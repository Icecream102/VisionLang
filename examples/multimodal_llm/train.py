"""Two-stage LLaVA-style captioning training with resumable checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoConfig, AutoTokenizer

from examples.multimodal_llm.data import (
    LlavaCaptionDataset,
    build_eval_transform,
    build_train_transform,
    collate_captioning,
)
from examples.multimodal_llm.evaluate import (
    evaluate_coco_captions,
    evaluate_mc_accuracy,
    evaluate_recognition_accuracy,
)
from examples.multimodal_llm.model import (
    IMAGE_TOKEN,
    LlavaCaptionModel,
    MLPProjector,
    build_llm,
    build_vision_tower,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--val-image-root", default=None)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init", choices=("random", "mae", "clip"), default="clip")
    parser.add_argument("--model-name", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=float, default=128.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--stage", choices=("projector", "lora"), required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--lr-proj", type=float, default=3e-4)
    parser.add_argument("--lr-lora", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val-eval", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-beams", type=int, default=1)
    parser.add_argument("--vision-lora", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--recognition-eval", action="store_true")
    parser.add_argument("--mc-eval", action="store_true")
    parser.add_argument("--vqa-eval", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--no-save-llm-base", action="store_true")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def restore_rng_state(state: Dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())


def build_scheduler(optimizer, warmup_steps: int, total_steps: int, min_ratio: float):
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (
            1 + math.cos(math.pi * min(1.0, progress))
        )

    return LambdaLR(optimizer, schedule)


def build_optimizer(
    model: LlavaCaptionModel,
    lr_proj: float,
    lr_lora: float,
    weight_decay: float,
) -> AdamW:
    projector = []
    lora = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("projector."):
            projector.append(parameter)
        elif "lora_" in name:
            lora.append(parameter)
        else:
            raise ValueError(f"Unexpected trainable parameter: {name}")
    groups = [{"params": projector, "lr": lr_proj}]
    if lora:
        groups.append({"params": lora, "lr": lr_lora})
    return AdamW([g for g in groups if g["params"]], weight_decay=weight_decay)


def save_checkpoint(
    path: Path,
    model: LlavaCaptionModel,
    optimizer: AdamW,
    scheduler: LambdaLR,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
) -> None:
    base_llm = model.llm.get_base_model() if hasattr(model.llm, "get_base_model") else model.llm
    adapter = {
        key: value
        for key, value in model.llm.state_dict().items()
        if "lora_" in key
    }
    checkpoint = {
        "args": vars(args),
        "epoch": epoch,
        "global_step": global_step,
        "vision": model.vision.state_dict() if model.vision is not None else None,
        "projector": model.projector.state_dict() if model.projector is not None else None,
        "llm_base": None if getattr(args, "no_save_llm_base", False) else base_llm.state_dict(),
        "adapter": adapter,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
    }
    torch.save(checkpoint, path)


def save_components(
    model: LlavaCaptionModel, output_dir: Path, save_llm_base: bool = True
) -> None:
    base_llm = model.llm.get_base_model() if hasattr(model.llm, "get_base_model") else model.llm
    if model.vision is not None:
        torch.save(model.vision.state_dict(), output_dir / "vision.pt")
    if model.projector is not None:
        torch.save(model.projector.state_dict(), output_dir / "projector.pt")
    if save_llm_base:
        torch.save(base_llm.state_dict(), output_dir / "llm_base.pt")
    if hasattr(model.llm, "save_pretrained"):
        model.llm.save_pretrained(output_dir / "lora_adapter")


def load_checkpoint(
    path: Path,
    model: LlavaCaptionModel,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if model.vision is not None and checkpoint.get("vision") is not None:
        model.vision.load_state_dict(checkpoint["vision"])
    if model.projector is not None and checkpoint.get("projector") is not None:
        model.projector.load_state_dict(checkpoint["projector"])
    base_llm = (
        model.llm.get_base_model()
        if hasattr(model.llm, "get_base_model")
        else model.llm
    )
    if checkpoint.get("llm_base") is not None:
        base_llm.load_state_dict(checkpoint["llm_base"])
    if checkpoint["adapter"]:
        model.llm.load_state_dict(checkpoint["adapter"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint["epoch"] + 1, checkpoint["global_step"]


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    set_seed(args.seed)
    is_distributed = args.distributed
    if is_distributed:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        is_main = torch.distributed.get_rank() == 0
        world_size = torch.distributed.get_world_size()
    else:
        local_rank = 0
        world_size = 1
        device = torch.device(
            args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        is_main = True
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.convert_tokens_to_ids(IMAGE_TOKEN) == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    llm_config = AutoConfig.from_pretrained(args.model_name)
    vision = (
        None
        if args.text_only
        else build_vision_tower(args.init, args.image_size, lora=args.vision_lora)
    )
    projector = None if args.text_only else MLPProjector(llm_dim=llm_config.hidden_size)
    llm = build_llm(
        args.model_name,
        tokenizer,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model = LlavaCaptionModel(
        vision,
        projector,
        llm,
        image_token_id,
        vision_lora=args.vision_lora,
        text_only=args.text_only,
    ).to(device)
    if args.text_only:
        # Text-only mode has no vision/projector activations; gradient
        # checkpointing would break the graph because the frozen embedding
        # lookup produces inputs_embeds without requires_grad.
        logger.info("Text-only mode: gradient checkpointing disabled")
        model.llm.config.use_cache = False
    elif args.gradient_checkpointing:
        model.llm.gradient_checkpointing_enable()
        model.llm.config.use_cache = False

    start_epoch, global_step = 0, 0

    train_dataset = LlavaCaptionDataset(
        args.train_manifest,
        args.image_root,
        tokenizer,
        build_train_transform(args.image_size),
        seed=args.seed,
        max_length=args.max_length,
        limit=args.limit_train,
        text_only=args.text_only,
    )
    val_dataset = LlavaCaptionDataset(
        args.val_manifest,
        args.val_image_root or args.image_root,
        tokenizer,
        build_eval_transform(args.image_size),
        seed=args.seed,
        max_length=args.max_length,
        eval_mode=True,
        text_only=args.text_only,
    )

    steps_per_epoch = math.ceil(
        len(train_dataset) / (args.batch_size * args.accum_steps * world_size)
    )
    model.set_trainable(args.stage, args.vision_lora)
    if is_distributed:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
    optimizer = build_optimizer(model, args.lr_proj, args.lr_lora, args.weight_decay)
    scheduler = build_scheduler(
        optimizer,
        int(args.warmup_ratio * args.epochs * steps_per_epoch),
        args.epochs * steps_per_epoch,
        0.1,
    )
    if args.resume:
        start_epoch, global_step = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, device
        )
        logger.info(
            "Resumed from %s at epoch %d step %d",
            args.resume,
            start_epoch,
            global_step,
        )
    elif args.init_checkpoint:
        init_dir = Path(args.init_checkpoint)
        if vision is not None:
            vision.load_state_dict(
                torch.load(init_dir / "vision.pt", map_location=device, weights_only=False)
            )
        if projector is not None:
            projector.load_state_dict(
                torch.load(
                    init_dir / "projector.pt", map_location=device, weights_only=False
                )
            )
        logger.info("Loaded stage-1 components from %s", init_dir)

    sampler = (
        DistributedSampler(
            train_dataset, shuffle=True, seed=args.seed, drop_last=True
        )
        if is_distributed
        else None
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_captioning,
    )
    history_path = output_dir / "metrics.jsonl"
    best_cider = float("-inf")
    if history_path.exists():
        for line in history_path.read_text().strip().splitlines():
            best_cider = max(best_cider, json.loads(line).get("val_CIDEr", float("-inf")))

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            images = batch["images"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(images, input_ids, attention_mask, labels)
                loss = outputs.loss / args.accum_steps
            loss.backward()
            total_loss += loss.detach().item()
            if (step + 1) % args.accum_steps == 0:
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 50 == 0:
                    logger.info(
                        "epoch=%d step=%d loss=%.4f lr_proj=%.2e lr_lora=%.2e",
                        epoch + 1,
                        global_step,
                        total_loss / (step + 1) * args.accum_steps,
                        optimizer.param_groups[0]["lr"],
                        optimizer.param_groups[-1]["lr"],
                    )
        if (step + 1) % args.accum_steps != 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        if not is_main:
            continue
        if args.recognition_eval:
            metrics = evaluate_recognition_accuracy(
                model,
                val_dataset,
                tokenizer,
                device,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                limit=args.limit_val_eval,
            )
        elif args.vqa_eval:
            from examples.multimodal_llm.eval_okvqa import evaluate_okvqa

            vqa_result = evaluate_okvqa(
                model,
                tokenizer,
                args.val_manifest,
                args.val_image_root or args.image_root,
                device,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                limit=args.limit_val_eval,
            )
            metrics = {
                "accuracy": vqa_result["accuracy"],
                "num_questions": vqa_result["num_questions"],
            }
        elif args.mc_eval:
            metrics = evaluate_mc_accuracy(
                model,
                val_dataset,
                tokenizer,
                device,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                limit=args.limit_val_eval,
            )
        else:
            metrics = evaluate_coco_captions(
                model,
                val_dataset,
                tokenizer,
                device,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                num_beams=args.eval_beams,
                limit=args.limit_val_eval,
            )
        result = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": total_loss / len(loader) * args.accum_steps,
            "lr_proj": optimizer.param_groups[0]["lr"],
            "lr_lora": optimizer.param_groups[-1]["lr"],
            **{f"val_{key}": value for key, value in metrics.items()},
        }
        with history_path.open("a") as handle:
            handle.write(json.dumps(result) + "\n")
        checkpoint = output_dir / "last.pt"
        save_checkpoint(
            checkpoint, model, optimizer, scheduler, epoch, global_step, args
        )
        selection_metric = (
            metrics.get("accuracy", float("-inf"))
            if (args.recognition_eval or args.mc_eval or args.vqa_eval)
            else metrics.get("CIDEr", float("-inf"))
        )
        if selection_metric >= best_cider:
            best_cider = selection_metric
            save_checkpoint(
                output_dir / "best.pt", model, optimizer, scheduler, epoch, global_step, args
            )
            save_components(
                model,
                output_dir,
                save_llm_base=not args.no_save_llm_base,
            )
            logger.info("New best %s=%.4f saved", "accuracy" if args.recognition_eval else "CIDEr", selection_metric)
if __name__ == "__main__":
    main()
