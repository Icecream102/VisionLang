"""Train MAE on an ImageFolder dataset with exact checkpoint recovery.

Example:
    python -m examples.representation_learning.mae_pretrain \
      --data-root /datasets/imagenet100/train --output-dir outputs/mae_tiny \
      --model-size tiny --epochs 100 --amp
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from examples.representation_learning.mae_utils import (
    build_mae,
    capture_rng_state,
    restore_rng_state,
    set_seed,
)
from examples.representation_learning.imagefolder_utils import stratified_split_indices
from torchmultimodal.modules.losses.reconstruction_loss import ReconstructionLoss
from torchmultimodal.transforms.mae_transform import (
    ImageEvalTransform,
    ImagePretrainTransform,
)


logger = logging.getLogger(__name__)


def build_scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    if total_steps <= 0:
        raise ValueError("The training loader must contain at least one batch.")

    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, scale)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
            "rng_state": capture_rng_state(),
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Tuple[int, int]:
    # This project owns the checkpoint and stores RNG state alongside tensors, which
    # PyTorch's 2.6+ weights-only default intentionally rejects.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint.get("scaler", {}))
    if "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])
    logger.info(
        "Resumed %s at epoch=%d step=%d",
        path,
        checkpoint["epoch"],
        checkpoint["global_step"],
    )
    return int(checkpoint["epoch"]) + 1, int(checkpoint["global_step"])


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: ReconstructionLoss,
    device: torch.device,
    optimizer: AdamW = None,
    scheduler: LambdaLR = None,
    scaler: torch.amp.GradScaler = None,
    amp: bool = False,
    accumulation_steps: int = 1,
) -> Tuple[float, int]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    optimizer_steps = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for batch_index, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp):
                output = model(images, return_reconstruction=True)
                loss = criterion(output.decoder_pred, output.label_patches, output.mask)
            total_loss += loss.detach().item()
            if training:
                scaler.scale(loss / accumulation_steps).backward()
                is_update = (
                    batch_index + 1
                ) % accumulation_steps == 0 or batch_index + 1 == len(loader)
                if is_update:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    optimizer_steps += 1
    return total_loss / len(loader), optimizer_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", required=True, help="ImageFolder train directory"
    )
    parser.add_argument(
        "--val-root", default=None, help="Optional ImageFolder validation directory"
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Create a deterministic per-class validation split from --data-root.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-size", choices=("tiny", "base"), default="tiny")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--amp", action="store_true", help="Enable CUDA mixed precision"
    )
    parser.add_argument("--resume", default=None, help="Path to a last.pt checkpoint")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--normalize-target", action="store_true", default=True)
    parser.add_argument(
        "--no-normalize-target", action="store_false", dest="normalize_target"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.accumulation_steps <= 0 or args.save_every <= 0:
        raise ValueError("epochs, accumulation-steps and save-every must be positive.")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    set_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    amp = args.amp and device.type == "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    if args.val_root and args.val_fraction is not None:
        raise ValueError("Use either --val-root or --val-fraction, not both.")
    dataset = ImageFolder(
        args.data_root, transform=ImagePretrainTransform(args.image_size)
    )
    val_dataset = None
    if args.val_fraction is not None:
        train_indices, val_indices = stratified_split_indices(
            dataset.targets, args.val_fraction, args.seed
        )
        dataset = Subset(dataset, train_indices)
        val_dataset = Subset(
            ImageFolder(args.data_root, transform=ImageEvalTransform(args.image_size)),
            val_indices,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = None
    if args.val_root:
        val_dataset = ImageFolder(
            args.val_root, transform=ImageEvalTransform(args.image_size)
        )
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
    model = build_mae(args.model_size, args.image_size, args.mask_ratio).to(device)
    criterion = ReconstructionLoss(normalize_target=args.normalize_target)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(loader) / args.accumulation_steps)
    scheduler = build_scheduler(
        optimizer,
        args.warmup_epochs * updates_per_epoch,
        args.epochs * updates_per_epoch,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, global_step = 0, 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device
        )

    history_path = output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        loss, optimizer_steps = run_epoch(
            model,
            loader,
            criterion,
            device,
            optimizer,
            scheduler,
            scaler,
            amp,
            args.accumulation_steps,
        )
        val_loss = None
        if val_loader is not None:
            val_loss, _ = run_epoch(model, val_loader, criterion, device, amp=amp)
        global_step += optimizer_steps
        result: Dict[str, float] = {
            "epoch": epoch + 1,
            "train_masked_mse": loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "global_step": global_step,
        }
        if val_loss is not None:
            result["val_masked_mse"] = val_loss
        with history_path.open("a") as history:
            history.write(json.dumps(result) + "\n")
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            args,
        )
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            save_checkpoint(
                output_dir / f"epoch_{epoch + 1:04d}.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                args,
            )
        logger.info(
            "epoch=%d masked_mse=%.6f lr=%.3e", epoch + 1, loss, result["learning_rate"]
        )


if __name__ == "__main__":
    main()
