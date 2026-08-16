"""GRPO on open-ended OK-VQA generation (extends the toy yes/no GRPO demo).

Loads the 3B OK-VQA SFT checkpoint, samples a group of free-form answers per
prompt from the current policy, scores them with the standard VQA rule reward
(normalized prediction matching >=3 of the 10 human answers), normalizes
advantages within each group, and optimizes with a per-token KL penalty
against the frozen SFT reference (DeepSeek-R1 style GRPO).

Protocol (documented honestly):
- Prompts come from OK-VQA val questions (each has 10 human answers needed
  for the reward).  The val split is partitioned into a disjoint "RL subset"
  (--prompts) and a held-out evaluation subset (--eval-limit), so the
  before/after accuracy is measured on questions never used for optimization.
- This is a method demonstration on a single GPU, not a state-of-the-art run.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoConfig, AutoTokenizer

from examples.multimodal_llm.data import (
    LlavaCaptionDataset,
    build_eval_transform,
)
from examples.multimodal_llm.eval_okvqa import evaluate_okvqa, normalize_answer
from examples.multimodal_llm.model import (
    IMAGE_TOKEN,
    LlavaCaptionModel,
    MLPProjector,
    build_llm,
    build_vision_tower,
)


logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_sft_model(
    init_dir: Path,
    model_name: str,
) -> Tuple[LlavaCaptionModel, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.convert_tokens_to_ids(IMAGE_TOKEN) == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    adapter_config = json.loads(
        (init_dir / "lora_adapter" / "adapter_config.json").read_text()
    )
    llm_config = AutoConfig.from_pretrained(model_name)
    vision = build_vision_tower("clip")
    projector = MLPProjector(llm_dim=llm_config.hidden_size)
    llm = build_llm(
        model_name,
        tokenizer,
        lora_r=int(adapter_config.get("r", 64)),
        lora_alpha=float(adapter_config.get("lora_alpha", 128.0)),
    )
    model = LlavaCaptionModel(vision, projector, llm, image_token_id).cuda()
    vision.load_state_dict(
        torch.load(init_dir / "vision.pt", weights_only=False)
    )
    projector.load_state_dict(
        torch.load(init_dir / "projector.pt", weights_only=False)
    )
    base_llm = llm.get_base_model()
    if (init_dir / "llm_base.pt").exists():
        base_llm.load_state_dict(
            torch.load(init_dir / "llm_base.pt", weights_only=False)
        )
    else:
        print("  llm_base.pt not found; using pretrained base weights", flush=True)
    llm.load_adapter(str(init_dir / "lora_adapter"), "default")
    model.set_trainable("lora")
    model.llm.gradient_checkpointing_enable()
    model.llm.config.use_cache = False
    return model, tokenizer


def vqa_rule_reward(prediction: str, answers: List[str]) -> float:
    normalized = normalize_answer(prediction)
    if not normalized:
        return 0.0
    matched = sum(1 for answer in answers if normalize_answer(answer) == normalized)
    return 1.0 if matched >= 3 else 0.0


def sequence_logprobs(
    model: LlavaCaptionModel,
    image_tokens: torch.Tensor,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    gen_ids: torch.Tensor,
    gen_mask: torch.Tensor,
    start: int = 0,
    end: int | None = None,
    chunk_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sequence summed log-probs and token counts of generated tokens."""
    if end is None:
        end = prompt_ids.size(0)
    batch_size = end - start
    all_logp = torch.zeros(batch_size, device=image_tokens.device)
    all_counts = torch.zeros(batch_size, device=image_tokens.device)
    for offset in range(0, batch_size, chunk_size):
        chunk = slice(start + offset, start + min(offset + chunk_size, batch_size))
        labels = torch.full_like(prompt_ids[chunk], -100)
        padded_gen = torch.where(
            gen_mask[chunk].bool(),
            gen_ids[chunk],
            torch.full_like(gen_ids[chunk], -100),
        )
        labels = torch.cat([labels, padded_gen], dim=1)
        full_ids = torch.cat([prompt_ids[chunk], gen_ids[chunk]], dim=1)
        full_mask = torch.cat(
            [prompt_mask[chunk], gen_mask[chunk].to(prompt_mask.dtype)], dim=1
        )
        logits, expanded_labels = model.forward_logprobs(
            None,
            full_ids,
            full_mask,
            labels,
            image_tokens=image_tokens[chunk],
        )
        vocab_size = logits.size(-1)
        shift_logits = logits[..., :-1, :].reshape(-1, vocab_size).contiguous()
        shift_labels = expanded_labels[..., 1:].reshape(-1).contiguous()
        token_loss = F.cross_entropy(shift_logits, shift_labels, reduction="none")
        token_loss = token_loss.reshape(chunk.stop - chunk.start, -1)
        mask = expanded_labels[..., 1:] != -100
        counts = mask.sum(dim=1).clamp(min=1)
        all_logp[offset : offset + (chunk.stop - chunk.start)] = (
            -token_loss * mask
        ).sum(dim=1)
        all_counts[offset : offset + (chunk.stop - chunk.start)] = counts
    return all_logp, all_counts


def write_subset_manifest(
    records, indices: List[int], output: Path
) -> None:
    with output.open("w") as handle:
        for i in indices:
            handle.write(json.dumps(records[i]) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--qa-val-manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--prompts", type=int, default=300)
    parser.add_argument("--eval-limit", type=int, default=1000)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--batch-prompts", type=int, default=2)
    parser.add_argument("--grad-chunk", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    set_seed(args.seed)
    device = torch.device("cuda")

    policy, tokenizer = load_sft_model(Path(args.init_checkpoint), args.model_name)
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    reference = copy.deepcopy(policy)
    reference.vision = None
    reference.projector = None
    for parameter in reference.parameters():
        parameter.requires_grad = False
    reference.eval()
    torch.cuda.empty_cache()

    full_dataset = LlavaCaptionDataset(
        args.qa_val_manifest,
        args.image_root,
        tokenizer,
        build_eval_transform(224),
        eval_mode=True,
        limit=None,
    )
    records = full_dataset.records
    total = len(records)
    if total < args.prompts + args.eval_limit:
        raise SystemExit(
            f"need {args.prompts + args.eval_limit} records, have {total}"
        )
    rng = random.Random(args.seed)
    all_indices = list(range(total))
    rng.shuffle(all_indices)
    train_indices = all_indices[: args.prompts]
    eval_indices = all_indices[args.prompts : args.prompts + args.eval_limit]
    train_indices.sort()
    eval_indices.sort()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2) + "\n"
    )
    train_manifest = output_dir / "rl_subset.jsonl"
    eval_manifest = output_dir / "eval_subset.jsonl"
    write_subset_manifest(records, train_indices, train_manifest)
    write_subset_manifest(records, eval_indices, eval_manifest)

    # Use the index subset via a lightweight dataset wrapper.
    class _Subset:
        def __init__(self, base, indices):
            self.base = base
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, index):
            return self.base[self.indices[index]]

    dataset = _Subset(full_dataset, train_indices)

    def evaluate(label: str) -> Dict[str, float]:
        policy.eval()
        result = evaluate_okvqa(
            policy,
            tokenizer,
            str(eval_manifest),
            args.image_root,
            device,
            batch_size=8,
            num_workers=0,
            limit=None,
        )
        metrics = {
            "accuracy": result["accuracy"],
            "num_questions": result["num_questions"],
            "mean_pred_len": result["mean_pred_len"],
        }
        logger.info(
            "%s OK-VQA acc=%.4f (n=%d)",
            label,
            metrics["accuracy"],
            metrics["num_questions"],
        )
        return metrics

    before = evaluate("GRPO before")
    torch.cuda.empty_cache()

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.0)
    group_size = args.group_size
    step = 0
    running = {"reward": 0.0, "kl": 0.0, "loss": 0.0, "n": 0}
    cumulative = {"reward": 0.0, "kl": 0.0, "loss": 0.0}
    metrics_path = output_dir / "grpo_metrics.jsonl"

    for epoch in range(args.epochs):
        for batch_start in range(0, len(dataset), args.batch_prompts):
            items = [
                dataset[i]
                for i in range(
                    batch_start, min(batch_start + args.batch_prompts, len(dataset))
                )
            ]
            batch_size = len(items)
            images = torch.stack([item["image"] for item in items]).to(device)
            prompt_ids = torch.nn.utils.rnn.pad_sequence(
                [item["input_ids"] for item in items],
                batch_first=True,
                padding_value=0,
            ).to(device)
            prompt_mask = torch.nn.utils.rnn.pad_sequence(
                [item["attention_mask"] for item in items],
                batch_first=True,
                padding_value=0,
            ).to(device)
            expected = [
                full_dataset.records[train_indices[i]]["captions"]
                for i in range(
                    batch_start, min(batch_start + args.batch_prompts, len(dataset))
                )
            ]
            group_images = images.repeat_interleave(group_size, dim=0)
            group_ids = prompt_ids.repeat_interleave(group_size, dim=0)
            group_mask = prompt_mask.repeat_interleave(group_size, dim=0)

            policy.eval()
            policy.llm.config.use_cache = True
            with torch.no_grad():
                outputs = policy.generate(
                    group_images,
                    group_ids,
                    group_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=1.3,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    bad_words_ids=[[image_token_id]],
                )
            policy.llm.config.use_cache = False
            prompt_lengths = group_mask.sum(dim=1)
            expanded = (prompt_lengths + (policy.image_token_count - 1)).long()
            gen_ids = []
            gen_masks = []
            rewards = []
            for row, length, offset in zip(
                outputs, expanded, range(0, batch_size * group_size)
            ):
                generated = row[length.item() :].cpu().reshape(-1)
                # Defensive: drop any special image tokens the sampler emitted.
                generated = generated[generated != image_token_id]
                answer = tokenizer.decode(generated, skip_special_tokens=True)
                gen_ids.append(generated)
                gen_masks.append(torch.ones_like(generated))
                rewards.append(
                    vqa_rule_reward(answer, expected[offset // group_size])
                )
            gen_ids = torch.nn.utils.rnn.pad_sequence(
                gen_ids,
                batch_first=True,
                padding_value=tokenizer.pad_token_id or tokenizer.eos_token_id,
            ).to(device)
            gen_mask = torch.nn.utils.rnn.pad_sequence(
                gen_masks, batch_first=True, padding_value=0
            ).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            rewards = rewards.view(-1, group_size)
            advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / (
                rewards.std(dim=1, keepdim=True) + 1e-4
            )
            advantages = advantages.reshape(-1)

            with torch.no_grad():
                ref_tokens = policy.get_image_tokens(group_images)
                ref_logp, ref_counts = sequence_logprobs(
                    reference,
                    ref_tokens,
                    group_ids,
                    group_mask,
                    gen_ids,
                    gen_mask,
                )
            policy.train()
            total_rows = group_images.size(0)
            step_loss = 0.0
            step_kl = 0.0
            for cstart in range(0, total_rows, args.grad_chunk):
                cend = min(cstart + args.grad_chunk, total_rows)
                chunk_tokens = policy.get_image_tokens(group_images[cstart:cend])
                policy_logp, counts = sequence_logprobs(
                    policy,
                    chunk_tokens,
                    group_ids[cstart:cend],
                    group_mask[cstart:cend],
                    gen_ids[cstart:cend],
                    gen_mask[cstart:cend],
                    chunk_size=args.grad_chunk,
                )
                per_token_policy = policy_logp / counts
                per_token_ref = ref_logp[cstart:cend] / ref_counts[cstart:cend]
                log_ratio = per_token_ref - per_token_policy
                kl = torch.exp(log_ratio) - log_ratio - 1.0
                adv = advantages[cstart:cend]
                chunk_loss = (
                    -(adv * per_token_policy).sum() + args.beta * kl.sum()
                ) / total_rows
                chunk_loss.backward()
                step_loss += chunk_loss.item()
                step_kl += kl.mean().item()
                del (
                    policy_logp,
                    counts,
                    per_token_policy,
                    per_token_ref,
                    log_ratio,
                    kl,
                    chunk_loss,
                )
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            running["reward"] += rewards.mean().item()
            running["kl"] += step_kl
            running["loss"] += step_loss
            running["n"] += 1
            cumulative["reward"] += rewards.mean().item()
            cumulative["kl"] += step_kl
            cumulative["loss"] += step_loss
            step += 1
            if step % args.log_every == 0:
                row = {
                    "step": step,
                    "reward": running["reward"] / running["n"],
                    "kl": running["kl"] / running["n"],
                    "loss": running["loss"] / running["n"],
                }
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                logger.info(
                    "step=%d reward=%.4f kl=%.4f loss=%.4f",
                    step,
                    row["reward"],
                    row["kl"],
                    row["loss"],
                )
                running = {"reward": 0.0, "kl": 0.0, "loss": 0.0, "n": 0}

    after = evaluate("GRPO after")
    summary = {
        "before": before,
        "after": after,
        "steps": step,
        "group_size": group_size,
        "beta": args.beta,
        "final_kl": float(cumulative["kl"] / max(1, step)),
        "mean_reward": float(cumulative["reward"] / max(1, step)),
        "rl_prompts": len(train_indices),
        "eval_questions": len(eval_indices),
        "disjoint_splits": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=1) + "\n"
    )

    policy.eval()
    policy.llm.save_pretrained(output_dir / "lora_adapter")
    torch.save(policy.vision.state_dict(), output_dir / "vision.pt")
    torch.save(policy.projector.state_dict(), output_dir / "projector.pt")
    logger.info("Saved GRPO policy to %s (no llm_base to save disk)", output_dir)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
