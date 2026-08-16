"""GRPO alignment (DeepSeek-R1 style) on the COCO yes/no recognition task.

Loads a supervised-fine-tuned LLaVA-style checkpoint, samples a group of
responses per prompt from the current policy, scores them with a rule-based
reward (correct yes/no answer), normalizes advantages inside each group, and
optimizes the policy with a per-token KL penalty against the frozen SFT
reference.  Runs on a single GPU with LoRA + projector trainable.

This demonstrates the RL/alignment training loop (reward design, group
advantage normalization, KL-constrained policy update) that is expected of a
multimodal LLM training role.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import re
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
from examples.multimodal_llm.evaluate import evaluate_recognition_accuracy
from examples.multimodal_llm.model import (
    IMAGE_TOKEN,
    LlavaCaptionModel,
    MLPProjector,
    build_llm,
    build_vision_tower,
)


logger = logging.getLogger(__name__)


class _Subset:
    """Minimal index subset wrapper around LlavaCaptionDataset."""

    def __init__(self, base, indices):
        self.base = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.base[self.indices[index]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_yes_no(answer: str) -> int | None:
    """Return 1 for yes, 0 for no, None if the answer is ambiguous."""
    if re.search(r"\byes\b", answer.lower()):
        return 1
    if re.search(r"\bno\b", answer.lower()):
        return 0
    return None


def rule_reward(answer: str, expected: str) -> float:
    prediction = parse_yes_no(answer)
    label = 1 if expected.strip().lower() == "yes" else 0
    if prediction is None:
        return -0.5  # gibberish / no answer: penalize
    return 1.0 if prediction == label else 0.0


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
    llm.get_base_model().load_state_dict(
        torch.load(init_dir / "llm_base.pt", weights_only=False)
    )
    llm.load_adapter(str(init_dir / "lora_adapter"), "default")
    model.set_trainable("lora")
    return model, tokenizer


def sequence_logprobs(
    model: LlavaCaptionModel,
    image_tokens: torch.Tensor,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    gen_ids: torch.Tensor,
    gen_mask: torch.Tensor,
    start: int = 0,
    end: int | None = None,
    chunk_size: int = 8,
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
            [
                prompt_mask[chunk],
                gen_mask[chunk].to(prompt_mask.dtype),
            ],
            dim=1,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--qa-train-manifest", required=True)
    parser.add_argument("--qa-val-manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--prompts", type=int, default=600)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--batch-prompts", type=int, default=4)
    parser.add_argument("--grad-chunk", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--eval-limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)
    device = torch.device("cuda")

    policy, tokenizer = load_sft_model(Path(args.init_checkpoint), args.model_name)
    reference = copy.deepcopy(policy)
    # The reference only needs the LLM for log-prob scoring; image tokens are
    # computed once by the policy and shared.  Dropping the vision/projector
    # copies saves ~0.4 GB and halves vision forwards.
    reference.vision = None
    reference.projector = None
    for parameter in reference.parameters():
        parameter.requires_grad = False
    reference.eval()
    torch.cuda.empty_cache()

    dataset = LlavaCaptionDataset(
        args.qa_train_manifest,
        args.image_root,
        tokenizer,
        build_eval_transform(224),
        seed=args.seed,
        eval_mode=True,
        limit=None,
    )
    if args.balanced:
        yes_idx = [
            i
            for i, record in enumerate(dataset.records)
            if record["captions"][0].strip().lower() == "yes"
        ]
        no_idx = [
            i
            for i, record in enumerate(dataset.records)
            if record["captions"][0].strip().lower() == "no"
        ]
        rng = random.Random(args.seed)
        half = args.prompts // 2
        indices = rng.sample(yes_idx, min(half, len(yes_idx))) + rng.sample(
            no_idx, min(half, len(no_idx))
        )
        rng.shuffle(indices)
        dataset = _Subset(dataset, indices)
        logger.info(
            "Balanced GRPO subset: %d yes + %d no = %d prompts",
            min(half, len(yes_idx)),
            min(half, len(no_idx)),
            len(indices),
        )
    else:
        dataset = _Subset(dataset, list(range(min(args.prompts, len(dataset)))))
    val_dataset = LlavaCaptionDataset(
        args.qa_val_manifest,
        args.image_root,
        tokenizer,
        build_eval_transform(224),
        eval_mode=True,
        limit=args.eval_limit,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2) + "\n"
    )
    metrics_path = output_dir / "grpo_metrics.jsonl"

    def evaluate(label: str) -> Dict[str, float]:
        policy.eval()
        metrics = evaluate_recognition_accuracy(
            policy,
            val_dataset,
            tokenizer,
            device,
            batch_size=8,
            num_workers=0,
        )
        logger.info("%s accuracy=%.4f", label, metrics["accuracy"])
        return metrics

    before = evaluate("GRPO before")
    torch.cuda.empty_cache()

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.0)
    group_size = args.group_size
    step = 0
    running = {"reward": 0.0, "kl": 0.0, "loss": 0.0, "n": 0}
    cumulative = {"reward": 0.0, "kl": 0.0, "loss": 0.0}

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
            expected = [item["caption"] for item in items]
            # Build group batch: repeat each prompt group_size times.
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
                    repetition_penalty=1.0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            policy.llm.config.use_cache = False
            prompt_lengths = group_mask.sum(dim=1)
            expanded = (
                prompt_lengths + (policy.image_token_count - 1)
            ).long()
            gen_ids = []
            gen_masks = []
            rewards = []
            for row, length, offset in zip(
                outputs, expanded, range(0, batch_size * group_size)
            ):
                generated = row[length.item() :].cpu().reshape(-1)
                answer = tokenizer.decode(generated, skip_special_tokens=True)
                gen_ids.append(generated)
                gen_masks.append(torch.ones_like(generated))
                rewards.append(
                    rule_reward(answer, expected[offset // group_size])
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
            # Policy scoring is chunked with an immediate backward per chunk:
            # keeping every chunk's activations alive until one big backward
            # exhausts the GPU (each forward-with-grad retains ~5 GB of saved
            # activations for 32 rows).
            policy.train()
            total_rows = group_images.size(0)
            step_loss = 0.0
            step_kl = 0.0
            for cstart in range(0, total_rows, args.grad_chunk):
                cend = min(cstart + args.grad_chunk, total_rows)
                # Fresh with-grad image tokens per chunk: a shared token
                # tensor would be freed by the first chunk's backward.
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
                # Non-negative per-token KL estimate from the GRPO paper:
                # KL_t = exp(log(pi_ref/pi_theta)) - log(pi_ref/pi_theta) - 1,
                # which is 0 iff pi_theta == pi_ref and penalizes divergence in
                # both directions.
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
                torch.nn.utils.clip_grad_norm_(
                    trainable, args.grad_clip_norm
                )
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
                    step, row["reward"], row["kl"], row["loss"],
                )
                running = {"reward": 0.0, "kl": 0.0, "loss": 0.0, "n": 0}

    after = evaluate("GRPO after")
    summary = {
        "before": {k: float(v) for k, v in before.items()},
        "after": {k: float(v) for k, v in after.items()},
        "steps": step,
        "group_size": group_size,
        "beta": args.beta,
        "final_kl": float(cumulative["kl"] / max(1, step)),
        "mean_reward": float(cumulative["reward"] / max(1, step)),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=1) + "\n"
    )

    policy.eval()
    policy.llm.save_pretrained(output_dir / "lora_adapter")
    torch.save(policy.vision.state_dict(), output_dir / "vision.pt")
    torch.save(policy.projector.state_dict(), output_dir / "projector.pt")
    base_llm = (
        policy.llm.get_base_model()
        if hasattr(policy.llm, "get_base_model")
        else policy.llm
    )
    torch.save(base_llm.state_dict(), output_dir / "llm_base.pt")
    logger.info("Saved GRPO policy to %s", output_dir)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
