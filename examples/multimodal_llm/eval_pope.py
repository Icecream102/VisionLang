"""POPE object-hallucination evaluation for completed VLM checkpoints.

Loads each finished run (vision.pt / projector.pt / llm_base.pt /
lora_adapter), answers the three POPE question sets (random, popular,
adversarial), and reports accuracy plus yes/no accuracy per split.  Results
are written to ``pope.json`` inside each run directory and aggregated into
``outputs/pope_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from examples.multimodal_llm.data import (
    LlavaCaptionDataset,
    build_eval_transform,
    collate_captioning,
)
from examples.multimodal_llm.eval_final import load_model
from examples.multimodal_llm.model import IMAGE_TOKEN


SPLITS = ["random", "popular", "adversarial"]


@torch.inference_mode()
def evaluate_pope_split(
    model,
    tokenizer,
    manifest: str,
    image_root: str,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Dict[str, float]:
    dataset = LlavaCaptionDataset(
        manifest,
        image_root,
        tokenizer,
        build_eval_transform(224),
        eval_mode=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_captioning,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    generator_kwargs = {
        "max_new_tokens": 8,
        "num_beams": 1,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    correct = total = yes_correct = yes_total = no_correct = no_total = 0
    start = 0
    for batch in loader:
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prompt_lengths = attention_mask.sum(dim=1).long()
        output_ids = model.generate(
            images, input_ids, attention_mask, **generator_kwargs
        )
        for offset, (row, prompt_length) in enumerate(
            zip(output_ids, prompt_lengths)
        ):
            expanded_length = (
                prompt_length
                if model.text_only
                else prompt_length + (model.image_token_count - 1)
            )
            answer = tokenizer.decode(
                row[expanded_length:], skip_special_tokens=True
            ).strip().lower()
            prediction = 1 if re.search(r"\byes\b", answer) else 0
            expected = dataset.records[start + offset]["captions"][0].strip().lower()
            label = 1 if expected == "yes" else 0
            total += 1
            correct += prediction == label
            if label:
                yes_total += 1
                yes_correct += prediction == 1
            else:
                no_total += 1
                no_correct += prediction == 0
        start += images.size(0)
    return {
        "accuracy": correct / total if total else float("nan"),
        "yes_accuracy": yes_correct / yes_total if yes_total else float("nan"),
        "no_accuracy": no_correct / no_total if no_total else float("nan"),
        "num_questions": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", required=True)
    parser.add_argument("--pope-manifest-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    outputs = Path(args.outputs_dir)
    manifest_dir = Path(args.pope_manifest_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.convert_tokens_to_ids(IMAGE_TOKEN) == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    llm_config = AutoConfig.from_pretrained(args.model_name)

    candidates = sorted(
        path
        for path in outputs.glob("*/")
        if (path / "vision.pt").exists()
        and (path / "projector.pt").exists()
    )
    if args.runs:
        names = set(args.runs)
        candidates = [path for path in candidates if path.name in names]
    if not candidates:
        raise SystemExit("No candidate runs found.")

    device = torch.device("cuda")
    summary = {}
    for run_dir in candidates:
        pope_path = run_dir / "pope.json"
        if pope_path.exists():
            print(f"{run_dir.name}: skip (exists)", flush=True)
            summary[run_dir.name] = json.loads(pope_path.read_text())
            continue
        print(f"{run_dir.name}: loading model", flush=True)
        model = load_model(run_dir, args.model_name)
        result = {}
        for split in SPLITS:
            metrics = evaluate_pope_split(
                model,
                tokenizer,
                str(manifest_dir / f"pope_{split}.jsonl"),
                args.image_root,
                device,
                batch_size=args.batch_size,
            )
            result[split] = metrics
            print(
                f"  {split}: acc={metrics['accuracy']:.4f} "
                f"yes={metrics['yes_accuracy']:.4f} no={metrics['no_accuracy']:.4f}",
                flush=True,
            )
        pope_path.write_text(json.dumps(result, indent=1) + "\n")
        summary[run_dir.name] = result
        del model
        torch.cuda.empty_cache()

    summary_path = outputs.parent / "pope_summary.json"
    summary_path.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
