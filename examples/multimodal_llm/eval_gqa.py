"""GQA testdev evaluation for a completed VLM checkpoint.

Loads a run (vision.pt / projector.pt / [llm_base.pt] / lora_adapter),
answers every GQA testdev_balanced question, and scores with the standard
GQA protocol: normalized exact match between the prediction and the short
answer.  Outputs per-question predictions, aggregate accuracy, and an
answer-length report.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

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


def normalize_answer(answer: str) -> str:
    """GQA-style normalization: lowercase, strip punctuation and articles."""
    answer = answer.splitlines()[0].strip() if answer.splitlines() else answer
    answer = re.sub(r"^[^a-z0-9]+", "", answer, flags=re.IGNORECASE).strip()
    answer = answer.lower()
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    answer = re.sub(r"[^a-z0-9 ]", " ", answer)
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer


def exact_match(prediction: str, reference: str) -> float:
    normalized = normalize_answer(prediction)
    if not normalized:
        return 0.0
    return 1.0 if normalized == normalize_answer(reference) else 0.0


@torch.inference_mode()
def evaluate_gqa(
    model,
    tokenizer,
    manifest: str,
    image_root: str,
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 4,
    limit: int | None = None,
) -> Dict:
    dataset = LlavaCaptionDataset(
        manifest,
        image_root,
        tokenizer,
        build_eval_transform(224),
        eval_mode=True,
        limit=limit,
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
        "max_new_tokens": 16,
        "num_beams": 1,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.3,
        "no_repeat_ngram_size": 3,
    }
    predictions: Dict[int, str] = {}
    correct = total = 0
    answer_lengths = Counter()
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
            ).strip()
            index = start + offset
            reference = dataset.records[index]["captions"][0]
            predictions[index] = answer
            correct += exact_match(answer, reference)
            total += 1
            answer_lengths[len(answer.split())] += 1
        start += images.size(0)
    return {
        "accuracy": correct / total if total else float("nan"),
        "num_questions": total,
        "mean_pred_len": (
            sum(k * v for k, v in answer_lengths.items()) / max(1, total)
        ),
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.convert_tokens_to_ids(IMAGE_TOKEN) == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    llm_config = AutoConfig.from_pretrained(args.model_name)

    model = load_model(Path(args.run_dir), args.model_name)
    result = evaluate_gqa(
        model,
        tokenizer,
        args.val_manifest,
        args.image_root,
        torch.device("cuda"),
        batch_size=args.batch_size,
        limit=args.limit,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(
            {
                "accuracy": result["accuracy"],
                "num_questions": result["num_questions"],
                "mean_pred_len": result["mean_pred_len"],
            },
            indent=1,
        )
        + "\n"
    )
    print(
        f"GQA accuracy={result['accuracy']:.4f} "
        f"({result['num_questions']} questions, mean len {result['mean_pred_len']:.2f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
