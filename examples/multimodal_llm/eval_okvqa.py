"""OK-VQA evaluation for a completed VLM checkpoint.

Loads a run (vision.pt / projector.pt / [llm_base.pt] / lora_adapter),
answers every OK-VQA val question, and scores with the standard VQA rule:
a question is correct when the predicted answer matches at least 3 of the 10
human answers after answer normalization.  Outputs per-question predictions,
aggregate accuracy, and an answer-length report.
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


CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "didnt": "didn't",
    "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hell": "he'll", "hes": "he's",
    "howd": "how'd", "howll": "how'll", "hows": "how's", "id": "i'd",
    "ill": "i'll", "im": "i'm", "ive": "i've", "isnt": "isn't", "itd": "it'd",
    "itll": "it'll", "its": "it's", "lets": "let's", "maam": "ma'am",
    "mightnt": "mightn't", "mightve": "might've", "mustnt": "mustn't",
    "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "shant": "shan't", "shed": "she'd", "shell": "she'll", "shes": "she's",
    "shouldve": "should've", "shouldnt": "shouldn't", "thats": "that's",
    "thered": "there'd", "theres": "there's", "theyd": "they'd", "theyll": "they'll",
    "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
    "wed": "we'd", "wedve": "we'd've", "well": "we'll", "were": "we're",
    "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",
    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
    "wheres": "where's", "whereve": "where've", "whod": "who'd", "wholl": "who'll",
    "whos": "who's", "whove": "who've", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't", "yall": "y'all", "youd": "you'd",
    "youll": "you'll", "youre": "you're", "youve": "you've",
}


def normalize_answer(answer: str) -> str:
    """VQA-style answer normalization."""
    # Keep only the first sentence/line; the 3B decoder often repeats the
    # answer and prepends "!" artifacts, which would break exact matching.
    first_line = answer.splitlines()[0].strip() if answer.splitlines() else answer
    answer = first_line
    answer = re.sub(r"^[^a-z0-9]+", "", answer, flags=re.IGNORECASE).strip()
    answer = answer.lower()
    for contraction, expansion in CONTRACTIONS.items():
        answer = answer.replace(contraction, expansion)
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    answer = re.sub(r"\b(i'm|i am)\b", "i am", answer)
    answer = re.sub(r"\b(s|t|d|ll|ve|re|m)\b", " ", answer)
    answer = re.sub(r"[^a-z0-9 ]", " ", answer)
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer


def vqa_accuracy(prediction: str, answers: List[str]) -> float:
    normalized = normalize_answer(prediction)
    if not normalized:
        return 0.0
    matched = sum(
        1 for answer in answers if normalize_answer(answer) == normalized
    )
    return 1.0 if matched >= 3 else 0.0


@torch.inference_mode()
def evaluate_okvqa(
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
        "max_new_tokens": 32,
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
            references = dataset.records[index]["captions"]
            predictions[index] = answer
            correct += vqa_accuracy(answer, references)
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
    result = evaluate_okvqa(
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
        f"OK-VQA accuracy={result['accuracy']:.4f} "
        f"({result['num_questions']} questions, mean len {result['mean_pred_len']:.2f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
