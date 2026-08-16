"""COCO captioning evaluation with pycocoevalcap."""

from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from examples.multimodal_llm.data import LlavaCaptionDataset, collate_captioning
from examples.multimodal_llm.model import LlavaCaptionModel


logger = logging.getLogger(__name__)


def normalize_caption(text: str) -> str:
    """Clean decoding artifacts before metric scoring.

    The Qwen2-0.5B checkpoint occasionally emits leading punctuation (``!!``)
    and, more importantly, continues past the first sentence (verbose
    multi-sentence descriptions) while COCO references are single short
    sentences.  We strip leading non-letter punctuation, collapse whitespace,
    and keep only the first sentence of the decoded text.  The raw text stays
    in prediction files; only the scored copy is normalized.
    """
    text = text.strip()
    # Drop leading punctuation artifacts like "!!", "!", "-" that the base
    # model emits at the start of generation.
    text = re.sub(r"^[^A-Za-z0-9]+", "", text).strip()
    # Collapse repeated newlines/whitespace into single spaces.
    text = re.sub(r"\s+", " ", text).strip()
    # Keep only the first sentence: COCO references are single sentences and
    # the 0.5B decoder continues with verbose follow-up sentences.
    match = re.search(r"^.*?[.!?](?:\s|$)", text)
    if match:
        return match.group(0).strip()
    return text


@torch.inference_mode()
def generate_captions(
    model: LlavaCaptionModel,
    dataset: LlavaCaptionDataset,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 4,
    max_new_tokens: int = 64,
    num_beams: int = 1,
    limit: int | None = None,
    repetition_penalty: float = 1.3,
    no_repeat_ngram_size: int = 3,
) -> Dict[int, Dict]:
    """Generate one caption per image; returns predictions and ground truth."""
    if limit is not None:
        dataset = LlavaCaptionDataset(
            manifest=dataset.manifest,
            image_root=str(dataset.image_root),
            tokenizer=tokenizer,
            transform=dataset.transform,
            seed=dataset.seed,
            max_length=dataset.max_length,
            limit=limit,
            eval_mode=True,
            text_only=dataset.text_only,
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
    predictions: Dict[int, List[str]] = {}
    ground_truth: Dict[int, List[str]] = {}
    generator_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }
    start = 0
    for batch in loader:
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prompt_lengths = attention_mask.sum(dim=1).long()
        output_ids = model.generate(
            images,
            input_ids,
            attention_mask,
            **generator_kwargs,
        )
        for offset, (row, prompt_length) in enumerate(
            zip(output_ids, prompt_lengths)
        ):
            # The <image> placeholder expands by (image_token_count - 1) tokens.
            expanded_length = (
                prompt_length
                if model.text_only
                else prompt_length + (model.image_token_count - 1)
            )
            generated = row[expanded_length:].cpu()
            caption = normalize_caption(
                tokenizer.decode(generated, skip_special_tokens=True)
            )
            index = start + offset
            predictions[index] = [caption] if caption else [" "]
        ground_truth.update(
            {
                start + offset: list(dataset.records[start + offset]["captions"])
                for offset in range(images.size(0))
            }
        )
        start += images.size(0)
    return predictions, ground_truth
def compute_metrics(
    predictions: Dict[int, List[str]],
    ground_truth: Dict[int, List[str]],
) -> Dict[str, float]:
    """Compute BLEU/METEOR/ROUGE-L/CIDEr/SPICE via pycocoevalcap."""
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        gt_path = tmp_path / "gt.json"
        res_path = tmp_path / "res.json"
        images = [{"id": index} for index in predictions]
        annotations = [
            {"image_id": index, "id": index * 100 + j, "caption": caption}
            for index, captions in ground_truth.items()
            for j, caption in enumerate(captions)
        ]
        gt_path.write_text(json.dumps({"images": images, "annotations": annotations}))
        res_path.write_text(
            json.dumps(
                [
                    {"image_id": index, "caption": captions[0]}
                    for index, captions in sorted(predictions.items())
                ]
            )
        )
        coco = COCO(str(gt_path))
        coco_result = coco.loadRes(str(res_path))
        evaluator = COCOEvalCap(coco, coco_result)
        evaluator.evaluate()
    return {key: float(value) for key, value in evaluator.eval.items()}


def evaluate_coco_captions(
    model: LlavaCaptionModel,
    dataset: LlavaCaptionDataset,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    **kwargs,
) -> Dict[str, float]:
    predictions, ground_truth = generate_captions(
        model, dataset, tokenizer, device, **kwargs
    )
    metrics = compute_metrics(predictions, ground_truth)
    logger.info(
        "CIDEr=%.2f BLEU-4=%.2f METEOR=%.2f ROUGE-L=%.2f",
        metrics.get("CIDEr", float("nan")),
        metrics.get("Bleu_4", float("nan")),
        metrics.get("METEOR", float("nan")),
        metrics.get("ROUGE_L", float("nan")),
    )
    return metrics


@torch.inference_mode()
def evaluate_recognition_accuracy(
    model: LlavaCaptionModel,
    dataset: LlavaCaptionDataset,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 4,
    limit: int | None = None,
) -> Dict[str, float]:
    """Accuracy for yes/no visual-recognition questions."""
    if limit is not None:
        dataset = LlavaCaptionDataset(
            manifest=dataset.manifest,
            image_root=str(dataset.image_root),
            tokenizer=tokenizer,
            transform=dataset.transform,
            seed=dataset.seed,
            max_length=dataset.max_length,
            limit=limit,
            eval_mode=True,
            text_only=dataset.text_only,
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
        output_ids = model.generate(images, input_ids, attention_mask, **generator_kwargs)
        for offset, (row, prompt_length) in enumerate(zip(output_ids, prompt_lengths)):
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
    metrics = {
        "accuracy": correct / total if total else float("nan"),
        "yes_accuracy": yes_correct / yes_total if yes_total else float("nan"),
        "no_accuracy": no_correct / no_total if no_total else float("nan"),
    }
    logger.info("recognition accuracy=%.4f", metrics["accuracy"])
    return metrics


@torch.inference_mode()
def evaluate_mc_accuracy(
    model: LlavaCaptionModel,
    dataset: LlavaCaptionDataset,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 4,
    limit: int | None = None,
) -> Dict[str, float]:
    """Accuracy for multiple-choice visual-recognition questions.

    Questions are formatted as "Which object is shown in this image? A) ...
    B) ... C) ... D) ..." and the expected answer is a single letter stored in
    the record's ``captions`` field.  The first A-D letter in the generated
    answer is compared against the ground truth.
    """
    if limit is not None:
        dataset = LlavaCaptionDataset(
            manifest=dataset.manifest,
            image_root=str(dataset.image_root),
            tokenizer=tokenizer,
            transform=dataset.transform,
            seed=dataset.seed,
            max_length=dataset.max_length,
            limit=limit,
            eval_mode=True,
            text_only=dataset.text_only,
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
        "repetition_penalty": 1.2,
    }
    correct = total = 0
    per_letter = {}
    start = 0
    for batch in loader:
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prompt_lengths = attention_mask.sum(dim=1).long()
        output_ids = model.generate(images, input_ids, attention_mask, **generator_kwargs)
        for offset, (row, prompt_length) in enumerate(zip(output_ids, prompt_lengths)):
            expanded_length = (
                prompt_length
                if model.text_only
                else prompt_length + (model.image_token_count - 1)
            )
            answer = tokenizer.decode(
                row[expanded_length:], skip_special_tokens=True
            ).strip().upper()
            match = re.search(r"\b([A-D])\b", answer)
            prediction = match.group(1) if match else ""
            expected = dataset.records[start + offset]["captions"][0].strip().upper()
            total += 1
            correct += prediction == expected
            per_letter.setdefault(expected, [0, 0])[1] += 1
            per_letter[expected][0] += prediction == expected
        start += images.size(0)
    metrics = {
        "accuracy": correct / total if total else float("nan"),
    }
    for letter, (hits, count) in sorted(per_letter.items()):
        metrics[f"acc_{letter}"] = hits / count if count else float("nan")
    logger.info("multiple-choice accuracy=%.4f", metrics["accuracy"])
    return metrics
