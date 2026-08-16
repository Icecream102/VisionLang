"""Skeleton of a multimodal pretraining data pipeline (interview material).

This module is a *design skeleton*, not a production data pipeline.  Each
function has a clear input/output contract and validation hook so the design
can be discussed and incrementally filled in.  Nothing here has been run on
real large-scale data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np


@dataclass
class Sample:
    image_path: str
    caption: str
    meta: Dict = field(default_factory=dict)


@dataclass
class FilterReport:
    name: str
    kept: int
    dropped: int
    stats: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. Cleaning layer
# ---------------------------------------------------------------------------


def clean_caption(text: str) -> str:
    """Normalize a caption: strip HTML, collapse whitespace, dedupe n-grams."""
    import html
    import re

    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def caption_repetition_rate(text: str, n: int = 3) -> float:
    """Fraction of repeated n-grams; high values indicate degenerate captions."""
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - len(set(grams)) / max(1, len(grams))


def clean_and_filter(
    samples: Iterable[Sample],
    min_chars: int = 12,
    max_repetition: float = 0.5,
    report: Optional[FilterReport] = None,
) -> List[Sample]:
    """Cleaning pass over raw samples.  Returns kept samples and (optionally)
    populates a report with drop statistics."""
    kept, dropped = [], 0
    drop_reasons: Dict[str, int] = {}
    for sample in samples:
        caption = clean_caption(sample.caption)
        sample.caption = caption
        if len(caption) < min_chars:
            dropped += 1
            drop_reasons["too_short"] = drop_reasons.get("too_short", 0) + 1
            continue
        if caption_repetition_rate(caption) > max_repetition:
            dropped += 1
            drop_reasons["repetitive"] = drop_reasons.get("repetitive", 0) + 1
            continue
        kept.append(sample)
    if report is not None:
        report.kept = len(kept)
        report.dropped = dropped
        report.stats["drop_reasons"] = drop_reasons
    return kept


# ---------------------------------------------------------------------------
# 2. Dedup layer
# ---------------------------------------------------------------------------


def perceptual_hash(image_array: np.ndarray, size: int = 8) -> str:
    """Simplified dHash: grayscale -> resize -> neighbor-difference bits."""
    from PIL import Image

    gray = Image.fromarray(image_array).convert("L").resize((size + 1, size))
    pixels = np.asarray(gray, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten().astype(np.uint8)
    return "".join(str(int(bit)) for bit in bits)


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def dedup_images(
    image_arrays: List[np.ndarray],
    threshold: int = 8,
    report: Optional[FilterReport] = None,
) -> List[int]:
    """Greedy dedup by dHash hamming distance.  Returns kept indices."""
    hashes = [perceptual_hash(img) for img in image_arrays]
    kept_indices: List[int] = []
    for i, h in enumerate(hashes):
        if all(hamming(h, hashes[j]) > threshold for j in kept_indices):
            kept_indices.append(i)
    if report is not None:
        report.kept = len(kept_indices)
        report.dropped = len(image_arrays) - len(kept_indices)
        report.stats["threshold"] = threshold
    return kept_indices


def semantic_dedup_keep_top_k(
    scores: np.ndarray,
    cluster_ids: np.ndarray,
    keep_ratio: float = 0.8,
    seed: int = 42,
) -> List[int]:
    """Within each semantic cluster keep the top-scoring fraction (by quality
    score), deterministically seeded.  Returns kept indices."""
    rng = np.random.default_rng(seed)
    kept: List[int] = []
    for cluster in np.unique(cluster_ids):
        idx = np.where(cluster_ids == cluster)[0]
        order = idx[np.argsort(-scores[idx])]
        keep_n = max(1, int(round(len(order) * keep_ratio)))
        chosen = order[:keep_n]
        kept.extend(rng.permutation(chosen).tolist())
    return sorted(kept)


# ---------------------------------------------------------------------------
# 3. Quality / relevance scoring layer (interfaces; CLIP not bundled here)
# ---------------------------------------------------------------------------


@dataclass
class QualityScorer:
    """Hook points for a CLIP scorer / OCR filter / caption-quality model."""

    clip_score_fn: Optional[Callable[[Sample], float]] = None
    ocr_fn: Optional[Callable[[Sample], float]] = None

    def score(self, sample: Sample) -> float:
        clip = self.clip_score_fn(sample) if self.clip_score_fn else 1.0
        ocr = self.ocr_fn(sample) if self.ocr_fn else 1.0
        return float(clip * ocr)


def filter_by_clip_score(
    samples: List[Sample],
    scorer: QualityScorer,
    min_score: float,
    report: Optional[FilterReport] = None,
) -> List[Sample]:
    kept = [s for s in samples if scorer.score(s) >= min_score]
    if report is not None:
        report.kept = len(kept)
        report.dropped = len(samples) - len(kept)
    return kept


# ---------------------------------------------------------------------------
# 4. Mixture ratio + curriculum + audit
# ---------------------------------------------------------------------------


@dataclass
class MixtureConfig:
    image_text_pairs: float = 1.0
    interleaved: float = 1.0
    text_only: float = 1.0
    instruction: float = 0.5

    def normalize(self) -> Dict[str, float]:
        total = (
            self.image_text_pairs
            + self.interleaved
            + self.text_only
            + self.instruction
        )
        return {
            "image_text_pairs": self.image_text_pairs / total,
            "interleaved": self.interleaved / total,
            "text_only": self.text_only / total,
            "instruction": self.instruction / total,
        }


def make_manifest(
    samples: List[Sample], output: Path, report: Optional[FilterReport] = None
) -> Path:
    """Versioned JSONL manifest (same schema as the project's training data)."""
    payload = "".join(
        json.dumps(
            {
                "image": s.image_path,
                "question": s.meta.get("question", ""),
                "captions": [s.caption],
            }
        )
        + "\n"
        for s in samples
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    target = output.with_name(f"{output.stem}_{digest}.jsonl")
    target.write_text(payload)
    if report is not None:
        report.stats["manifest_digest"] = digest
    return target


def audit_snapshot(
    samples: List[Sample], k: int = 200, seed: int = 42
) -> Dict[str, object]:
    """Deterministic audit stats for a data version (for A/B benchmarking)."""
    rng = np.random.default_rng(seed)
    sampled = rng.choice(len(samples), size=min(k, len(samples)), replace=False)
    lengths = [len(s.caption.split()) for s in samples]
    return {
        "num_samples": len(samples),
        "mean_caption_len": float(np.mean(lengths)),
        "std_caption_len": float(np.std(lengths)),
        "audit_indices": sampled.tolist(),
    }
