"""Evaluate a pretrained CLIP model on a JSONL image-text retrieval manifest.

This is a zero-shot reference point for the MAE-initialized dual encoder.  It
uses the same multi-positive I2T/T2I metrics and exactly the same manifest as
the trainable model, so values are directly comparable.

Example:
    python -m examples.representation_learning.clip_retrieval_baseline \
      --image-root /datasets/coco --manifest data/coco_karpathy_test.jsonl \
      --output outputs/clip_vit_b32_karpathy_test.json

``transformers`` is intentionally imported only when ``main`` runs, allowing
the rest of the project and unit tests to work without downloading CLIP.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from examples.representation_learning.retrieval_metrics import compute_retrieval_metrics


logger = logging.getLogger(__name__)


class CaptionManifestImages(Dataset):
    """Manifest reader that returns RGB PIL images for CLIP preprocessing."""

    def __init__(self, manifest: str, image_root: str) -> None:
        self.image_root = Path(image_root)
        self.records: List[Dict] = []
        with Path(manifest).open() as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                if not isinstance(record.get("image"), str) or not record.get(
                    "captions"
                ):
                    raise ValueError(f"Invalid record at {manifest}:{line_number}")
                self.records.append(record)
        if not self.records:
            raise ValueError(f"No records found in {manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[int, Image.Image, Sequence[str]]:
        record = self.records[index]
        with Image.open(self.image_root / record["image"]) as source:
            image = source.convert("RGB").copy()
        return index, image, record["captions"]


def evaluation_collate(
    batch: Sequence[Tuple[int, Image.Image, Sequence[str]]],
) -> Tuple[List[Image.Image], List[str], Tensor]:
    """Flatten captions while retaining their corresponding image indices."""
    images, captions, caption_image_ids = [], [], []
    for image_id, image, image_captions in batch:
        images.append(image)
        captions.extend(image_captions)
        caption_image_ids.extend([image_id] * len(image_captions))
    return images, captions, torch.tensor(caption_image_ids, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--similarity-chunk-size", type=int, default=1024)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    # Imported lazily because model weights are downloaded only for this baseline.
    from transformers import CLIPModel, CLIPProcessor

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name).to(device).eval()
    dataset = CaptionManifestImages(args.manifest, args.image_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=evaluation_collate,
        pin_memory=device.type == "cuda",
    )

    all_images, all_texts, all_caption_ids = [], [], []
    for images, captions, caption_image_ids in loader:
        image_inputs = processor.image_processor(images=images, return_tensors="pt")
        text_inputs = processor.tokenizer(
            captions, padding=True, truncation=True, return_tensors="pt"
        )
        image_features = model.get_image_features(
            pixel_values=image_inputs["pixel_values"].to(device, non_blocking=True)
        )
        text_features = model.get_text_features(
            input_ids=text_inputs["input_ids"].to(device, non_blocking=True),
            attention_mask=text_inputs["attention_mask"].to(device, non_blocking=True),
        )
        all_images.append(F.normalize(image_features, dim=-1).cpu())
        all_texts.append(F.normalize(text_features, dim=-1).cpu())
        all_caption_ids.append(caption_image_ids)

    metrics = compute_retrieval_metrics(
        torch.cat(all_images),
        torch.cat(all_texts),
        torch.cat(all_caption_ids),
        chunk_size=args.similarity_chunk_size,
        device=device,
    ).as_dict()
    result = {
        "model": args.model_name,
        "manifest": str(Path(args.manifest).resolve()),
        "num_images": len(dataset),
        "num_captions": int(sum(len(record["captions"]) for record in dataset.records)),
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    logger.info(
        "Saved CLIP retrieval metrics to %s; mR=%.2f", output, metrics["mean_recall"]
    )


if __name__ == "__main__":
    main()
