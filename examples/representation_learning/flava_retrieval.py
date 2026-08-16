"""Evaluate pretrained FLAVA with the standard multi-caption retrieval protocol.

Example:
    python -m examples.representation_learning.flava_retrieval \
      --image-root /datasets/coco/val2014 \
      --annotations /datasets/coco/annotations/captions_val2014.json \
      --output outputs/flava_coco_5k.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CocoCaptions

from examples.flava.data.transforms import (
    default_image_pretraining_transforms,
    default_text_transform,
)
from examples.representation_learning.retrieval_metrics import compute_retrieval_metrics
from torchmultimodal.models.flava.model import flava_model


logger = logging.getLogger(__name__)


class CocoMultiCaptionDataset(Dataset):
    """COCO samples retaining every caption and its corresponding image index."""

    def __init__(self, image_root: str, annotations: str) -> None:
        self.dataset = CocoCaptions(root=image_root, annFile=annotations)
        _, self.image_transform = default_image_pretraining_transforms()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor, Sequence[str]]:
        image, captions = self.dataset[index]
        return index, self.image_transform(image)["image"], captions


def collate_multi_caption(
    batch: List[Tuple[int, torch.Tensor, Sequence[str]]]
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    image_indices, images, captions_by_image = zip(*batch)
    captions: List[str] = []
    caption_image_ids: List[int] = []
    for image_index, image_captions in zip(image_indices, captions_by_image):
        captions.extend(image_captions)
        caption_image_ids.extend([image_index] * len(image_captions))
    return (
        torch.stack(images),
        captions,
        torch.tensor(caption_image_ids, dtype=torch.long),
    )


@torch.inference_mode()
def encode_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    text_transform = default_text_transform()
    image_embeddings = []
    text_embeddings = []
    caption_image_ids = []
    model.eval()
    for batch_index, (images, captions, image_ids) in enumerate(loader):
        tokens = text_transform(captions)["input_ids"]
        _, image_features = model.encode_image(images.to(device), projection=True)
        _, text_features = model.encode_text(tokens.to(device), projection=True)
        image_embeddings.append(image_features.cpu())
        text_embeddings.append(text_features.cpu())
        caption_image_ids.append(image_ids)
        if batch_index % 20 == 0:
            logger.info("Encoded %d image batches", batch_index + 1)
    return (
        torch.cat(image_embeddings),
        torch.cat(text_embeddings),
        torch.cat(caption_image_ids),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True, help="COCO image directory")
    parser.add_argument("--annotations", required=True, help="COCO captions JSON")
    parser.add_argument("--output", required=True, help="Metrics JSON output path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional deterministic prefix for a fast smoke test.",
    )
    parser.add_argument("--similarity-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--device", default=None, help="Defaults to cuda when available"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    dataset: Dataset = CocoMultiCaptionDataset(args.image_root, args.annotations)
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError("--max-images must be positive.")
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_multi_caption,
    )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Loading pretrained FLAVA on %s", device)
    model = flava_model(pretrained=True).to(device)
    image_embeddings, text_embeddings, caption_image_ids = encode_dataset(
        model, loader, device
    )
    metrics = compute_retrieval_metrics(
        image_embeddings,
        text_embeddings,
        caption_image_ids,
        chunk_size=args.similarity_chunk_size,
        device=device,
    )
    result = {
        "dataset": "COCO captions",
        "num_images": len(image_embeddings),
        "num_captions": len(text_embeddings),
        "metrics": metrics.as_dict(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    logger.info("Saved metrics to %s", output)
    for name, value in metrics.as_dict().items():
        logger.info("%s: %.2f", name, value)


if __name__ == "__main__":
    main()
