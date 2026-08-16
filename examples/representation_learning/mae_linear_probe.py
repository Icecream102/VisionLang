"""Measure MAE transfer quality with a frozen-encoder linear probe.

Example:
    python -m examples.representation_learning.mae_linear_probe \
      --train-root /datasets/imagenet100/train --val-root /datasets/imagenet100/val \
      --checkpoint outputs/mae_tiny/last.pt --output outputs/mae_tiny/linear_probe.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from examples.representation_learning.mae_utils import build_mae, encoder_dim, set_seed
from examples.representation_learning.imagefolder_utils import stratified_split_indices
from torchmultimodal.transforms.mae_transform import ImageEvalTransform


logger = logging.getLogger(__name__)


class LinearProbe(nn.Module):
    """Global-average-pool a frozen MAE encoder, then train a single linear head."""

    def __init__(self, encoder: nn.Module, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        # MAE masks patches in train mode. A probe must use the full frozen encoder.
        self.encoder.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            encoder_output = self.encoder(images).encoder_output.last_hidden_state
            features = encoder_output[:, 1:].mean(dim=1)
        return self.classifier(features)


def topk_correct(logits: torch.Tensor, labels: torch.Tensor, k: int) -> torch.Tensor:
    predicted = logits.topk(k, dim=1).indices
    return predicted.eq(labels[:, None]).any(dim=1).float().sum()


@torch.inference_mode()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    correct1, correct5, total = 0.0, 0.0, 0
    for images, labels in loader:
        labels = labels.to(device)
        logits = model(images.to(device, non_blocking=True))
        correct1 += topk_correct(logits, labels, 1).item()
        correct5 += topk_correct(logits, labels, min(5, logits.shape[1])).item()
        total += len(labels)
    return 100 * correct1 / total, 100 * correct5 / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-size", choices=("tiny", "base"), default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--mask-ratio", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    set_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    model_size = args.model_size or checkpoint_args.get("model_size")
    image_size = args.image_size or checkpoint_args.get("image_size")
    mask_ratio = args.mask_ratio or checkpoint_args.get("mask_ratio")
    if model_size is None or image_size is None or mask_ratio is None:
        raise ValueError(
            "Checkpoint lacks model metadata; provide model-size, image-size and mask-ratio."
        )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    if args.val_root and args.val_fraction is not None:
        raise ValueError("Use either --val-root or --val-fraction, not both.")
    if not args.val_root and args.val_fraction is None:
        raise ValueError("Provide --val-root or --val-fraction.")
    train_dataset = ImageFolder(args.train_root, transform=train_transform)
    if args.val_root:
        val_dataset = ImageFolder(
            args.val_root, transform=ImageEvalTransform(image_size)
        )
        if train_dataset.classes != val_dataset.classes:
            raise ValueError(
                "Train and validation ImageFolder classes must have identical ordering."
            )
    else:
        train_indices, val_indices = stratified_split_indices(
            train_dataset.targets, args.val_fraction, args.seed
        )
        val_dataset = Subset(
            ImageFolder(args.train_root, transform=ImageEvalTransform(image_size)),
            val_indices,
        )
        train_dataset = Subset(train_dataset, train_indices)
    train_loader = DataLoader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    mae = build_mae(model_size, image_size, mask_ratio)
    mae.load_state_dict(checkpoint["model"])
    num_classes = len(
        train_dataset.dataset.classes
        if isinstance(train_dataset, Subset)
        else train_dataset.classes
    )
    model = LinearProbe(mae, encoder_dim(model_size), num_classes).to(device)
    optimizer = AdamW(
        model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    best_top1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            logits = model(images.to(device, non_blocking=True))
            loss = criterion(logits, labels.to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        top1, top5 = evaluate(model, val_loader, device)
        best_top1 = max(best_top1, top1)
        logger.info("epoch=%d top1=%.2f top5=%.2f", epoch + 1, top1, top5)

    result = {
        "best_top1": best_top1,
        "final_top1": top1,
        "final_top5": top5,
        "epochs": args.epochs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    logger.info("Saved linear-probe result to %s", output)


if __name__ == "__main__":
    main()
