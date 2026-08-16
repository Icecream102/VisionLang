"""Fine-tune an MAE-initialized image encoder for low-resource image-text retrieval.

The manifest is JSONL. Each record contains an image path relative to
``--image-root`` and one or more captions:

    {"image": "train2014/COCO_train2014_000000.jpg", "captions": ["..."]}

Example:
    python -m examples.representation_learning.mae_dual_encoder \
      --image-root /datasets/coco --train-manifest data/train.jsonl \
      --val-manifest data/val.jsonl --mae-checkpoint outputs/mae_base/last.pt \
      --output-dir outputs/mae_coco_retrieval --epochs 20 --amp
"""

import argparse
import json
import logging
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from examples.representation_learning.mae_utils import (
    build_mae,
    capture_rng_state,
    restore_rng_state,
    set_seed,
)
from examples.representation_learning.retrieval_metrics import compute_retrieval_metrics


logger = logging.getLogger(__name__)

PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"


class CaptionManifest(Dataset):
    """Image/caption records supporting multiple positive captions per image."""

    def __init__(self, manifest: str, image_root: str, transform=None) -> None:
        self.image_root = Path(image_root)
        self.transform = transform
        self.records = []
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

    def __getitem__(self, index: int):
        record = self.records[index]
        path = self.image_root / record["image"]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return index, image, record["captions"]


class WordTokenizer:
    """A fitted word tokenizer with explicit, checkpointed vocabulary state."""

    def __init__(self, vocabulary: Dict[str, int], max_length: int) -> None:
        self.vocabulary = vocabulary
        self.max_length = max_length

    @classmethod
    def fit(
        cls, records: Sequence[Dict], max_length: int, min_frequency: int
    ) -> "WordTokenizer":
        counts = Counter(
            token
            for record in records
            for caption in record["captions"]
            for token in caption.lower().split()
        )
        vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for token, frequency in sorted(counts.items()):
            if frequency >= min_frequency:
                vocabulary[token] = len(vocabulary)
        return cls(vocabulary, max_length)

    def encode(self, captions: Sequence[str]) -> Tensor:
        result = torch.zeros((len(captions), self.max_length), dtype=torch.long)
        for row, caption in enumerate(captions):
            ids = [self.vocabulary.get(token, 1) for token in caption.lower().split()]
            result[row, : min(len(ids), self.max_length)] = torch.tensor(
                ids[: self.max_length], dtype=torch.long
            )
        return result


class HuggingFaceTokenizer:
    """Small adapter that gives a HuggingFace tokenizer the local encode API."""

    def __init__(self, model_name: str, max_length: int) -> None:
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            raise ValueError(f"Tokenizer {model_name} has no pad token.")
        self.pad_token_id = self.tokenizer.pad_token_id

    def encode(self, captions: Sequence[str]) -> Tensor:
        return self.tokenizer(
            list(captions),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"]


class TextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        max_length: int,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(
            vocab_size, hidden_dim, padding_idx=pad_token_id
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, max_length, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, token_ids: Tensor) -> Tensor:
        padding_mask = token_ids.eq(self.pad_token_id)
        x = (
            self.token_embedding(token_ids)
            + self.position_embedding[:, : token_ids.size(1)]
        )
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        lengths = (~padding_mask).sum(dim=1, keepdim=True).clamp_min(1)
        return self.norm((x * (~padding_mask).unsqueeze(-1)).sum(dim=1) / lengths)


class PretrainedTextEncoder(nn.Module):
    """Mean-pool a pretrained HuggingFace encoder for a comparable text baseline."""

    def __init__(self, model_name: str, pad_token_id: int) -> None:
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        self.pad_token_id = pad_token_id

    def forward(self, token_ids: Tensor) -> Tensor:
        attention_mask = token_ids.ne(self.pad_token_id)
        hidden = self.model(
            input_ids=token_ids, attention_mask=attention_mask
        ).last_hidden_state
        lengths = attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return (hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / lengths


class MAEDualEncoder(nn.Module):
    """A contrastive dual encoder whose visual tower is initialized from MAE."""

    def __init__(
        self,
        mae: nn.Module,
        text_encoder: nn.Module,
        text_hidden_dim: int,
        projection_dim: int,
    ) -> None:
        super().__init__()
        self.mae = mae
        self.text_encoder = text_encoder
        self.image_projection = nn.Linear(
            mae.embeddings.conv_projection.out_channels, projection_dim
        )
        self.text_projection = nn.Linear(text_hidden_dim, projection_dim)
        self.itm_head = nn.Sequential(
            nn.Linear(projection_dim * 3, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, 1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, images: Tensor) -> Tensor:
        embedded = self.mae.embeddings(images, apply_patch_drop=False).embeddings
        encoded = self.mae.encoder(embedded).last_hidden_state[:, 1:].mean(dim=1)
        return F.normalize(self.image_projection(encoded), dim=-1)

    def encode_text(self, token_ids: Tensor) -> Tensor:
        return F.normalize(self.text_projection(self.text_encoder(token_ids)), dim=-1)

    def matching_logits(self, image_features: Tensor, text_features: Tensor) -> Tensor:
        return self.itm_head(
            torch.cat(
                [image_features, text_features, image_features * text_features], dim=-1
            )
        ).squeeze(-1)

    def forward(
        self, images: Tensor, token_ids: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        image_features = self.encode_image(images)
        text_features = self.encode_text(token_ids)
        scale = self.logit_scale.exp().clamp(max=100)
        return image_features, text_features, scale


def contrastive_loss(
    image_features: Tensor, text_features: Tensor, scale: Tensor
) -> Tensor:
    targets = torch.arange(len(image_features), device=image_features.device)
    logits = scale * image_features @ text_features.T
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


def hard_negative_itm_loss(
    model: MAEDualEncoder, image_features: Tensor, text_features: Tensor
) -> Tensor:
    """Binary ITM loss with the most similar in-batch non-matching captions/images."""
    if len(image_features) < 2:
        return image_features.new_zeros(())
    similarities = image_features.detach() @ text_features.detach().T
    similarities.fill_diagonal_(float("-inf"))
    negative_text = similarities.argmax(dim=1)
    negative_image = similarities.argmax(dim=0)
    positive = model.matching_logits(image_features, text_features)
    image_negative = model.matching_logits(image_features, text_features[negative_text])
    text_negative = model.matching_logits(image_features[negative_image], text_features)
    logits = torch.cat((positive, image_negative, text_negative))
    labels = torch.cat(
        (
            torch.ones_like(positive),
            torch.zeros_like(image_negative),
            torch.zeros_like(text_negative),
        )
    )
    return F.binary_cross_entropy_with_logits(logits, labels)


def build_optimizer(
    model: MAEDualEncoder,
    lr: float,
    vision_lr: float | None,
    text_lr: float | None,
    weight_decay: float,
) -> AdamW:
    """Separate backbone and projection learning rates for stable transfer."""
    vision_parameters = list(model.mae.parameters())
    text_parameters = list(model.text_encoder.parameters())
    backbone_ids = {id(parameter) for parameter in vision_parameters + text_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in backbone_ids
    ]
    return AdamW(
        [
            {"params": vision_parameters, "lr": vision_lr or lr},
            {"params": text_parameters, "lr": text_lr or lr},
            {"params": head_parameters, "lr": lr},
        ],
        weight_decay=weight_decay,
    )


def build_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:
    if not 0 <= min_lr_ratio <= 1:
        raise ValueError("--min-lr-ratio must be in [0, 1].")

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * min(1.0, progress))
        )

    return LambdaLR(optimizer, schedule)


def train_collate(tokenizer):
    def collate(batch):
        _, images, caption_sets = zip(*batch)
        # Sampling one caption avoids treating paired captions of the same image as negatives.
        captions = [random.choice(captions) for captions in caption_sets]
        return torch.stack(images), tokenizer.encode(captions)

    return collate


@torch.inference_mode()
def evaluate(
    model: MAEDualEncoder,
    dataset: CaptionManifest,
    tokenizer,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    similarity_chunk_size: int,
) -> Dict[str, float]:
    def collate(batch):
        image_ids, images, captions_by_image = zip(*batch)
        captions, caption_image_ids = [], []
        for image_id, image_captions in zip(image_ids, captions_by_image):
            captions.extend(image_captions)
            caption_image_ids.extend([image_id] * len(image_captions))
        return torch.stack(images), captions, torch.tensor(caption_image_ids)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    image_features, text_features, caption_image_ids = [], [], []
    for images, captions, image_ids in loader:
        image_features.append(model.encode_image(images.to(device)).cpu())
        text_features.append(
            model.encode_text(tokenizer.encode(captions).to(device)).cpu()
        )
        caption_image_ids.append(image_ids)
    metrics = compute_retrieval_metrics(
        torch.cat(image_features),
        torch.cat(text_features),
        torch.cat(caption_image_ids),
        chunk_size=similarity_chunk_size,
        device=device,
    )
    return metrics.as_dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument(
        "--test-manifest",
        default=None,
        help="Optional held-out manifest evaluated once after training; never used for selection.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--train-limit-images",
        type=int,
        default=None,
        help="Deterministically subsample paired images for low-resource experiments.",
    )
    parser.add_argument("--mae-checkpoint", default=None)
    parser.add_argument("--model-size", choices=("tiny", "base"), default="tiny")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument(
        "--text-backbone",
        choices=("word", "bert"),
        default="word",
        help="word trains a local tokenizer/Transformer; bert loads a pretrained encoder.",
    )
    parser.add_argument("--text-model-name", default="bert-base-uncased")
    parser.add_argument("--text-hidden-dim", type=int, default=256)
    parser.add_argument("--text-layers", type=int, default=2)
    parser.add_argument("--text-heads", type=int, default=4)
    parser.add_argument("--max-text-length", type=int, default=32)
    parser.add_argument("--min-token-frequency", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--vision-lr",
        type=float,
        default=None,
        help="MAE learning rate; defaults to --lr.",
    )
    parser.add_argument(
        "--text-lr",
        type=float,
        default=None,
        help="Text-backbone learning rate; defaults to --lr.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Set to 0 to disable gradient clipping.",
    )
    parser.add_argument(
        "--itm-loss-weight",
        type=float,
        default=0.1,
        help="Weight of batch-hard image-text matching loss; zero disables ITM.",
    )
    parser.add_argument("--freeze-vision-epochs", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--similarity-chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def set_vision_trainable(model: MAEDualEncoder, enabled: bool) -> None:
    for parameter in model.mae.parameters():
        parameter.requires_grad = enabled


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    set_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    amp = args.amp and device.type == "cuda"
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(args.image_size / 0.875)),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_dataset = CaptionManifest(
        args.train_manifest, args.image_root, train_transform
    )
    if args.train_limit_images is not None:
        if args.train_limit_images <= 0:
            raise ValueError("--train-limit-images must be positive.")
        sample_size = min(args.train_limit_images, len(train_dataset.records))
        sampler = random.Random(args.seed)
        selected = sorted(
            sampler.sample(range(len(train_dataset.records)), sample_size)
        )
        train_dataset.records = [train_dataset.records[index] for index in selected]
        logger.info("Using %d paired images for low-resource alignment", sample_size)
    val_dataset = CaptionManifest(args.val_manifest, args.image_root, eval_transform)
    test_dataset = (
        CaptionManifest(args.test_manifest, args.image_root, eval_transform)
        if args.test_manifest
        else None
    )
    if args.itm_loss_weight < 0:
        raise ValueError("--itm-loss-weight must be non-negative.")
    if args.text_backbone == "word":
        tokenizer = WordTokenizer.fit(
            train_dataset.records, args.max_text_length, args.min_token_frequency
        )
        text_encoder = TextTransformer(
            len(tokenizer.vocabulary),
            args.text_hidden_dim,
            args.text_layers,
            args.text_heads,
            args.max_text_length,
        )
        tokenizer_state = {"type": "word", "vocabulary": tokenizer.vocabulary}
        text_hidden_dim = args.text_hidden_dim
    else:
        tokenizer = HuggingFaceTokenizer(args.text_model_name, args.max_text_length)
        text_encoder = PretrainedTextEncoder(
            args.text_model_name, tokenizer.pad_token_id
        )
        tokenizer_state = {"type": "bert", "model_name": args.text_model_name}
        text_hidden_dim = text_encoder.hidden_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    mae = build_mae(args.model_size, args.image_size, args.mask_ratio)
    if args.mae_checkpoint:
        checkpoint = torch.load(
            args.mae_checkpoint, map_location="cpu", weights_only=False
        )
        mae.load_state_dict(checkpoint["model"], strict=True)
        logger.info("Loaded MAE initialization from %s", args.mae_checkpoint)
    model = MAEDualEncoder(
        mae,
        text_encoder=text_encoder,
        text_hidden_dim=text_hidden_dim,
        projection_dim=args.projection_dim,
    ).to(device)
    optimizer = build_optimizer(
        model, args.lr, args.vision_lr, args.text_lr, args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch = 0
    resume_checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        resume_checkpoint = checkpoint
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        if checkpoint.get("tokenizer_state", {}).get("type") == "word":
            tokenizer = WordTokenizer(
                checkpoint["tokenizer_state"]["vocabulary"],
                checkpoint["max_text_length"],
            )
        elif "vocabulary" in checkpoint:  # backward compatibility with initial runs
            tokenizer = WordTokenizer(
                checkpoint["vocabulary"], checkpoint["max_text_length"]
            )
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = checkpoint["epoch"] + 1

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=train_collate(tokenizer),
    )
    scheduler = build_scheduler(
        optimizer,
        args.warmup_epochs * len(loader),
        args.epochs * len(loader),
        args.min_lr_ratio,
    )
    if resume_checkpoint and "scheduler" in resume_checkpoint:
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
    history_path = output_dir / "metrics.jsonl"
    best_mr = float("-inf")
    for epoch in range(start_epoch, args.epochs):
        set_vision_trainable(model, epoch >= args.freeze_vision_epochs)
        model.train()
        total_loss = 0.0
        total_itm_loss = 0.0
        for images, tokens in loader:
            images, tokens = images.to(device, non_blocking=True), tokens.to(
                device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                image_features, text_features, scale = model(images, tokens)
                contrastive = contrastive_loss(image_features, text_features, scale)
                itm = hard_negative_itm_loss(model, image_features, text_features)
                loss = contrastive + args.itm_loss_weight * itm
            scaler.scale(loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.detach().item()
            total_itm_loss += itm.detach().item()
        metrics = evaluate(
            model,
            val_dataset,
            tokenizer,
            args.batch_size,
            args.num_workers,
            device,
            args.similarity_chunk_size,
        )
        result = {
            "epoch": epoch + 1,
            "train_contrastive_loss": total_loss / len(loader),
            "train_itm_loss": total_itm_loss / len(loader),
            "vision_lr": optimizer.param_groups[0]["lr"],
            "text_lr": optimizer.param_groups[1]["lr"],
            "head_lr": optimizer.param_groups[2]["lr"],
            **metrics,
        }
        with history_path.open("a") as handle:
            handle.write(json.dumps(result) + "\n")
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "tokenizer_state": tokenizer_state,
            "vocabulary": getattr(tokenizer, "vocabulary", None),
            "max_text_length": tokenizer.max_length,
            "rng_state": capture_rng_state(),
        }
        torch.save(state, output_dir / "last.pt")
        if metrics["mean_recall"] >= best_mr:
            best_mr = metrics["mean_recall"]
            torch.save(state, output_dir / "best.pt")
        logger.info(
            "epoch=%d loss=%.4f itm=%.4f mR=%.2f",
            epoch + 1,
            result["train_contrastive_loss"],
            result["train_itm_loss"],
            metrics["mean_recall"],
        )

    if test_dataset is not None:
        # The test split is intentionally evaluated only once, after all model
        # selection has happened on --val-manifest.
        checkpoint = torch.load(
            output_dir / "best.pt", map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        test_metrics = evaluate(
            model,
            test_dataset,
            tokenizer,
            args.batch_size,
            args.num_workers,
            device,
            args.similarity_chunk_size,
        )
        test_result = {
            "checkpoint": "best.pt",
            "selection_manifest": str(Path(args.val_manifest).resolve()),
            "test_manifest": str(Path(args.test_manifest).resolve()),
            **test_metrics,
        }
        (output_dir / "test_metrics.json").write_text(
            json.dumps(test_result, indent=2) + "\n"
        )
        logger.info("held_out_test mR=%.2f", test_metrics["mean_recall"])


if __name__ == "__main__":
    main()
