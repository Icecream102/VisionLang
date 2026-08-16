"""COCO caption instruction-tuning data with deterministic caption rotation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import PreTrainedTokenizerBase

from examples.multimodal_llm.model import IMAGE_TOKEN


PROMPT_TEMPLATES = [
    "Describe this image in one sentence.",
    "What is shown in this image?",
    "Please provide a short description of this image.",
    "What do you see in this picture?",
    "Give a brief caption for this image.",
    "Describe the scene in this photo.",
]


class LlavaCaptionDataset(Dataset):
    """Records with image path + captions; one caption selected per epoch."""

    def __init__(
        self,
        manifest: str,
        image_root: str,
        tokenizer: PreTrainedTokenizerBase,
        transform=None,
        seed: int = 42,
        max_length: int = 256,
        limit: int | None = None,
        eval_mode: bool = False,
        text_only: bool = False,
    ) -> None:
        self.image_root = Path(image_root)
        self.transform = transform
        self.tokenizer = tokenizer
        self.seed = seed
        self.max_length = max_length
        self.eval_mode = eval_mode
        self.text_only = text_only
        self.manifest = str(Path(manifest))
        self.records: List[Dict] = []
        with Path(manifest).open() as handle:
            for line in handle:
                record = json.loads(line)
                if not isinstance(record.get("image"), str) or not record.get("captions"):
                    raise ValueError(f"Invalid record: {record}")
                self.records.append(record)
        if limit is not None:
            self.records = self.records[:limit]
        if not self.records:
            raise ValueError(f"No records found in {manifest}")

    def _rotation(self, index: int) -> int:
        # Deterministic per seed: each seed sees a different caption/template.
        return (index + self.seed) % len(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        path = self.image_root / record["image"]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        rotation = self._rotation(index)
        captions = record["captions"]
        caption = captions[rotation % len(captions)].strip()
        prompt = record.get("question") or PROMPT_TEMPLATES[
            rotation % len(PROMPT_TEMPLATES)
        ]
        if self.eval_mode:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt if self.text_only else f"{IMAGE_TOKEN}\n{prompt}",
                },
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            ids = self.tokenizer(
                text,
                add_special_tokens=False,
                max_length=self.max_length,
                truncation=True,
            ).input_ids
            return {
                "image": image,
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.ones(len(ids), dtype=torch.long),
                "prompt": prompt,
                "caption": caption,
            }
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt if self.text_only else f"{IMAGE_TOKEN}\n{prompt}",
            },
            {"role": "assistant", "content": caption},
        ]
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        ).input_ids
        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        ).input_ids
        labels = [-100] * len(full_ids)
        labels[len(prompt_ids) :] = full_ids[len(prompt_ids) :]
        return {
            "image": image,
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "caption": caption,
        }


def collate_captioning(batch: Sequence[Dict]) -> Dict[str, Tensor]:
    images = torch.stack([item["image"] for item in batch])
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    result: Dict[str, Tensor] = {
        "images": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if "labels" in batch[0]:
        result["labels"] = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-100,
        )
    return result


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size / 0.875)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
