"""Create a deterministic image-caption dataset for end-to-end smoke tests.

The generated data verifies pipeline correctness only; it must never be reported as
an image-text retrieval benchmark.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw


PALETTE: List[Tuple[int, int, int]] = [
    (220, 20, 60),
    (30, 144, 255),
    (50, 205, 50),
    (255, 165, 0),
    (138, 43, 226),
    (0, 206, 209),
    (255, 105, 180),
    (154, 205, 50),
    (255, 99, 71),
    (70, 130, 180),
    (199, 21, 133),
    (218, 165, 32),
]


def draw_pattern(
    class_index: int, split: str, image_size: int, seed: int
) -> Image.Image:
    rng = random.Random(seed)
    image = Image.new("RGB", (image_size, image_size), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    color = PALETTE[class_index % len(PALETTE)]
    margin = image_size // 5 + rng.randint(-8, 8)
    if class_index % 3 == 0:
        draw.rectangle(
            (margin, margin, image_size - margin, image_size - margin), fill=color
        )
    elif class_index % 3 == 1:
        draw.ellipse(
            (margin, margin, image_size - margin, image_size - margin), fill=color
        )
    else:
        draw.polygon(
            [
                (image_size // 2, margin),
                (image_size - margin, image_size - margin),
                (margin, image_size - margin),
            ],
            fill=color,
        )
    stripe = 8 + (class_index % 4) * 6
    for offset in range(-image_size, image_size * 2, stripe * 3):
        draw.line(
            (offset, 0, offset + image_size, image_size), fill=(35, 35, 35), width=2
        )
    return image


def write_manifest(records: List[Dict], path: Path) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classes", type=int, default=12)
    parser.add_argument("--train-images-per-class", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.classes < 10:
        raise ValueError("Use at least 10 classes so Recall@10 is defined.")
    if args.train_images_per_class <= 0:
        raise ValueError("--train-images-per-class must be positive.")
    root = Path(args.output_dir)
    train_records: List[Dict] = []
    val_records: List[Dict] = []
    for split, images_per_class, records in (
        ("train", args.train_images_per_class, train_records),
        ("val", 1, val_records),
    ):
        for class_index in range(args.classes):
            class_name = f"class_{class_index:02d}"
            directory = root / split / class_name
            directory.mkdir(parents=True, exist_ok=True)
            for image_index in range(images_per_class):
                filename = f"{image_index:03d}.png"
                image = draw_pattern(
                    class_index,
                    split,
                    args.image_size,
                    args.seed
                    + 1000 * class_index
                    + image_index
                    + (100_000 if split == "val" else 0),
                )
                image.save(directory / filename)
                records.append(
                    {
                        "image": str(Path(split) / class_name / filename),
                        "captions": [
                            f"synthetic pattern category {class_name}",
                            f"geometric symbol label {class_name}",
                        ],
                    }
                )
    write_manifest(train_records, root / "train.jsonl")
    write_manifest(val_records, root / "val.jsonl")
    print(
        f"Created synthetic smoke data at {root}: "
        f"{len(train_records)} train images, {len(val_records)} validation images."
    )


if __name__ == "__main__":
    main()
