"""Build a 4-option multiple-choice visual-recognition task from COCO.

Each record asks "Which object is shown in this image? A) <present> B) <absent>
C) <absent> D) <absent>" and the expected answer is the letter of the category
that is actually present in the image.  Distractors are hard negatives sampled
from categories absent from the image.  The output reuses the caption-manifest
schema (``question`` + ``captions``), so the same training loop can consume it
with ``--mc-eval``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Set


LETTERS = ["A", "B", "C", "D"]


def load_instances(instances_json: str) -> Dict[str, Dict]:
    data = json.loads(Path(instances_json).read_text())
    category_id_to_name = {item["id"]: item["name"] for item in data["categories"]}
    image_id_to_file = {item["id"]: item["file_name"] for item in data["images"]}
    image_to_categories: Dict[int, Set[str]] = {}
    for annotation in data["annotations"]:
        image_to_categories.setdefault(annotation["image_id"], set()).add(
            category_id_to_name[annotation["category_id"]]
        )
    return {
        "file_names": image_id_to_file,
        "categories": image_to_categories,
        "category_names": sorted(category_id_to_name.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-json", required=True)
    parser.add_argument("--image-prefix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-images", type=int, default=None)
    args = parser.parse_args()

    info = load_instances(args.instances_json)
    file_names = info["file_names"]
    categories = info["categories"]
    all_categories = info["category_names"]
    rng = random.Random(args.seed)
    image_ids = sorted(categories)
    if args.limit_images is not None:
        image_ids = rng.sample(image_ids, min(args.limit_images, len(image_ids)))

    rows = []
    for image_id in image_ids:
        present = categories[image_id]
        answer = rng.choice(sorted(present))
        distractors = rng.sample(
            [name for name in all_categories if name not in present],
            min(3, len(all_categories) - len(present)),
        )
        if len(distractors) < 3:
            continue
        options = [answer] + distractors
        rng.shuffle(options)
        answer_letter = LETTERS[options.index(answer)]
        option_text = " ".join(
            f"{letter}) {name}" for letter, name in zip(LETTERS, options)
        )
        rows.append(
            {
                "image": f"{args.image_prefix}/{file_names[image_id]}",
                "question": f"Which object is shown in this image? {option_text}",
                "captions": [answer_letter],
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} records to {output}")


if __name__ == "__main__":
    main()
