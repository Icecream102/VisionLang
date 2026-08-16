"""Build a yes/no visual-recognition task from COCO instances annotations.

Each record asks "Is there a <category> in this image?" with the expected
answer "yes" (category present) or "no" (hard-negative category absent from
the image).  The output reuses the caption manifest schema plus a
``question`` field, so the same training loop can consume it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Set


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
    parser.add_argument("--negatives-per-image", type=int, default=2)
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
        positive = rng.choice(sorted(present))
        negatives = rng.sample(
            [name for name in all_categories if name not in present],
            min(args.negatives_per_image, len(all_categories) - len(present)),
        )
        file_name = file_names[image_id]
        rows.append(
            {
                "image": f"{args.image_prefix}/{file_name}",
                "question": f"Is there a {positive} in this image?",
                "captions": ["yes"],
            }
        )
        for negative in negatives:
            rows.append(
                {
                    "image": f"{args.image_prefix}/{file_name}",
                    "question": f"Is there a {negative} in this image?",
                    "captions": ["no"],
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
