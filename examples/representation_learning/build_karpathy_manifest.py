"""Create a retrieval JSONL manifest from Karpathy's COCO split annotation.

The commonly used ``dataset_coco.json`` contains 113,287 train/restval images,
5,000 validation images and 5,000 test images.  This utility preserves every
caption and writes split metadata next to the manifest so training and final
test evaluation cannot be confused.

Example:
    python -m examples.representation_learning.build_karpathy_manifest \
      --karpathy-json data/dataset_coco.json --splits train,restval \
      --output data/coco_karpathy_train.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


def parse_splits(value: str) -> List[str]:
    splits = [item.strip() for item in value.split(",") if item.strip()]
    if not splits:
        raise argparse.ArgumentTypeError("At least one split is required.")
    return splits


def build_records(annotation: Dict, splits: Iterable[str]) -> List[Dict]:
    requested = set(splits)
    records = []
    for item in annotation.get("images", []):
        if item.get("split") not in requested:
            continue
        captions = [sentence["raw"] for sentence in item.get("sentences", [])]
        if not captions or not item.get("filename"):
            continue
        filepath = item.get("filepath", "")
        records.append(
            {
                "image": str(Path(filepath) / item["filename"]),
                "captions": captions,
            }
        )
    if not records:
        raise ValueError(f"No captioned images found for splits: {sorted(requested)}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--karpathy-json", required=True)
    parser.add_argument(
        "--splits",
        type=parse_splits,
        required=True,
        help="Comma-separated Karpathy splits, e.g. train,restval or test.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.karpathy_json)
    records = build_records(json.loads(source.read_text()), args.splits)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata = {
        "dataset": "COCO Karpathy split",
        "source": str(source),
        "splits": args.splits,
        "num_images": len(records),
        "num_captions": sum(len(record["captions"]) for record in records),
        "split_counts": dict(Counter(args.splits)),
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Wrote {len(records)} image-caption records to {output}")


if __name__ == "__main__":
    main()
