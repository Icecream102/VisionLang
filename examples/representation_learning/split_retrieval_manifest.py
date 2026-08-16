"""Deterministically split a JSONL retrieval manifest into validation and test sets.

This is intended for COCO2017 when only the official ``val2017`` images are
available.  It splits by image record, never by caption, so every caption of
an image remains a positive for that same image.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def split_records(
    records: List[Dict], test_fraction: float, seed: int
) -> Tuple[List[Dict], List[Dict]]:
    if not 0 < test_fraction < 1:
        raise ValueError("--test-fraction must be in (0, 1).")
    if len(records) < 2:
        raise ValueError("At least two image records are required.")
    test_size = round(len(records) * test_fraction)
    test_size = min(max(test_size, 1), len(records) - 1)
    test_indices = set(random.Random(seed).sample(range(len(records)), test_size))
    validation = [
        record for index, record in enumerate(records) if index not in test_indices
    ]
    test = [record for index, record in enumerate(records) if index in test_indices]
    return validation, test


def write_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--test-output", required=True)
    parser.add_argument("--test-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line) for line in Path(args.input).read_text().splitlines() if line
    ]
    validation, test = split_records(records, args.test_fraction, args.seed)
    val_output, test_output = Path(args.val_output), Path(args.test_output)
    write_jsonl(validation, val_output)
    write_jsonl(test, test_output)
    metadata = {
        "source": str(Path(args.input).resolve()),
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "num_validation_images": len(validation),
        "num_test_images": len(test),
        "split_unit": "image",
    }
    val_output.with_suffix(val_output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Wrote {len(validation)} validation and {len(test)} test image records")


if __name__ == "__main__":
    main()
