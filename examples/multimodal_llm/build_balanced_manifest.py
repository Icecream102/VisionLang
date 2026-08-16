"""Build a balanced yes/no recognition manifest from the COCO recognition data.

The original manifest is 2:1 no-biased (yes 11829 / no 23658).  Balancing keeps
all yes rows and samples an equal number of no rows (seeded), yielding an equal
yes/no train split so longer LoRA training cannot exploit the majority class.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    with Path(args.input).open() as handle:
        for line in handle:
            rows.append(json.loads(line))

    yes_rows = [row for row in rows if row["captions"][0].strip().lower() == "yes"]
    no_rows = [row for row in rows if row["captions"][0].strip().lower() == "no"]
    rng = random.Random(args.seed)
    no_sample = rng.sample(no_rows, len(yes_rows))
    balanced = yes_rows + no_sample
    rng.shuffle(balanced)

    with Path(args.output).open("w") as handle:
        for row in balanced:
            handle.write(json.dumps(row) + "\n")

    from collections import Counter

    counts = Counter(row["captions"][0].strip().lower() for row in balanced)
    print(f"input rows: {len(rows)} (yes {len(yes_rows)} / no {len(no_rows)})", flush=True)
    print(f"balanced rows: {len(balanced)} -> {dict(counts)}", flush=True)


if __name__ == "__main__":
    main()
