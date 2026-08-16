"""Aggregate repeated retrieval runs into mean/std metrics for an ablation table.

Example:
    python -m examples.representation_learning.summarize_retrieval_runs \
      --run random=outputs/karpathy_random_seed42/metrics.jsonl \
      --run random=outputs/karpathy_random_seed43/metrics.jsonl \
      --run mae_finetune=outputs/karpathy_mae_finetune_seed42/metrics.jsonl \
      --output outputs/karpathy_ablation_summary.json
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Each --run must be NAME=PATH.")
    name, path = value.split("=", maxsplit=1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Each --run must be NAME=PATH.")
    return name, Path(path)


def selected_metrics(path: Path, select: str) -> Dict[str, float]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No metric rows found in {path}")
    row = (
        max(rows, key=lambda item: item["mean_recall"])
        if select == "best"
        else rows[-1]
    )
    return {
        key: float(value)
        for key, value in row.items()
        if isinstance(value, (int, float))
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--select", choices=("best", "final"), default="best")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for name, path in args.run:
        grouped[name].append(selected_metrics(path, args.select))

    summary = {}
    for name, runs in grouped.items():
        keys = set.intersection(*(set(run) for run in runs))
        summary[name] = {
            "num_seeds": len(runs),
            "metrics": {
                key: {
                    "mean": statistics.mean(run[key] for run in runs),
                    "std": statistics.pstdev(run[key] for run in runs),
                }
                for key in sorted(keys)
                if key != "epoch"
            },
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
