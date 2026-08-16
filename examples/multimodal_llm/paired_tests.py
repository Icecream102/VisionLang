"""Paired significance tests across seeds for the final v3/v4 metrics.

Computes a paired sign test (three matched seeds) for the key comparisons:
CLIP vs MAE, CLIP vs random, MAE vs random, on the held-out test metrics.
Uses scipy's Wilcoxon when available, otherwise an exact two-sided sign test.
With n=3 seeds the test has little power; the p-values are reported as
descriptive evidence alongside mean differences, never as a strong claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def wilcoxon_or_sign_test(
    a: List[float], b: List[float]
) -> Tuple[Optional[float], float]:
    """Wilcoxon signed-rank when scipy exists, else an exact sign test."""
    try:
        from scipy import stats

        stat, p = stats.wilcoxon(a, b)
        return float(stat), float(p)
    except Exception:
        pass
    positives = sum(1 for x, y in zip(a, b) if x > y)
    negatives = sum(1 for x, y in zip(a, b) if x < y)
    n = positives + negatives
    if n == 0:
        return None, 1.0
    k = min(positives, negatives)
    p = 0.0
    for i in range(k + 1):
        p += math.comb(n, i) * 0.5**n
    return None, float(2 * p)


GROUPS = {
    "clip": ["clip_seed42", "clip_seed43", "clip_seed44"],
    "mae": ["mae_seed42", "mae_seed43", "mae_seed44"],
    "random": ["random_seed42", "random_seed43", "random_seed44"],
}

METRICS = ("Bleu_1", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    results: Dict[str, Dict] = {}
    for metric in METRICS + ("chair_s",):
        results[metric] = {}
        for (a_name, b_name), (a_runs, b_runs) in [
            (("CLIP", "MAE"), (GROUPS["clip"], GROUPS["mae"])),
            (("CLIP", "Random"), (GROUPS["clip"], GROUPS["random"])),
            (("MAE", "Random"), (GROUPS["mae"], GROUPS["random"])),
        ]:
            key = f"{a_name}_vs_{b_name}"
            a_values: List[float] = []
            b_values: List[float] = []
            for run in a_runs:
                value = summary[run]["test"].get(metric) or summary[run].get(
                    "chair", {}
                ).get(metric)
                if value is not None:
                    a_values.append(value)
            for run in b_runs:
                value = summary[run]["test"].get(metric) or summary[run].get(
                    "chair", {}
                ).get(metric)
                if value is not None:
                    b_values.append(value)
            if len(a_values) == len(b_values) and len(a_values) >= 3:
                stat, p = wilcoxon_or_sign_test(a_values, b_values)
                results[metric][key] = {
                    "mean_a": float(sum(a_values) / len(a_values)),
                    "mean_b": float(sum(b_values) / len(b_values)),
                    "diff": float(sum(a_values) / len(a_values) - sum(b_values) / len(b_values)),
                    "p": float(p),
                    "n": len(a_values),
                }
    Path(args.output).write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
