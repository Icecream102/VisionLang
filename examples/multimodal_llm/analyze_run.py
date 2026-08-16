"""Analyze one completed v3 run and append the result to PROGRESS.md."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def load_best_metrics(run_dir: Path, key: str = "val_CIDEr") -> Optional[Dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if key in row]
    if not rows:
        return None
    return max(rows, key=lambda row: row.get(key, float("-inf")))


def run_config(run_dir: Path) -> Dict:
    path = run_dir / "config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def family_stats(outputs: Path, init: str, limit_train, key: str) -> str:
    values: List[float] = []
    for run_dir in outputs.glob(f"*{init}_seed*"):
        config = run_config(run_dir)
        if config.get("limit_train") != limit_train:
            continue
        best = load_best_metrics(run_dir, key)
        if best is not None:
            values.append(best.get(key, float("nan")))
    values = [v for v in values if v == v]
    if not values:
        return "_no peers yet_"
    if len(values) == 1:
        return f"{values[0]:.4f} (1 run)"
    return f"{statistics.mean(values):.4f} ± {statistics.pstdev(values):.4f} (n={len(values)})"


def cross_family_ranking(outputs: Path, limit_train, key: str) -> str:
    ranking = []
    for init in ("random", "mae", "clip"):
        values = []
        for run_dir in outputs.glob(f"{init}_seed*"):
            config = run_config(run_dir)
            if config.get("limit_train") != limit_train:
                continue
            best = load_best_metrics(run_dir, key)
            if best is not None:
                values.append(best.get(key, float("nan")))
        values = [v for v in values if v == v]
        if values:
            ranking.append((init, statistics.mean(values), len(values)))
    ranking.sort(key=lambda item: item[1], reverse=True)
    return " > ".join(f"{init} {mean:.4f} (n={n})" for init, mean, n in ranking) or "_pending_"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--progress-md", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    outputs = Path(args.outputs_dir)
    config = run_config(run_dir)
    init = config.get("init", "?")
    seed = config.get("seed", "?")
    limit_train = config.get("limit_train")
    scale = "100%" if limit_train is None else f"{limit_train} images"
    best = load_best_metrics(run_dir)
    best_acc = load_best_metrics(run_dir, "val_accuracy")

    lines = [f"\n## {run_dir.name} — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"]
    lines.append(f"- config: init={init}, seed={seed}, data={scale}")
    if best is not None:
        lines.append(
            f"- best val: CIDEr {best.get('val_CIDEr', float('nan')):.4f}, "
            f"BLEU-4 {best.get('val_Bleu_4', float('nan')):.4f}, "
            f"METEOR {best.get('val_METEOR', float('nan')):.4f}, "
            f"ROUGE-L {best.get('val_ROUGE_L', float('nan')):.4f}, "
            f"SPICE {best.get('val_SPICE', float('nan')):.4f}"
        )
        if limit_train is not None:
            lines.append(f"- family ({init} @ {scale}): {family_stats(outputs, init, limit_train, 'val_CIDEr')}")
        else:
            lines.append(f"- family ({init} @ 100%): {family_stats(outputs, init, None, 'val_CIDEr')}")
            lines.append(f"- cross-family ranking @ 100%: {cross_family_ranking(outputs, None, 'val_CIDEr')}")
    if best_acc is not None:
        lines.append(
            f"- recognition: accuracy {best_acc.get('val_accuracy', float('nan')):.4f}, "
            f"yes {best_acc.get('val_yes_accuracy', float('nan')):.4f}, "
            f"no {best_acc.get('val_no_accuracy', float('nan')):.4f}"
        )
    chair_path = run_dir / "chair.json"
    if chair_path.exists():
        chair = json.loads(chair_path.read_text())
        lines.append(f"- CHAIR_s {chair.get('chair_s', float('nan')):.4f}, CHAIR_i {chair.get('chair_i', float('nan')):.4f}")
    report = "\n".join(lines) + "\n"
    progress = Path(args.progress_md)
    progress.parent.mkdir(parents=True, exist_ok=True)
    with progress.open("a") as handle:
        handle.write(report)
    print(report)


if __name__ == "__main__":
    main()
