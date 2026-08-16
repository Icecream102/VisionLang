"""Merge per-run pope.json files into one summary (0.5B + 3B checkpoints)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pope-jsons", nargs="+", required=True)
    parser.add_argument("--keys", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.pope_jsons) != len(args.keys):
        raise SystemExit("--pope-jsons and --keys must have equal length")

    summary = {}
    for path, key in zip(args.pope_jsons, args.keys):
        summary[key] = json.loads(Path(path).read_text())
    Path(args.output).write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n"
    )
    print(f"wrote {args.output} with keys {list(summary)}", flush=True)


if __name__ == "__main__":
    main()
