"""Download lmms-lab/GQA testdev_balanced parquet files (mirror-friendly)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repo", default="lmms-lab/GQA")
    parser.add_argument("--split", default="testdev_balanced")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "images": f"{args.split}_images/testdev-00000-of-00001.parquet",
        "instructions": f"{args.split}_instructions/testdev-00000-of-00001.parquet",
    }
    for key, rel in files.items():
        target = out / rel
        if target.exists() and target.stat().st_size > 0:
            print(f"skip {key}: {target} exists", flush=True)
        else:
            print(f"downloading {key} ...", flush=True)
            hf_hub_download(
                repo_id=args.repo,
                filename=rel,
                repo_type="dataset",
                local_dir=str(out),
            )
            print(f"downloaded {key}", flush=True)

    for key in files:
        path = out / files[key]
        schema = pq.ParquetFile(path).schema
        print(f"== {key} schema ==", flush=True)
        print(schema, flush=True)


if __name__ == "__main__":
    main()
