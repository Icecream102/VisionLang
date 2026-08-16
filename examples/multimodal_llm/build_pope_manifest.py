"""Convert the lmms-lab/POPE parquet splits into image files + JSONL manifests.

The POPE dataset (Li et al., 2023) asks "Is there a <object> in the image?"
over COCO images with three question sets: random, popular, adversarial.
Each set has 3,000 questions (1,500 yes / 1,500 no).  This script writes the
embedded images under ``output_dir/images`` and a JSONL manifest per split
reusing the caption-manifest schema (``image`` / ``question`` / ``captions``),
so the existing training/eval loop can consume it unchanged.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", required=True, help="HF snapshot Full/ dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir = Path(args.parquet_dir)

    splits = ["adversarial", "popular", "random"]
    for split in splits:
        parquet_path = parquet_dir / f"{split}-00000-of-00001.parquet"
        table = pq.read_table(
            parquet_path,
            columns=["question_id", "question", "answer", "category", "image"],
        )
        data = table.to_pydict()
        manifest = out / f"pope_{split}.jsonl"
        with manifest.open("w") as handle:
            for i in range(len(data["question"])):
                image_bytes = data["image"][i]["bytes"]
                question = data["question"][i]
                answer = data["answer"][i].strip().lower()
                if answer not in ("yes", "no"):
                    raise ValueError(f"Unexpected POPE answer: {answer!r}")
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image_name = f"{split}_{data['question_id'][i]}.jpg"
                image.save(images_dir / image_name, quality=95)
                handle.write(
                    json.dumps(
                        {
                            "image": f"images/{image_name}",
                            "question": question,
                            "captions": [answer],
                        }
                    )
                    + "\n"
                )
        print(f"{split}: {len(data['question'])} records -> {manifest}")


if __name__ == "__main__":
    main()
