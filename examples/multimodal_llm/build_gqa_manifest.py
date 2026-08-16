"""Build a GQA testdev_balanced manifest for the existing train/eval loop.

Downloads (or reuses) the lmms-lab/GQA testdev_balanced parquet files, unpacks
the embedded images next to the repo, and writes the standard manifest schema
``{"image": ..., "question": ..., "captions": [answer]}`` so the LLaVA-style
evaluator consumes GQA unchanged.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def read_table_columns(path: Path):
    table = pq.read_table(path)
    return table, table.column_names


def pick(columns, *candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(f"none of {candidates} in columns {columns}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="testdev_balanced")
    args = parser.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.output_dir)
    images_dir = out / "gqa_testdev_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    images_path = raw / f"{args.split}_images/testdev-00000-of-00001.parquet"
    instr_path = raw / f"{args.split}_instructions/testdev-00000-of-00001.parquet"
    if not images_path.exists() or not instr_path.exists():
        raise SystemExit(
            "parquet files missing; run download_gqa_files.py first with "
            f"--out-dir {raw}"
        )

    images_table, images_columns = read_table_columns(images_path)
    instr_table, instr_columns = read_table_columns(instr_path)
    print("image cols:", images_columns, flush=True)
    print("instr cols:", instr_columns, flush=True)

    image_id_col = pick(images_columns, "image_id", "img_id", "id")
    image_col = pick(images_columns, "image", "img")
    q_image_id_col = pick(instr_columns, "imageId", "image_id", "img_id", "image")
    qid_col = pick(instr_columns, "questionId", "question_id", "qid", "id")
    question_col = pick(instr_columns, "question")
    answer_col = pick(instr_columns, "answer", "answers", "short_answer")

    images_data = images_table.to_pydict()
    image_bytes = {}
    for i, image_id in enumerate(images_data[image_id_col]):
        raw_image = images_data[image_col][i]
        if isinstance(raw_image, dict) and "bytes" in raw_image:
            image_bytes[str(image_id)] = raw_image["bytes"]
        else:
            raise TypeError(f"unexpected image cell type: {type(raw_image)}")

    instr_data = instr_table.to_pydict()
    rows = []
    for i in range(len(instr_data[question_col])):
        image_id = instr_data[q_image_id_col][i]
        qid = instr_data[qid_col][i]
        answers = instr_data[answer_col][i]
        if isinstance(answers, list):
            answer = answers[0] if answers else ""
        else:
            answer = answers
        name = f"gqa_{qid}.jpg"
        if not (images_dir / name).exists():
            pil_image = Image.open(io.BytesIO(image_bytes[str(image_id)])).convert("RGB")
            pil_image.save(images_dir / name, quality=95)
        rows.append(
            {
                "image": f"gqa_testdev_images/{name}",
                "question": instr_data[question_col][i],
                "captions": [answer],
            }
        )

    manifest_path = out / "gqa_testdev.jsonl"
    with manifest_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"wrote {manifest_path} with {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
