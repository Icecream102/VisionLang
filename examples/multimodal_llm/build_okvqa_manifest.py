"""Build OK-VQA train/val manifests from official annotations + lmms-lab parquet.

Training split: OK-VQA train2014 questions/answers are joined and written as
``{"image": "train2017/<id>.jpg", "question": ..., "captions": [answer]}``.
The answer for training is the most common ground-truth answer (ties broken
by the first answer), matching standard VQA practice.  The image root must
contain COCO ``train2017`` (OK-VQA train questions all use train2014 image
IDs, which are a subset of COCO2017 train).

Validation split: ``lmms-lab/OK-VQA`` parquet (images embedded) is unpacked
into ``images/`` and written as the same manifest schema, so the existing
train/eval loop consumes it unchanged.  Evaluation of OK-VQA val uses the
standard accuracy = mean over questions of (num matching answers / 3).
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def load_train_records(questions_zip: str, annotations_zip: str):
    with zipfile.ZipFile(questions_zip) as handle:
        questions_name = next(
            name for name in handle.namelist() if name.endswith("questions.json")
        )
        questions_data = json.loads(handle.read(questions_name))
    with zipfile.ZipFile(annotations_zip) as handle:
        annotations_name = next(
            name for name in handle.namelist() if name.endswith("annotations.json")
        )
        annotations_data = json.loads(handle.read(annotations_name))
    question_by_id = {item["question_id"]: item for item in questions_data["questions"]}
    answer_by_id = {
        item["question_id"]: item for item in annotations_data["annotations"]
    }
    rows = []
    for qid, question in question_by_id.items():
        answers = answer_by_id[qid]["answers"]
        # Standard VQA training target: most common answer, tie -> first.
        counter = Counter(answer["answer"].strip().lower() for answer in answers)
        target = counter.most_common(1)[0][0]
        image_id = question["image_id"]
        rows.append(
            {
                "image": f"train2017/{image_id:012d}.jpg",
                "question": question["question"],
                "captions": [target],
            }
        )
    return rows


def build_val_split(parquet_paths, output_dir: Path):
    images_dir = output_dir / "okvqa_val_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for parquet_path in parquet_paths:
        table = pq.read_table(
            parquet_path,
            columns=["question_id", "question", "answers", "image"],
        )
        data = table.to_pydict()
        for i in range(len(data["question"])):
            image = Image.open(io.BytesIO(data["image"][i]["bytes"])).convert("RGB")
            name = f"val_{data['question_id'][i]}.jpg"
            image.save(images_dir / name, quality=95)
            rows.append(
                {
                    "image": f"okvqa_val_images/{name}",
                    "question": data["question"][i],
                    "captions": list(data["answers"][i]),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-questions-zip", required=True)
    parser.add_argument("--train-annotations-zip", required=True)
    parser.add_argument("--val-parquet", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_train_records(
        args.train_questions_zip, args.train_annotations_zip
    )
    with (output_dir / "okvqa_train.jsonl").open("w") as handle:
        for row in train_rows:
            handle.write(json.dumps(row) + "\n")
    print(f"train: {len(train_rows)} records")

    val_rows = build_val_split(args.val_parquet, output_dir)
    with (output_dir / "okvqa_val.jsonl").open("w") as handle:
        for row in val_rows:
            handle.write(json.dumps(row) + "\n")
    print(f"val: {len(val_rows)} records")


if __name__ == "__main__":
    main()
