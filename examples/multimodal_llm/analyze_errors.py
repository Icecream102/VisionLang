"""Error analysis over full held-out test predictions for one checkpoint.

Loads a completed run, regenerates captions for the full test manifest,
and cross-checks the generated text against COCO instance annotations:

* object-level hallucination: objects mentioned in the caption but absent
  from the image's instance annotations;
* object omission: frequent ground-truth objects that the caption never
  mentions;
* caption quality proxy: unigram precision against all reference captions,
  used only to surface low-quality examples for manual inspection.

Outputs ``predictions_test.json`` (all predictions), ``error_analysis.json``
(per-sample metrics) and a human-readable ``error_report.md`` with the worst
examples and aggregate statistics.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from examples.multimodal_llm.data import LlavaCaptionDataset, build_eval_transform
from examples.multimodal_llm.eval_final import load_model
from examples.multimodal_llm.evaluate import generate_captions
from examples.multimodal_llm.hallucination import (
    SYNONYMS,
    detect_mentioned_categories,
    load_gt_categories,
)
from examples.multimodal_llm.model import IMAGE_TOKEN


def unigram_precision(prediction: str, references: list[str]) -> float:
    """Cheap quality proxy: fraction of prediction unigrams covered by refs."""
    pred_tokens = [t for t in re.findall(r"[a-z0-9']+", prediction.lower()) if t]
    if not pred_tokens:
        return 0.0
    ref_tokens = set()
    for ref in references:
        ref_tokens.update(re.findall(r"[a-z0-9']+", ref.lower()))
    if not ref_tokens:
        return 0.0
    return sum(t in ref_tokens for t in pred_tokens) / len(pred_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--instances-json", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/error_analysis")
    parser.add_argument("--num-worst", type=int, default=20)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_gt_categories(args.instances_json)
    image_id_to_file = {}
    data = json.loads(Path(args.instances_json).read_text())
    for item in data["images"]:
        image_id_to_file[item["id"]] = item["file_name"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.convert_tokens_to_ids(IMAGE_TOKEN) == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    llm_config = AutoConfig.from_pretrained(args.model_name)

    model = load_model(run_dir, args.model_name)
    dataset = LlavaCaptionDataset(
        args.test_manifest,
        args.image_root,
        tokenizer,
        build_eval_transform(224),
        eval_mode=True,
        limit=args.limit,
    )
    predictions, ground_truth = generate_captions(
        model,
        dataset,
        tokenizer,
        torch.device("cuda"),
        batch_size=16,
        num_workers=4,
        max_new_tokens=64,
        num_beams=1,
    )

    rows = []
    category_count = Counter()
    for index, captions in predictions.items():
        record = dataset.records[index]
        image_file = record["image"]
        image_id = int(Path(image_file).stem)
        prediction = captions[0]
        references = ground_truth[index]
        score = unigram_precision(prediction, references)
        mentioned = detect_mentioned_categories(prediction)
        gt_cats = gt.get(image_id, set())
        hallucinated = sorted(mentioned - gt_cats)
        omitted = sorted(gt_cats - mentioned)
        rows.append(
            {
                "image_id": image_id,
                "image_file": image_file,
                "prediction": prediction,
                "references": references,
                "quality": round(score, 4),
                "mentioned": sorted(mentioned),
                "gt_categories": sorted(gt_cats),
                "hallucinated_objects": hallucinated,
                "omitted_objects": omitted,
                "pred_len": len(prediction.split()),
            }
        )
        for cat in hallucinated:
            category_count[cat] += 1

    # Aggregate stats
    n = len(rows)
    with_mention = sum(1 for r in rows if r["mentioned"])
    halluc_any = sum(1 for r in rows if r["hallucinated_objects"])
    halluc_objects = sum(len(r["hallucinated_objects"]) for r in rows)
    mentioned_total = sum(len(r["mentioned"]) for r in rows)
    avg_quality = sum(r["quality"] for r in rows) / max(1, n)
    stats = {
        "num_images": n,
        "avg_quality_proxy": round(avg_quality, 4),
        "images_mentioning_object": with_mention,
        "images_with_hallucinated_object": halluc_any,
        "hallucinated_objects_total": halluc_objects,
        "hallucination_rate_mentioned": (
            round(halluc_objects / mentioned_total, 4) if mentioned_total else 0.0
        ),
        "hallucinated_object_categories": dict(
            category_count.most_common(15)
        ),
    }

    rows.sort(key=lambda r: r["quality"])
    worst = rows[: args.num_worst]

    # Write artifacts
    predictions_path = out_dir / "predictions_test.json"
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "image_id": r["image_id"],
                    "image_file": r["image_file"],
                    "prediction": r["prediction"],
                    "references": r["references"],
                }
                for r in rows
            ],
            indent=1,
            ensure_ascii=False,
        )
    )
    analysis_path = out_dir / "error_analysis.json"
    analysis_path.write_text(
        json.dumps({"stats": stats, "rows": rows}, indent=1, ensure_ascii=False)
    )

    # Markdown report
    lines = [
        "# 错误样本分析（held-out test）",
        "",
        f"- 模型：`{run_dir.name}`",
        f"- 评测集：`{args.test_manifest}`（{n} 张图）",
        f"- 质量代理（unigram precision vs 参考句）：均值 {stats['avg_quality_proxy']}",
        f"- 提到对象的图：{with_mention}/{n}",
        f"- 含幻觉对象的图：{halluc_any}/{n}",
        f"- 幻觉对象占全部提到对象比例：{stats['hallucination_rate_mentioned']}",
        "",
        "## 高频幻觉对象类别",
        "",
        "| 类别 | 出现次数 |",
        "| --- | ---: |",
    ]
    for cat, count in category_count.most_common(15):
        lines.append(f"| {cat} | {count} |")
    lines += ["", f"## 最差 {len(worst)} 个样本（按质量代理升序）", ""]
    for i, row in enumerate(worst, 1):
        lines += [
            f"### {i}. image_id={row['image_id']}（质量 {row['quality']}）",
            "",
            f"- 图像：`{row['image_file']}`",
            f"- 预测：{row['prediction']}",
            f"- 参考：{row['references'][0]}",
            f"- 图内真实对象：{', '.join(row['gt_categories']) or '无'}",
            f"- 提到但图内没有（幻觉）：{', '.join(row['hallucinated_objects']) or '无'}",
            f"- 图内有但未提到（遗漏）：{', '.join(row['omitted_objects']) or '无'}",
            "",
        ]
    report_path = out_dir / "error_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    print(f"Wrote {predictions_path}, {analysis_path}, {report_path}")


if __name__ == "__main__":
    main()
