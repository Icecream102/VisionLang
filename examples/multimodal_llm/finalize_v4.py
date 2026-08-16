"""Render the final v4 results markdown from eval_final outputs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional


REPORT_KEYS = ("Bleu_1", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE")
LABELS = {
    "Bleu_1": "BLEU-1",
    "Bleu_4": "BLEU-4",
    "METEOR": "METEOR",
    "ROUGE_L": "ROUGE-L",
    "CIDEr": "CIDEr",
    "SPICE": "SPICE",
}


def mean_std(rows: List[Dict], key: str, digits: int = 4) -> str:
    values = [row[key] for row in rows if key in row]
    if not values:
        return "_pending_"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{statistics.mean(values):.{digits}f} ± {statistics.pstdev(values):.{digits}f}"


def render(
    summary: Dict,
    grpo: Optional[List[Dict]],
    paired: Optional[Dict],
) -> str:
    lines = [
        "# v3/v4 Multimodal LLM Experiment Results (final)",
        "",
        "Protocol: LLaVA-style VLM (ViT-B/16 vision + MLP projector + Qwen2-0.5B with "
        "LoRA). Trained on COCO2017 `train2017`; **model selection on a 500-image val "
        "subset**; **held-out test (2,500 images) reported once below**. COCO2017 fixed "
        "split (not Karpathy). Decoding: greedy + repetition penalty 1.3 + "
        "no-repeat-ngram 3 + caption normalization (the v3 CIDEr=0 artefact came from "
        "repetitive multi-sentence decoding, not from training).",
        "",
    ]

    def table_rows(inits, split: str):
        for init, label in inits:
            rows = [
                summary.get(f"{init}_seed{seed}", {}).get(split, {})
                for seed in (42, 43, 44)
            ]
            rows = [r for r in rows if r]
            cells = " | ".join(mean_std(rows, key) for key in REPORT_KEYS)
            lines.append(f"| {label} | {cells} |")

    lines.append("## A. Visual initialization transfer — held-out TEST (2,500 images)")
    lines.append("")
    lines.append("| Initialization | " + " | ".join(LABELS[k] for k in REPORT_KEYS) + " |")
    lines.append("| --- | " + " | ".join(["---:"] * len(REPORT_KEYS)) + " |")
    table_rows([("random", "Random"), ("mae", "MAE"), ("clip", "CLIP")], "test")
    lines.append("")

    lines.append("## A'. Same comparison on the 500-image val subset (continuity with v3)")
    lines.append("")
    lines.append("| Initialization | " + " | ".join(LABELS[k] for k in REPORT_KEYS) + " |")
    lines.append("| --- | " + " | ".join(["---:"] * len(REPORT_KEYS)) + " |")
    table_rows([("random", "Random"), ("mae", "MAE"), ("clip", "CLIP")], "val")
    lines.append("")

    text_prior = []
    for init in ("random", "mae", "clip"):
        rows = [
            summary.get(f"{init}_seed{seed}", {}).get("text_prior_val", {})
            for seed in (42, 43, 44)
        ]
        rows = [r for r in rows if r]
        if rows:
            cells = " | ".join(
                mean_std(rows, key) for key in ("Bleu_4", "METEOR", "ROUGE_L", "SPICE")
            )
            text_prior.append(f"| {init} | {cells} |")
    if text_prior:
        lines.append(
            "## Text-prior floor (same trained weights, image omitted, val 500)"
        )
        lines.append("")
        lines.append("| Init | BLEU-4 | METEOR | ROUGE-L | SPICE |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        lines.extend(text_prior)
        lines.append("")
        lines.append(
            "Interpretation: removing the image collapses caption quality (e.g. "
            "CLIP METEOR drops from ~0.19 to the text-prior level), i.e. the "
            "vision pathway is doing the work, not the language prior."
        )
        lines.append("")

    lines.append("## E. Data-scaling curve (CLIP init) — held-out TEST")
    lines.append("")
    lines.append("| Data fraction | BLEU-4 | METEOR | ROUGE-L | SPICE |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for tag, fraction in (("e10", "10%"), ("e50", "50%")):
        rows = [
            summary.get(f"{tag}_clip_seed{seed}", {}).get("test", {})
            for seed in (42, 43)
        ]
        rows = [r for r in rows if r]
        if not rows:
            continue
        cells = " | ".join(
            mean_std(rows, key) for key in ("Bleu_4", "METEOR", "ROUGE_L", "SPICE")
        )
        lines.append(f"| {fraction} | {cells} |")
    rows = [
        summary.get(f"clip_seed{seed}", {}).get("test", {})
        for seed in (42, 43)
    ]
    rows = [r for r in rows if r]
    if rows:
        cells = " | ".join(
            mean_std(rows, key) for key in ("Bleu_4", "METEOR", "ROUGE_L", "SPICE")
        )
        lines.append(f"| 100% | {cells} |")
    lines.append("")

    lines.append("## A''. LR sensitivity (5,000 images, seed 42, val 500)")
    lines.append("")
    lines.append("| Init / LR | METEOR | ROUGE-L | SPICE |")
    lines.append("| --- | ---: | ---: | ---: |")
    for init in ("random", "mae", "clip"):
        for lr_tag, lr_label in (("lr1", "2e-4"), ("lr05", "1e-4")):
            row = summary.get(f"{lr_tag}_{init}_seed42", {}).get("val", {})
            if not row:
                continue
            lines.append(
                f"| {init} / {lr_label} | {row['METEOR']:.4f} | "
                f"{row['ROUGE_L']:.4f} | {row['SPICE']:.4f} |"
            )
    lines.append("")

    lora16 = summary.get("lora_r16_clip_seed42", {}).get("val", {})
    e10 = summary.get("e10_clip_seed42", {}).get("val", {})
    if lora16 and e10:
        lines.append(
            f"## LoRA rank (10% data, val 500)\n\n- r=16 METEOR {lora16['METEOR']:.4f} "
            f"vs r=64 METEOR {e10['METEOR']:.4f}; SPICE {lora16['SPICE']:.4f} vs "
            f"{e10['SPICE']:.4f}."
        )
        lines.append("")

    chair_rows = []
    for init, label in (("random", "Random"), ("mae", "MAE"), ("clip", "CLIP")):
        rows = [
            summary.get(f"{init}_seed{seed}", {}).get("chair", {})
            for seed in (42, 43, 44)
        ]
        rows = [r for r in rows if r]
        if rows:
            chair_rows.append(
                f"| {label} | {mean_std(rows, 'chair_s', 4)} | "
                f"{mean_std(rows, 'chair_i', 4)} |"
            )
    if chair_rows:
        lines.append("## Hallucination (CHAIR, new decoding, val 300)")
        lines.append("")
        lines.append("| Initialization | CHAIR_s | CHAIR_i |")
        lines.append("| --- | ---: | ---: |")
        lines.extend(chair_rows)
        lines.append("")

    textonly = {}
    for tag, label in (
        ("textonly_e10_clip_seed42", "text-only 10% s42"),
        ("textonly_e10_clip_seed43", "text-only 10% s43"),
        ("textonly_full_clip_seed42", "text-only 100% s42"),
    ):
        row = summary.get(tag, {})
        if row.get("test"):
            textonly[label] = (row["test"], "test")
        elif row.get("val"):
            textonly[label] = (row["val"], "val")
    if textonly:
        lines.append("## Text-only trained baseline (no vision pathway)")
        lines.append("")
        lines.append("| Run | Split | BLEU-4 | METEOR | ROUGE-L | SPICE |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for label, (row, split) in textonly.items():
            lines.append(
                f"| {label} | {split} | {row.get('Bleu_4', 0):.4f} | "
                f"{row.get('METEOR', 0):.4f} | {row.get('ROUGE_L', 0):.4f} | "
                f"{row.get('SPICE', 0):.4f} |"
            )
        lines.append("")
        lines.append(
            "Interpretation: training the same Qwen2-0.5B + LoRA recipe with the "
            "image removed performs far below the vision-conditioned model, "
            "quantifying how much the VLM gains from the visual modality."
        )
        lines.append("")

    if grpo:
        lines.append("## GRPO alignment (yes/no recognition, single GPU)")
        lines.append("")
        lines.append(
            f"| Stage | Accuracy | Yes acc | No acc | Steps | Group | Beta | KL |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for label, g in grpo:
            before = g.get("before", {})
            after = g.get("after", {})
            lines.append(
                f"| SFT before GRPO ({label}) | "
                f"{before.get('accuracy', float('nan')):.4f} | "
                f"{before.get('yes_accuracy', float('nan')):.4f} | "
                f"{before.get('no_accuracy', float('nan')):.4f} | — | — | — | — |"
            )
            lines.append(
                f"| After GRPO ({label}) | "
                f"{after.get('accuracy', float('nan')):.4f} | "
                f"{after.get('yes_accuracy', float('nan')):.4f} | "
                f"{after.get('no_accuracy', float('nan')):.4f} | {g.get('steps')} | "
                f"{g.get('group_size')} | {g.get('beta')} | "
                f"{g.get('final_kl', float('nan')):.4f} |"
            )
        lines.append("")
        lines.append(
            "Implementation: group sampling (temperature 1.2), rule-based reward "
            "(correct yes/no), within-group advantage normalization, per-token KL "
            "penalty against the frozen SFT reference (DeepSeek-R1 style GRPO), "
            "LoRA + projector trainable. The balanced variant samples equal "
            "yes/no prompts to avoid class-imbalance drift in the rule reward."
        )
        lines.append("")

    if paired:
        lines.append("## Paired significance (Wilcoxon, n=3 seeds, held-out test)")
        lines.append("")
        lines.append("| Metric | Comparison | Diff (A−B) | p |")
        lines.append("| --- | --- | ---: | ---: |")
        for metric, comparisons in paired.items():
            for comp, data in comparisons.items():
                lines.append(
                    f"| {LABELS.get(metric, metric)} | {comp} | "
                    f"{data['diff']:+.4f} | {data['p']:.4f} |"
                )
        lines.append("")
        lines.append(
            "Caveat: with only three matched seeds the test has low power; treat "
            "p-values as descriptive, not as proof of significance."
        )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--grpo-json", nargs="+", default=None)
    parser.add_argument("--paired-json", default=None)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text())
    grpo = (
        [
            (Path(path).parent.name, json.loads(Path(path).read_text()))
            for path in args.grpo_json
        ]
        if args.grpo_json
        else None
    )
    paired = (
        json.loads(Path(args.paired_json).read_text()) if args.paired_json else None
    )
    markdown = render(summary, grpo, paired)
    Path(args.output_md).write_text(markdown + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
