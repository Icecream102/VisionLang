"""Final evaluation protocol for every v3/v4 checkpoint.

Re-scores all completed caption runs with the fixed decoding recipe
(repetition penalty + no-repeat-ngram + caption normalization) on:

* val split (default 500 images, matching the training-time selection subset);
* held-out test split (default 2,500 images, reported once);
* CHAIR hallucination on val (300 images);
* a text-prior floor for selected runs (same weights, image omitted), which
  isolates how much of the caption quality comes from language alone.

Outputs per run: ``metrics_final.json`` (val/test/chair/text_prior), plus a
global ``outputs/v4_final_summary.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoConfig, AutoTokenizer

from examples.multimodal_llm.data import LlavaCaptionDataset, build_eval_transform
from examples.multimodal_llm.evaluate import (
    compute_metrics,
    generate_captions,
)
from examples.multimodal_llm.hallucination import compute_chair, load_gt_categories
from examples.multimodal_llm.model import (
    IMAGE_TOKEN,
    LlavaCaptionModel,
    MLPProjector,
    build_llm,
    build_vision_tower,
)


def infer_init(run_dir: Path) -> str:
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            init = json.loads(config_path.read_text()).get("init")
            if init in ("random", "mae", "clip"):
                return init
        except Exception:
            pass
    name = run_dir.name
    if "clip" in name:
        return "clip"
    if "mae" in name:
        return "mae"
    return "random"


def read_lora_config(run_dir: Path):
    adapter_config = run_dir / "lora_adapter" / "adapter_config.json"
    if adapter_config.exists():
        data = json.loads(adapter_config.read_text())
        return int(data.get("r", 64)), float(data.get("lora_alpha", 128.0))
    if "lora_r16" in run_dir.name:
        return 16, 32.0
    return 64, 128.0


def load_model(
    run_dir: Path,
    model_name: str,
    text_only: bool = False,
) -> LlavaCaptionModel:
    init = infer_init(run_dir)
    lora_r, lora_alpha = read_lora_config(run_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.convert_tokens_to_ids(IMAGE_TOKEN) == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    llm_config = AutoConfig.from_pretrained(model_name)
    has_vision = (run_dir / "vision.pt").exists()
    vision = None
    projector = None
    if has_vision:
        if not text_only:
            vision = build_vision_tower(init)
            projector = MLPProjector(llm_dim=llm_config.hidden_size)
    llm = build_llm(
        model_name,
        tokenizer,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )
    model = LlavaCaptionModel(
        vision,
        projector,
        llm,
        image_token_id,
        text_only=text_only or not has_vision,
    ).cuda()
    if vision is not None and (run_dir / "vision.pt").exists():
        vision.load_state_dict(
            torch.load(run_dir / "vision.pt", weights_only=False)
        )
    if projector is not None and (run_dir / "projector.pt").exists():
        projector.load_state_dict(
            torch.load(run_dir / "projector.pt", weights_only=False)
        )
    base_llm = llm.get_base_model()
    if (run_dir / "llm_base.pt").exists():
        base_llm.load_state_dict(
            torch.load(run_dir / "llm_base.pt", weights_only=False)
        )
    else:
        print("  llm_base.pt not found; using pretrained base weights", flush=True)
    if (run_dir / "lora_adapter").exists():
        llm.load_adapter(str(run_dir / "lora_adapter"), "default")
    return model


def evaluate_manifest(
    model: LlavaCaptionModel,
    manifest: str,
    image_root: str,
    tokenizer,
    limit: int,
    text_only: bool = False,
    max_new_tokens: int = 48,
) -> Dict:
    dataset = LlavaCaptionDataset(
        manifest,
        image_root,
        tokenizer,
        build_eval_transform(224),
        eval_mode=True,
        limit=limit,
        text_only=text_only,
    )
    predictions, ground_truth = generate_captions(
        model,
        dataset,
        tokenizer,
        torch.device("cuda"),
        batch_size=8,
        num_workers=4,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
    )
    metrics = compute_metrics(predictions, ground_truth)
    return metrics, predictions, ground_truth, dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--instances-json", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--limit-val", type=int, default=500)
    parser.add_argument("--limit-test", type=int, default=2500)
    parser.add_argument("--limit-chair", type=int, default=300)
    parser.add_argument("--runs", nargs="+", default=None)
    parser.add_argument("--test-runs", nargs="+", default=None)
    parser.add_argument("--text-prior-runs", nargs="+", default=None)
    parser.add_argument("--skip-chair", action="store_true")
    args = parser.parse_args()

    outputs = Path(args.outputs_dir)
    if args.runs:
        run_dirs = [outputs / name for name in args.runs]
    else:
        run_dirs = sorted(
            path
            for path in outputs.glob("*/")
            if path.name != "stage1"
            and not path.name.startswith("qa_")
            and not path.name.startswith("smoke")
            and (path / "llm_base.pt").exists()
        )
    test_set = set(args.test_runs or [path.name for path in run_dirs])
    text_prior_set = set(args.text_prior_runs or [])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.convert_tokens_to_ids(IMAGE_TOKEN) == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    gt_categories = (
        load_gt_categories(args.instances_json)
        if not args.skip_chair
        else None
    )

    summary: Dict[str, Dict] = {}
    for run_dir in run_dirs:
        name = run_dir.name
        print(f"[eval_final] {name}: loading", flush=True)
        model = load_model(run_dir, args.model_name, text_only=False)
        result: Dict[str, Dict] = {}

        val_metrics, val_preds, val_gts, val_dataset = evaluate_manifest(
            model,
            args.val_manifest,
            args.image_root,
            tokenizer,
            limit=args.limit_val,
        )
        result["val"] = {k: float(v) for k, v in val_metrics.items()}
        print(f"  val: METEOR={val_metrics.get('METEOR'):.4f} "
              f"CIDEr={val_metrics.get('CIDEr'):.6f} "
              f"SPICE={val_metrics.get('SPICE'):.4f}", flush=True)

        if not args.skip_chair:
            chair_preds = {}
            for i in list(val_preds)[: args.limit_chair]:
                image_id = int(Path(val_dataset.records[i]["image"]).stem)
                chair_preds[image_id] = val_preds[i][0]
            chair = compute_chair(chair_preds, gt_categories)
            result["chair"] = {
                "chair_s": float(chair["chair_s"]),
                "chair_i": float(chair["chair_i"]),
                "num_images": min(args.limit_chair, len(val_preds)),
            }
            print(f"  chair_s={chair['chair_s']:.4f}", flush=True)

        if name in test_set:
            test_metrics, test_preds, test_gts, test_dataset = evaluate_manifest(
                model,
                args.test_manifest,
                args.image_root,
                tokenizer,
                limit=args.limit_test,
            )
            result["test"] = {k: float(v) for k, v in test_metrics.items()}
            print(f"  test: METEOR={test_metrics.get('METEOR'):.4f} "
                  f"CIDEr={test_metrics.get('CIDEr'):.6f} "
                  f"SPICE={test_metrics.get('SPICE'):.4f}", flush=True)
            samples = []
            for i in list(test_preds)[:10]:
                samples.append(
                    {
                        "image_id": i,
                        "reference": test_gts[i][0],
                        "generated": test_preds[i][0],
                    }
                )
            (run_dir / "samples_test.json").write_text(
                json.dumps(samples, indent=1, ensure_ascii=False)
            )

        if name in text_prior_set:
            torch.cuda.empty_cache()
            text_model = load_model(run_dir, args.model_name, text_only=True)
            prior_metrics, _, _, _ = evaluate_manifest(
                text_model,
                args.val_manifest,
                args.image_root,
                tokenizer,
                limit=min(500, args.limit_val),
                text_only=True,
            )
            result["text_prior_val"] = {
                k: float(v) for k, v in prior_metrics.items()
            }
            print(f"  text-prior val: METEOR={prior_metrics.get('METEOR'):.4f} "
                  f"SPICE={prior_metrics.get('SPICE'):.4f}", flush=True)
            del text_model
            torch.cuda.empty_cache()

        summary[name] = result
        (run_dir / "metrics_final.json").write_text(
            json.dumps(result, indent=1) + "\n"
        )
        del model
        torch.cuda.empty_cache()

    out_path = outputs.parent / "v4_final_summary.json"
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    else:
        existing = {}
    existing.update(summary)
    out_path.write_text(json.dumps(existing, indent=1) + "\n")
    print(f"[eval_final] wrote {out_path}", flush=True)
if __name__ == "__main__":
    main()
