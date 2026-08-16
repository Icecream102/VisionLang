"""Compute CHAIR hallucination metrics for every completed v3 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from examples.multimodal_llm.data import (
    LlavaCaptionDataset,
    build_eval_transform,
)
from examples.multimodal_llm.evaluate import generate_captions
from examples.multimodal_llm.hallucination import compute_chair, load_gt_categories
from examples.multimodal_llm.model import (
    IMAGE_TOKEN,
    LlavaCaptionModel,
    MLPProjector,
    build_llm,
    build_vision_tower,
)


def infer_init(run_dir: Path) -> str:
    name = run_dir.name
    if "clip" in name:
        return "clip"
    if "mae" in name:
        return "mae"
    return "random"


def infer_lora_config(run_dir: Path):
    """Recover the LoRA rank used by a run so the adapter loads cleanly."""
    if "lora_r16" in run_dir.name:
        return 16, 32.0
    return 64, 128.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--instances-json", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--limit-images", type=int, default=300)
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()

    gt = load_gt_categories(args.instances_json)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.convert_tokens_to_ids(IMAGE_TOKEN) == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    llm_config = AutoConfig.from_pretrained(args.model_name)

    run_dirs = sorted(
        path
        for path in Path(args.outputs_dir).glob("*/")
        if path.name != "stage1"
        and not path.name.startswith("qa_")
        and (path / "vision.pt").exists()
        and (path / "projector.pt").exists()
        and (path / "llm_base.pt").exists()
    )
    results = {}
    for run_dir in run_dirs:
        chair_path = run_dir / "chair.json"
        if chair_path.exists():
            results[run_dir.name] = json.loads(chair_path.read_text())
            continue
        try:
            init = infer_init(run_dir)
            lora_r, lora_alpha = infer_lora_config(run_dir)
            vision = build_vision_tower(init)
            projector = MLPProjector(llm_dim=llm_config.hidden_size)
            llm = build_llm(
                args.model_name,
                tokenizer,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
            )
            model = LlavaCaptionModel(vision, projector, llm, image_token_id).cuda()
            vision.load_state_dict(
                torch.load(run_dir / "vision.pt", weights_only=False)
            )
            projector.load_state_dict(
                torch.load(run_dir / "projector.pt", weights_only=False)
            )
            base_llm = llm.get_base_model()
            base_llm.load_state_dict(
                torch.load(run_dir / "llm_base.pt", weights_only=False)
            )
            if (run_dir / "lora_adapter").exists():
                llm.load_adapter(str(run_dir / "lora_adapter"), "default")
            dataset = LlavaCaptionDataset(
                args.val_manifest,
                args.image_root,
                tokenizer,
                build_eval_transform(224),
                eval_mode=True,
                limit=args.limit_images,
            )
            predictions, _ = generate_captions(
                model,
                dataset,
                tokenizer,
                torch.device("cuda"),
                batch_size=8,
                num_workers=4,
                num_beams=1,
            )
            image_ids = [
                int(Path(dataset.records[i]["image"]).stem)
                for i in sorted(predictions)
            ]
            generated = {
                image_ids[i]: captions[0]
                for i, captions in enumerate(predictions.values())
            }
            chair = compute_chair(generated, gt)
            chair.update(
                {
                    "model_dir": run_dir.name,
                    "init": init,
                    "num_images": len(generated),
                }
            )
            chair_path.write_text(json.dumps(chair, indent=2) + "\n")
            results[run_dir.name] = chair
            print(
                f"{run_dir.name}: CHAIR_s={chair['chair_s']:.4f} "
                f"CHAIR_i={chair['chair_i']:.4f}"
            )
        except Exception as exc:  # keep going when a single run fails
            print(f"{run_dir.name}: CHAIR eval failed: {exc}")
        finally:
            try:
                del model
            except UnboundLocalError:
                pass
            torch.cuda.empty_cache()
    summary_path = Path(args.outputs_dir) / "chair_summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
