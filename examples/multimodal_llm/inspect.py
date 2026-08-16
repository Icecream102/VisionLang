"""Print real generated captions from a completed v3 run for sanity checks."""

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
from examples.multimodal_llm.model import (
    IMAGE_TOKEN,
    LlavaCaptionModel,
    MLPProjector,
    build_llm,
    build_vision_tower,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--init", default="clip")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Skip loading the LoRA adapter (tests the raw backbone).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw token ids / full decoded strings for debugging.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.convert_tokens_to_ids(IMAGE_TOKEN) == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    llm_config = AutoConfig.from_pretrained(args.model_name)

    vision = build_vision_tower(args.init)
    projector = MLPProjector(llm_dim=llm_config.hidden_size)
    llm = build_llm(args.model_name, tokenizer)
    model = LlavaCaptionModel(vision, projector, llm, image_token_id).cuda()
    vision.load_state_dict(torch.load(output_dir / "vision.pt", weights_only=False))
    projector.load_state_dict(
        torch.load(output_dir / "projector.pt", weights_only=False)
    )
    base_llm = llm.get_base_model()
    base_llm.load_state_dict(torch.load(output_dir / "llm_base.pt", weights_only=False))
    if (output_dir / "lora_adapter").exists() and not args.no_adapter:
        llm.load_adapter(str(output_dir / "lora_adapter"), "default")

    dataset = LlavaCaptionDataset(
        args.val_manifest,
        args.image_root,
        tokenizer,
        build_eval_transform(224),
        seed=args.seed,
        eval_mode=True,
        limit=args.num_samples,
    )
    if args.raw:
        model.eval()
        generator_kwargs = {
            "max_new_tokens": 32,
            "num_beams": 1,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        with torch.inference_mode():
            for i in range(min(args.num_samples, len(dataset))):
                record = dataset[i]
                image = record["image"].unsqueeze(0).cuda()
                input_ids = record["input_ids"].unsqueeze(0).cuda()
                attention_mask = record["attention_mask"].unsqueeze(0).cuda()
                output_ids = model.generate(
                    image, input_ids, attention_mask, **generator_kwargs
                )
                print(f"--- sample {i} raw ---")
                print("output shape:", tuple(output_ids.shape))
                print("prompt length:", input_ids.size(1))
                print(
                    "image positions:",
                    (input_ids[0] == image_token_id).nonzero().flatten().tolist(),
                )
                print("prompt ids:", input_ids[0].tolist()[:12])
                print("output first ids:", output_ids[0].tolist()[:12])
                print("output last ids:", output_ids[0].tolist()[-12:])
                print(
                    "full decode:",
                    repr(tokenizer.decode(output_ids[0], skip_special_tokens=False)),
                )
                print(
                    "after-prompt decode:",
                    repr(
                        tokenizer.decode(
                            output_ids[0, input_ids.size(1) + model.image_token_count - 1 :],
                            skip_special_tokens=True,
                        )
                    ),
                )
        return

    predictions, ground_truth = generate_captions(
        model,
        dataset,
        tokenizer,
        torch.device("cuda"),
        batch_size=args.num_samples,
        num_workers=0,
        num_beams=args.beams,
    )
    for index in sorted(predictions):
        print(f"--- sample {index} ---")
        print(f"prompt : {dataset.records[index]['captions']}")
        print(f"gold   : {ground_truth[index][0]!r}")
        print(f"pred   : {predictions[index][0]!r}")
        if args.raw:
            record = dataset[index]
            print(f"input_ids len={len(record['input_ids'])}")
            image_positions = (record["input_ids"] == image_token_id).nonzero().flatten()
            print(f"image token positions: {image_positions.tolist()}")


if __name__ == "__main__":
    main()
