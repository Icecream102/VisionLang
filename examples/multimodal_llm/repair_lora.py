"""Regenerate a run's lora_adapter (and components) from its best.pt.

Used when a run's final `save_pretrained(lora_adapter)` failed because the
disk filled up, even though training completed and best.pt was written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from examples.multimodal_llm.model import (
    IMAGE_TOKEN,
    LlavaCaptionModel,
    MLPProjector,
    build_llm,
    build_vision_tower,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)

    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.convert_tokens_to_ids(IMAGE_TOKEN) == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    llm_config = AutoConfig.from_pretrained(model_name)
    vision = build_vision_tower(config.get("init", "clip"))
    projector = MLPProjector(llm_dim=llm_config.hidden_size)
    llm = build_llm(
        model_name,
        tokenizer,
        lora_r=int(config.get("lora_r", 64)),
        lora_alpha=float(config.get("lora_alpha", 128.0)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
    )
    model = LlavaCaptionModel(vision, projector, llm, image_token_id)
    if checkpoint.get("vision") is not None:
        vision.load_state_dict(checkpoint["vision"])
    if checkpoint.get("projector") is not None:
        projector.load_state_dict(checkpoint["projector"])
    if checkpoint.get("adapter"):
        llm.load_state_dict(checkpoint["adapter"], strict=False)
    base_llm = llm.get_base_model() if hasattr(llm, "get_base_model") else llm
    if checkpoint.get("llm_base") is not None:
        base_llm.load_state_dict(checkpoint["llm_base"])

    torch.save(vision.state_dict(), run_dir / "vision.pt")
    torch.save(projector.state_dict(), run_dir / "projector.pt")
    if hasattr(llm, "save_pretrained"):
        llm.save_pretrained(run_dir / "lora_adapter")
    print(f"repaired {run_dir}", flush=True)


if __name__ == "__main__":
    main()
