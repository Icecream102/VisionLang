"""Shared construction and reproducibility utilities for MAE experiments."""

import random
from typing import Any, Dict, Tuple

import numpy as np
import torch

from torchmultimodal.models.masked_auto_encoder.model import (
    MaskedAutoEncoder,
    image_mae,
)


MODEL_CONFIGS: Dict[str, Dict[str, int]] = {
    # Intended for a laptop/smoke test; base follows the paper's ViT-B configuration.
    "tiny": {
        "encoder_layers": 4,
        "encoder_hidden_dim": 192,
        "encoder_heads": 3,
        "encoder_dim_feedforward": 768,
        "decoder_layers": 2,
        "decoder_hidden_dim": 128,
        "decoder_heads": 4,
        "decoder_dim_feedforward": 512,
    },
    "base": {
        "encoder_layers": 12,
        "encoder_hidden_dim": 768,
        "encoder_heads": 12,
        "encoder_dim_feedforward": 3072,
        "decoder_layers": 8,
        "decoder_hidden_dim": 512,
        "decoder_heads": 16,
        "decoder_dim_feedforward": 2048,
    },
}


def build_mae(model_size: str, image_size: int, mask_ratio: float) -> MaskedAutoEncoder:
    if model_size not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model_size={model_size}. Choose from {sorted(MODEL_CONFIGS)}"
        )
    if not 0 < mask_ratio < 1:
        raise ValueError("mask_ratio must be strictly between 0 and 1.")
    return image_mae(
        image_size=image_size,
        masking_ratio=mask_ratio,
        **MODEL_CONFIGS[model_size],
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda"]])


def encoder_dim(model_size: str) -> int:
    return MODEL_CONFIGS[model_size]["encoder_hidden_dim"]
