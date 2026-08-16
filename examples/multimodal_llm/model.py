"""Vision tower, projector and Qwen2 backbone for LLaVA-style captioning.

The three vision initializations share the same ViT-B/16 architecture
(768 hidden, 197 tokens at 224x224), so any difference between conditions
comes from the pretrained weights rather than model capacity.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor, nn
from transformers import (
    AutoConfig,
    CLIPVisionModel,
    PreTrainedTokenizerBase,
    Qwen2ForCausalLM,
    ViTMAEConfig,
    ViTMAEModel,
)
from transformers.utils import logging as hf_logging

from peft import LoraConfig, get_peft_model


hf_logging.set_verbosity_error()
logger = logging.getLogger(__name__)


IMAGE_TOKEN = "<image>"


def build_vision_tower(
    init: str,
    image_size: int = 224,
    lora: bool = False,
    lora_r: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.05,
) -> nn.Module:
    """Return a frozen (or LoRA-adaptable) ViT-B/16 vision tower.

    ``init`` selects only the source of the weights: ``random`` builds the
    same ViT-B/16 architecture with default initialization; ``mae`` and
    ``clip`` load pretrained weights.
    """
    if init in ("random", "mae"):
        if init == "random":
            config = ViTMAEConfig.from_pretrained("facebook/vit-mae-base")
            vision: nn.Module = ViTMAEModel(config)
        else:
            vision = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        lora_targets = ["query", "key", "value", "dense", "fc1", "fc2"]
    elif init == "clip":
        vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        lora_targets = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    else:
        raise ValueError(f"Unknown vision init: {init}")

    if lora:
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_targets,
            lora_dropout=lora_dropout,
            bias="none",
        )
        vision = get_peft_model(vision, config)
        logger.info(
            "Vision LoRA: %d trainable parameters",
            sum(p.numel() for p in vision.parameters() if p.requires_grad),
        )
    else:
        for parameter in vision.parameters():
            parameter.requires_grad = False
    return vision


class MLPProjector(nn.Module):
    """Two-layer MLP mapping vision tokens to the LLM hidden dimension."""

    def __init__(self, vision_dim: int = 768, hidden_dim: int = 2048, llm_dim: int = 896):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        return self.layers(tokens)


def build_llm(
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    lora: bool = True,
    lora_r: int = 64,
    lora_alpha: float = 128.0,
    lora_dropout: float = 0.05,
) -> Qwen2ForCausalLM:
    """Load Qwen2 and optionally wrap it with LoRA."""
    config = AutoConfig.from_pretrained(model_name)
    llm = Qwen2ForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    if tokenizer.convert_tokens_to_ids(IMAGE_TOKEN) == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
        llm.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            embedding = llm.get_input_embeddings().weight
            image_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
            embedding[image_id] = embedding[:-1].mean(dim=0)
    if lora:
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, config)
        llm.print_trainable_parameters()
    else:
        for parameter in llm.parameters():
            parameter.requires_grad = False
    return llm


class LlavaCaptionModel(nn.Module):
    """Vision tower + projector + Qwen2 with image tokens injected as embeddings."""

    def __init__(
        self,
        vision: nn.Module,
        projector: MLPProjector,
        llm: Qwen2ForCausalLM,
        image_token_id: int,
        vision_lora: bool = False,
        text_only: bool = False,
    ) -> None:
        super().__init__()
        self.vision = vision
        self.projector = projector
        self.llm = llm
        self.image_token_id = image_token_id
        self.vision_lora = vision_lora
        self.text_only = text_only
        self.llm_dtype = (
            llm.base_model.dtype if hasattr(llm, "base_model") else llm.dtype
        )
        self.image_token_count: int = 197  # ViT-B/16 at 224x224 (196 patches + CLS)

    def get_image_tokens(self, images: Tensor) -> Tensor:
        """Project vision features into LLM space: [B, num_tokens, llm_dim]."""
        if self.text_only:
            return None
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=images.is_cuda and images.dtype != torch.bfloat16,
        ):
            hidden = self.vision(images).last_hidden_state
        self.image_token_count = hidden.size(1)
        projected = self.projector(hidden.float())
        return projected.to(self.llm_dtype)

    def expand_batch(
        self,
        input_ids: Tensor,
        image_tokens: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ):
        """Expand each <image> placeholder into the projected vision token run.

        The chat template contains exactly one <image> token per sample; the
        model replaces it with ``image_token_count`` embedding vectors, growing
        the sequence by ``image_token_count - 1``. Attention and labels are
        expanded consistently (image positions are masked out of the loss).
        """
        mask = input_ids == self.image_token_id
        if not torch.all(mask.sum(dim=1) == 1):
            if not self.text_only:
                raise ValueError(
                    "Every training/generation sample must contain one <image> token."
                )
            base_embeds = self.llm.get_input_embeddings()(input_ids)
            return base_embeds, attention_mask, labels
        base_embeds = self.llm.get_input_embeddings()(input_ids)
        batch_size, seq_len, hidden = base_embeds.shape
        image_count = self.image_token_count
        new_len = seq_len - 1 + image_count
        device = base_embeds.device
        new_embeds = torch.zeros(batch_size, new_len, hidden, dtype=base_embeds.dtype, device=device)
        new_mask = None
        if attention_mask is not None:
            new_mask = torch.ones(
                batch_size, new_len, dtype=attention_mask.dtype, device=device
            )
        new_labels = None
        if labels is not None:
            new_labels = torch.full(
                (batch_size, new_len), -100, dtype=labels.dtype, device=device
            )
        positions = mask.nonzero()[:, 1]
        for i in range(batch_size):
            position = positions[i].item()
            new_embeds[i, :position] = base_embeds[i, :position]
            new_embeds[i, position : position + image_count] = image_tokens[i]
            new_embeds[i, position + image_count :] = base_embeds[i, position + 1 :]
            if new_mask is not None:
                new_mask[i, :position] = attention_mask[i, :position]
                new_mask[i, position + image_count :] = attention_mask[i, position + 1 :]
            if new_labels is not None:
                new_labels[i, :position] = labels[i, :position]
                new_labels[i, position + image_count :] = labels[i, position + 1 :]
        return new_embeds, new_mask, new_labels

    def forward(
        self,
        images: Tensor,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ):
        if self.text_only:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            return self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
        image_tokens = self.get_image_tokens(images)
        inputs_embeds, attention_mask, labels = self.expand_batch(
            input_ids, image_tokens, attention_mask, labels
        )
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    def generate(
        self,
        images: Tensor,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        **generation_kwargs,
    ):
        if self.text_only:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            dummy_input_ids = torch.zeros_like(input_ids)
            self.llm.config.use_cache = True
            return self.llm.generate(
                input_ids=dummy_input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        image_tokens = self.get_image_tokens(images)
        inputs_embeds, attention_mask, _ = self.expand_batch(
            input_ids, image_tokens, attention_mask
        )
        # transformers >= 4.5x returns only the newly generated tokens when
        # generating from inputs_embeds without input_ids.  Passing a dummy
        # input_ids of the expanded length restores the standard
        # "prompt + generated" output shape used by the eval slicing.
        dummy_input_ids = torch.zeros(
            (inputs_embeds.size(0), inputs_embeds.size(1)),
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        self.llm.config.use_cache = True  # re-enable KV cache for fast inference
        return self.llm.generate(
            input_ids=dummy_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

    def forward_logprobs(
        self,
        images: Tensor,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
        labels: Optional[Tensor],
        image_tokens: Optional[Tensor] = None,
    ):
        """Return logits and expanded labels for per-token log-prob scoring."""
        if image_tokens is None:
            image_tokens = self.get_image_tokens(images)
        inputs_embeds, attention_mask, expanded_labels = self.expand_batch(
            input_ids, image_tokens, attention_mask, labels
        )
        logits = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        ).logits
        return logits, expanded_labels

    def set_trainable(self, stage: str, vision_lora: bool = False) -> None:
        """Enable the parameter groups used by the requested training stage."""
        allow_llm = stage == "lora"
        allow_vision = vision_lora and stage == "lora"
        for name, parameter in self.named_parameters():
            in_projector = name.startswith("projector.")
            in_llm_lora = "llm." in name and "lora_" in name
            in_vision_lora = (
                vision_lora and name.startswith("vision.") and "lora_" in name
            )
            parameter.requires_grad = (
                in_projector
                or (allow_llm and in_llm_lora)
                or (allow_vision and in_vision_lora)
            )
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("Stage %s: %d trainable parameters", stage, trainable)
