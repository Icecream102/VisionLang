"""Memory-bounded metrics for many-to-many image-text retrieval."""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class RetrievalMetrics:
    """Standard bidirectional retrieval metrics, expressed as percentages."""

    image_to_text_r1: float
    image_to_text_r5: float
    image_to_text_r10: float
    text_to_image_r1: float
    text_to_image_r5: float
    text_to_image_r10: float
    mean_recall: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _validate_inputs(
    image_embeddings: Tensor, text_embeddings: Tensor, caption_image_ids: Tensor
) -> None:
    if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("Embeddings must have shape [num_examples, embedding_dim].")
    if image_embeddings.shape[1] != text_embeddings.shape[1]:
        raise ValueError("Image and text embeddings must have the same dimension.")
    if caption_image_ids.ndim != 1 or len(caption_image_ids) != len(text_embeddings):
        raise ValueError("caption_image_ids must contain one image index per caption.")
    if len(image_embeddings) == 0 or len(text_embeddings) == 0:
        raise ValueError("At least one image and one caption are required.")
    if caption_image_ids.min().item() < 0 or caption_image_ids.max().item() >= len(
        image_embeddings
    ):
        raise ValueError("caption_image_ids contains an out-of-range image index.")


def _recall_at_k(hits: Tensor, ks: Sequence[int]) -> Dict[int, float]:
    return {k: float(hits[:, :k].any(dim=1).float().mean().item() * 100) for k in ks}


@torch.inference_mode()
def compute_retrieval_metrics(
    image_embeddings: Tensor,
    text_embeddings: Tensor,
    caption_image_ids: Tensor,
    ks: Iterable[int] = (1, 5, 10),
    chunk_size: int = 1024,
    device: torch.device = None,
) -> RetrievalMetrics:
    """Compute COCO/Flickr-style image-text retrieval metrics.

    ``caption_image_ids[j]`` identifies the image described by caption ``j``. This
    supports the five-positive-caption COCO protocol instead of incorrectly treating
    the diagonal of a square score matrix as the only positive pair. Similarities are
    evaluated in chunks, so the full image-by-caption matrix is never materialized.
    """
    ks = tuple(sorted(set(int(k) for k in ks)))
    if not ks or min(ks) <= 0:
        raise ValueError("ks must contain positive integers.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    caption_image_ids = caption_image_ids.to(dtype=torch.long, device="cpu")
    _validate_inputs(image_embeddings, text_embeddings, caption_image_ids)
    max_k = max(ks)
    if max_k > len(text_embeddings) or max_k > len(image_embeddings):
        raise ValueError("Every requested k must not exceed both gallery sizes.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_embeddings = F.normalize(image_embeddings.float(), dim=-1)
    text_embeddings = F.normalize(text_embeddings.float(), dim=-1)
    gallery_images = image_embeddings.to(device)
    gallery_texts = text_embeddings.to(device)

    image_hits = []
    for start in range(0, len(image_embeddings), chunk_size):
        end = min(start + chunk_size, len(image_embeddings))
        scores = image_embeddings[start:end].to(device) @ gallery_texts.T
        indices = scores.topk(max_k, dim=1).indices.cpu()
        targets = torch.arange(start, end).unsqueeze(1)
        image_hits.append(caption_image_ids[indices].eq(targets))

    text_hits = []
    for start in range(0, len(text_embeddings), chunk_size):
        end = min(start + chunk_size, len(text_embeddings))
        scores = text_embeddings[start:end].to(device) @ gallery_images.T
        indices = scores.topk(max_k, dim=1).indices.cpu()
        targets = caption_image_ids[start:end].unsqueeze(1)
        text_hits.append(indices.eq(targets))

    image_recall = _recall_at_k(torch.cat(image_hits), ks)
    text_recall = _recall_at_k(torch.cat(text_hits), ks)
    values = [image_recall[k] for k in ks] + [text_recall[k] for k in ks]
    return RetrievalMetrics(
        image_to_text_r1=image_recall.get(1, float("nan")),
        image_to_text_r5=image_recall.get(5, float("nan")),
        image_to_text_r10=image_recall.get(10, float("nan")),
        text_to_image_r1=text_recall.get(1, float("nan")),
        text_to_image_r5=text_recall.get(5, float("nan")),
        text_to_image_r10=text_recall.get(10, float("nan")),
        mean_recall=sum(values) / len(values),
    )
