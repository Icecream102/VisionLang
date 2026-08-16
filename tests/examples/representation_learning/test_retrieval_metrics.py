import pytest
import torch

from examples.representation_learning.retrieval_metrics import compute_retrieval_metrics


def test_multi_caption_perfect_retrieval_is_not_diagonal_only():
    images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    # Captions are deliberately not grouped in image order.
    texts = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    caption_image_ids = torch.tensor([1, 0, 1, 0])

    metrics = compute_retrieval_metrics(
        images,
        texts,
        caption_image_ids,
        ks=(1,),
        chunk_size=1,
        device=torch.device("cpu"),
    )

    assert metrics.image_to_text_r1 == 100.0
    assert metrics.text_to_image_r1 == 100.0
    assert metrics.mean_recall == 100.0


def test_invalid_caption_mapping_is_rejected():
    with pytest.raises(ValueError, match="out-of-range"):
        compute_retrieval_metrics(
            torch.randn(2, 4),
            torch.randn(2, 4),
            torch.tensor([0, 2]),
            ks=(1,),
            device=torch.device("cpu"),
        )
