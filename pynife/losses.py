from collections.abc import Sequence
from enum import Enum

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch import nn


class CosineLoss(torch.nn.Module):
    def __init__(self, model: SentenceTransformer) -> None:
        """Cosine loss."""
        super().__init__()
        self.model = model
        self.loss_fct = nn.CosineSimilarity()  # type: ignore

    def forward(self, sentence_features: Sequence[dict[str, torch.Tensor]], labels: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass."""
        embeddings = self.model(sentence_features[0])["sentence_embedding"]
        labels = labels[:, : embeddings.shape[1]]
        loss = 1 - self.loss_fct(embeddings, labels[:, : embeddings.shape[1]]).mean()
        return loss


class DistillationCosineLoss(CosineLoss):
    def __init__(self, model: SentenceTransformer, tau: float = 0.07) -> None:
        """Cosine loss."""
        super().__init__(model=model)
        self.tau = tau

    def forward(self, sentence_features: Sequence[dict[str, torch.Tensor]], labels: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Forward pass."""
        embeddings = self.model(sentence_features[0])["sentence_embedding"]
        labels = labels[:, : embeddings.shape[1]]

        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        labels = torch.nn.functional.normalize(labels, p=2, dim=1)

        sim = embeddings @ labels.T

        return F.cross_entropy(sim / self.tau, torch.arange(embeddings.size(0), device=embeddings.device))


class LossFunction(str, Enum):
    """Different loss functions."""

    COSINE = "cosine"
    DISTILLATION_COSINE = "distillation_cosine"


def select_loss(name: str | LossFunction) -> type[nn.Module]:
    """Select loss function by name."""
    try:
        function = LossFunction(name)
    except ValueError as e:
        raise ValueError(
            f"Unknown loss function: {name}, available options are: {[e.value for e in LossFunction]}"
        ) from e
    match function:
        case LossFunction.COSINE:
            return CosineLoss
        case LossFunction.DISTILLATION_COSINE:
            return DistillationCosineLoss
