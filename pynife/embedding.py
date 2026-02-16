from __future__ import annotations

from typing import Any, TypeVar, cast

import numpy as np
import torch
from sentence_transformers.models import Module, StaticEmbedding
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

T = TypeVar("T", bound=Module)


def _load_weights_for_model(
    model: T,
    model_name_or_path: str,
    subfolder: str = "",
    token: bool | str | None = None,
    cache_folder: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> T:
    """Load weights for a given model from a pretrained model."""
    weights = Module.load_torch_weights(
        model=model,
        model_name_or_path=model_name_or_path,
        subfolder=subfolder,
        token=token,
        cache_folder=cache_folder,
        revision=revision,
        local_files_only=local_files_only,
    )
    return cast(T, weights)


def _load_weights_as_tensor_dict(
    model_name_or_path: str,
    subfolder: str = "",
    token: bool | str | None = None,
    cache_folder: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Load embedding weights from a pretrained model as a tensor."""
    weights = Module.load_torch_weights(
        model_name_or_path=model_name_or_path,
        subfolder=subfolder,
        token=token,
        cache_folder=cache_folder,
        revision=revision,
        local_files_only=local_files_only,
    )
    return cast(dict[str, torch.Tensor], weights)


class TrainableStaticEmbedding(StaticEmbedding):
    def __init__(
        self,
        tokenizer: Tokenizer | PreTrainedTokenizerFast,
        embedding_weights: np.ndarray | torch.Tensor | None = None,
        embedding_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize static embedding layer."""
        super().__init__(tokenizer, embedding_weights, embedding_dim, **kwargs)
        self._max_seq_length = 512

    def tokenize(self, texts: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Tokenize the texts."""
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        encodings_ids = [torch.Tensor(encoding.ids[: self.max_seq_length]).long() for encoding in encodings]

        input_ids = torch.nn.utils.rnn.pad_sequence(encodings_ids, batch_first=True, padding_value=0)
        return {"input_ids": input_ids}

    def forward(self, features: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass."""
        x = self.embedding(features["input_ids"])
        features["sentence_embedding"] = x
        return features

    @property
    def max_seq_length(self) -> int:
        """Get the maximum sequence length."""
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """Set the maximum sequence length."""
        self._max_seq_length = value

    @classmethod
    def load(  # type: ignore[misc]
        cls: type[T],
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> T:
        """Load a TrainableStaticEmbedding from a pretrained model."""
        tokenizer_path = cls.load_file_path(
            model_name_or_path,
            filename="tokenizer.json",
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        assert tokenizer_path is not None
        tokenizer = Tokenizer.from_file(tokenizer_path)

        weights_dict = _load_weights_as_tensor_dict(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        try:
            weights = weights_dict["embedding.weight"]
        except KeyError:  # pragma: no cover
            weights = weights_dict["embeddings"]  # pragma: no cover
        initialized = cls(dim=weights.shape[1], tokenizer=tokenizer, embedding_weights=weights, **kwargs)
        return _load_weights_for_model(
            initialized,
            model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
