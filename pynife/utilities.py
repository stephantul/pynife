import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from huggingface_hub import HfApi, ModelCard

logger = logging.getLogger(__name__)


T = TypeVar("T")


def iterable_iterator_dispatch(
    stream: Iterable[T] | Iterator[T],
) -> Iterator[T]:
    """Convert an iterable or iterator into an iterator."""
    return iter(stream)


def batchify(stream: Iterator[T] | Iterable[T], batch_size: int) -> Iterator[list[T]]:
    """Turn an iterator over something into batches."""
    # If we got an iterable, turn it into an iterator
    stream = iterable_iterator_dispatch(stream)

    batch: list[T] = []
    while True:
        if len(batch) < batch_size:
            try:
                batch.append(next(stream))
            except StopIteration:
                if batch:
                    yield batch
                break
        else:
            yield batch
            batch = []


def get_teacher_from_metadata(path: str | Path, key: str = "base_model") -> str:
    """Get metadata file for a given model or dataset from the Hugging Face Hub or a local path."""
    path = Path(path)
    if path.exists() and path.is_dir():
        readme_path = str(path / "README.md")
    else:
        api = HfApi()
        try:
            readme_path = api.hf_hub_download(repo_id=str(path), filename="README.md")
        except Exception as e:
            raise FileNotFoundError(f"Could not find README.md for model at {path}") from e

    model_card = ModelCard.load(readme_path)
    model_name: str | None = getattr(model_card.data, key, None)
    if model_name is None:
        raise ValueError(f"Could not find '{key}' in metadata for model at {path}")
    return model_name
