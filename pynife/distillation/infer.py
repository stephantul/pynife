import json
import logging
from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeVar

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizerBase

from pynife.data import build_parquet_shards_from_folder
from pynife.utilities import batchify

logger = logging.getLogger(__name__)


T = TypeVar("T")


def _batchify(records: Iterator[T] | Iterable[T], batch_size: int) -> Iterator[list[T]]:
    """Turn a list of texts into batches."""
    batch_iterator = batchify(records, batch_size)
    pbar = tqdm(total=0, unit="batches", desc="Creating batches")
    for batch in batch_iterator:
        pbar.update(1)
        yield batch

    pbar.close()


def _write_data(path: Path, pooled: list[torch.Tensor], records: list[dict[str, str]], shard_index: int) -> None:
    """Write out the data to disk."""
    pooled_tensor = torch.cat(pooled, dim=0).float()
    torch.save(pooled_tensor, path / f"pooled_{shard_index:04d}.pt")
    with open(path / f"texts_{shard_index:04d}.txt", "w", encoding="utf-8") as f:
        for record in records:
            record["text"] = record.pop("truncated")
            line = json.dumps(
                record,
                ensure_ascii=False,
            )
            f.write(line + "\n")


def _tokenize(
    strings: list[str], tokenizer: PreTrainedTokenizerBase, max_length: int
) -> tuple[BatchEncoding, list[str]]:
    """Tokenize a list of strings using a HuggingFace tokenizer.

    This is mainly a helper function; it also returns the truncated strings, so that we don't have to
    re-tokenize them later to find out how they were truncated.

    Args:
        strings: The list of strings to tokenize.
        tokenizer: The HuggingFace tokenizer to use.
        max_length: The maximum sequence length.

    Returns:
        A tuple of (BatchEncoding, list of truncated strings).

    """
    strings = [x.strip()[:10000] for x in strings]  # Hard limit to 10k chars
    tokenized = tokenizer(
        strings,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offset_mapping = tokenized.pop("offset_mapping")

    # The offset mapping is a 3D array (batch, max_length_in_batch, 2), with offsets
    # Where the final dimension is start, end indices. So by taking the max end index
    # we know the length to which the tokenizer tokenized the string.
    lengths = np.asarray(offset_mapping)[:, :, 1].max(axis=1)
    return tokenized, [string[:length] for string, length in zip(strings, lengths, strict=False)]


def _generate_embeddings(
    model: SentenceTransformer,
    records: Iterator[dict[str, str]] | Iterable[dict[str, str]],
    output_dir: str | Path,
    batch_size: int = 96,
    max_length: int = 512,
    save_every: int = 8192,
    limit_batches: int | None = None,
) -> None:
    """Generate embeddings for a stream of texts using a SentenceTransformer model.

    This is mainly used as an inner loop for the knowledge distillation process.
    We get N texts, and create embeddings for them using the teacher model.
    These embeddings are then used to train the student model.

    We return the pooled output for each text.

    Args:
        model: The SentenceTransformer model to use for inference.
        records: A sequence of records to infer embeddings for.
        output_dir: The name of the directory to save the results to.
        batch_size: The batch size to use for inference. Defaults to 96.
        max_length: The maximum sequence length for tokenization. Defaults to 512.
        save_every: Save intermediate results every N batches. Defaults to 8192.
        limit_batches: An optional limit on the number of batches to process.

    """
    model.eval()

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    shards_saved = 0

    all_pooled, accumulated_records = [], []
    tokenizer: PreTrainedTokenizerBase = model.tokenizer

    original_max_length = model[0].max_seq_length
    assert original_max_length is not None
    assert isinstance(original_max_length, int)
    if max_length > original_max_length:
        logger.warning(
            f"Warning: max_length {max_length} is greater than the model's max_length {original_max_length}. "
            "Not changing it."
        )
    else:
        model[0].max_seq_length = max_length  # type: ignore[assignment]

    seen = 0
    with torch.inference_mode(), torch.autocast(device_type=model.device.type, dtype=torch.bfloat16):
        for batch in _batchify(records, batch_size=batch_size):
            texts = [record["text"] for record in batch]
            features, truncated_strings = _tokenize(texts, tokenizer, max_length)
            features_dict = {k: v.to(model.device) for k, v in features.items()}

            # One forward through the whole SentenceTransformer
            out = model(features_dict)
            pooled = out["sentence_embedding"].cpu()

            for record, truncated in zip(batch, truncated_strings, strict=False):
                record["truncated"] = truncated

                accumulated_records.append(record)
            all_pooled.append(pooled.cpu())
            del pooled
            torch.cuda.empty_cache()

            seen += 1
            if seen % save_every == 0:
                logger.info(f"Seen {seen * batch_size} texts, saving intermediate results to disk.")
                _write_data(path, all_pooled, accumulated_records, shards_saved)
                shards_saved += 1
                all_pooled = []
                accumulated_records = []

            if limit_batches is not None and seen >= limit_batches:
                logger.info(f"Reached limit of {limit_batches} batches, stopping inference.")
                break

    if accumulated_records:
        _write_data(path, all_pooled, accumulated_records, shards_saved)


def _workdir(temp_folder: str | None) -> AbstractContextManager[str]:
    """Create a temp folder if needed. For <= 3.11 temporary directories."""
    if temp_folder is None:
        return TemporaryDirectory()
    Path(temp_folder).mkdir(parents=True, exist_ok=True)
    return nullcontext(temp_folder)


def generate_and_save_embeddings(
    model: SentenceTransformer,
    model_name: str,
    dataset_name: str,
    output_folder: str | Path,
    records: Iterator[dict[str, str]] | Iterable[dict[str, str]],
    temp_folder: str | None = None,
    limit_batches: int | None = None,
    batch_size: int = 512,
    save_every: int = 256,
    max_length: int = 512,
) -> None:
    """Run inference and save the results to parquet shards.

    This runs inference over an iterable or iterator of records. Each record is a dictionary with
    at the very least a "text" key, which is the field that will be featurized.
    """
    with _workdir(temp_folder) as dir_name:
        _generate_embeddings(
            model,
            records,
            batch_size=batch_size,
            output_dir=dir_name,
            save_every=save_every,
            limit_batches=limit_batches,
            max_length=max_length,
        )

        logger.info("Converting dataset to shards...")
        build_parquet_shards_from_folder(dir_name, output_folder, model_name=model_name, dataset_name=dataset_name)
        logger.info(f"Converted dataset saved to {output_folder}")
