from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal, cast, overload

import numpy as np
import pyarrow.parquet as pq
import torch
from datasets import Dataset, Features, IterableDataset, Value, load_dataset
from datasets import Sequence as DatasetSequenceFeature
from huggingface_hub import HfApi
from tqdm import tqdm

from pynife.cards.dataset_card import generate_dataset_card
from pynife.utilities import get_teacher_from_metadata


def _pair_stream(
    txt_path: Path,
    emb_path: Path,
) -> Iterator[tuple[dict[str, str], np.ndarray]]:
    """Stream aligned (text, embedding_row) pairs from a txt jsonl and tensor file.

    The text path can contain arbitrary dictionary fields, but will also contain a "text" field.

    Args:
        txt_path: Path to the text file (jsonl).
        emb_path: Path to the embeddings file (torch tensor).

    Yields:
        Tuples of (record dict, embedding numpy array).

    """
    embs = torch.load(emb_path)
    embs = embs.float().numpy()

    with open(txt_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            yield item, embs[i]


def _iter_all_pairs(
    root: Path,
) -> Iterator[tuple[dict[str, str], np.ndarray]]:
    """Iterate all (text, embedding) pairs across the folder in a streaming fashion."""
    for txt_path in tqdm(sorted(root.glob("**/*.txt"))):
        name = txt_path.stem.split("_")[1]
        emb_path = txt_path.parent / f"pooled_{name}.pt"
        if not emb_path.exists():
            raise ValueError(f"Embedding file {emb_path} does not exist")

        yield from _pair_stream(txt_path, emb_path)


def build_parquet_shards_from_folder(
    path: str | Path,
    out_dir: str | Path,
    *,
    rows_per_shard: int = 100_000,
    model_name: str = "unknown-model",
    dataset_name: str = "unknown-dataset",
) -> None:
    """Stream over (text, embedding) pairs and write sharded parquet files to disk."""
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_iter = _iter_all_pairs(path)
    try:
        first_record, first_emb = next(pair_iter)
    except StopIteration as e:
        raise RuntimeError("No data found under the given path.") from e

    keys = list(first_record.keys())
    first_key = keys[0]
    D = int(first_emb.shape[0])
    features = Features(
        {
            **{k: Value("string") for k in keys},
            "embedding": DatasetSequenceFeature(Value("float32"), length=D),
        }
    )

    # Start first buffer with the peeked row
    buffers = {k: [first_record[k]] for k in keys}
    buf_embeddings = [first_emb.astype("float32")]

    shard_id = 0
    rows_emitted = 1

    def _flush_buffer() -> None:
        nonlocal shard_id, buffers, buf_embeddings
        if not buffers:
            return
        dataset = Dataset.from_dict(
            {k: buffers[k] for k in keys} | {"embedding": np.vstack(buf_embeddings)}, features=features
        )
        # Write a single parquet file per shard for simple globbing later
        shard_path = out_dir / "train" / f"shard_{shard_id:05d}.parquet"
        dataset.to_parquet(str(shard_path))
        # release memory
        del dataset
        buffers.clear()
        buf_embeddings.clear()
        shard_id += 1

    for record, emb in pair_iter:
        for k, v in record.items():
            buffers.setdefault(k, []).append(str(v))
        buf_embeddings.append(emb.astype("float32"))
        rows_emitted += 1
        if len(buffers[first_key]) >= rows_per_shard:
            _flush_buffer()

    # Flush remainder
    _flush_buffer()

    generate_dataset_card(
        model_name=model_name,
        dataset_name=dataset_name,
        length=D,
        size=rows_emitted,
        size_kind="examples",
    )
    # Also write the dataset card to README.md in the output folder so HF metadata is available
    readme = generate_dataset_card(
        model_name=model_name,
        dataset_name=dataset_name,
        length=D,
        size=rows_emitted,
        size_kind="examples",
    )
    with open(out_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)

    # Also write a small metadata.json next to the README so callers (or hf_hub_download)
    # can easily discover which model produced this converted dataset and the full
    # HF dataset identifier used during conversion.
    meta = {"model_name": model_name, "dataset_name": dataset_name}
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _post_process_dataset(dataset: Dataset | IterableDataset, to_keep: set[str]) -> Dataset | IterableDataset:
    # Rename only if the source exists and the target is kept.
    assert dataset.column_names is not None
    columns = set(dataset.column_names)
    # Sentence transformers specific stuff.
    if "text" in columns and "sentence" in to_keep:
        dataset = dataset.rename_column("text", "sentence")
        columns.remove("text")
        columns.add("sentence")
    if "embedding" in columns and "label" in to_keep:
        dataset = dataset.rename_column("embedding", "label")
        columns.remove("embedding")
        columns.add("label")

    # Remove all columns not in to_keep
    to_remove = columns - to_keep
    dataset = dataset.remove_columns(list(to_remove))

    return dataset


def _collect_parquet_shards(path_or_repo: Path) -> list[Path]:
    """Return a sorted list of train/*.parquet for local dir or HF dataset repo.

    The sort order is determined lexicographically by path.
    If a HF dataset repo is given, we first download a local snapshot, and then return any shards
    downloaded there. We never load the dataset as is.

    Args:
        path_or_repo: Local path or HF dataset repo name.

    Returns:
        List of parquet shard paths.

    """
    if path_or_repo.is_dir():
        shards = path_or_repo.glob("**/*.parquet")
    else:
        api = HfApi()  # pragma: no cover
        path = api.snapshot_download(repo_id=path_or_repo.as_posix(), repo_type="dataset")
        local = Path(path)  # pragma: no cover
        shards = local.glob("**/*.parquet")  # pragma: no cover
    return sorted(shards, key=lambda p: p.as_posix())


@overload
def get_datasets(
    paths: Sequence[str] | Sequence[Path],
    in_memory: Literal[True],
    limit_shards: int | None = None,
    columns_to_keep: frozenset[str] | set[str] = frozenset({"sentence", "label", "question"}),
) -> tuple[Dataset, int]: ...


@overload
def get_datasets(
    paths: Sequence[str] | Sequence[Path],
    in_memory: Literal[False],
    limit_shards: int | None = None,
    columns_to_keep: frozenset[str] | set[str] = frozenset({"sentence", "label", "question"}),
) -> tuple[IterableDataset, int]: ...


@overload
def get_datasets(
    paths: Sequence[str] | Sequence[Path],
    in_memory: bool,
    limit_shards: int | None = None,
    columns_to_keep: frozenset[str] | set[str] = frozenset({"sentence", "label", "question"}),
) -> tuple[IterableDataset | Dataset, int]: ...


def get_datasets(
    paths: Sequence[Path] | Sequence[str],
    in_memory: bool = True,
    limit_shards: int | None = None,
    columns_to_keep: frozenset[str] | set[str] = frozenset({"sentence", "label"}),
) -> tuple[Dataset | IterableDataset, int]:
    """Get datasets from the given paths.

    The datasets can be loaded in memory or streamed from disk. In either case, we assume that
    the datasets have a "train" split. In all cases, we assume that the datasets have "text" and "embedding"
    columns, which we rename to "sentence" and "label" respectively. We drop all other columns
    except those specified in `columns_to_keep`, which are "sentence" and "label" by default.

    Args:
        paths: Paths to the datasets.
        in_memory: Whether to load the datasets in memory or stream them from disk.
        limit_shards: If streaming, limit the number of shards to load from each dataset.
        columns_to_keep: Columns to keep in the dataset.

    Returns:
        A tuple of (dataset, length), where dataset is either a Dataset or IterableDataset
        depending on the `in_memory` flag, and length is the total number of records across all datasets.

    """
    paths = [Path(p) for p in paths]
    shards: list[Path] = []
    for path in paths:
        ps = _collect_parquet_shards(path)
        if limit_shards is not None:
            ps = ps[:limit_shards]
        shards.extend(ps)

    # Get the length by reading metadata only.
    # We might stream later, so we can't rely on length.
    length = sum(pq.read_metadata(p).num_rows for p in shards)

    data_files = [p.as_posix() for p in shards]
    ds = cast(
        Dataset | IterableDataset,
        load_dataset(
            "parquet", data_files=data_files, split="train", streaming=not in_memory, columns=["text", "embedding"]
        ),
    )

    if not in_memory:
        ds = cast(IterableDataset, ds)
        ds = ds.shuffle(buffer_size=50_000, seed=42)
    else:
        ds = cast(Dataset, ds)
        ds = ds.shuffle(seed=42)

    dataset = _post_process_dataset(ds, to_keep=set(columns_to_keep))
    return dataset, length


def get_model_name_from_datasets(datasets: list[str]) -> str | None:
    """Get a model name based on the datasets."""
    model_names = set()
    for dataset in datasets:
        model_name = get_teacher_from_metadata(dataset, key="model_name")
        if model_name:
            model_names.add(model_name)
    if len(model_names) > 1:
        raise ValueError(f"Multiple base models found: {model_names}")
    if not model_names:
        return None
    return next(iter(model_names))
