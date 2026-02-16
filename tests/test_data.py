import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pyarrow.parquet as pq
import pytest
import torch
from datasets import Dataset, IterableDataset

from pynife import data
from pynife.data import build_parquet_shards_from_folder, get_datasets, get_model_name_from_datasets


def _make_input_folder(tmp: Path, *, n_rows: int = 3) -> Path:
    """Create a temporary input folder with texts_0000.txt and pooled_0000.pt."""
    inp = tmp / "input"
    inp.mkdir(parents=True, exist_ok=True)

    texts_path = inp / "texts_0000.txt"
    records = []
    for i in range(n_rows):
        rec = {"text": f"example {i}", "meta": f"m{i}"}
        records.append(rec)

    # write jsonl
    with open(texts_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # write embeddings tensor with shape (n_rows, dim)
    emb = torch.arange(n_rows * 4, dtype=torch.float32).reshape(n_rows, 4)
    pooled_path = inp / "pooled_0000.pt"
    torch.save(emb, pooled_path)

    return inp


def test_build_parquet_shards_and_get_datasets_in_memory() -> None:
    """End-to-end: build parquet shards from jsonl+tensor files and load them in-memory."""
    with TemporaryDirectory() as td:
        td_path = Path(td)
        inp = _make_input_folder(td_path, n_rows=5)
        out = td_path / "out"

        # build parquet shards with small rows_per_shard to force flushes
        build_parquet_shards_from_folder(inp, out, rows_per_shard=2)

        # check shards exist
        train_dir = out / "train"
        shards = sorted(train_dir.glob("*.parquet"))
        assert len(shards) >= 1

        # load datasets in memory
        ds, length = get_datasets([out], in_memory=True)
        assert isinstance(ds, Dataset)
        # length should match number of rows we wrote
        assert length == 5
        # dataset should have 'sentence' and 'label' columns after post-processing
        assert "sentence" in ds.column_names
        assert "label" in ds.column_names


def test_get_datasets_streaming() -> None:
    """Ensure streaming path returns an IterableDataset and correct length."""
    with TemporaryDirectory() as td:
        td_path = Path(td)
        inp = _make_input_folder(td_path, n_rows=4)
        out = td_path / "out2"
        build_parquet_shards_from_folder(inp, out, rows_per_shard=10)

        ds, length = get_datasets([out], in_memory=False)
        # streaming returns an iterable-style dataset and reports correct length
        assert isinstance(ds, IterableDataset)
        assert length == 4


def test_missing_embedding_file_raises() -> None:
    """If an embedding file is missing for a texts file, building shards raises ValueError."""
    with TemporaryDirectory() as td:
        td_path = Path(td)
        inp = td_path / "input2"
        inp.mkdir()
        # write a texts file but no pooled file
        (inp / "texts_0000.txt").write_text(json.dumps({"text": "a"}) + "\n", encoding="utf-8")

        out = td_path / "out3"
        try:
            build_parquet_shards_from_folder(inp, out)
        except ValueError as e:
            assert "does not exist" in str(e)
        else:
            raise AssertionError("Expected ValueError due to missing embedding file")


def test_build_parquet_shards_empty_raises() -> None:
    """If the input folder contains no texts files, a RuntimeError is raised."""
    with TemporaryDirectory() as td:
        td_path = Path(td)
        inp = td_path / "empty"
        inp.mkdir()
        out = td_path / "out_empty"
        try:
            build_parquet_shards_from_folder(inp, out)
        except RuntimeError as e:
            assert "No data found" in str(e)
        else:
            raise AssertionError("Expected RuntimeError for empty input folder")


def test_parquet_shard_content_matches_embeddings() -> None:
    """Verify parquet shard contains expected rows and embedding vectors."""
    with TemporaryDirectory() as td:
        td_path = Path(td)
        inp = _make_input_folder(td_path, n_rows=3)
        out = td_path / "out_parquet"
        build_parquet_shards_from_folder(inp, out, rows_per_shard=10)

        train_dir = out / "train"
        shards = sorted(train_dir.glob("*.parquet"))
        assert len(shards) == 1

        table = pq.read_table(shards[0])
        assert table.num_rows == 3
        # embedding column should be present and be list-like per row
        emb_col = table.column("embedding").to_pylist()
        assert isinstance(emb_col, list)
        assert len(emb_col) == 3
        assert all(len(row) == 4 for row in emb_col)


def test_get_datasets_columns_to_keep_and_limit_shards() -> None:
    """Test that get_datasets respects columns_to_keep and limit_shards."""
    with TemporaryDirectory() as td:
        td_path = Path(td)
        inp = _make_input_folder(td_path, n_rows=3)
        out = td_path / "out_cols"
        # produce one row per shard to create multiple shards
        build_parquet_shards_from_folder(inp, out, rows_per_shard=1)

        # columns_to_keep only 'sentence' should drop 'label'
        ds, length = get_datasets([out], in_memory=True, columns_to_keep={"sentence"})
        assert "sentence" in ds.column_names
        assert "label" not in ds.column_names
        assert length == 3

        # limit_shards restricts how many shards are counted
        _, length2 = get_datasets([out], in_memory=True, limit_shards=1)
        # compute expected number of rows from the first shard file(s)
        shards = sorted((out / "train").glob("*.parquet"))
        expected = 0
        if shards:
            # sum rows from the first `limit_shards` files
            expected = sum(pq.read_metadata(shards[i]).num_rows for i in range(min(1, len(shards))))
        assert length2 == expected


def test_get_model_name_from_single_dataset() -> None:
    """When a single dataset reports a model_name, return it."""
    calls: list[tuple[str, str]] = []

    def fake_get_teacher(path: str, key: str = "model_name") -> str:
        calls.append((path, key))
        return "my-base-model"

    with mock.patch.object(data, "get_teacher_from_metadata", side_effect=fake_get_teacher):
        result = data.get_model_name_from_datasets(["dataset/a"])

    assert result == "my-base-model"
    assert calls == [("dataset/a", "model_name")]


def test_get_model_name_from_no_datasets() -> None:
    """If none of the datasets expose a model_name, return None."""
    with mock.patch.object(data, "get_teacher_from_metadata", return_value=None):
        result = get_model_name_from_datasets(["dataset/a"])  # returns None when nothing found
    assert result is None


def test_get_model_name_from_multiple_datasets_raises() -> None:
    """If datasets report different model names, raise ValueError."""
    responses = {"d1": "model-a", "d2": "model-b"}

    def fake_get_teacher(path: str, key: str = "model_name") -> str | None:
        return responses.get(path, None)

    with mock.patch.object(data, "get_teacher_from_metadata", side_effect=fake_get_teacher):
        with pytest.raises(ValueError):
            get_model_name_from_datasets(["d1", "d2"])
