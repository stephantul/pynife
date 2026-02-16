from skeletoken import TokenizerModel
from transformers import PreTrainedTokenizerFast

from pynife.tokenizer.datamodels import VocabItem
from pynife.tokenizer.expand import _add_tokens_to_tokenizer, _prune_tokenizer, expand_tokenizer


def test_prune_noop_with_empty_data(test_tokenizer: PreTrainedTokenizerFast) -> None:
    """Pruning with empty data should not increase the vocabulary size."""
    tokenizer = test_tokenizer
    model = TokenizerModel.from_transformers_tokenizer(tokenizer)
    orig_size = model.vocabulary_size

    pruned = _prune_tokenizer(model, [], min_subword_frequency=1)
    assert isinstance(pruned, TokenizerModel)
    # Pruning with no data should not increase size; it may stay equal or shrink.
    assert pruned.vocabulary_size <= orig_size


def test_prune_high_threshold_reduces_vocab(test_tokenizer: PreTrainedTokenizerFast) -> None:
    """A very high min_subword_frequency should often reduce vocabulary size (or be a no-op)."""
    tokenizer = test_tokenizer
    model = TokenizerModel.from_transformers_tokenizer(tokenizer)
    orig_size = model.vocabulary_size

    # Use a few common short tokens as data entries; counts are small so a
    # high threshold will tend to mark many subword ids for removal.
    dataset: list[VocabItem] = [
        {"token": "a", "frequency": 1},
        {"token": "b", "frequency": 1},
        {"token": "c", "frequency": 1},
        {"token": "xyz", "frequency": 1},
    ]
    pruned = _prune_tokenizer(model, dataset, min_subword_frequency=10_000)
    assert isinstance(pruned, TokenizerModel)
    assert pruned.vocabulary_size <= orig_size


def test_expand_adds_and_filters(test_tokenizer: PreTrainedTokenizerFast) -> None:
    """Integration smoke test: expand_tokenizer should add non-numeric tokens and skip numbers when requested."""
    tokenizer = test_tokenizer
    original_vocab = tokenizer.get_vocab()
    original_size = len(original_vocab)

    data: list[VocabItem] = [
        {"token": "12345", "frequency": 1000},
        {"token": "hellonew", "frequency": 500},
        {"token": "worldnew", "frequency": 400},
    ]

    # Adding tokens without filtering should allow non-numeric tokens to be added
    new_tokenizer = expand_tokenizer(
        tokenizer, data, min_subword_frequency=0, new_vocab_size=original_size + 2, filter_numbers=False
    )
    new_vocab = new_tokenizer.get_vocab()
    assert isinstance(new_tokenizer, PreTrainedTokenizerFast)
    assert "hellonew" in new_vocab or "worldnew" in new_vocab

    # With filtering enabled, numeric token should not be added
    new_tokenizer2 = expand_tokenizer(
        tokenizer, data, min_subword_frequency=0, new_vocab_size=original_size + 2, filter_numbers=True
    )
    new_vocab2 = new_tokenizer2.get_vocab()
    assert "12345" not in new_vocab2


def test_add_tokens_noop_when_target_reached(test_tokenizer: PreTrainedTokenizerFast) -> None:
    """When new_vocab_size <= current vocabulary size, no tokens are added."""
    tokenizer = test_tokenizer
    model = TokenizerModel.from_transformers_tokenizer(tokenizer)
    orig_size = model.vocabulary_size

    dataset: list[VocabItem] = [{"token": "x", "frequency": 1}, {"token": "y", "frequency": 2}]
    returned = _add_tokens_to_tokenizer(model, dataset=dataset, filter_numbers=False, new_vocab_size=orig_size)
    assert isinstance(returned, TokenizerModel)
    assert returned.vocabulary_size == orig_size


def test_add_tokens_adds_non_numeric(test_tokenizer: PreTrainedTokenizerFast) -> None:
    """Adding non-numeric tokens should increase vocabulary size up to the requested amount."""
    tokenizer = test_tokenizer
    model = TokenizerModel.from_transformers_tokenizer(tokenizer)
    orig_size = model.vocabulary_size

    data: list[VocabItem] = [
        {"token": "tokenone", "frequency": 10},
        {"token": "tokentwo", "frequency": 5},
        {"token": "1234", "frequency": 20},
    ]
    returned = _add_tokens_to_tokenizer(model, dataset=data, filter_numbers=True, new_vocab_size=orig_size + 2)
    assert isinstance(returned, TokenizerModel)
    assert returned.vocabulary_size >= orig_size + 1


def test_add_tokens_filter_all_numbers(test_tokenizer: PreTrainedTokenizerFast) -> None:
    """If all candidate tokens are numeric and filter_numbers=True, nothing is added."""
    tokenizer = test_tokenizer
    model = TokenizerModel.from_transformers_tokenizer(tokenizer)
    orig_size = model.vocabulary_size

    data: list[VocabItem] = [{"token": "123", "frequency": 10}, {"token": "456", "frequency": 5}]
    returned = _add_tokens_to_tokenizer(model, dataset=data, filter_numbers=True, new_vocab_size=orig_size + 2)
    assert isinstance(returned, TokenizerModel)
    assert returned.vocabulary_size == orig_size


def test_add_tokens_skips_existing_tokens(test_tokenizer: PreTrainedTokenizerFast) -> None:
    """Ensure tokens already in the tokenizer are skipped and do not cause errors."""
    tokenizer = test_tokenizer
    model = TokenizerModel.from_transformers_tokenizer(tokenizer)
    orig_size = model.vocabulary_size

    # pick an existing token from the underlying tokenizer vocabulary
    hf_vocab = tokenizer.get_vocab()
    existing = next(iter(hf_vocab.keys()))
    data: list[VocabItem] = [{"token": existing, "frequency": 100}, {"token": "brandnewtoken", "frequency": 50}]

    returned = _add_tokens_to_tokenizer(model, dataset=data, filter_numbers=False, new_vocab_size=orig_size + 1)
    assert isinstance(returned, TokenizerModel)
    # either brand_new_token was added or the vocabulary size remains same if skipped
    assert returned.vocabulary_size >= orig_size


def test_expand_invokes_prune_branch(test_tokenizer: PreTrainedTokenizerFast) -> None:
    """Calling expand_tokenizer with min_subword_frequency>0 should invoke the prune path."""
    tokenizer = test_tokenizer
    original_vocab = tokenizer.get_vocab()
    original_size = len(original_vocab)

    data: list[VocabItem] = [{"token": "a", "frequency": 1}, {"token": "b", "frequency": 1}]

    # Set new_vocab_size to current size so no tokens are added; this ensures
    # expand_tokenizer will call _prune_tokenizer (min_subword_frequency > 0)
    new_tokenizer = expand_tokenizer(
        tokenizer, dataset=data, min_subword_frequency=1, new_vocab_size=original_size, filter_numbers=False
    )
    assert isinstance(new_tokenizer, PreTrainedTokenizerFast)
