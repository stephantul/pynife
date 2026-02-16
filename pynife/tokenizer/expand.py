import logging
import re
from collections.abc import Iterable, Iterator

import numpy as np
from skeletoken import TokenizerModel
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from pynife.tokenizer.datamodels import VocabItem
from pynife.utilities import batchify

FreqTuple = tuple[str, int]
DatasetIterable = Iterator[VocabItem] | Iterable[VocabItem]

_NUMBERS_RE = re.compile(r"^\d+$")

logger = logging.getLogger(__name__)


def _prune_tokenizer(
    tokenizer_model: TokenizerModel, dataset: DatasetIterable, min_subword_frequency: int
) -> TokenizerModel:
    """Prune a tokenizer by removing tokens that occur less than min_subword_frequency times in the provided data."""
    tokenizer_object = tokenizer_model.to_tokenizer()

    old_vocab_size = tokenizer_object.get_vocab_size()
    original_vocab_counts = np.zeros(old_vocab_size, dtype=np.int32)

    for batch in tqdm(batchify(dataset, batch_size=10_000), desc="Counting subword frequencies"):
        # Find the frequency of each subword in the original vocabulary
        token_strings = [item["token"] for item in batch]
        token_counts = [item["frequency"] for item in batch]

        tokenized = tokenizer_object.encode_batch_fast(token_strings, add_special_tokens=False)
        for encoding, count in zip(tokenized, token_counts, strict=False):
            counts = np.bincount(encoding.ids, minlength=old_vocab_size)
            original_vocab_counts += counts * count

    vocab_to_remove = original_vocab_counts < min_subword_frequency

    vocabulary = {index: token for token, index in tokenizer_model.vocabulary.items()}
    tokens_to_remove = [vocabulary[index] for index in np.flatnonzero(vocab_to_remove)]
    tokenizer_model = tokenizer_model.remove_tokens_from_vocabulary(tokens_to_remove)

    logger.info("Removed %d tokens from vocabulary.", np.sum(vocab_to_remove))
    return tokenizer_model


def _add_tokens_to_tokenizer(
    tokenizer_model: TokenizerModel, dataset: DatasetIterable, filter_numbers: bool, new_vocab_size: int
) -> TokenizerModel:
    """Add new tokens to a tokenizer up to the specified new vocabulary size."""
    n_tokens_to_add = new_vocab_size - tokenizer_model.vocabulary_size
    if n_tokens_to_add <= 0:
        logger.info("No tokens to add to vocabulary.")
        return tokenizer_model
    logger.info("Adding %d new tokens to vocabulary.", n_tokens_to_add)

    tokens_added = 0

    dataset = sorted(dataset, key=lambda x: x["frequency"], reverse=True)
    # Sort by frequency and add tokens until we reach the new vocab size
    for item in dataset:
        token = item["token"]
        if filter_numbers and _NUMBERS_RE.match(token):
            continue
        if tokens_added >= n_tokens_to_add:
            break
        if token in tokenizer_model.vocabulary:
            continue
        tokenizer_model = tokenizer_model.add_token_to_vocabulary(token)
        tokens_added += 1

    return tokenizer_model


def expand_tokenizer(
    tokenizer: PreTrainedTokenizerFast,
    dataset: DatasetIterable,
    min_subword_frequency: int,
    new_vocab_size: int,
    filter_numbers: bool,
) -> PreTrainedTokenizerFast:
    """Expand a tokenizer's vocabulary by counting frequencies.

    Args:
        tokenizer: The tokenizer to expand.
        dataset: A dataset with a "token" and "frequency" column.
        min_subword_frequency: Minimum frequency for subwords to be kept in the tokenizer.
        new_vocab_size: The desired vocabulary size after expansion.
        filter_numbers: Whether to filter out tokens that are purely numeric.

    Returns:
        The expanded tokenizer.

    """
    tokenizer_model = TokenizerModel.from_transformers_tokenizer(tokenizer)
    if min_subword_frequency > 0:
        tokenizer_model = _prune_tokenizer(tokenizer_model, dataset, min_subword_frequency)
    new_tokenizer = _add_tokens_to_tokenizer(tokenizer_model, dataset, filter_numbers, new_vocab_size)
    return new_tokenizer.to_transformers()
