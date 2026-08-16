# tests the sliding-window dataset on a throwaway corpus in tmp_path

import pytest
import torch
from torch.utils.data import DataLoader

from minigpt.data.dataset import CorpusDataset
from minigpt.tokenizer.create_tokenizer import Tokenizer

LINES = ["the cat runs quickly", "a dog sees the house", "she builds the answer"] * 40
CONTEXT_LEN = 8


@pytest.fixture(scope="module")
def tokenizer():
    tok = Tokenizer(lowercase=False)
    tok.train_bpe(LINES, num_merges=20)
    tok.build_vocab(corpus=LINES)
    return tok


@pytest.fixture
def dataset(tmp_path, tokenizer):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(LINES), encoding="utf-8")
    return CorpusDataset(str(corpus), tokenizer, context_len=CONTEXT_LEN)


def test_item_is_input_and_shifted_target(dataset):
    x, y = dataset[0]
    assert x.shape == (CONTEXT_LEN,) and y.shape == (CONTEXT_LEN,)
    assert torch.equal(y[:-1], x[1:])
    assert x.dtype == torch.long and y.dtype == torch.long


def test_length_is_tokens_minus_context(dataset):
    assert len(dataset) == len(dataset.ids) - CONTEXT_LEN


def test_ids_are_inside_the_vocab(dataset, tokenizer):
    assert dataset.ids.min().item() >= 0
    assert dataset.ids.max().item() < tokenizer.vocab_size


def test_batches_stack(dataset):
    xb, yb = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
    assert xb.shape == (4, CONTEXT_LEN) and yb.shape == (4, CONTEXT_LEN)


def test_corpus_shorter_than_context_is_rejected(tmp_path, tokenizer):
    corpus = tmp_path / "tiny.txt"
    corpus.write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError):
        CorpusDataset(str(corpus), tokenizer, context_len=512)
