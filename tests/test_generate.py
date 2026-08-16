# smoke test for the sampling loop: tiny model + tiny tokenizer, short decode

import pytest
import torch

from minigpt.config import LLMConfig
from minigpt.generate import apply_top_k, apply_top_p, generate
from minigpt.model.gpt import MiniGPT
from minigpt.tokenizer.create_tokenizer import Tokenizer

CORPUS = ["the cat runs quickly", "a dog sees the house"] * 20


@pytest.fixture(scope="module")
def tokenizer():
    tok = Tokenizer(lowercase=False)
    tok.train_bpe(CORPUS, num_merges=20)
    tok.build_vocab(corpus=CORPUS)
    return tok


@pytest.fixture(scope="module")
def model(tokenizer):
    cfg = LLMConfig(vocab_size=tokenizer.vocab_size, context_len=32, n_layers=2,
                    n_heads=2, d_model=32, d_ff=64, dropout=0.0)
    return MiniGPT(cfg).eval()


def test_greedy_decode_returns_text(model, tokenizer):
    out = generate(model, tokenizer, prompt="the cat", max_new_tokens=8,
                   greedy=True, stop_at_eos=False)
    assert isinstance(out, str)
    assert out.startswith("the cat")
    # 8 new tokens on top of the prompt, so it can only have grown
    assert len(out) > len("the cat")


def test_greedy_decode_is_deterministic(model, tokenizer):
    kwargs = dict(prompt="a dog", max_new_tokens=6, greedy=True, stop_at_eos=False)
    assert generate(model, tokenizer, **kwargs) == generate(model, tokenizer, **kwargs)


def test_decode_length_in_tokens(model, tokenizer):
    prompt = "the cat"
    n_prompt = len(tokenizer.encode(prompt))
    out = generate(model, tokenizer, prompt=prompt, max_new_tokens=5,
                   greedy=True, stop_at_eos=False)
    assert len(tokenizer.encode(out)) >= n_prompt


def test_empty_prompt_still_generates(model, tokenizer):
    out = generate(model, tokenizer, prompt="", max_new_tokens=4,
                   greedy=True, stop_at_eos=False)
    assert isinstance(out, str)


def test_sampling_respects_context_crop(model, tokenizer):
    # prompt longer than context_len must not blow up the rope tables
    long_prompt = " ".join(["the cat runs quickly"] * 20)
    out = generate(model, tokenizer, prompt=long_prompt, max_new_tokens=3,
                   greedy=True, stop_at_eos=False)
    assert isinstance(out, str) and out


def test_top_k_keeps_k_finite_logits():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    filtered = apply_top_k(logits, 2)
    assert torch.isfinite(filtered).sum().item() == 2
    assert filtered[0, 1].item() == 5.0 and filtered[0, 4].item() == 4.0


def test_top_p_keeps_the_nucleus():
    logits = torch.tensor([[10.0, 0.0, -10.0]])
    filtered = apply_top_p(logits, 0.9)
    assert torch.isfinite(filtered[0, 0])
    assert filtered[0, 2].item() == float("-inf")
