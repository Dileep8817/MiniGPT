# tests the MiniGPT stack: shapes, weight tying, gradients, causality

import pytest
import torch

from minigpt.config import LLMConfig
from minigpt.embeddings.positional_encoding import SinusoidalPositionalEncoding
from minigpt.model.gpt import MiniGPT

# small enough to run on CPU in a second, same code paths as the real config
TINY = dict(vocab_size=97, context_len=32, n_layers=3, n_heads=4,
            d_model=32, d_ff=64, dropout=0.0)


@pytest.fixture(scope="module")
def cfg():
    return LLMConfig(**TINY)


@pytest.fixture(scope="module")
def model(cfg):
    return MiniGPT(cfg)


@pytest.mark.parametrize("B,T", [(1, 1), (2, 7), (4, 16), (2, 32)])
def test_forward_shape(model, cfg, B, T):
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    assert model(ids).shape == (B, T, cfg.vocab_size)


def test_weight_tying_shares_one_parameter(model):
    assert model.head.weight is model.embedding.tok.table.weight


def test_parameters_dedup_the_tied_weight(model, cfg):
    n_dedup = sum(p.numel() for p in model.parameters())
    n_naive = (sum(p.numel() for p in model.embedding.parameters())
               + sum(sum(p.numel() for p in b.parameters()) for b in model.blocks)
               + sum(p.numel() for p in model.ln_f.parameters())
               + sum(p.numel() for p in model.head.parameters()))
    assert n_naive - n_dedup == cfg.vocab_size * cfg.d_model


def test_gradients_reach_the_last_block(cfg):
    model = MiniGPT(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    model(ids).sum().backward()

    assert model.embedding.tok.table.weight.grad is not None
    deep = model.blocks[-1].ff.w_down.weight
    assert deep.grad is not None
    assert deep.grad.abs().sum() > 0


def test_causality(model, cfg):
    # changing only the last token must not move any earlier position's logits
    model.eval()
    ids1 = torch.randint(0, cfg.vocab_size, (1, 16))
    ids2 = ids1.clone()
    ids2[0, -1] = (ids1[0, -1] + 1) % cfg.vocab_size
    with torch.no_grad():
        l1, l2 = model(ids1), model(ids2)

    assert torch.allclose(l1[:, :-1, :], l2[:, :-1, :], atol=1e-5)
    # ...and the last position must move, or the model is ignoring its input
    assert not torch.allclose(l1[:, -1, :], l2[:, -1, :])


def test_loss_is_finite_against_targets(model, cfg):
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.vocab_size), targets.reshape(-1)
    )
    assert loss.dim() == 0 and torch.isfinite(loss) and loss.item() > 0


def test_sinusoidal_positional_encoding():
    pe = SinusoidalPositionalEncoding(context_len=512, d_model=512)
    assert pe(32).shape == (32, 512)
    assert sum(p.numel() for p in pe.parameters() if p.requires_grad) == 0
    assert "pe" in dict(pe.named_buffers())

    row0 = pe(1)[0]
    assert torch.allclose(row0[0::2], torch.zeros(256))
    assert torch.allclose(row0[1::2], torch.ones(256))

    full = pe(512)
    assert full.min() >= -1.0 and full.max() <= 1.0
    for i in range(10):
        for j in range(i + 1, 10):
            assert not torch.allclose(full[i], full[j])
