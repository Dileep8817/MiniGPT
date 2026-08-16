# tests the training-loop helpers that are easy to get wrong

import pytest
import torch

from minigpt.config import LLMConfig
from minigpt.model.gpt import MiniGPT
from minigpt.train import (build_param_groups, load_checkpoint, lr_at_step,
                           save_checkpoint, unwrap)

CFG = LLMConfig(vocab_size=64, context_len=16, n_layers=2, n_heads=2,
                d_model=32, d_ff=64, dropout=0.0)


@pytest.fixture
def model():
    return MiniGPT(CFG)


def test_lr_schedule_warms_up_then_decays():
    warmup, total, max_lr, min_lr = 10, 100, 3e-4, 3e-5
    assert lr_at_step(0, warmup, total, max_lr, min_lr) == pytest.approx(max_lr / warmup)
    assert lr_at_step(warmup - 1, warmup, total, max_lr, min_lr) == pytest.approx(max_lr)
    mid = lr_at_step(55, warmup, total, max_lr, min_lr)
    assert min_lr < mid < max_lr
    assert lr_at_step(total, warmup, total, max_lr, min_lr) == pytest.approx(min_lr)


def test_param_groups_exclude_norms_from_decay(model):
    decay, nodecay = build_param_groups(model, 0.1)
    assert decay["weight_decay"] == 0.1 and nodecay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in nodecay["params"])


def test_unwrap_returns_the_plain_module(model):
    assert unwrap(model) is model

    class FakeCompiled(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self._orig_mod = inner

    assert unwrap(FakeCompiled(model)) is model


def test_checkpoint_roundtrip_is_loadable_by_a_plain_model(tmp_path, model):
    opt = torch.optim.AdamW(build_param_groups(model, 0.1), lr=1e-3)
    path = tmp_path / "step_000001.pt"
    save_checkpoint(model, opt, 1, CFG, str(path))

    # generate.py rebuilds a bare MiniGPT, so the keys must not be prefixed
    fresh = MiniGPT(CFG)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    assert all(not k.startswith("_orig_mod.") for k in ckpt["model"])
    fresh.load_state_dict(ckpt["model"])

    step = load_checkpoint(fresh, opt, str(path), torch.device("cpu"))
    assert step == 1
    ids = torch.randint(0, CFG.vocab_size, (1, 8))
    model.eval(), fresh.eval()
    with torch.no_grad():
        assert torch.allclose(model(ids), fresh(ids))
