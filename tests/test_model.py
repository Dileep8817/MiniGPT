# file to test the embeddings of the model

import torch
from torch import embedding
from minigpt.embeddings.positional_encoding import LearnedPositionalEncoding, SinusoidalPositionalEncoding


def test_embeddings():
    pe = SinusoidalPositionalEncoding(context_len=512, d_model=512)
    out = pe(32)
    assert out.shape == (32, 512)

    n_trainable = sum(p.numel() for p in pe.parameters() if p.requires_grad)
    assert n_trainable == 0, f"sinusoidal PE should have 0 trainable params, got {n_trainable}"

    assert "pe" in dict(pe.named_buffers())

    pe1 = SinusoidalPositionalEncoding(512, 512)
    pe2 = SinusoidalPositionalEncoding(512, 512)
    assert torch.equal(pe1(32), pe2(32))

    row0 = pe(1)[0]
    assert torch.allclose(row0[0::2], torch.zeros(256))
    assert torch.allclose(row0[1::2], torch.ones(256))

    full = pe(512)
    assert full.min() >= -1.0 and full.max() <= 1.0

    full = pe(512)
    for i in range(10):
        for j in range(i+1, 10):
            assert not torch.allclose(full[i], full[j])
    
    from minigpt.embeddings.token_embedding import TokenEmbedding
    from minigpt.config import cfg

    tok = TokenEmbedding(cfg.vocab_size, cfg.d_model)
    pe_mod = SinusoidalPositionalEncoding(cfg.context_len, cfg.d_model)
    ids = torch.randint(0, cfg.vocab_size, (4,32))
    combined = tok(ids) + pe_mod(32)
    assert combined.shape == (4, 32, cfg.d_model)


