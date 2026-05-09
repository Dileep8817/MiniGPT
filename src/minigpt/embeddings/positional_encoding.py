# EMBEDDING LAYER 
#
# Self-attention is permutation-invariant — without position info,
# "dog bites man" = "man bites dog". Positional encoding fixes this
# by injecting a unique signal per position into the token vectors.
#
# Two implementations:
#   LEARNED      — each position has its own trainable vector (GPT-2 style)
#   SINUSOIDAL   — fixed sin/cos encoding from original Transformer paper
# ─────────────────────────────────────────────────────────────────────────────

import math
import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    """One trainable vector per position 0 … context_len-1."""
    
    def __init__(self, context_len: int, d_model: int):
        super().__init__()
        self.table = nn.Embedding(context_len, d_model)
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)

    def forward(self, seq_len: int) -> torch.Tensor:
        device = self.table.weight.device

        pos = torch.arange(seq_len, device=device)
        return self.table(pos) # (T, d_model)
    
class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sin/cos encoding — no trainable parameters.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, context_len: int, d_model: int):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        position = torch.arange(0, context_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(context_len, d_model)
        # fill even and odd columns separately; even columns with sin and odd columns with cos
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, seq_len: int):
        return self.pe[:seq_len]
    
if __name__ == "__main__":
    pe = SinusoidalPositionalEncoding(context_len=512, d_model=512)
    out = pe(32)
    assert out.shape == (32, 512)

    n_trainable = sum(p.numel() for p in pe.parameters() if p.requires_grad())
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
    combined = tok(ids) + pe(32)
    assert combined.shape == (4, 32, cfg.d_model)


