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
    

