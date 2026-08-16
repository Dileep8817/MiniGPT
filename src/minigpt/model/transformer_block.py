from __future__ import annotations

import torch
import torch.nn as nn

from minigpt.model.attention import CausalMultiHeadAttention
from minigpt.model.feed_forward import FeedForward
from minigpt.model.rms_norm import RMSNorm

class TransformerBlock(nn.Module):
    # pre-norm, so shape is preserved and blocks stack cleanly
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int,
            rotary,
            dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            rotary=rotary,
            dropout=dropout
        )
        self.ln2 = RMSNorm(d_model)
        self.ff =FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3
        assert x.size(-1) == self.d_model

        # (B, T, d_model) in and out
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
