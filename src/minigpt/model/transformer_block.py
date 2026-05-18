# TRANSFORMER BLOCK
#
# Core computation unit of a GPT-style transformer.
#
# Each block performs two operations:
#
#   1. Multi-head causal self-attention
#      → allows tokens to exchange information across the sequence
#
#   2. Position-wise feed-forward network (FFN)
#      → performs nonlinear feature transformation independently
#        on each token representation
#
# Both sublayers are wrapped in:
#
#   • LayerNorm (pre-norm configuration)
#   • Residual connections
#
# Mathematical form:
#
#   x = x + Attention(LayerNorm(x))
#   x = x + FFN(LayerNorm(x))
#
# Pre-norm transformers are significantly more stable to train at depth
# than the original post-norm formulation from "Attention Is All You Need".
#
# Input / output shape:
#
#   (B, T, d_model)
#
# Shape is preserved so blocks can be stacked repeatedly.

from __future__ import annotations

import torch
import torch.nn as nn

from minigpt.model.attention import CausalMultiHeadAttention
from minigpt.model.feed_forward import FeedForward
from minigpt.model.rms_norm import RMSNorm

class TransformerBlock(nn.Module):
    """
    Pre-norm GPT block (Llama-style):
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """
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
        self.ln1 = RMSNorm(d_model) # pre-norm before attention
        self.attn = CausalMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            rotary=rotary,
            dropout=dropout
        ) # multi-head self-attention with causal mask
        self.ln2 = RMSNorm(d_model)
        self.ff =FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3
        assert x.size(-1) == self.d_model

        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
