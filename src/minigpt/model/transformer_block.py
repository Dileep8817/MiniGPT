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

class TransformerBlock(nn.Module):
    """
    Pre-norm GPT transformer block.

    Combines:
        • causal multi-head self-attention
        • position-wise feed-forward network

    with residual connections and LayerNorm.
    """
    
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            context_len: int,
            d_ff: int | None = None,
            dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff or 4*d_model

        self.ln1 = nn.LayerNorm(d_model) # pre-norm before attention
        self.attn = CausalMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            context_len=context_len,
            dropout=dropout
        ) # multi-head self-attention with causal mask

        self.ln2 = nn.LayerNorm(d_model) # seperate normalization before FFN
        self.ff = FeedForward(
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
