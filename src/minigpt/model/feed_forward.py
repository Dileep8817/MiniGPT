
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    Position-wise SwiGLU feed-forward (Llama-style).
    Replaces the GELU MLP with a gated linear unit:
        FFN(x) = (silu(W_gate x) * (W_up x)) @ W_down
    SwiGLU uses 3 matrices instead of GELU's 2. To keep total parameter
    count comparable to a 2x GELU FFN at d_ff = 4 * d_model, choose
    d_ff ~= (8/3) * d_model. For d_model = 512 the calling code uses
    d_ff = 1408 (multiple of 64, hardware-friendly).
    Biases are removed (Llama convention) — no quality cost, slightly
    faster, fewer parameters.
    """
    def __init__(self, d_model: int, d_ff: int, dropout:float = 0.1):
        super().__init__()

        self.d_ff = d_ff
        self.d_model = d_model
        
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.w_gate.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w_up.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w_down.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3
        assert x.size(-1) == self.d_model
        
        gated = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.dropout(self.w_down(gated))
    
# Backwards-compatible alias so transformer_block.py doesn't need to change
# its import name even though the implementation is now SwiGLU.
FeedForward = SwiGLU
