
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    # gated ffn: silu(w_gate x) * (w_up x) -> w_down. 3 matrices, so callers
    # pick d_ff ~= 8/3 * d_model to stay near a 4x GELU ffn's param count
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
    
# alias kept so transformer_block.py imports the same name as before
FeedForward = SwiGLU
