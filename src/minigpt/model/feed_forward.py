# FEED-FORWARD LAYER
#
# Feed-forward network used inside each transformer block.
#
# After attention mixes information across tokens, the FFN performs
# nonlinear feature transformation independently on each token vector.
#
# This layer:
#   1. Expands the representation into a larger feature space
#   2. Applies a nonlinear activation (GELU)
#   3. Projects back to d_model
#
# No communication occurs across sequence positions here —
# token interaction happens only in the attention mechanism.
# ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    Position-wise transformer MLP.

    Expands token representations into a higher-dimensional feature
    space, applies nonlinear transformation, then projects back to
    d_model.

    Operates independently on each token position.
    """
    def __init__(self, d_model: int, d_ff: int | None = None, dropout:float = 0.1):
        super().__init__()

        self.d_ff = d_ff or 4*d_model
        self.d_model = d_model
        
        self.fc1 = nn.Linear(d_model, self.d_ff)
        self.gelu = nn.GELU()
        self.act_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.d_ff, d_model)

        for proj in (self.fc1, self.fc2):
            nn.init.normal_(proj.weight, mean = 0.0, std = 0.02)
            nn.init.zeros_(proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3
        assert x.size(-1) == self.d_model
        
        x = self.fc1(x)           # (B, T, d_model) -> (B, T, d_ff)
        x = self.gelu(x)
        x = self.act_dropout(x)
        x = self.fc2(x)           # (B, T, d_ff)    -> (B, T, d_model)
        x = self.out_dropout(x)
        return x
