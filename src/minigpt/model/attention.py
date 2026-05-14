# Creates a GPT-style causal multi-head self-attention.
# Each token attends to all previous tokens (including itself)
# but cannot see future positions due to the causal mask.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with a causal mask (GPT-style).
    
    Input:  (B, T, d_model)
    Output: (B, T, d_model)
    """
    def __init__(
            self, 
            d_model: int, 
            n_heads: int, 
            context_len: int, 
            dropout: float = 0.1
            ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.normal(proj.weight, mean=0.0, std=0.02)
            nn.init.zeros(proj.bias)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(
            torch.ones(context_len, context_len, dtype=torch.bool)
            )
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, T, d_model)
        """
        B, T, _ = x.shape

        assert T <= self.causal_mask.size(0)

        Q = self.q_proj(x)  # (B, T, d_model)
        K = self.k_proj(x)  # (B, T, d_model)
        V = self.v_proj(x)  # (B, T, d_model)

        h = self.n_heads
        d_h = self.d_head
        Q = Q.view(B, T, h, d_h).transpose(1, 2) # (B, n_heads, T, d_head)
        K = K.view(B, T, h, d_h).transpose(1, 2) # (B, n_heads, T, d_head)
        V = V.view(B, T, h, d_h).transpose(1, 2) # (B, n_heads, T, d_head)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head) # (B, n_heads, T, T)
        scores = scores.masked_fill(
            ~self.causal_mask[:T, :T], 
            torch.finfo(scores.dtype).min
        )

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out