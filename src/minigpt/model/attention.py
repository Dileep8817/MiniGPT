import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalMultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention with:
      * RoPE applied to Q and K (passed in by the parent block)
      * F.scaled_dot_product_attention (uses MPS/CUDA fast kernels)
      * No bias on projections (Llama convention)
      * Internal attention dropout via SDPA's dropout_p (training only)
      * Residual dropout after out_proj

    Input:  (B, T, d_model)
    Output: (B, T, d_model)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rotary,                     # shared RotaryEmbedding instance
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = dropout
        self.rotary = rotary

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        nn.init.normal_(self.q_proj.weight,   mean=0.0, std=0.02)
        nn.init.normal_(self.k_proj.weight,   mean=0.0, std=0.02)
        nn.init.normal_(self.v_proj.weight,   mean=0.0, std=0.02)
        # out_proj gets the GPT-2 scaled init applied later in MiniGPT.
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)

        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h, d_h = self.n_heads, self.d_head

        Q = self.q_proj(x).view(B, T, h, d_h).transpose(1, 2)   # (B, h, T, d_h)
        K = self.k_proj(x).view(B, T, h, d_h).transpose(1, 2)
        V = self.v_proj(x).view(B, T, h, d_h).transpose(1, 2)

        Q, K = self.rotary(Q, K)

        out = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )                                                       # (B, h, T, d_h)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)
        return self.resid_dropout(out)