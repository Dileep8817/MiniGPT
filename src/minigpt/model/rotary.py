
import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    # rope: precompute cos/sin, rotate q,k in pairs. zero learnable params
    def __init__(self, d_head: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        assert d_head % 2 == 0, "RoPE requires an even d_head"
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)                # (T, d_head/2)
        emb = torch.cat([freqs, freqs], dim=-1)         # (T, d_head)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        # q, k: (B, n_heads, T, d_head)
        T = q.size(-2)
        cos = self.cos[:T].view(1, 1, T, -1).to(q.dtype)
        sin = self.sin[:T].view(1, 1, T, -1).to(q.dtype)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot
