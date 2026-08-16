# attention is permutation-invariant, so position has to be injected somewhere.
# these two variants back the v1 baseline; v2 uses rope inside attention instead.

import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    # one trainable vector per position 0 .. context_len-1 (gpt-2 style)
    def __init__(self, context_len: int, d_model: int):
        super().__init__()
        self.table = nn.Embedding(context_len, d_model)
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)

    def forward(self, seq_len: int) -> torch.Tensor:
        device = self.table.weight.device

        pos = torch.arange(seq_len, device=device)
        return self.table(pos) # (T, d_model)
    
class SinusoidalPositionalEncoding(nn.Module):
    # fixed sin/cos table from the original transformer paper, no parameters
    def __init__(self, context_len: int, d_model: int):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        position = torch.arange(0, context_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(context_len, d_model)
        # even columns get sin, odd columns get cos
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, seq_len: int):
        return self.pe[:seq_len]
