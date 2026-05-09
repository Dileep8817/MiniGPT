# EMBEDDING LAYER  
#
# Converts integer token IDs → dense float vectors of size d_model.
# This is a lookup table: each row = the learned vector for one token.
#
# Input:  (batch, seq_len)          integer token IDs
# Output: (batch, seq_len, d_model) float vectors
# ─────────────────────────────────────────────────────────────────────────────

import math
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    """
    Learnable token embedding table — shape (vocab_size, d_model).
 
    The PAD token's row is always zeroed and never updated by gradients
    (padding_idx). Every other row is trained via backprop.
 
    Output is scaled by √d_model to keep embedding
    magnitudes consistent with positional encodings.
    """

    def __init__(self, vocab_size: int, d_model: int, pad_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.scale = math.sqrt(d_model)
 
        self.table = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.table.weight[pad_id].zero_()
 
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, T) → (B, T, d_model)
        return self.table(token_ids) * self.scale
 
    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"TokenEmbedding(vocab={self.vocab_size}, d_model={self.d_model}, params={n:,})"
 
 
if __name__ == "__main__":
    import sys; sys.path.insert(0, "..")
    from minigpt.config import cfg
    emb = TokenEmbedding(cfg.vocab_size, cfg.d_model)
    x   = torch.randint(0, cfg.vocab_size, (4, 32))
    out = emb(x)
    assert out.shape == (4, 32, cfg.d_model)
    print(emb)
    print(f"Input {x.shape} → Output {out.shape}  ✅")