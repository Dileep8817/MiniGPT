
import torch
import torch.nn as nn
from minigpt.embeddings.positional_encoding import (
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding
)
from minigpt.embeddings.token_embedding import TokenEmbedding

class GPTEmbedding(nn.Module):
    # token embedding + optional positional encoding + dropout -> (B, T, d_model)
    def __init__(self, vocab_size: int, context_len: int, d_model: int, 
                 dropout: float, pos_type: str = "learned"):
        super().__init__()
        self.tok = TokenEmbedding(vocab_size, d_model)
        if pos_type == "rope":
            # rope is applied to q,k inside attention, nothing to add here
            self.pos = None
        elif pos_type == "learned":
            self.pos = LearnedPositionalEncoding(context_len, d_model)
        elif pos_type == "sinusoidal":
            self.pos = SinusoidalPositionalEncoding(context_len, d_model)
        else:
            raise ValueError(
                f"Unknown pos_type '{pos_type}'. Use 'rope', 'learned', or 'sinusoidal'."
            )
        self.drop = nn.Dropout(dropout)
        self._d_model = d_model
        self._vocab_size = vocab_size
    
    @classmethod
    def from_config(cls, cfg) -> "GPTEmbedding":
        return cls(
            vocab_size=cfg.vocab_size,
            context_len=cfg.context_len,
            d_model=cfg.d_model,
            dropout=cfg.dropout,
            pos_type=cfg.pos_type
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: (B, T) -> (B, T, d_model); pos(T) is (T, d_model) and broadcasts
        _, T = ids.shape
        tok = self.tok(ids)
        if self.pos is not None:
            tok = tok + self.pos(T)
        return self.drop(tok)

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        pos_cls = type(self.pos).__name__
        return (f"GPTEmbedding(vocab={self._vocab_size}, "
                f"d_model={self._d_model}, pos={pos_cls}, params={n:,})")
