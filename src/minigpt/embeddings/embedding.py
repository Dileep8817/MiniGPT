
import torch
import torch.nn as nn
from minigpt.embeddings.positional_encoding import (
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding
)
from minigpt.embeddings.token_embedding import TokenEmbedding

class GPTEmbedding(nn.Module):
    """
    Token embedding + positional encoding + dropout.
    This is the input to transformer block 0.

    Output shape: (B, T, d_model)
    """
    def __init__(self, vocab_size: int, context_len: int, d_model: int, 
                 dropout: float, pos_type: str = "learned"):
        super().__init__()
        self.tok = TokenEmbedding(vocab_size, d_model)
        if pos_type == "learned":
            self.pos = LearnedPositionalEncoding(context_len, d_model)
        elif pos_type == "sinusoidal":
            self.pos = SinusoidalPositionalEncoding(context_len, d_model)
        else:
            raise ValueError(f"Unknown pos_type '{pos_type}'. Use 'learned' or 'sinusoidal'.")
        self.drop = nn.Dropout(dropout)
        self._d_model = d_model
        self._vocab_size = vocab_size
    
    @classmethod
    def from_config(cls, cfg) -> "GPTEmbedding":
        """Construct directly from LLMConfig instance."""
        return cls(
            vocab_size=cfg.vocab_size,
            context_len=cfg.context_len,
            d_model=cfg.d_model,
            dropout=cfg.dropout,
            pos_type=cfg.pos_type
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: (B, T) — token IDs
        # tok(ids): (B, T, d_model)
        # pos(T):      (T, d_model) — broadcasts across batch
        _, T = ids.shape    # batch size isn't needed
        return self.drop(self.tok(ids) + self.pos(T))

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        pos_cls = type(self.pos).__name__
        return (f"GPTEmbedding(vocab={self._vocab_size}, "
                f"d_model={self._d_model}, pos={pos_cls}, params={n:,})")