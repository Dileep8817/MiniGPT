
import torch
import torch.nn as nn
from minigpt.embeddings.positional_encoding import (
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding
)
from minigpt.embeddings.token_embedding import TokenEmbedding

class GPTEmbedding(nn.Module):
    """Token + positional embedding + dropout, what gets fed into block 0."""
    def __init__(self, vocab_size, context_len, d_model, dropout, pos_type="learned"):
        super().__init__()
        self.tok = TokenEmbedding(vocab_size, d_model)
        if pos_type == "learned":
            self.pos = LearnedPositionalEncoding(context_len, d_model)
        elif pos_type == "sinusoidal":
            self.pos = SinusoidalPositionalEncoding(context_len, d_model)
        else:
            raise ValueError(f"Unknown pos_type {pos_type}")
        self.drop = nn.Dropout(dropout)

    def forward(self, ids):
        _, T = ids.shape    # batch size isn't needed
        return self.drop(self.tok(ids) + self.pos(T))
