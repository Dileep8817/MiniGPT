import math

import torch
import torch.nn as nn

from minigpt.config import LLMConfig
from minigpt.embeddings.embedding import GPTEmbedding
from minigpt.model.transformer_block import TransformerBlock
from minigpt.model.rotary import RotaryEmbedding
from minigpt.model.rms_norm import RMSNorm

class MiniGPT(nn.Module):
    def __init__(self, cfg: LLMConfig):
        super().__init__()

        self.cfg = cfg
        self.embedding = GPTEmbedding.from_config(cfg)

        d_head = cfg.d_model // cfg.n_heads
        # one rotary table shared by every block
        self.rotary = RotaryEmbedding(d_head=d_head, max_seq_len=cfg.context_len)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                d_ff=cfg.d_ff,
                rotary = self.rotary,
                dropout=cfg.dropout
            ) 
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        
        # tie input embedding to output head — same Parameter, not a copy
        self.head.weight = self.embedding.tok.table.weight

        self._apply_scaled_init()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(ids)  # (B, T) -> (B, T, d_model)

        for block in self.blocks:
            x = block(x)  # (B, T, d_model) -> (B, T, d_model)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, d_model) -> (B, T, vocab_size)
        return logits
    
    def _apply_scaled_init(self):
        # gpt-2 §2.3: shrink the projections that write into the residual
        # stream by 1/sqrt(2*n_layers) so activations don't grow with depth
        scaled_std = 0.02 / math.sqrt(2 * self.cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=scaled_std)
            nn.init.normal_(block.ff.w_down.weight,     mean=0.0, std=scaled_std)
        
    
    def num_params(self) -> int:
        # .parameters() dedups by id, so the tied weight is counted once
        total = sum(p.numel() for p in self.parameters())
        n_tied = self.embedding.tok.table.weight.numel()
        print(f"MiniGPT: {total:,} parameters "
              f"(weight tying saved {n_tied:,})")
        return total
    
    def __repr__(self) -> str:
        c = self.cfg
        return (f"MiniGPT(layers={c.n_layers}, d_model={c.d_model}, "
                f"heads={c.n_heads}, vocab={c.vocab_size}, "
                f"context_len={c.context_len})")
    

if __name__ == "__main__":
    from minigpt.config import cfg

    # quick demo — the real assertions live in tests/test_model.py
    model = MiniGPT(cfg)
    print(model)
    model.num_params()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    print(f"logits: {tuple(model(ids).shape)}")
