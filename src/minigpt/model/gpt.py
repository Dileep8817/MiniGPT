# MINI GPT
#
# Stitches everything together into a working autoregressive language model.
#
#   ids (B, T)
#       │
#   GPTEmbedding (token + positional + dropout)
#       │
#   ┌───▼────┐
#   │ block  │  × n_layers   (pre-norm: LN → attn → +res, LN → ffn → +res)
#   └───┬────┘
#       │
#   final LayerNorm
#       │
#   linear head  (weight-tied to token embedding)
#       │
#       ▼
#   logits (B, T, vocab_size)
# ───────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn

from minigpt.config import LLMConfig
from minigpt.embeddings.embedding import GPTEmbedding
from minigpt.model.transformer_block import TransformerBlock
from minigpt.model.rotary import RotaryEmbedding
from minigpt.model.rms_norm import RMSNorm

class MiniGPT(nn.Module):
    """A small GPT-style decoder-only language model."""
    def __init__(self, cfg: LLMConfig):
        super().__init__()

        self.cfg = cfg
        self.embedding = GPTEmbedding.from_config(cfg)

        d_head = cfg.d_model // cfg.n_heads
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
        
        # Weight tying: the input embedding matrix and the output unembedding
        # matrix are the SAME Parameter. Counts once in num_params() and trains
        # in lockstep.
        self.head.weight = self.embedding.tok.table.weight

        self._apply_scaled_init()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(ids)  # (B, T) -> (B, T, d_model)

        for block in self.blocks:
            x = block(x)  # (B, T, d_model) -> (B, T, d_model)

        x = self.ln_f(x)  # final layer norm
        logits = self.head(x)  # (B, T, d_model) -> (B, T, vocab_size)
        return logits
    
    def _apply_scaled_init(self):
        """
        GPT-2 § 2.3: rescale the std of every projection that writes into the
        residual stream (attn out_proj and SwiGLU down_proj) by 1/sqrt(2*n_layers).
        Keeps residual activations from blowing up as depth grows.
        """
        import math
        scaled_std = 0.02 / math.sqrt(2 * self.cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=scaled_std)
            nn.init.normal_(block.ff.w_down.weight,     mean=0.0, std=scaled_std)
        
    
    def num_params(self) -> int:
        """
        Total trainable parameter count.
        head.weight is tied to embedding.tok.table.weight (same Parameter
        object), and PyTorch's .parameters() iterator deduplicates by ID,
        so the tied weight is counted exactly once — no manual subtraction
        needed. We still print the would-be saving for visibility.
        """
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
    
    model = MiniGPT(cfg)
    print(model)
    model.num_params()
    
    # Forward pass shape
    ids = torch.randint(0, cfg.vocab_size, (4, 32))
    logits = model(ids)
    assert logits.shape == (4, 32, cfg.vocab_size), f"shape: {logits.shape}"
    
    # Variable T
    for T in [1, 16, cfg.context_len]:
        ids = torch.randint(0, cfg.vocab_size, (2, T))
        assert model(ids).shape == (2, T, cfg.vocab_size)
    
    # Weight tying — must be the SAME tensor object, not just equal values
    assert model.head.weight is model.embedding.tok.table.weight, \
        "Weight tying failed — assignment didn't share the underlying Parameter"
    
    # Sanity check: dedup in .parameters() is actually happening
    n_dedup = sum(p.numel() for p in model.parameters())
    n_naive = (sum(p.numel() for p in model.embedding.parameters())
               + sum(sum(p.numel() for p in b.parameters()) for b in model.blocks)
               + sum(p.numel() for p in model.ln_f.parameters())
               + sum(p.numel() for p in model.head.parameters()))
    diff = n_naive - n_dedup
    expected_diff = cfg.vocab_size * cfg.d_model
    assert diff == expected_diff, \
        f"dedup mismatch: naive - dedup = {diff:,}, expected {expected_diff:,}"
    
    # Gradient flow — verify both the tied weight and a deep block param get gradients
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(ids)
    loss = logits.sum()
    loss.backward()
    assert model.embedding.tok.table.weight.grad is not None
    assert model.blocks[0].attn.q_proj.weight.grad is not None
    
    # Causality — most important test for an autoregressive LM
    model.eval()
    ids1 = torch.randint(0, cfg.vocab_size, (1, 16))
    ids2 = ids1.clone()
    ids2[0, -1] = (ids1[0, -1] + 1) % cfg.vocab_size  # change only the LAST token
    with torch.no_grad():
        l1 = model(ids1)
        l2 = model(ids2)
    # All earlier positions' logits must be IDENTICAL — they can't see the future
    assert torch.allclose(l1[:, :-1, :], l2[:, :-1, :], atol=1e-5), \
        "CAUSALITY BROKEN: changing the last token affected earlier logits"
    # And the last position SHOULD differ (otherwise the model is ignoring input)
    assert not torch.allclose(l1[:, -1, :], l2[:, -1, :])
    
    # Verify model can compute loss against targets (sanity for training)
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, cfg.vocab_size),
        targets.reshape(-1),
    )
    assert loss.dim() == 0 and loss.item() > 0
    
    print("All MiniGPT tests passed.")