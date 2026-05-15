# DATASET
#
# Wraps a tokenized corpus as a sliding-window dataset for next-token
# language modeling.
#
# The full corpus is tokenized ONCE on disk-load and held in memory as a
# single 1D LongTensor of token IDs. Each training example is a sliding
# window of length context_len; (x, y) are the input/target pair where
# y is x shifted right by one position.
# ────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset

from minigpt.config import LLMConfig
from minigpt.tokenizer.create_tokenizer import Tokenizer
from minigpt.tokenizer.train_tokenizer import load_trained_tokenizer

class CorpusDataset(Dataset):
    """
    Sliding-window dataset over a tokenized text corpus.
    __getitem__(i) returns (x, y) where:
        chunk = ids[i : i + context_len]
        x     = chunk[:-1]   # positions 0 .. T-1
        y     = chunk[1:]    # positions 1 .. T   (next-token targets)
    Both x and y have length context_len.
    """
    def __init__(
            self,
            corpus_path: str,
            tokenizer: Tokenizer,
            context_len: int,
            cache: bool = True
    ):
        super().__init__()

        self.corpus_path = corpus_path
        self.context_len = context_len

        cache_path = corpus_path + ".tokens.pt"
        cache_valid = (
            cache
            and os.path.exists(cache_path)
            and os.path.getmtime(cache_path) > os.path.getmtime(corpus_path)
        )
        if cache_valid:
            print(f"  [dataset] Loading cached tokens ← {cache_path}")
            self.ids = torch.load(cache_path)
        else:
            print(f"  [dataset] Encoding corpus: {corpus_path}")
            self.ids = self._encode_file(corpus_path, tokenizer)
            if cache:
                torch.save(self.ids, cache_path)
                print(f"  [dataset] Cached encoded tokens → {cache_path}")
        if len(self.ids) <= context_len:
            raise ValueError(
                f"Corpus has only {len(self.ids):,} tokens, "
                f"need more than context_len={context_len}"
            )
        print(
            f"  [dataset] {len(self.ids):,} total tokens, "
            f"{len(self):,} training windows of length {context_len}"
        )

    @staticmethod
    def _encode_file(path: str, tokenizer: Tokenizer) -> torch.Tensor:
        """
        Encode one document per line, with <EOS> between documents so the
        model learns boundaries. Streams line-by-line to keep memory low
        and provide progress feedback (BPE encoding is slow in pure Python).
        """
        ids: list[int] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start = 1):
                line = line.strip()
                if not line:
                    continue
                ids.extend(tokenizer.encode(line, add_eos=True))
                if line_no % 1000 == 0:
                    print(f"    encoded {line_no:>8,} lines | {len(ids):>10,} tokens")
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        # Total number of (x, y) pairs is total tokens minus context length
        return len(self.ids) - self.context_len

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Get a sliding window of tokens for input (x) and target (y)
        chunk = self.ids[i : i + self.context_len + 1]
        x = chunk[: -1]
        y = chunk[1 : ]
        return x, y
    
    @classmethod
    def from_config(
        cls,
        cfg: LLMConfig,
        tokenizer: Tokenizer | None = None
    ) -> "CorpusDataset":
        """Build dataset from cfg. Loads the trained tokenizer if not given."""
        if tokenizer is None:
            tokenizer = load_trained_tokenizer(cfg)   # also updates cfg.vocab_size
        return cls(
            corpus_path=cfg.raw_data_path,
            tokenizer=tokenizer,
            context_len=cfg.context_len,
        )
    def __repr__(self) -> str:
        return (
            f"CorpusDataset(tokens={len(self.ids):,}, "
            f"context_len={self.context_len}, "
            f"windows={len(self):,})"
        )

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from minigpt.config import cfg
    ds = CorpusDataset.from_config(cfg)
    print(ds)
    x, y = ds[0]
    T_expected = cfg.context_len
    assert x.shape == (T_expected,), f"x shape: {x.shape}"
    assert y.shape == (T_expected,), f"y shape: {y.shape}"
    assert torch.equal(y[:-1], x[1:]), "y must be x shifted by one"
    assert x.dtype == torch.long and y.dtype == torch.long, \
        "Token IDs must be int64 for nn.Embedding"
    assert len(ds) == len(ds.ids) - cfg.context_len
    assert ds.ids.min().item() >= 0
    assert ds.ids.max().item() < cfg.vocab_size, (
        f"Token ID {ds.ids.max().item()} >= vocab_size {cfg.vocab_size}. "
        "Did you reload the trained tokenizer to update cfg.vocab_size?"
    )
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    xb, yb = next(iter(loader))
    assert xb.shape == (4, T_expected)
    assert yb.shape == (4, T_expected)
    print(f"  [dataset] Batch OK: x={tuple(xb.shape)}, y={tuple(yb.shape)}")
    from minigpt.model.gpt import MiniGPT
    model = MiniGPT(cfg)
    logits = model(xb)
    assert logits.shape == (4, T_expected, cfg.vocab_size)
    print(f"  [dataset] Model forward on batch OK: logits={tuple(logits.shape)}")
    print("CorpusDataset: all tests passed.")


