from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset

from minigpt.config import LLMConfig
from minigpt.tokenizer.create_tokenizer import Tokenizer
from minigpt.tokenizer.train_tokenizer import load_trained_tokenizer

class CorpusDataset(Dataset):
    # the whole corpus lives in memory as one 1D LongTensor of ids; each item
    # is a sliding window where y is x shifted right by one
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

        # cache is stale if the corpus was rewritten after it
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
        # one doc per line with <EOS> between them so boundaries are learnable.
        # streamed line by line because pure-python BPE encoding is slow
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
        return len(self.ids) - self.context_len

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.ids[i : i + self.context_len + 1]
        x = chunk[: -1]
        y = chunk[1 : ]
        return x, y
    
    @classmethod
    def from_ids(
        cls,
        ids: torch.Tensor,
        context_len: int,
        corpus_path: str = ""
    ) -> "CorpusDataset":
        # for slicing an already-encoded stream (train/val split) without
        # re-reading or re-encoding the corpus
        if len(ids) <= context_len:
            raise ValueError(
                f"Got {len(ids):,} tokens, need more than context_len={context_len}"
            )
        ds = cls.__new__(cls)
        ds.ids = ids
        ds.context_len = context_len
        ds.corpus_path = corpus_path
        return ds

    @classmethod
    def from_config(
        cls,
        cfg: LLMConfig,
        tokenizer: Tokenizer | None = None
    ) -> "CorpusDataset":
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
    from minigpt.config import cfg

    # quick demo — the real assertions live in tests/test_dataset.py
    ds = CorpusDataset.from_config(cfg)
    print(ds)
    x, y = ds[0]
    print(f"x: {tuple(x.shape)}  y: {tuple(y.shape)}")
