# Responsibility: orchestrate the full tokenizer pipeline:
#   load corpus → train BPE → build vocab → analyze lengths → save
#
# This is the only file you need to run to produce a trained tokenizer.
# ─────────────────────────────────────────────────────────────────────────────

import os
from minigpt.tokenizer.create_tokenizer import Tokenizer

from minigpt.config import LLMConfig, cfg as default_cfg

def analyze_corpus(path: str) -> dict:
    """Read corpus and return basic statistics"""
    lines = []
    char_count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                lines.append(line)
                char_count += len(line)

    lengths = [len(line) for line in lines]
    return {
        "n_docs": len(lines),
        "total_chars": char_count,
        "avg_len": sum(lengths)/max(len(lengths), 1),
        "max_len": max(lengths) if lengths else 0,
        "lines": lines
    }

def estimate_token_length(
        lines: list[str], 
        tokenizer: Tokenizer,
        sample_size: int = 1000
        ) -> dict:
    """Estimate token-sequence lengths to recommend context_len."""
    sample = lines[:sample_size]
    lengths = sorted(len(tokenizer.encode(doc)) for doc in sample)
    n = len(lengths)
    if not n:
        return {}
    
    return {
        "min":  lengths[0],
        "p50":  lengths[n // 2],
        "p90":  lengths[int(n * 0.90)],
        "p99":  lengths[int(n * 0.99)],
        "max":  lengths[-1],
        "mean": sum(lengths) / n,
    }

def train_tokenizer(cfg: LLMConfig = default_cfg) -> Tokenizer:
    """
    Full Part 1 pipeline:
 
    1. Check corpus exists (run load_data.py first if not)
    2. Analyze corpus stats
    3. Train BPE on the corpus
    4. Build final vocabulary
    5. Estimate token sequence lengths → recommend context_len
    6. Save tokenizer to disk
    7. Update cfg.vocab_size in memory
    """
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Training Tokenizer")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
 
    # ── 1. Ensure corpus exists ────────────────────────────────────────────
    if not os.path.exists(cfg.raw_data_path):
        print(f"  Corpus not found at {cfg.raw_data_path}.")
        print("  Running load_data.py first...\n")
        from minigpt.tokenizer.load_data import load_corpus, save_corpus
        docs = load_corpus(cfg)
        save_corpus(docs, cfg.raw_data_path)

    # ── 2. Analyze corpus ──────────────────────────────────────────────────
    print(f"  Analyzing corpus: {cfg.raw_data_path}")
    stats = analyze_corpus(cfg.raw_data_path)

    max_chars = 5_000_000
    sampled_lines = []
    total = 0

    for line in stats["lines"]:
        if total >= max_chars:
            break
        sampled_lines.append(line)
        total += len(line)

    stats["lines"] = sampled_lines
    print(f"  Using {len(sampled_lines):,} docs / {total:,} chars for tokenizer training")

 
    print(f"  ├─ Documents:    {stats['n_docs']:>10,}")
    print(f"  ├─ Total chars:  {stats['total_chars']:>10,}")
    print(f"  ├─ Avg doc len:  {stats['avg_len']:>10.1f} chars")
    print(f"  └─ Max doc len:  {stats['max_len']:>10,} chars")

    # ── 3. Train BPE on corpus ──────────────────────────────────────────────────
    print()
    tokenizer = Tokenizer(lowercase=cfg.lowercase)
    tokenizer.train_bpe(stats["lines"], num_merges=cfg.bpe_num_merges)

    # ── 4. Build vocab ──────────────────────────────────────────────────
    tokenizer.build_vocab(corpus=stats["lines"])
    
    # ── 5. Estimate sequence lengths ──────────────────────────────────────────────────
    print("\n  Estimating token sequence lengths (sample of 1000 docs)...")
    lens = estimate_token_length(stats["lines"], tokenizer)
    if lens:
        print(f"  ├─ Min:    {lens['min']}")
        print(f"  ├─ Mean:   {lens['mean']:.1f}")
        print(f"  ├─ p50:    {lens['p50']}")
        print(f"  ├─ p90:    {lens['p90']}")
        print(f"  ├─ p99:    {lens['p99']}")
        print(f"  └─ Max:    {lens['max']}")
        recommended = min(lens["p90"] + 64, 2048)
        print(f"\n  ✅ Recommended context_len: {recommended}")
        print(f"     (current cfg.context_len: {cfg.context_len})")
    
    # ── 6. Save ────────────────────────────────────────────────────────────
    tokenizer.save(cfg.tokenizer_path)
 
    # ── 7. Update cfg ──────────────────────────────────────────────────────
    cfg.vocab_size = tokenizer.vocab_size
    print(f"\n  cfg.vocab_size updated → {cfg.vocab_size:,}")
    print("\n  ✅ Tokenizer training complete.\n")
 
    return tokenizer

def load_trained_tokenizer(cfg: LLMConfig = default_cfg) -> Tokenizer:
    """Load a previously trained tokenizer from disk."""
    if not os.path.exists(cfg.tokenizer_path):
        raise FileNotFoundError(
            f"No tokenizer at {cfg.tokenizer_path}. Run train_tokenizer() first."
        )
    tok = Tokenizer()
    tok.load(cfg.tokenizer_path)
    cfg.vocab_size = tok.vocab_size
    return tok

# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tokenizer = train_tokenizer(default_cfg)
 
    test = "The quick brown fox jumps over the lazy dog."
    ids  = tokenizer.encode(test)
    back = tokenizer.decode(ids)
    print(f"\nRound-trip test:")
    print(f"  Original: {test!r}")
    print(f"  Decoded:  {back!r}")
    print(f"  Tokens:   {len(ids)}")
 




