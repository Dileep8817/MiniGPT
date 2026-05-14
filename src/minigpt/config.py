# ─────────────────────────────────────────────────────────────────────────────
# Central configuration — general-purpose LLM.
# Every other module imports from here. Never hardcode values elsewhere.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field

# ── Special tokens ────────────────────────────────────────────────────────────
# Kept minimal for a general LLM. These always occupy the lowest IDs so
# their positions are fixed regardless of what BPE learns.
SPECIAL_TOKENS = [
    "<PAD>",    # 0 — padding (fills short sequences in a batch)
    "<UNK>",    # 1 — unknown token (char not in vocab)
    "<BOS>",    # 2 — beginning of sequence
    "<EOS>",    # 3 — end of sequence
]

@dataclass
class LLMConfig:
    # ──── Tokenizer ──────────────────────────
    vocab_size: int = 8000   # target BPE vocab size (updated after training)
    bpe_num_merges: int = 1000    # how many BPE merge rules to learn
    lowercase: bool = False # normalize to lowercase before tokenizing

    # ── Architecture ──────────────────────────────────────────────────────────
    context_len: int   = 512     # max tokens the model sees at once
    n_layers: int   = 6       # number of stacked transformer blocks
    n_heads: int   = 8       # attention heads (d_model must be divisible)
    d_model: int   = 512     # embedding dimension
    d_ff:   int   = 2048    # feed-forward hidden size (4 × d_model)
    dropout: float = 0.1
    pos_type: str = "learned"

    # ── Paths ─────────────────────────────────────────────────────────────────
    raw_data_path:   str   = "data/corpus.txt"
    tokenizer_path:  str   = "data/tokenizer.json"
    checkpoint_dir:  str   = "checkpoints/"

# ── Shared default instance ───────────────────────────────────────────────────
cfg = LLMConfig()


