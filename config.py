# ─────────────────────────────────────────────────────────────────────────────
# Central configuration for the Math LLM.
# All hyperparameters and special-token definitions live here.
# Every other module imports from this file — never hardcode values elsewhere.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field

SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "[PROBLEM]",
    "[SOLUTION]",
    "[STEP]",
    "[ANSWER]",
]

@dataclass
class MathLLMConfig:
    # ──── Tokenizer ──────────────────────────
    vocab_size: int = 512   # grows after tokenizer training; updated in main.py
    digit_by_digit: bool = True   # always tokenize digits individually (critical for math)
    raw_data_path: str = "data/corpus.txt"

# ── Default instance ──────────────────────────────────────────────────────────
cfg = MathLLMConfig()
