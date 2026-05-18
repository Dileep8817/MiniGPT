# PART 1 — TOKENIZER 

# Responsibility: define the MathTokenizer class — the object that converts
# raw text strings into integer IDs and back again.

# Implements the full 5-step tokenization pipeline:
#
#   1. NORMALIZATION   — unicode cleanup, optional lowercase, whitespace
#   2. PRE-TOKENIZATION — split into "words" on whitespace + punctuation
#   3. BASE TOKENS     — split each word into individual characters
#   4. BPE MERGES      — apply learned merge rules in order
#   5. IDs             — map token strings → integer IDs
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import re 
import json
import unicodedata
from collections import Counter
from pathlib import Path
from minigpt.config import SPECIAL_TOKENS

# ── Base vocabulary components ────────────────────────────────────────────────
 
# Common punctuation and operators — always single tokens, never merged
BASE_PUNCTUATION = list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

EOW = "</w>"

def has_eow(token: str) -> bool:
    return token.endswith(EOW)

def strip_eow(token: str) ->str:
    return token[:-len(EOW)] if token.endswith(EOW) else token

def is_digit_token(token: str) -> bool:
    return strip_eow(token).isdigit()

# GPT-2 style pre-tokenization pattern:
# Handles contractions ('s, 't, 're …), words, punctuation, whitespace
PRETOKENIZE_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d"  # contractions
    r"| ?\w+"                     # optional-space + word chars
    r"| ?[^\s\w]+"                # optional-space + non-word chars (punctuation)
    r"|\s+(?!\S)"                 # trailing whitespace
    r"|\s+"                       # any remaining whitespace
)

class Tokenizer:
    """
    General-purpose BPE tokenizer.
 
    Pipeline (called in order inside encode()):
        normalize() → pre_tokenize() → to_base_chars() → apply_bpe() → tokens_to_ids()
 
    Train with train_bpe() on a corpus, then save() to persist.
    Reload with load() — no need to retrain.
    """

    def __init__(self, lowercase: bool = False):
        self.lowercase: bool = lowercase

        self.stoi: dict[str, int] = {} # token string -> int ID
        self.itos: dict[int, str] = {} # int ID -> token string
        self.vocab_size: int = 0

        self.bpe_merges: list[tuple[str, str]] = []

        self._multi_char: list[str] = []

    
    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — NORMALIZATION
    # ══════════════════════════════════════════════════════════════════════════

    def normalize(self, text: str) -> str:
        """
        Clean raw text before anything else:
          • NFC unicode normalization (collapses equivalent unicode forms)
          • Optional lowercase
          • Collapse multiple whitespace into a single space
          • Strip leading/trailing whitespace
        """
        # NFC: "é" represented as one codepoint rather than e + combining accent
        text = unicodedata.normalize("NFC", text)
        if self.lowercase:
            text = text.lower()
        
        # Collapse runs of whitespace (spaces, tabs, newlines) → single space
        text = re.sub(r"\s+", " ", text).strip()
 
        return text
    
    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — PRE-TOKENIZATION
    # ══════════════════════════════════════════════════════════════════════════
    def pre_tokenize(self, text: str) -> list[str]:
        """
        Split normalized text into a list of "pre-tokens" (rough words).
 
        Uses the GPT-2 regex that keeps:
          • Contractions together ("don't" → ["don", "'t"])
          • Punctuation as separate tokens
          • Leading spaces attached to the word that follows them
            (this is how GPT-2 distinguishes " dog" from "dog")
 
        Each pre-token is a string that will be further split into characters
        in step 3. Pre-tokens NEVER span multiple words.
        """
        return re.findall(PRETOKENIZE_PATTERN, text)
    
    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — SPLIT INTO BASE CHARACTERS
    # ══════════════════════════════════════════════════════════════════════════
    def to_base_chars(self, pre_tokens: list[str]) -> list[str]:
        """
        Split each pre-token into characters and mark the end of each pre-token
        with </w>. BPE training should not merge across tokens ending in </w>.
        """
        chars: list[str] = []
        for token in pre_tokens:
            if not token:
                continue
        
            chars_list = list(token)
            chars_list[-1] = chars_list[-1] + EOW
            chars.extend(chars_list)

        return chars
    
    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — APPLY BPE MERGE RULES
    # ══════════════════════════════════════════════════════════════════════════
    def apply_bpe(self, tokens: list[str]) -> list[str]:
        """
        Apply all learned BPE merge rules to a flat token list, in the order
        they were learned (most frequent pair first).
 
        Each merge rule (a, b) replaces every adjacent occurrence of a then b
        with the merged token ab.  Rules are applied sequentially — earlier
        merges produce tokens that later merges can use.
 
        This is O(n_merges × n_tokens) — fine for inference, slow for large
        batch training (use train_bpe's internal merge loop for that).
        """
        if not self.bpe_merges:
            return tokens
        
        tokens = list(tokens) # copy so we don't mutate the input

        for (a, b) in self.bpe_merges:
            merged = a + b
            i = 0
            while i <len(tokens) - 1:
                if tokens[i] == a and tokens[i+1] == b:
                    tokens[i] = merged
                    del tokens[i+1]
                    # Don't move i back — re-check from same position
                    # (the newly merged token could form a new pair with i-1,
                    # but BPE rules are applied left-to-right greedily)
                else:
                    i += 1
        return tokens
    
    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — CONVERT TOKENS TO IDs
    # ══════════════════════════════════════════════════════════════════════════
    def tokens_to_id(self, tokens: list[str]) -> list[int]:
        """
        Map each token string to its integer ID.
        Unknown tokens → <UNK> (ID 1).
        """
        unk_id = self.stoi.get("<UNK>", 1)
        return [self.stoi.get(t, unk_id) for t in tokens]
    
    # ── Full encode/decode ──────
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """
        Run the full 5-step pipeline on a string and return integer IDs.
 
        add_bos / add_eos: prepend/append boundary tokens (use for full documents)
        """
        # Step 1: Normalize
        text = self.normalize(text)
        # Step 2: Pre-tokenize
        pre_tokens = self.pre_tokenize(text)
        # Step 3: Base Characters
        base = self.to_base_chars(pre_tokens)
        # step 4: BPE merges
        bpe = self.apply_bpe(base)
        # step 5: IDs
        ids = self.tokens_to_id(bpe)

        if add_bos:
            ids = [self.stoi["<BOS>"]] + ids
        if add_eos:
            ids = ids + [self.stoi["<EOS>"]]

        return ids
    
    def decode(self, ids: list[int], skip_special: bool = False) -> str:
        """
        Convert integer IDs back to a text string.
        Strips the </w> end-of-word markers and reconstructs spaces.
        """
        special_set = set(SPECIAL_TOKENS)
        parts = []

        for i in ids:
            tok = self.itos.get(i, "<UNK>")
            if skip_special and tok in special_set:
                continue
            parts.append(tok)

        text = "".join(parts)
        text = text.replace("</w>", "")
        return text.strip()
    
    # ── Vocabulary construction ───────────
    def _base_vocab(self) -> list[str]:
        """Tokens always present regardless of corpus."""
        vocab: list[str] = []

        vocab.extend(SPECIAL_TOKENS)
        vocab.extend([str(d) for d in range(10)])
        vocab.extend([chr(c) for c in range(ord("A"), ord("Z") + 1)])
        vocab.extend([chr(c) for c in range(ord("a"), ord("z") + 1)])
        vocab.extend(BASE_PUNCTUATION)
        vocab.extend([" ", "\n", "\t"])

        # End-of-pre-token variants.
        for c in range(ord("a"), ord("z") + 1):
            vocab.append(chr(c) + EOW)
        for c in range(ord("A"), ord("Z") + 1):
            vocab.append(chr(c) + EOW)
        for d in range(10):
            vocab.append(str(d) + EOW)
        for p in BASE_PUNCTUATION:
            vocab.append(p + EOW)

        return vocab

    def build_vocab(
        self,
        extra_tokens: list[str] | None = None,
        corpus: list[str] | None = None
    ) -> None:
        """
        Construct stoi / itos from: base vocab + corpus extras + BPE merge tokens.
        Called once after BPE training is complete.
        """
        all_tokens: list[str] = []
        seen: set[str] = set()

        def add(tok: str) -> None:
            if tok not in seen:
                seen.add(tok)
                all_tokens.append(tok)
        
        for t in self._base_vocab():
            add(t)
        
        if corpus:
            for doc in corpus:
                for token in self.to_base_chars(self.pre_tokenize(self.normalize(doc))):
                    add(token)
        
        if extra_tokens:
            for t in extra_tokens:
                add(t)

        # Add every token produced by BPE merges
        for (a, b) in self.bpe_merges:
            add(a+b)

        self.stoi = {tok: idx for idx, tok in enumerate(all_tokens)}
        self.itos = {idx: tok for idx, tok in enumerate(all_tokens)}
        self.vocab_size = len(all_tokens)
        self._multi_char = sorted(
            [t for t in self.stoi if len(t) > 1],
            key=len, reverse=True
        )
 
        print(f"  [tokenizer] Vocabulary built — {self.vocab_size:,} tokens")

    # ── BPE Training  ────────
    def train_bpe(self, corpus: list[str], num_merges: int = 7000) -> None:
        """
        Fast BPE training: word-frequency aggregation + incremental pair counter
        + reverse index from pair → words containing it.

        Same merge rules, same blocking constraints as the original, but
        O(unique_words × avg_merges_per_word) instead of O(corpus × num_merges).
        """
        print(f"  [tokenizer] Training BPE ({num_merges} merges, fast)...")

        # ── 1. Aggregate pre-tokens into a word-frequency dict ────────────────
        # Each "word" = tuple of base char-tokens (last char carries </w>).
        word_counts: Counter[tuple[str, ...]] = Counter()
        for doc in corpus:
            norm = self.normalize(doc)
            for token in self.pre_tokenize(norm):
                if not token:
                    continue
                chars = list(token)
                chars[-1] = chars[-1] + EOW
                word_counts[tuple(chars)] += 1

        # words[i] is a *mutable* list of tokens for unique word i; counts[i] = freq
        words: list[list[str]] = [list(w) for w in word_counts]
        counts: list[int] = list(word_counts.values())
        print(f"  [tokenizer] {len(words):,} unique pre-tokens "
            f"(from {sum(counts):,} total)")

        # ── 2. Helper: is this pair forbidden? ─────────────────────────────────
        def blocked(a: str, b: str) -> bool:
            return (
                a in SPECIAL_TOKENS or b in SPECIAL_TOKENS
                or is_digit_token(a) or is_digit_token(b)
                or a == " " or b == " "
                or has_eow(a)
            )

        # ── 3. Initial pair counts + reverse index pair → {word indices} ──────
        pair_counts: Counter = Counter()
        pair_words: dict[tuple[str, str], set[int]] = {}
        for wi, w in enumerate(words):
            c = counts[wi]
            for i in range(len(w) - 1):
                p = (w[i], w[i + 1])
                if blocked(*p):
                    continue
                pair_counts[p] += c
                pair_words.setdefault(p, set()).add(wi)

        # ── 4. Merge loop ──────────────────────────────────────────────────────
        for merge_idx in range(num_merges):
            if not pair_counts:
                print(f"  [tokenizer] No more pairs to merge at step {merge_idx}.")
                break

            # Pick the highest-frequency unblocked pair. Counter has no
            # max-key shortcut, but max() over .items() is O(n_unique_pairs)
            # which is way smaller than the corpus.
            best_pair, best_freq = max(pair_counts.items(), key=lambda kv: kv[1])
            if best_freq <= 0:
                break

            self.bpe_merges.append(best_pair)
            merged = best_pair[0] + best_pair[1]
            a, b = best_pair

            # Only the words containing best_pair are affected
            affected = pair_words.pop(best_pair, set())
            pair_counts.pop(best_pair, None)

            for wi in affected:
                w = words[wi]
                c = counts[wi]

                # Old pair multiset in this word (excluding blocked, excluding best_pair itself)
                old_pairs = Counter()
                for i in range(len(w) - 1):
                    p = (w[i], w[i + 1])
                    if blocked(*p) or p == best_pair:
                        continue
                    old_pairs[p] += 1

                # Apply the merge greedily, left-to-right
                new_w: list[str] = []
                i = 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                        new_w.append(merged)
                        i += 2
                    else:
                        new_w.append(w[i])
                        i += 1
                words[wi] = new_w

                # New pair multiset
                new_pairs = Counter()
                for i in range(len(new_w) - 1):
                    p = (new_w[i], new_w[i + 1])
                    if blocked(*p):
                        continue
                    new_pairs[p] += 1

                # Incremental update of pair_counts and pair_words
                for p in old_pairs.keys() - new_pairs.keys():
                    if p in pair_words:
                        pair_words[p].discard(wi)
                        if not pair_words[p]:
                            pair_words.pop(p, None)
                for p in new_pairs.keys() - old_pairs.keys():
                    pair_words.setdefault(p, set()).add(wi)

                delta = Counter(new_pairs)
                delta.subtract(old_pairs)
                for p, d in delta.items():
                    if d == 0:
                        continue
                    pair_counts[p] += c * d
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                        pair_words.pop(p, None)

            if merge_idx % 500 == 0 or merge_idx == num_merges - 1:
                print(f"    merge {merge_idx:>5}: {best_pair[0]!r} + {best_pair[1]!r}"
                    f" → {merged!r}   "
                    f"(freq={best_freq:,}, n_pairs={len(pair_counts):,})")

        print(f"  [tokenizer] BPE training done. {len(self.bpe_merges):,} merges learned.")

    # ── Utility helpers ───────────────────────────────────────────────────────

    def batch_encode(self, texts: list[str], max_len: int | None = None,
                     pad: bool = True) -> list[list[int]]:
        encoded = [self.encode(t) for t in texts]
        if max_len is not None:
            encoded = [ids[:max_len] for ids in encoded]
            if pad:
                pad_id = self.stoi.get("<PAD>", 0)
                encoded = [
                    ids + [pad_id] * (max_len - len(ids))
                    for ids in encoded
                ]
        return encoded
 
    def token_to_id(self, token: str) -> int:
        return self.stoi.get(token, self.stoi.get("<UNK>", 1))
 
    def id_to_token(self, idx: int) -> str:
        return self.itos.get(idx, "<UNK>")
    
    # ── Persistence ───────────────────────────────────────────────────────────
 
    def save(self, path: str) -> None:
        import os
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "stoi":       self.stoi,
                "vocab_size": self.vocab_size,
                "lowercase":  self.lowercase,
                "bpe_merges": self.bpe_merges,   # list of [a, b] pairs
            }, f, indent=2, ensure_ascii=False)
        print(f"  [tokenizer] Saved → {path}")
 
    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.stoi       = data["stoi"]
        self.itos       = {int(v): k for k, v in self.stoi.items()}
        self.vocab_size = data["vocab_size"]
        self.lowercase  = data.get("lowercase", False)
        self.bpe_merges = [tuple(pair) for pair in data.get("bpe_merges", [])]
        self._multi_char = sorted(
            [t for t in self.stoi if len(t) > 1],
            key=len, reverse=True
        )
        print(f"  [tokenizer] Loaded ({self.vocab_size:,} tokens) ← {path}")
    
    def __repr__(self) -> str:
        return (f"Tokenizer(vocab_size={self.vocab_size}, "
                f"bpe_merges={len(self.bpe_merges)}, lowercase={self.lowercase})")
 


    




