from __future__ import annotations
import os
import re 
import json
import unicodedata
from collections import Counter
from minigpt.config import SPECIAL_TOKENS

# punctuation and operators stay single tokens, never merged
BASE_PUNCTUATION = list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

EOW = "</w>"

def has_eow(token: str) -> bool:
    return token.endswith(EOW)

def strip_eow(token: str) ->str:
    return token[:-len(EOW)] if token.endswith(EOW) else token

def is_digit_token(token: str) -> bool:
    return strip_eow(token).isdigit()

# gpt-2 style pre-tokenizer: contractions, words, punctuation, whitespace
PRETOKENIZE_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d"  # contractions
    r"| ?\w+"                     # optional-space + word chars
    r"| ?[^\s\w]+"                # optional-space + non-word chars (punctuation)
    r"|\s+(?!\S)"                 # trailing whitespace
    r"|\s+"                       # any remaining whitespace
)

class Tokenizer:
    # encode() runs: normalize -> pre_tokenize -> to_base_chars -> apply_bpe -> ids
    def __init__(self, lowercase: bool = False):
        self.lowercase: bool = lowercase

        self.stoi: dict[str, int] = {} # token string -> int ID
        self.itos: dict[int, str] = {} # int ID -> token string
        self.vocab_size: int = 0

        self.bpe_merges: list[tuple[str, str]] = []

    def normalize(self, text: str) -> str:
        # NFC folds equivalent unicode forms, e.g. "é" as one codepoint
        text = unicodedata.normalize("NFC", text)
        if self.lowercase:
            text = text.lower()

        text = re.sub(r"\s+", " ", text).strip()

        return text

    def pre_tokenize(self, text: str) -> list[str]:
        # pre-tokens never span words; a leading space stays glued to its word
        # so " dog" and "dog" get different tokens
        return re.findall(PRETOKENIZE_PATTERN, text)

    def to_base_chars(self, pre_tokens: list[str]) -> list[str]:
        # </w> marks the last char of a pre-token so bpe can't merge across words
        chars: list[str] = []
        for token in pre_tokens:
            if not token:
                continue
        
            chars_list = list(token)
            chars_list[-1] = chars_list[-1] + EOW
            chars.extend(chars_list)

        return chars

    def apply_bpe(self, tokens: list[str]) -> list[str]:
        # merge rules applied in learned order, greedily left-to-right.
        # O(n_merges * n_tokens) — fine for inference, too slow for training
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
                else:
                    i += 1
        return tokens

    def tokens_to_id(self, tokens: list[str]) -> list[int]:
        unk_id = self.stoi.get("<UNK>", 1)
        return [self.stoi.get(t, unk_id) for t in tokens]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        text = self.normalize(text)
        pre_tokens = self.pre_tokenize(text)
        base = self.to_base_chars(pre_tokens)
        bpe = self.apply_bpe(base)
        ids = self.tokens_to_id(bpe)

        if add_bos:
            ids = [self.stoi["<BOS>"]] + ids
        if add_eos:
            ids = ids + [self.stoi["<EOS>"]]

        return ids

    def decode(self, ids: list[int], skip_special: bool = False) -> str:
        special_set = set(SPECIAL_TOKENS)
        parts = []

        for i in ids:
            tok = self.itos.get(i, "<UNK>")
            if skip_special and tok in special_set:
                continue
            parts.append(tok)

        # dropping </w> is what puts the spaces back
        text = "".join(parts)
        text = text.replace("</w>", "")
        return text.strip()

    def _base_vocab(self) -> list[str]:
        # tokens always present regardless of corpus
        vocab: list[str] = []

        vocab.extend(SPECIAL_TOKENS)
        vocab.extend([str(d) for d in range(10)])
        vocab.extend([chr(c) for c in range(ord("A"), ord("Z") + 1)])
        vocab.extend([chr(c) for c in range(ord("a"), ord("z") + 1)])
        vocab.extend(BASE_PUNCTUATION)
        vocab.extend([" ", "\n", "\t"])

        # end-of-pre-token variants
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
        # base vocab + corpus chars + every token bpe produced. call once,
        # after train_bpe
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

        for (a, b) in self.bpe_merges:
            add(a+b)

        self.stoi = {tok: idx for idx, tok in enumerate(all_tokens)}
        self.itos = {idx: tok for idx, tok in enumerate(all_tokens)}
        self.vocab_size = len(all_tokens)

        print(f"  [tokenizer] Vocabulary built — {self.vocab_size:,} tokens")

    def train_bpe(self, corpus: list[str], num_merges: int = 7000) -> None:
        # sennrich-style: word-frequency dict + incremental pair counts + a
        # reverse index pair -> words, so each merge only touches affected words
        print(f"  [tokenizer] Training BPE ({num_merges} merges, fast)...")

        # each "word" is a tuple of base char-tokens, last one carrying </w>
        word_counts: Counter[tuple[str, ...]] = Counter()
        for doc in corpus:
            norm = self.normalize(doc)
            for token in self.pre_tokenize(norm):
                if not token:
                    continue
                chars = list(token)
                chars[-1] = chars[-1] + EOW
                word_counts[tuple(chars)] += 1

        # words[i] is mutable, counts[i] is its frequency
        words: list[list[str]] = [list(w) for w in word_counts]
        counts: list[int] = list(word_counts.values())
        print(f"  [tokenizer] {len(words):,} unique pre-tokens "
            f"(from {sum(counts):,} total)")

        def blocked(a: str, b: str) -> bool:
            return (
                a in SPECIAL_TOKENS or b in SPECIAL_TOKENS
                or is_digit_token(a) or is_digit_token(b)
                or a == " " or b == " "
                or has_eow(a)
            )

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

        for merge_idx in range(num_merges):
            if not pair_counts:
                print(f"  [tokenizer] No more pairs to merge at step {merge_idx}.")
                break

            # max() over items is O(n_unique_pairs), far smaller than the corpus
            best_pair, best_freq = max(pair_counts.items(), key=lambda kv: kv[1])
            if best_freq <= 0:
                break

            self.bpe_merges.append(best_pair)
            merged = best_pair[0] + best_pair[1]
            a, b = best_pair

            affected = pair_words.pop(best_pair, set())
            pair_counts.pop(best_pair, None)

            for wi in affected:
                w = words[wi]
                c = counts[wi]

                old_pairs = Counter()
                for i in range(len(w) - 1):
                    p = (w[i], w[i + 1])
                    if blocked(*p) or p == best_pair:
                        continue
                    old_pairs[p] += 1

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

                new_pairs = Counter()
                for i in range(len(new_w) - 1):
                    p = (new_w[i], new_w[i + 1])
                    if blocked(*p):
                        continue
                    new_pairs[p] += 1

                # only the pairs whose membership changed need reindexing
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

    def save(self, path: str) -> None:
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
        # json gives back lists; merges must be tuples to compare against pairs
        self.bpe_merges = [tuple(pair) for pair in data.get("bpe_merges", [])]
        print(f"  [tokenizer] Loaded ({self.vocab_size:,} tokens) ← {path}")

    def __repr__(self) -> str:
        return (f"Tokenizer(vocab_size={self.vocab_size}, "
                f"bpe_merges={len(self.bpe_merges)}, lowercase={self.lowercase})")
