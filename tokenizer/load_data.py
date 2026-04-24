
# Responsibility: pull raw text into memory from various sources.
# For a general LLM this means: plain text files, Wikipedia dumps,
# books, web-scraped content, or custom JSON datasets.

from __future__ import annotations
import os
import json
import random
from pathlib import Path 

# ── Loaders ───────────────────────
def load_plain_text(path: str) -> list[str]:
    """
    Load a .txt file. Each non-empty paragraph (blank-line separated) becomes
    one training document. Falls back to line-by-line if no blank lines found.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Try paragraph splitting first
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    
    # Fallback: line by line
    return [line.strip() for line in content.splitlines() if line.strip()]

def load_jsonl(path: str, text_key: str = "text") -> list[str]:
    """
    Load a .jsonl file (one JSON object per line).
    Looks for a field named text_key in each object.
 
    Compatible with: OpenWebText, The Pile, C4, custom datasets.
    Each line expected: {"text": "...", ...}
    """
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get(text_key, "").strip()
                if text:
                    docs.append(text)
            except json.JSONDecodeError:
                continue
    return docs

def load_json(path: str, text_key: str = "text") -> list[str]:
    """
    Load a .json file containing a list of objects or a list of strings.
 
    Formats supported:
        ["doc1 text", "doc2 text", ...]
        [{"text": "doc1 text"}, {"text": "doc2 text"}, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
 
    docs = []
    for item in data:
        if isinstance(item, str):
            if item.strip():
                docs.append(item.strip())
        elif isinstance(item, dict):
            text = item.get(text_key, "").strip()   # FIX: was `.` instead of `,`
            if text:
                docs.append(text)
    return docs
 
 
def load_directory(path: str, extensions: tuple = (".txt",)) -> list[str]:
    """
    Recursively load all text files in a directory.
    Each file becomes one document.
    """
    docs = []
    for file_path in Path(path).rglob("*"):
        if file_path.suffix in extensions:
            try:
                text = file_path.read_text(encoding="utf-8").strip()
                if text:
                    docs.append(text)
            except (UnicodeDecodeError, OSError):
                continue
    return docs

def load_huggingface(dataset_name: str, split: str = "train",
                     text_key: str = "text", max_samples: int | None = None) -> list[str]:
    """
    Load from HuggingFace datasets (requires: pip install datasets).
 
    Examples:
        load_huggingface("wikitext", ...)    — Wikipedia text
        load_huggingface("bookcorpus", ...)  — Books
        load_huggingface("c4", ...)          — Web text
 
    Note: some datasets need a config name, e.g.:
        datasets.load_dataset("wikitext", "wikitext-103-raw-v1")
    """
    try:
        from datasets import load_dataset
    except:
        raise ImportError("Run: pip install datasets")
    
    ds = load_dataset(dataset_name, split=split)
    docs = [row[text_key].strip() for row in ds if row[text_key].strip()]

    if max_samples:
        docs = docs[:max_samples]
    return docs

# ── Synthetic fallback ─────────
def generate_synthetic_text(n: int = 2000, seed: int = 42) -> list[str]:
    """
    Generate simple synthetic text so the pipeline runs immediately with
    no external data. Covers diverse sentence patterns.
    Replace with real data before serious training.
    """
    random.seed(seed)
    subjects  = ["The cat", "A dog", "She", "He", "The team", "The model", "An idea"]
    verbs     = ["runs", "sees", "builds", "finds", "creates", "understands", "explores"]
    objects   = ["the house", "a new path", "the answer", "something unexpected", "a solution"]
    adverbs   = ["quickly", "carefully", "slowly", "easily", "often", "rarely"]
    connectors= ["However,", "Therefore,", "In addition,", "As a result,", "Meanwhile,"]

    docs = []
    for _ in range(n):
        # Vary dodcument length
        n_sentences = random.randint(2,6)
        sentences = []
        for _ in range(n_sentences):
            s = f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)} {random.choice(adverbs)}."
            if random.random() > 0.6:
                s = random.choice(connectors) + " " + s
            sentences.append(s)
        docs.append(" ".join(sentences))
 
    return docs

# ── Master loader ────────
def load_corpus(cfg) -> list[str]:
    """
    Try all supported data sources in order. Falls back to synthetic text.
    Returns a shuffled list of document strings.
    """
    docs: list[str] = []

    # 1. Plain text file (drop-in: just put a .txt in data/)
    if os.path.exists(cfg.raw_data_path):
        loaded = load_plain_text(cfg.raw_data_path)
        docs.extend(loaded)
        print(f"  [load_data] Loaded {len(loaded):,} docs from {cfg.raw_data_path}")
    
    # 2. JSONL (OpenWebText style)
    for fname in ["data/train.jsonl", "data/corpus.jsonl"]:
        if os.path.exists(fname):
            loaded = load_jsonl(fname)
            docs.extend(loaded)
            print(f"  [load_data] Loaded {len(loaded):,} docs from {fname}")
 
    # 3. JSON list
    for fname in ["data/train.json", "data/corpus.json"]:
        if os.path.exists(fname):
            loaded = load_json(fname)
            docs.extend(loaded)
            print(f"  [load_data] Loaded {len(loaded):,} docs from {fname}")
 
    # 4. Directory of text files
    if os.path.isdir("data/texts"):
        loaded = load_directory("data/texts")
        docs.extend(loaded)
        print(f"  [load_data] Loaded {len(loaded):,} docs from data/texts/")
 
    # 5. Synthetic fallback
    if not docs:
        print("  [load_data] No data found — generating synthetic text...")
        docs = generate_synthetic_text(n=2000)
        print(f"  [load_data] Generated {len(docs):,} synthetic docs")
 
    random.shuffle(docs)
    return docs
 
def save_corpus(docs: list[str], path: str) -> None:
    """Write all documents to a flat text file — one document per line."""
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        # Collapse internal newlines so each document is truly one line
        f.write("\n".join(doc.replace("\n", " ") for doc in docs))
    print(f"  [load_data] Corpus saved → {path}  ({len(docs):,} docs)")

# ── Quick test ───────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import cfg  

    docs = load_corpus(cfg)
    save_corpus(docs, cfg.raw_data_path)
    print(f"\n--- Sample 0 ---\n{docs[0][:200]}")
    print(f"\n--- Sample 1 ---\n{docs[1][:200]}")






