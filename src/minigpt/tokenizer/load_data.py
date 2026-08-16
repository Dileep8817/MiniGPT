from __future__ import annotations
import os
import json
import random
from pathlib import Path 

def load_plain_text(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # blank-line separated paragraphs are one doc each, else fall back to lines
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs

    return [line.strip() for line in content.splitlines() if line.strip()]

def load_jsonl(path: str, text_key: str = "text") -> list[str]:
    # one json object per line, e.g. openwebtext / the pile / c4 dumps
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
    # accepts a list of strings or a list of {text_key: ...} objects
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for item in data:
        if isinstance(item, str):
            if item.strip():
                docs.append(item.strip())
        elif isinstance(item, dict):
            text = item.get(text_key, "").strip()
            if text:
                docs.append(text)
    return docs


def load_directory(path: str, extensions: tuple = (".txt",)) -> list[str]:
    # each file becomes one document
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
    # note some datasets need a config name, e.g. ("wikitext", "wikitext-103-raw-v1")
    try:
        from datasets import load_dataset
    except:
        raise ImportError("Run: pip install datasets")
    
    ds = load_dataset(dataset_name, split=split)
    docs = [row[text_key].strip() for row in ds if row[text_key].strip()]

    if max_samples:
        docs = docs[:max_samples]
    return docs

def generate_synthetic_text(n: int = 2000, seed: int = 42) -> list[str]:
    # so the pipeline runs with no external data; replace before real training
    random.seed(seed)
    subjects  = ["The cat", "A dog", "She", "He", "The team", "The model", "An idea"]
    verbs     = ["runs", "sees", "builds", "finds", "creates", "understands", "explores"]
    objects   = ["the house", "a new path", "the answer", "something unexpected", "a solution"]
    adverbs   = ["quickly", "carefully", "slowly", "easily", "often", "rarely"]
    connectors= ["However,", "Therefore,", "In addition,", "As a result,", "Meanwhile,"]

    docs = []
    for _ in range(n):
        # vary document length
        n_sentences = random.randint(2,6)
        sentences = []
        for _ in range(n_sentences):
            s = f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)} {random.choice(adverbs)}."
            if random.random() > 0.6:
                s = random.choice(connectors) + " " + s
            sentences.append(s)
        docs.append(" ".join(sentences))

    return docs

def load_corpus(cfg) -> list[str]:
    # try each source in turn, fall back to synthetic, return shuffled docs
    docs: list[str] = []

    if os.path.exists(cfg.raw_data_path):
        loaded = load_plain_text(cfg.raw_data_path)
        docs.extend(loaded)
        print(f"  [load_data] Loaded {len(loaded):,} docs from {cfg.raw_data_path}")

    for fname in ["data/train.jsonl", "data/corpus.jsonl"]:
        if os.path.exists(fname):
            loaded = load_jsonl(fname)
            docs.extend(loaded)
            print(f"  [load_data] Loaded {len(loaded):,} docs from {fname}")

    for fname in ["data/train.json", "data/corpus.json"]:
        if os.path.exists(fname):
            loaded = load_json(fname)
            docs.extend(loaded)
            print(f"  [load_data] Loaded {len(loaded):,} docs from {fname}")

    if os.path.isdir("data/texts"):
        loaded = load_directory("data/texts")
        docs.extend(loaded)
        print(f"  [load_data] Loaded {len(loaded):,} docs from data/texts/")

    if not docs:
        print("  [load_data] No data found — generating synthetic text...")
        docs = generate_synthetic_text(n=2000)
        print(f"  [load_data] Generated {len(docs):,} synthetic docs")

    random.shuffle(docs)
    return docs

def save_corpus(docs: list[str], path: str) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        # collapse internal newlines so each doc really is one line
        f.write("\n".join(doc.replace("\n", " ") for doc in docs))
    print(f"  [load_data] Corpus saved → {path}  ({len(docs):,} docs)")

if __name__ == "__main__":
    from minigpt.config import cfg 

    docs = load_corpus(cfg)
    save_corpus(docs, cfg.raw_data_path)
    print(f"\n--- Sample 0 ---\n{docs[0][:200]}")
    print(f"\n--- Sample 1 ---\n{docs[1][:200]}")
