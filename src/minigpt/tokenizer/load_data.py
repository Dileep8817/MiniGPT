from __future__ import annotations
import os
import random

def load_plain_text(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # blank-line separated paragraphs are one doc each, else fall back to lines
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs

    return [line.strip() for line in content.splitlines() if line.strip()]

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
    # scripts/build_corpus.py writes the real corpus; synthetic is the fallback
    if os.path.exists(cfg.raw_data_path):
        docs = load_plain_text(cfg.raw_data_path)
        print(f"  [load_data] Loaded {len(docs):,} docs from {cfg.raw_data_path}")
    else:
        print("  [load_data] No corpus found — generating synthetic text...")
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
