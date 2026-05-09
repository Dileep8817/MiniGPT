# this is to scrape data to build the corpus to train the tokenizer  as load_data.py got too little data

from datasets import load_dataset
from pathlib import Path
import random

def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = " ".join(text.split())
    return text.strip()

def collect_wikitext(max_samples: int = 50_000) -> list[str]:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    docs = []
    for row in ds:
        text = clean_text(row["text"])
        if len(text) >= 80:
            docs.append(text)
        if len(docs) >= max_samples:
            break

    return docs

def collect_tinystories(max_samples: int = 50_000) -> list[str]:
    ds = load_dataset("roneneldan/TinyStories", split="train")

    docs = []
    for row in ds:
        text = clean_text(row["text"])
        if len(text) >= 80:
            docs.append(text)
        if len(docs) >= max_samples:
            break
    
    return docs

def main() -> None:
    output_path = Path("data/corpus.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    docs = []
    docs.extend(collect_wikitext(max_samples=20_000))
    docs.extend(collect_tinystories(max_samples=30_000))
    random.seed(42)
    random.shuffle(docs)
    with output_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc.replace("\n", " ") + "\n")
    print(f"Wrote {len(docs):,} docs to {output_path}")
    
if __name__ == "__main__":
    main()
