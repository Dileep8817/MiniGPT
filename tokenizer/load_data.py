
# Responsibility: pull raw math text into memory and format every
# problem/solution pair into a consistent string template before the
# tokenizer ever sees it.

import os
import json
import random
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Template ───────────────────────
TEMPLATE = (
    "<BOS>[PROBLEM] {problem} "
    "[SOLUTION]{steps}"
    "[ANSWER] {answer}<EOS>"
)

STEP_SEP = " [STEP] "       # separates individual solution steps

def format_sample(problem: str, steps: list[str], answer: str) -> str:
    """
    Convert a structured problem dict into a single formatted string.
    The model learns the token boundaries  <BOS> … [PROBLEM] … [SOLUTION]
    … [STEP] … [ANSWER] … <EOS>  purely from seeing many examples.
    """
    steps_str = STEP_SEP.join(s.strip() for s in steps) if steps else ""
    if steps_str:
        steps_str = STEP_SEP + steps_str + " "
    return TEMPLATE.format(
        problem=problem.strip(),
        steps=steps_str,
        answer=answer.strip()
    )

# ── Loaders for common open datasets ─────────────────
def load_gsm8k(path: str) -> list[str]:
    """
    Load GSM8K-style JSONL files.
    Each line: {"question": "...", "answer": "..."}
    The answer field in GSM8K contains step-by-step reasoning separated
    by '\\n', with the final number after '####'.
 
    Download: https://github.com/openai/grade-school-math
    """
    samples = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line.strip())
            question = obj["question"]
            raw_answer = obj["answer"]

            # Split reasoning steps from final answer
            if '####' in raw_answer:
                reasoning, final = raw_answer.split("####")
                steps = [s.strip() for s in reasoning.strip().split("\n") if s.strip()]
                answer = final.strip()
            else:
                step= []
                answer = raw_answer.strip()
            
            samples.append(format_sample(question, steps, answer))
    return samples

def load_math_dataset(path: str) -> list[str]:
    """
    Load the MATH dataset (Hendrycks et al., 2021).
    Each problem is a JSON file:  {"problem": "...", "solution": "...", "answer": "..."}
 
    Directory structure expected:
        path/
          algebra/problem1.json
          geometry/problem2.json ...
 
    Download: https://github.com/hendrycks/math
    """
    samples = []
    for json_path in Path(path).rglob("*.json"):
        with open(json_path) as f:
            obj = json.load(f)
        problem = obj.get("problem", "")
        solution = obj.get("solution", "")
        answer = obj.get("answer", "")
        # Treat the full solution as one step (no sub-steps in this dataset)
        samples.append(format_sample(problem, [solution], answer))
    return samples

def load_plain_text(path: str) -> list[str]:
    """
    Fallback: load a plain .txt file where each non-empty line is
    treated as one training sample (no special structure assumed).
    Useful for custom scraped data or textbook passages.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines

def load_custom_json(path: str) -> list[str]:
    """
    Load a custom JSON file in the format:
    [
      {
        "problem":  "What is 2 + 2?",
        "steps":    ["Think about it.", "2 and 2 make 4."],
        "answer":   "4"
      },
      ...
    ]
    """
    with open(path, "r") as f:
        data = json.load(f)
    return [
        format_sample(
            obj.get("problem", ""),
            obj.get("steps", []).
            obj.get("answer", ""),
        )
        for obj in data
    ]

# ── Synthetic fallback data ────────
def generate_synthetic_arithmetic(n: int = 5000, seed: int = 42) -> list[str]:
    """
    Generate n simple arithmetic problems so you can run the full pipeline
    immediately, before downloading any real dataset.
 
    Covers: addition, subtraction, multiplication, integer division,
            linear equations, and basic fractions.
    """
    random.seed(seed)
    samples = []
    for _ in range(n):
        kind = random.choice(["add", "sub", "mul", "div", "linear", "fraction"])

        if kind == "add":
            a, b = random.randint(1, 999), random.randint(1, 999)
            problem = f"Calculate {a} + {b}"
            steps = f"Add {a} and {b} together."
            answer = str(a + b)

        elif kind == "sub":
            a, b = random.randint(1, 999), random.randint(1, 999)
            lo, hi = min(a, b), max(a, b)
            problem = f"Calculate: {hi} - {lo}"
            steps   = [f"Subtract {lo} from {hi}."]
            answer  = str(hi - lo)
 
        elif kind == "mul":
            a, b = random.randint(1, 99), random.randint(1, 99)
            problem = f"Calculate: {a} × {b}"
            steps   = [f"Multiply {a} by {b}."]
            answer  = str(a * b)
 
        elif kind == "div":
            b = random.randint(1, 20)
            a = b * random.randint(1, 50)
            problem = f"Calculate: {a} ÷ {b}"
            steps   = [f"Divide {a} by {b}."]
            answer  = str(a // b)
 
        elif kind == "linear":
            # ax + b = c  →  x = (c - b) / a
            a = random.randint(1, 10)
            x = random.randint(-20, 20)
            b = random.randint(-50, 50)
            c = a * x + b
            problem = f"Solve for x:  {a}x + ({b}) = {c}"
            steps   = [
                f"Subtract {b} from both sides: {a}x = {c - b}",
                f"Divide both sides by {a}: x = {(c - b) // a}",
            ]
            answer  = f"x = {x}"
        
        else: # fraction
            num = random.randint(1, 10)
            den = random.randint(2, 20)
            problem = f"Simplify the fraction: {num*2}/{den*2}"
            from math import gcd
            g = gcd(num*2, den*2)
            steps = [f"Find GCD({num*2}, {den*2}) = {g}",
                       f"Divide numerator and denominator by {g}."]
            answer  = f"{(num*2)//g}/{(den*2)//g}"
 
        samples.append(format_sample(problem, steps, answer))
 
    return samples

# ── Main entry point ────────
def load_corpus(cfg) -> list[str]:
    """
    Master loader: tries real datasets first, falls back to synthetic data.
    Returns a list of formatted strings ready for the tokenizer.
    """
    samples: list[str] = []

    # Try GSM8K
    gsm_path = "data/gsm8k_train.json1"
    if os.path.exists(gsm_path):
        gsm = load_gsm8k(gsm_path)
        samples.extend(gsm)
        print(f" [load_data] Loaded {len(gsm):,} GSM8K samples")
    
    # Try MATH dataset
    math_path = "data/MATH"
    if os.path.exists(math_path):
        math = load_math_dataset(math_path)
        samples.extend(math)
        print(f" [load_data] Loaded {len(math):,} MATH dataset samples")
    
    # Try custom JSON
    custom_path = "data/custom_math.json"
    if os.path.exists(custom_path):
        custom = load_custom_json(custom_path)
        samples.extend(custom)
        print(f"  [load_data] Loaded {len(custom):,} custom samples")
 
    # Fallback: synthetic arithmetic
    if not samples:
        print("  [load_data] No external datasets found — generating synthetic arithmetic...")
        samples = generate_synthetic_arithmetic(n=5000)
        print(f"  [load_data] Generated {len(samples):,} synthetic samples")
 
    random.shuffle(samples)
    return samples

def save_corpus(samples: list[str], path: str) -> None:
    """Write all formatted samples to a single text file, one sample per line."""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(samples))
    print(f" [load_data] Corpus saved -> {path}  ({len(samples):,} samples)")

# ── Quick test ─────
if __name__ == "__main__":
    from config import cfg

    samples = load_corpus(cfg)
    save_corpus(samples, cfg.raw_data_path)
    print(f"\nSample 0:\n{samples[0]}")
    print(f"\nSample 1:\n{samples[1]}")






