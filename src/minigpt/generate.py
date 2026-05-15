"""Text generation from a trained MiniGPT checkpoint"""
from __future__ import annotations

import argparse
import os
import torch
import torch.nn.functional as F

from minigpt.config import cfg
from minigpt.model.gpt import MiniGPT
from minigpt.tokenizer.train_tokenizer import load_trained_tokenizer

# choosing optimal device to use
def get_device() -> torch.device:
    if torch.backends.mps.is_available(): 
        return torch.device("mps")
    if torch.cuda.is_available():         
        return torch.device("cuda")
    return torch.device("cpu") 

# loads model and tokenizer for inference 
def load_model_for_inference(ckpt_path: str, device: torch.device):
    tokenizer = load_trained_tokenizer(cfg)
    model = MiniGPT(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"]) # puts trained weights into transformer
    model.eval()
    step = ckpt.get("step", "?")
    print(f"  [generate] loaded checkpoint step={step} ← {ckpt_path}")
    return model, tokenizer

# from probability of all tokens, keeps only top-k tokens and zeroes out the rest
def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Zero-out probability mass outside the top-k logits."""
    if top_k is None or top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))  # safety check
    vals, _ = torch.topk(logits, top_k) # finds highest k logits
    cutoff = vals[..., -1, None] # gets the k logit val
    return logits.masked_fill(logits < cutoff, float("-inf"))

# instead of fixed number tokens like top_k; it keeps the smallest set of tokens whose cumulative probability exceeds top_p
def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus sampling — keep smallest set whose cumulative prob >= top_p."""
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = logits.sort(descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    mask = cumulative > top_p
    mask[..., 1:] = mask[..., :-1].clone()    # keep first token over threshold
    mask[..., 0] = False
    sorted_logits[mask] = float("-inf")
    return torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)

@torch.no_grad() # disables gradient tracking
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
    device: torch.device | None = None,
    stop_at_eos: bool = True,
) -> str:
    """
    Autoregressive sampling.
        greedy=True       → argmax each step (ignores temperature/top-k/top-p)
        temperature=1.0   → no scaling
        top_k=40          → restrict to top-40 logits before sampling
        top_p=0.9         → nucleus sampling
    """
    device = device or next(model.parameters()).device
    eos_id = tokenizer.stoi.get("<EOS>", None)
    context_len = model.cfg.context_len
    ids = tokenizer.encode(prompt, add_bos=False, add_eos=False)
    if not ids:
        ids = [tokenizer.stoi.get("<BOS>", 0)]
    ids = torch.tensor([ids], dtype=torch.long, device=device)   # (1, T)
    for _ in range(max_new_tokens):
        ids_in = ids if ids.size(1) <= context_len else ids[:, -context_len:]
        logits = model(ids_in)[:, -1, :]                          # (1, V)
        if greedy:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-6)
            logits = apply_top_k(logits, top_k)
            logits = apply_top_p(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)     # (1, 1)
        ids = torch.cat([ids, next_id], dim=1)
        if stop_at_eos and eos_id is not None and next_id.item() == eos_id:
            break
    return tokenizer.decode(ids[0].tolist(), skip_special=True)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/step_002000.pt")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--n_samples", type=int, default=3)
    args = parser.parse_args()
    device = get_device()
    print(f"  [generate] device = {device}")
    model, tokenizer = load_model_for_inference(args.ckpt, device)
    for i in range(args.n_samples):
        print(f"\n──────── Sample {i+1} ────────")
        text = generate(
            model, tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            greedy=args.greedy,
            device=device,
        )
        print(text)
if __name__ == "__main__":
    main()

