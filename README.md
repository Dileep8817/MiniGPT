# MiniGPT

I built a small-scale, from-scratch GPT-style decoder-only llm to understand
the transformer architecutre. The llm consists of a byte-pair encoding tokenizer, 
custom training loop, and well the architecture itself. I based my architectural
and training decisions off of the empircal laws on llms discussed in the original
Chinchilla paper: https://arxiv.org/pdf/2203.15556

The repository contains two trained variants:

- **v1** — a simple GPT-2 model (LayerNorm, GELU, learned
  positional embeddings, manual scaled-dot-product attention).
- **v2.1** — a more mdoern Llama-style variant (RMSNorm, SwiGLU, RoPE,
  fused SDPA attention, GPT-2 scaled residual init, bias-free projections,
  bigger BPE vocab).

Both were trained on the same ~37 MB English corpus (Wikitext-103 +
TinyStories sample) on an Apple-Silicon MacBook Air using MPS.

---

## Headline result

| | v1 | v2.1 |
|---|---|---|
| Parameters | 19.97 M | 27.75 M |
| Vocab | 1,545 | 16,545 |
| Final val loss (step 1800) | **1.802** | 2.021 |
| **Bits per byte** (apples-to-apples) | 1.362 | **1.252** |
| Train / val gap (step 1800) | 0.06 | 0.38 |
| Tok/s on MPS (avg) | ~4,000 | ~2,000 |
| Wall-clock (2000 steps) | ~5 h | ~6 h |

v2.1 is **~8 % better in bits-per-byte** despite the higher raw val loss
(different vocabs ⇒ different absolute loss scales). It also clearly
**overfits**, which v1 did not. This was definitly because of the larger
architecture, being trained on the same small data corpus. The architecture
changes worked exactly as designed; the bottleneck for further progress
is now **data**, not architecture.

![Loss curves — v1 vs v2.1](figures/loss_comparison.png)
![Parameter breakdown — v1 vs v2.1](figures/param_breakdown.png)

---

## Repository layout

```
MiniGPT/
├── src/minigpt/
│   ├── config.py                 ── single source of truth for all hyperparams
│   ├── tokenizer/                ── custom BPE: train, save, load, encode
│   │   ├── create_tokenizer.py
│   │   ├── train_tokenizer.py    ── BPE training pipeline (fast Sennrich-style)
│   │   └── load_data.py
│   ├── embeddings/
│   │   ├── token_embedding.py    ── learnable lookup table (tied with head)
│   │   ├── positional_encoding.py── learned + sinusoidal variants (legacy)
│   │   └── embedding.py          ── combines token + (optional) positional
│   ├── model/
│   │   ├── attention.py          ── causal MHA with RoPE + SDPA
│   │   ├── feed_forward.py       ── SwiGLU FFN (Llama-style)
│   │   ├── rms_norm.py           ── Root-Mean-Square LayerNorm
│   │   ├── rotary.py             ── Rotary Positional Embedding (RoPE)
│   │   ├── transformer_block.py  ── pre-norm block: attn + ffn + residuals
│   │   └── gpt.py                ── MiniGPT: stitches everything together
│   ├── data/dataset.py           ── sliding-window CorpusDataset over tokens
│   ├── train.py                  ── full training loop with eval + checkpoints
│   └── generate.py               ── autoregressive sampling from a checkpoint
├── scripts/
│   ├── build_corpus.py           ── pulls Wikitext + TinyStories → data/corpus.txt
│   ├── plot_loss.py              ── renders loss curves into figures/
│   └── plot_params.py            ── renders parameter breakdown chart
├── tests/                        ── pytest unit tests (tokenizer + embeddings)
├── data/                         ── corpus.txt, tokenizer.json (gitignored)
├── checkpoints/                  ── *.pt model checkpoints (gitignored)
├── logs/                         ── training CSVs (kept for reproducibility)
├── figures/                      ── plots used in this README
├── pyproject.toml
└── README.md
```

---

## Quick start

```bash
# 1. Install
git clone <repo> && cd MiniGPT
python -m venv venv && source venv/bin/activate
pip install -e .[dev]            # core deps + pytest

# 2. Build the corpus (pulls ~50k docs from HuggingFace)
python scripts/build_corpus.py

# 3. Train the tokenizer (16,000 BPE merges, ~5 min)
PYTHONPATH=src python -m minigpt.tokenizer.train_tokenizer

# 4. Encode the corpus (cached after first run)
PYTHONPATH=src python -m minigpt.data.dataset

# 5. Train (2000 steps ≈ 6 h on M-series Air, faster on CUDA)
caffeinate -i python -m minigpt.train

# 6. Plot the curves
python scripts/plot_loss.py
python scripts/plot_params.py

# 7. Generate text from a checkpoint
PYTHONPATH=src python -m minigpt.generate \
    --ckpt checkpoints/step_002000.pt --prompt "Once upon a time"
```

---

## Architecture (v2.1)

Each transformer block is the Llama-style pre-norm pattern:

```
x ← x + Attention( RMSNorm(x) )
x ← x + SwiGLU(    RMSNorm(x) )
```

Stacked 6 deep, plus a final RMSNorm and a tied output head. RoPE is
applied to `Q` and `K` inside attention rather than added to the input
embedding. The full forward pass:

```
ids  (B, T) ──► token embedding (16545 × 512, tied)
                       │
                       ▼
                 GPTEmbedding (+ dropout, no positional add)
                       │
        ┌──────────────▼──────────────┐
        │  TransformerBlock × 6        │
        │   RMSNorm → Attention(SDPA, │
        │            RoPE) → +res     │
        │   RMSNorm → SwiGLU → +res   │
        └──────────────┬──────────────┘
                       │
                  final RMSNorm
                       │
              linear head (tied)
                       │
                       ▼
              logits  (B, T, 16545)
```

---

## v1 — Baseline

### Architecture (v1)

| Component | Calculation | Parameters |
|---|---|---|
| Token embedding (tied) | 1545 × 512 | 791,040 |
| Learned positional encoding | 512 × 512 | 262,144 |
| Per block — Attention (Q, K, V, O, with bias) | 4 × (512² + 512) | 1,050,624 |
| Per block — FFN (GELU, with bias) | 2 × (512 × 2048) + 2048 + 512 | 2,099,712 |
| Per block — 2 × LayerNorm | 2 × (2 × 512) | 2,048 |
| **Per block total** | | **3,152,384** |
| 6 blocks | 6 × 3,152,384 | 18,914,304 |
| Final LayerNorm | 2 × 512 | 1,024 |
| Output head (tied) | — | 0 |
| **Total** | | **19,968,512 (~19.97 M)** |

Weight tying saves ~791k parameters; without it the model would be
~20.76 M.

### Training run (v1)

- **Hardware**: Apple-Silicon MPS (MacBook Air).
- **Tokens / optimizer step**: `batch × grad_accum × (T − 1) = 16 × 4 × 511 = 32,704`.
- **2000 steps**: ~65.4 M token-positions of gradient signal ≈ 3.36
  effective passes over the 19.5 M-token corpus.
- **Wall-clock**: ~4.5–5 h at an average of ~4,000 tok/s.

### Chinchilla check (v1)

Compute-optimal data for a 19.97 M-param model is ~20 × params ≈
**399 M unique tokens** (Hoffmann et al., 2022). The corpus has only
19.5 M unique tokens — **4.9 % of compute-optimal**. The loss curve was
still descending at step 2000 with virtually no train/val gap
(`train=1.75`, `val=1.80`). v1 is *not* capacity-limited; it is
**data-limited**.

---

## Improvements made for v2.1

I made some specific changes in arhitecture and training in roughly the
same order they were applied. Everything else (optimizer, LR schedule,
batch×grad-accum effective batch of 64, dropout 0.1) was held constant
so the comparison is honest.

### Tokenizer

- I bumped `num_merges` from 1k to **16k**, and vocab from 1,545 to **16,545**.
- I re-implemented the BPE training loop using the **classical
  Sennrich et al. (2016) algorithm**: a word-frequency dict plus an
  incremental pair counter with a reverse index `pair → {word indices}`.
  Complexity dropped from `O(corpus × num_merges)` to
  `O(affected_words × num_merges)`. The slow original would have taken
  ~19 hours for 16k merges; the optimized version finishes in
  **~5 minutes**, with provably identical output (same merge rules in
  the same order — mathematically equivalent, just bookkept differently).
- Re-encoded the corpus → **15.96 M tokens** (vs v1's 19.5 M). Each token
  now carries more information per byte.

### Architecture

- **Removed the `√d_model` scaling on token embeddings.** This was from
  the original *Attention Is All You Need* paper, but with `std=0.02`
  init the resulting magnitude (~10) is too large for what RMSNorm
  expects (~1). Modern LMs (GPT-2, Llama) drop it.

- **Learned positional encoding → RoPE.** The old learned table assigns
  each absolute position its own vector, so the model has no concept
  that position 48 comes right after 47, or that the gap between
  positions 3 and 5 is the same as between 102 and 104. At inference
  time, sequences longer than `context_len` simply break because there
  are no learned vectors for unseen positions. RoPE instead rotates
  `Q` and `K` in pairs inside attention by an angle proportional to
  position, so the dot product `Q·K` naturally depends on the
  **relative offset** `(i − j)`. Zero learned parameters, generalizes
  past the training context length, and is the modern standard.

- **LayerNorm → RMSNorm.** Normalization keeps activations at a
  consistent scale across the depth of the network. LayerNorm centers
  to zero mean *and* divides by std; RMSNorm just divides by
  `sqrt(mean(x²))`. In deep pre-norm transformers, activations
  naturally hover near zero mean already, so the mean-subtraction step
  is redundant. RMSNorm is faster, has fewer parameters (no bias), and
  matches LayerNorm on every quality benchmark.

- **Manual attention → `F.scaled_dot_product_attention` (SDPA).** The
  previous code did four separate ops: `Q·Kᵀ → causal mask → softmax →
  ·V`. Each materialized the full `(B, h, T, T)` attention matrix to
  memory and read it back for the next op — on MPS, memory bandwidth
  is the bottleneck, so this was wasteful. SDPA fuses the whole pipeline
  into one kernel that never writes the full matrix out. On MPS this is
  ~1.5–2× faster and uses far less memory.

- **GELU FFN → SwiGLU.** The classic FFN is `GELU(W₁·x) → W₂`, a single
  branch with the activation wrapping the first projection. SwiGLU is
  gated: it computes `silu(W_gate·x) ⊙ (W_up·x)` (element-wise product)
  and then projects with `W_down`. The "gate" lets the network learn
  which features to open or close per token, giving a more expressive
  function family. Empirically this gives ~5–10 % better validation
  loss at iso-parameter count (Shazeer 2020, Llama paper). Because
  SwiGLU uses 3 matrices instead of 2, `d_ff` was reduced from `2048`
  to `1408` (≈ ⅔ × 2048, rounded to a multiple of 64) to keep total
  parameters comparable.

- **GPT-2 scaled residual init.** The matrices that *write into* the
  residual stream (`attn.out_proj` and `swiglu.w_down`) are
  re-initialized with `std = 0.02 / √(2·n_layers)` instead of the
  default `0.02`. With 6 layers the multiplier is ~0.408, so each
  block contributes ~⅔ less to the residual stream at init. This
  prevents activations from accumulating across depth and is what keeps
  6-layer (and deeper) stacks from diverging in the first few hundred
  steps. (GPT-2 paper §2.3.)

- **Bias-free Q, K, V, O, FFN projections.** Llama convention — biases
  on linear projections inside pre-norm transformers carry no useful
  signal (the norm zeroes any constant offset on the next layer
  anyway). Removing them saves a few thousand parameters per block and
  is slightly faster.

### Training-loop hygiene

- Baseline val-loss eval at step 0 so the random-init starting point
  (`≈ ln(vocab) ≈ 9.7`) is explicit instead of inferred.
- Final val-loss eval after the last training step (the original loop's
  `step > start_step` guard skipped the last `eval_every` boundary).
- Gradient-norm capture from `clip_grad_norm_` printed every log step
  — the earliest warning sign of training instability.
- `torch.compile(..., mode="default")` enabled, with safe failover to
  eager mode if compile cannot engage on the device.
- `pin_memory=True` is now conditional on `device.type == "cuda"` so
  MPS doesn't print the no-op warning.

### Data scale-up (deferred)

- For Chinchilla-optimal training of the 27.75 M-param v2 model, the
  corpus would need ~554 M unique tokens (~1.5 GB of text). The
  realistic source is **FineWeb-Edu** via `datasets.load_dataset(...)`.
  At ~2,000 tok/s on MPS, one pass would take ~77 hours — not feasible
  on this hardware. **Not implemented in this project; flagged as the
  obvious next step**, ideally on a cloud GPU.

---

## v2.1 — Pre-training smoke test (step 100)

Before committing to the full overnight run, a 100-step diagnostic was
done to compare v2 architecture against v1 on the same hardware:

| Step | v1 train loss | v2 train loss |
|---|---|---|
| 0   | 7.54 | 9.85 |
| 10  | 5.76 | 6.02 |
| 30  | 4.49 | 4.85 |
| 50  | 3.94 | 4.07 |
| 90  | 3.34 | 3.38 |
| 100 | 3.30 | 3.39 |

The raw numbers aren't directly comparable because v2 has an 11× bigger
vocabulary (16,545 vs 1,545), so a higher absolute loss can still mean
better per-byte quality. Converting to **bits per byte** with
`bpb = loss × (tokens / corpus_bytes) / ln(2)`:

- v1 @ step 100: `3.30 × (19.5M / 37M) / ln(2) ≈ 2.51 bpb`
- v2 @ step 100: `3.39 × (15.96M / 37M) / ln(2) ≈ 2.11 bpb`

So by step 100, v2 has already reached a per-byte quality v1 needed
several hundred steps to hit (~**~25 % better per-step quality**). `gnorm`
behaved cleanly (settled into the 0.6–2.5 range, no instability).

### The throughput regression

v2.1 ran at ~2,000 tok/s vs v1's ~4,000 tok/s on the same hardware,
despite all the supposed speed wins from SDPA and bf16. Root cause
analysis:

| Cost added | Impact on tok/s |
|---|---|
| Output head: vocab 1,545 → 16,545 | 11× more compute in head matmul + softmax (this single change dominates) |
| SwiGLU: 2 matrices → 3 | +50 % FFN compute, even at smaller `d_ff` |
| RMSNorm fp32 cast | Small constant overhead per norm |
| SDPA on MPS | ~1.5–2× faster attention (real, but doesn't compensate for the above) |
| MacBook Air thermal throttling | Sustained throughput drops 30–50 % from peak on passive cooling |

Net effect: v2 trades raw throughput for *information* per token. The
tokenizer compresses the corpus 19.5M → 16M tokens, so processing fewer
tokens to see the same text partially offsets the slowdown.

**Quantization is the wrong tool** to recover the lost throughput.
Training needs precise gradients; INT8 quantization is for inference,
not training. The only "quantization" that helps training is bf16/fp16
mixed precision — which v2 already uses. The realistic levers are
bigger batches (already at 32), `torch.compile`, active cooling, or
moving training to a beefier GPU.

---

## v2.1 — Full 2000-step run

| Metric | Value |
|---|---|
| Wall clock | ~7 hours |
| Average throughput | ~2,000 tok/s |
| Step-0 baseline val_loss | 9.67 |
| Final train loss (step 1990) | 1.685 |
| Final val loss (step 1800) | **2.021** |
| Final val loss (step 2000) | 2.026 (↑ — overfitting onset) |
| Train / val gap at step 1800 | 0.38 |
| Bits per byte (val, step 1800) | **1.252** |

### Train/val gap analysis

Val loss progression by eval step:

| Step | Val loss | Δ |
|---|---|---|
| 0 | 9.67 | — |
| 200 | 2.79 | −6.88 |
| 400 | 2.44 | −0.35 |
| 600 | 2.30 | −0.14 |
| 800 | 2.19 | −0.11 |
| 1000 | 2.14 | −0.05 |
| 1200 | 2.09 | −0.05 |
| 1400 | 2.06 | −0.03 |
| 1600 | 2.03 | −0.03 |
| 1800 | 2.021 | −0.01 |
| **2000** | **2.026** | **+0.005 ← val went UP** |

By step ~1200 generalization improvements had essentially plateaued;
val loss *increased* between steps 1800 → 2000. Meanwhile train loss
kept dropping. Classic data-starvation overfitting, exactly what the
Chinchilla scaling laws predict for a 27.75 M model on only 15.96 M
unique tokens (the model has memorization capacity to spare).

Two factors compound here:

1. **Capacity overfit.** v2 is ~40 % larger than v1, same data, same
   dropout. More capacity ⇒ more room to memorize specific training
   sequences. v1 didn't overfit because it didn't have the capacity to.
2. **Train/val distribution mismatch.** `split_dataset` in `train.py`
   takes the *last* 10 % of the contiguous token stream as val. If the
   corpus has any document-level ordering (it does — wikitext articles
   then tinystories), the val set is from a slightly different
   distribution than train. The tokenizer is also trained on only the
   first 5 MB, which biases BPE merges toward the train distribution
   and inflates val loss further.

Both factors are fixable but neither matters more than getting more
data — see the conclusion.

---

## Conclusion

**The architecture changes worked.** v2.1 reaches better per-byte
quality than v1 at every comparable step, sees less raw text per unit
of wall-clock, and has more headroom for further training. RoPE,
RMSNorm, SwiGLU, SDPA, and GPT-2 scaled init are all unambiguous
improvements at this scale.

**The model is now provably data-limited.** v1 didn't overfit because
its smaller capacity couldn't memorize the 19.5 M-token corpus. v2 can,
and so it does. With only ~5 % of Chinchilla-optimal data, no amount of
additional training-step compute on the same corpus will improve val
loss — the curve has already plateaued.

**The right next step is more data, not more steps.** Scaling the
corpus to ~554 M tokens (FineWeb-Edu) and training one Chinchilla-optimal
pass would unlock the rest of v2's capacity. At ~2,000 tok/s on a
MacBook Air that would take ~77 hours of continuous training, which is
why it wouldn't be that optimal for me. I guessthe recommended path is to 
move training to a free Colab T4 (~8 h) or a paid Lambda Labs A100 (~1 h).

This project deliberately stops short of that scale-up: the architecture
work is complete, the empirical findings are clean, and the next phase
is an infrastructure project rather than a research one.

---

## Future work

In priority order, if this project is picked up again:

1. **Random / chunk-shuffled train/val split.** Replace the contiguous
   tail split with a deterministic shuffle of ~8k-token chunks. Most
   likely closes ~0.1–0.2 of the v2.1 train/val gap immediately, no
   retraining needed (just re-evaluate the existing checkpoint).
2. **Memory-mapped `CorpusDataset`.** At 554 M tokens the current
   `self.ids = torch.LongTensor(...)` would consume 4.4 GB of RAM.
   Switching to a `numpy.uint16` memmap drops that to <100 MB while
   supporting random window access. Required before any data scale-up.
3. **FineWeb-Edu data pipeline.** Stream the dataset, tokenize once
   with an LRU-cached `apply_bpe`, save to the memmap. Roughly 30 min
   of work end-to-end.
4. **Cloud training notebook.** Push the repo + memmap to Google Drive,
   run training in a Colab T4 / Kaggle notebook. The training loop
   itself is already device-agnostic via `get_device()`.
5. **Early-stopping by best val loss.** Save the model at the lowest
   val loss seen, not just at the last step.
6. **`bits_per_byte()` evaluation utility** so future comparisons
   don't need manual normalization.

---

## Setup details

Tested on Python 3.9, PyTorch 2.8, macOS 15.7, Apple M-series MPS.
`pyproject.toml` declares the runtime deps; install with:

```bash
pip install -e .            # core
pip install -e .[dev]       # core + pytest
```

The trained tokenizer and tokenized corpus are gitignored (you regenerate
them with the steps above). Checkpoints are also gitignored to keep the
repo small (each is ~333 MB).

---

## References

- Vaswani et al. (2017). *Attention Is All You Need.*
- Radford et al. (2019). *Language Models Are Unsupervised Multitask
  Learners* (GPT-2). Scaled residual init recipe in §2.3.
- Sennrich, Haddow, Birch (2016). *Neural Machine Translation of Rare
  Words with Subword Units.* (BPE training algorithm.)
- Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary
  Position Embedding.*
- Zhang, Sennrich (2019). *Root Mean Square Layer Normalization.*
- Shazeer (2020). *GLU Variants Improve Transformer.*
- Touvron et al. (2023). *Llama: Open and Efficient Foundation Language
  Models.* (RMSNorm + SwiGLU + RoPE in production.)
- Hoffmann et al. (2022). *Training Compute-Optimal Large Language
  Models* (Chinchilla scaling laws — 20 tokens per parameter heuristic).
