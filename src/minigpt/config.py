# every module reads hyperparams from here, nothing is hardcoded elsewhere

from dataclasses import dataclass

# these always take the lowest IDs so their positions are fixed whatever BPE learns
SPECIAL_TOKENS = [
    "<PAD>",    # 0 — padding (fills short sequences in a batch)
    "<UNK>",    # 1 — unknown token (char not in vocab)
    "<BOS>",    # 2 — beginning of sequence
    "<EOS>",    # 3 — end of sequence
]

@dataclass
class LLMConfig:
    # tokenizer
    vocab_size: int = 16500   # target BPE vocab size (updated after training)
    bpe_num_merges: int = 16000    # how many BPE merge rules to learn
    lowercase: bool = False # normalize to lowercase before tokenizing

    # architecture
    context_len: int   = 512     # max tokens the model sees at once
    n_layers: int   = 6       # number of stacked transformer blocks
    n_heads: int   = 8       # attention heads (d_model must be divisible)
    d_model: int   = 512     # embedding dimension
    d_ff:   int   = 1408     # SwiGLU: ~8/3 * d_model, rounded to multiple of 64
    dropout: float = 0.1
    pos_type: str = "rope"

    # paths
    raw_data_path:   str   = "data/corpus.txt"
    tokenizer_path:  str   = "data/tokenizer.json"
    checkpoint_dir:  str   = "checkpoints/"

    # training
    batch_size:          int   = 32
    grad_accum_steps:    int   = 2      # effective batch = 64
    max_steps:           int   = 2000
    warmup_steps:        int   = 100
    max_lr:              float = 3e-4
    min_lr:              float = 3e-5
    weight_decay:        float = 0.1
    grad_clip:           float = 1.0
    log_every:           int   = 10
    eval_every:          int   = 200
    save_every:          int   = 500
    val_split:           float = 0.1
    use_bf16:            bool  = True
    use_grad_checkpoint: bool  = False
    use_sdpa:            bool  = True

cfg = LLMConfig()
