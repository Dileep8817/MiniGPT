from __future__ import annotations

import csv
import math, os, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from minigpt.config import cfg
from minigpt.data.dataset import CorpusDataset
from minigpt.model.gpt import MiniGPT
from minigpt.tokenizer.train_tokenizer import load_trained_tokenizer


def get_device() -> torch.device:
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


def build_param_groups(model, weight_decay: float):
    # decay 2D+ tensors only, so norms and biases stay out of weight decay
    decay, nodecay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad: continue
        (decay if p.dim() >= 2 else nodecay).append(p)
    return [
        {"params": decay,   "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]


def lr_at_step(step, warmup, total, max_lr, min_lr):
    # linear warmup then cosine decay to min_lr
    if step < warmup:           
        return max_lr * (step + 1) / warmup
    if step >= total:           
        return min_lr
    p = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * p))


def split_dataset(full_ds: CorpusDataset, val_ratio: float):
    # contiguous tail split — val is the last val_ratio of the token stream
    n = int((1 - val_ratio) * len(full_ds.ids))
    def view(ids):
        return CorpusDataset.from_ids(ids, full_ds.context_len, full_ds.corpus_path)
    return view(full_ds.ids[:n]), view(full_ds.ids[n:])


@torch.no_grad()
def evaluate(model, loader, device, max_batches=100):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches: 
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.item())
    return sum(losses) / max(len(losses), 1)


def save_checkpoint(model, optimizer, step, cfg, path):
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step, "cfg": cfg.__dict__}, path)
    print(f"  [ckpt] saved → {path}")


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]


def main():
    device = get_device()
    print(f"  [train] device = {device}")

    tokenizer = load_trained_tokenizer(cfg)
    full_ds = CorpusDataset.from_config(cfg, tokenizer)
    train_ds, val_ds = split_dataset(full_ds, cfg.val_split)

    # pin_memory is a no-op on MPS (unified memory), so only enable it on cuda
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              drop_last=True, pin_memory=pin, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              drop_last=True, pin_memory=pin, num_workers=2)
    print(f"  [train] train={len(train_ds):,} val={len(val_ds):,}")

    model = MiniGPT(cfg).to(device)
    # "default" mode skips the cuda-graph passes that warn on MPS
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="default")
            print("  [train] torch.compile enabled")
        except Exception as e:
            print(f"  [train] torch.compile unavailable: {e}")
    model.num_params()

    optimizer = torch.optim.AdamW(
        build_param_groups(model, cfg.weight_decay),
        lr=cfg.max_lr, betas=(0.9, 0.95), eps=1e-8,
    )

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "run.csv")
    new_log = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if new_log:
        log_writer.writerow(["step", "train_loss", "val_loss", "lr"])
    start_step = 0
    ckpts = sorted(f for f in os.listdir(cfg.checkpoint_dir) if f.startswith("step_"))
    if ckpts:
        path = os.path.join(cfg.checkpoint_dir, ckpts[-1])
        start_step = load_checkpoint(model, optimizer, path, device)
        print(f"  [train] resumed step {start_step} ← {path}")

    use_amp = cfg.use_bf16 and device.type != "cpu"

    model.train()
    data_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()

    # baseline eval so the random-init starting point is explicit, not inferred
    val_loss0 = evaluate(model, val_loader, device, max_batches=100)
    print(f"  [train] step 0 val_loss (random init / resume) = {val_loss0:.4f}")
    log_writer.writerow([start_step, "", f"{val_loss0:.6f}", ""])
    log_file.flush()
    model.train()

    for step in range(start_step, cfg.max_steps):
        accum_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            try:               x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                    ) / cfg.grad_accum_steps
            else:
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                ) / cfg.grad_accum_steps

            loss.backward()
            accum_loss += loss.item()

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        lr = lr_at_step(step, cfg.warmup_steps, cfg.max_steps, cfg.max_lr, cfg.min_lr)
        for pg in optimizer.param_groups: pg["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % cfg.log_every == 0:
            dt = time.time() - t0
            tok = cfg.batch_size * cfg.grad_accum_steps * (cfg.context_len - 1)
            tps = (tok * cfg.log_every) / max(dt, 1e-9) if step > start_step else 0.0
            # gnorm is the earliest warning sign of instability
            print(f"step {step:>5} | loss {accum_loss:.4f} | lr {lr:.2e} | gnorm {gnorm:.2f} | tok/s {tps:>7,.0f}")
            log_writer.writerow([step, f"{accum_loss:.6f}", "", f"{lr:.6e}"])
            log_file.flush()
            t0 = time.time()

        if step > start_step and step % cfg.eval_every == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"  ↳ val_loss = {val_loss:.4f}")
            log_writer.writerow([step, "", f"{val_loss:.6f}", ""])
            log_file.flush()
            model.train()

        if step > start_step and step % cfg.save_every == 0:
            save_checkpoint(model, optimizer, step, cfg,
                            os.path.join(cfg.checkpoint_dir, f"step_{step:06d}.pt"))

    # the step > start_step guard above skips the last eval_every boundary
    final_val = evaluate(model, val_loader, device, max_batches=200)
    print(f"  [train] FINAL val_loss = {final_val:.4f}")
    log_writer.writerow([cfg.max_steps, "", f"{final_val:.6f}", ""])
    log_file.flush()

    save_checkpoint(model, optimizer, cfg.max_steps, cfg,
                    os.path.join(cfg.checkpoint_dir, f"step_{cfg.max_steps:06d}.pt"))
    print("  [train] done.")


if __name__ == "__main__":
    main()
