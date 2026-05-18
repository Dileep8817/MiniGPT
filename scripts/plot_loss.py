"""
Plot training and validation loss curves from logs/*.csv into figures/.

Two modes:
  1. Single-run plot (one log):
       python scripts/plot_loss.py --log logs/training_v2.1.csv \
           --out figures/v2_loss.png --title "MiniGPT v2.1"

  2. Comparison plot (default — overlays multiple runs):
       python scripts/plot_loss.py

The default behavior compares logs/initial_training_v1.csv against
logs/training_v2.1.csv and writes figures/loss_comparison.png.

The CSV format is:
  step,train_loss,val_loss,lr
with train rows every `log_every` steps (train_loss + lr filled) and
val rows at each eval step (val_loss filled). Repeated step entries
from interrupted/restarted runs are deduplicated by keeping the last
value seen for each (step, metric) combination.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # save-to-file only; no interactive display needed
import matplotlib.pyplot as plt


def load_run(path: str):
    """
    Return (train_steps, train_losses, val_steps, val_losses) for a run.

    If the CSV contains multiple entries for the same (step, metric) — which
    happens when a training run is restarted from a checkpoint — the latest
    value wins. This naturally drops false-start data from interrupted runs.
    """
    train: "OrderedDict[int, float]" = OrderedDict()
    val: "OrderedDict[int, float]" = OrderedDict()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                step = int(row["step"])
            except (KeyError, ValueError):
                continue
            if row.get("train_loss"):
                train[step] = float(row["train_loss"])
            if row.get("val_loss"):
                val[step] = float(row["val_loss"])
    train_steps = sorted(train.keys())
    val_steps = sorted(val.keys())
    return (
        train_steps,
        [train[s] for s in train_steps],
        val_steps,
        [val[s] for s in val_steps],
    )


def plot_runs(runs: list[tuple[str, str]], out_path: str, title: str) -> None:
    """
    runs : list of (label, csv_path) tuples
    """
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    plt.figure(figsize=(10, 6))
    for (label, path), color in zip(runs, colors):
        if not Path(path).exists():
            print(f"  [warn] log not found: {path}")
            continue
        s_train, tr, s_val, vl = load_run(path)
        plt.plot(s_train, tr, label=f"{label} train", color=color,
                 alpha=0.5, linewidth=1.0)
        if s_val:
            plt.plot(s_val, vl, label=f"{label} val", color=color,
                     marker="o", markersize=4, linewidth=2.0)

    plt.xlabel("optimizer step")
    plt.ylabel("cross-entropy loss (nats / token)")
    plt.title(title)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", action="append", default=None,
                        help="CSV log to plot. Pass multiple times to overlay runs.")
    parser.add_argument("--label", action="append", default=None,
                        help="Label for the corresponding --log (in matching order).")
    parser.add_argument("--out", default=None, help="Output figure path.")
    parser.add_argument("--title", default=None, help="Figure title.")
    args = parser.parse_args()

    if args.log is None:
        runs = [
            ("v1",   "logs/initial_training_v1.csv"),
            ("v2.1", "logs/training_v2.1.csv"),
        ]
        out_path = args.out or "figures/loss_comparison.png"
        title = args.title or "MiniGPT — training & validation loss (v1 vs v2.1)"
    else:
        labels = args.label or [Path(p).stem for p in args.log]
        if len(labels) != len(args.log):
            raise SystemExit("--label count must match --log count")
        runs = list(zip(labels, args.log))
        if len(runs) == 1:
            out_path = args.out or f"figures/{labels[0]}_loss.png"
            title = args.title or f"MiniGPT {labels[0]} — training & validation loss"
        else:
            out_path = args.out or "figures/loss_comparison.png"
            title = args.title or "MiniGPT — training & validation loss"

    plot_runs(runs, out_path, title)


if __name__ == "__main__":
    main()
