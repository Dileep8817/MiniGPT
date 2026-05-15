"""Plot training and validation loss from logs/*.csv into figures/."""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt

def load_run(path: str):
    steps_train, train_losses = [], []
    steps_val,   val_losses   = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            if row["train_loss"]:
                steps_train.append(step)
                train_losses.append(float(row["train_loss"]))
            if row["val_loss"]:
                steps_val.append(step)
                val_losses.append(float(row["val_loss"]))
    return steps_train, train_losses, steps_val, val_losses
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/run.csv")
    parser.add_argument("--out", default="figures/loss.png")
    parser.add_argument("--title", default="MiniGPT training")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    s_train, tr, s_val, vl = load_run(args.log)
    plt.figure(figsize=(8, 5))
    plt.plot(s_train, tr, label="train loss", alpha=0.7)
    if s_val:
        plt.plot(s_val, vl, "o-", label="val loss", linewidth=2)
    plt.xlabel("step")
    plt.ylabel("cross-entropy loss")
    plt.title(args.title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    print(f"  wrote {args.out}")
if __name__ == "__main__":
    main()
