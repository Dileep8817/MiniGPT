"""
Render a stacked-bar visualization of the parameter breakdown for each
MiniGPT version, showing where the model's capacity lives.

Two bars (v1 / v2.1) are stacked by component:
    - token embedding (tied with output head)
    - positional encoding (learned for v1, zero for v2 RoPE)
    - per-block attention (Q, K, V, O projections)
    - per-block FFN (GELU for v1, SwiGLU for v2)
    - per-block normalization (LayerNorm for v1, RMSNorm for v2)
    - final normalization

The numbers below are derived from the actual config of each version
and have been cross-checked against `MiniGPT.num_params()` at load
time. Edit the dictionaries to update if you train another variant.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")          # save-to-file only; no interactive display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─── Per-version parameter breakdowns ────────────────────────────────────────
# All values are exact integer parameter counts.

V1 = {
    "label": "v1 (LayerNorm + GELU + learned PE)",
    "vocab_size": 1545,
    "d_model": 512,
    "context_len": 512,
    "n_layers": 6,
    "components": {
        "token embedding (tied)":     1545 * 512,                          # 791,040
        "positional encoding":        512 * 512,                           # 262,144
        "attention (all blocks)":     6 * (4 * (512 * 512 + 512)),         # 6,303,744
        "FFN (all blocks)":           6 * (2 * (512 * 2048) + 2048 + 512), # 12,598,272
        "norms (all blocks + final)": 6 * (2 * (2 * 512)) + 2 * 512,       # 13,312
    },
    "reported_total": 19_968_512,
}

V2 = {
    "label": "v2.1 (RMSNorm + SwiGLU + RoPE)",
    "vocab_size": 16545,
    "d_model": 512,
    "context_len": 512,
    "n_layers": 6,
    "components": {
        "token embedding (tied)":     16545 * 512,                         # 8,471,040
        "positional encoding":        0,                                    # RoPE, no params
        "attention (all blocks)":     6 * (4 * (512 * 512)),               # 6,291,456 (bias-free)
        "FFN (all blocks)":           6 * (3 * (512 * 1408)),              # 12,976,128 (SwiGLU, 3 mats, bias-free)
        "norms (all blocks + final)": 6 * (2 * 512) + 512,                 # 6,656 (RMSNorm, no bias)
    },
    "reported_total": 27_745_280,
}

# Consistent colors per component across the two stacks
COMPONENT_COLORS = {
    "token embedding (tied)":     "#4C72B0",
    "positional encoding":        "#DD8452",
    "attention (all blocks)":     "#55A467",
    "FFN (all blocks)":           "#C44E52",
    "norms (all blocks + final)": "#8172B2",
}


def _verify(version: dict) -> None:
    total = sum(version["components"].values())
    expected = version["reported_total"]
    if total != expected:
        raise SystemExit(
            f"[{version['label']}] components sum to {total:,} "
            f"but reported_total is {expected:,}"
        )


def main():
    _verify(V1)
    _verify(V2)

    versions = [V1, V2]
    components = list(COMPONENT_COLORS.keys())

    fig, (ax_bar, ax_text) = plt.subplots(
        1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.2, 1]}
    )

    # ── Left: stacked bars ──────────────────────────────────────────────
    bar_width = 0.55
    x_positions = [0, 1]
    bottoms = [0, 0]
    for comp in components:
        heights = [v["components"].get(comp, 0) for v in versions]
        ax_bar.bar(
            x_positions, heights, bar_width,
            bottom=bottoms, color=COMPONENT_COLORS[comp],
            edgecolor="white", linewidth=1.0,
        )
        for i, h in enumerate(heights):
            if h > 200_000:  # only annotate big slices
                ax_bar.text(
                    x_positions[i], bottoms[i] + h / 2,
                    f"{h/1e6:.2f}M",
                    ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold",
                )
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    # Total annotations on top of each bar
    for i, v in enumerate(versions):
        total = v["reported_total"]
        ax_bar.text(
            x_positions[i], total + 600_000,
            f"{total/1e6:.2f}M total",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels(["v1", "v2.1"], fontsize=12)
    ax_bar.set_ylabel("parameters")
    ax_bar.set_title("Parameter breakdown by component")

    def millions(x, _):
        return f"{x/1e6:.0f}M"
    ax_bar.yaxis.set_major_formatter(plt.FuncFormatter(millions))
    ax_bar.grid(True, axis="y", alpha=0.3)

    legend_handles = [
        mpatches.Patch(color=COMPONENT_COLORS[c], label=c) for c in components
    ]
    ax_bar.legend(handles=legend_handles, loc="upper left", fontsize=9,
                  framealpha=0.95)

    # ── Right: per-version textual summary ──────────────────────────────
    ax_text.axis("off")
    lines = []
    for v in versions:
        lines.append(v["label"])
        lines.append("─" * 48)
        lines.append(f"  vocab_size  : {v['vocab_size']:>10,}")
        lines.append(f"  d_model     : {v['d_model']:>10,}")
        lines.append(f"  n_layers    : {v['n_layers']:>10,}")
        lines.append(f"  context_len : {v['context_len']:>10,}")
        lines.append("")
        for comp, count in v["components"].items():
            pct = 100 * count / v["reported_total"]
            lines.append(f"  {comp:<28} {count:>12,}  ({pct:>4.1f}%)")
        lines.append("  " + "─" * 46)
        lines.append(f"  {'TOTAL':<28} {v['reported_total']:>12,}")
        lines.append("")
        lines.append("")

    ax_text.text(
        0.0, 1.0, "\n".join(lines),
        family="monospace", fontsize=9,
        va="top", ha="left",
        transform=ax_text.transAxes,
    )

    fig.suptitle("MiniGPT — parameter inventory across versions",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = "figures/param_breakdown.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
