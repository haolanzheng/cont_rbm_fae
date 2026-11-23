"""Plot strong-selector largest-error curves from strong_baselines.py output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLES = (
    ("dense_eigh", "Dense eigensolver", "#222222", "-", "x"),
    ("lanczos", "Lanczos", "#CC79A7", "--", "+"),
    ("gd100", "GD--100", "#0072B2", "-", "o"),
    ("lbfgs100", "L-BFGS--100", "#D55E00", ":", "^"),
    ("fixed_random4096", "Fixed random--4096", "#E69F00", "--", "D"),
    ("fresh_random4096", "Resampled random--4096", "#009E73", "-.", "s"),
    ("adaptive_cov_pool4096", "Adaptive pool--4096", "#6A3D9A", ":", "*"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parents[2] / "runs/report_baseline"
    parser.add_argument(
        "--data", type=Path, default=here / "strong_baselines_fe_p3_grid40.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "strong_baselines_largest_error_vs_rank.pdf",
    )
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), constrained_layout=True)
    for ax, d in zip(axes.flat, (8, 16, 32, 64)):
        case = data["cases"][str(d)]
        rank = case["rank"]
        stride = max(1, rank // 7)
        for key, label, color, linestyle, marker in STYLES:
            curve = case["methods"][key]["curve_median"]
            ax.semilogy(
                np.arange(rank + 1),
                curve,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                marker=marker,
                markersize=4,
                markevery=stride,
            )
        ax.set(
            title=f"$d={d}$", xlabel="RB dimension $n$", ylabel="Largest energy error"
        )
        ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
