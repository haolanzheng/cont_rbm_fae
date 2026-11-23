"""Plot the saved baseline results using the main paper's round index N.

Run from any directory with NumPy and Matplotlib installed.
This script does not run experiments or change the baseline report.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
METHODS = (
    ("exact_indicator", "Exact indicator", "#222222", ":", "x"),
    ("normalized_gd_T100", "GD--100", "#0072B2", "-", "o"),
    ("pool_M256", "Pool--256", "#E69F00", "-.", "s"),
    ("pool_M4096", "Pool--4096", "#D55E00", "--", "D"),
    ("pool_M10000", "Pool--10,000", "#009E73", (0, (3, 1, 1, 1)), "v"),
)


def main():
    data = json.loads((ROOT / "runs/report_baseline/fe_p3_grid40_results.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8))
    for ax, d in zip(axes.flat, (8, 16, 32, 64)):
        ranks = np.arange(d // 2 + 1)
        stride = max(2, len(ranks) // 7)
        ax.semilogy(
            ranks,
            data["cases"][str(d)]["discrete_singular_values"][: len(ranks)],
            color="#666666",
            linestyle=(0, (5, 2)),
            marker="^",
            markersize=4.3,
            markerfacecolor="white",
            markevery=(0, stride),
            lw=1.4,
            label="Kolmogorov width",
        )
        for j, (key, label, color, style, marker) in enumerate(METHODS):
            method = data["cases"][str(d)]["methods"].get(key)
            if method is None:
                continue
            ax.semilogy(
                ranks,
                method["worst_median_by_n"],
                linestyle=style,
                color=color,
                lw=1.8,
                marker=marker,
                markersize=4.2,
                markerfacecolor="white" if marker != "x" else color,
                markeredgewidth=1.0,
                markevery=(j % stride, stride),
                label=label,
            )
        ax.set_title(rf"$d={d}$, final rank $N={d // 2}$")
        ax.set_xlabel("Reduced-basis dimension $N$")
        ax.set_ylabel(r"exact largest energy error $E_N$")
        ax.grid(True, which="both", alpha=0.22)
    handles, labels = axes.flat[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(PAPER / "figures/baseline_gd_largest_error.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
