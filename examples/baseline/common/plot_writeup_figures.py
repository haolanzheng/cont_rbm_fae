"""Export the writeup's figures at their final print sizes from archived data.

No experiments are run. Only paper/writeup/figs and the export manifest change.
Run from the repository root with .venv_cpu/bin/python.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/writeup_matplotlib")
os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ.get("PATH", "")
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, NullFormatter, ScalarFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper/writeup/figs"
AUDIT = ROOT / "runs/writeup_figures_20261121"
SOURCE = ROOT / "runs/piecewise_forcing_2d_20261112/slides_summary"
SPECTRAL = ROOT / "runs/energy_residual_20261120"
FULL = 6.5
SINGLE = 4.8
FONT_PT = 9
INPUTS = {}
EXPORTS = {}

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[T1]{fontenc}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": FONT_PT,
    "axes.labelsize": FONT_PT,
    "axes.titlesize": FONT_PT,
    "xtick.labelsize": FONT_PT,
    "ytick.labelsize": FONT_PT,
    "legend.fontsize": FONT_PT,
    "axes.titlepad": 4,
    "axes.labelpad": 2,
    "axes.linewidth": .65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "lines.linewidth": 1.35,
    "lines.markersize": 3.3,
    "legend.frameon": False,
    "legend.handlelength": 2.0,
    "legend.handletextpad": .45,
    "legend.columnspacing": 1.15,
    "legend.labelspacing": .3,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

SOURCE_STYLES = [
    ("gd80", r"GD, $T=80$", "#0072B2", "-", "o"),
    ("pool256", "Pool 256", "#555555", ":", "s"),
    ("pool4096", "Pool 4096", "#009E73", "-.", "^"),
    ("pool10000", "Pool 10000", "#C74432", "--", "D"),
]
RATIO_STYLES = [
    ("gd", r"GD, $T=40$", "#245E91", "-", "o"),
    ("pool256", "Pool 256", "#BC762C", "--", "^"),
    ("pool4096", "Pool 4096", "#477A68", "-.", "D"),
    ("pool10000", "Pool 10000", "#B44E49", ":", "s"),
]
CURVE_STYLES = [
    ("gd", r"GD, $T=40$", "#24557A", "-", "o"),
    ("pool256", "Pool 256", "#C17E3C", "--", "^"),
    ("pool4096", "Pool 4096", "#438273", "-.", "D"),
    ("pool10000", "Pool 10000", "#993F4D", ":", "s"),
    ("grid1", r"Grid $h=1$", "#555555", "--", "v"),
    ("grid0.5", r"Grid $h=1/2$", "#555555", "-.", "P"),
    ("grid0.25", r"Grid $h=1/4$", "#555555", ":", "X"),
]


def remember(path):
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(path):
    return json.loads(remember(path).read_text())


def save(fig, name):
    """Preserve physical size, and record every plotted series for verification."""
    fig.canvas.draw()
    # Catch clipped labels before accepting an export.
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        box = ax.get_tightbbox(renderer)
        bounds = fig.bbox
        assert box.x0 >= -1 and box.y0 >= -1, (name, box)
        assert box.x1 <= bounds.x1 + 1 and box.y1 <= bounds.y1 + 1, (name, box)
    fig.savefig(OUT / (name + ".pdf"))
    EXPORTS[name] = {
        "size_inches": fig.get_size_inches().tolist(),
        "font_pt": FONT_PT,
        "axes": [{"lines": [{"label": line.get_label(),
                               "x": np.asarray(line.get_xdata()).tolist(),
                               "y": np.asarray(line.get_ydata()).tolist()}
                              for line in ax.lines],
                  "images": [np.asarray(im.get_array()).tolist() for im in ax.images]}
                 for ax in fig.axes],
    }
    plt.close(fig)
    print(name, flush=True)


def line(ax, x, y, spec, log=False, every=None):
    _, label, color, ls, marker = spec
    fn = ax.semilogy if log else ax.plot
    return fn(x, y, color=color, ls=ls, marker=marker, label=label,
              markevery=every or max(1, len(x) // 7), mfc="white", mew=.75)[0]


def grid(ax):
    ax.grid(alpha=.17, linewidth=.55)
    ax.set_axisbelow(True)


def median_curve(rows, method, key):
    values = [r[key] for r in rows if r["method"] == method]
    length = min(map(len, values))
    return np.median([v[:length] for v in values], axis=0)


def shared_legend(fig, ax, ncol=4):
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=ncol,
               bbox_to_anchor=(.51, .005), borderaxespad=0, borderpad=.2)


def source_errors(data):
    dims = sorted(map(int, data))
    fig, ax = plt.subplots(figsize=(SINGLE, 2.65))
    for spec in SOURCE_STYLES:
        values = [median_curve(data[str(d)]["curves"], spec[0], "energy_error")[d // 2]
                  for d in dims]
        line(ax, dims, values, spec, every=1)
    ax.set(xscale="log", xlabel=r"Parameter dimension $d$", ylabel=r"$E_N$")
    ax.set_xticks([4, 16, 32, 64, 128, 256], labels=["4", "16", "32", "64", "128", "256"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    grid(ax)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout(pad=.45)
    save(fig, "piecewise_source_error_vs_d")

    fig, axes = plt.subplots(2, 2, figsize=(FULL, 3.7))
    for ax, d in zip(axes.flat, [4, 16, 64, 256]):
        rows = data[str(d)]["curves"]
        for spec in SOURCE_STYLES:
            values = median_curve(rows, spec[0], "energy_error")
            line(ax, np.arange(len(values)), values, spec, log=True)
        if d == 4:
            for m, label, ls in [("grid1", r"Grid $h=1$", "--"),
                                 ("grid0.5", r"Grid $h=1/2$", ":"),
                                 ("grid0.25", r"Grid $h=1/4$", "-.")]:
                values = median_curve(rows, m, "energy_error")
                ax.semilogy(range(len(values)), values, label=label, color="#9A6B26", ls=ls)
        ax.set(title=rf"$d={d}$", xlabel=r"$N$", xlim=(0, len(values)-1))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        grid(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$E_N$")
    shared_legend(fig, axes[0, 0])
    fig.tight_layout(pad=.45, h_pad=.7, w_pad=.75, rect=(0, .14, 1, 1))
    save(fig, "piecewise_source_error_vs_N")


def source_history():
    history = np.load(remember(SOURCE / "d64_n8_history.npz"))["parameters"]
    pools = read_json(SOURCE / "pool_source_comparison.json")["selected"]
    fig = plt.figure(figsize=(FULL, 2.65))
    gs = fig.add_gridspec(2, 5, left=.055, right=.88, bottom=.13, top=.91,
                          hspace=.62, wspace=.32)
    cmap = LinearSegmentedColormap.from_list("source", ["#0072B2", "#FAFAF8", "#C53B32"])
    for row, columns, labels, values in [
        (0, range(5), [rf"GD: $t={t}$" for t in [0, 20, 40, 60, 80]],
         [history[t] for t in [0, 20, 40, 60, 80]]),
        (1, range(1, 4), [f"Pool {m}" for m in [256, 4096, 10000]],
         [pools[str(m)]["parameter"] for m in [256, 4096, 10000]])
    ]:
        for i, (col, label, value) in enumerate(zip(columns, labels, values)):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(np.asarray(value).reshape(8, 8).T, origin="lower", extent=[0, 1, 0, 1],
                           cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
            ax.set(title=label, xticks=[0, 1], yticks=[0, 1])
            if i == 0:
                ax.set_xlabel(r"$x_1$", labelpad=-7)
                ax.set_ylabel(r"$x_2$", labelpad=-5)
            ax.tick_params(length=2)
    cax = fig.add_axes([.912, .25, .015, .51])
    bar = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1])
    bar.set_label(r"$f(x;z)$", labelpad=3)
    save(fig, "piecewise_source_gd_history")

    result = read_json(SOURCE / "fixed_space_enrichment/results.json")
    fig, ax = plt.subplots(figsize=(SINGLE, 2.65))
    line(ax, [r["t"] for r in result["gd"]], [r["E9"] for r in result["gd"]],
         ("gd", "GD candidate", "#0072B2", "-", "o"), every=20)
    for r, spec in zip(result["pools"], SOURCE_STYLES[1:]):
        ax.axhline(r["E9"], label=spec[1], color=spec[2], ls=spec[3])
    ax.set(xlabel=r"GD iteration $t$", ylabel=r"$E_9$", xlim=(0, 80), xticks=[0, 20, 40, 60, 80])
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    grid(ax)
    shared_legend(fig, ax, ncol=2)
    fig.tight_layout(pad=.45, rect=(0, .17, 1, 1))
    save(fig, "piecewise_source_enrichment_history")


def boundary_errors():
    result = read_json(ROOT / "runs/piecewise_boundary_updates_20261121/summary.json")
    styles = [("clipping", "Clipping", "#0072B2", "-", "o"),
              ("active_face", "Active face", "#C74432", "--", "s"),
              ("adaptive_normal", "Adaptive normal updates", "#009E73", "-.", "^")]
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 3.45))
    for ax, d in zip(axes.flat, [4, 16, 64, 256]):
        rows = next(r["curves"] for r in result["results"] if r["d"] == d)
        for spec in styles:
            values = median_curve(rows, spec[0], "energy_error")
            line(ax, np.arange(len(values)), values, spec, log=True)
        ax.set(title=rf"$d={d}$", xlabel=r"$N$", xlim=(0, len(values)-1))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        grid(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$E_N$")
    shared_legend(fig, axes[0, 0], ncol=3)
    fig.tight_layout(pad=.45, h_pad=.7, w_pad=.75, rect=(0, .08, 1, 1))
    save(fig, "piecewise_boundary_updates_error_vs_N")


def spectral_errors(rows):
    def select(dom, d, m, curves=False):
        return [r for r in rows if (r["domain"], r["d"], r["method"]) == (dom, d, m)
                and (not curves or "errors" in r)]

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.65))
    dims = list(range(8, 65, 4))
    for dom, ax in zip(["sphere", "ball"], axes):
        all_values = []
        for spec in RATIO_STYLES:
            values = [float(np.median([r["ratio"] for r in select(dom, d, spec[0])])) for d in dims]
            all_values.extend(values)
            line(ax, dims, values, spec, every=1)
        ax.set(title=dom.capitalize(), xlabel=r"$d$", xticks=[8, 16, 32, 48, 64],
               ylim=(.96, max(2.25, 1.05*max(all_values))))
        ax.axhline(1, color="#777777", lw=.65, ls=":", zorder=0)
        grid(ax)
    axes[0].set_ylabel(r"$E_N/d_N$")
    shared_legend(fig, axes[0])
    fig.tight_layout(pad=.45, w_pad=.9, rect=(0, .115, 1, 1))
    save(fig, "test_width_ratio_vs_d")

    fig, axes = plt.subplots(2, 4, figsize=(FULL, 3.95), sharey="col")
    for ri, dom in enumerate(["sphere", "ball"]):
        for ax, d in zip(axes[ri], [6, 16, 32, 64]):
            for spec in CURVE_STYLES:
                rr = select(dom, d, spec[0], curves=True)
                if not rr:
                    continue
                values = median_curve(rr, spec[0], "errors")
                line(ax, np.arange(len(values)), values, spec, log=True)
            ax.set(title=rf"{dom.capitalize()}, $d={d}$", xlabel=r"$N$", xlim=(0, d-1),
                   xticks=[0, d//2, d-1])
            if d in (6, 16):
                ax.set_yticks([.3, .6, 1] if d == 6 else [.03, .1, 1])
                ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            elif d == 64:
                ax.set_yticks([1e-6, 1e-3, 1])
            ax.yaxis.set_minor_formatter(NullFormatter())
            grid(ax)
        axes[ri, 0].set_ylabel(r"$E_N$")
    shared_legend(fig, axes[0, 0])
    fig.tight_layout(pad=.45, h_pad=.85, w_pad=.6, rect=(0, .135, 1, 1))
    save(fig, "test_error_vs_N_dimensions")


def probability_figures(records):
    dims = [8, 16, 32, 48, 64]
    colors = ["#27647B", "#B1813D", "#527B61", "#A55359", "#64717D"]
    styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
    markers = ["o", "^", "D", "s", "v"]
    for name, key, xkey, xlabel, ylabel in [
        ("step_size_probability_bounds", "step_lower", "alpha", r"$\alpha$", r"Lower bound for $q_{40}$"),
        ("iteration_probability_bounds", "time_lower", "T", r"$T$", r"Lower bound for $q_T$")
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3), sharey=True)
        for dom, ax in zip(["sphere", "ball"], axes):
            for d, color, ls, marker in zip(dims, colors, styles, markers):
                r = next(r for r in records if (r["domain"], r["d"]) == (dom, d))
                line(ax, r[xkey], r[key], (str(d), rf"$d={d}$", color, ls, marker),
                     every=max(1, len(r[xkey])//8))
            ax.set(title=dom.capitalize(), ylim=(0, 1.025), xlabel=xlabel)
            ax.set_xlim(left=0)
            ax.set_yticks([0, .5, 1])
            if xkey == "T":
                ax.set_xlim(0, 100)
                ax.axvline(40, color="#68727A", ls=":", lw=.7, alpha=.65)
            grid(ax)
        axes[0].set_ylabel(ylabel)
        shared_legend(fig, axes[0], ncol=5)
        fig.tight_layout(pad=.45, w_pad=.7, rect=(0, .13, 1, 1))
        save(fig, name)

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.65), sharey=True)
    for dom, ax in zip(["sphere", "ball"], axes):
        rr = [next(r for r in records if (r["domain"], r["d"]) == (dom, d)) for d in dims]
        line(ax, dims, [r["lower"] for r in rr],
             ("gd", r"GD: lower bound for $q_{40}(\tau)$", "#245E91", "--", "s"), log=True, every=1)
        for m, color, ls, marker in [(256, "#BC762C", ":", "^"),
                                      (4096, "#477A68", "-.", "D"),
                                      (10000, "#B44E49", "-", "v")]:
            line(ax, dims, [r["pool_upper"][str(m)] for r in rr],
                 (str(m), rf"Pool: upper bound for $P_{{{m}}}(\tau)$", color, ls, marker),
                 log=True, every=1)
        ax.set(xticks=dims, title=dom.capitalize(), xlabel=r"$d$")
        ax.set_ylim(top=1.5)
        grid(ax)
    axes[0].set_ylabel("Probability bound")
    shared_legend(fig, axes[0], ncol=2)
    fig.tight_layout(pad=.45, w_pad=.7, rect=(0, .19, 1, 1))
    save(fig, "probability_bounds_dimension")


def main():
    OUT.mkdir(exist_ok=True)
    AUDIT.mkdir(exist_ok=True)
    source_errors(read_json(SOURCE / "shared_test_errors.json"))
    source_history()
    boundary_errors()
    spectral_errors(read_json(SPECTRAL / "results.json"))
    probability_figures(read_json(SPECTRAL / "probability_theory.json"))
    assert len(EXPORTS) == 10
    (AUDIT / "export_manifest.json").write_text(json.dumps({
        "input_sha256": INPUTS,
        "font": f"Computer Modern, TeX-rendered, {FONT_PT} pt at native size",
        "figures": EXPORTS,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
