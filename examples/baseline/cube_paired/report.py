"""Summarize saved paired-cube experiments without rerunning a search.

The fixed-space probabilities use the independent starts in frozen.json.
The RB curves describe separate complete greedy runs and are never treated
as independent samples of the fixed-space probability.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "runs" / "cube_paired_rb_20261104"
METHODS = ("gd", "pool256", "pool4096", "pool10000")
STYLES = {
    "gd": ("GD", "#205B7B", "-", "o"),
    "pool256": ("Pool 256", "#B58B3E", "--", "s"),
    "pool4096": ("Pool 4096", "#4B8067", "-.", "^"),
    "pool10000": ("Pool 10000", "#A45B55", ":", "D"),
}
MODES = {
    "paired_prefix": "Prescribed paired prefix",
    "empty_start": "Greedy from the empty space",
}


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_median(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return median(values) if values else None


def curve_summary(curves: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    seen = set()
    for row in curves:
        key = (row["mode"], int(row["d"]), row["method"], int(row["seed"]), int(row["N"]))
        if key in seen:
            raise ValueError(f"Duplicate saved RB point: {key}")
        seen.add(key)
        if row["method"] not in STYLES or row["mode"] not in MODES:
            raise ValueError(f"Unknown method or mode: {key}")
        error, width = float(row["test_worst_error"]), float(row["width"])
        if not math.isfinite(error) or error < 0 or not math.isfinite(width) or width < 0:
            raise ValueError(f"Invalid error or width at {key}")
        groups[(row["mode"], int(row["d"]), row["method"], int(row["N"]))].append(row)
    result = []
    for (mode, d, method, N), rows in sorted(groups.items()):
        errors = [float(r["test_worst_error"]) for r in rows]
        # A width of zero makes E_N/d_N undefined, including when E_N is zero.
        ratios = [float(r["test_worst_error"]) / float(r["width"])
                  for r in rows if float(r["width"]) > 0]
        result.append({
            "mode": mode, "d": d, "method": method, "N": N,
            "seed_count": len(rows), "median_test_worst_error": median(errors),
            "minimum_test_worst_error": min(errors), "maximum_test_worst_error": max(errors),
            "width": finite_median(r["width"] for r in rows),
            "median_error_ratio": finite_median(ratios),
        })
    return result


def final_summary(curves: list[dict], timings: list[dict]) -> list[dict]:
    by_run = defaultdict(list)
    for row in curves:
        by_run[(row["mode"], int(row["d"]), row["method"], int(row["seed"]))].append(row)
    timing_map = {}
    for row in timings:
        key = (row["mode"], int(row["d"]), row["method"], int(row["seed"]))
        if key in timing_map:
            raise ValueError(f"Duplicate timing record: {key}")
        timing_map[key] = row
    if set(by_run) != set(timing_map):
        raise ValueError("RB runs and timing records do not identify the same runs")
    common_N = {}
    for mode, d in sorted({(k[0], k[1]) for k in by_run}):
        rank_sets = [{int(r["N"]) for r in rows}
                     for k, rows in by_run.items() if k[:2] == (mode, d)]
        shared = set.intersection(*rank_sets)
        if not shared:
            raise ValueError(f"No common saved N for mode={mode}, d={d}")
        common_N[(mode, d)] = max(shared)
    groups = defaultdict(list)
    for key, rows in by_run.items():
        mode, d, method, seed = key
        final = max(rows, key=lambda r: int(r["N"]))
        timing = timing_map[key]
        if int(timing["final_N"]) != int(final["N"]):
            raise ValueError(f"Final N disagrees between curves and timings: {key}")
        shared = next(r for r in rows if int(r["N"]) == common_N[(mode, d)])
        groups[(mode, d, method)].append((final, shared, timing))
    result = []
    for (mode, d, method), rows in sorted(groups.items()):
        final_Ns = [int(r[0]["N"]) for r in rows]
        shared_ratios = [float(r[1]["test_worst_error"]) / float(r[1]["width"])
                         for r in rows if float(r[1]["width"]) > 0]
        final_ratios = [float(r[0]["test_worst_error"]) / float(r[0]["width"])
                        for r in rows if float(r[0]["width"]) > 0]
        result.append({
            "mode": mode, "d": d, "method": method, "seed_count": len(rows),
            "final_N_min": min(final_Ns), "final_N_max": max(final_Ns),
            "median_final_test_worst_error": finite_median(r[0]["test_worst_error"] for r in rows),
            "median_final_error_ratio": finite_median(final_ratios),
            "common_N": common_N[(mode, d)],
            "median_common_N_test_worst_error": finite_median(r[1]["test_worst_error"] for r in rows),
            "common_N_width": finite_median(r[1]["width"] for r in rows),
            "median_common_N_error_ratio": finite_median(shared_ratios),
            "median_train_seconds": finite_median(r[2]["train_seconds"] for r in rows),
            "minimum_train_seconds": min(float(r[2]["train_seconds"]) for r in rows),
            "maximum_train_seconds": max(float(r[2]["train_seconds"]) for r in rows),
            "stopped_reasons": "; ".join(sorted({str(r[2].get("stopped_reason") or "completed") for r in rows})),
        })
    return result


def frozen_summary(frozen: list[dict]) -> list[dict]:
    result, seen = [], set()
    for row in sorted(frozen, key=lambda r: int(r["d"])):
        d, starts = int(row["d"]), int(row["starts"])
        if d in seen:
            raise ValueError(f"More than one fixed-space record for d={d}")
        seen.add(d)
        if starts <= 0:
            raise ValueError("Fixed-space start count must be positive")
        for name in ("gd_q", "gd_ci_low", "gd_ci_high", "gd_lower", "pool_upper_81", "vertex_pool_exact_81"):
            if not math.isfinite(float(row[name])) or not 0 <= float(row[name]) <= 1:
                raise ValueError(f"Invalid probability {name}, d={d}")
        q = int(row["gd_hits"]) / starts
        if not math.isclose(q, float(row["gd_q"]), rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"Hit count disagrees with probability for d={d}")
        if int(row["T"]) != 80:
            raise ValueError("The agreed pool fields use M=81; update labels if the experiment changes T")
        result.append({
            "d": d, "N": d // 2, "starts": starts, "T": int(row["T"]),
            "alpha": float(row["alpha"]), "gamma": float(row["gamma"]),
            "gd_hits": int(row["gd_hits"]), "gd_q": q,
            "gd_ci_low": float(row["gd_ci_low"]), "gd_ci_high": float(row["gd_ci_high"]),
            "initial_hits": int(row["initial_hits"]),
            "initial_q": int(row["initial_hits"]) / starts,
            "gd_lower": float(row["gd_lower"]),
            "interior_pool_upper_M81": float(row["pool_upper_81"]),
            "vertex_pool_exact_M81": float(row["vertex_pool_exact_81"]),
            "mean_final_J": float(row["mean_final_J"]),
            "min_final_J": float(row["min_final_J"]),
            "wall_seconds": float(row["wall_seconds"]),
        })
    return result


def save_figure(fig, out: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def first_search_summary(choices: list[dict]) -> list[dict]:
    """Compare searches in exactly the same paired-prefix space."""
    groups = defaultdict(list)
    for row in choices:
        if row["mode"] == "paired_prefix" and int(row["n"]) == int(row["d"])//2:
            groups[(int(row["d"]), row["method"])].append(float(row["selected_J"]))
    return [dict(d=d, N=d//2, method=method, seed_count=len(values),
                 median_selected_J=median(values),
                 median_selected_indicator_ratio=median(math.sqrt(max(v,0.)) for v in values),
                 minimum_selected_J=min(values), maximum_selected_J=max(values))
            for (d,method), values in sorted(groups.items())]


def probability_figure(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    d = np.asarray([r["d"] for r in rows])
    zero_notes = []
    series = (
        ("gd_q", "GD: empirical", "#205B7B", "-", "o"),
        ("gd_lower", "GD: lower bound", "#205B7B", "--", "s"),
        ("interior_pool_upper_M81", "Interior pool, M=81: upper bound", "#B58B3E", "--", "^"),
        ("vertex_pool_exact_M81", "Vertex pool, M=81: exact probability", "#A45B55", "-.", "D"),
    )
    for key, label, color, style, marker in series:
        values = np.asarray([r[key] for r in rows], dtype=float)
        positive = values > 0
        displayed = np.where(positive, values, np.nan)
        ax.plot(d, displayed, color=color, linestyle=style, marker=marker,
                linewidth=1.7, markersize=5, label=label)
        if np.any(~positive):
            zero_notes.append(f"{label}: d=" + ",".join(str(x) for x in d[~positive]))
        if key == "gd_q":
            # Draw only positive portions of supplied confidence intervals.
            # A true zero is never replaced by machine precision.
            for x, q, r in zip(d, values, rows):
                low, high = r["gd_ci_low"], r["gd_ci_high"]
                if q > 0 and low > 0 and high >= q >= low:
                    ax.errorbar(x, q, yerr=[[q-low], [high-q]], fmt="none",
                                ecolor=color, capsize=3, linewidth=1)
                elif q == 0 and high > 0:
                    ax.plot(x, high, marker="v", fillstyle="none", color=color,
                            linestyle="none", label="GD: upper confidence limit (zero hits)")
    ax.set_yscale("log")
    ax.set_xlabel("Parameter dimension d")
    ax.set_ylabel("Probability of reaching the threshold")
    ax.set_xticks(d)
    ax.set_title(r"Fixed reduced space: $N=d/2$, $J\geq\gamma^2\max J$")
    ax.set_ylim(top=1.4)
    ax.grid(axis="y", which="major", alpha=.2)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=9, loc="best")
    values_T = sorted({r["T"] for r in rows})
    values_alpha = sorted({r["alpha"] for r in rows})
    values_gamma = sorted({r["gamma"] for r in rows})
    detail = (f"T={values_T[0] if len(values_T)==1 else values_T}; "
              f"alpha={values_alpha[0] if len(values_alpha)==1 else values_alpha}; "
              f"gamma={values_gamma[0] if len(values_gamma)==1 else values_gamma}. "
              "GD error bars: supplied confidence intervals for independent starts.")
    if zero_notes:
        detail += "\nZero values omitted on the logarithmic axis: " + "; ".join(zero_notes) + "."
    fig.text(.5, .01, detail, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, .065 if not zero_notes else .10, 1, 1))
    save_figure(fig, out, "fixed_space_probabilities")


def error_figure(rows: list[dict], mode: str, out: Path) -> None:
    selected = [r for r in rows if r["mode"] == mode]
    dims = sorted({r["d"] for r in selected})
    if not dims:
        return
    ncols, nrows = min(2, len(dims)), math.ceil(len(dims) / 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0*ncols, 3.8*nrows), squeeze=False)
    zeros_omitted = False
    for ax, d in zip(axes.flat, dims):
        for method in METHODS:
            part = sorted((r for r in selected if r["d"] == d and r["method"] == method),
                          key=lambda r: r["N"])
            if not part:
                continue
            label, color, style, marker = STYLES[method]
            N = np.asarray([r["N"] for r in part])
            errors = np.asarray([r["median_test_worst_error"] for r in part])
            zeros_omitted |= bool(np.any(errors == 0))
            ax.plot(N, np.where(errors > 0, errors, np.nan), label=label,
                    color=color, linestyle=style, linewidth=1.5, marker=marker,
                    markersize=3.8, markevery=max(1, len(N)//10))
        widths = {}
        for r in selected:
            if r["d"] == d:
                widths[int(r["N"])] = float(r["width"])
        width_N = np.asarray(sorted(widths))
        width_values = np.asarray([widths[n] for n in width_N])
        ax.plot(width_N, np.where(width_values > 0, width_values, np.nan),
                color="#333333", linestyle=(0, (5, 3)), linewidth=1.2,
                label=r"Exact width $d_N$")
        ax.set_yscale("log")
        ax.set_title(f"d = {d}")
        ax.set_xlabel("Reduced-space dimension N")
        ax.set_ylabel("Maximum test energy error")
        ax.set_xticks(np.linspace(int(width_N.min()), int(width_N.max()), 5, dtype=int))
        ax.grid(axis="y", which="major", alpha=.18)
    for ax in list(axes.flat)[len(dims):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(.5, .035))
    fig.suptitle(MODES[mode], fontsize=13)
    detail = "Pointwise medians over saved seeds; the same test set is used for every method at each d."
    if zeros_omitted:
        detail += " Zero errors are omitted on the logarithmic axis."
    fig.text(.5, .012, detail, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .95))
    save_figure(fig, out, f"rb_test_error_vs_N_{mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT,
                        help="Directory containing the completed run JSON files")
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    required = ("settings.json", "validation.json", "frozen.json", "rb_curves.json", "timings.json")
    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        raise FileNotFoundError("Run data are not complete: " + ", ".join(missing))
    settings = read_json(out / "settings.json")
    validation = read_json(out / "validation.json")
    frozen = frozen_summary(read_json(out / "frozen.json"))
    curves = read_json(out / "rb_curves.json")
    timings = read_json(out / "timings.json")
    expected_runs = len(settings["dimensions"])*len(settings["seeds"])*len(settings["modes"])*len(METHODS)
    if len(timings) != expected_runs:
        raise ValueError(f"Run incomplete: {len(timings)} / {expected_runs} RB runs saved")
    summaries = curve_summary(curves)
    finals = final_summary(curves, timings)
    if not frozen or not summaries:
        raise ValueError("Empty fixed-space or RB results; wait for the experiment to finish")
    write_csv(out / "summary.csv", summaries)
    write_csv(out / "frozen_summary.csv", frozen)
    write_csv(out / "final_summary.csv", finals)
    write_json(out / "final_summary.json", finals)
    first_search = first_search_summary(read_json(out/"selected_parameters.json"))
    write_csv(out/"first_search_summary.csv", first_search)
    write_json(out/"first_search_summary.json", first_search)
    write_json(out / "summary.json", {
        "settings": settings, "validation": validation,
        "definitions": {
            "fixed_space_probability": "Independent starts on one prescribed space V_(d/2), not greedy-round frequencies.",
            "test_error": "Maximum energy error on the saved common test set, not a certified supremum unless the test set is exhaustive.",
            "common_N": "Largest saved N present in every run of every method at the same d and initialization mode.",
            "zero_width": "E_N/d_N is undefined when d_N=0 and is saved as null, not zero.",
            "timing": "Train time as recorded by run.py; test evaluation is governed by that runner's timing boundary.",
        },
        "record_counts": {"frozen": len(frozen), "rb_curves": len(curves), "timings": len(timings)},
        "frozen": frozen, "curves": summaries, "final": finals,
        "first_paired_prefix_search": first_search,
    })
    probability_figure(frozen, out)
    for mode in MODES:
        error_figure(summaries, mode, out)
    print(f"Saved paired-cube summaries and plots to {out}")
    for r in finals:
        ratio = r["median_common_N_error_ratio"]
        ratio_text = "undefined" if ratio is None else f"{ratio:.6g}"
        print(f"{r['mode']:13s} d={r['d']:3d} {r['method']:9s} "
              f"common N={r['common_N']:3d}, test E/width={ratio_text}, "
              f"time={r['median_train_seconds']:.4g}s")


if __name__ == "__main__":
    main()
