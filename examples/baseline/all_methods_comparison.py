"""Fixed-rank or threshold-stopped comparison of RB selectors.

All methods use the same P3 finite-element problem. Fixed-rank mode builds
exactly d/2 snapshots; the original threshold-stopped mode remains available.
Exact full-sphere errors are independent post-selection audits.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from contgreedy.sampling.laws import uniform_sphere
from examples.baseline.continuous_fem import FEMProblem
from examples.baseline.rom import a_orthonormal_append, matrices
from examples.baseline.sampling import normalized_gd_search
from examples.baseline.literature_sampling import (
    METHODS as LITERATURE_METHODS,
)
from examples.baseline.literature_sampling import audit_error, summarize
from examples.baseline.literature_sampling import run as run_literature
from examples.baseline.strong_baselines import select, sobol_sphere

STANDARD_METHODS = (
    "dense_eigh",
    "lanczos",
    "gd100",
    "lbfgs100",
    "sobol8192",
    "fixed_random10000",
)
METHODS = (*STANDARD_METHODS, *LITERATURE_METHODS)


def run_standard(
    problem: FEMProblem,
    method: str,
    seed: int,
    *,
    pool_size: int = 10000,
    sobol_size: int = 8192,
    target_rank: int | None = None,
    fixed_rank: bool = False,
) -> dict:
    if method not in STANDARD_METHODS:
        raise ValueError(method)
    d = problem.d
    target_rank = d // 2 if target_rank is None else target_rank
    max_rank = target_rank if fixed_rank else min(d, target_rank + 4)
    tolerance = problem.spectrum_ratio**target_rank
    basis = np.empty((problem.dofs, 0))
    rng = np.random.default_rng(seed)
    pool_rng = np.random.default_rng(seed + 1_000_003)
    timings = {
        "training_set_seconds": 0.0,
        "indicator_matrix_seconds": 0.0,
        "selection_seconds": 0.0,
        "enrichment_seconds": 0.0,
    }
    pool = None
    begin = time.perf_counter()
    if method == "sobol8192":
        pool = sobol_sphere(d, sobol_size, seed)
    elif method == "fixed_random10000":
        pool = uniform_sphere(pool_rng, pool_size, d)
    timings["training_set_seconds"] += time.perf_counter() - begin
    curve = [audit_error(problem, basis)]
    cumulative_seconds = [sum(timings.values())]
    evaluations = 0
    rounds = []
    stop_reason = "rank_cap"
    for _ in range(max_rank):
        begin = time.perf_counter()
        H, _, _ = matrices(problem, basis)
        timings["indicator_matrix_seconds"] += time.perf_counter() - begin
        begin = time.perf_counter()
        if method == "sobol8192":
            point, statistics = select(H, "sobol4096", None, pool)
        elif method == "fixed_random10000":
            point, statistics = select(H, "fixed_random4096", None, pool)
        elif method == "fresh_random":
            raise AssertionError("Fresh random is handled by the literature runner")
        else:
            point, statistics = select(H, method, rng.standard_normal(d), None)
        timings["selection_seconds"] += time.perf_counter() - begin
        count = statistics.get("h_actions")
        if count is not None:
            evaluations += int(count)
        selected_squared = max(float(point @ H @ point), 0.0)
        if not fixed_rank and selected_squared <= tolerance**2:
            stop_reason = "sampled_indicator_below_tolerance"
            break
        # The following is an audit and does not influence selection.
        exact_indicator = max(float(la.eigvalsh(H)[-1]), 0.0)
        quality = math.sqrt(selected_squared / exact_indicator)
        begin = time.perf_counter()
        basis, novelty = a_orthonormal_append(basis, problem.S @ point, problem.A)
        timings["enrichment_seconds"] += time.perf_counter() - begin
        curve.append(audit_error(problem, basis))
        cumulative_seconds.append(sum(timings.values()))
        rounds.append(
            {
                "rank": basis.shape[1],
                "indicator_evaluations": count,
                "indicator_quality": quality,
                "novelty": novelty,
            }
        )
    timings["total_training_seconds"] = sum(timings.values())
    return {
        "method": method,
        "seed": seed,
        "rank": basis.shape[1],
        "target_tolerance": tolerance,
        "stop_reason": stop_reason,
        "exact_worst_error": curve[-1],
        "indicator_evaluations": evaluations,
        "curve": curve,
        "cumulative_training_seconds": cumulative_seconds,
        "rounds": rounds,
        "timings": timings,
    }


def plot(data: dict, path: Path) -> None:
    colors = {
        "dense_eigh": "#222222",
        "lanczos": "#777777",
        "gd100": "#0072B2",
        "lbfgs100": "#56B4E9",
        "sobol8192": "#CC79A7",
        "fixed_random10000": "#E69F00",
        "fresh_random": "#009E73",
        "aega": "#D55E00",
        "smm": "#6A3D9A",
    }
    markers = {
        "dense_eigh": "x",
        "lanczos": "+",
        "gd100": "o",
        "lbfgs100": "^",
        "sobol8192": "D",
        "fixed_random10000": "s",
        "fresh_random": "v",
        "aega": "P",
        "smm": "*",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1), constrained_layout=True)
    for ax, d in zip(axes, data["setup"]["dimensions"]):
        case = data["cases"][str(d)]
        for method in data["setup"]["methods"]:
            result = case["methods"][method]
            ax.scatter(
                result["training_seconds_median"],
                result["exact_worst_error_median"],
                color=colors[method],
                marker=markers[method],
                s=70,
                label=method.replace("_", " "),
            )
        if not data["setup"].get("fixed_rank", False):
            ax.axhline(
                case["target_tolerance"],
                color="#444444",
                linestyle=":",
                linewidth=1.2,
            )
        ax.set(
            title=f"$d={d}$",
            xlabel="Median training time (s)",
            ylabel="Median exact worst error",
            yscale="log",
        )
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=8)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--grid", type=int, default=40)
    parser.add_argument("--pool", type=int, default=10000)
    parser.add_argument("--sobol", type=int, default=8192)
    parser.add_argument("--safety", type=int, default=100000)
    parser.add_argument("--fixed-rank", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.output is None:
        filename = ("fixed_rank_four_methods_fe_p3_m10000.json"
                    if args.fixed_rank else "all_methods_fe_p3_m10000.json")
        args.output = (ROOT / "runs/report_baseline").joinpath(filename)
    data = {
        "setup": {
            "problem": "continuous-source P3 FEM baseline",
            "grid": args.grid,
            "dimensions": args.dimensions,
            "seeds": args.seeds,
            "parameter_domain": "Euclidean unit sphere",
            "fixed_rank": args.fixed_rank,
            "target_indicator_tolerance": None if args.fixed_rank else "0.78^(d/2)",
            "aega_replenishment_threshold": "0.78^(d/2)",
            "rank_cap": "d/2" if args.fixed_rank else "d/2+4",
            "random_training_set_size": args.pool,
            "sobol_training_set_size": args.sobol,
            "aega_safety_draws": None if args.fixed_rank else args.safety,
            "smm_surrogate_size": "20*(outer_loop+1)",
            "timing": "single CPU, one thread; common FE setup and exact audits excluded",
            "methods": list(args.methods),
        },
        "cases": {},
    }
    for d in args.dimensions:
        begin = time.perf_counter()
        problem = FEMProblem.build(args.grid, d, element_order=3)
        H0, _, _ = matrices(problem, np.empty((problem.dofs, 0)))
        warm_start = np.ones(d)
        for warm_method in ("dense_eigh", "lanczos", "lbfgs100"):
            select(H0, warm_method, warm_start, None)
        normalized_gd_search(np.eye(d), np.ones(d), 100, 1.0)
        setup_seconds = time.perf_counter() - begin
        case = {
            "hf_setup_seconds": setup_seconds,
            "hf_dofs": problem.dofs,
            "target_rank": d // 2,
            "target_tolerance": problem.spectrum_ratio ** (d // 2),
            "methods": {},
        }
        for method in args.methods:
            if method in STANDARD_METHODS:
                runs = [
                    run_standard(
                        problem,
                        method,
                        1000 + seed,
                        pool_size=args.pool,
                        sobol_size=args.sobol,
                        fixed_rank=args.fixed_rank,
                    )
                    for seed in range(args.seeds)
                ]
            else:
                runs = [
                    run_literature(
                        problem,
                        method,
                        1000 + seed,
                        pool_size=args.pool,
                        full_size=args.pool,
                        safety_size=args.safety,
                        fixed_rank=args.fixed_rank,
                    )
                    for seed in range(args.seeds)
                ]
            case["methods"][method] = {**summarize(runs), "runs": runs}
            result = case["methods"][method]
            print(
                f"d={d:2d} {method:19s} N={result['rank_median']:4.1f} "
                f"E={result['exact_worst_error_median']:.6g} "
                f"hit={result['success_fraction']:.2f} "
                f"eval={result['indicator_evaluations_median']:.0f} "
                f"train={result['training_seconds_median']:.3f}s",
                flush=True,
            )
            data["cases"][str(d)] = case
            args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    plot(data, args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
