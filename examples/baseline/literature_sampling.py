"""Published sampling policies on the continuous-source P3 finite-element baseline.

Fresh random training sets: Cohen--Dahmen--DeVore--Nichols (2020), Sec. 4.
AEGA: Hesthaven--Stamm--Zhang (2014), Algorithm 3.
SMM: Jiang--Chen--Narayan (2017), Algorithm 1 and Sec. 3.1.

The three methods stop at the same indicator threshold. Exact full-sphere
error and greedy quality are audited outside the measured training time.

Run from the repository root:
  OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. .venv/bin/python \
    examples/baseline/literature_sampling.py
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
from examples.baseline.rom import a_orthonormal_append, matrices, true_worst

METHODS = ("fresh_random", "aega", "smm")


def score_pool(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Squared residual indicators for a finite training set."""
    return np.einsum("mi,ij,mj->m", points, H, points, optimize=True)


def aega_scan(
    H: np.ndarray,
    points: np.ndarray,
    saved: np.ndarray,
    saturation: float,
    tolerance: float,
) -> tuple[int, float, np.ndarray, int]:
    """Algorithm 3's saturation scan over the current active training set.

    The saved indicators may come from different earlier RB spaces. The
    saturation bound remains valid because the energy projection error
    decreases as the RB space grows.
    """
    order = np.argsort(-saved, kind="stable")
    best_index = -1
    best_squared = -1.0
    checked = np.zeros(len(points), dtype=bool)
    evaluations = 0
    for index in order:
        upper_squared = (saturation * saved[index]) ** 2
        if upper_squared <= best_squared:
            break
        value = max(float(points[index] @ H @ points[index]), 0.0)
        saved[index] = math.sqrt(value)
        checked[index] = True
        evaluations += 1
        if value > best_squared:
            best_squared = value
            best_index = int(index)
    remove = checked & (saved < tolerance)
    return best_index, best_squared, remove, evaluations


def replenish(
    rng: np.random.Generator,
    points: np.ndarray,
    saved: np.ndarray,
    remove: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Replace the active-set points marked as resolved in AEGA."""
    count = int(np.count_nonzero(remove))
    if count:
        points[remove] = uniform_sphere(rng, count, points.shape[1])
        saved[remove] = np.inf
    return points, saved, count


def smm_subset(
    points: np.ndarray,
    scores: np.ndarray,
    tolerance: float,
    size: int,
) -> np.ndarray:
    """SMM Section 3.1: select points at equispaced indicator levels."""
    values = np.sqrt(np.maximum(scores, 0.0))
    maximum = float(np.max(values))
    levels = tolerance + (maximum - tolerance) * np.arange(size) / size
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    # Treat round-off-scale differences as equal indicator levels.
    slack = 8 * np.finfo(float).eps * max(1.0, maximum)
    chosen = order[np.searchsorted(sorted_values, levels - slack, side="left")]
    return points[np.unique(chosen)].copy()


def audit_error(problem: FEMProblem, basis: np.ndarray) -> float:
    if basis.shape[1] == 0:
        return true_worst(problem.solution_gram)
    C = basis.T @ problem.B
    residual = problem.solution_gram - C.T @ C
    return true_worst(0.5 * (residual + residual.T))


def run(
    problem: FEMProblem,
    method: str,
    seed: int,
    *,
    pool_size: int = 10000,
    full_size: int = 10000,
    safety_size: int = 100000,
    target_rank: int | None = None,
    fixed_rank: bool = False,
) -> dict:
    """Train one reduced basis with a published parameter sampling policy."""
    if method not in METHODS:
        raise ValueError(method)
    d = problem.d
    target_rank = d // 2 if target_rank is None else target_rank
    max_rank = target_rank if fixed_rank else min(d, target_rank + 4)
    tolerance = problem.spectrum_ratio**target_rank
    rng = np.random.default_rng(seed + 1_000_003)
    basis = np.empty((problem.dofs, 0))
    timings = {
        "training_set_seconds": 0.0,
        "indicator_matrix_seconds": 0.0,
        "selection_seconds": 0.0,
        "enrichment_seconds": 0.0,
    }
    evaluations = 0
    safety_evaluations = 0
    refresh_count = 0
    stage = 0
    stage_error = math.inf
    selected_error = math.inf
    subset = None
    points = None
    saved = None
    saturation = math.sqrt((1.0 + problem.contrast) / (1.0 - problem.contrast))
    begin = time.perf_counter()
    if method == "aega":
        points = uniform_sphere(rng, pool_size, d)
    elif method == "smm":
        points = uniform_sphere(rng, full_size, d)
    timings["training_set_seconds"] += time.perf_counter() - begin
    curve = [audit_error(problem, basis)]
    cumulative_seconds = [sum(timings.values())]
    rounds = []

    # AEGA and SMM start with a random snapshot as in their source algorithms.
    if method in {"aega", "smm"}:
        begin = time.perf_counter()
        basis, novelty = a_orthonormal_append(basis, problem.S @ points[0], problem.A)
        timings["enrichment_seconds"] += time.perf_counter() - begin
        if method == "aega":
            saved = np.full(pool_size, np.inf)
        curve.append(audit_error(problem, basis))
        cumulative_seconds.append(sum(timings.values()))
        rounds.append(
            {
                "rank": 1,
                "indicator_evaluations": 0,
                "initial_random": True,
                "novelty": novelty,
            }
        )

    stop_reason = "rank_cap"
    consecutive_safe = 0
    safety_batches = math.ceil(safety_size / pool_size)
    while basis.shape[1] < max_rank:
        begin = time.perf_counter()
        H, _, _ = matrices(problem, basis)
        timings["indicator_matrix_seconds"] += time.perf_counter() - begin

        begin = time.perf_counter()
        new_evaluations = 0
        maintenance = {"full_scans": 0, "safety_batches": 0, "replaced": 0}
        if method == "fresh_random":
            candidates = uniform_sphere(rng, pool_size, d)
            scores = score_pool(H, candidates)
            index = int(np.argmax(scores))
            point = candidates[index]
            selected_squared = float(scores[index])
            new_evaluations = pool_size
        elif method == "aega":
            while True:
                index, selected_squared, remove, count = aega_scan(
                    H, points, saved, saturation, tolerance
                )
                new_evaluations += count
                if fixed_rank or selected_squared > tolerance**2:
                    point = points[index].copy()
                    points, saved, replaced = replenish(rng, points, saved, remove)
                    maintenance["replaced"] += replaced
                    consecutive_safe = 0
                    break
                consecutive_safe += 1
                maintenance["safety_batches"] += 1
                safety_evaluations += count
                if consecutive_safe >= safety_batches:
                    point = None
                    break
                points = uniform_sphere(rng, pool_size, d)
                saved = np.full(pool_size, np.inf)
        else:
            # Algorithm 1: periodically sweep the full fixed training set;
            # between global sweeps, use the indicator-level subset of Sec. 3.1.
            if subset is None or selected_error <= stage_error / (stage + 1):
                scores = score_pool(H, points)
                index = int(np.argmax(scores))
                point = points[index]
                selected_squared = float(scores[index])
                stage += 1
                refresh_count += 1
                maintenance["full_scans"] = 1
                new_evaluations = full_size
                stage_error = math.sqrt(max(selected_squared, 0.0))
                if selected_squared > tolerance**2:
                    subset = smm_subset(points, scores, tolerance, 20 * (stage + 1))
            else:
                scores = score_pool(H, subset)
                index = int(np.argmax(scores))
                point = subset[index]
                selected_squared = float(scores[index])
                new_evaluations = len(subset)
            selected_error = math.sqrt(max(selected_squared, 0.0))
        timings["selection_seconds"] += time.perf_counter() - begin
        evaluations += new_evaluations
        if point is None or (not fixed_rank and selected_squared <= tolerance**2):
            stop_reason = "sampled_indicator_below_tolerance"
            break

        # The independent exact maximization is an audit, not an algorithm input.
        optimum = float(la.eigvalsh(H, subset_by_index=[d - 1, d - 1])[0])
        quality = math.sqrt(max(selected_squared / optimum, 0.0))
        begin = time.perf_counter()
        basis, novelty = a_orthonormal_append(basis, problem.S @ point, problem.A)
        timings["enrichment_seconds"] += time.perf_counter() - begin
        curve.append(audit_error(problem, basis))
        cumulative_seconds.append(sum(timings.values()))
        rounds.append(
            {
                "rank": basis.shape[1],
                "indicator_evaluations": new_evaluations,
                "indicator_quality": quality,
                "novelty": novelty,
                **maintenance,
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
        "safety_evaluations": safety_evaluations,
        "full_scans": refresh_count,
        "curve": curve,
        "cumulative_training_seconds": cumulative_seconds,
        "rounds": rounds,
        "timings": timings,
    }


def summarize(runs: list[dict]) -> dict:
    def median(key: str) -> float:
        return float(np.median([run[key] for run in runs]))

    timing_keys = runs[0]["timings"]
    return {
        "replications": len(runs),
        "rank_median": median("rank"),
        "exact_worst_error_median": median("exact_worst_error"),
        "exact_worst_error_min": float(min(r["exact_worst_error"] for r in runs)),
        "exact_worst_error_max": float(max(r["exact_worst_error"] for r in runs)),
        "training_seconds_median": float(
            np.median([r["timings"]["total_training_seconds"] for r in runs])
        ),
        "indicator_evaluations_median": median("indicator_evaluations"),
        "success_fraction": float(
            np.mean([r["exact_worst_error"] <= r["target_tolerance"] for r in runs])
        ),
        "timings_median_seconds": {
            key: float(np.median([run["timings"][key] for run in runs]))
            for key in timing_keys
        },
    }


def plot(data: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4), constrained_layout=True)
    styles = {
        "fresh_random": ("Fresh random", "#009E73", "o-"),
        "aega": ("AEGA", "#D55E00", "s--"),
        "smm": ("SMM", "#0072B2", "^:"),
    }
    for d, case in data["cases"].items():
        for method in METHODS:
            runs = case["methods"][method]["runs"]
            label, color, style = styles[method]
            cap = max(len(run["curve"]) for run in runs)
            padded = [
                np.pad(
                    run["curve"],
                    (0, cap - len(run["curve"])),
                    constant_values=run["curve"][-1],
                )
                for run in runs
            ]
            curve = np.median(padded, axis=0)
            if d == "32":
                axes[0].semilogy(
                    np.arange(cap),
                    curve,
                    style,
                    color=color,
                    label=label,
                    markevery=max(1, cap // 8),
                )
            axes[1].scatter(
                case["methods"][method]["training_seconds_median"],
                case["methods"][method]["exact_worst_error_median"],
                marker=style[0],
                color=color,
                s=50,
            )
            if d in {"16", "32", "64"}:
                axes[1].annotate(
                    f"{label}, d={d}",
                    (
                        case["methods"][method]["training_seconds_median"],
                        case["methods"][method]["exact_worst_error_median"],
                    ),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )
    axes[0].set(xlabel="RB dimension (d=32)", ylabel="Exact full-sphere worst error")
    axes[1].set(
        xlabel="Median training time (s)",
        ylabel="Median exact full-sphere worst error",
        yscale="log",
    )
    for ax in axes:
        ax.grid(alpha=0.25)
    if "32" in data["cases"]:
        axes[0].legend(fontsize=8)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--grid", type=int, default=40)
    parser.add_argument("--pool", type=int, default=10000)
    parser.add_argument("--full", type=int, default=10000)
    parser.add_argument("--safety", type=int, default=100000)
    parser.add_argument(
        "--output",
        type=Path,
        default=(ROOT / "runs/report_baseline").joinpath("literature_sampling_fe_p3_m10000.json"),
    )
    args = parser.parse_args()
    data = {
        "setup": {
            "problem": "continuous-source P3 FEM baseline",
            "grid": args.grid,
            "dimensions": args.dimensions,
            "seeds": args.seeds,
            "parameter_domain": "Euclidean unit sphere",
            "target_indicator_tolerance": "rho^(d/2), rho=0.78",
            "rank_cap": "d/2+4",
            "fresh_random_pool": args.pool,
            "aega_active_set": args.pool,
            "aega_safety_draws": args.safety,
            "aega_saturation": "sqrt((1+contrast)/(1-contrast)), contrast=0.25",
            "smm_full_training_set": args.full,
            "smm_surrogate_size": "20*(outer_loop+1)",
            "smm_damping": 1,
            "timing": "single CPU process; FE setup and exact audits excluded from training",
            "sources": {
                "fresh_random": "Cohen et al., ESAIM M2AN 54 (2020), Sec. 4",
                "aega": "Hesthaven et al., ESAIM M2AN 48 (2014), Algorithm 3",
                "smm": "Jiang et al., J. Sci. Comput. 73 (2017), Algorithm 1 and Sec. 3.1",
            },
        },
        "cases": {},
    }
    for d in args.dimensions:
        begin = time.perf_counter()
        problem = FEMProblem.build(args.grid, d, element_order=3)
        # Warm the common residual assembly before timing any method.
        matrices(problem, np.empty((problem.dofs, 0)))
        setup_seconds = time.perf_counter() - begin
        case = {
            "hf_setup_seconds": setup_seconds,
            "hf_dofs": problem.dofs,
            "target_rank": d // 2,
            "target_tolerance": problem.spectrum_ratio ** (d // 2),
            "methods": {},
        }
        for method in METHODS:
            runs = [
                run(
                    problem,
                    method,
                    1000 + seed,
                    pool_size=args.pool,
                    full_size=args.full,
                    safety_size=args.safety,
                )
                for seed in range(args.seeds)
            ]
            summary = summarize(runs)
            case["methods"][method] = {**summary, "runs": runs}
            print(
                f"d={d:2d} {method:13s} rank={summary['rank_median']:4.1f} "
                f"worst={summary['exact_worst_error_median']:.6g} "
                f"hit={summary['success_fraction']:.2f} "
                f"eval={summary['indicator_evaluations_median']:.0f} "
                f"train={summary['training_seconds_median']:.3f}s",
                flush=True,
            )
        data["cases"][str(d)] = case
        args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    plot(data, args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
