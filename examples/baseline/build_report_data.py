"""Reproduce the GD--100, exact-oracle, and pool comparison.

Run from the repository root:
  PYTHONPATH=src:. python examples/baseline/build_report_data.py
  PYTHONPATH=src:. python examples/baseline/build_report_data.py --hf-solver legacy_fd --grid 48
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import betainc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from examples.baseline.hf import Problem
from examples.baseline.rom import a_orthonormal_append, matrices
from examples.baseline.run import run_greedy

REPORT_DIR = ROOT / "runs/report_baseline"
FIG_DIR = ROOT / "paper/report_baseline"
DIMENSIONS = (8, 16, 32, 64)
GRID = 48
RESULTS_FILE = REPORT_DIR / "grid48_results.json"
FE_GRID = 40
FE_RESULTS_FILE = REPORT_DIR / "fe_p3_grid40_results.json"
METHODS = (
    ("exact_indicator", "Exact indicator", "#222222", ":", "x"),
    ("normalized_gd_T100", "GD--100", "#0072B2", "-", "o"),
    ("pool_M256", "Pool--256", "#E69F00", "-.", "s"),
    ("pool_M4096", "Pool--4096", "#D55E00", "--", "D"),
    ("pool_M10000", "Pool--10,000", "#009E73", (0, (3, 1, 1, 1)), "v"),
)


def exact_path_geometry(problem: Problem, rank: int, steps: int) -> dict:
    """Certified p upper and GD q lower on the frozen exact-indicator path."""
    d = problem.d
    a = 0.9**2
    basis = np.empty((problem.A.shape[0], 0))
    rows = []
    for n in range(rank):
        H, _, _ = matrices(problem, basis)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        lam1 = float(eigenvalues[-1])
        lam2 = float(eigenvalues[-2])
        r2 = max(lam2 / lam1, 0.0)
        xi = lam1 / float(np.linalg.norm(H, ord="fro"))
        ratio = (1.0 + xi * r2) / (1.0 + xi)
        amplification = ratio ** (2 * steps)
        threshold_gd = a * amplification / (1.0 - a + a * amplification)
        failure_upper = float(betainc(0.5, (d - 1) / 2, threshold_gd))
        q_lower = 1.0 - failure_upper
        if r2 < a:
            threshold_pool = (a - r2) / (1.0 - r2)
            p_upper = float(betainc((d - 1) / 2, 0.5, 1.0 - threshold_pool))
        else:
            p_upper = 1.0
        rows.append(
            {
                "n": n,
                "lambda_1": lam1,
                "lambda_2_over_lambda_1": r2,
                "xi": xi,
                "p_upper": p_upper,
                "gd100_q_lower": q_lower,
                "gd100_failure_upper": failure_upper,
            }
        )
        basis, _ = a_orthonormal_append(
            basis, problem.S @ eigenvectors[:, -1], problem.A
        )
    return {
        "gamma": 0.9,
        "path": "exact-indicator greedy, one deterministic path",
        "minimum_gd100_q_lower": min(row["gd100_q_lower"] for row in rows),
        "maximum_gd100_failure_upper": max(row["gd100_failure_upper"] for row in rows),
        "median_p_upper": float(np.median([row["p_upper"] for row in rows])),
        "rows": rows,
    }


def generate_all(
    *,
    hf_solver: str = "continuous_fe",
    grid: int = FE_GRID,
    element_order: int = 3,
    results_file: Path = FE_RESULTS_FILE,
) -> dict:
    output = {
        "hf_solver": hf_solver,
        "grid": grid,
        "element_order": element_order if hf_solver == "continuous_fe" else None,
        "contrast": 0.25,
        "spectrum_ratio": 0.78,
        "steps": 100,
        "learning_rate": 1.0,
        "normalization": "Frobenius norm of the current residual matrix H",
        "return_policy": "best iterate among t=0,...,100",
        "cases": {},
    }
    for d in DIMENSIONS:
        if hf_solver == "continuous_fe":
            from examples.baseline.continuous_fem import FEMProblem

            problem = FEMProblem.build(grid, d, 0.25, 0.78, element_order=element_order)
        elif hf_solver == "legacy_fd":
            problem = Problem.build(grid, d, 0.25, 0.78, "controlled")
        else:
            raise ValueError(f"Unknown HF solver: {hf_solver}")
        rank = d // 2
        count = 12 if d == 32 else 8
        # The first call pays one-time library/setup overhead. Warm up
        # the complete rank path before recording method wall times.
        run_greedy(problem, "exact_indicator", rank, 999)
        methods = {}
        method_specs = [
            ("exact_indicator", "exact_indicator", None),
            ("normalized_gd_T100", "normalized_gd", None),
            ("pool_M256", "pool", 256),
            ("pool_M4096", "pool", 4096),
        ]
        if d > 8:
            method_specs.append(("pool_M10000", "pool", 10_000))
        for key, method, pool_size in method_specs:
            seeds = (
                range(1000, 1003)
                if key == "exact_indicator"
                else range(1000, 1000 + count)
            )
            runs = []
            for seed in seeds:
                kwargs = {"steps": 100, "adam_lr": 1.0}
                if pool_size is not None:
                    kwargs["pool_size"] = pool_size
                result = run_greedy(problem, method, rank, seed, **kwargs)
                runs.append(
                    {
                        "seed": seed,
                        "elapsed_seconds": result["elapsed_seconds"],
                        "rows": [
                            {
                                "n": row["n"],
                                "worst_true": row["worst_true"],
                                "rho": row.get("rho"),
                                "gd_failure_upper": row.get("gd_failure_upper"),
                                "direct_hit_p_upper": row.get("direct_hit_p_upper"),
                            }
                            for row in result["rows"]
                        ],
                    }
                )
            med = np.median(
                np.array([[row["worst_true"] for row in run["rows"]] for run in runs]),
                axis=0,
            )
            methods[key] = {
                "seeds": len(runs),
                "worst_median_by_n": med.tolist(),
                "wall_time_median_seconds": float(
                    np.median([run["elapsed_seconds"] for run in runs])
                ),
                "runs": runs,
            }
            print(
                f"d={d}, {key}: final median E={med[-1]:.9g}, "
                f"median wall={methods[key]['wall_time_median_seconds']:.3f}s",
                flush=True,
            )
        gd_runs = methods["normalized_gd_T100"]["runs"]
        search_rows = [
            row
            for run in gd_runs
            for row in run["rows"]
            if row["gd_failure_upper"] is not None
        ]
        output["cases"][str(d)] = {
            "rank": rank,
            "seeds": count,
            "hf_dofs": problem.A.shape[0],
            "discrete_singular_values": problem.singular_values.tolist(),
            "continuous_width_at_rank": 0.78**rank
            if hf_solver == "continuous_fe"
            else None,
            "discrete_width_at_rank": float(problem.singular_values[rank]),
            "max_relative_energy_projection_error": (
                problem.projection_error if hf_solver == "continuous_fe" else None
            ),
            "methods": methods,
            "maximum_gd_failure_upper_on_runs": max(
                row["gd_failure_upper"] for row in search_rows
            ),
            "maximum_direct_hit_p_upper_on_runs": max(
                row["direct_hit_p_upper"] for row in search_rows
            ),
            "exact_path_geometry": exact_path_geometry(problem, rank, 100),
        }
        results_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def make_plot(
    gd: dict, *, output_prefix: Path = REPORT_DIR / "fe_p3_grid40_largest_error_vs_rank"
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), sharex=False)
    for ax, d in zip(axes.flat, DIMENSIONS):
        n = np.arange(d // 2 + 1)
        marker_stride = max(2, len(n) // 7)
        widths = (
            gd["cases"][str(d)]["discrete_singular_values"][: len(n)]
            if gd.get("hf_solver") == "continuous_fe"
            else 0.78**n
        )
        ax.semilogy(
            n,
            widths,
            color="#666666",
            linestyle=(0, (5, 2)),
            marker="^",
            markersize=4.3,
            markerfacecolor="white",
            markevery=(0, marker_stride),
            lw=1.4,
            label="Kolmogorov width",
        )
        for j, (key, label, color, style, marker) in enumerate(METHODS):
            if key not in gd["cases"][str(d)]["methods"]:
                continue
            error = gd["cases"][str(d)]["methods"][key]["worst_median_by_n"]
            ax.semilogy(
                n,
                error,
                linestyle=style,
                color=color,
                lw=1.8,
                marker=marker,
                markersize=4.2,
                markerfacecolor="white" if marker != "x" else color,
                markeredgewidth=1.0,
                markevery=(j % marker_stride, marker_stride),
                label=label,
            )
        ax.set_title(rf"$d={d}$, final rank $n={d // 2}$")
        ax.set_xlabel("Reduced-basis dimension $n$")
        ax.set_ylabel(r"exact largest energy error $E_n$")
        ax.grid(True, which="both", alpha=0.22)
    handles, labels = axes.flat[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(output_prefix.with_suffix(".pdf"))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--hf-solver", choices=("legacy_fd", "continuous_fe"), default="continuous_fe"
    )
    parser.add_argument("--grid", type=int)
    parser.add_argument("--element-order", type=int, default=3)
    args = parser.parse_args()
    grid = (
        args.grid
        if args.grid is not None
        else (FE_GRID if args.hf_solver == "continuous_fe" else GRID)
    )
    results_file = (
        REPORT_DIR / f"fe_p{args.element_order}_grid{grid}_results.json"
        if args.hf_solver == "continuous_fe"
        else RESULTS_FILE
    )
    plot_prefix = (
        FIG_DIR / f"fe_p{args.element_order}_grid{grid}_largest_error_vs_rank"
        if args.hf_solver == "continuous_fe"
        else FIG_DIR / "largest_error_vs_rank"
    )
    data = (
        json.loads(results_file.read_text(encoding="utf-8"))
        if args.plot_only
        else generate_all(
            hf_solver=args.hf_solver,
            grid=grid,
            element_order=args.element_order,
            results_file=results_file,
        )
    )
    make_plot(data, output_prefix=plot_prefix)
