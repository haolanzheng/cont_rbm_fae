"""Controlled extreme-versus-typical experiment; does not modify the paper.

The parameter law is deliberately concentrated on modes just beyond the
worst-case rank cutoff. Algorithms only receive its independent training
sample, not its density. A separate validation sample evaluates typical error;
the full-sphere worst-case error is computed exactly from a small Gram matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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
from examples.baseline.sampling import normalized_gd_search

ALPHAS = (0.0, 0.01, 0.05, 0.2, 0.5, 1.0)


def controlled_law(
    rng: np.random.Generator, count: int, d: int, rank: int, contrast: float
) -> np.ndarray:
    """Normalized anisotropic Gaussian; positive density on the whole sphere."""
    scales = np.ones(d)
    scales[rank : 2 * rank] = contrast
    raw = rng.standard_normal((count, d)) * scales
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def mixture_sample(
    rng: np.random.Generator, observed: np.ndarray, count: int, alpha: float
) -> np.ndarray:
    """Empirical resampling plus uniform exploration, without density fitting."""
    d = observed.shape[1]
    explore = rng.random(count) < alpha
    points = observed[rng.integers(len(observed), size=count)].copy()
    if np.any(explore):
        points[explore] = uniform_sphere(rng, int(explore.sum()), d)
    return points


def metric(
    residual_gram: np.ndarray,
    validation: np.ndarray,
    covariance: np.ndarray,
    total_mean_energy: float,
) -> dict:
    residual_gram = 0.5 * (residual_gram + residual_gram.T)
    worst = math.sqrt(max(float(la.eigvalsh(residual_gram)[-1]), 0.0))
    squared = np.einsum(
        "mi,ij,mj->m", validation, residual_gram, validation, optimize=True
    )
    squared = np.maximum(squared, 0.0)
    rms = math.sqrt(max(float(np.sum(residual_gram * covariance)), 0.0))
    return {
        "worst": worst,
        "rms": rms,
        "relative_rms": rms / math.sqrt(total_mean_energy),
        "q95": float(np.sqrt(np.quantile(squared, 0.95))),
        "q99": float(np.sqrt(np.quantile(squared, 0.99))),
    }


def residual_gram(G: np.ndarray, Q: np.ndarray) -> np.ndarray:
    if Q.shape[1] == 0:
        return G
    result = G - (G @ Q) @ (Q.T @ G)
    return 0.5 * (result + result.T)


def append_g_orthonormal(G: np.ndarray, Q: np.ndarray, point: np.ndarray) -> np.ndarray:
    direction = point - Q @ (Q.T @ G @ point)
    direction -= Q @ (Q.T @ G @ direction)
    size = math.sqrt(max(float(direction @ G @ direction), 0.0))
    if size < 1e-12:
        raise RuntimeError("Dependent greedy snapshot")
    return np.column_stack((Q, direction / size))


def mixture_pod(
    G: np.ndarray, empirical_covariance: np.ndarray, alpha: float, rank: int
) -> np.ndarray:
    """A-orthonormal coefficient basis optimal for mixture-law mean square."""
    d = G.shape[0]
    eigenvalues, eigenvectors = la.eigh(G)
    sqrt_g = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    invsqrt_g = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    covariance = (1.0 - alpha) * empirical_covariance + alpha * np.eye(d) / d
    weighted = sqrt_g @ covariance @ sqrt_g
    _, directions = la.eigh(weighted)
    return invsqrt_g @ directions[:, -rank:]


def run_greedy(
    G: np.ndarray,
    K: np.ndarray,
    observed: np.ndarray,
    validation: np.ndarray,
    validation_covariance: np.ndarray,
    total_mean_energy: float,
    alpha: float,
    method: str,
    seed: int,
    rank: int,
    pool_size: int,
) -> dict:
    rng = np.random.default_rng(seed)
    d = G.shape[0]
    Q = np.empty((d, 0))
    path = [metric(G, validation, validation_covariance, total_mean_energy)]
    for _ in range(rank):
        projection = np.eye(d) - Q @ (Q.T @ G)
        H = projection.T @ K @ projection
        H = 0.5 * (H + H.T)
        if method == "resampled_pool":
            candidates = mixture_sample(rng, observed, pool_size, alpha)
            scores = np.einsum("mi,ij,mj->m", candidates, H, candidates, optimize=True)
            point = candidates[int(np.argmax(scores))]
        elif method == "gd100":
            start = mixture_sample(rng, observed, 1, alpha)[0]
            point, _ = normalized_gd_search(H, start, 100, 1.0)
        else:
            raise ValueError(method)
        Q = append_g_orthonormal(G, Q, point)
        path.append(
            metric(
                residual_gram(G, Q),
                validation,
                validation_covariance,
                total_mean_energy,
            )
        )
    return {"seed": seed, "final": path[-1], "path": path}


def summarize(runs: list[dict]) -> dict:
    keys = ("worst", "rms", "relative_rms", "q95", "q99")
    return {
        "seeds": len(runs),
        "median": {
            key: float(np.median([run["final"][key] for run in runs])) for key in keys
        },
        "runs": runs,
    }


def plot(result: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    styles = (
        ("mixture_pod", "Mixture POD", "#222222", "s-"),
        ("resampled_pool", "Resampled pool", "#D55E00", "o--"),
        ("gd100", "GD-100", "#0072B2", "^:"),
    )
    for method, label, color, style in styles:
        rows = [result["results"][str(alpha)][method] for alpha in ALPHAS]
        if method != "mixture_pod":
            rows = [row["median"] for row in rows]
        worst = [row["worst"] for row in rows]
        rms = [row["rms"] for row in rows]
        axes[0].plot(ALPHAS, worst, style, color=color, label=label, markersize=5)
        axes[1].plot(rms, worst, style, color=color, label=label, markersize=5)
        for alpha, x, y in zip(ALPHAS, rms, worst):
            if alpha in (0.0, 0.2, 1.0):
                axes[1].annotate(
                    f"{alpha:g}",
                    (x, y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )
    axes[0].set(
        xlabel="Uniform exploration fraction $\\alpha$",
        ylabel="Exact worst-case energy error",
    )
    axes[1].set(
        xlabel="Held-out RMS energy error", ylabel="Exact worst-case energy error"
    )
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--grid", type=int, default=40)
    parser.add_argument("--contrast", type=float, default=10.0)
    parser.add_argument("--train", type=int, default=8192)
    parser.add_argument("--validation", type=int, default=100000)
    parser.add_argument("--pool", type=int, default=4096)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=(ROOT / "runs/report_baseline").joinpath("distribution_tradeoff_d16.json"),
    )
    args = parser.parse_args()
    if not 0 < args.rank * 2 <= args.d:
        raise ValueError("Require 0 < 2*rank <= d")
    problem = FEMProblem.build(args.grid, args.d, element_order=3)
    G = problem.solution_gram
    XB = np.column_stack([problem.solve_X(problem.B[:, j]) for j in range(args.d)])
    K = 0.5 * (problem.B.T @ XB + XB.T @ problem.B)
    rng = np.random.default_rng(713)
    observed = controlled_law(rng, args.train, args.d, args.rank, args.contrast)
    validation = controlled_law(rng, args.validation, args.d, args.rank, args.contrast)
    empirical_covariance = observed.T @ observed / len(observed)
    validation_covariance = validation.T @ validation / len(validation)
    total_mean_energy = float(np.sum(G * validation_covariance))
    result = {
        "setup": {
            "d": args.d,
            "rank": args.rank,
            "hf": "P3 FEM",
            "hf_grid": args.grid,
            "hf_dofs": problem.dofs,
            "parameter_law": "normalized diagonal Gaussian",
            "std_on_modes_rank_to_2rank_minus_1": args.contrast,
            "std_on_other_modes": 1.0,
            "empirical_train_count": args.train,
            "independent_validation_count": args.validation,
            "pool_size_per_round": args.pool,
            "replications": args.seeds,
            "alpha": list(ALPHAS),
            "alpha_meaning": "uniform exploration fraction in empirical-plus-uniform mixture",
            "mixture_pod_objective": "(1-alpha) empirical mean squared energy error plus alpha uniform-sphere mean squared energy error",
            "worst_domain": "entire Euclidean unit sphere; largest residual Gram eigenvalue",
            "rms_mean_energy_mc_se": float(
                np.std(
                    np.einsum("mi,ij,mj->m", validation, G, validation, optimize=True),
                    ddof=1,
                )
                / math.sqrt(len(validation))
            ),
        },
        "results": {},
    }
    for alpha in ALPHAS:
        Q = mixture_pod(G, empirical_covariance, alpha, args.rank)
        pod = metric(
            residual_gram(G, Q),
            validation,
            validation_covariance,
            total_mean_energy,
        )
        row = {"mixture_pod": pod}
        for method in ("resampled_pool", "gd100"):
            runs = [
                run_greedy(
                    G,
                    K,
                    observed,
                    validation,
                    validation_covariance,
                    total_mean_energy,
                    alpha,
                    method,
                    1000 + seed,
                    args.rank,
                    args.pool,
                )
                for seed in range(args.seeds)
            ]
            row[method] = summarize(runs)
        result["results"][str(alpha)] = row
        print(
            f"alpha={alpha:g} POD worst={pod['worst']:.6g} "
            f"RMS={pod['rms']:.6g}; "
            f"pool worst={row['resampled_pool']['median']['worst']:.6g} "
            f"RMS={row['resampled_pool']['median']['rms']:.6g}; "
            f"GD worst={row['gd100']['median']['worst']:.6g} "
            f"RMS={row['gd100']['median']['rms']:.6g}",
            flush=True,
        )
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot(result, args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
