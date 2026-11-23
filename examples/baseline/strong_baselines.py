"""Cost-separated strong selectors for the continuous-source FEM benchmark.

Run from the repository root:
  OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. python examples/baseline/strong_baselines.py

The expensive exact-error and quality audits occur *after* timing each greedy
round.  The common FEM construction is reported separately from method costs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.stats import norm, qmc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from contgreedy.sampling.laws import uniform_sphere
from examples.baseline.continuous_fem import FEMProblem
from examples.baseline.rom import a_orthonormal_append, matrices, true_worst
from examples.baseline.sampling import normalized_gd_search

REPORT_METHODS = (
    "dense_eigh",
    "lanczos",
    "gd100",
    "lbfgs100",
    "sobol4096",
    "fixed_random4096",
    "fresh_random4096",
    "adaptive_cov_pool4096",
)
METHODS = REPORT_METHODS


def sobol_sphere(d: int, m: int, seed: int) -> np.ndarray:
    """Scrambled Sobol normals projected radially to the Euclidean sphere."""
    if m < 1 or m & (m - 1):
        raise ValueError("Sobol candidate count must be a power of two")
    unit = qmc.Sobol(d, scramble=True, seed=seed).random_base2(int(math.log2(m)))
    candidates = norm.ppf(np.clip(unit, 1e-12, 1 - 1e-12))
    return candidates / np.linalg.norm(candidates, axis=1, keepdims=True)


def lbfgs_rayleigh(H: np.ndarray, start: np.ndarray, maxiter: int):
    """Optimize the scale-invariant Rayleigh quotient with its analytic gradient."""
    calls = 0
    scale = max(float(la.norm(H, ord="fro")), 1e-30)

    def value_gradient(x):
        nonlocal calls
        hx = H @ x
        calls += 1
        xx = float(x @ x)
        quotient = float(x @ hx) / xx
        return -quotient / scale, -2.0 * (hx - quotient * x) / (xx * scale)

    result = minimize(
        value_gradient,
        start / np.linalg.norm(start),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "maxfun": maxiter + 15, "ftol": 1e-14},
    )
    point = result.x / np.linalg.norm(result.x)
    return point, calls, bool(result.success)


def adaptive_cov_pool(
    H: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Value-only elite-covariance pool: 4 batches of 1024, 4096 scores total.

    The covariance update is axial, respecting J(z)=J(-z). Each later batch
    contains 64 independent uniform exploration points and 960 points drawn
    from the regularized elite covariance. No indicator gradients are used.
    """
    d = H.shape[0]
    batch_size = 1024
    elite_count = 32
    uniform_count = 64
    covariance = np.eye(d) / d
    best_point = None
    best_score = -np.inf
    for stage in range(4):
        if stage == 0:
            pool = uniform_sphere(rng, batch_size, d)
        else:
            factor = la.cholesky(covariance, lower=True, check_finite=False)
            adaptive = rng.standard_normal((batch_size - uniform_count, d))
            adaptive = adaptive @ factor.T
            adaptive /= np.linalg.norm(adaptive, axis=1, keepdims=True)
            pool = np.vstack((uniform_sphere(rng, uniform_count, d), adaptive))
        scores = np.einsum("mi,ij,mj->m", pool, H, pool, optimize=True)
        index = int(np.argmax(scores))
        if scores[index] > best_score:
            best_point, best_score = pool[index].copy(), float(scores[index])
        elite = pool[np.argpartition(scores, -elite_count)[-elite_count:]]
        empirical = elite.T @ elite / elite_count
        covariance = 0.95 * empirical + 0.05 * np.eye(d) / d
    return best_point, {
        "h_actions": 4096,
        "stages": 4,
        "batch_size": batch_size,
        "elite_count": elite_count,
        "uniform_exploration_per_later_batch": uniform_count,
    }


def select(
    H: np.ndarray,
    method: str,
    z0: np.ndarray,
    pool: np.ndarray | None,
) -> tuple[np.ndarray, dict]:
    d = H.shape[0]
    if method == "dense_eigh":
        _, vectors = la.eigh(H, subset_by_index=[d - 1, d - 1])
        return vectors[:, 0], {"h_actions": None}
    if method == "lanczos":
        calls = 0

        def matvec(x):
            nonlocal calls
            calls += 1
            return H @ x

        operator = LinearOperator((d, d), matvec=matvec, dtype=H.dtype)
        _, vectors = eigsh(
            operator, k=1, which="LA", v0=z0 / np.linalg.norm(z0), tol=1e-8
        )
        return vectors[:, 0], {"h_actions": calls}
    if method == "gd100":
        point, _ = normalized_gd_search(H, z0, 100, 1.0)
        return point, {"h_actions": 101}
    if method == "lbfgs100":
        point, calls, converged = lbfgs_rayleigh(H, z0, 100)
        return point, {"h_actions": calls, "converged": converged}
    if method in {"sobol4096", "fixed_random4096", "fresh_random4096"}:
        if pool is None:
            raise ValueError("Pool selectors require candidate points")
        scores = np.einsum("mi,ij,mj->m", pool, H, pool, optimize=True)
        best = int(np.argmax(scores))
        return pool[best], {"h_actions": len(pool)}
    raise ValueError(f"Unknown selector: {method}")


def run(problem: FEMProblem, method: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pool_rng = np.random.default_rng(seed + 1_000_003)
    d = problem.d
    rank = d // 2
    basis = np.empty((problem.A.shape[0], 0))
    pool = None
    pool_seconds = 0.0
    if method.startswith("sobol"):
        begin = time.perf_counter()
        pool = sobol_sphere(d, 4096, seed)
        pool_seconds = time.perf_counter() - begin
    elif method == "fixed_random4096":
        begin = time.perf_counter()
        pool = uniform_sphere(pool_rng, 4096, d)
        pool_seconds = time.perf_counter() - begin
    timings = {
        "matrix_seconds": 0.0,
        "selector_seconds": 0.0,
        "enrichment_seconds": 0.0,
        "pool_setup_seconds": pool_seconds,
    }
    curve = [true_worst(problem.solution_gram)]
    rounds = []
    for n in range(rank):
        begin = time.perf_counter()
        H, _, _ = matrices(problem, basis)
        timings["matrix_seconds"] += time.perf_counter() - begin

        z0 = rng.standard_normal(d)
        begin = time.perf_counter()
        if method == "fresh_random4096":
            pool = uniform_sphere(pool_rng, 4096, d)
        if method == "adaptive_cov_pool4096":
            point, statistics = adaptive_cov_pool(H, pool_rng)
        else:
            point, statistics = select(H, method, z0, pool)
        timings["selector_seconds"] += time.perf_counter() - begin

        begin = time.perf_counter()
        basis, novelty = a_orthonormal_append(basis, problem.S @ point, problem.A)
        timings["enrichment_seconds"] += time.perf_counter() - begin

        # Audit is excluded from all method timings.
        optimum = float(la.eigvalsh(H, subset_by_index=[d - 1, d - 1])[0])
        quality = math.sqrt(max(float(point @ H @ point) / optimum, 0.0))
        C = basis.T @ problem.B
        G = problem.solution_gram - C.T @ C
        curve.append(true_worst(0.5 * (G + G.T)))
        rounds.append({"n": n, "rho": quality, "novelty": novelty, **statistics})

    return {
        "seed": seed,
        "method": method,
        "curve": curve,
        "timings": timings,
        "rounds": rounds,
    }


def summarize(runs: list[dict]) -> dict:
    keys = runs[0]["timings"]
    times = {key: float(np.median([r["timings"][key] for r in runs])) for key in keys}
    times["total_algorithm_seconds"] = sum(times.values())
    final = np.array([r["curve"][-1] for r in runs])
    qualities = np.array([row["rho"] for run in runs for row in run["rounds"]])
    actions = [row["h_actions"] for run in runs for row in run["rounds"]]
    return {
        "seeds": len(runs),
        "final_error_median": float(np.median(final)),
        "final_error_min": float(np.min(final)),
        "final_error_max": float(np.max(final)),
        "curve_median": np.median([r["curve"] for r in runs], axis=0).tolist(),
        "rho_median": float(np.median(qualities)),
        "rho_min": float(np.min(qualities)),
        "gamma_0_9_success_rate": float(np.mean(qualities >= 0.9)),
        "h_actions_median_per_round": (
            float(np.median(actions)) if all(x is not None for x in actions) else None
        ),
        "timings_median_seconds": times,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--grid", type=int, default=40)
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=REPORT_METHODS
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=(ROOT / "runs/report_baseline").joinpath("strong_baselines_fe_p3_grid40.json"),
    )
    args = parser.parse_args()
    output = {
        "setup": {
            "hf_solver": "P3 FEM",
            "grid": args.grid,
            "contrast": 0.25,
            "spectrum_ratio": 0.78,
            "timing": "single CPU process; post-selection audits excluded",
            "threads": "set OPENBLAS_NUM_THREADS=1 externally",
            "method_classes": {
                "spectral_solver": ["dense_eigh", "lanczos"],
                "continuous_optimizer": ["gd100", "lbfgs100"],
                "value_only_pool": [
                    "sobol4096",
                    "fixed_random4096",
                    "fresh_random4096",
                ],
                "adaptive_value_only_pool": ["adaptive_cov_pool4096"],
            },
        },
        "cases": {},
    }
    if args.resume and args.output.exists():
        output = json.loads(args.output.read_text(encoding="utf-8"))
    for d in args.dimensions:
        begin = time.perf_counter()
        problem = FEMProblem.build(args.grid, d, 0.25, 0.78, element_order=3)
        setup_seconds = time.perf_counter() - begin
        # JAX JIT compilation belongs to setup, not a repeated greedy selection.
        if "gd100" in args.methods:
            normalized_gd_search(np.eye(d), np.ones(d), 100, 1.0)
        case = {
            "rank": d // 2,
            "hf_dofs": problem.dofs,
            "hf_setup_seconds": setup_seconds,
            "methods": {},
        }
        if args.resume and str(d) in output["cases"]:
            case = output["cases"][str(d)]
        else:
            output["cases"][str(d)] = case
        for method in args.methods:
            runs = [run(problem, method, 1000 + i) for i in range(args.seeds)]
            case["methods"][method] = {
                **summarize(runs),
                "runs": runs,
            }
            summary = case["methods"][method]
            print(
                f"d={d:2d} {method:18s} E={summary['final_error_median']:.7g} "
                f"rho_min={summary['rho_min']:.4f} "
                f"select={summary['timings_median_seconds']['selector_seconds']:.3f}s",
                flush=True,
            )
            args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
