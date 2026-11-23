"""Time one frozen greedy round on the continuous-source P3 FEM baseline.

From the repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORM_NAME=cpu \
    .venv/bin/python examples/baseline/one_round_walltime.py

The deterministic oracle path constructing V_{d/2}, FE setup, JAX compilation,
and exact spectral quality audit are outside all timed round components.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from contgreedy.sampling.laws import uniform_sphere
from examples.baseline.continuous_fem import FEMProblem
from examples.baseline.rom import a_orthonormal_append, matrices
from examples.baseline.strong_baselines import select


def summary(values: list[float]) -> dict[str, float]:
    a = np.asarray(values)
    return {
        "median": float(np.median(a)),
        "q25": float(np.quantile(a, 0.25)),
        "q75": float(np.quantile(a, 0.75)),
    }


def oracle_basis(problem: FEMProblem) -> np.ndarray:
    basis = np.empty((problem.dofs, 0))
    for _ in range(problem.d // 2):
        H, _, _ = matrices(problem, basis)
        _, vectors = la.eigh(H, subset_by_index=[problem.d - 1, problem.d - 1])
        basis, novelty = a_orthonormal_append(
            basis, problem.S @ vectors[:, 0], problem.A
        )
        if novelty <= 0:
            raise RuntimeError("Oracle path did not enrich the reduced space")
    return basis


def run_case(d: int, repeats: int, pool_size: int, seed: int) -> dict:
    problem = FEMProblem.build(40, d, 0.25, 0.78, element_order=3)
    basis = oracle_basis(problem)
    H, _, _ = matrices(problem, basis)
    eigs = la.eigvalsh(H)
    lam1 = float(eigs[-1])
    if lam1 <= 0:
        raise RuntimeError("Frozen indicator has terminated")

    rng = np.random.default_rng(seed + d)
    z0 = rng.standard_normal(d)
    begin = time.perf_counter()
    pool = uniform_sphere(rng, pool_size, d)
    pool_setup = time.perf_counter() - begin

    pool_name = f"fixed_pool{pool_size}"
    specs = {
        "oracle": ("dense_eigh", None),
        "gd100": ("gd100", None),
        pool_name: ("fixed_random4096", pool),
    }
    for method, candidates in specs.values():
        point, _ = select(H, method, z0, candidates)
        a_orthonormal_append(basis, problem.S @ point, problem.A)

    records = {name: [] for name in specs}
    qualities = {}
    for repeat in range(repeats):
        names = list(specs)
        names = names[repeat % len(names) :] + names[: repeat % len(names)]
        for name in names:
            method, candidates = specs[name]
            t0 = time.perf_counter()
            H_now, _, _ = matrices(problem, basis)
            t1 = time.perf_counter()
            point, _ = select(H_now, method, z0, candidates)
            t2 = time.perf_counter()
            enriched, novelty = a_orthonormal_append(
                basis, problem.S @ point, problem.A
            )
            t3 = time.perf_counter()
            if enriched.shape[1] != basis.shape[1] + 1 or novelty <= 0:
                raise RuntimeError(f"{name} produced a dependent snapshot")
            qualities[name] = float(np.sqrt(max(point @ H @ point, 0.0) / lam1))
            records[name].append(
                {
                    "H_seconds": t1 - t0,
                    "selection_seconds": t2 - t1,
                    "enrichment_seconds": t3 - t2,
                    "total_seconds": t3 - t0,
                }
            )

    return {
        "d": d,
        "n": d // 2,
        "hf_dofs": problem.dofs,
        "max_relative_energy_projection_error": problem.projection_error,
        "lambda2_over_lambda1": float(eigs[-2] / lam1),
        "frobenius_over_lambda1": float(la.norm(H, ord="fro") / lam1),
        "pool_generation_seconds": pool_setup,
        "methods": {
            name: {
                "indicator_fraction": qualities[name],
                "timings": {
                    key: summary([row[key] for row in rows])
                    for key in rows[0]
                },
            }
            for name, rows in records.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--pool-size", type=int, default=10_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=(ROOT / "runs/report_baseline").joinpath("one_round_walltime_fe_p3_grid40.json"),
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.pool_size < 1:
        raise ValueError("Positive repeats and pool size are required")
    if args.output.exists():
        raise FileExistsError(args.output)
    pool_name = f"fixed_pool{args.pool_size}"
    results = {
        "protocol": {
            "hf": "P3 FEM, grid 40, contrast 0.25, rho 0.78",
            "frozen_basis": "oracle path to rank d/2",
            "methods": ["oracle", "GD-100", f"fixed pool-{args.pool_size}"],
            "repeats": args.repeats,
            "seed": 7129,
            "timing": "single-thread CPU; FE setup, basis preparation, JAX compilation, and exact audits excluded; H construction, selection, and enrichment timed",
            "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "jax_platform": os.environ.get("JAX_PLATFORM_NAME"),
        },
        "cases": {},
    }
    for d in args.dimensions:
        case = run_case(d, args.repeats, args.pool_size, seed=7129)
        results["cases"][str(d)] = case
        print(
            f"d={d:3d} dofs={case['hf_dofs']:5d} "
            f"GD={case['methods']['gd100']['timings']['total_seconds']['median']:.6f}s "
            f"pool={case['methods'][pool_name]['timings']['total_seconds']['median']:.6f}s",
            flush=True,
        )
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
