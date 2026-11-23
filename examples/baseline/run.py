"""Spectral experiment orchestration. Use run.py spectral for isolated runs."""

from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.special import betainc
from .hf import Problem
from .rom import matrices, true_worst, equivalence_constants, a_orthonormal_append
from .sampling import (
    normalize_rows, power_search, adam_search, normalized_gd_search, pool_search,
)
from .evaluate import estimate_high_set_probability, landscape_audit, aggregate
from .plot import make_figures


def run_greedy(
    problem: Problem,
    method: str,
    rank: int,
    seed: int,
    *,
    steps: int = 25,
    pool_size: int = 256,
    adam_lr: float = 0.05,
    audit: bool = False,
    gamma: float = 0.9,
    audit_samples: int = 100000,
) -> dict:
    from contgreedy.rom import greedy

    rng = np.random.default_rng(seed)
    V = np.empty((problem.A.shape[0], 0))
    fixed_pool = (
        normalize_rows(rng.standard_normal((pool_size, problem.d)))
        if method == "pool"
        else None
    )
    rows = []
    start_time = time.perf_counter()

    def freeze_row(basis):
        H, G, _ = matrices(problem, basis)
        eigs, q = la.eigh(H)
        jstar = max(float(eigs[-1]), 0.0)
        row = {
            "n": basis.shape[1],
            "worst_true": true_worst(G),
            "indicator_max": math.sqrt(jstar),
        }
        cmin, cmax = equivalence_constants(H, G)
        row.update({"c_minus": cmin, "c_plus": cmax, "equiv_ratio": cmin / cmax})
        return H, eigs, q, jstar, row

    def choose(basis, iteration):
        H, eigs, q, jstar, row = freeze_row(basis)
        if audit:
            p, hits = estimate_high_set_probability(eigs, gamma, audit_samples, rng)
            land = landscape_audit(H, gamma, starts=300, steps=100, rng=rng)
            row.update({"p_high": p, "p_hits": hits, **land})
        if method == "exact_indicator":
            z, selected = q[:, -1], jstar
        elif method in {"power", "adam", "normalized_gd"}:
            z0 = rng.standard_normal(problem.d)
            if method == "power":
                z, hist = power_search(H, z0, steps)
            elif method == "normalized_gd":
                z, hist = normalized_gd_search(H, z0, steps, adam_lr)
                lam1 = float(eigs[-1])
                lam2 = max(float(eigs[-2]), 0.0)
                relative_second = lam2 / lam1
                xi = adam_lr * lam1 / float(la.norm(H, ord="fro"))
                ratio = (1.0 + xi * relative_second) / (1.0 + xi)
                attenuation = ratio ** (2 * steps)
                gamma_sq = gamma * gamma
                threshold = gamma_sq * attenuation / (
                    1.0 - gamma_sq + gamma_sq * attenuation
                )
                row["gd_failure_upper"] = float(
                    betainc(0.5, (problem.d - 1) / 2, threshold)
                )
                if relative_second < gamma_sq:
                    direct_threshold = (
                        gamma_sq - relative_second
                    ) / (1.0 - relative_second)
                    row["direct_hit_p_upper"] = float(
                        betainc(
                            (problem.d - 1) / 2, 0.5, 1.0 - direct_threshold
                        )
                    )
                else:
                    row["direct_hit_p_upper"] = 1.0
            else:
                z, hist = adam_search(H, z0, steps, adam_lr)
            selected = float(z @ H @ z)
            row["inner_history"] = hist
        elif method == "pool":
            z, selected = pool_search(H, fixed_pool)
        elif method == "fresh_pool":
            pool = normalize_rows(rng.standard_normal((pool_size, problem.d)))
            z, selected = pool_search(H, pool)
        else:
            raise ValueError(method)
        row["selected_indicator"] = math.sqrt(max(selected, 0.0))
        row["rho"] = math.sqrt(max(selected / max(jstar, 1e-300), 0.0))
        rows.append(row)
        return z, {}

    def enrich(basis, snapshot):
        updated, novelty = a_orthonormal_append(basis, snapshot, problem.A)
        rows[-1]["snapshot_novelty"] = novelty
        return updated

    def solved(basis):
        C = basis.T @ problem.B
        G = problem.solution_gram - C.T @ C
        return true_worst(0.5 * (G + G.T)) < 1e-11

    V, history = greedy(
        V,
        rounds=rank,
        select=choose,
        truth=lambda z: problem.S @ z,
        enrich=enrich,
        rank=lambda basis: basis.shape[1],
        stop=solved,
        stop_on_dependent=True,
    )
    if not history or history[-1]["rank_after"] > history[-1]["rank_before"]:
        rows.append(freeze_row(V)[-1])
    return {
        "method": method,
        "seed": seed,
        "steps": steps if method in {"power", "adam", "normalized_gd"} else None,
        "pool_size": pool_size if "pool" in method else None,
        "adam_lr": adam_lr if method == "adam" else None,
        "normalized_gd_lr": adam_lr if method == "normalized_gd" else None,
        "search_return_policy": "best_iterate_0_to_T"
        if method in {"power", "adam", "normalized_gd"}
        else "maximum_candidate",
        "elapsed_seconds": time.perf_counter() - start_time,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir", type=Path, default=Path("regularity_guided_greedy/results")
    )
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--contrast", type=float, default=0.25)
    ap.add_argument("--spectrum-ratio", type=float, default=0.78)
    ap.add_argument("--load-mode", choices=("controlled", "raw"), default="controlled")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--audit-samples", type=int, default=100000)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    problem = Problem.build(
        args.grid, args.d, args.contrast, args.spectrum_ratio, args.load_mode
    )
    config = vars(args).copy()
    config["outdir"] = str(config["outdir"])

    # One exact-indicator trajectory carries the expensive landscape audit.
    audit_run = run_greedy(
        problem,
        "exact_indicator",
        args.rank,
        7103,
        audit=True,
        gamma=0.9,
        audit_samples=args.audit_samples,
    )
    runs = [run_greedy(problem, "exact_indicator", args.rank, 7103)]
    for seed in range(args.seeds):
        s = 1000 + seed
        for t in (2, 5, 10, 25, 100):
            runs.append(run_greedy(problem, "power", args.rank, s, steps=t))
        runs.append(run_greedy(problem, "adam", args.rank, s, steps=100, adam_lr=0.05))
        for m in (64, 256, 1024, 4096):
            runs.append(run_greedy(problem, "pool", args.rank, s, pool_size=m))
        runs.append(run_greedy(problem, "fresh_pool", args.rank, s, pool_size=256))

    data = {
        "config": config,
        "problem": {
            "description": (
                "2D variable-coefficient elliptic PDE; energy-orthogonal sine-load subspace with controlled spectrum"
                if args.load_mode == "controlled"
                else "2D variable-coefficient elliptic PDE with unmodified low-frequency sine loads"
            ),
            "coefficient_bounds": [1.0 - args.contrast, 1.0 + args.contrast],
            "singular_values": problem.singular_values.tolist(),
        },
        "audit_run": audit_run,
        "runs": runs,
        "aggregate": aggregate(runs),
    }
    (args.outdir / "results.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
    make_figures(data, args.outdir)
    print(
        json.dumps(
            {
                "outdir": str(args.outdir),
                "methods": list(data["aggregate"]),
                "audit_steps": len(audit_run["rows"]) - 1,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
