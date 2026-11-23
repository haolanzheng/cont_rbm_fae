#!/usr/bin/env python3
"""Increase full-greedy replication to 24 seeds on one shared deployment set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from examples.nonsmooth.nonsmooth_continuous_greedy import (  # noqa: E402
    ContactModel,
    ObstacleModel,
    deployment_errors,
    precompute_truths,
    run_greedy,
    sobol_points,
)


def bootstrap_comparison(pool: np.ndarray, adam: np.ndarray, seed: int, nboot: int = 20000) -> dict:
    rng = np.random.default_rng(seed)
    n = len(pool)
    diffs = np.empty(nboot)
    ratios = np.empty(nboot)
    for b in range(nboot):
        ip = rng.integers(0, n, n)
        ia = rng.integers(0, n, n)
        mp = np.median(pool[ip])
        ma = np.median(adam[ia])
        diffs[b] = mp - ma
        ratios[b] = mp / ma
    return {
        "median_pool_minus_adam": float(np.median(pool) - np.median(adam)),
        "median_ratio_pool_over_adam": float(np.median(pool) / np.median(adam)),
        "bootstrap95_difference": np.quantile(diffs, [0.025, 0.975]).tolist(),
        "bootstrap95_ratio": np.quantile(ratios, [0.025, 0.975]).tolist(),
        "probability_adam_better_all_pairs": float(np.mean(adam[:, None] < pool[None, :])),
        "mann_whitney_two_sided_p": float(mannwhitneyu(pool, adam, alternative="two-sided").pvalue),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="nonsmooth_pde_bench/results_full/results.json")
    ap.add_argument("--output", default="nonsmooth_pde_bench/results_full/greedy_24seeds.json")
    ap.add_argument("--seeds", type=int, default=24)
    args = ap.parse_args()
    base = json.loads(Path(args.input).read_text())
    root_seed = int(base["protocol"]["seed"])
    models = [ObstacleModel(24), ContactModel(28, 8)]
    out = {"n_seeds": args.seeds, "n_basis": 12, "test_size": 2048, "models": {}}
    for i, model in enumerate(models):
        seed = root_seed + 1000 * i
        fixed = base["models"][model.name]["fixed_space"]
        lr = float(fixed["chosen_lr"])
        test = sobol_points(2048, model.d, seed + 200)
        print(f"[{model.name}] precomputing common truth set", flush=True)
        truths, truth_norms = precompute_truths(model, test)
        methods = {}
        for method in ["pool100", "adam100"]:
            print(f"[{model.name}] {method}, {args.seeds} seeds", flush=True)
            runs = []
            for k in range(args.seeds):
                run_seed = seed + 300 + 19 * k
                V, history = run_greedy(model, method, 12, run_seed, lr)
                errs = deployment_errors(model, V, test, truths, truth_norms)
                runs.append({
                    "seed": run_seed,
                    "rank": int(V.shape[1]),
                    "max_error": float(np.max(errs)),
                    "q99_error": float(np.quantile(errs, 0.99)),
                    "mean_error": float(np.mean(errs)),
                    "history": history,
                })
                print(f"  {k + 1:02d}/{args.seeds}: max={runs[-1]['max_error']:.5f}", flush=True)
            methods[method] = {
                "runs": runs,
                "max_error_median": float(np.median([r["max_error"] for r in runs])),
                "max_error_iqr": np.quantile([r["max_error"] for r in runs], [0.25, 0.75]).tolist(),
                "q99_error_median": float(np.median([r["q99_error"] for r in runs])),
                "mean_error_median": float(np.median([r["mean_error"] for r in runs])),
            }
        pool = np.array([r["max_error"] for r in methods["pool100"]["runs"]])
        adam = np.array([r["max_error"] for r in methods["adam100"]["runs"]])
        out["models"][model.name] = {
            "methods": methods,
            "comparison": bootstrap_comparison(pool, adam, seed + 999),
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps({n: b["comparison"] for n, b in out["models"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
