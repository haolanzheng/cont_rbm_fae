"""Paired bootstrap audit of archived contact/obstacle greedy runs.

No PDE solves or training are performed. The input is the archived
retuned_results.json; resampling units are matched greedy seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def pair_max_errors(pool_runs, adam_runs):
    """Align two run lists by seed; reject ambiguous or incomplete pairings."""
    def index(runs):
        result = {}
        for run in runs:
            seed = int(run["seed"])
            if seed in result:
                raise ValueError(f"Duplicate greedy seed: {seed}")
            value = float(run["max_error"])
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Maximum relative errors must be finite and positive")
            result[seed] = (int(run["rank"]), value)
        return result

    pool, adam = index(pool_runs), index(adam_runs)
    if not pool or pool.keys() != adam.keys():
        raise ValueError("Both methods must have the same nonempty seed set")
    seeds = sorted(pool)
    ranks = {pool[s][0] for s in seeds} | {adam[s][0] for s in seeds}
    if len(ranks) != 1:
        raise ValueError("All runs must have the same reduced rank")
    return (
        seeds,
        np.array([pool[s][1] for s in seeds]),
        np.array([adam[s][1] for s in seeds]),
    )


def paired_bootstrap(pool, adam, *, seed=74119, n_resamples=20000):
    """Percentile interval for median(pool) / median(adam), not median ratios."""
    pool = np.asarray(pool, dtype=float)
    adam = np.asarray(adam, dtype=float)
    if pool.ndim != 1 or pool.size == 0 or pool.shape != adam.shape:
        raise ValueError("Expected nonempty equal-length vectors")
    if not np.all(np.isfinite(pool)) or not np.all(np.isfinite(adam)):
        raise ValueError("Errors must be finite")
    if np.any(pool <= 0) or np.any(adam <= 0) or n_resamples <= 0:
        raise ValueError("Errors and number of resamples must be positive")
    rng = np.random.default_rng(seed)
    # Common indices preserve dependence within each greedy-seed pair.
    indices = rng.integers(0, pool.size, size=(n_resamples, pool.size))
    ratios = np.median(pool[indices], axis=1) / np.median(adam[indices], axis=1)
    return {
        "n_pairs": int(pool.size),
        "median_pool": float(np.median(pool)),
        "median_adam": float(np.median(adam)),
        "ratio_of_medians_pool_over_adam": float(np.median(pool) / np.median(adam)),
        "paired_percentile95": np.quantile(
            ratios, [0.025, 0.975], method="linear"
        ).tolist(),
    }


def audit(source, *, seed=74119, n_resamples=20000):
    source = Path(source)
    raw = source.read_bytes()
    data = json.loads(raw)
    results = {}
    for name in sorted(data["models"]):
        greedy = data["models"][name]["full_greedy"]
        seeds, pool, adam = pair_max_errors(
            greedy["pool_summary"]["runs"], greedy["adam_runs"]
        )
        # Resetting to the same bootstrap seed makes each model independent
        # of traversal order; inference is still performed model by model.
        result = paired_bootstrap(pool, adam, seed=seed, n_resamples=n_resamples)
        result["greedy_seeds"] = seeds
        results[name] = result
    return {
        "source_name": source.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "numpy_version": np.__version__,
        "bootstrap_seed_per_model": seed,
        "n_resamples": n_resamples,
        "estimand": "median(pool maximum relative error) / median(Adam maximum relative error)",
        "resampling_unit": "paired greedy seed; same indices for both methods",
        "interval": "95% percentile bootstrap; NumPy linear quantiles",
        "scope": "conditional on the fixed shared validation set; no PDE rerun",
        "models": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=74119)
    parser.add_argument("--resamples", type=int, default=20000)
    args = parser.parse_args()
    result = audit(args.input, seed=args.seed, n_resamples=args.resamples)
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        # Preserve existing audits: a changed protocol needs a new output path.
        with args.output.open("x") as stream:
            stream.write(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
