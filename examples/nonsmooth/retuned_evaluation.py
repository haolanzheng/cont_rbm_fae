#!/usr/bin/env python3
"""Independent evaluation after expanding the Adam learning-rate grid."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from examples.nonsmooth.nonsmooth_continuous_greedy import (  # noqa: E402
    ContactModel,
    ObstacleModel,
    adam_search,
    build_common_basis,
    deployment_errors,
    precompute_truths,
    run_greedy,
    sobol_points,
    wilson_interval,
)
from examples.nonsmooth.replicate_full_greedy import bootstrap_comparison  # noqa: E402


def main() -> None:
    root = Path("nonsmooth_pde_bench/results_full")
    base = json.loads((root / "results.json").read_text())
    rep = json.loads((root / "greedy_24seeds.json").read_text())
    chosen = {"membrane_obstacle": 0.55, "elastic_contact": 0.8}
    out = {"selection": "largest mean best-J on 40 disjoint tuning starts", "models": {}}
    for i, model in enumerate([ObstacleModel(24), ContactModel(28, 8)]):
        seed = int(base["protocol"]["seed"]) + 1000 * i
        lr = chosen[model.name]
        fixed0 = base["models"][model.name]["fixed_space"]
        V, _ = build_common_basis(model, 16, 7, seed + 1)
        starts = sobol_points(320, 3, seed + 5)
        print(f"[{model.name}] independent fixed-space trajectories, lr={lr}", flush=True)
        traces = [adam_search(model, V, z, lr, 100) for z in starts]
        best = np.array([t["best_J"] for t in traces])
        fixed = {"lr": lr, "trials": 320, "q_T100": {}, "q_T100_wilson95": {}, "q_by_step": {}}
        for q in ["0.99", "0.995", "0.999"]:
            tau = fixed0["thresholds"][q]
            k = int(np.sum(best >= tau))
            fixed["q_T100"][q] = k / 320
            fixed["q_T100_wilson95"][q] = wilson_interval(k, 320)
        for T in [1, 5, 10, 25, 50, 100]:
            vals = np.array([t["best_values"][T - 1] for t in traces])
            fixed["q_by_step"][str(T)] = {
                q: float(np.mean(vals >= fixed0["thresholds"][q])) for q in ["0.99", "0.995", "0.999"]
            }
        fixed["best_J"] = {
            "mean": float(np.mean(best)), "median": float(np.median(best)),
            "q10": float(np.quantile(best, 0.1)), "q90": float(np.quantile(best, 0.9)),
            "max": float(np.max(best)), "values": best.tolist(),
        }
        fixed["activity_count_changes_mean"] = float(np.mean([t["active_count_changes"] for t in traces]))

        test = sobol_points(2048, 3, seed + 200)
        print(f"[{model.name}] common deployment truths", flush=True)
        truths, truth_norms = precompute_truths(model, test)
        print(f"[{model.name}] full Adam greedy, 24 seeds", flush=True)
        runs = []
        for k in range(24):
            run_seed = seed + 300 + 19 * k
            Vr, history = run_greedy(model, "adam100", 12, run_seed, lr)
            errs = deployment_errors(model, Vr, test, truths, truth_norms)
            runs.append({
                "seed": run_seed, "rank": int(Vr.shape[1]),
                "max_error": float(np.max(errs)),
                "q99_error": float(np.quantile(errs, 0.99)),
                "mean_error": float(np.mean(errs)), "history": history,
            })
            print(f"  {k + 1:02d}/24 max={runs[-1]['max_error']:.5f}", flush=True)
        pool_runs = rep["models"][model.name]["methods"]["pool100"]["runs"]
        pool_max = np.array([r["max_error"] for r in pool_runs])
        adam_max = np.array([r["max_error"] for r in runs])
        greedy = {
            "adam_runs": runs,
            "adam_summary": {
                "max_error_median": float(np.median(adam_max)),
                "max_error_iqr": np.quantile(adam_max, [0.25, 0.75]).tolist(),
                "q99_error_median": float(np.median([r["q99_error"] for r in runs])),
                "mean_error_median": float(np.median([r["mean_error"] for r in runs])),
            },
            "pool_summary": rep["models"][model.name]["methods"]["pool100"],
            "comparison": bootstrap_comparison(pool_max, adam_max, seed + 1999),
        }
        out["models"][model.name] = {"fixed_space": fixed, "full_greedy": greedy}
        (root / "retuned_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({n: b["full_greedy"]["comparison"] for n, b in out["models"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
