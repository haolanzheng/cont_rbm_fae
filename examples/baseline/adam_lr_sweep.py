#!/usr/bin/env python3
"""Learning-rate robustness check for the Riemannian Adam search."""

import argparse
import json
from pathlib import Path

import numpy as np

from examples.baseline.experiment import Problem, run_greedy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("regularity_guided_greedy/adam_lr_sweep.json"))
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    problem = Problem.build(24, 32, 0.25, 0.78, "controlled")
    lrs = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    results = []
    for lr in lrs:
        runs = [run_greedy(problem, "adam", 16, 4000 + s, steps=100, adam_lr=lr) for s in range(args.seeds)]
        finals = np.array([r["rows"][-1]["worst_true"] for r in runs])
        per_run_min_rho = np.array([min(row["rho"] for row in r["rows"][:-1]) for r in runs])
        results.append({
            "lr": lr,
            "final_error_median": float(np.median(finals)),
            "final_error_q10": float(np.quantile(finals, 0.1)),
            "final_error_q90": float(np.quantile(finals, 0.9)),
            "min_rho_median": float(np.median(per_run_min_rho)),
            "min_rho_q10": float(np.quantile(per_run_min_rho, 0.1)),
            "success_all_rounds_rho_ge_0.9": float(np.mean(per_run_min_rho >= 0.9)),
        })
    out = {
        "problem": "d=32 controlled-spectrum variable-coefficient elliptic benchmark",
        "rank": 16,
        "steps": 100,
        "seeds": args.seeds,
        "optimal_width_rank16": float(problem.singular_values[16]),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
