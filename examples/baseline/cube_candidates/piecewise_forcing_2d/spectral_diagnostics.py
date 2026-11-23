"""Reconstruct spectral diagnostics from saved square-partition greedy paths.

No training is repeated. Run from the continuous_greedy repository root, or
provide --repo. All quantities use the saved parameter paths and the existing
finite sine-series model. This script does not modify source experiment data.
"""

import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path,
                        default=Path("/tmp/cg_dimension_mechanism_diagnostics_20261112.json"))
    parser.add_argument("--m", type=int, nargs="+", default=[5, 8, 12, 16])
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "examples/baseline/cube_candidates/piecewise_forcing_2d"))
    from run import model, append, residual_gram

    base = repo / "runs/piecewise_forcing_2d_20261112/square_dimension_study"
    reports = []
    for m in args.m:
        d = m * m
        N = d // 2
        B, G, _ = model(m)
        eigenvalues = np.sum(B * B, axis=1)
        order = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[order]
        data = np.load(base / f"d{d}_snapshots.npz")
        reference = json.loads((base / f"d{d}.json").read_text())
        rows = []
        for method in ["gd80", "pool10000"]:
            for seed in [1000, 1001, 1002, 1003]:
                Q = np.zeros((d, 0))
                statistics = []
                for n, z in enumerate(data[f"z_s{seed}_{method}"]):
                    H = residual_gram(B, G, Q)
                    hz = H @ z
                    J = z @ hz
                    largest = np.linalg.eigvalsh(H)[-1]
                    statistics.append(dict(
                        n=n,
                        source_energy=float(z @ z / d),
                        selected_direction_quality=float(J / (z @ z) / largest),
                        mean_gain_over_lmax=float((hz @ hz) / J / largest),
                        mean_squared_error_drop=float((hz @ hz) / (3 * J)),
                    ))
                    Q, _ = append(B, Q, z)

                R = B - Q @ (Q.T @ B)
                H = R.T @ R
                spectrum = np.linalg.eigvalsh(H)
                trace = float(np.trace(H))
                largest = float(spectrum[-1])
                modal_residual = np.sum(R * R, axis=1)
                K = N // 2
                previous = next(r for r in reference["results"]
                                if r["method"] == method and r["seed"] == seed)
                rows.append(dict(
                    method=method,
                    seed=seed,
                    trace=trace,
                    lmax=largest,
                    trace_over_optimal_tail=trace / sum(sorted_eigenvalues[N:]),
                    lmax_over_optimal_sphere_squared=largest / sorted_eigenvalues[N],
                    low_mode_residual_fraction=float(sum(modal_residual[order[:K]]) / trace),
                    shape_factor=previous["energy_error"] ** 2 / (d * largest),
                    error_over_E0=previous["error_over_E0"],
                    latter_half={key: float(np.mean([s[key] for s in statistics[N // 2:]]))
                                 for key in ["source_energy", "selected_direction_quality",
                                             "mean_gain_over_lmax", "mean_squared_error_drop"]},
                    rounds=statistics,
                ))

        reports.append(dict(
            d=d,
            N=N,
            optimal_trace=float(sum(sorted_eigenvalues[N:])),
            optimal_lmax=float(sorted_eigenvalues[N]),
            low_mode_count=N // 2,
            rows=rows,
        ))

    output = dict(definitions={
        "source_energy": "||z||^2/d=||f_z||_L2^2",
        "selected_direction_quality": "(z^T H z)/(||z||^2 lambda_max(H))",
        "mean_gain_over_lmax": "(z^T H^2 z)/(z^T H z lambda_max(H)); actual uniform-cube mean squared error drop=(z^T H^2 z)/(3 z^T H z)",
        "trace_over_optimal_tail": "trace(H_N)/sum_{j>N}lambda_j(G); square-error metric",
        "shape_factor": "E_test_N^2/(d lambda_max(H_N))",
        "latter_half": "mean over n=floor(N/2),...,N-1, evaluated along each method own greedy path",
        "reported_summary_ratio": "previous trace/lmax comparison ratios were ratios of four-seed medians, NOT median of paired ratios; other supplied per-method metrics were medians of per-seed metrics.",
    }, reports=reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
