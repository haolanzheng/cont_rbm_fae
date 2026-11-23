"""Check HF-grid sensitivity for the same continuum sine-load family.

The load mixing is frozen at the 48x48 reported baseline. For each grid,
the sine modes are resampled and multiplied by the same coefficients;
the loads are not re-whitened. The discrete energy Gram is rescaled by
the cell area and the prescribed singular values before comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from examples.baseline.hf import elliptic_matrix, inv_sqrt_spd, sine_loads


CALIBRATION_GRID = 48


def gram(n: int, d: int, mixing: np.ndarray) -> np.ndarray:
    A = elliptic_matrix(n, 0.25)
    # sine_loads normalizes each discrete column by n+1. This factor
    # recovers the same pointwise continuum sine fields on each grid.
    B = ((n + 1) / (CALIBRATION_GRID + 1)) * (sine_loads(n, d) @ mixing)
    solve = spla.factorized(A)
    S = np.column_stack([solve(B[:, j]) for j in range(d)])
    G = (B.T @ S) / (n + 1) ** 2
    return 0.5 * (G + G.T)


def audit(d: int) -> dict:
    n0 = CALIBRATION_GRID
    A0 = elliptic_matrix(n0, 0.25)
    b0 = sine_loads(n0, d)
    solve0 = spla.factorized(A0)
    gram0 = b0.T @ np.column_stack([solve0(b0[:, j]) for j in range(d)])
    singular = 0.78 ** np.arange(d)
    mixing = inv_sqrt_spd(gram0) * singular[None, :]
    diagonal_inverse = np.diag(1.0 / singular)
    normalized = {
        n: (n0 + 1.0) ** 2
        * diagonal_inverse
        @ gram(n, d, mixing)
        @ diagonal_inverse
        for n in (24, 48, 96)
    }
    reference = normalized[96]
    rows = []
    for n in (24, 48, 96):
        G = normalized[n]
        eig = la.eigvalsh(G)
        rows.append(
            {
                "grid": n,
                "relative_gram_operator_difference_from_96": float(
                    la.norm(G - reference, ord=2) / la.norm(reference, ord=2)
                ),
                "max_relative_diagonal_difference_from_96": float(
                    np.max(np.abs(np.diag(G) / np.diag(reference) - 1.0))
                ),
                "scaled_gram_min_eigenvalue": float(eig[0]),
                "scaled_gram_max_eigenvalue": float(eig[-1]),
            }
        )
    return {"d": d, "reference_grid": 96, "rows": rows}


if __name__ == "__main__":
    output = {
        "load_policy": "fixed continuum loads calibrated on the reported 48x48 grid",
        "cases": [],
    }
    for d in (32, 64):
        result = audit(d)
        output["cases"].append(result)
        print(f"d={d}: {result['rows']}", flush=True)
    path = (ROOT / "runs/report_baseline").joinpath("mesh_refinement_audit.json")
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
