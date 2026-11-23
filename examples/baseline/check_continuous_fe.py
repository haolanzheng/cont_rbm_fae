"""Check HF mesh error for the mesh-independent spectral baseline.

Run from the repository root with ``PYTHONPATH=src:.``.  The reported error is
the operator norm of I-P_h on the entire d-dimensional exact solution space,
in the continuous PDE energy norm; it is not a Monte Carlo estimate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from examples.baseline.continuous_fem import FEMProblem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", type=int, nargs="+", default=[24, 32, 40])
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--element-order", type=int, default=3)
    parser.add_argument("--target", type=float, default=0.003)
    parser.add_argument(
        "--output",
        type=Path,
        default=(ROOT / "runs/report_baseline").joinpath("fe_p3_convergence.json"),
    )
    args = parser.parse_args()
    rows = []
    for grid in args.grids:
        p = FEMProblem.build(grid, args.dimension, element_order=args.element_order)
        rank = args.dimension // 2
        row = {
            "grid_interior_vertices_per_axis": grid,
            "degrees_of_freedom": p.dofs,
            "element_order": p.element_order,
            "quadrature_order": p.quadrature_order,
            "max_relative_energy_projection_error": p.projection_error,
            "continuous_width_at_half_rank": p.spectrum_ratio**rank,
            "discrete_width_at_half_rank": float(p.singular_values[rank]),
        }
        row["relative_width_difference"] = abs(
            row["discrete_width_at_half_rank"] / row["continuous_width_at_half_rank"]
            - 1.0
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "problem": "continuous manufactured right-hand side; variable-coefficient elliptic PDE",
        "dimension": args.dimension,
        "contrast": 0.25,
        "spectrum_ratio": 0.78,
        "target_max_relative_energy_projection_error": args.target,
        "target_met": rows[-1]["max_relative_energy_projection_error"] <= args.target,
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["target_met"]:
        raise SystemExit("finest grid does not meet the requested HF accuracy")


if __name__ == "__main__":
    main()
