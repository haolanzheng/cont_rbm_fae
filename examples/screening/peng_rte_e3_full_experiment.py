#!/usr/bin/env python3
"""Paper-faithful E3 screen for Peng et al.'s angular RTE RB method.

This script uses the paper's 1-D Example 3 data, P1 upwind DG with dx=.125,
GL2 initialization, symmetry-paired enrichment, polynomial least-squares
density reconstruction, GL24 training and GL32 testing.  The spatial DG
coordinates are L2-orthonormalized, so coefficient Euclidean norms are the
physical discrete L2 norms reported by the paper.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import svd
from scipy.sparse import diags

from examples.screening.peng_rte_continuous_greedy import (
    DG1D,
    Indicator,
    adam_choice,
    examples,
    pool_choice,
    selected_coupled_solution,
)


def orthonormal_e3() -> DG1D:
    model = DG1D(examples()[2])
    mass = np.tile([model.dx, model.dx / 3.0], model.nx)
    transform = diags(1.0 / np.sqrt(mass))
    model.T = (transform @ model.T @ transform).tocsc()
    model.S = (transform @ model.S @ transform).tocsc()
    model.Up = (transform @ model.Up @ transform).tocsc()
    model.Um = (transform @ model.Um @ transform).tocsc()
    model.g = np.asarray(transform @ model.g)
    old_boundary_rhs = model.boundary_rhs
    model.boundary_rhs = lambda v: np.asarray(transform @ old_boundary_rhs(v))
    return model


def spectral_ratio(F: np.ndarray) -> float:
    singular = svd(F, full_matrices=False, compute_uv=False, check_finite=False)
    return float(singular[-1] / singular.sum())


def audit(
    model: DG1D,
    selected: np.ndarray,
    test_nodes: np.ndarray,
    truth_rho: np.ndarray,
    training_truth_rho: np.ndarray | None = None,
) -> dict:
    rho, F, weights = selected_coupled_solution(model, selected)
    Q, singular, _ = svd(F, full_matrices=False, check_finite=False)
    keep = singular > 1e-12 * singular[0]
    Q = Q[:, keep]
    absolute = []
    relative = []
    projection_relative = []
    truth_density_relative = []
    for v in test_nodes:
        true = model.sweep(float(v), truth_rho)
        reduced_matrix = Q.T @ (model.A(float(v)) @ Q)
        rhs = Q.T @ (model.g + model.boundary_rhs(float(v)) + model.S @ rho)
        rb = Q @ np.linalg.solve(reduced_matrix, rhs)
        error = np.linalg.norm(true - rb)
        absolute.append(error)
        relative.append(error / np.linalg.norm(true))
        projection_relative.append(np.linalg.norm(true - Q @ (Q.T @ true)) / np.linalg.norm(true))
        rhs_truth_density = Q.T @ (
            model.g + model.boundary_rhs(float(v)) + model.S @ truth_rho
        )
        rb_truth_density = Q @ np.linalg.solve(reduced_matrix, rhs_truth_density)
        truth_density_relative.append(
            np.linalg.norm(true - rb_truth_density) / np.linalg.norm(true)
        )
    density_reference = truth_rho if training_truth_rho is None else training_truth_rho
    return {
        "rank": int(Q.shape[1]),
        "spectral_ratio": spectral_ratio(F),
        "max_absolute_f": float(np.max(absolute)),
        "max_relative_f": float(np.max(relative)),
        "mean_relative_f": float(np.mean(relative)),
        "max_projection_relative_f": float(np.max(projection_relative)),
        "max_relative_f_with_truth_density_diagnostic": float(np.max(truth_density_relative)),
        "relative_density_vs_training_fom": float(
            np.linalg.norm(rho - density_reference) / np.linalg.norm(density_reference)
        ),
        "relative_density_vs_test_fom": float(
            np.linalg.norm(rho - truth_rho) / np.linalg.norm(truth_rho)
        ),
        "density_weight_l1": float(np.linalg.norm(weights, 1)),
        "density_weight_min": float(weights.min()),
        "selected": selected.tolist(),
    }


def train_to_dimensions(
    model: DG1D,
    method: str,
    seed: int,
    lr: float,
    dimensions: tuple[int, ...] = (4, 6, 8, 10),
) -> dict[int, dict]:
    rng = np.random.default_rng(seed)
    selected = np.array([-1 / math.sqrt(3), 1 / math.sqrt(3)])
    if method == "gl24":
        pool = leggauss(24)[0]
    elif method.startswith("pool"):
        pool = rng.uniform(-1, 1, int(method[4:]))
    else:
        pool = None
    saved = {}
    selection_history = []
    while len(selected) < max(dimensions):
        rho, F, _ = selected_coupled_solution(model, selected)
        indicator = Indicator.build(model, selected, rho, F)
        dense_grid = np.linspace(-1, 1, 20001)
        oracle_v, oracle_value = pool_choice(indicator, dense_grid)
        indicator.evaluations = 0
        if method == "adam100":
            chosen, value, start = adam_choice(indicator, rng, lr, steps=100)
        else:
            chosen, value = pool_choice(indicator, pool)
            start = None
        selection_history.append(
            {
                "rank_before": len(selected),
                "start": start,
                "chosen": chosen,
                "indicator": value,
                "oracle_v": oracle_v,
                "oracle_indicator": oracle_value,
                "indicator_ratio": value / oracle_value,
            }
        )
        selected = np.r_[selected, chosen, -chosen]
        if len(selected) in dimensions:
            saved[len(selected)] = {
                "selected": selected.copy(),
                "history": list(selection_history),
            }
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("peng_rte_e3_full_results.json"))
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--adam-lr", type=float, default=0.02)
    args = parser.parse_args()

    model = orthonormal_e3()
    training_nodes, training_weights = leggauss(24)
    test_nodes, test_weights = leggauss(32)
    training_truth_rho, _ = model.coupled_density(training_nodes, training_weights / 2)
    truth_rho, _ = model.coupled_density(test_nodes, test_weights / 2)
    methods = ["gl24", "pool100", "pool10000", "adam100"]
    output = {
        "problem": "Peng et al. 1-D Example 3 (two-material problem 1)",
        "paper_settings": {
            "domain": [0, 20],
            "dx": 0.125,
            "spatial_method": "P1 upwind DG in L2-orthonormal scaled Legendre coordinates",
            "source": "5 on (0,10), 0 on (10,20)",
            "sigma_t": 100,
            "sigma_s": "90 on (0,10), 100 on (10,20)",
            "inflow": 0,
            "initial_angles": "GL2",
            "paper_training_angles": "GL24",
            "test_angles": "GL32",
            "density_reconstruction": "paper polynomial least squares, s=m+1",
        },
        "seed": args.seed,
        "adam_lr": args.adam_lr,
        "adam_steps": 100,
        "methods": {},
    }
    for method_index, method in enumerate(methods):
        trained = train_to_dimensions(
            model, method, args.seed + 1000 * method_index, args.adam_lr
        )
        method_result = {}
        for dimension, item in trained.items():
            item_audit = audit(
                model, item["selected"], test_nodes, truth_rho, training_truth_rho
            )
            item_audit["history"] = item["history"]
            method_result[str(dimension)] = item_audit
        output["methods"][method] = method_result
        output["methods"][method]["stopping_dimension"] = {
            str(tol): next(
                (
                    d
                    for d in (4, 6, 8, 10)
                    if method_result[str(d)]["spectral_ratio"] <= tol
                ),
                None,
            )
            for tol in (1e-4, 1e-6)
        }
        print(
            method,
            {d: method_result[str(d)]["max_relative_f"] for d in (4, 6, 8, 10)},
        )
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
