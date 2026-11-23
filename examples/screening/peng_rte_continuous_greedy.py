#!/usr/bin/env python3
"""Continuous-greedy screen on the 1-D RTE examples of Peng et al.

The physical coefficients, mesh width, angular initialization, symmetry
augmentation, least-squares density reconstruction, and residual-free L1
indicator follow Section 3--4.1 of:

  Z. Peng, Y. Chen, Y. Cheng, F. Li,
  "A Reduced Basis Method for Radiative Transfer Equation", JSC (2022).

The spatial full model here is a P1 upwind DG implementation.  A direct
density Schur complement is used instead of SASI; this changes only the
linear solver, not the discrete equations.  Continuous selection uses one
uniform random start and exactly 100 projected Adam steps.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import svd
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import splu


T_ADAM = 100


@dataclass(frozen=True)
class Example:
    name: str
    x0: float
    x1: float
    sigma_t: Callable[[np.ndarray], np.ndarray]
    sigma_s: Callable[[np.ndarray], np.ndarray]
    source: Callable[[np.ndarray], np.ndarray]
    inflow_left: float = 0.0
    inflow_right: float = 0.0


def const(c: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda x: np.full_like(np.asarray(x, dtype=float), c, dtype=float)


def examples() -> list[Example]:
    return [
        Example("E1_scattering", 0, 10, const(100), const(100), const(0.01)),
        Example(
            "E2_varying_scattering",
            0,
            10,
            lambda x: 100 * (1 + x),
            lambda x: 100 * (1 + x),
            const(0.01),
        ),
        Example(
            "E3_two_material_source",
            0,
            20,
            const(100),
            lambda x: np.where(x < 10, 90.0, 100.0),
            lambda x: np.where(x < 10, 5.0, 0.0),
        ),
        Example(
            "E4_absorber_scatterer",
            0,
            11,
            lambda x: np.where(x < 1, 2.0, 100.0),
            lambda x: np.where(x < 1, 0.0, 100.0),
            const(0.0),
            inflow_left=5.0,
        ),
        Example("E5_transport", 0, 10, const(1.2), const(1.0), const(0.01)),
    ]


class DG1D:
    """P1 upwind DG matrices on a uniform mesh with dx=0.125."""

    def __init__(self, ex: Example, dx: float = 0.125):
        self.ex = ex
        self.nx = int(round((ex.x1 - ex.x0) / dx))
        assert abs(self.nx * dx - (ex.x1 - ex.x0)) < 1e-12
        self.dx = dx
        self.ndof = 2 * self.nx
        self.T = self._reaction_matrix(ex.sigma_t)
        self.S = self._reaction_matrix(ex.sigma_s)
        self.g = self._load_vector(ex.source)
        self.Up = self._streaming(+1.0)
        self.Um = self._streaming(-1.0)

    def _reaction_matrix(self, coefficient: Callable[[np.ndarray], np.ndarray]) -> csc_matrix:
        z, w = leggauss(4)
        out = lil_matrix((self.ndof, self.ndof))
        for i in range(self.nx):
            xl = self.ex.x0 + i * self.dx
            xc = xl + 0.5 * self.dx
            x = xc + 0.5 * self.dx * z
            phi = np.vstack((np.ones_like(z), z))
            local = (phi * (w * coefficient(x) * 0.5 * self.dx)) @ phi.T
            sl = slice(2 * i, 2 * i + 2)
            out[sl, sl] = local
        return out.tocsc()

    def _load_vector(self, source: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        z, w = leggauss(4)
        out = np.zeros(self.ndof)
        for i in range(self.nx):
            xl = self.ex.x0 + i * self.dx
            xc = xl + 0.5 * self.dx
            x = xc + 0.5 * self.dx * z
            phi = np.vstack((np.ones_like(z), z))
            out[2 * i : 2 * i + 2] = phi @ (w * source(x) * 0.5 * self.dx)
        return out

    def _streaming(self, sign: float) -> csc_matrix:
        """Streaming matrix at v=+1 or v=-1."""
        v = float(sign)
        out = lil_matrix((self.ndof, self.ndof))
        for i in range(self.nx):
            sl = slice(2 * i, 2 * i + 2)
            # - integral v (d test/dx) trial
            out[2 * i + 1, 2 * i] += -2 * v
            if v > 0:
                out[sl, sl] += v * np.array([[1, 1], [1, 1]], dtype=float)
                if i > 0:
                    prev = slice(2 * (i - 1), 2 * (i - 1) + 2)
                    out[sl, prev] += v * np.array([[-1, -1], [1, 1]], dtype=float)
            else:
                out[sl, sl] += (-v) * np.array([[1, -1], [-1, 1]], dtype=float)
                if i + 1 < self.nx:
                    nxt = slice(2 * (i + 1), 2 * (i + 1) + 2)
                    out[sl, nxt] += v * np.array([[1, -1], [1, -1]], dtype=float)
        return out.tocsc()

    def A(self, v: float) -> csc_matrix:
        if v >= 0:
            return self.T + v * self.Up
        return self.T + (-v) * self.Um

    def boundary_rhs(self, v: float) -> np.ndarray:
        b = np.zeros(self.ndof)
        if v > 0 and self.ex.inflow_left:
            b[:2] = v * self.ex.inflow_left * np.array([1.0, -1.0])
        elif v < 0 and self.ex.inflow_right:
            b[-2:] = (-v) * self.ex.inflow_right * np.array([1.0, 1.0])
        return b

    def sweep(self, v: float, rho: np.ndarray) -> np.ndarray:
        return splu(self.A(v)).solve(self.g + self.boundary_rhs(v) + self.S @ rho)

    def coupled_density(self, nodes: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Solve the angularly coupled system by a density Schur complement."""
        eye = np.eye(self.ndof)
        schur = eye.copy()
        rhs = np.zeros(self.ndof)
        solvers = []
        Sd = self.S.toarray()
        for v, weight in zip(nodes, weights):
            lu = splu(self.A(float(v)))
            solvers.append(lu)
            schur -= weight * lu.solve(Sd)
            rhs += weight * lu.solve(self.g + self.boundary_rhs(float(v)))
        rho = np.linalg.solve(schur, rhs)
        F = np.column_stack(
            [lu.solve(self.g + self.boundary_rhs(float(v)) + self.S @ rho) for lu, v in zip(solvers, nodes)]
        )
        return rho, F


def ls_density_weights(nodes: np.ndarray, degree: int) -> np.ndarray:
    """Weights obtained by integrating the LS polynomial on [-1,1]."""
    degree = min(int(degree), len(nodes) - 1)
    X = np.vander(nodes, N=degree + 1, increasing=True)
    moments = np.array([1.0 / (k + 1) if k % 2 == 0 else 0.0 for k in range(degree + 1)])
    return moments @ np.linalg.pinv(X, rcond=1e-12)


def selected_coupled_solution(model: DG1D, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = max(0, (len(nodes) - 2) // 2)
    degree = 1 if m == 0 else m + 1
    weights = ls_density_weights(nodes, degree)
    rho, F = model.coupled_density(nodes, weights)
    return rho, F, weights


@dataclass
class Indicator:
    model: DG1D
    selected: np.ndarray
    rho: np.ndarray
    Q: np.ndarray
    singular_values: np.ndarray
    Vt: np.ndarray
    evaluations: int = 0

    @classmethod
    def build(cls, model: DG1D, selected: np.ndarray, rho: np.ndarray, F: np.ndarray) -> "Indicator":
        Q, s, Vt = svd(F, full_matrices=False, check_finite=False)
        keep = s > (1e-12 * s[0])
        return cls(model, selected.copy(), rho.copy(), Q[:, keep], s[keep], Vt[keep, :])

    def __call__(self, v: float) -> float:
        self.evaluations += 1
        v = float(np.clip(v, -1.0, 1.0))
        Ared = self.Q.T @ (self.model.A(v) @ self.Q)
        rhs = self.Q.T @ (self.model.g + self.model.boundary_rhs(v) + self.model.S @ self.rho)
        try:
            c = np.linalg.solve(Ared, rhs)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(Ared, rhs, rcond=1e-12)[0]
        coeff_snapshot = self.Vt.T @ (c / self.singular_values)
        return float(np.linalg.norm(coeff_snapshot, ord=1))


def near_selected(v: float, selected: np.ndarray, tol: float = 1e-8) -> bool:
    return bool(np.min(np.abs(selected - v)) < tol or np.min(np.abs(selected + v)) < tol)


def pool_choice(indicator: Indicator, pool: np.ndarray) -> tuple[float, float]:
    best_v, best_y = 0.0, -np.inf
    for v in pool:
        if near_selected(float(v), indicator.selected):
            continue
        y = indicator(float(v))
        if y > best_y:
            best_v, best_y = float(v), y
    return best_v, best_y


def adam_choice(
    indicator: Indicator,
    rng: np.random.Generator,
    lr: float,
    steps: int = T_ADAM,
    fd_h: float = 2e-5,
) -> tuple[float, float, float]:
    """One random start, exactly `steps` projected Adam updates; best checkpoint returned."""
    x = float(rng.uniform(-1, 1))
    start = x
    m1 = 0.0
    m2 = 0.0
    beta1, beta2 = 0.9, 0.999
    best_x = x
    best_y = indicator(x) if not near_selected(x, indicator.selected) else -np.inf
    for t in range(1, steps + 1):
        xm = max(-1.0, x - fd_h)
        xp = min(1.0, x + fd_h)
        grad = (indicator(xp) - indicator(xm)) / max(xp - xm, 1e-14)
        if not np.isfinite(grad):
            grad = 0.0
        grad = float(np.clip(grad, -1e6, 1e6))
        m1 = beta1 * m1 + (1 - beta1) * grad
        m2 = beta2 * m2 + (1 - beta2) * grad * grad
        mhat = m1 / (1 - beta1**t)
        vhat = m2 / (1 - beta2**t)
        x = float(np.clip(x + lr * mhat / (math.sqrt(vhat) + 1e-8), -1, 1))
        y = indicator(x) if not near_selected(x, indicator.selected) else -np.inf
        if y > best_y:
            best_x, best_y = x, y
    if not np.isfinite(best_y):
        # This is a deterministic escape, not another random start.
        best_x = float(np.clip(start + 1e-4, -1, 1))
        best_y = indicator(best_x)
    return best_x, best_y, start


def dense_indicator_max(indicator: Indicator, n: int = 4001) -> tuple[float, float]:
    grid = np.linspace(-1, 1, n)
    return pool_choice(indicator, grid)


def train(
    model: DG1D,
    method: str,
    seed: int,
    target_dim: int,
    adam_lr: float,
) -> dict:
    rng = np.random.default_rng(seed)
    selected = np.array([-1 / math.sqrt(3), 1 / math.sqrt(3)], dtype=float)
    if method == "gl24":
        fixed_pool = leggauss(24)[0]
    elif method.startswith("pool"):
        fixed_pool = rng.uniform(-1, 1, int(method[4:]))
    else:
        fixed_pool = None

    history = []
    total_evals = 0
    t0 = time.perf_counter()
    while len(selected) < target_dim:
        rho, F, weights = selected_coupled_solution(model, selected)
        indicator = Indicator.build(model, selected, rho, F)
        oracle_v, oracle_y = dense_indicator_max(indicator)
        oracle_evals = indicator.evaluations
        indicator.evaluations = 0
        if method == "adam100":
            chosen, chosen_y, start = adam_choice(indicator, rng, adam_lr, T_ADAM)
        else:
            chosen, chosen_y = pool_choice(indicator, fixed_pool)
            start = None
        total_evals += indicator.evaluations
        history.append(
            {
                "dimension_before": int(len(selected)),
                "start": start,
                "chosen": chosen,
                "chosen_indicator": chosen_y,
                "oracle_v": oracle_v,
                "oracle_indicator": oracle_y,
                "indicator_ratio": chosen_y / oracle_y,
                "selection_evaluations": indicator.evaluations,
                "oracle_audit_evaluations": oracle_evals,
                "ls_weights": weights.tolist(),
            }
        )
        selected = np.r_[selected, chosen, -chosen]

    rho, F, weights = selected_coupled_solution(model, selected)
    Q, s, _ = svd(F, full_matrices=False, check_finite=False)
    keep = s > 1e-12 * s[0]
    return {
        "method": method,
        "seed": seed,
        "adam_steps": T_ADAM if method == "adam100" else None,
        "adam_lr": adam_lr if method == "adam100" else None,
        "selected": selected.tolist(),
        "history": history,
        "selection_indicator_evaluations": total_evals,
        "training_seconds": time.perf_counter() - t0,
        "rho": rho,
        "Q": Q[:, keep],
        "singular_values": s[keep],
        "density_weights": weights,
    }


def audit(
    model: DG1D,
    run: dict,
    truth_rho: np.ndarray,
    audit_nodes: np.ndarray,
) -> dict:
    Q = run["Q"]
    rho = run["rho"]
    rel = []
    abs_err = []
    for v in audit_nodes:
        f_true = model.sweep(float(v), truth_rho)
        Ared = Q.T @ (model.A(float(v)) @ Q)
        rhs = Q.T @ (model.g + model.boundary_rhs(float(v)) + model.S @ rho)
        c = np.linalg.solve(Ared, rhs)
        f_rb = Q @ c
        e = np.linalg.norm(f_true - f_rb)
        abs_err.append(e)
        rel.append(e / max(np.linalg.norm(f_true), 1e-14))
    truth_norm = np.linalg.norm(truth_rho)
    return {
        "max_relative_f": float(np.max(rel)),
        "mean_relative_f": float(np.mean(rel)),
        "p95_relative_f": float(np.quantile(rel, 0.95)),
        "max_absolute_f": float(np.max(abs_err)),
        "mean_absolute_f": float(np.mean(abs_err)),
        "relative_density": float(np.linalg.norm(truth_rho - rho) / max(truth_norm, 1e-14)),
        "worst_v": float(audit_nodes[int(np.argmax(rel))]),
    }


def serializable(run: dict) -> dict:
    return {k: v for k, v in run.items() if k not in {"rho", "Q", "singular_values", "density_weights"}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("peng_rte_results.json"))
    ap.add_argument("--examples", default="1,2,3,4,5")
    ap.add_argument("--methods", default="gl24,pool24,pool100,pool10000,adam100")
    ap.add_argument("--target-dim", type=int, default=10)
    ap.add_argument("--truth-angles", type=int, default=128)
    ap.add_argument("--audit-angles", type=int, default=501)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--adam-lr", type=float, default=0.02)
    args = ap.parse_args()

    ids = [int(x) for x in args.examples.split(",")]
    methods = args.methods.split(",")
    results = {
        "paper": "Peng, Chen, Cheng, Li (2022), A Reduced Basis Method for Radiative Transfer Equation",
        "spatial_method": "P1 upwind DG, dx=0.125; direct density Schur solve",
        "continuous_optimizer": "single uniform start, projected finite-difference Adam, best indicator checkpoint",
        "T_adam": T_ADAM,
        "target_dimension": args.target_dim,
        "truth_angles_gauss_legendre": args.truth_angles,
        "audit_angles_uniform": args.audit_angles,
        "seed": args.seed,
        "adam_lr": args.adam_lr,
        "examples": {},
    }

    for idx in ids:
        ex = examples()[idx - 1]
        print(f"[{ex.name}] building DG and truth", flush=True)
        model = DG1D(ex)
        truth_nodes, truth_weights = leggauss(args.truth_angles)
        t_truth = time.perf_counter()
        truth_rho, _ = model.coupled_density(truth_nodes, truth_weights / 2)
        truth_seconds = time.perf_counter() - t_truth
        audit_nodes = np.linspace(-1, 1, args.audit_angles)
        ex_result = {
            "domain": [ex.x0, ex.x1],
            "nx": model.nx,
            "ndof": model.ndof,
            "truth_seconds": truth_seconds,
            "runs": {},
        }
        for j, method in enumerate(methods):
            print(f"  {method}", flush=True)
            run = train(model, method, args.seed + 1000 * idx + j, args.target_dim, args.adam_lr)
            metrics = audit(model, run, truth_rho, audit_nodes)
            compact = serializable(run)
            compact["audit"] = metrics
            ex_result["runs"][method] = compact
            print(
                f"    max={metrics['max_relative_f']:.3e}, mean={metrics['mean_relative_f']:.3e}, "
                f"indicator_ratio={np.mean([h['indicator_ratio'] for h in run['history']]):.3f}",
                flush=True,
            )
        results["examples"][ex.name] = ex_result
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
