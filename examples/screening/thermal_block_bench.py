"""14-parameter thermal-block benchmark for continuous vs finite-pool greedy.

This follows the PDE and parameter geometry in Nielen, Tse, and Veroy
(arXiv:2410.17938): a 2-by-7 block conductivity field with p_i in [1, 10].
The discretization here is a compact finite-difference analogue so that learning-
rate sweeps and repeated seeds can be run cheaply before a full FEM reproduction.

Search uses only the affine residual indicator.  A full solve is performed after
the next parameter has been selected and in the independent final audit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu, spsolve

from contgreedy.sampling.host import T_INNER, adam_ascent


class ThermalBlock:
    def __init__(self, grid=23, rows=2, cols=7):
        self.grid = grid
        self.rows = rows
        self.cols = cols
        self.d = rows * cols
        self.n = grid * grid
        self.parts = self._assemble_parts()
        self.xnorm = sum(self.parts)
        self.riesz = splu(csc_matrix(self.xnorm))
        self.f = np.full(self.n, 1.0 / (grid + 1) ** 2)

    def _stiffness_from_nodal_kappa(self, kap):
        g = self.grid
        h2 = (g + 1) ** 2
        kap = kap.reshape(g, g)
        row, col, val = [], [], []
        for i in range(g):
            for j in range(g):
                k = i * g + j
                diag = 0.0
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < g and 0 <= jj < g:
                        ell = ii * g + jj
                        weight = 0.5 * (kap[i, j] + kap[ii, jj]) * h2
                        row.append(k); col.append(ell); val.append(-weight)
                        diag += weight
                    else:
                        diag += kap[i, j] * h2
                row.append(k); col.append(k); val.append(diag)
        return csc_matrix((val, (row, col)), shape=(self.n, self.n))

    def _assemble_parts(self):
        g = self.grid
        x = (np.arange(g) + 1) / (g + 1)
        xx, yy = np.meshgrid(x, x, indexing="ij")
        block_col = np.minimum((xx * self.cols).astype(int), self.cols - 1)
        block_row = np.minimum((yy * self.rows).astype(int), self.rows - 1)
        block = block_row * self.cols + block_col
        return [self._stiffness_from_nodal_kappa((block == j).astype(float).ravel())
                for j in range(self.d)]

    def matrix(self, p):
        out = self.parts[0] * float(p[0])
        for coeff, part in zip(p[1:], self.parts[1:]):
            out = out + float(coeff) * part
        return csc_matrix(out)

    def fom(self, p):
        return spsolve(self.matrix(p), self.f)

    def add(self, v, u):
        if v.shape[1]:
            u = u - v @ (v.T @ (self.xnorm @ u))
            u = u - v @ (v.T @ (self.xnorm @ u))
        norm = np.sqrt(max(float(u @ (self.xnorm @ u)), 0.0))
        return np.column_stack((v, u / norm)) if norm > 1e-13 else v

    def cache(self, v):
        av = np.stack([part @ v for part in self.parts])
        arv = np.stack([v.T @ item for item in av])
        return av, arv

    def indicator_grad(self, p, v, cached=None, gradient=True):
        av, arv = cached if cached is not None else self.cache(v)
        ar = np.tensordot(p, arv, axes=(0, 0))
        y = np.linalg.solve(ar, v.T @ self.f)
        av_full = np.tensordot(p, av, axes=(0, 0))
        r = self.f - av_full @ y
        w = self.riesz.solve(r)
        value = float(r @ w)
        if not gradient:
            return value
        grad = np.empty(self.d)
        for j in range(self.d):
            gj = av[j] @ y
            correction = av_full @ np.linalg.solve(ar, v.T @ gj)
            dr = -gj + correction
            grad[j] = 2.0 * dr @ w
        return value, grad

    def true_relative_error(self, p, v, uf=None):
        a = self.matrix(p)
        if uf is None:
            uf = spsolve(a, self.f)
        ur = v @ np.linalg.solve(v.T @ (a @ v), v.T @ self.f)
        e = uf - ur
        return float(np.sqrt(max(e @ (self.xnorm @ e), 0.0)) /
                     np.sqrt(max(uf @ (self.xnorm @ uf), 1e-30)))


def train(problem, method, rank, pool, seed, learning_rate):
    rng = np.random.default_rng(seed)
    v = problem.add(np.empty((problem.n, 0)), problem.fom(np.full(problem.d, 5.5)))
    selected = []
    selected_p = []
    start = time.perf_counter()
    for _ in range(1, rank):
        cached = problem.cache(v)
        if method == "continuous":
            p0 = rng.uniform(1.0, 10.0, problem.d)
            fg = lambda p: problem.indicator_grad(p, v, cached, True)
            p, value = adam_ascent(fg, p0, lambda x: np.clip(x, 1.0, 10.0),
                                   steps=T_INNER, lr=learning_rate)
        else:
            pp = rng.uniform(1.0, 10.0, (pool, problem.d))
            values = np.fromiter(
                (problem.indicator_grad(p, v, cached, False) for p in pp),
                dtype=float, count=pool)
            index = int(np.argmax(values))
            p, value = pp[index], float(values[index])
        selected.append(value)
        selected_p.append(p.tolist())
        v = problem.add(v, problem.fom(p))
    return v, time.perf_counter() - start, selected, selected_p


def audit(problem, trained, random_count=2000, audit_starts=32, seed=9001):
    """Shared audit: random mean/q99 plus corner and adversarial maxima."""
    rng = np.random.default_rng(seed)
    random_points = rng.uniform(1.0, 10.0, (random_count, problem.d))
    all_points = list(random_points)
    corner_indicator_max = {}
    if problem.d <= 16:
        index = np.arange(1 << problem.d, dtype=np.uint32)[:, None]
        bits = ((index >> np.arange(problem.d, dtype=np.uint32)) & 1).astype(float)
        corners = 1.0 + 9.0 * bits
        # Add the most difficult corners for every trained basis to a common
        # true-error audit set; the indicator maximum over all corners is exact.
        for name, (v, _, _, _) in trained.items():
            cache = problem.cache(v)
            values = np.fromiter(
                (problem.indicator_grad(p, v, cache, False) for p in corners),
                dtype=float, count=len(corners))
            corner_indicator_max[name] = float(np.sqrt(values.max()))
            top = np.argpartition(values, -32)[-32:]
            all_points.extend(corners[top])
    # Audit maximization may use multiple starts; training never does.
    for v, _, _, _ in trained.values():
        cache = problem.cache(v)
        for p0 in rng.uniform(1.0, 10.0, (audit_starts, problem.d)):
            fg = lambda p: problem.indicator_grad(p, v, cache, True)
            p, _ = adam_ascent(fg, p0, lambda x: np.clip(x, 1.0, 10.0),
                               steps=300, lr=0.15)
            all_points.append(p)
    fom = [problem.fom(p) for p in all_points]
    result = {}
    for name, (v, seconds, selected, _) in trained.items():
        errors = np.array([problem.true_relative_error(p, v, u)
                           for p, u in zip(all_points, fom)])
        indicators = np.sqrt(np.array([problem.indicator_grad(p, v, None, False)
                                       for p in all_points]))
        result[name] = {
            "random_mean": float(errors[:random_count].mean()),
            "random_q99": float(np.quantile(errors[:random_count], 0.99)),
            "shared_adversarial_max": float(errors.max()),
            "shared_indicator_max": float(indicators.max()),
            "all_corners_indicator_max": corner_indicator_max.get(name),
            "indicator_error_corr": float(np.corrcoef(indicators, errors)[0, 1]),
            "training_seconds": seconds,
            "selected_indicator_squared": selected,
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=23)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--learning-rate", type=float, default=0.15)
    ap.add_argument("--pool-large", type=int, default=16384)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--audit-starts", type=int, default=32)
    ap.add_argument("--output", type=Path, default=Path("thermal_block_result.json"))
    args = ap.parse_args()
    problem = ThermalBlock(grid=args.grid)
    trained = {}
    for name, method, pool in (
        ("pool100", "pool", 100),
        (f"pool{args.pool_large}", "pool", args.pool_large),
        ("continuous", "continuous", 0),
    ):
        trained[name] = train(problem, method, args.rank, pool, args.seed,
                              args.learning_rate)
        print(name, "trained in", trained[name][1], flush=True)
    result = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "T_inner": T_INNER,
        "audit": audit(problem, trained, args.test, args.audit_starts),
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["audit"], indent=2))


if __name__ == "__main__":
    main()
