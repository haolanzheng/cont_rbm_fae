"""Low-diffusion reaction-field stress test for continuous greedy.

This intentionally audits whether a fixed-Riesz residual remains aligned with
the true relative deployment error when continuous optimization explores tails.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from contgreedy.sampling.host import T_INNER, adam_ascent


class ReactionDiffusion:
    def __init__(self, n=127, d=64, eps=2e-4, alpha=1.2, seed=5):
        self.n, self.d, self.eps, self.alpha = n, d, eps, alpha
        self.h = 1.0 / (n + 1)
        k = np.diag(np.full(n, 2.0)) - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)
        self.k = k / self.h
        x = np.arange(1, n + 1) * self.h
        modes = 12
        phi = np.stack([np.sin(np.pi * (j + 1) * x) / (j + 1) ** 0.8
                        for j in range(modes)], axis=1)
        rng = np.random.default_rng(seed)
        mix = rng.normal(size=(modes, d))
        mix /= np.linalg.norm(mix, axis=1, keepdims=True)
        b = phi @ mix
        b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-15
        self.b = b
        self.f = np.ones(n) * self.h
        self.xnorm = eps * self.k + self.h * np.eye(n)
        self.xcf = cho_factor(self.xnorm)

    def matrix_coeff(self, z):
        c = np.exp(self.alpha * (self.b @ z))
        return self.eps * self.k + np.diag(self.h * c), c

    def fom(self, z):
        a, _ = self.matrix_coeff(z)
        return np.linalg.solve(a, self.f)

    def add(self, v, u):
        if v.shape[1]:
            u -= v @ (v.T @ (self.xnorm @ u))
            u -= v @ (v.T @ (self.xnorm @ u))
        nu = np.sqrt(max(u @ (self.xnorm @ u), 0))
        return np.column_stack((v, u / nu)) if nu > 1e-12 else v

    def indicator_grad(self, z, v, gradient=True):
        a, c = self.matrix_coeff(z)
        ar = v.T @ a @ v
        y = np.linalg.solve(ar, v.T @ self.f)
        u = v @ y
        r = self.f - a @ u
        w = cho_solve(self.xcf, r)
        q = float(r @ w)
        if not gradient:
            return q
        # Columns are A_j u.
        gmat = (self.h * c * u)[:, None] * self.b * self.alpha
        corr = (a @ v) @ np.linalg.solve(ar, v.T @ gmat)
        dr = -gmat + corr
        return q, 2.0 * dr.T @ w

    def true_error(self, z, v, uf=None):
        a, _ = self.matrix_coeff(z)
        if uf is None:
            uf = np.linalg.solve(a, self.f)
        ur = v @ np.linalg.solve(v.T @ a @ v, v.T @ self.f)
        e = uf - ur
        return float(np.sqrt(e @ (self.xnorm @ e)) /
                     np.sqrt(uf @ (self.xnorm @ uf)))


def train(p, method, rank, pool, seed, learning_rate=0.08):
    rng = np.random.default_rng(seed)
    v = p.add(np.empty((p.n, 0)), p.fom(np.zeros(p.d)))
    selected = []
    start = time.perf_counter()
    for _ in range(1, rank):
        if method == "continuous":
            z0 = rng.uniform(-1, 1, p.d)
            fg = lambda z: p.indicator_grad(z, v, True)
            z, q = adam_ascent(fg, z0, lambda x: np.clip(x, -1, 1),
                               steps=T_INNER, lr=learning_rate)
        else:
            zz = rng.uniform(-1, 1, (pool, p.d))
            qs = np.array([p.indicator_grad(z, v, False) for z in zz])
            ii = int(np.argmax(qs)); z, q = zz[ii], float(qs[ii])
        selected.append(q)
        v = p.add(v, p.fom(z))
    return v, time.perf_counter() - start, selected


def audit(p, trained, n_random=500, paths=24, seed=909):
    rng = np.random.default_rng(seed)
    zall = [z for z in rng.uniform(-1, 1, (n_random, p.d))]
    # Build a shared adversarial audit set from every trained basis.
    for v, _, _ in trained.values():
        for z0 in rng.uniform(-1, 1, (paths, p.d)):
            fg = lambda z: p.indicator_grad(z, v, True)
            z, _ = adam_ascent(fg, z0, lambda x: np.clip(x, -1, 1),
                               steps=T_INNER, lr=0.08)
            zall.append(z)
    fom = [p.fom(z) for z in zall]
    out = {}
    for name, (v, seconds, _) in trained.items():
        err = np.array([p.true_error(z, v, uf) for z, uf in zip(zall, fom)])
        ind = np.sqrt(np.array([p.indicator_grad(z, v, False) for z in zall]))
        out[name] = {
            "random_mean": float(np.mean(err[:n_random])),
            "random_q99": float(np.quantile(err[:n_random], 0.99)),
            "shared_adversarial_max": float(np.max(err)),
            "shared_indicator_max": float(np.max(ind)),
            "indicator_error_corr": float(np.corrcoef(ind, err)[0, 1]),
            "seconds": seconds,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dimension", type=int, default=64)
    ap.add_argument("--rank", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--audit-paths", type=int, default=24)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--learning-rate", type=float, default=0.08)
    ap.add_argument("--output", type=Path, default=Path("reaction_result.json"))
    args = ap.parse_args()
    p = ReactionDiffusion(d=args.dimension, alpha=args.alpha)
    trained = {}
    for name, pool in (("pool100", 100), ("pool10000", 10_000), ("continuous", 0)):
        kind = "continuous" if name == "continuous" else "pool"
        trained[name] = train(p, kind, args.rank, pool, args.seed,
                              learning_rate=args.learning_rate)
        print(name, "trained", trained[name][1], flush=True)
    result = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "T": T_INNER,
              "audit": audit(p, trained, args.test, args.audit_paths)}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["audit"], indent=2))


if __name__ == "__main__":
    main()
