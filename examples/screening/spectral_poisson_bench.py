"""Exact spectral Poisson benchmark for pool vs one-start 100-step Adam."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from contgreedy.sampling.host import T_INNER, adam_ascent, sphere_project, sphere_samples


def add_vector(v, y):
    if v.shape[1]:
        y = y - v @ (v.T @ y)
        y = y - v @ (v.T @ y)
    ny = np.linalg.norm(y)
    return np.column_stack((v, y / ny)) if ny > 1e-12 else v


def objective_grad(z, sigma, v):
    y = sigma * z
    r = y - v @ (v.T @ y)
    return float(r @ r), 2.0 * sigma * r


def train(d, rank, method, pool, seed, exponent):
    rng = np.random.default_rng(seed)
    # Laplacian eigenvalues grow as j^2; energy-scaled solution singular values
    # are proportional to 1/j for Poisson. exponent permits stronger smoothing.
    sigma = 1.0 / np.arange(1, d + 1, dtype=float) ** exponent
    v = np.empty((d, 0))
    v = add_vector(v, sigma * sphere_samples(rng, 1, d)[0])
    start = time.perf_counter()
    for _ in range(1, rank):
        if method == "continuous":
            z0 = sphere_samples(rng, 1, d)[0]
            fg = lambda z: objective_grad(z, sigma, v)
            z, _ = adam_ascent(fg, z0, sphere_project, steps=T_INNER, lr=0.035)
        else:
            zz = sphere_samples(rng, pool, d)
            yy = zz * sigma
            rr = yy - (yy @ v) @ v.T
            q = np.einsum("ij,ij->i", rr, rr)
            z = zz[int(np.argmax(q))]
        v = add_vector(v, sigma * z)
    return sigma, v, time.perf_counter() - start


def audit(sigma, v):
    d = sigma.size

    def mv(z):
        y = sigma * z
        return sigma * (y - v @ (v.T @ y))

    op = LinearOperator((d, d), matvec=mv, dtype=float)
    lam = eigsh(op, k=1, which="LA", return_eigenvectors=False, tol=1e-10)[0]
    # E_z ||error||^2 for z uniform on the sphere.
    pv = sigma[:, None] * v
    trace = float(sigma @ sigma - np.sum(pv * pv))
    return {"exact_max": float(np.sqrt(max(lam, 0))),
            "sphere_rms": float(np.sqrt(max(trace, 0) / d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dimension", type=int, default=512)
    ap.add_argument("--rank", type=int, default=25)
    ap.add_argument("--exponent", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--output", type=Path, default=Path("spectral_result.json"))
    args = ap.parse_args()
    out = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "T": T_INNER, "methods": {}}
    for name, pool in (("pool100", 100), ("pool10000", 10_000), ("continuous", 0)):
        kind = "continuous" if name == "continuous" else "pool"
        sigma, v, seconds = train(args.dimension, args.rank, kind, pool,
                                  args.seed, args.exponent)
        out["methods"][name] = audit(sigma, v) | {"seconds": seconds}
        print(name, out["methods"][name], flush=True)
    args.output.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
