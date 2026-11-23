"""Continuous-vs-pool greedy benchmarks with a fixed 100-step inner solve.

The script contains:
  1. a linear spectral Poisson sanity check;
  2. a 2-D high-contrast latent-channel diffusion problem;
  3. a 1-D low-diffusion lognormal reaction problem (indicator stress test).

No FOM solution is used inside the greedy search.  A FOM solve is performed only
after a parameter is selected and in the independent final audit.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import splu, spsolve


from contgreedy.sampling.host import T_INNER, adam_ascent, sphere_project, sphere_samples

@dataclass
class GreedyResult:
    method: str
    basis: np.ndarray
    seconds: float
    selected_objective: list[float]
    selected_cap: list[float]


class LatentChannelDiffusion:
    """SPD elliptic problem with rare high-conductivity channels.

    kappa(x,z) = 1 + contrast * sum_k sigmoid((a_k'z-theta)/tau) chi_k(x).
    The dense directions a_k are orthonormal in a high-dimensional latent sphere.
    """

    def __init__(
        self,
        grid=23,
        latent_dim=256,
        channels=8,
        contrast=999.0,
        theta=0.62,
        tau=0.075,
        seed=4,
    ):
        self.grid = grid
        self.n = grid * grid
        self.d = latent_dim
        self.k = channels
        self.contrast = contrast
        self.theta = theta
        self.tau = tau
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.normal(size=(latent_dim, channels)))
        self.directions = q.T
        self.A_parts = self._assemble_parts()
        self.A0 = self.A_parts[0]
        self.riesz = splu(csc_matrix(self.A0))
        self.f = self._make_rhs()

    def _make_rhs(self):
        g = self.grid
        x = np.linspace(0, 1, g + 2)[1:-1]
        xx, yy = np.meshgrid(x, x, indexing="ij")
        f = np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.012)
        return f.ravel() / np.linalg.norm(f)

    def _channel_fields(self):
        g = self.grid
        x = np.linspace(0, 1, g + 2)[1:-1]
        xx, yy = np.meshgrid(x, x, indexing="ij")
        fields = []
        # Rays from the source to separated boundary locations.
        ends = []
        for j in range(self.k):
            side = j % 4
            t = 0.18 + 0.64 * ((j // 4 + 1) / (self.k // 4 + 1))
            if side == 0:
                ends.append((0.0, t))
            elif side == 1:
                ends.append((1.0, t))
            elif side == 2:
                ends.append((t, 0.0))
            else:
                ends.append((t, 1.0))
        p0 = np.array([0.5, 0.5])
        width = 0.030
        for ex, ey in ends:
            p1 = np.array([ex, ey])
            vx, vy = p1 - p0
            den = vx * vx + vy * vy
            tt = np.clip(((xx - p0[0]) * vx + (yy - p0[1]) * vy) / den, 0, 1)
            px = p0[0] + tt * vx
            py = p0[1] + tt * vy
            dist2 = (xx - px) ** 2 + (yy - py) ** 2
            fields.append(np.exp(-0.5 * dist2 / width**2).ravel())
        return fields

    def _stiffness_from_nodal_kappa(self, kap):
        g = self.grid
        h2 = (g + 1) ** 2
        kap = kap.reshape(g, g)
        rows = []
        cols = []
        vals = []
        for i in range(g):
            for j in range(g):
                p = i * g + j
                diagv = 0.0
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < g and 0 <= nj < g:
                        q = ni * g + nj
                        w = 0.5 * (kap[i, j] + kap[ni, nj]) * h2
                        rows.append(p); cols.append(q); vals.append(-w)
                        diagv += w
                    else:
                        diagv += kap[i, j] * h2
                rows.append(p); cols.append(p); vals.append(diagv)
        return csc_matrix((vals, (rows, cols)), shape=(self.n, self.n))

    def _assemble_parts(self):
        parts = [self._stiffness_from_nodal_kappa(np.ones(self.n))]
        for field in self._channel_fields():
            parts.append(self._stiffness_from_nodal_kappa(field))
        return parts

    def activation(self, z):
        s = self.directions @ z
        x = np.clip((s - self.theta) / self.tau, -40, 40)
        g = 1.0 / (1.0 + np.exp(-x))
        gp = g * (1.0 - g) / self.tau
        return s, g, gp

    def matrix(self, z):
        _, g, _ = self.activation(z)
        a = self.A0.copy()
        for coeff, part in zip(self.contrast * g, self.A_parts[1:]):
            a = a + coeff * part
        return csc_matrix(a)

    def fom(self, z):
        return spsolve(self.matrix(z), self.f)

    def orth_add(self, v, u):
        if v.shape[1]:
            u = u - v @ (v.T @ (self.A0 @ u))
            u = u - v @ (v.T @ (self.A0 @ u))
        nu = np.sqrt(max(u @ (self.A0 @ u), 0.0))
        return np.column_stack((v, u / nu)) if nu > 1e-12 else v

    def cache(self, v):
        av = [part @ v for part in self.A_parts]
        arv = [v.T @ item for item in av]
        return av, arv

    def indicator_grad(self, z, v, cached=None, gradient=True):
        av, arv = cached if cached is not None else self.cache(v)
        _, act, actp = self.activation(z)
        alpha = self.contrast * act
        ar = arv[0].copy()
        for c, ark in zip(alpha, arv[1:]):
            ar += c * ark
        y = np.linalg.solve(ar, v.T @ self.f)
        au = av[0] @ y
        for c, avk in zip(alpha, av[1:]):
            au += c * (avk @ y)
        r = self.f - au
        w = self.riesz.solve(r)
        val = float(r @ w)
        if not gradient:
            return val
        # Derivatives first with respect to the K active coordinates.
        full_av = av[0].copy()
        for c, avk in zip(alpha, av[1:]):
            full_av += c * avk
        ds = np.empty(self.k)
        for j in range(self.k):
            scale = self.contrast * actp[j]
            gj = scale * (av[j + 1] @ y)
            corr = full_av @ np.linalg.solve(ar, v.T @ gj)
            dr = -gj + corr
            ds[j] = 2.0 * dr @ w
        grad = self.directions.T @ ds
        return val, grad

    def true_relative_error(self, z, v, uf=None):
        a = self.matrix(z)
        if uf is None:
            uf = spsolve(a, self.f)
        y = np.linalg.solve(v.T @ (a @ v), v.T @ self.f)
        e = uf - v @ y
        den = np.sqrt(uf @ (self.A0 @ uf))
        return float(np.sqrt(max(e @ (self.A0 @ e), 0.0)) / den)


def train_channel(problem, method, rank, pool, seed, density_floor=0.01,
                  learning_rate=0.035):
    rng = np.random.default_rng(seed)
    v = np.empty((problem.n, 0))
    # Common non-rare starting snapshot.
    zc = sphere_project(np.sum(problem.directions, axis=0))
    zc = sphere_project(zc - problem.directions.T @ (problem.directions @ zc))
    v = problem.orth_add(v, problem.fom(zc))
    vals, caps = [], []
    start = time.perf_counter()
    for _ in range(1, rank):
        cached = problem.cache(v)
        if method in ("continuous", "density"):
            z0 = sphere_samples(rng, 1, problem.d)[0]
            if method == "continuous":
                fg = lambda z: problem.indicator_grad(z, v, cached, gradient=True)
            else:
                # Marginal density of the active coordinates under a uniform
                # high-dimensional sphere, with a nonzero floor retaining tails.
                def fg(z):
                    q, gq = problem.indicator_grad(z, v, cached, gradient=True)
                    s = problem.directions @ z
                    core = np.exp(-0.5 * problem.d * float(s @ s))
                    weight = density_floor + (1.0 - density_floor) * core
                    dw_ds = -(1.0 - density_floor) * core * problem.d * s
                    gw = problem.directions.T @ dw_ds
                    return q * weight, weight * gq + q * gw
            z, val = adam_ascent(fg, z0, sphere_project, steps=T_INNER,
                                 lr=learning_rate)
        else:
            zz = sphere_samples(rng, pool, problem.d)
            q = np.array([
                problem.indicator_grad(z, v, cached, gradient=False) for z in zz
            ])
            ii = int(np.argmax(q))
            z, val = zz[ii], float(q[ii])
        vals.append(val)
        caps.append(float(np.max(problem.directions @ z)))
        v = problem.orth_add(v, problem.fom(z))
    return GreedyResult(method, v, time.perf_counter() - start, vals, caps)


def audit_channel(problem, results, n_random=500, n_paths=24, seed=987):
    rng = np.random.default_rng(seed)
    zr = sphere_samples(rng, n_random, problem.d)
    # Include exact caps and deterministic one-channel paths s in [0,1].
    # The orthogonal component does not affect the PDE, but keeps z on the sphere.
    b = rng.normal(size=problem.d)
    b = b - problem.directions.T @ (problem.directions @ b)
    b = sphere_project(b)
    active_paths = []
    for ak in problem.directions:
        for s in np.linspace(0.0, 1.0, 81):
            active_paths.append(s * ak + np.sqrt(max(1.0 - s * s, 0.0)) * b)
    za = np.vstack((zr, problem.directions, -problem.directions, active_paths))
    # Add adversarial candidates generated against every method, then evaluate
    # every basis on the same union (avoids method-specific test sets).
    adversarial = []
    for res in results:
        cached = problem.cache(res.basis)
        for z0 in sphere_samples(rng, n_paths, problem.d):
            fg = lambda z: problem.indicator_grad(z, res.basis, cached, gradient=True)
            z, _ = adam_ascent(fg, z0, sphere_project, steps=T_INNER, lr=0.035)
            adversarial.append(z)
    za = np.vstack((za, adversarial))
    fom = [problem.fom(z) for z in za]
    out = {}
    for res in results:
        err = np.array([
            problem.true_relative_error(z, res.basis, uf)
            for z, uf in zip(za, fom)
        ])
        indicator = np.sqrt(np.array([
            problem.indicator_grad(z, res.basis, gradient=False) for z in za
        ]))
        path_end = n_random + 2 * problem.k + len(active_paths)
        out[f"{res.method}_{len(out)}"] = {
            "random_mean": float(np.mean(err[:n_random])),
            "random_q99": float(np.quantile(err[:n_random], 0.99)),
            "common_max": float(np.max(err)),
            "cap_max": float(np.max(err[n_random:n_random + 2 * problem.k])),
            "active_path_max": float(np.max(err[n_random + 2 * problem.k:path_end])),
            "shared_adversarial_max": float(np.max(err)),
            "shared_indicator_max": float(np.max(indicator)),
            "indicator_error_corr": float(np.corrcoef(indicator, err)[0, 1]),
            "seconds": res.seconds,
            "selected_cap_mean": float(np.mean(res.selected_cap)),
            "selected_cap_max": float(np.max(res.selected_cap)),
        }
    return out


def run_channel(args):
    p = LatentChannelDiffusion(
        grid=args.grid,
        latent_dim=args.dimension,
        channels=args.channels,
        contrast=args.contrast,
        theta=args.theta,
        tau=args.tau,
        seed=4,
    )
    results = []
    methods = [("pool100", 100), ("pool10000", 10_000), ("continuous", 0)]
    if args.include_density:
        methods.append(("density", 0))
    for method, pool in methods:
        kind = method if method in ("continuous", "density") else "pool"
        r = train_channel(p, kind, args.rank, pool, args.seed,
                          density_floor=args.density_floor,
                          learning_rate=args.learning_rate)
        r.method = method
        results.append(r)
        print(method, "trained", r.seconds, "cap", np.max(r.selected_cap), flush=True)
    return {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "T": T_INNER,
        "audit": audit_channel(p, results, n_random=args.test, n_paths=args.audit_paths),
        "selected": {
            r.method: {
                "indicator": r.selected_objective,
                "cap": r.selected_cap,
            }
            for r in results
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dimension", type=int, default=256)
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--grid", type=int, default=23)
    ap.add_argument("--rank", type=int, default=10)
    ap.add_argument("--contrast", type=float, default=999.0)
    ap.add_argument("--theta", type=float, default=0.62)
    ap.add_argument("--tau", type=float, default=0.075)
    ap.add_argument("--density-floor", type=float, default=1e-5)
    ap.add_argument("--learning-rate", type=float, default=0.035)
    ap.add_argument("--include-density", action="store_true")
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--audit-paths", type=int, default=24)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--output", type=Path, default=Path("channel_result.json"))
    args = ap.parse_args()
    result = run_channel(args)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["audit"], indent=2))


if __name__ == "__main__":
    main()
