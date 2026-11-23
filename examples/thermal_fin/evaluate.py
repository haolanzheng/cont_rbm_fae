"""Thermal-fin error and local-optimality audits."""

from __future__ import annotations
import numpy as np
from .rom import RB, PoolCache
from .sampling import kkt


def uncached_values(rb, z, batch=128):
    n = rb.n
    q = rb.p.q
    result = []
    for k in range(0, len(z), batch):
        c = rb.p.coeff(z[k : k + batch])
        a = np.einsum("bq,qij->bij", c, rb.A[:, :n, :n], optimize=True)
        y = np.linalg.solve(a, np.broadcast_to(rb.fr[:n], (len(c), n))[..., None])[
            ..., 0
        ]
        xx = (y[:, :, None] * c[:, None, :]).reshape(len(c), n * q)
        num = (
            rb.p.ff
            - 2 * (xx @ rb.h[: n * q])
            + np.sum((xx @ rb.G[: n * q, : n * q]) * xx, axis=1)
        )
        result.extend(np.log(np.maximum(num, 1e-290)) - np.log(np.sum(y * y, axis=1)))
    return np.asarray(result)


def local_audit(rb, z, j, g):
    active = np.abs(z) > 1 - 1e-8
    strong = active & (g * z > 1e-5)
    free = np.where(~strong)[0]
    pg = kkt(z, g)
    lam = None
    hess = np.empty((len(free), len(free)))
    for i, k in enumerate(free):
        ep = np.zeros(len(z))
        ep[k] = 2e-5
        hess[:, i] = (rb.fg(z + ep)[1][free] - rb.fg(z - ep)[1][free]) / (4e-5)
    if len(free):
        lam = float(np.linalg.eigvalsh((hess + hess.T) / 2)[-1])
    return {
        "projected_gradient_inf": pg,
        "critical_span_dimension": len(free),
        "max_hessian_eigenvalue": lam,
        "numerical_local_max": bool(pg <= 1e-5 and (lam is None or lam < -1e-6)),
        "first_order_stationary": bool(pg <= 1e-5),
    }


def evaluate(p, V, z, u, norms, ranks):
    coeff = p.coeff(z)
    ar = np.array([V.T @ (a @ V) for a in p.parts])
    fr = V.T @ p.f
    cross = (p.X @ V).T @ u.T
    ans = {}
    for n in ranks:
        err = []
        for k in range(0, len(z), 128):
            c = coeff[k : k + 128]
            a = np.einsum("bq,qij->bij", c, ar[:, :n, :n], optimize=True)
            y = np.linalg.solve(a, np.broadcast_to(fr[:n], (len(c), n))[..., None])[
                ..., 0
            ]
            nn = norms[k : k + len(c)]
            v = (
                nn
                - 2 * np.einsum("bi,ib->b", y, cross[:n, k : k + len(c)])
                + np.sum(y * y, axis=1)
            )
            err.extend(np.sqrt(np.maximum(v, 0) / nn))
        e = np.asarray(err)
        ans[str(n)] = {
            "max": float(e.max()),
            "uniform_max": float(e[:4096].max()),
            "uniform_mean": float(e[:4096].mean()),
            "uniform_q99": float(np.quantile(e[:4096], 0.99)),
            "vertices_max": float(e[4096:4352].max()) if len(e) > 4096 else None,
            "tilted_max": float(e[4352:].max()) if len(e) > 4352 else None,
            "worst_index": int(e.argmax()),
        }
    return ans


def validate(p):
    rb = RB(p, 5)
    rb.add(p.fom(np.zeros(p.d)))
    rng = np.random.default_rng(641)
    for _ in range(3):
        rb.add(p.fom(rng.uniform(-1, 1, p.d)))
    z = rng.uniform(-0.7, 0.7, p.d)
    j, g = rb.fg(z)
    fd = []
    for k in range(p.d):
        e = np.zeros(p.d)
        e[k] = 2e-6
        fd.append((rb.fg(z + e, False) - rb.fg(z - e, False)) / 4e-6)
    pc = PoolCache(rb, rng.uniform(-1, 1, (23, p.d)))
    vp = pc.values()
    vs = np.array([rb.fg(zz, False) for zz in pc.z])
    rel = abs(np.exp(j) - rb.explicit(z)) / rb.explicit(z)
    out = {
        "dofs": p.n,
        "q_raw": 49,
        "q_compressed": p.q,
        "gradient_relative_error": float(np.linalg.norm(g - fd) / np.linalg.norm(g)),
        "gram_vs_explicit_relative_error": rel,
        "pool_vs_scalar_log_error": float(np.max(abs(vp - vs))),
        "orthogonality": float(
            np.linalg.norm(rb.V[:, : rb.n].T @ (p.X @ rb.V[:, : rb.n]) - np.eye(rb.n))
        ),
        "root_flux_integral": float(p.f.sum()),
    }
    assert (
        out["gradient_relative_error"] < 1e-5
        and rel < 1e-8
        and out["pool_vs_scalar_log_error"] < 1e-7
    )
    return out
