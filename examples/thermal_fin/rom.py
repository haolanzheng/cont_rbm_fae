"""Offline basis/Gram updates, reference NumPy evaluation and JAX online ROM."""

from __future__ import annotations
import numpy as np
from scipy import linalg


class ReferenceRB:
    def __init__(self, p, capacity):
        self.p = p
        self.capacity = capacity
        self.n = 0
        q = p.q
        h = p.n
        self.V = np.zeros((h, capacity))
        self.A = np.zeros((q, capacity, capacity))
        self.fr = np.zeros(capacity)
        self.B = np.zeros((h, capacity * q))
        self.W = np.zeros_like(self.B)
        self.G = np.zeros((capacity * q, capacity * q))
        self.h = np.zeros(capacity * q)

    def add(self, u):
        p = self.p
        n = self.n
        q = p.q
        v = u.copy()
        vv = self.V[:, :n]
        for _ in range(2):
            v -= vv @ (vv.T @ (p.X @ v))
        vn = np.sqrt(v @ (p.X @ v))
        if vn < 1e-12:
            raise RuntimeError("Dependent enrichment")
        v /= vn
        self.V[:, n] = v
        self.fr[n] = v @ p.f
        bv = np.column_stack([a @ v for a in p.parts])
        w = p.riesz.solve(bv)
        k = n * q
        self.B[:, k : k + q] = bv
        self.W[:, k : k + q] = w
        g = self.B[:, : k + q].T @ w
        self.G[: k + q, k : k + q] = g
        self.G[k : k + q, : k + q] = g.T
        self.h[k : k + q] = bv.T @ p.xf
        av = self.V[:, : n + 1].T @ bv
        self.A[:, : n + 1, n] = av.T
        self.A[:, n, : n + 1] = av.T
        self.n += 1

    def fg(self, z, gradient=True):
        p = self.p
        n = self.n
        q = p.q
        c, dc = p.coeff(z, True)
        a = np.einsum("q,qij->ij", c, self.A[:, :n, :n])
        cf = linalg.cho_factor(a, check_finite=False)
        y = linalg.cho_solve(cf, self.fr[:n], check_finite=False)
        x = (y[:, None] * c).ravel()
        g = self.G[: n * q, : n * q] @ x
        num = p.ff - 2 * self.h[: n * q] @ x + x @ g
        den = y @ y
        num = max(num, 1e-290)
        j = np.log(num) - np.log(den)
        if not gradient:
            return j
        w = (g - self.h[: n * q]).reshape(n, q)
        ja = 2 * (w @ c) / num - 2 * y / den
        lam = linalg.cho_solve(cf, ja, check_finite=False)
        jc = 2 * (y @ w) / num - np.einsum("i,qij,j->q", lam, self.A[:, :n, :n], y)
        return j, dc.T @ jc

    def explicit(self, z):
        p = self.p
        n = self.n
        c = p.coeff(z)
        a = np.einsum("q,qij->ij", c, self.A[:, :n, :n])
        y = linalg.solve(a, self.fr[:n], assume_a="pos")
        r = p.f - p.matrix_from_coeff(c) @ (self.V[:, :n] @ y)
        return float((r @ p.riesz.solve(r)) / (y @ y))


class PoolCache:
    def __init__(self, rb, z):
        self.rb = rb
        self.z = z
        self.c = rb.p.coeff(z)
        self.cc = (self.c[:, :, None] * self.c[:, None, :]).reshape(len(z), -1)
        n = rb.capacity
        m = len(z)
        self.A = np.empty((m, n, n))
        self.G = np.empty_like(self.A)
        self.h = np.empty((m, n))
        self.n = 0

    def values(self):
        rb = self.rb
        q = rb.p.q
        n = rb.n
        while self.n < n:
            k = self.n
            av = self.c @ rb.A[:, : k + 1, k]
            self.A[:, : k + 1, k] = av
            self.A[:, k, : k + 1] = av
            block = (
                rb.G[k * q : (k + 1) * q, : (k + 1) * q]
                .reshape(q, k + 1, q)
                .transpose(1, 0, 2)
                .reshape(k + 1, q * q)
            )
            gv = self.cc @ block.T
            self.G[:, : k + 1, k] = gv
            self.G[:, k, : k + 1] = gv
            self.h[:, k] = self.c @ rb.h[k * q : (k + 1) * q]
            self.n += 1
        y = np.linalg.solve(
            self.A[:, :n, :n], np.broadcast_to(rb.fr[:n], (len(self.z), n))[..., None]
        )[..., 0]
        num = (
            rb.p.ff
            - 2 * np.einsum("bi,bi->b", self.h[:, :n], y)
            + np.einsum("bi,bij,bj->b", y, self.G[:, :n, :n], y, optimize=True)
        )
        return np.log(np.maximum(num, 1e-290)) - np.log(np.sum(y * y, axis=1))


from contgreedy._jax import jax, jnp


@jax.jit
def log_relative_residual(z, lo, hi, log_mask, center, exps, A, fr, G, h, ff):
    unit = (z + 1) * 0.5
    physical = jnp.where(
        log_mask, jnp.exp(jnp.log(lo) + jnp.log(hi / lo) * unit), lo + (hi - lo) * unit
    )
    c = jnp.exp(jnp.log(physical / center) @ exps.T)
    matrix = jnp.einsum("q,qij->ij", c, A)
    # Explicit Cholesky preserves the SPD numerical path of the reference.
    from jax.scipy.linalg import cho_solve

    factor = jnp.linalg.cholesky(matrix)
    y = cho_solve((factor, True), fr)
    x = (y[:, None] * c).ravel()
    numerator = ff - 2 * h @ x + x @ G @ x
    return jnp.log(jnp.maximum(numerator, 1e-290)) - jnp.log(y @ y)


value_grad = jax.jit(jax.value_and_grad(log_relative_residual))
batch_values = jax.jit(
    jax.vmap(
        log_relative_residual,
        in_axes=(0, None, None, None, None, None, None, None, None, None, None),
    )
)


class RB(ReferenceRB):
    """JAX float64 online evaluation with unchanged NumPy/SciPy offline data."""

    def freeze(self):
        p = self.p
        n = self.n
        q = p.q
        mask = np.arange(p.d) <= p.m
        return tuple(
            jnp.asarray(a)
            for a in (
                p.lo,
                p.hi,
                mask,
                p.center,
                p.exps,
                self.A[:, :n, :n],
                self.fr[:n],
                self.G[: n * q, : n * q],
                self.h[: n * q],
                p.ff,
            )
        )

    def fg(self, z, gradient=True):
        args = self.freeze()
        if not gradient:
            return float(log_relative_residual(jnp.asarray(z), *args))
        value, gradient = value_grad(jnp.asarray(z), *args)
        return float(value), np.asarray(gradient)

    def values(self, z):
        return np.asarray(batch_values(jnp.asarray(z), *self.freeze()))
