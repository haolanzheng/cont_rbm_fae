"""Thermal-fin P1 assembly; SciPy sparse LU remains an offline operation."""

from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from .domain import physical


class Fin:
    def __init__(self, d=11, cells=32):
        assert d % 2 == 1 and 1 <= (d - 1) // 2 <= 16
        self.d = d
        self.m = (d - 1) // 2
        self.cells = cells
        self.base_k = np.array([0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.2, 1.2] + [1.0] * 8)
        self.base_b = np.array([0.75, 2.5, 2.5, 0.25] * 4 + [1.0])
        self.lo = np.r_[np.full(self.m, 0.1), 0.01, [0.1] * self.m]
        self.hi = np.r_[
            np.full(self.m, 10.0),
            5.0,
            [5.0 if j % 4 == 0 else 2.5 if j % 4 == 3 else 10.0 for j in range(self.m)],
        ]
        self.log = np.arange(d) <= self.m
        self.center = self.physical(np.zeros(d))[0]
        self._mesh_parts()
        self._compress()
        self.X = self.matrix_from_coeff(
            self.coeff_physical(
                np.r_[self.base_k[: self.m], 0.1, self.base_b[: self.m]]
            )
        )
        self.riesz = splu(self.X.tocsc())
        self.xf = self.riesz.solve(self.f)
        self.ff = float(self.f @ self.xf)

    def physical(self, z):
        return physical(z, self.lo, self.hi, self.log)

    def _mesh_parts(self):
        h = 1 / self.cells
        nx = int(6 * self.cells)
        ny = 4 * self.cells
        # Integer coordinates x in [-2.5,3.5], y in [0,4].
        x0 = int(2.5 * self.cells)
        x1 = x0 + self.cells
        rects = []
        for i in range(4):
            ya = int((i + 0.75) * self.cells)
            yb = (i + 1) * self.cells
            rects.extend(
                [
                    (2 * i, x1, nx, ya, yb),
                    (2 * i + 1, 0, x0, ya, yb),
                    (8 + 2 * i, x0, x1, i * self.cells, ya),
                    (9 + 2 * i, x0, x1, ya, yb),
                ]
            )
        cells = {}
        for r, xa, xb, ya, yb in rects:
            for j in range(ya, yb):
                for i in range(xa, xb):
                    cells[i, j] = r
        verts = sorted(
            {
                v
                for i, j in cells
                for v in [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
            },
            key=lambda v: (v[1], v[0]),
        )
        ids = {v: k for k, v in enumerate(verts)}
        self.xy = np.asarray(verts) * h
        self.n = len(verts)
        self.f = np.zeros(self.n)
        rows = [[] for _ in range(49)]
        cols = [[] for _ in range(49)]
        vals = [[] for _ in range(49)]

        def add(q, ii, a):
            for i, u in enumerate(ii):
                for j, v in enumerate(ii):
                    if a[i, j] != 0:
                        rows[q].append(u)
                        cols[q].append(v)
                        vals[q].append(a[i, j])

        gradients = [
            np.array([[-1, 0], [1, -1], [0, 1]]) / h,
            np.array([[0, -1], [1, 0], [-1, 1]]) / h,
        ]
        for (i, j), r in cells.items():
            a, b, c, e = [
                ids[v] for v in [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            ]
            for ii, g in zip([(a, b, c), (a, c, e)], gradients):
                add(2 * r, ii, 0.5 * h * h * np.outer(g[:, 0], g[:, 0]))
                add(2 * r + 1, ii, 0.5 * h * h * np.outer(g[:, 1], g[:, 1]))
            for di, dj, edge in [
                (0, -1, [(i, j), (i + 1, j)]),
                (0, 1, [(i, j + 1), (i + 1, j + 1)]),
                (-1, 0, [(i, j), (i, j + 1)]),
                (1, 0, [(i + 1, j), (i + 1, j + 1)]),
            ]:
                if (i + di, j + dj) in cells:
                    continue
                ii = [ids[v] for v in edge]
                if j == 0 and dj == -1:
                    self.f[ii] += h / 2
                    continue
                if dj:
                    q = 32 + r if r < 8 else 40
                else:
                    q = 45 + r // 2 if r < 8 else 41 + (r - 8) // 2
                add(q, ii, h * np.array([[2, 1], [1, 2]]) / 6)
        self.rawparts = [
            sparse.coo_matrix((v, (r, c)), shape=(self.n, self.n)).tocsr()
            for r, c, v in zip(rows, cols, vals)
        ]
        assert self.n == 10017 if self.cells == 32 else True
        assert abs(self.f.sum() - 1) < 1e-12

    def _compress(self):
        # Each coefficient is exactly a monomial in the active physical inputs.
        # Merge identical monomials; this is algebraic, with no truncation.
        centerk = self.base_k.copy()
        centerk[: self.m] = self.center[: self.m]
        centerb = self.base_b.copy()
        centerb[: self.m] = self.center[self.m + 1 :]
        entries = []

        def expvar(e, kind, index, power):
            if kind == "k" and index < self.m:
                e[index] += power
            elif kind == "bi":
                e[self.m] += power
            elif kind == "b" and index < self.m:
                e[self.m + 1 + index] += power

        for r in range(16):
            if r < 8:
                i = r // 2
                ix = 4 * i + (1 if r % 2 == 0 else 2)
                iy = 4 * i + 3
            else:
                i = (r - 8) // 2
                ix = 16
                iy = 4 * i + (0 if r % 2 == 0 else 3)
            sx = centerb[ix] / self.base_b[ix]
            sy = centerb[iy] / self.base_b[iy]
            for direction in [0, 1]:
                e = np.zeros(self.d, dtype=int)
                expvar(e, "k", r, 1)
                expvar(e, "b", ix, -1 if direction == 0 else 1)
                expvar(e, "b", iy, 1 if direction == 0 else -1)
                entries.append(
                    (tuple(e), centerk[r] * (sy / sx if direction == 0 else sx / sy))
                )
        for r in range(9):
            ix = 4 * (r // 2) + (1 if r % 2 == 0 else 2) if r < 8 else 16
            e = np.zeros(self.d, dtype=int)
            expvar(e, "bi", 0, 1)
            expvar(e, "b", ix, 1)
            entries.append(
                (tuple(e), self.center[self.m] * centerb[ix] / self.base_b[ix])
            )
        for j in range(8):
            iy = 4 * j if j < 4 else 4 * (j - 4) + 3
            e = np.zeros(self.d, dtype=int)
            expvar(e, "bi", 0, 1)
            expvar(e, "b", iy, 1)
            entries.append(
                (tuple(e), self.center[self.m] * centerb[iy] / self.base_b[iy])
            )
        groups = {}
        for a, (key, c) in zip(self.rawparts, entries):
            groups[key] = groups.get(key, 0) + c * a
        self.exps = np.array(list(groups))
        self.parts = [p.tocsr() for p in groups.values()]
        self.q = len(self.parts)
        pattern = sum(self.parts).tocsr()
        self.indices = pattern.indices
        self.indptr = pattern.indptr
        rr = np.repeat(np.arange(self.n), np.diff(pattern.indptr))
        keys = rr * self.n + pattern.indices
        self.data = np.zeros((self.q, len(pattern.data)))
        for i, p in enumerate(self.parts):
            pr = np.repeat(np.arange(self.n), np.diff(p.indptr))
            loc = np.searchsorted(keys, pr * self.n + p.indices)
            self.data[i, loc] = p.data
        self.entries = entries

    def coeff_physical(self, p):
        return np.exp(np.log(np.asarray(p) / self.center) @ self.exps.T)

    def coeff(self, z, grad=False):
        p, dp = self.physical(z)
        c = self.coeff_physical(p)
        return (c, c[:, None] * self.exps * (dp / p)[None, :]) if grad else c

    def matrix_from_coeff(self, c):
        return sparse.csr_matrix(
            (c @ self.data, self.indices, self.indptr), shape=(self.n, self.n)
        )

    def fom(self, z):
        return splu(self.matrix_from_coeff(self.coeff(z)).tocsc()).solve(self.f)
