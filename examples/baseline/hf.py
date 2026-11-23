"""Spectral benchmark: physical assembly and fixed-operator truth solves."""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def elliptic_matrix(n: int, contrast: float) -> sp.csc_matrix:
    """Five-point discretization of -div(a grad), a=1+c sin(2pi x)sin(2pi y)."""
    h = 1.0 / (n + 1)
    rows, cols, vals = [], [], []

    def idx(i: int, j: int) -> int:
        return i * n + j

    def coeff(x: float, y: float) -> float:
        return 1.0 + contrast * math.sin(2.0 * math.pi * x) * math.sin(
            2.0 * math.pi * y
        )

    for i in range(n):
        x = (i + 1) * h
        for j in range(n):
            y = (j + 1) * h
            p = idx(i, j)
            diag = 0.0
            for di, dj, xe, ye in (
                (-1, 0, x - 0.5 * h, y),
                (1, 0, x + 0.5 * h, y),
                (0, -1, x, y - 0.5 * h),
                (0, 1, x, y + 0.5 * h),
            ):
                w = coeff(xe, ye) / (h * h)
                diag += w
                ii, jj = i + di, j + dj
                if 0 <= ii < n and 0 <= jj < n:
                    rows.append(p)
                    cols.append(idx(ii, jj))
                    vals.append(-w)
            rows.append(p)
            cols.append(p)
            vals.append(diag)
    return sp.coo_matrix((vals, (rows, cols)), shape=(n * n, n * n)).tocsc()


def sine_loads(n: int, d: int) -> np.ndarray:
    h = 1.0 / (n + 1)
    x = np.arange(1, n + 1) * h
    xx, yy = np.meshgrid(x, x, indexing="ij")
    pairs = sorted(
        ((kx, ky) for kx in range(1, n + 1) for ky in range(1, n + 1)),
        key=lambda t: (t[0] * t[0] + t[1] * t[1], t[0], t[1]),
    )[:d]
    b = np.column_stack(
        [
            2.0 * np.sin(math.pi * kx * xx).ravel() * np.sin(math.pi * ky * yy).ravel()
            for kx, ky in pairs
        ]
    )
    b /= np.linalg.norm(b, axis=0, keepdims=True)
    return b


def inv_sqrt_spd(g: np.ndarray) -> np.ndarray:
    w, q = la.eigh(0.5 * (g + g.T))
    if w[0] <= 1e-13 * w[-1]:
        raise RuntimeError("Load Gram matrix is numerically rank deficient")
    return (q * (1.0 / np.sqrt(w))) @ q.T


@dataclass
class Problem:
    n_grid: int
    d: int
    contrast: float
    spectrum_ratio: float
    A: sp.csc_matrix
    X: sp.csc_matrix
    B: np.ndarray
    S: np.ndarray
    solution_gram: np.ndarray
    singular_values: np.ndarray
    solve_X: object

    @classmethod
    def build(
        cls,
        n_grid: int,
        d: int,
        contrast: float,
        spectrum_ratio: float,
        load_mode: str = "controlled",
    ) -> "Problem":
        A = elliptic_matrix(n_grid, contrast)
        X = elliptic_matrix(n_grid, 0.0)
        solve_A = spla.factorized(A)
        solve_X = spla.factorized(X)
        b0 = sine_loads(n_grid, d)
        s0 = np.column_stack([solve_A(b0[:, j]) for j in range(d)])
        gram = b0.T @ s0
        if load_mode == "controlled":
            b_orth = b0 @ inv_sqrt_spd(gram)
            target_sing = spectrum_ratio ** np.arange(d, dtype=float)
            B = b_orth * target_sing[None, :]
            S = np.column_stack([solve_A(B[:, j]) for j in range(d)])
            solution_gram = 0.5 * (B.T @ S + S.T @ B)
            err = la.norm(solution_gram - np.diag(target_sing * target_sing), ord=2)
            if err > 2e-9:
                raise RuntimeError(f"Energy whitening failed: {err:g}")
        elif load_mode == "raw":
            B = b0
            S = s0
            solution_gram = 0.5 * (B.T @ S + S.T @ B)
        else:
            raise ValueError(f"Unknown load mode {load_mode}")
        sing = np.sqrt(np.maximum(la.eigvalsh(solution_gram)[::-1], 0.0))
        return cls(
            n_grid,
            d,
            contrast,
            spectrum_ratio,
            A,
            X,
            B,
            S,
            solution_gram,
            sing,
            solve_X,
        )
