"""Residual quadratic form and exact energy-error audit."""

from __future__ import annotations
import math
import numpy as np
import scipy.linalg as la
from .hf import Problem


def a_orthonormal_append(V, u, A):
    from contgreedy.rom import orthonormal_append

    result, norm = orthonormal_append(V, u, metric=lambda x: A @ np.asarray(x))
    return np.asarray(result), norm


def matrices(
    problem: Problem, V: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if V.shape[1]:
        C = V.T @ problem.B
        R = problem.B - (problem.A @ V) @ C
        G = problem.solution_gram - C.T @ C
    else:
        R = problem.B.copy()
        G = problem.solution_gram.copy()
    XR = np.column_stack([problem.solve_X(R[:, j]) for j in range(problem.d)])
    H = 0.5 * (R.T @ XR + XR.T @ R)
    G = 0.5 * (G + G.T)
    return H, G, R


def true_worst(G: np.ndarray) -> float:
    return math.sqrt(
        max(
            float(la.eigvalsh(G, subset_by_index=[G.shape[0] - 1, G.shape[0] - 1])[0]),
            0.0,
        )
    )


def equivalence_constants(H: np.ndarray, G: np.ndarray) -> tuple[float, float]:
    wg, qg = la.eigh(G)
    keep = wg > max(1e-13, 1e-10 * wg[-1])
    if not np.any(keep):
        return 1.0, 1.0
    W = qg[:, keep] / np.sqrt(wg[keep])[None, :]
    vals = la.eigvalsh(0.5 * (W.T @ H @ W + W.T @ H.T @ W))
    return math.sqrt(max(vals[0], 0.0)), math.sqrt(max(vals[-1], 0.0))
