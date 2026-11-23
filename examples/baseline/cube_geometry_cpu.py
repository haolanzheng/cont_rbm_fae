"""CPU searches and exact finite-vertex audits for the parameter cube [-1,1]^d.

No eigenvalue formula for the sphere is used for cube maxima or widths.
"""
from __future__ import annotations

import numpy as np


def vertices(d):
    return 2.0 * ((np.arange(1 << d)[:, None] >> np.arange(d)) & 1) - 1.0


def values(H, points):
    return np.einsum("ij,ij->i", points, points @ H.T)


def cube_search(H, starts, steps=100, omega=0.5, checkpoints=(), threshold=None):
    """Relaxed projected ascent of J/||H||_F with base step eta=1/2."""
    points = np.atleast_2d(np.asarray(starts, dtype=np.float64)).copy()
    scale = np.linalg.norm(H, "fro")
    best = points.copy()
    best_values = values(H, points)
    hit_time = np.full(len(points), -1, dtype=int)
    snapshots = {}
    min_increment = np.inf
    previous = best_values.copy()
    active = np.zeros(len(points), dtype=bool)
    for t in range(steps + 1):
        scores = values(H, points)
        if t:
            min_increment = min(min_increment, float(np.min(scores - previous)))
        previous = scores.copy()
        improved = scores > best_values
        best[improved] = points[improved]
        best_values[improved] = scores[improved]
        if threshold is not None:
            hit_time[(hit_time < 0) & (best_values >= threshold)] = t
        if t in checkpoints:
            snapshots[str(t)] = {"hits": int(np.sum(hit_time >= 0)),
                                  "q_hat": float(np.mean(hit_time >= 0)),
                                  "projection_active_fraction": float(np.mean(active))}
        if t == steps:
            break
        trial = points + points @ H.T / scale if scale else points.copy()
        active |= np.max(np.abs(trial), axis=1) > 1
        points = (1 - omega) * points + omega * np.clip(trial, -1, 1)
    return (best[0] if np.ndim(starts) == 1 else best), {
        "snapshots": snapshots, "minimum_increment": min_increment if steps else 0.0,
        "max_feasibility_defect": float(np.max(np.maximum(np.abs(points)-1, 0))),
        "h_actions_per_start": steps+1,
    }


def width_bounds(G0, cube_vertices):
    """Certified HF Kolmogorov-width brackets.

    The lower bound is the tail trace, from the average over Rademacher
    vertices. The upper bound is the exact error of a left-singular subspace,
    evaluated on every vertex. The bounds coincide for orthogonal generators.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(G0)
    lower, upper = [], []
    d = len(G0)
    for n in range(d):
        tail = np.maximum(eigenvalues[:d-n], 0)
        matrix = (eigenvectors[:, :d-n] * tail) @ eigenvectors[:, :d-n].T
        lower.append(float(np.sqrt(np.sum(tail))))
        upper.append(float(np.sqrt(max(np.max(values(matrix, cube_vertices)), 0))))
    return np.array(lower), np.array(upper)
