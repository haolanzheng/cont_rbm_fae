"""Normalized projected ascent on [-1,1]^d; return the best visited point."""
import numpy as np


def scores(H, Z):
    return np.sum((Z @ H) * Z, axis=-1)


def search(H, starts, alpha=.6, T=40, threshold=None, diagnostics=True):
    """Reuse H z for both value and derivative; keep audits optional."""
    if not diagnostics and threshold is not None:
        raise ValueError("Probability audits require diagnostics=True")
    z = np.atleast_2d(starts).copy()
    best = z.copy()
    best_value = np.full(len(z), -np.inf)
    ell = np.linalg.norm(H)
    if diagnostics:
        times = {0, 1, 5, 10, 20, T}
        history = []
        minimum_increment = 0.
        previous = None
    for t in range(T + 1):
        hz = z @ H
        current = np.sum(hz * z, axis=-1)
        improved = current > best_value
        best[improved] = z[improved]
        best_value = np.maximum(best_value, current)
        if diagnostics:
            if previous is not None:
                minimum_increment = min(minimum_increment, float(np.min(current - previous)))
            previous = current
            if t in times:
                history.append({
                    'T': t,
                    'hits': None if threshold is None else int(np.sum(best_value >= threshold)),
                    'mean_best_value': float(best_value.mean()),
                    'mean_boundary_coordinates': float(np.mean(np.abs(z) >= 1 - 1e-12)),
                })
        if t == T:
            break
        grad = 2 * hz / ell if ell else np.zeros_like(z)
        eta = alpha / np.sqrt(1 + np.sum(grad * grad, axis=1, keepdims=True))
        z = np.clip(z + eta * grad, -1, 1)
    data = {}
    if diagnostics:
        data = {'history': history, 'min_increment': minimum_increment,
                'feasibility_defect': float(np.max(np.maximum(np.abs(z) - 1, 0)))}
    return (best[0] if np.ndim(starts) == 1 else best), data
