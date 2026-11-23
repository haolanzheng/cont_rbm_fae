"""Reference host selector for objectives backed by SciPy/C++.

Preserved arithmetic and NumPy RNGs for historical experiments. New JAX-native
objectives use contgreedy.sampling.adam; this callback path is not GPU-batched.
"""

import numpy as np

T_INNER = 100


def adam_ascent(fun_grad, z0, project, steps=T_INNER, lr=0.05):
    """Exactly ``steps`` Adam updates from one initial point."""
    z = z0.copy()
    m = np.zeros_like(z)
    v = np.zeros_like(z)
    best_q = -np.inf
    best_z = z.copy()
    for t in range(1, steps + 1):
        q, g = fun_grad(z)
        if q > best_q:
            best_q, best_z = q, z.copy()
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mh = m / (1.0 - 0.9**t)
        vh = v / (1.0 - 0.999**t)
        z = project(z + lr * mh / (np.sqrt(vh) + 1e-10))
    q, _ = fun_grad(z)
    if q > best_q:
        best_q, best_z = q, z.copy()
    return best_z, float(best_q)


def sphere_project(z):
    nz = np.linalg.norm(z)
    return z / nz if nz > 0 else np.r_[1.0, np.zeros(z.size - 1)]


def sphere_samples(rng, count, d):
    z = rng.normal(size=(count, d))
    return z / np.linalg.norm(z, axis=1, keepdims=True)


class BestObserved:
    """Track the maximum raw objective seen by a host optimizer.

    The caller supplies feasible points (possibly in transformed coordinates).
    Optimizer evaluations are never cached or skipped; ensure() only avoids an
    extra endpoint evaluation when that point was already evaluated. Line-search
    trial points are eligible too, a superset of accepted trajectory iterates.
    """

    def __init__(self, value_grad):
        self.value_grad = value_grad
        self.calls = 0
        self.point = None
        self.value = -np.inf
        self._seen = set()

    def evaluate(self, point):
        point = np.asarray(point, dtype=float)
        value, gradient = self.value_grad(point)
        self.calls += 1
        self._seen.add((point.shape, point.tobytes()))
        if np.isfinite(value) and value > self.value:
            self.point, self.value = point.copy(), float(value)
        return value, gradient

    def ensure(self, point):
        point = np.asarray(point, dtype=float)
        if (point.shape, point.tobytes()) not in self._seen:
            self.evaluate(point)

    def result(self):
        if self.point is None:
            raise ValueError("Search did not evaluate a finite feasible objective")
        return self.point.copy(), self.value
