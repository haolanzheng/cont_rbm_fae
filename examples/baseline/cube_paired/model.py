"""Analytic energy-coordinate model for a paired cube diffusion baseline.

Let psi_j be energy-orthonormal solutions of one fixed linear elliptic
operator.  The parameterized solution is u(z) = sum_j z_j psi_j / sqrt(d),
with z in [-1, 1]^d.  Consequently all reduced projections and energy errors
are computed exactly in R^d, without a spatial discretization.

Columns of Q are energy-orthonormal reduced basis vectors in these normalized
solution coordinates.  At the paired prefix Q, the residual is the sum of
squared differences between neighboring parameter coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


def hadamard(order: int) -> np.ndarray:
    """Return a Sylvester Hadamard matrix; order must be a positive power of 2."""
    order = int(order)
    if order < 1 or order & (order - 1):
        raise ValueError("Hadamard order must be a positive power of two")
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < order:
        h = np.block([[h, h], [h, -h]])
    return h


def paired_prefix_parameters(d: int) -> np.ndarray:
    """Return m=d/2 cube vertices forming a valid exact-greedy prefix.

    Each Hadamard row is repeated coordinatewise.  The resulting snapshots
    have unit norm and are mutually orthogonal, so every prefix selection
    attains the largest possible squared error, namely one.
    """
    d = int(d)
    if d < 2 or d % 2:
        raise ValueError("The paired construction requires even d >= 2")
    return np.repeat(hadamard(d // 2), 2, axis=1)


def paired_prefix(d: int) -> np.ndarray:
    """Return an orthonormal d-by-(d/2) basis for the pair-average space."""
    return paired_prefix_parameters(d).T / np.sqrt(d)


@dataclass(frozen=True)
class PairedCubeModel:
    """A linear solution map with an exactly known cube Kolmogorov width."""

    d: int

    def __post_init__(self) -> None:
        if not isinstance(self.d, (int, np.integer)) or self.d < 1:
            raise ValueError("d must be a positive integer")

    def solution_coefficients(self, z: np.ndarray) -> np.ndarray:
        """Map one parameter or a batch to coefficients in the psi_j basis."""
        z = np.asarray(z, dtype=np.float64)
        if z.shape[-1] != self.d:
            raise ValueError("Last parameter dimension must equal d")
        return z / np.sqrt(self.d)

    def _basis(self, Q: np.ndarray | None) -> np.ndarray:
        if Q is None:
            return np.empty((self.d, 0), dtype=np.float64)
        Q = np.asarray(Q, dtype=np.float64)
        if Q.ndim != 2 or Q.shape[0] != self.d or Q.shape[1] > self.d:
            raise ValueError("Q must have shape (d, N), with N <= d")
        return Q

    def matrix(self, Q: np.ndarray | None = None) -> np.ndarray:
        """Return H=(I-QQ^T)/d for an orthonormal reduced basis.

        At full rank the analytic residual is exactly zero.  Return the zero
        matrix explicitly to avoid magnifying roundoff by Frobenius scaling.
        """
        Q = self._basis(Q)
        if Q.shape[1] == self.d:
            return np.zeros((self.d, self.d), dtype=np.float64)
        H = (np.eye(self.d) - Q @ Q.T) / self.d
        return (H + H.T) * 0.5

    def error_squared(self, Q: np.ndarray | None, z: np.ndarray) -> np.ndarray:
        """Return squared Galerkin energy error, using a stable residual norm."""
        Q = self._basis(Q)
        u = self.solution_coefficients(z)
        if Q.shape[1] == self.d:
            return np.zeros(u.shape[:-1], dtype=np.float64)
        residual = u - (u @ Q) @ Q.T
        return np.sum(residual * residual, axis=-1)

    def append(
        self, Q: np.ndarray | None, z: np.ndarray, tol: float = 1e-12
    ) -> tuple[np.ndarray, float]:
        """Enrich Q with u(z), using two-pass modified Gram--Schmidt.

        Return (new_Q, novelty), where novelty is the norm of the component
        outside Q.  Dependent snapshots return the unchanged Q and zero.
        """
        Q = self._basis(Q)
        if Q.shape[1] == self.d:
            return Q, 0.0
        u = self.solution_coefficients(z)
        if u.ndim != 1:
            raise ValueError("append expects one parameter vector")
        residual = u.copy()
        for _ in range(2):
            residual -= Q @ (Q.T @ residual)
        novelty = float(np.linalg.norm(residual))
        if novelty <= tol * max(1.0, float(np.linalg.norm(u))):
            return Q, 0.0
        return np.column_stack((Q, residual / novelty)), novelty

    def width(self, N: int) -> float:
        """Return d_N=sqrt((d-N)/d), with zero width at N >= d."""
        if N < 0:
            raise ValueError("N must be nonnegative")
        return float(np.sqrt(max(self.d - N, 0) / self.d))


def scores(H: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Evaluate z^T H z for one parameter or a batch, retaining its shape."""
    z = np.asarray(z, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    return np.maximum(np.sum((z @ H) * z, axis=-1), 0.0)


def _projected_search(
    starts: np.ndarray,
    score: Callable[[np.ndarray], np.ndarray],
    normalized_gradient: Callable[[np.ndarray], np.ndarray],
    alpha: float,
    T: int,
    checkpoints: Iterable[int],
    threshold: float | None,
) -> tuple[np.ndarray, dict]:
    original = np.asarray(starts, dtype=np.float64)
    if original.ndim not in (1, 2) or original.size == 0:
        raise ValueError("starts must be one parameter or a nonempty batch")
    if not np.all(np.isfinite(original)) or np.any(np.abs(original) > 1 + 1e-13):
        raise ValueError("starts must belong to [-1,1]^d")
    if alpha <= 0 or T < 0 or int(T) != T:
        raise ValueError("alpha must be positive and T a nonnegative integer")
    T = int(T)
    z = np.clip(np.atleast_2d(original).copy(), -1.0, 1.0)
    best = z.copy()
    values = score(z)
    previous = values.copy()
    keep = {0, T} | {int(t) for t in checkpoints if 0 <= int(t) <= T}
    history = []
    min_increment = 0.0
    for t in range(T + 1):
        current = score(z)
        improved = current > values
        best[improved] = z[improved]
        values = np.maximum(values, current)
        if t:
            min_increment = min(min_increment, float(np.min(current - previous)))
        previous = current.copy()
        if t in keep:
            history.append(
                {
                    "T": t,
                    "best_values": values.copy(),
                    "hits": None if threshold is None else int(np.count_nonzero(values >= threshold)),
                    "mean_best_value": float(np.mean(values)),
                    "mean_boundary_coordinates": float(np.mean(np.abs(z) >= 1 - 1e-12)),
                }
            )
        if t == T:
            break
        grad = normalized_gradient(z)
        eta = alpha / np.sqrt(1.0 + np.sum(grad * grad, axis=1, keepdims=True))
        z = np.clip(z + eta * grad, -1.0, 1.0)
    single = original.ndim == 1
    stats = {
        "history": history,
        "best_values": values.copy(),
        "final_values": current.copy(),
        "final_points": z[0].copy() if single else z.copy(),
        "min_increment": min_increment,
        "feasibility_defect": float(np.max(np.maximum(np.abs(z) - 1, 0))),
        "updates": T,
        "gradient_evaluations": T * len(z),
    }
    return (best[0] if single else best), stats


def projected_search(
    H: np.ndarray,
    starts: np.ndarray,
    alpha: float = 0.6,
    T: int = 80,
    checkpoints: Iterable[int] = (0, 10, 20, 40, 80),
    threshold: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Run the baseline Frobenius-normalized projected gradient ascent.

    Each trajectory uses eta_t=alpha/sqrt(1+||2Hz/||H||_F||^2), then clips
    coordinatewise and returns its best visited parameter.  All trajectories
    in a batch have their own adaptive stepsize.
    """
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")
    if np.asarray(starts).shape[-1] != H.shape[0]:
        raise ValueError("H and starts have incompatible dimensions")
    ell = float(np.linalg.norm(H, "fro"))

    def gradient(z: np.ndarray) -> np.ndarray:
        return 2 * (z @ H) / ell if ell > 0 else np.zeros_like(z)

    return _projected_search(
        starts, lambda z: scores(H, z), gradient, alpha, T, checkpoints, threshold
    )


def paired_search(
    starts: np.ndarray,
    alpha: float = 0.6,
    T: int = 80,
    checkpoints: Iterable[int] = (0, 10, 20, 40, 80),
    threshold: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Run exactly the same update at the paired prefix in O(batch*d) work."""
    d = np.asarray(starts).shape[-1]
    if d < 2 or d % 2:
        raise ValueError("The paired residual requires even d >= 2")
    m = d // 2

    def score(z: np.ndarray) -> np.ndarray:
        delta = z[:, 0::2] - z[:, 1::2]
        return np.sum(delta * delta, axis=1) / (2 * d)

    def gradient(z: np.ndarray) -> np.ndarray:
        delta = (z[:, 0::2] - z[:, 1::2]) / np.sqrt(m)
        grad = np.empty_like(z)
        grad[:, 0::2] = delta
        grad[:, 1::2] = -delta
        return grad

    return _projected_search(starts, score, gradient, alpha, T, checkpoints, threshold)


def verify_model(d: int = 16, seed: int = 913) -> dict:
    """Check analytic identities, the greedy prefix, and the two search paths.

    This lightweight audit is not an experiment.  It raises AssertionError
    if energy projection, gradients, or efficient paired updates disagree.
    """
    rng = np.random.default_rng(seed)
    model = PairedCubeModel(d)
    Q = paired_prefix(d)
    H = model.matrix(Q)
    Z = rng.uniform(-0.8, 0.8, size=(17, d))
    delta = Z[:, 0::2] - Z[:, 1::2]
    pair_values = np.sum(delta * delta, axis=1) / (2 * d)
    matrix_error = float(np.max(np.abs(scores(H, Z) - pair_values)))
    energy_error = float(np.max(np.abs(model.error_squared(Q, Z) - pair_values)))
    orthogonality = float(np.linalg.norm(Q.T @ Q - np.eye(d // 2)))
    eps = 1e-6
    z = Z[0]
    finite_difference = np.empty(d)
    for j in range(d):
        direction = np.zeros(d)
        direction[j] = eps
        finite_difference[j] = (
            model.error_squared(Q, z + direction) - model.error_squared(Q, z - direction)
        ) / (2 * eps)
    gradient_error = float(np.max(np.abs(finite_difference - 2 * H @ z)))
    prefix_errors = []
    basis = np.empty((d, 0))
    for parameter in paired_prefix_parameters(d):
        prefix_errors.append(abs(float(model.error_squared(basis, parameter)) - 1))
        basis, novelty = model.append(basis, parameter)
        assert novelty > 0
    prefix_error = float(max(prefix_errors))
    full, full_stats = projected_search(H, Z, T=20, threshold=0.81)
    fast, fast_stats = paired_search(Z, T=20, threshold=0.81)
    update_error = float(np.max(np.abs(full - fast)))
    min_increment = min(full_stats["min_increment"], fast_stats["min_increment"])
    assert matrix_error < 1e-12
    assert energy_error < 1e-12
    assert orthogonality < 1e-12
    assert gradient_error < 1e-8
    assert prefix_error < 1e-12
    assert update_error < 1e-11
    assert min_increment > -1e-12
    assert np.all(model.matrix(np.eye(d)) == 0)
    return {
        "d": d,
        "matrix_vs_pair_error": matrix_error,
        "energy_vs_pair_error": energy_error,
        "orthogonality_error": orthogonality,
        "finite_difference_gradient_error": gradient_error,
        "exact_greedy_prefix_error": prefix_error,
        "dense_vs_pair_update_error": update_error,
        "min_increment": min_increment,
    }
