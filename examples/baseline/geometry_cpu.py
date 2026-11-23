"""CPU-only quadratic searches on the round sphere and the closed unit ball.

H is the residual-indicator matrix of one fixed greedy round.  The search
objective is J(z)/||H||_F, where J(z)=z.T H z.  All arrays are float64 NumPy
arrays; this module does not import a GPU-capable numerical backend.
"""

from __future__ import annotations

import numpy as np
from scipy.special import betaincc


def sym(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (matrix + matrix.T)


def uniform_parameters(rng, count, dimension, domain):
    """Normalized Gaussian directions; ball radii have distribution r**d."""
    if domain not in ("sphere", "ball") or count < 1 or dimension < 2:
        raise ValueError("Require sphere/ball, positive count, and dimension >= 2")
    points = rng.standard_normal((count, dimension))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    if domain == "ball":
        points *= rng.random(count)[:, None] ** (1.0 / dimension)
    return points


def sphere_dj(H, points):
    """Round-metric tangent derivative of the normalized objective."""
    points = np.asarray(points, dtype=np.float64)
    scale = float(np.linalg.norm(H, "fro"))
    if scale == 0:
        return np.zeros_like(points)
    hz = points @ H.T
    values = np.sum(points * hz, axis=-1, keepdims=True)
    return 2.0 * (hz - values * points) / scale


def sphere_exp(points, tangent):
    """Exp_z(v) for the unit round sphere, with a stable zero-step limit."""
    length = np.linalg.norm(tangent, axis=-1, keepdims=True)
    result = np.cos(length) * points + np.sinc(length / np.pi) * tangent
    # Only repair floating-point norm drift; the update is the exponential map.
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def sphere_step(H, points, alpha=0.2):
    derivative = sphere_dj(H, points)
    length_squared = np.sum(derivative * derivative, axis=-1, keepdims=True)
    tangent = alpha * derivative / np.sqrt(1.0 + length_squared)
    return sphere_exp(points, tangent)


def ball_step(H, points, omega=1.0):
    """(1-omega)z + omega P_B(z + H z / ||H||_F).

    Direct projection (omega=1) is the default. The omega argument remains
    available to reproduce archived relaxed runs. The unconstrained trial
    uses step 1/2 for the derivative of J/||H||_F.
    Activity records whether the *trial* lies outside the ball; omega < 1
    need not place the actual next iterate on the boundary.
    """
    if not 0 < omega <= 1:
        raise ValueError("omega must be in (0,1]")
    scale = float(np.linalg.norm(H, "fro"))
    trial = points if scale == 0 else points + (points @ H.T) / scale
    radius = np.linalg.norm(trial, axis=-1, keepdims=True)
    projected = trial / np.maximum(1.0, radius)
    return (1.0 - omega) * points + omega * projected, (radius[..., 0] > 1.0)


def search(H, initial, steps=100, domain="sphere", alpha=0.2, omega=1.0,
           checkpoints=None, threshold=None, ball_step_rule="fixed", diagnostics=True):
    """Return earliest best among 0..steps, for one start or a batch of starts.

    q estimates, when requested, count best-so-far threshold attainment.  The
    supplied batch is never treated as a candidate pool for a single search.
    Use diagnostics=False for training timings; probability audits retain
    diagnostics=True. Diagnostic quantities do not affect the updates.
    """
    if ball_step_rule not in ("fixed", "normalized") or alpha <= 0:
        raise ValueError("Require a positive alpha and a known ball step rule")
    if not diagnostics and (checkpoints is not None or threshold is not None):
        raise ValueError("Probability audits require diagnostics=True")
    H = sym(H)
    scale = float(np.linalg.norm(H, "fro"))
    points = np.atleast_2d(np.asarray(initial, dtype=np.float64)).copy()
    if points.shape[1] != H.shape[0] or steps < 0:
        raise ValueError("Incompatible dimension or negative update count")
    radii = np.linalg.norm(points, axis=1)
    if domain == "sphere":
        if np.any(radii == 0):
            raise ValueError("Sphere starts cannot be zero")
        points /= radii[:, None]
    elif domain == "ball":
        if np.any(radii > 1.0 + 1e-12):
            raise ValueError("Ball starts must be feasible")
    else:
        raise ValueError("Unknown domain")
    best = points.copy()
    best_values = np.full(len(points), -np.inf)
    if diagnostics:
        activity_count = np.zeros(len(points), dtype=int)
        hit_time = np.full(len(points), -1, dtype=int)
        norm_defect = 0.0
        minimum_increment = np.inf
        previous_values = None
        snapshots = {}
        requested = set(checkpoints or ())
    for t in range(steps + 1):
        hz = points @ H.T
        values = np.sum(points * hz, axis=1)
        improved = values > best_values
        best[improved] = points[improved]
        best_values[improved] = values[improved]
        if diagnostics:
            if threshold is not None:
                newly_hit = (hit_time < 0) & (best_values >= threshold)
                hit_time[newly_hit] = t
            if previous_values is not None:
                minimum_increment = min(minimum_increment, float(np.min(values - previous_values)))
            previous_values = values
            norm = np.linalg.norm(points, axis=1)
            defect = np.abs(norm - 1.0) if domain == "sphere" else np.maximum(norm - 1.0, 0.0)
            norm_defect = max(norm_defect, float(np.max(defect)))
            if t in requested:
                snapshots[str(t)] = {
                    "hits": int(np.count_nonzero(hit_time >= 0)),
                    "mean_best_value": float(np.mean(best_values)),
                    "projection_active_fraction": float(np.mean(activity_count > 0)),
                    "mean_radius": float(np.mean(norm)),
                    ("unit_radius_fraction_1e12" if domain == "sphere" else "boundary_contact_fraction_1e12"): float(np.mean(norm >= 1.0 - 1e-12)),
                }
        if t == steps:
            break
        if domain == "sphere":
            derivative = np.zeros_like(points) if scale == 0 else 2.0 * (hz - values[:, None] * points) / scale
            tangent = alpha * derivative / np.sqrt(1.0 + np.sum(derivative**2, axis=1, keepdims=True))
            points = sphere_exp(points, tangent)
        else:
            derivative = np.zeros_like(points) if scale == 0 else 2.0 * hz / scale
            eta = (alpha / np.sqrt(1.0 + np.sum(derivative**2, axis=1, keepdims=True))
                   if ball_step_rule == "normalized" else 0.5)
            trial = points + eta * derivative
            norm = np.linalg.norm(trial, axis=1, keepdims=True)
            if diagnostics:
                activity_count += norm[:, 0] > 1.0
            points = (1.0 - omega) * points + omega * trial / np.maximum(1.0, norm)
    data = {"h_actions_per_start": steps + 1}
    if diagnostics:
        data.update({
            "max_feasibility_defect": norm_defect,
            "minimum_objective_increment": 0.0 if steps == 0 else minimum_increment,
            "projection_active_steps_mean": float(np.mean(activity_count)),
            "final_radius_mean": float(np.mean(np.linalg.norm(points, axis=1))),
            "snapshots": snapshots,
        })
    return (best[0] if np.ndim(initial) == 1 else best), data


def pool_search(H, pool):
    scores = np.sum(pool * (pool @ H.T), axis=1)
    index = int(np.argmax(scores))
    return pool[index].copy(), {"h_actions_per_start": len(pool)}


def ball_probability_lower(H, gamma, steps, omega=1.0):
    """Discrete sufficient-event bound for uniform volume initialization.

    Before J >= gamma**2 lambda_1, the top coefficient grows by at least c,
    with a=1+lambda_1/ell, b=sqrt(1+gamma**2*(a*a-1)),
    c=1-omega+omega*a/b.  Its initial square is Beta(1/2,(d+1)/2).
    """
    if not 0 < gamma < 1 or not 0 < omega <= 1:
        raise ValueError("Invalid gamma or omega")
    top = float(np.linalg.eigvalsh(H)[-1])
    scale = float(np.linalg.norm(H, "fro"))
    if top <= 0 or scale == 0:
        raise ValueError("Probability certificate requires nonzero PSD H")
    a = 1.0 + top / scale
    b = np.sqrt(1.0 + gamma**2 * (a*a - 1.0))
    c = 1.0 - omega + omega*a/b
    coordinate_threshold = gamma**2 * c**(-2 * np.asarray(steps))
    return betaincc(0.5, (H.shape[0] + 1.0) / 2.0, coordinate_threshold)


def ball_normalized_probability_lower(H, gamma, steps, alpha=0.2):
    """Bound for eta(z)=alpha/sqrt(1+||2Hz/ell||^2), direct projection.

    Below J=gamma^2*sigma, ||DJ||^2 <= 4*gamma^2*sigma^2.
    Thus eta >= h=alpha/sqrt(1+4*gamma^2*sigma^2). The coefficient
    amplification lower bound is increasing in eta, so evaluate it at h.
    """
    if not 0 < gamma < 1 or alpha < 0:
        raise ValueError("Require 0<gamma<1 and alpha>=0")
    scale = float(np.linalg.norm(H, "fro"))
    top = float(np.linalg.eigvalsh(sym(H))[-1])
    if scale == 0 or top <= 0:
        raise ValueError("Require nonzero PSD H")
    sigma = top / scale
    h = alpha / np.sqrt(1 + 4*gamma**2*sigma**2)
    c = (1 + 2*h*sigma) / np.sqrt(1 + 4*gamma**2*h*sigma*(1+h*sigma))
    bound = betaincc(0.5, (len(H)+1)/2, gamma**2*c**(-2*np.asarray(steps)))
    return bound, {"alpha": float(alpha), "sigma": sigma,
                   "eta_lower_before_target": float(h), "growth_factor": float(c)}


def direct_probability_upper(H, gamma, domain):
    """Gap-based upper bound on one uniform draw, valid also with a kernel."""
    eigenvalues = np.linalg.eigvalsh(sym(H))
    ratio = max(float(eigenvalues[-2] / eigenvalues[-1]), 0.0)
    if ratio >= gamma**2:
        return 1.0
    cutoff = (gamma**2 - ratio) / (1.0 - ratio)
    second = (H.shape[0] - 1.0) / 2.0 if domain == "sphere" else (H.shape[0] + 1.0) / 2.0
    return float(betaincc(0.5, second, cutoff))


class CoordinateROM:
    """Coefficient representation V=S Q of a linear solution family.

    Q.T G0 Q=I. If A S=B, P=I-Q Q.T G0 and the residual is B P.
    For sampled analytic snapshots, cross=S.T B keeps the original
    Galerkin equations and the extra Gram matrices retain B-A S exactly.
    No diagonal surrogate replaces the discrete residual objective.
    """

    def __init__(self, G0, H0, cross=None, residual_error_cross=None, residual_error_gram=None):
        self.G0 = sym(G0)
        self.H0 = sym(H0)
        self.cross = self.G0 if cross is None else np.asarray(cross)
        self.residual_error_cross = residual_error_cross
        self.residual_error_gram = residual_error_gram
        self.Q = np.empty((len(G0), 0), dtype=np.float64)

    def matrices(self, *, include_error=True):
        # Galerkin coefficients use S.T B, even when sampled exact snapshots
        # do not exactly satisfy A S=B on the retained finite-element grid.
        M = self.Q @ (self.Q.T @ self.cross)
        P = np.eye(len(self.G0)) - M
        H = P.T @ self.H0 @ P
        if self.residual_error_cross is not None:
            # B-A S M = B P + (B-A S)M. This form avoids subtracting
            # nearly equal O(1) Gram matrices at late greedy rounds.
            mixed = P.T @ self.residual_error_cross @ M
            H += mixed + mixed.T + M.T @ self.residual_error_gram @ M
        return sym(H), sym(P.T @ self.G0 @ P) if include_error else None

    def append(self, parameter):
        q = np.asarray(parameter, dtype=np.float64).copy()
        for _ in range(2):
            q -= self.Q @ (self.Q.T @ (self.G0 @ q))
        novelty = np.sqrt(max(float(q @ self.G0 @ q), 0.0))
        if novelty <= 1e-14:
            raise RuntimeError("Selected snapshot does not enrich the ROM")
        self.Q = np.column_stack((self.Q, q / novelty))
        return float(novelty)

    def orthogonality_defect(self):
        return float(np.linalg.norm(self.Q.T @ self.G0 @ self.Q - np.eye(self.Q.shape[1]), ord=2)) if self.Q.shape[1] else 0.0


def rom_options(data):
    """Additional Galerkin/residual matrices for sampled analytic snapshots."""
    return {k:data[k] for k in ('cross','residual_error_cross','residual_error_gram') if k in data}
