"""Limited-memory BFGS on an exact Euclidean sphere.

The update uses the sphere exponential map and exact parallel transport.
No coordinate clipping or post-step radial projection is used.
"""

from __future__ import annotations
import numpy as np


def tangent(x, v, radius):
    return v - x * (x @ v) / (radius * radius)


def sphere_exp(x, v, radius):
    v = tangent(x, v, radius)
    nv = float(np.linalg.norm(v))
    if nv < 1e-15:
        return x.copy()
    angle = nv / radius
    return np.cos(angle) * x + radius * np.sin(angle) * (v / nv)


def sphere_transport(x, y, v, radius):
    """Exact parallel transport along the short geodesic x -> y."""
    v = tangent(x, v, radius)
    den = radius * radius + float(x @ y)
    if den < 1e-12 * radius * radius:
        return tangent(y, v, radius)
    out = v - (float(v @ y) / den) * (x + y)
    return tangent(y, out, radius)


def rlbfgs_ascent(
    fg,
    value_only,
    x0,
    radius,
    steps,
    memory=7,
    initial_trust_fraction=0.25,
    c1=1e-4,
    max_linesearch=12,
):
    """Maximize a positive objective through its logarithm."""
    x = np.asarray(x0, float).copy()
    x *= radius / np.linalg.norm(x)
    raw, graw = fg(x)
    eps = 1e-30
    obj = float(np.log(max(raw, eps)))
    grad = tangent(x, np.asarray(graw) / (max(raw, eps)), radius)
    h = -grad  # gradient of the minimization -log f
    history = []
    best = (float(raw), x.copy())
    value_evals = 0
    grad_evals = 1
    delta = initial_trust_fraction * radius
    delta_max = 0.45 * np.pi * radius
    trace = []
    for it in range(steps):
        q = h.copy()
        alphas = []
        for s, y, rho in reversed(history):
            a = rho * float(s @ q)
            alphas.append(a)
            q -= a * y
        if history:
            s, y, _ = history[-1]
            gamma = float(s @ y) / max(float(y @ y), 1e-30)
            gamma = float(np.clip(gamma, 1e-4, 1e4))
        else:
            gamma = 1.0
        p = gamma * q
        for (s, y, rho), a in zip(history, reversed(alphas)):
            b = rho * float(y @ p)
            p += s * (a - b)
        p = tangent(x, -p, radius)  # descent for -log f = ascent for log f
        directional = float(grad @ p)
        if directional <= 1e-14:
            p = grad.copy()
            directional = float(grad @ p)
            history = []
        npnorm = float(np.linalg.norm(p))
        if npnorm > delta:
            p *= delta / npnorm
            directional = float(grad @ p)

        alpha = 1.0
        accepted = False
        candidate = None
        cand_obj = None
        cand_raw = None
        for ls in range(max_linesearch):
            candidate = sphere_exp(x, alpha * p, radius)
            cand_raw = float(value_only(candidate))
            value_evals += 1
            cand_obj = float(np.log(max(cand_raw, eps)))
            if cand_obj >= obj + c1 * alpha * directional:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            delta *= 0.5
            history = []
            trace.append(
                {
                    "iteration": it + 1,
                    "accepted": False,
                    "raw": float(raw),
                    "trust_radius": delta,
                    "line_search_evals": max_linesearch,
                }
            )
            if delta < 1e-8 * radius:
                break
            continue

        new_raw, new_graw = fg(candidate)
        grad_evals += 1
        new_obj = float(np.log(max(new_raw, eps)))
        new_grad = tangent(candidate, np.asarray(new_graw) / max(new_raw, eps), radius)
        new_h = -new_grad
        transported = []
        for s, y, rho in history:
            st = sphere_transport(x, candidate, s, radius)
            yt = sphere_transport(x, candidate, y, radius)
            transported.append((st, yt, rho))
        step_vec = sphere_transport(x, candidate, alpha * p, radius)
        old_h = sphere_transport(x, candidate, h, radius)
        yvec = new_h - old_h
        sy = float(step_vec @ yvec)
        if sy > 1e-10 * np.linalg.norm(step_vec) * max(np.linalg.norm(yvec), 1e-30):
            transported.append((step_vec, yvec, 1.0 / sy))
            transported = transported[-memory:]
        else:
            transported = []
        actual = new_obj - obj
        pred = max(alpha * directional, 1e-30)
        ratio = actual / pred
        if ratio > 0.75 and alpha > 0.99:
            delta = min(1.5 * delta, delta_max)
        elif ratio < 0.1:
            delta = max(0.5 * delta, 1e-8 * radius)
        x, raw, obj, grad, h = candidate, float(new_raw), new_obj, new_grad, new_h
        history = transported
        if raw > best[0]:
            best = (raw, x.copy())
        trace.append(
            {
                "iteration": it + 1,
                "accepted": True,
                "raw": raw,
                "alpha": alpha,
                "trust_radius": delta,
                "line_search_evals": ls + 1,
                "grad_norm": float(np.linalg.norm(grad)),
            }
        )
    return (
        best[1],
        best[0],
        {
            "gradient_evaluations": grad_evals,
            "value_evaluations": value_evals,
            "trace": trace,
        },
    )
