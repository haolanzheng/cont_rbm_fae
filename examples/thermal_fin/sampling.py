"""Thermal-fin selection policy; historical local solvers retain their settings."""

from __future__ import annotations
import numpy as np
from scipy import optimize
from contgreedy._jax import jnp
from contgreedy.domain import Box
from contgreedy.sampling import AdamConfig, adam_ascent
from .rom import value_grad


def kkt(z, g):
    return float(np.max(np.abs(np.minimum(1, np.maximum(-1, z + g)) - z)))


def select(rb, method, z0):
    calls = 0
    best = (-np.inf, None)
    history = []

    def fun(z):
        nonlocal calls, best
        j, g = rb.fg(z)
        calls += 1
        if j > best[0]:
            best = (j, z.copy())
        return -j, -g

    if method.startswith("adam"):
        steps = int(method[4:])
        result = adam_ascent(
            value_grad,
            jnp.asarray(z0),
            Box().project,
            AdamConfig(steps=steps, learning_rate=0.05, beta2=0.99),
            args=rb.freeze(),
        )
        best = (float(result.value), np.asarray(result.point))
        calls = int(result.gradient_evaluations)
        nit = steps
        message = "JAX projected Adam; preserved thermal-fin policy"
        success = False
    else:
        limit = 1000 if method == "local" else int(method[5:])

        def callback(z):
            history.append(float(-fun(z)[0]))

        res = optimize.minimize(
            fun,
            z0,
            jac=True,
            method="L-BFGS-B",
            bounds=[(-1, 1)] * len(z0),
            options={
                "maxiter": limit,
                "maxfun": max(5000, limit * 10),
                "ftol": 0.0,
                "gtol": 1e-7,
                "maxls": 40,
                "maxcor": 15,
            },
        )
        nit = int(res.nit)
        message = str(res.message)
        success = bool(res.success)
        if method == "local" and kkt(res.x, -res.jac) > 1e-5:
            # Continue the SAME trajectory with a different local solver; no restart.
            polish = optimize.minimize(
                fun,
                res.x,
                jac=True,
                method="SLSQP",
                bounds=[(-1, 1)] * len(z0),
                options={"maxiter": 300, "ftol": 1e-13},
            )
            nit += int(polish.nit)
            message += "; SLSQP: " + str(polish.message)
            success = bool(polish.success)
    z = best[1]
    j, g = rb.fg(z)
    return z, {
        "iterations": nit,
        "fg_calls": calls,
        "log_indicator_squared": float(j),
        "projected_gradient_inf": kkt(z, g),
        "solver_success": success,
        "solver_message": message,
    }
