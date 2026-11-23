"""Tangent-gradient sphere searches returning the best iterate, including 0..T."""

from functools import partial

import numpy as np

from contgreedy._jax import jax, jnp
from contgreedy.domain import Sphere
from contgreedy.sampling import AdamConfig, adam_ascent
from contgreedy.sampling import pool_search as select_pool

SPHERE = Sphere()


def normalize_rows(z):
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def quadratic(z, H):
    return z @ H @ z


def tangent_value_grad(z, H):
    hz = H @ z
    return z @ hz, 2 * (hz - (z @ hz) * z)


def adam_search(H, z0, steps, lr):
    z0 = np.asarray(z0) / np.linalg.norm(z0)
    result = adam_ascent(
        tangent_value_grad,
        z0,
        SPHERE.project,
        AdamConfig(steps=steps, learning_rate=lr, return_best=True),
        args=(jnp.asarray(H),),
    )
    return np.asarray(result.point), np.asarray(result.values).tolist()


@jax.jit
def _power_step(H, z):
    hz = H @ z
    norm = jnp.linalg.norm(hz)
    return hz / jnp.maximum(norm, 1e-300), norm


def power_search(H, z0, steps):
    H = jnp.asarray(H)
    z = SPHERE.project(z0)
    history = [float(z @ H @ z)]
    best_z, best_value = z, history[0]
    for _ in range(steps):
        candidate, norm = _power_step(H, z)
        if float(norm) < 1e-15:
            break
        z = candidate
        history.append(float(z @ H @ z))
        if history[-1] > best_value:
            best_z, best_value = z, history[-1]
    return np.asarray(best_z), history


@partial(jax.jit, static_argnames=("steps",))
def _normalized_gd_trajectory(H, z0, steps, learning_rate):
    """Retracted Euclidean ascent for z^T H z with Frobenius scaling."""
    H = 0.5 * (H + H.T)
    scale = jnp.maximum(jnp.linalg.norm(H), 1e-30)
    z0 = z0 / jnp.linalg.norm(z0)
    hz0 = H @ z0
    value0 = z0 @ hz0

    def step(carry, _):
        z, hz, best_z, best_value = carry
        candidate = z + learning_rate * hz / scale
        candidate = candidate / jnp.linalg.norm(candidate)
        hc = H @ candidate
        value = candidate @ hc
        improved = value > best_value
        best_z = jnp.where(improved, candidate, best_z)
        best_value = jnp.maximum(best_value, value)
        return (candidate, hc, best_z, best_value), value

    (_, _, best_z, _), values = jax.lax.scan(
        step, (z0, hz0, z0, value0), None, length=steps
    )
    return best_z, jnp.concatenate((value0[None], values))


def normalized_gd_search(H, z0, steps, learning_rate=1.0):
    point, values = _normalized_gd_trajectory(
        jnp.asarray(H), jnp.asarray(z0), steps, learning_rate
    )
    return np.asarray(point), np.asarray(values).tolist()


def pool_search(H, pool):
    z, value = select_pool(quadratic, jnp.asarray(pool), args=(jnp.asarray(H),))
    return np.asarray(z), float(value)
