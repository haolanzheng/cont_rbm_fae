"""Explicit initialization laws using caller-owned NumPy RNGs.

Keeping historical RNG streams avoids changing paired experimental samples
during backend migration. Arrays are converted to JAX at the numerical boundary.
"""

import numpy as np


def uniform_sphere(rng, count, dimension, radius=1.0):
    points = rng.standard_normal((count, dimension))
    return radius * points / np.linalg.norm(points, axis=1, keepdims=True)


def uniform_box(rng, count, dimension, lower=-1.0, upper=1.0):
    return rng.uniform(lower, upper, (count, dimension))


def uniform_ball(rng, count, dimension, radius=1.0):
    points = uniform_sphere(rng, count, dimension, radius)
    return points * rng.random((count, 1)) ** (1.0 / dimension)
