"""Feasible sets. Initialization laws live in contgreedy.sampling."""

from dataclasses import dataclass
from contgreedy._jax import jnp


@dataclass(frozen=True)
class Box:
    lower: float = -1.0
    upper: float = 1.0

    def __post_init__(self):
        if self.lower >= self.upper:
            raise ValueError("Box requires lower < upper")

    def project(self, z):
        return jnp.clip(z, self.lower, self.upper)


@dataclass(frozen=True)
class Sphere:
    radius: float = 1.0

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError("Sphere requires positive radius")

    def project(self, z):
        z = jnp.asarray(z, dtype=jnp.float64)
        norm = jnp.linalg.norm(z)
        fallback = jnp.zeros_like(z).at[0].set(self.radius)
        return jnp.where(
            norm > 0, z * self.radius / jnp.maximum(norm, 1e-300), fallback
        )

    def tangent(self, z, g):
        return g - z * jnp.vdot(z, g).real / self.radius**2


@dataclass(frozen=True)
class Ball:
    radius: float = 1.0

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError("Ball requires positive radius")

    def project(self, z):
        z = jnp.asarray(z, dtype=jnp.float64)
        return z / jnp.maximum(1.0, jnp.linalg.norm(z) / self.radius)
