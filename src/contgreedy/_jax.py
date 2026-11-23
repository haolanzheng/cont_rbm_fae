"""Numerical policy: float64 is required for subtractive residuals."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

__all__ = ["jax", "jnp"]
