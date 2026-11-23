"""Numerical checks for the array-based pure-JAX FAE implementation."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from .model import apply, fourier_transfer, init_model, parameter_count
from .training import objective


def compact_config(base, pooling_kind):
    config = dict(base or {})
    config.update(
        latent_dimension=8,
        width=4,
        modes=2,
        pool_channels=2,
        kernel_hidden=8,
        pooling_kind=pooling_kind,
        decoder_kind="conditional_mlp",
        decoder_width=8,
        decoder_blocks=1,
        decoder_latent_scale=1.0,
        coordinate_frequencies=2,
    )
    if pooling_kind == "orthogonal_integral":
        config["orthogonal_epsilon"] = 1e-3
    if pooling_kind == "quadrature_attention":
        config.update(
            attention_queries=2,
            attention_value_dimension=4,
            attention_key_dimension=8,
        )
    return config


def run_checks(model_config=None):
    spectral = []
    for n, m in ((24, 18), (18, 12), (12, 8), (8, 12)):
        positive = jnp.ones((1, 4, 4), dtype=jnp.complex64)
        negative = jnp.ones_like(positive)
        constant = jnp.ones((1, 1, n, n), dtype=jnp.float32)
        constant_error = float(
            jnp.max(jnp.abs(fourier_transfer(constant, (m, m), positive, negative) - 1))
        )

        def mode(resolution):
            axis = jnp.arange(resolution, dtype=jnp.float32) / resolution
            yy, xx = jnp.meshgrid(axis, axis, indexing="ij")
            return jnp.sin(2 * math.pi * (xx + yy))[None, None]

        mode_error = float(
            jnp.max(
                jnp.abs(
                    fourier_transfer(mode(n), (m, m), positive, negative) - mode(m)
                )
            )
        )
        assert constant_error < 2e-6 and mode_error < 2e-5
        spectral.append(
            {
                "grid": [n, m],
                "constant_max_error": constant_error,
                "sinusoid_max_error": mode_error,
            }
        )

    pooling = {}
    for index, kind in enumerate(
        ("orthogonal_integral", "quadrature_attention", "nonlinear_integral")
    ):
        config = compact_config(model_config, kind)
        params = init_model(jax.random.PRNGKey(170 + index), config)
        field = jax.random.normal(jax.random.PRNGKey(180 + index), (2, 12, 12))
        prediction, code = apply(params, field, config)
        loss, gradients = jax.value_and_grad(
            lambda values: objective(values, field, config, 1e-4)[0]
        )(params)
        finite = bool(
            jnp.isfinite(loss)
            & jnp.all(jnp.isfinite(prediction))
            & jnp.all(jnp.isfinite(code))
            & jnp.all(
                jnp.stack(
                    [
                        jnp.all(jnp.isfinite(value))
                        for value in jax.tree_util.tree_leaves(gradients)
                    ]
                )
            )
        )
        assert prediction.shape == field.shape and code.shape == (2, 8) and finite
        pooling[kind] = {
            "real_parameters": parameter_count(params),
            "latent_shape": list(code.shape),
            "loss": float(loss),
            "all_gradients_finite": True,
        }

    return {
        "backend": "jax",
        "spectral_amplitude": spectral,
        "pooling": pooling,
        "scope": "Implementation checks only; not trained reconstruction accuracy.",
    }
