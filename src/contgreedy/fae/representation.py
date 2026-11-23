"""Physical-field adapter for array-based FAE checkpoints."""

import jax
import jax.numpy as jnp

from .model import coordinates, decode_coordinates, encode


class FAERepresentation:
    def __init__(self, params, model_config, mean, std):
        if std <= 0:
            raise ValueError("Normalization std must be positive")
        self.params = params
        self.model_config = model_config
        self.mean = mean
        self.std = std

    def encode(self, field):
        return encode(
            self.params,
            (jnp.asarray(field) - self.mean) / self.std,
            self.model_config,
        )

    def decode(self, latent, shape):
        values = decode_coordinates(
            self.params,
            jnp.asarray(latent),
            coordinates(shape),
            self.model_config,
        )
        return values.reshape(len(latent), *shape) * self.std + self.mean

    def pullback(self, latent, field_gradient, shape):
        """Compute decoder transpose times a physical field gradient."""
        _, transpose = jax.vjp(lambda code: self.decode(code, shape), latent)
        return transpose(jnp.asarray(field_gradient))[0]
