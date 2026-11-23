"""Corrupted-input training objectives for function reconstruction."""

import jax.numpy as jnp

from .model import coordinates, decode_coordinates, encode, trapezoid_weights


def reconstruction_objective(
    params,
    encoder_input,
    target,
    observation_mask,
    decoder_indices,
    config,
    penalty,
):
    """Encode a degraded field and reconstruct sampled nodes of the clean target."""
    code = encode(params, encoder_input, config, observation_mask)
    target_shape = target.shape[-2:]
    flat_target = target.reshape(len(target), -1)
    xy = coordinates(target_shape)[decoder_indices]
    prediction = decode_coordinates(params, code, xy, config)
    weights = trapezoid_weights(*target_shape)[decoder_indices]
    # Uniform sampling without replacement: N/m makes the weighted spatial sum
    # an unbiased estimator of the full quadrature loss.
    weights = weights * (flat_target.shape[1] / decoder_indices.shape[0])
    reconstruction = 0.5 * jnp.mean(
        jnp.sum(
            (prediction - flat_target[:, decoder_indices]) ** 2
            * weights[None, :],
            axis=1,
        )
    )
    latent = jnp.mean(jnp.sum(code**2, axis=1))
    return reconstruction + penalty * latent, (reconstruction, latent)
