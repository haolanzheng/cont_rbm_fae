import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from contgreedy.fae import model

ROOT = Path(__file__).resolve().parents[1]


def decoder_config(kind):
    config = {
        "latent_dimension": 6,
        "width": 4,
        "modes": 2,
        "pool_channels": 2,
        "kernel_hidden": 8,
        "decoder_kind": kind,
        "decoder_width": 16,
        "decoder_blocks": 2,
        "coordinate_frequencies": 2,
    }
    if kind == "conditional_mlp":
        config["decoder_latent_scale"] = 1.0
    else:
        config.update(
            siren_omega_initial=10.0,
            siren_omega_hidden=30.0,
            siren_scale_limit=0.1,
        )
    return config


@pytest.mark.parametrize("kind", ["conditional_mlp", "modulated_siren"])
def test_nonlinear_decoders_are_finite_latent_dependent_and_have_no_norm(kind):
    config = decoder_config(kind)
    params = model.init_model(jax.random.PRNGKey(71), config)
    decoder_names = [name for name in params if name.startswith("decoder.")]
    assert decoder_names and not any("norm" in name for name in decoder_names)

    code = 0.2 * jax.random.normal(jax.random.PRNGKey(72), (3, 6))
    xy = model.coordinates((7, 9))
    decoded = model.decode_coordinates(params, code, xy, config)
    assert decoded.shape == (3, 63)
    assert jnp.all(jnp.isfinite(decoded))
    assert not np.allclose(decoded[0], decoded[1])

    code_gradient = jax.grad(
        lambda z: jnp.mean(model.decode_coordinates(params, z, xy, config) ** 2)
    )(code)
    assert jnp.all(jnp.isfinite(code_gradient))
    assert jnp.linalg.norm(code_gradient) > 0

    midpoint = model.decode_coordinates(
        params, (code[:1] + code[1:2]) / 2, xy, config
    )
    endpoint_average = (
        model.decode_coordinates(params, code[:1], xy, config)
        + model.decode_coordinates(params, code[1:2], xy, config)
    ) / 2
    assert jnp.linalg.norm(midpoint - endpoint_average) > 1e-5

    loss, gradients = jax.value_and_grad(
        lambda p: jnp.mean(model.decode_coordinates(p, code, xy, config) ** 2)
    )(params)
    assert jnp.isfinite(loss)
    assert all(
        jnp.all(jnp.isfinite(value))
        for value in jax.tree_util.tree_leaves(gradients)
    )


def test_siren_initialization_obeys_frequency_scaled_bounds():
    config = decoder_config("modulated_siren")
    params = model.init_model(jax.random.PRNGKey(73), config)
    first = params["decoder.siren.input.weight"]
    hidden = params["decoder.siren.blocks.0.linear.weight"]
    assert jnp.max(jnp.abs(first)) <= 1 / 2
    expected = math.sqrt(6 / config["decoder_width"]) / config["siren_omega_hidden"]
    assert jnp.max(jnp.abs(hidden)) <= expected


def test_full_nonlinear_decoder_configs_share_encoder_initialization_and_budget():
    names = (
        "fae64_jax_conditional_mlp_n4096",
        "fae64_jax_modulated_siren_n4096",
    )
    configs = [
        json.loads(
            (
                ROOT
                / "examples/hyperelasticity/configs"
                / f"{name}.json"
            ).read_text()
        )
        for name in names
    ]
    key = jax.random.PRNGKey(310519)
    parameters = [model.init_model(key, config["model"]) for config in configs]
    shared = set(parameters[0]) & set(parameters[1])
    shared = {name for name in shared if not name.startswith("decoder.")}
    assert shared
    for name in shared:
        np.testing.assert_array_equal(parameters[0][name], parameters[1][name])
    for params, config in zip(parameters, configs):
        count = model.parameter_count(params)
        assert count <= config["training"]["max_real_parameters"]
        assert config["model"]["latent_dimension"] == 64
        assert config["model"]["decoder_width"] == 256
        assert config["model"]["decoder_blocks"] == 4
