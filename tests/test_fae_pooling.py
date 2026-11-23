import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from contgreedy.fae import model, training
from examples.hyperelasticity.fae_pooling_comparison import NAMES, initialization_audit

ROOT = Path(__file__).resolve().parents[1]


def experiment_configs():
    directory = ROOT / "examples/hyperelasticity/configs"
    return {
        name: json.loads((directory / f"{name}.json").read_text()) for name in NAMES
    }


def compact_model_config(config):
    result = dict(config["model"])
    result.update(
        latent_dimension=8,
        width=4,
        modes=2,
        pool_channels=2,
        kernel_hidden=8,
        decoder_width=8,
        decoder_blocks=1,
        coordinate_frequencies=2,
    )
    if result["pooling_kind"] == "quadrature_attention":
        result.update(
            attention_queries=2,
            attention_value_dimension=4,
            attention_key_dimension=8,
        )
    return result


def test_pooling_control_has_exact_shared_initialization_and_64_scalars():
    configurations = experiment_configs()
    audit = initialization_audit(configurations)
    assert audit["shared_initialization_exact"]
    assert audit["shared_parameter_arrays"] > 0
    assert all(
        config["model"]["latent_dimension"] == 64
        and config["training"]["epochs"] == 50
        and config["training"]["learning_rate"] == 1e-3
        for config in configurations.values()
    )
    assert all(count < 2_000_000 for count in audit["real_parameters"].values())


@pytest.mark.parametrize("name", NAMES)
def test_pooling_forward_and_gradients_are_finite(name):
    config = compact_model_config(experiment_configs()[name])
    params = model.init_model(jax.random.PRNGKey(91), config)
    field = jax.random.normal(jax.random.PRNGKey(92), (2, 12, 12))
    prediction, code = model.apply(params, field, config)
    assert prediction.shape == field.shape
    assert code.shape == (2, 8)
    assert jnp.all(jnp.isfinite(prediction)) and jnp.all(jnp.isfinite(code))
    loss, gradients = jax.value_and_grad(
        lambda values: training.objective(values, field, config, 1e-4)[0]
    )(params)
    assert jnp.isfinite(loss)
    assert all(
        jnp.all(jnp.isfinite(value))
        for value in jax.tree_util.tree_leaves(gradients)
    )
    pooling_gradient = sum(
        float(jnp.linalg.norm(value))
        for key, value in gradients.items()
        if key.startswith("pool.")
    )
    assert pooling_gradient > 0


def test_weighted_kernel_orthogonalization():
    kernel = jax.random.normal(jax.random.PRNGKey(93), (49, 4, 3))
    weights = model.trapezoid_weights(7, 7)
    normalized = model.orthogonalize_kernel(kernel, weights, 1e-7)
    gram = jnp.einsum("nlc,nmc,n->lm", normalized, normalized, weights)
    np.testing.assert_allclose(gram, np.eye(4), atol=2e-5, rtol=2e-5)


def test_attention_rejects_a_nonmatching_latent_factorization():
    config = compact_model_config(
        experiment_configs()["fae64_pool_attention_e50"]
    )
    config["attention_queries"] = 3
    with pytest.raises(ValueError, match="must equal latent_dimension"):
        model.init_model(jax.random.PRNGKey(94), config)
