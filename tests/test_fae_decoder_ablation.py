"""Contract tests for the controlled nonlinear-decoder ablation."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.hyperelasticity.fae import build_autoencoder, grid


ROOT = Path(__file__).resolve().parents[1]
KINDS = (
    "concat",
    "film_shift",
    "film_scale_shift",
    "low_rank",
    "cross_attention",
    "multiscale_local",
)


def load_config(kind):
    path = (
        ROOT
        / "examples/hyperelasticity/configs"
        / f"fae100_decoder_{kind}_l64_e50.json"
    )
    return json.loads(path.read_text())


def test_ablation_configs_only_change_decoder_kind():
    controlled = []
    for kind in KINDS:
        config = load_config(kind)
        assert config["model"]["latent_dimension"] == 64
        assert config["model"]["decoder_kind"] == kind
        config["model"].pop("decoder_kind")
        controlled.append(config)
    assert all(config == controlled[0] for config in controlled[1:])


@pytest.mark.parametrize("kind", KINDS)
def test_decoder_is_finite_and_latent_dependent(kind):
    decoder = build_autoencoder(load_config(kind)).decoder
    key = jax.random.PRNGKey(41)
    z = jax.random.normal(key, (2, 64))
    xy = jnp.broadcast_to(jnp.asarray(grid(5))[None], (2, 25, 2))
    variables = decoder.init(key, z, xy)
    decoded = decoder.apply(variables, z, xy)
    gradient = jax.grad(
        lambda code: jnp.mean(decoder.apply(variables, code, xy) ** 2)
    )(z)

    assert decoded.shape == (2, 25, 1)
    assert np.all(np.isfinite(decoded))
    assert np.all(np.isfinite(gradient))
    assert float(jnp.linalg.norm(gradient)) > 0
    assert not np.allclose(decoded[0], decoded[1])
