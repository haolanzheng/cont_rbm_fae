"""Numerical migration checks; PyTorch is only the independent test reference."""

import copy
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from contgreedy.fae import model
from contgreedy.fae import training




def reference_module():
    path = Path(__file__).parent / "fixtures/fae_reference_model.py"
    spec = importlib.util.spec_from_file_location("fae_reference_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
ROOT = Path(__file__).resolve().parents[1]


def small_config():
    config = json.loads(
        (ROOT / "examples/hyperelasticity/configs/fae128_jax_w32.json").read_text()
    )
    config["model"].update(
        latent_dimension=8,
        width=4,
        modes=2,
        pool_channels=2,
        kernel_hidden=8,
        decoder_width=8,
        decoder_blocks=1,
        condition_width=8,
        coordinate_frequencies=2,
    )
    return config


def test_model_matches_reference_forward_gradients_and_adam():
    torch = pytest.importorskip("torch")
    reference_code = reference_module()
    KernelPoolFAE = reference_code.KernelPoolFAE
    trapezoid_weights = reference_code.trapezoid_weights

    torch.manual_seed(41)
    config = small_config()
    reference = KernelPoolFAE(**config["model"])
    params = {}
    for name, value in reference.named_parameters():
        raw = value.detach().numpy().copy()
        params[name] = jnp.asarray(
            np.stack((raw.real, raw.imag), -1) if np.iscomplexobj(raw) else raw
        )
    initialized = model.init_model(jax.random.PRNGKey(1), config["model"])
    assert params.keys() == initialized.keys()
    for name in params:
        assert params[name].shape == initialized[name].shape
    raw = np.random.default_rng(4).normal(size=(2, 12, 12)).astype(np.float32)
    tx = torch.tensor(raw, requires_grad=True)
    prediction, code = reference(tx)
    jpred, jcode = jax.jit(lambda p, x: model.apply(p, x, config["model"]))(params, raw)
    np.testing.assert_allclose(jpred, prediction.detach().numpy(), rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(jcode, code.detach().numpy(), rtol=2e-4, atol=2e-5)
    weights = trapezoid_weights(12, 12)
    loss = (
        0.5 * ((prediction - tx).flatten(1) ** 2 * weights).sum(1).mean()
        + 1e-4 * code.square().sum(1).mean()
    )
    loss.backward()
    fn = lambda p, x: training.objective(p, x, config["model"], 1e-4)[0]
    jloss, (grads, input_grad) = jax.jit(jax.value_and_grad(fn, argnums=(0, 1)))(
        params, raw
    )
    np.testing.assert_allclose(jloss, loss.detach().numpy(), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(input_grad, tx.grad.numpy(), rtol=2e-3, atol=2e-6)
    differences, norms = [], []
    for name, value in reference.named_parameters():
        g = value.grad.numpy()
        g = np.stack((g.real, g.imag), -1) if np.iscomplexobj(g) else g
        differences.append(np.linalg.norm(np.asarray(grads[name]) - g) ** 2)
        norms.append(np.linalg.norm(g) ** 2)
    assert np.sqrt(sum(differences) / sum(norms)) < 2e-4
    opt = torch.optim.Adam(
        reference.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8
    )
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    opt.step()
    updated, _ = training.adam_update(
        training.init_adam(params), grads, jnp.float32(1e-3), config["training"]
    )
    for name, value in reference.named_parameters():
        p = value.detach().numpy()
        p = np.stack((p.real, p.imag), -1) if np.iscomplexobj(p) else p
        np.testing.assert_allclose(
            updated["params"][name], p, rtol=2e-4, atol=1e-5, err_msg=name
        )


def test_widened_parameter_count_and_schedule():
    torch = pytest.importorskip("torch")
    reference_code = reference_module()
    KernelPoolFAE = reference_code.KernelPoolFAE
    parameter_counts = reference_code.parameter_counts

    config = json.loads(
        (ROOT / "examples/hyperelasticity/configs/fae128_jax_w32.json").read_text()
    )
    params = jax.jit(lambda key: model.init_model(key, config["model"]))(
        jax.random.PRNGKey(0)
    )
    expected = parameter_counts(KernelPoolFAE(**config["model"]))["real_scalars"]
    assert model.parameter_count(params) == expected
    assert expected <= config["training"]["max_real_parameters"]
    assert config["model"]["width"] == 32
    assert config["training"]["warmup_epochs"] == 0
    assert training.learning_rate(1, config["training"]) == pytest.approx(1e-3)
    assert training.learning_rate(6400, config["training"]) == pytest.approx(1e-5)


@pytest.mark.parametrize("decoder_kind", ["pod", "pod_film"])
def test_pod_decoders_have_no_layer_norm_and_use_latent_scale(decoder_kind):
    config = small_config()
    config["model"] = {
        **config["model"],
        "decoder_kind": decoder_kind,
        "n_basis": 8,
        "basis_n_layers": 2,
        "basis_hidden_dim": 12,
        "coefs_n_layers": 2,
        "coefs_hidden_dim": 12,
        "basis_film_scale": 0.01,
    }
    params = model.init_model(jax.random.PRNGKey(17), config["model"])
    decoder_names = [name for name in params if name.startswith("decoder.")]
    assert decoder_names
    assert not any("norm" in name for name in decoder_names)
    code = jax.random.normal(jax.random.PRNGKey(18), (3, 8))
    xy = model.coordinates((9, 11))
    decoded = model.decode_coordinates(params, code, xy, config["model"])
    scaled = model.decode_coordinates(params, 2 * code, xy, config["model"])
    assert decoded.shape == (3, 99)
    assert jnp.all(jnp.isfinite(decoded))
    assert not np.allclose(decoded, scaled)
    loss, grads = jax.value_and_grad(
        lambda p: jnp.mean(model.decode_coordinates(p, code, xy, config["model"]) ** 2)
    )(params)
    assert jnp.isfinite(loss)
    assert all(
        jnp.all(jnp.isfinite(value)) for value in jax.tree_util.tree_leaves(grads)
    )
    if decoder_kind == "pod_film":
        film = grads["decoder.basis.blocks.0.film.weight"]
        assert jnp.linalg.norm(film) > 0


def test_pod_and_pod_film_share_exact_initial_base_parameters():
    config = small_config()["model"]
    pod = {
        **config,
        "decoder_kind": "pod",
        "n_basis": 8,
        "basis_n_layers": 2,
        "basis_hidden_dim": 12,
        "coefs_n_layers": 2,
        "coefs_hidden_dim": 12,
    }
    film = {**pod, "decoder_kind": "pod_film", "basis_film_scale": 0.01}
    key = jax.random.PRNGKey(52)
    fixed_params = model.init_model(key, pod)
    film_params = model.init_model(key, film)
    for name, value in fixed_params.items():
        np.testing.assert_array_equal(value, film_params[name])
    extra = set(film_params) - set(fixed_params)
    assert extra == {
        f"decoder.basis.blocks.{i}.film.{leaf}"
        for i in range(2)
        for leaf in ("weight", "bias")
    }
    assert all(jnp.all(film_params[name] == 0) for name in extra)


@pytest.mark.parametrize("batch_size", [16, 64])
def test_training_resume_reproduces_uninterrupted_run(tmp_path, batch_size):
    config = small_config()
    config["training"].update(
        train_size=batch_size, batch_size=batch_size, epochs=2, evaluation_interval_epochs=1
    )
    data = tmp_path / "data"
    data.mkdir()
    fields = np.random.default_rng(9).normal(size=(batch_size + 32, 144)).astype(np.float32)
    np.save(data / "fields.npy", fields)
    (data / "metadata.json").write_text(
        json.dumps(
            {
                "shape": [batch_size + 32, 144],
                "grid_resolution": 12,
                "splits": {
                    "train": [0, batch_size],
                    "validation": [batch_size, batch_size + 16],
                    "test": [batch_size + 16, batch_size + 32],
                },
                "mean": float(fields[:batch_size].mean()),
                "std": float(fields[:batch_size].std()),
            }
        )
    )
    full, split = tmp_path / "full", tmp_path / "split"
    training.train(config, data_directory=data, run_directory=full, require_gpu=False)
    training.train(
        config,
        data_directory=data,
        run_directory=split,
        stop_after_epochs=1,
        require_gpu=False,
    )
    assert not (split / "metrics.json").exists()
    training.train(
        config, data_directory=data, run_directory=split, resume=True, require_gpu=False
    )
    a, am = training.load_checkpoint(full / "last.npz", config)
    b, bm = training.load_checkpoint(split / "last.npz", config)
    for group in ("params", "m", "v"):
        for name in a[group]:
            np.testing.assert_allclose(
                a[group][name], b[group][name], rtol=1e-6, atol=1e-8,
                err_msg="GPU restart may differ in the last float32 reduction bits",
            )
    assert int(a["step"]) == int(b["step"]) == 2
    assert am["numpy_rng"] == bm["numpy_rng"]
    result = json.loads((split / "metrics.json").read_text())
    assert result["status"] == "completed" and result["test"]["n"] == 16
