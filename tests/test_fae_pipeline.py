"""Numerical checks for the native FAE infrastructure."""
import json
import jax
import jax.numpy as jnp
import numpy as np
import optax

from examples.hyperelasticity.fae import (
    FunctionalLoader, RelativeL2Metric, TrapezoidL2,
    build_autoencoder, grid, resample_fields, weights,
)
from contgreedy.fae.losses import make_fae_loss
from contgreedy.fae.trainer import Trainer


def config():
    return dict(model=dict(latent_dimension=4, encoder_channels=[4, 8], pool_channels=2,
                           kernel_hidden=8, orthogonal_epsilon=1e-3,
                           decoder_width=8, decoder_blocks=1, coordinate_frequencies=2),
                training=dict(batch_size=4, microbatch_size=2, gradient_clip=1.0))


def test_resampling_preserves_same_affine_field_and_endpoints():
    xy = grid(9)
    u = (2*xy[:, 0] + 3*xy[:, 1])[None, :]
    v = resample_fields(u, 6)[0, :, 0]
    new_xy = grid(6)
    np.testing.assert_allclose(v, 2*new_xy[:, 0]+3*new_xy[:, 1], atol=1e-6)


def test_loader_keeps_clean_targets_and_pairs():
    u = np.arange(4*16, dtype=np.float32).reshape(4, 16, 1)
    loader = FunctionalLoader(u, u, np.arange(4), batch_size=4, microbatch_size=2,
                              noise_std=0.05, seed=1, paired=True)
    batches = list(loader)
    assert len(batches) == 4
    for clean, noisy in zip(batches[:2], batches[2:]):
        np.testing.assert_array_equal(clean[0], noisy[0])
        np.testing.assert_array_equal(clean[0], clean[2])
        assert not np.array_equal(noisy[0], noisy[2])
    assert all(b[1].shape == (2, 16, 2) for b in batches)


def test_fae_loss_and_accumulated_adam_match_full_batch(tmp_path):
    cfg = config()
    ae = build_autoencoder(cfg)
    u = np.random.default_rng(2).normal(size=(4, 16, 1)).astype(np.float32)
    loader = FunctionalLoader(u, u, np.arange(4), batch_size=4, microbatch_size=2,
                              noise_std=0.05, seed=1, paired=True)
    loss_fn = make_fae_loss(ae, TrapezoidL2(), beta=1e-4)
    trainer = Trainer(ae, loss_fn, loader, loader, [], accumulation_steps=4,
                      gradient_clip=1.0, run_directory=tmp_path)
    init = trainer.initialize(jax.random.PRNGKey(1), learning_rate=1e-3,
                              minimum_learning_rate=1e-3, total_updates=1)
    batches = list(loader)
    combined = tuple(jnp.concatenate([b[i] for b in batches], axis=0) for i in range(4))
    truth, x_dec, observed, x_enc = combined
    (expected_loss, _), gradient = jax.value_and_grad(loss_fn, has_aux=True)(
        init.params, jax.random.PRNGKey(2), init.batch_stats,
        observed, x_enc, truth, x_dec)
    prediction = ae.apply(dict(params=init.params), observed, x_enc, x_dec)
    z = ae.encoder.apply(dict(params=init.params["encoder"]), observed, x_enc)
    analytic = 0.5*jnp.mean(TrapezoidL2().squared_norm(prediction-truth, x_dec)) + 1e-4*jnp.mean(jnp.sum(z*z, -1))
    np.testing.assert_allclose(expected_loss, analytic, rtol=1e-6)
    reference_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
    updates, reference_opt_state = reference_tx.update(gradient, reference_tx.init(init.params), init.params)
    expected_params = optax.apply_updates(init.params, updates)
    state = init
    step_fn = trainer.train_step
    losses = []
    for batch in batches:
        loss, state = step_fn(state, batch)
        losses.append(loss)
    np.testing.assert_allclose(jnp.mean(jnp.stack(losses)), expected_loss, rtol=2e-5)
    # GroupNorm cancels some bias gradients exactly in real arithmetic; tiny
    # float32 cancellation differences are amplified by Adam's 1e-8 epsilon.
    # Check ALL optimizer moments, and parameter updates on resolved gradients.
    for actual, expected in zip(jax.tree_util.tree_leaves(state.opt_state.inner_opt_state),
                               jax.tree_util.tree_leaves(reference_opt_state)):
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-4)
    for actual, expected, g in zip(jax.tree_util.tree_leaves(state.params),
                                  jax.tree_util.tree_leaves(expected_params),
                                  jax.tree_util.tree_leaves(gradient)):
        resolved = np.abs(np.asarray(g)) > 1e-5
        np.testing.assert_allclose(np.asarray(actual)[resolved], np.asarray(expected)[resolved],
                                   atol=3e-6, rtol=1e-4)
    assert int(state.opt_state.gradient_step) == 1


def test_reference_evaluation_uses_original_grid_and_physical_units():
    cfg = config()
    ae = build_autoencoder(cfg)
    u = np.ones((2, 16, 1), np.float32)
    target = np.ones((2, 36, 1), np.float32)*2
    loader = FunctionalLoader(u, target, np.arange(2), batch_size=2)
    batch = next(iter(loader))
    truth, xd, obs, xe = batch
    variables = ae.init(jax.random.PRNGKey(4), obs, xe, xd)
    class State:
        params = variables["params"]
        batch_stats = None
    metric = RelativeL2Metric(ae, mean=3, std=2)
    score = metric(State(), jax.random.PRNGKey(1), loader)
    predicted = ae.apply(variables, obs, xe, xd)
    w = weights(36)[None, :, None]
    expected = jnp.mean(jnp.sqrt(jnp.sum(((predicted-truth)*2)**2*w, (1,2))/
                                jnp.sum((truth*2+3)**2*w, (1,2))))
    np.testing.assert_allclose(score, expected, rtol=1e-6)


def test_minimal_trainer_fit_and_artifacts(tmp_path):
    cfg = config()
    ae = build_autoencoder(cfg)
    u = np.random.default_rng(5).normal(size=(4, 16, 1)).astype(np.float32)
    loader = FunctionalLoader(u, u, np.arange(4), batch_size=4, microbatch_size=2)
    trainer = Trainer(
        ae,
        make_fae_loss(ae, TrapezoidL2(), beta=1e-4),
        loader,
        accumulation_steps=2,
        run_directory=tmp_path,
    )
    state = trainer.initialize(
        jax.random.PRNGKey(6), learning_rate=1e-3,
        minimum_learning_rate=1e-4, total_updates=2,
    )
    result = trainer.fit(state, epochs=2)
    assert int(result["last_state"].step) == 4
    assert len(result["history"]) == 2
    assert (tmp_path / "last.msgpack").is_file()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["epoch"] == 2 and status["completed_updates"] == 2
