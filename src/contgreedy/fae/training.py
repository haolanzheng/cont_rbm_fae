"""Pure-JAX training utilities for the array-based FAE implementation."""

from __future__ import annotations

import hashlib
import copy
import json
import math
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from .model import apply, init_model, parameter_count, trapezoid_weights
from .encoder import fit_input_normalization
from .corruption import reconstruction_objective


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def objective(params, target, config, penalty):
    prediction, code = apply(params, target, config)
    weights = trapezoid_weights(*target.shape[-2:])
    reconstruction = 0.5 * jnp.mean(
        jnp.sum((prediction - target).reshape(len(target), -1) ** 2 * weights, axis=1)
    )
    latent = jnp.mean(jnp.sum(code**2, axis=1))
    return reconstruction + penalty * latent, (reconstruction, latent)


def init_adam(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"params": params, "m": zeros, "v": zeros, "step": jnp.array(0, jnp.int32)}


def learning_rate(step, training):
    """First update is exactly the configured initial LR; last is the minimum."""
    total = training["epochs"] * training["train_size"] // training["batch_size"]
    fraction = (step - 1) / max(total - 1, 1)
    return training["minimum_learning_rate"] + 0.5 * (
        training["learning_rate"] - training["minimum_learning_rate"]
    ) * (1 + math.cos(math.pi * fraction))


def adam_update(state, gradients, lr, training):
    norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree_util.tree_leaves(gradients)))
    scale = jnp.minimum(
        jnp.float32(1), training["gradient_clip"] / (norm + jnp.float32(1e-6))
    )
    gradients = jax.tree_util.tree_map(lambda g: g * scale, gradients)
    b1, b2 = jnp.float32(training["beta1"]), jnp.float32(training["beta2"])
    step = state["step"] + 1
    m = jax.tree_util.tree_map(
        lambda m, g: b1 * m + (1 - b1) * g, state["m"], gradients
    )
    v = jax.tree_util.tree_map(
        lambda v, g: b2 * v + (1 - b2) * g * g, state["v"], gradients
    )
    params = jax.tree_util.tree_map(
        lambda p, a, b: (
            p
            - lr
            * (a / (1 - b1**step))
            / (jnp.sqrt(b / (1 - b2**step)) + training["epsilon"])
        ),
        state["params"],
        m,
        v,
    )
    return {"params": params, "m": m, "v": v, "step": step}, norm


def make_step(config):
    def step(state, target, lr):
        (loss, (reconstruction, latent)), grads = jax.value_and_grad(
            objective, has_aux=True
        )(
            state["params"],
            target,
            config["model"],
            config["training"]["latent_penalty"],
        )
        updated, norm = adam_update(state, grads, lr, config["training"])
        finite = jnp.isfinite(loss) & jnp.isfinite(norm)
        finite = finite & jnp.all(
            jnp.stack(
                [
                    jnp.all(jnp.isfinite(v))
                    for v in jax.tree_util.tree_leaves(updated["params"])
                ]
            )
        )
        return updated, {
            "loss": loss,
            "reconstruction": reconstruction,
            "latent_squared_norm": latent,
            "gradient_norm": norm,
            "finite": finite,
        }

    return jax.jit(step)


def make_corrupted_step(config):
    """Adam step with distinct encoder input and clean high-resolution target."""

    def step(state, encoder_input, target, observation_mask, decoder_indices, lr):
        (loss, (reconstruction, latent)), grads = jax.value_and_grad(
            reconstruction_objective, has_aux=True
        )(
            state["params"],
            encoder_input,
            target,
            observation_mask,
            decoder_indices,
            config["model"],
            config["training"]["latent_penalty"],
        )
        updated, norm = adam_update(state, grads, lr, config["training"])
        finite = jnp.isfinite(loss) & jnp.isfinite(norm)
        finite = finite & jnp.all(
            jnp.stack(
                [
                    jnp.all(jnp.isfinite(v))
                    for v in jax.tree_util.tree_leaves(updated["params"])
                ]
            )
        )
        return updated, {
            "loss": loss,
            "reconstruction": reconstruction,
            "latent_squared_norm": latent,
            "gradient_norm": norm,
            "finite": finite,
        }

    return jax.jit(step)


def save_checkpoint(path, state, metadata):
    path = Path(path)
    arrays = {}
    for group in ("params", "m", "v"):
        arrays.update({group + "/" + k: np.asarray(v) for k, v in state[group].items()})
    arrays["step"] = np.asarray(state["step"])
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def load_checkpoint(path, config):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata["config"] != config:
            raise ValueError(
                "Resume requires an identical model and training configuration"
            )
        state = {
            group: {
                k[len(group) + 1 :]: jnp.asarray(archive[k])
                for k in archive.files
                if k.startswith(group + "/")
            }
            for group in ("params", "m", "v")
        }
        state["step"] = jnp.asarray(archive["step"])
    return state, metadata


def stats(values):
    return dict(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        p95=float(np.quantile(values, 0.95)),
        max=float(np.max(values)),
    )


def make_evaluator(config, mean, std):
    @jax.jit
    def evaluate_batch(params, raw):
        predicted, code = apply(params, (raw - mean) / std, config["model"])
        weights = trapezoid_weights(*raw.shape[-2:])
        prediction = predicted * std + mean
        error = jnp.sqrt(
            jnp.sum((prediction - raw).reshape(len(raw), -1) ** 2 * weights, axis=1)
            / jnp.maximum(
                jnp.sum(raw.reshape(len(raw), -1) ** 2 * weights, axis=1), 1e-12
            )
        )
        return error, jnp.sum(code**2, axis=1)

    return evaluate_batch


def evaluate(evaluator, params, fields, ids, n, batch_size):
    errors, latent = [], []
    for first in range(0, len(ids), batch_size):
        batch = np.array(
            fields[ids[first : first + batch_size]], dtype=np.float32
        ).reshape(-1, n, n)
        err, code_norm = jax.device_get(evaluator(params, batch))
        errors.extend(err.tolist())
        latent.extend(code_norm.tolist())
    return {
        "n": len(errors),
        "relative_L2": stats(errors),
        "latent_squared_norm_mean": float(np.mean(latent)),
    }


def train(
    config,
    *,
    data_directory,
    run_directory,
    resume=False,
    stop_after_epochs=0,
    require_gpu=True,
):
    config = copy.deepcopy(config)
    training = config["training"]
    if training.get("warmup_epochs", 0) != 0:
        raise ValueError(
            "This protocol starts at the configured LR; warmup must be zero"
        )
    if training.get("weight_decay", 0) != 0:
        raise ValueError("This protocol uses Adam without weight decay")
    if not 1 <= training["train_size"] <= 8192 or training["batch_size"] not in (
        16,
        32,
        64,
    ):
        raise ValueError("Require train_size<=8192 and batch_size in {16,32,64}")
    if training["train_size"] % training["batch_size"]:
        raise ValueError("Require complete training batches")
    if require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("JAX CUDA is required; refusing to train on the login CPU")
    jax.config.update("jax_default_matmul_precision", "highest")
    run, data = Path(run_directory), Path(data_directory)
    run.mkdir(parents=True, exist_ok=True)
    last = run / "last.npz"
    if last.exists() and not resume:
        raise ValueError(
            "Checkpoint exists; choose a fresh run directory or explicitly resume"
        )
    metadata = json.loads((data / "metadata.json").read_text())
    fields = np.load(data / "fields.npy", mmap_mode="r")
    if list(fields.shape) != metadata["shape"]:
        raise ValueError("Input array and metadata shapes disagree")
    n = metadata["grid_resolution"]
    train0, train1 = metadata["splits"]["train"]
    if training["train_size"] > train1 - train0:
        raise ValueError("Insufficient training samples")
    train_ids = np.arange(train0, train0 + training["train_size"])
    validation_ids = np.arange(*metadata["splits"]["validation"])
    test_ids = np.arange(*metadata["splits"]["test"])
    mean, std = metadata["mean"], metadata["std"]
    input_mode = training.get("input_mode", "clean")
    if input_mode not in ("clean", "multiresolution", "masked", "clean_noisy_pair"):
        raise ValueError(f"Unknown input_mode: {input_mode}")
    if input_mode != "clean":
        if config["model"].get("encoder_kind") != "darcy_unet":
            raise ValueError("Corrupted-input training requires the Darcy U-Net encoder")
        if config["model"].get("encoder_input_features") != "raw":
            raise ValueError("Corrupted-input training requires raw-only field features")
    if input_mode == "multiresolution":
        if config["model"].get("encoder_use_mask_channel", False):
            raise ValueError("Multiresolution input must not use a mask channel")
        input_resolutions = tuple(int(v) for v in training["input_resolutions"])
        probabilities = np.asarray(
            training["input_resolution_probabilities"], dtype=np.float64
        )
        if (
            len(input_resolutions) != len(probabilities)
            or min(input_resolutions) < 4
            or max(input_resolutions) > n
            or not np.isclose(probabilities.sum(), 1)
            or np.any(probabilities <= 0)
        ):
            raise ValueError("Invalid multiresolution sampling distribution")
        decoder_sample_count = int(training["decoder_sample_count"])
        if not 1 <= decoder_sample_count <= n * n:
            raise ValueError("decoder_sample_count exceeds the target grid")
    elif input_mode == "masked":
        if not config["model"].get("encoder_use_mask_channel", False):
            raise ValueError("Masked input requires an explicit observation-mask channel")
        mask_fraction = float(training["mask_fraction"])
        if not 0 < mask_fraction < 1:
            raise ValueError("mask_fraction must lie strictly between zero and one")
    elif input_mode == "clean_noisy_pair":
        if config["model"].get("encoder_use_mask_channel", False):
            raise ValueError("Clean/noisy pairs must not use a mask channel")
        noise_kind = str(training["noise_kind"])
        if noise_kind != "iid_gaussian":
            raise ValueError("This protocol currently supports iid_gaussian noise")
        noise_std = float(training["noise_std"])
        if not 0 < noise_std < 1:
            raise ValueError("noise_std must lie strictly between zero and one")
    if config["model"].get("encoder_kind") == "darcy_unet":
        if resume:
            saved_config = json.loads((run / "config.json").read_text())
            requested, saved_base = copy.deepcopy(config), copy.deepcopy(saved_config)
            requested["model"].pop("encoder_input_normalization", None)
            saved_base["model"].pop("encoder_input_normalization", None)
            if requested != saved_base:
                raise ValueError("Resume requires an identical model and training configuration")
            config = saved_config
        else:
            config["model"]["encoder_input_normalization"] = fit_input_normalization(
                fields, train_ids, n, mean, std, config["model"]
            )
    rng = np.random.default_rng(training["seed"])
    history, start_epoch, best, previous_seconds = [], 0, float("inf"), 0.0
    if resume:
        state, saved = load_checkpoint(last, config)
        rng.bit_generator.state = saved["numpy_rng"]
        history, start_epoch = saved["history"], saved["epoch"]
        best, previous_seconds = saved["best"], saved["elapsed_seconds"]
    else:
        params = jax.jit(lambda key: init_model(key, config["model"]))(
            jax.random.PRNGKey(training["seed"])
        )
        state = init_adam(params)
    count = parameter_count(state["params"])
    if count > training["max_real_parameters"]:
        raise ValueError("Model exceeds the real-parameter budget")
    steps_per_epoch = len(train_ids) // training["batch_size"]
    if int(state["step"]) != start_epoch * steps_per_epoch:
        raise ValueError("Checkpoint epoch and optimizer update count disagree")
    stop = min(stop_after_epochs or training["epochs"], training["epochs"])
    if stop <= start_epoch:
        raise ValueError("Requested stopping epoch is already complete")
    source_dir = Path(__file__).resolve().parent
    provenance = {
        "backend": "jax",
        "jax": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "platform": jax.default_backend(),
        "real_parameters": count,
        "float_dtype": "float32",
        "complex_dtype": "complex64",
        "matmul_precision": "highest",
        "sources_sha256": {
            name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
            for name in ("model.py", "encoder.py", "corruption.py", "training.py")
        },
        "data_sha256": hashlib.sha256((data / "fields.npy").read_bytes()).hexdigest(),
        "data_metadata": metadata,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "random_initialization": not resume,
    }
    atomic_json(run / "config.json", config)
    atomic_json(run / "provenance.json", provenance)
    atomic_json(
        run / "status.json",
        {
            "status": "initializing",
            "epoch": start_epoch,
            "completed_updates": int(state["step"]),
            "planned_epochs": training["epochs"],
        },
    )
    print(
        json.dumps({**provenance, "start_epoch": start_epoch, "stop_epoch": stop}),
        flush=True,
    )
    step_fn = (
        make_step(config)
        if input_mode == "clean"
        else make_corrupted_step(config)
    )
    evaluator = make_evaluator(config, mean, std)
    full_decoder_indices = np.arange(n * n, dtype=np.int32)
    resizers = {}
    if input_mode == "multiresolution":
        for resolution in input_resolutions:
            if resolution == n:
                continue

            def resize(values, output_resolution=resolution):
                return jax.image.resize(
                    values,
                    (values.shape[0], output_resolution, output_resolution),
                    method="linear",
                    antialias=True,
                )

            resizers[resolution] = jax.jit(resize)
    started = time.perf_counter()
    for epoch in range(start_epoch + 1, stop + 1):
        order = rng.permutation(train_ids)
        metrics = []
        resolution_counts = {value: 0 for value in input_resolutions} if input_mode == "multiresolution" else {}
        visible_fractions = []
        for b in range(steps_per_epoch):
            step = (epoch - 1) * steps_per_epoch + b + 1
            raw = np.array(
                fields[
                    order[b * training["batch_size"] : (b + 1) * training["batch_size"]]
                ],
                dtype=np.float32,
            ).reshape(-1, n, n)
            target = np.asarray((raw - mean) / std, dtype=np.float32)
            lr = learning_rate(step, training)
            if input_mode == "clean":
                state, row = step_fn(state, target, np.float32(lr))
            elif input_mode == "multiresolution":
                resolution = int(rng.choice(input_resolutions, p=probabilities))
                resolution_counts[resolution] += 1
                target_device = jnp.asarray(target)
                encoder_input = (
                    target_device
                    if resolution == n
                    else resizers[resolution](target_device)
                )
                observation_mask = jnp.ones_like(encoder_input)
                decoder_indices = np.sort(
                    rng.choice(
                        n * n, size=decoder_sample_count, replace=False
                    ).astype(np.int32)
                )
                state, row = step_fn(
                    state,
                    encoder_input,
                    target_device,
                    observation_mask,
                    decoder_indices,
                    np.float32(lr),
                )
            elif input_mode == "masked":
                observation_mask = (
                    rng.random(target.shape) >= mask_fraction
                ).astype(np.float32)
                visible_fractions.append(float(observation_mask.mean()))
                encoder_input = target * observation_mask
                state, row = step_fn(
                    state,
                    encoder_input,
                    target,
                    observation_mask,
                    full_decoder_indices,
                    np.float32(lr),
                )
            else:
                target_device = jnp.asarray(target)
                noise_key = jax.random.fold_in(
                    jax.random.PRNGKey(training["seed"] + 104729), step
                )
                noisy_input = target_device + noise_std * jax.random.normal(
                    noise_key, target_device.shape, dtype=target_device.dtype
                )
                encoder_input = jnp.concatenate((target_device, noisy_input), axis=0)
                paired_target = jnp.concatenate((target_device, target_device), axis=0)
                state, row = step_fn(
                    state,
                    encoder_input,
                    paired_target,
                    jnp.ones_like(encoder_input),
                    full_decoder_indices,
                    np.float32(lr),
                )
            row = jax.device_get(row)
            if not bool(row["finite"]):
                raise FloatingPointError(
                    f"Nonfinite loss/gradient/parameters at epoch {epoch}, batch {b}"
                )
            metrics.append(row)
        row = {
            "epoch": epoch,
            "step": epoch * steps_per_epoch,
            **{
                k: float(np.mean([m[k] for m in metrics]))
                for k in ("loss", "reconstruction", "latent_squared_norm")
            },
            "learning_rate": lr,
            "last_gradient_norm": float(metrics[-1]["gradient_norm"]),
            "elapsed_seconds": previous_seconds + time.perf_counter() - started,
        }
        if input_mode == "multiresolution":
            row["input_resolution_batches"] = {
                str(key): value for key, value in resolution_counts.items()
            }
            row["decoder_sample_count"] = decoder_sample_count
        elif input_mode == "masked":
            row["visible_fraction_mean"] = float(np.mean(visible_fractions))
        elif input_mode == "clean_noisy_pair":
            row["pair_copies_per_field"] = 2
            row["noise_kind"] = noise_kind
            row["noise_std"] = noise_std
        if (
            epoch == 1
            or epoch % training["evaluation_interval_epochs"] == 0
            or epoch == stop
        ):
            row["validation"] = evaluate(
                evaluator,
                state["params"],
                fields,
                validation_ids,
                n,
                training["batch_size"],
            )
            row["train_monitor_128"] = evaluate(
                evaluator,
                state["params"],
                fields,
                train_ids[:128],
                n,
                training["batch_size"],
            )
            score = row["validation"]["relative_L2"]["mean"]
            if score < best:
                best = score
                best_state = {
                    "params": state["params"],
                    "m": {},
                    "v": {},
                    "step": state["step"],
                }
                save_checkpoint(
                    run / "best.npz",
                    best_state,
                    {"config": config, "epoch": epoch, "validation": row["validation"]},
                )
        history.append(row)
        save_checkpoint(
            last,
            state,
            {
                "config": config,
                "epoch": epoch,
                "history": history,
                "best": best,
                "numpy_rng": rng.bit_generator.state,
                "elapsed_seconds": row["elapsed_seconds"],
            },
        )
        atomic_json(run / "history.json", history)
        atomic_json(
            run / "status.json",
            {
                "status": "running",
                "epoch": epoch,
                "planned_epochs": training["epochs"],
                "completed_updates": epoch * steps_per_epoch,
                "best_validation_relative_L2": best,
            },
        )
        print("EPOCH", json.dumps(row), flush=True)
    result = {
        "status": "completed" if stop == training["epochs"] else "paused",
        "epoch": stop,
        "planned_epochs": training["epochs"],
        "completed_updates": stop * steps_per_epoch,
        "real_parameters": count,
        "last_epoch": history[-1],
        "best_validation_relative_L2": best,
    }
    if result["status"] == "completed":
        chosen, selected = load_checkpoint(run / "best.npz", config)
        result["selected_epoch"] = selected["epoch"]
        result["test"] = evaluate(
            evaluator, chosen["params"], fields, test_ids, n, training["batch_size"]
        )
        atomic_json(run / "metrics.json", result)
    atomic_json(run / "status.json", result)
    print("FINAL_STATUS", json.dumps(result), flush=True)
    return result
