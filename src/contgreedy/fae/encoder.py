"""Pure-JAX convolutional encoder and input scaling.

The U-Net follows fae-rebano/src/uno_encoder.py: 3x3 convolutions,
pre-normalized residual blocks, linear resize, concatenated multiscale skips,
and a zero-gated raw-input projection. Pooling remains in model.py.
"""

import jax
import jax.numpy as jnp
import numpy as np


def field_feature_count(config):
    """Number of deterministic channels derived from the observed field."""
    kind = config.get("encoder_input_features", "raw_smooth_residual")
    if kind == "raw":
        return 1
    if kind == "raw_smooth_residual":
        return 3
    raise ValueError(f"Unknown encoder_input_features: {kind}")


def input_features(field, config):
    """Return either raw-only or raw/smoothed/residual field channels."""
    if config.get("encoder_input_features", "raw_smooth_residual") == "raw":
        return field[..., None]
    radius = int(config.get("encoder_smooth_radius", 2))
    sigma = float(config.get("encoder_smooth_sigma", 0.8))
    if radius < 1 or sigma <= 0 or min(field.shape[1:]) <= radius:
        raise ValueError("Require positive smoothing radius/sigma and a larger grid")
    offsets = jnp.arange(-radius, radius + 1, dtype=field.dtype)
    kernel = jnp.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / jnp.sum(kernel)
    smooth = field
    for axis in (1, 2):
        padding = [(0, 0)] * 3
        padding[axis] = (radius, radius)
        padded = jnp.pad(smooth, padding, mode="reflect")
        smooth = sum(
            kernel[i] * jax.lax.slice_in_dim(
                padded, i, i + field.shape[axis], axis=axis
            )
            for i in range(2 * radius + 1)
        )
    return jnp.stack((field, smooth, field - smooth), axis=-1)


def fit_input_normalization(fields, ids, resolution, mean, std, config):
    """Fit only on the explicit training IDs, in the standardized-target units."""
    channels = field_feature_count(config)
    extract = jax.jit(lambda field: input_features(field, config))
    total = np.zeros(channels, dtype=np.float64)
    squares = np.zeros(channels, dtype=np.float64)
    count = 0
    for first in range(0, len(ids), 32):
        raw = np.asarray(fields[ids[first:first + 32]], dtype=np.float32)
        field = ((raw - mean) / std).reshape(-1, resolution, resolution)
        features = np.asarray(extract(field), dtype=np.float64).reshape(-1, channels)
        total += features.sum(axis=0)
        squares += np.square(features).sum(axis=0)
        count += len(features)
    center = total / count
    scale = np.sqrt(np.maximum(squares / count - center**2, 0))
    scale = np.maximum(scale, 1e-8)
    return {
        "mean": center.astype(np.float32).tolist(),
        "std": scale.astype(np.float32).tolist(),
        "sample_count": int(len(ids)),
        "training_id_first": int(ids[0]),
        "training_id_last": int(ids[-1]),
        "units": "standardized target",
    }


def normalized_inputs(field, config, observation_mask=None):
    stats = config.get("encoder_input_normalization")
    if stats is None:
        raise ValueError("Darcy encoder requires saved training-only input normalization")
    normalized = (
        input_features(field, config) - jnp.asarray(stats["mean"])
    ) / jnp.asarray(stats["std"])
    if config.get("encoder_use_mask_channel", False):
        if observation_mask is None:
            observation_mask = jnp.ones_like(field)
        if observation_mask.shape != field.shape:
            raise ValueError("Observation mask must have the field shape")
        normalized = jnp.concatenate(
            (normalized, observation_mask[..., None]), axis=-1
        )
    return normalized


def init_unet(key, config):
    params = {}
    channels = tuple(config.get("encoder_channels", (32, 64, 128)))
    blocks = int(config.get("encoder_blocks_per_level", 1))
    if not channels or min(channels) <= 0 or blocks < 1:
        raise ValueError("Require positive encoder channels and residual block count")

    def conv(name, inputs, outputs):
        nonlocal key
        key, draw = jax.random.split(key)
        params[name + ".weight"] = jax.nn.initializers.glorot_normal()(
            draw, (3, 3, inputs, outputs), jnp.float32
        )
        params[name + ".bias"] = jnp.zeros(outputs, jnp.float32)

    def norm(name, width):
        params[name + ".weight"] = jnp.ones(width, jnp.float32)
        params[name + ".bias"] = jnp.zeros(width, jnp.float32)

    def stage(name, inputs, outputs):
        conv(name + ".input", inputs, outputs)
        for i in range(blocks):
            block = f"{name}.blocks.{i}"
            norm(block + ".norm_0", outputs)
            conv(block + ".conv_0", outputs, outputs)
            norm(block + ".norm_1", outputs)
            conv(block + ".conv_1", outputs, outputs)

    field_channels = field_feature_count(config) + int(
        config.get("encoder_use_mask_channel", False)
    )
    inputs = field_channels + 2  # Append two unencoded coordinates.
    for i, width in enumerate(channels):
        stage(f"encoder.down.{i}", inputs, width)
        inputs = width
    for i in reversed(range(len(channels) - 1)):
        stage(f"encoder.up.{i}", inputs + channels[i], channels[i])
        inputs = channels[i]
    norm("encoder.output_norm", inputs)
    conv("encoder.output", inputs, config["pool_channels"])
    key, draw = jax.random.split(key)
    params["encoder.token_skip.weight"] = jax.nn.initializers.lecun_normal()(
        draw, (field_channels, config["pool_channels"]), jnp.float32
    )
    params["encoder.token_skip.alpha"] = jnp.float32(0)
    return params


def spatial_conv(params, name, values):
    return jax.lax.conv_general_dilated(
        values, params[name + ".weight"], (1, 1), "SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    ) + params[name + ".bias"]


def grouped_norm(params, name, values):
    channels = values.shape[-1]
    groups = next(g for g in range(min(8, channels), 0, -1) if channels % g == 0)
    grouped = values.reshape(*values.shape[:-1], groups, channels // groups)
    mean = jnp.mean(grouped, axis=(1, 2, 4), keepdims=True)
    variance = jnp.maximum(
        jnp.mean(grouped**2, axis=(1, 2, 4), keepdims=True) - mean**2, 0
    )
    normalized = ((grouped - mean) * jax.lax.rsqrt(variance + 1e-6)).reshape(values.shape)
    return normalized * params[name + ".weight"] + params[name + ".bias"]


def unet_features(params, inputs, xy, config):
    """Return full-resolution NHWC features; ``inputs`` are already normalized."""
    batch, height, width, _ = inputs.shape
    coords = jnp.broadcast_to(xy.reshape(1, height, width, 2), (batch, height, width, 2))
    values = jnp.concatenate((inputs, coords), axis=-1)
    channels = tuple(config.get("encoder_channels", (32, 64, 128)))

    def stage(name, values):
        values = spatial_conv(params, name + ".input", values)
        for i in range(config.get("encoder_blocks_per_level", 1)):
            prefix = f"{name}.blocks.{i}"
            hidden = jax.nn.gelu(grouped_norm(params, prefix + ".norm_0", values))
            hidden = spatial_conv(params, prefix + ".conv_0", hidden)
            hidden = jax.nn.gelu(grouped_norm(params, prefix + ".norm_1", hidden))
            values = values + spatial_conv(params, prefix + ".conv_1", hidden)
        return values

    skips = []
    for i in range(len(channels)):
        values = stage(f"encoder.down.{i}", values)
        skips.append(values)
        if i < len(channels) - 1:
            values = jax.image.resize(
                values, (batch, max(1, values.shape[1] // 2),
                         max(1, values.shape[2] // 2), values.shape[-1]), method="linear"
            )
    for i in reversed(range(len(channels) - 1)):
        values = jax.image.resize(
            values, (*skips[i].shape[:3], values.shape[-1]), method="linear"
        )
        values = stage(f"encoder.up.{i}", jnp.concatenate((values, skips[i]), axis=-1))
    values = jax.nn.gelu(grouped_norm(params, "encoder.output_norm", values))
    values = spatial_conv(params, "encoder.output", values)
    return values + params["encoder.token_skip.alpha"] * (
        inputs @ params["encoder.token_skip.weight"]
    )
