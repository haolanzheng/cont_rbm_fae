"""Array-based pure-JAX UNO, pooling, and decoder implementation.

Parameters are real float32 arrays. Spectral coefficients store real/imaginary
parts on their last axis so Adam treats the two components independently.
Names and layouts match the reference model for numerical migration checks.
"""

from __future__ import annotations

import math
from itertools import pairwise

import jax
import jax.numpy as jnp

from .encoder import init_unet, normalized_inputs, unet_features


def init_model(key, config):
    if config.get("encoder_kind", "spectral") == "darcy_unet":
        # Retain the historical RNG stream for the shared decoder/position/pool
        # parameters. Discarded spectral arrays are dead-code-eliminated by JIT.
        reference = {**config, **config["shared_initialization_reference"],
                     "encoder_kind": "spectral"}
        common = init_model(key, reference)
        params = {name: value for name, value in common.items()
                  if not name.startswith("encoder.")}
        params.update(init_unet(jax.random.fold_in(key, 1701), config))
        return params
    if config.get("encoder_kind", "spectral") != "spectral":
        raise ValueError("Unknown encoder_kind")
    params = {}

    def uniform(name, shape, bound):
        nonlocal key
        key, draw = jax.random.split(key)
        params[name] = jax.random.uniform(
            draw, shape, dtype=jnp.float32, minval=-bound, maxval=bound
        )

    def linear(name, inputs, outputs, bias=True, conv=False):
        shape = (outputs, inputs, 1, 1) if conv else (outputs, inputs)
        uniform(name + ".weight", shape, 1 / math.sqrt(inputs))
        if bias:
            uniform(name + ".bias", (outputs,), 1 / math.sqrt(inputs))

    def glorot_linear(name, inputs, outputs, bias=True):
        """Flax-style dense initialization used by the POD decoder."""
        nonlocal key
        key, draw = jax.random.split(key)
        params[name + ".weight"] = jax.random.normal(
            draw, (outputs, inputs), dtype=jnp.float32
        ) * math.sqrt(2 / (inputs + outputs))
        if bias:
            params[name + ".bias"] = jnp.zeros(outputs, jnp.float32)

    def siren_linear(name, inputs, outputs, omega, first=False, bias=True):
        """SIREN initialization before the explicit multiplication by omega."""
        bound = 1 / inputs if first else math.sqrt(6 / inputs) / omega
        uniform(name + ".weight", (outputs, inputs), bound)
        if bias:
            params[name + ".bias"] = jnp.zeros(outputs, jnp.float32)

    def norm(name, channels):
        params[name + ".weight"] = jnp.ones(channels, jnp.float32)
        params[name + ".bias"] = jnp.zeros(channels, jnp.float32)

    def operator(name, inputs, outputs):
        nonlocal key
        rank = min(32, inputs, outputs)
        norm(name + ".in_norm", inputs)
        linear(name + ".spectral_in", inputs, rank, False, True)
        for side in ("positive", "negative"):
            key, draw = jax.random.split(key)
            params[name + ".weight_" + side] = jax.random.normal(
                draw, (rank, config["modes"], config["modes"], 2), jnp.float32
            ) / math.sqrt(2 * rank)
        linear(name + ".spectral_out", rank, outputs, False, True)
        linear(name + ".local", inputs, outputs, conv=True)
        norm(name + ".out_norm", outputs)
        linear(name + ".mixing.0", outputs, 2 * outputs, conv=True)
        linear(name + ".mixing.2", 2 * outputs, outputs, conv=True)

    frequencies = config["coordinate_frequencies"]
    position_channels = 4 + 4 * frequencies
    for name in ("frequency_offsets", "phase_x", "phase_y"):
        params["position_encoding." + name] = jnp.zeros(frequencies, jnp.float32)
    linear("position_encoding.refinement.0", position_channels, 2 * position_channels)
    linear("position_encoding.refinement.2", 2 * position_channels, position_channels)
    for name in ("weight", "bias"):
        k = "position_encoding.refinement.2." + name
        params[k] = jnp.zeros_like(params[k])

    width = config["width"]
    channels = [width, round(1.5 * width), 3 * width, 6 * width]
    linear("encoder.lift", 1 + position_channels, width, conv=True)
    for i, (a, b) in enumerate(pairwise(channels)):
        operator(f"encoder.contract.{i}", a, b)
    operator("encoder.bottleneck", channels[-1], channels[-1])
    linear("encoder.project.0", channels[-1], 32, conv=True)
    linear("encoder.project.2", 32, config["pool_channels"], conv=True)

    latent, hidden = config["latent_dimension"], config["kernel_hidden"]
    pooling_kind = config.get("pooling_kind", "coordinate_kernel")
    if pooling_kind in ("coordinate_kernel", "orthogonal_integral"):
        linear("pool.kernel.0", position_channels, hidden)
        linear("pool.kernel.2", hidden, hidden)
        linear("pool.kernel.4", hidden, latent * config["pool_channels"])
    elif pooling_kind == "quadrature_attention":
        queries = config["attention_queries"]
        values = config["attention_value_dimension"]
        keys = config["attention_key_dimension"]
        if queries * values != latent:
            raise ValueError(
                "attention_queries * attention_value_dimension must equal "
                "latent_dimension"
            )
        joint = position_channels + config["pool_channels"]
        linear("pool.attention.key", joint, keys)
        uniform("pool.attention.value_scale", (queries, values), 1 / math.sqrt(values))
        params["pool.attention.value_scale"] = 1 + 0.01 * params[
            "pool.attention.value_scale"
        ]
        linear("pool.attention.value", joint, values)
        uniform("pool.attention.queries", (queries, keys), 1 / math.sqrt(keys))
    elif pooling_kind == "nonlinear_integral":
        joint = position_channels + config["pool_channels"]
        linear("pool.nonlinear.0", joint, hidden)
        linear("pool.nonlinear.2", hidden, hidden)
        linear("pool.nonlinear.4", hidden, latent)
    else:
        raise ValueError(f"Unknown pooling_kind: {pooling_kind}")
    params["pool.readout.weight"] = jnp.eye(latent, dtype=jnp.float32)
    decoder_kind = config.get("decoder_kind", "film")
    if decoder_kind == "film":
        decoder, condition = config["decoder_width"], config["condition_width"]
        norm("decoder.condition.0", latent)
        linear("decoder.condition.1", latent, condition)
        linear("decoder.condition.3", condition, condition)
        linear("decoder.coordinate_lift.0", position_channels, decoder)
        linear("decoder.coordinate_lift.2", decoder, decoder)
        for i in range(config["decoder_blocks"]):
            name = f"decoder.blocks.{i}"
            norm(name + ".norm", decoder)
            linear(name + ".linear_in", decoder, 2 * decoder)
            linear(name + ".film", condition, 4 * decoder)
            linear(name + ".linear_out", 2 * decoder, decoder)
        norm("decoder.output.0", decoder)
        linear("decoder.output.1", decoder, 1)
    elif decoder_kind in ("pod", "pod_film"):
        basis = config["n_basis"]

        def tower(name, inputs, hidden, blocks):
            glorot_linear(f"{name}.input", inputs, hidden)
            for i in range(blocks):
                prefix = f"{name}.blocks.{i}"
                glorot_linear(prefix + ".dense_0", hidden, hidden)
                glorot_linear(prefix + ".dense_1", hidden, hidden)

        tower(
            "decoder.basis",
            position_channels,
            config["basis_hidden_dim"],
            config["basis_n_layers"],
        )
        if decoder_kind == "pod_film":
            film_outputs = 2 * config["basis_hidden_dim"]
            for i in range(config["basis_n_layers"]):
                prefix = f"decoder.basis.blocks.{i}.film"
                params[prefix + ".weight"] = jnp.zeros(
                    (film_outputs, latent), jnp.float32
                )
                params[prefix + ".bias"] = jnp.zeros(film_outputs, jnp.float32)
        glorot_linear(
            "decoder.basis.output", config["basis_hidden_dim"], basis, bias=False
        )
        tower(
            "decoder.coefs",
            latent,
            config["coefs_hidden_dim"],
            config["coefs_n_layers"],
        )
        glorot_linear(
            "decoder.coefs.output", config["coefs_hidden_dim"], basis, bias=False
        )
    elif decoder_kind == "conditional_mlp":
        decoder = config["decoder_width"]
        glorot_linear("decoder.coordinate_lift", position_channels, decoder)
        glorot_linear("decoder.latent_lift", latent, decoder, bias=False)
        for i in range(config["decoder_blocks"]):
            prefix = f"decoder.blocks.{i}"
            glorot_linear(prefix + ".dense_0", decoder, decoder)
            glorot_linear(prefix + ".latent", latent, decoder, bias=False)
            glorot_linear(prefix + ".dense_1", decoder, decoder)
        glorot_linear("decoder.output", decoder, 1)
    elif decoder_kind == "modulated_siren":
        decoder = config["decoder_width"]
        omega_initial = config["siren_omega_initial"]
        omega_hidden = config["siren_omega_hidden"]
        siren_linear("decoder.siren.input", 2, decoder, omega_initial, first=True)
        glorot_linear("decoder.siren.input_shift", latent, decoder, bias=False)
        for i in range(config["decoder_blocks"]):
            prefix = f"decoder.siren.blocks.{i}"
            siren_linear(prefix + ".linear", decoder, decoder, omega_hidden)
            glorot_linear(prefix + ".film", latent, 2 * decoder)
        siren_linear("decoder.siren.output", decoder, 1, omega_hidden)
    else:
        raise ValueError(f"Unknown decoder_kind: {decoder_kind}")
    return params


def parameter_count(params):
    return sum(value.size for value in jax.tree_util.tree_leaves(params))


def gelu(x):
    return jax.nn.gelu(x, approximate=False)


def linear(params, name, x):
    value = x @ params[name + ".weight"].T
    if name + ".bias" in params:
        value = value + params[name + ".bias"]
    return value


def conv(params, name, x):
    value = jnp.einsum("bchw,oc->bohw", x, params[name + ".weight"][:, :, 0, 0])
    if name + ".bias" in params:
        value = value + params[name + ".bias"][None, :, None, None]
    return value


def layer_norm(params, name, x):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + 1e-5) * params[
        name + ".weight"
    ] + params[name + ".bias"]


def residual_tower(params, name, x, blocks):
    """Ordinary two-layer residual MLP without normalization."""
    values = gelu(linear(params, name + ".input", x))
    for i in range(blocks):
        prefix = f"{name}.blocks.{i}"
        hidden = gelu(linear(params, prefix + ".dense_0", values))
        residual = linear(params, prefix + ".dense_1", hidden)
        values = values + residual / math.sqrt(2)
    return values


def conditioned_residual_basis(params, position, code, blocks, film_scale):
    """Residual basis Psi(x; z) with raw-latent FiLM and no normalization."""
    values = gelu(linear(params, "decoder.basis.input", position))[None]
    values = jnp.broadcast_to(values, (len(code), *values.shape[1:]))
    for i in range(blocks):
        prefix = f"decoder.basis.blocks.{i}"
        hidden = gelu(linear(params, prefix + ".dense_0", values))
        scale, shift = jnp.split(linear(params, prefix + ".film", code), 2, -1)
        hidden = hidden * (1 + film_scale * scale[:, None])
        hidden = hidden + film_scale * shift[:, None]
        residual = linear(params, prefix + ".dense_1", hidden)
        values = values + residual / math.sqrt(2)
    return values


def conditional_residual_mlp(params, position, code, config):
    """Nonlinear coordinate field with latent injection in every residual block."""
    latent_scale = config.get("decoder_latent_scale", 1.0)
    values = linear(params, "decoder.coordinate_lift", position)[None]
    values = values + latent_scale * linear(
        params, "decoder.latent_lift", code
    )[:, None]
    values = gelu(values)
    for i in range(config["decoder_blocks"]):
        prefix = f"decoder.blocks.{i}"
        hidden = linear(params, prefix + ".dense_0", values)
        hidden = hidden + latent_scale * linear(
            params, prefix + ".latent", code
        )[:, None]
        residual = linear(params, prefix + ".dense_1", gelu(hidden))
        values = values + residual / math.sqrt(2)
    return linear(params, "decoder.output", values)[..., 0]


def modulated_siren(params, xy, code, config):
    """SIREN coordinate field with raw-latent phase and bounded scale modulation."""
    omega_initial = config["siren_omega_initial"]
    omega_hidden = config["siren_omega_hidden"]
    scale_limit = config.get("siren_scale_limit", 0.1)
    phase = omega_initial * linear(params, "decoder.siren.input", xy)[None]
    phase = phase + linear(params, "decoder.siren.input_shift", code)[:, None]
    values = jnp.sin(phase)
    for i in range(config["decoder_blocks"]):
        prefix = f"decoder.siren.blocks.{i}"
        phase = omega_hidden * linear(params, prefix + ".linear", values)
        scale, shift = jnp.split(linear(params, prefix + ".film", code), 2, -1)
        phase = phase * (1 + scale_limit * jnp.tanh(scale)[:, None])
        phase = phase + shift[:, None]
        values = jnp.sin(phase)
    return linear(params, "decoder.siren.output", values)[..., 0]


def group_norm(params, name, x):
    mean = jnp.mean(x, axis=(1, 2, 3), keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=(1, 2, 3), keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + 1e-5) * params[name + ".weight"][
        None, :, None, None
    ] + params[name + ".bias"][None, :, None, None]


def coordinates(shape):
    yy, xx = jnp.meshgrid(
        jnp.linspace(-1, 1, shape[0], dtype=jnp.float32),
        jnp.linspace(-1, 1, shape[1], dtype=jnp.float32),
        indexing="ij",
    )
    return jnp.stack((xx, yy), -1).reshape(-1, 2)


def position_encoding(params, xy):
    xx, yy = xy[..., 0], xy[..., 1]
    base = jnp.stack((xx, yy, 1 - xx**2, 1 - yy**2), -1)
    prefix = "position_encoding."
    offsets = params[prefix + "frequency_offsets"]
    freq = jnp.pi * 2.0 ** jnp.arange(offsets.size, dtype=jnp.float32)
    freq = freq * jnp.exp(0.25 * jnp.tanh(offsets))
    px = xx[..., None] * freq + params[prefix + "phase_x"]
    py = yy[..., None] * freq + params[prefix + "phase_y"]
    features = jnp.concatenate(
        (base, jnp.sin(px), jnp.cos(px), jnp.sin(py), jnp.cos(py)), -1
    )
    residual = linear(
        params,
        prefix + "refinement.2",
        gelu(linear(params, prefix + "refinement.0", features)),
    )
    return features + 0.1 * residual


def trapezoid_weights(height, width):
    if min(height, width) < 2:
        raise ValueError("Quadrature requires at least two nodes per axis")
    wy = jnp.ones(height, jnp.float32).at[jnp.array([0, height - 1])].set(0.5)
    wx = jnp.ones(width, jnp.float32).at[jnp.array([0, width - 1])].set(0.5)
    return (wy[:, None] * wx[None, :] / ((height - 1) * (width - 1))).reshape(-1)


def orthogonalize_kernel(kernel, weights, epsilon):
    """Whiten latent kernel functions in the weighted spatial-channel inner product."""
    gram = jnp.einsum("nlc,nmc,n->lm", kernel, kernel, weights)
    gram = gram + jnp.float32(epsilon) * jnp.eye(gram.shape[0], dtype=gram.dtype)
    factor = jnp.linalg.cholesky(gram)
    # If G = L L^T, K L^{-T} has weighted Gram equal to the identity.
    return jax.scipy.linalg.solve_triangular(
        factor, kernel.transpose(1, 0, 2).reshape(kernel.shape[1], -1), lower=True
    ).reshape(kernel.shape[1], kernel.shape[0], kernel.shape[2]).transpose(1, 0, 2)


def pool_features(params, features, position, shape, config):
    """Map a spatial feature field to exactly ``latent_dimension`` scalars."""
    height, width = shape
    weights = trapezoid_weights(height, width)
    pooling_kind = config.get("pooling_kind", "coordinate_kernel")
    if pooling_kind in ("coordinate_kernel", "orthogonal_integral"):
        kernel = gelu(linear(params, "pool.kernel.0", position))
        kernel = gelu(linear(params, "pool.kernel.2", kernel))
        kernel = linear(params, "pool.kernel.4", kernel).reshape(
            height * width, config["latent_dimension"], config["pool_channels"]
        )
        if pooling_kind == "orthogonal_integral":
            kernel = orthogonalize_kernel(
                kernel, weights, config.get("orthogonal_epsilon", 1e-4)
            )
        code = jnp.einsum("bnc,nlc,n->bl", features, kernel, weights)
    elif pooling_kind == "quadrature_attention":
        batch = features.shape[0]
        positions = jnp.broadcast_to(position[None], (batch, *position.shape))
        joint = jnp.concatenate((positions, features), axis=-1)
        keys = gelu(linear(params, "pool.attention.key", joint))
        values = linear(params, "pool.attention.value", joint)
        queries = params["pool.attention.queries"]
        logits = jnp.einsum("qk,bnk->bqn", queries, keys) / math.sqrt(keys.shape[-1])
        # Log quadrature weights make this a discretization of continuum attention,
        # rather than an unweighted average over the node sampling density.
        attention = jax.nn.softmax(logits + jnp.log(weights)[None, None], axis=-1)
        code = jnp.einsum("bqn,bnv->bqv", attention, values)
        code = (code * params["pool.attention.value_scale"][None]).reshape(batch, -1)
    elif pooling_kind == "nonlinear_integral":
        batch = features.shape[0]
        positions = jnp.broadcast_to(position[None], (batch, *position.shape))
        joint = jnp.concatenate((positions, features), axis=-1)
        moments = gelu(linear(params, "pool.nonlinear.0", joint))
        moments = gelu(linear(params, "pool.nonlinear.2", moments))
        moments = linear(params, "pool.nonlinear.4", moments)
        code = jnp.einsum("bnl,n->bl", moments, weights)
    else:
        raise ValueError(f"Unknown pooling_kind: {pooling_kind}")
    return linear(params, "pool.readout", code)


def bilinear(values, shape, extent=(1.0, 1.0)):
    """Endpoint-aligned interpolation; extent crops the padded physical domain."""
    ih, iw = values.shape[-2:]
    y = jnp.linspace(0, (ih - 1) * extent[0], shape[0], dtype=jnp.float32)
    x = jnp.linspace(0, (iw - 1) * extent[1], shape[1], dtype=jnp.float32)
    y0, x0 = jnp.floor(y).astype(jnp.int32), jnp.floor(x).astype(jnp.int32)
    y1, x1 = jnp.minimum(y0 + 1, ih - 1), jnp.minimum(x0 + 1, iw - 1)
    fy, fx = y - y0, x - x0
    rows = (
        values[:, :, y0, :] * (1 - fy)[None, None, :, None]
        + values[:, :, y1, :] * fy[None, None, :, None]
    )
    return (
        rows[:, :, :, x0] * (1 - fx)[None, None, None, :]
        + rows[:, :, :, x1] * fx[None, None, None, :]
    )


def fourier_transfer(values, shape, positive, negative):
    ih, iw = values.shape[-2:]
    oh, ow = shape
    my = min(positive.shape[-2], (ih - 1) // 2, (oh - 1) // 2)
    mx = min(positive.shape[-1], (iw - 1) // 2 + 1, (ow - 1) // 2 + 1)
    if min(my, mx) < 1:
        raise ValueError("Spatial grid is too small for spectral transfer")
    ft = jnp.fft.rfft2(values, norm="forward")
    output = jnp.zeros((*values.shape[:-2], oh, ow // 2 + 1), ft.dtype)
    output = output.at[..., :my, :mx].set(
        ft[..., :my, :mx] * positive[None, :, :my, :mx]
    )
    output = output.at[..., -my:, :mx].set(
        ft[..., -my:, :mx] * negative[None, :, :my, :mx]
    )
    return jnp.fft.irfft2(output, s=shape, norm="forward")


def operator(params, name, values, shape):
    normalized = group_norm(params, name + ".in_norm", values)
    pos, neg = params[name + ".weight_positive"], params[name + ".weight_negative"]
    pos, neg = (
        jax.lax.complex(pos[..., 0], pos[..., 1]),
        jax.lax.complex(neg[..., 0], neg[..., 1]),
    )
    spectral = conv(
        params,
        name + ".spectral_out",
        fourier_transfer(
            conv(params, name + ".spectral_in", normalized),
            shape,
            pos,
            neg,
        ),
    )
    local = conv(params, name + ".local", bilinear(normalized, shape))
    output = gelu(spectral + local)
    mixed = group_norm(params, name + ".out_norm", output)
    mixed = conv(
        params, name + ".mixing.2", gelu(conv(params, name + ".mixing.0", mixed))
    )
    return output + mixed


def encode(params, field, config, observation_mask=None):
    if field.ndim == 4:
        if field.shape[1] != 1:
            raise ValueError("Expected one input field channel")
        field = field[:, 0]
    batch, height, width = field.shape
    position = position_encoding(params, coordinates((height, width)))
    if config.get("encoder_kind", "spectral") == "darcy_unet":
        inputs = normalized_inputs(field, config, observation_mask)
        features = unet_features(params, inputs, coordinates((height, width)), config)
        return pool_features(
            params, features.reshape(batch, height * width, -1), position,
            (height, width), config,
        )
    extra = jnp.broadcast_to(
        position.T.reshape(1, -1, height, width),
        (batch, position.shape[-1], height, width),
    )
    features = conv(params, "encoder.lift", jnp.concatenate((field[:, None], extra), 1))
    py, px = max(1, round(height * 0.1)), max(1, round(width * 0.1))
    features = jnp.pad(features, ((0, 0), (0, 0), (0, py), (0, px)))
    for i, factor in enumerate((0.75, 2 / 3, 0.5)):
        shape = tuple(max(5, round(n * factor)) for n in features.shape[-2:])
        features = operator(params, f"encoder.contract.{i}", features, shape)
    features = operator(params, "encoder.bottleneck", features, features.shape[-2:])
    features = conv(
        params, "encoder.project.2", gelu(conv(params, "encoder.project.0", features))
    )
    features = bilinear(
        features,
        (height, width),
        ((height - 1) / (height + py - 1), (width - 1) / (width + px - 1)),
    )
    features = features.transpose(0, 2, 3, 1).reshape(batch, height * width, -1)
    return pool_features(params, features, position, (height, width), config)


def decode_coordinates(params, code, xy, config):
    position = position_encoding(params, xy)
    decoder_kind = config.get("decoder_kind", "film")
    if decoder_kind in ("pod", "pod_film"):
        if decoder_kind == "pod":
            basis = residual_tower(
                params, "decoder.basis", position, config["basis_n_layers"]
            )
            basis = linear(params, "decoder.basis.output", basis)
        else:
            basis = conditioned_residual_basis(
                params,
                position,
                code,
                config["basis_n_layers"],
                config["basis_film_scale"],
            )
            basis = linear(params, "decoder.basis.output", basis)
        coefficients = residual_tower(
            params, "decoder.coefs", code, config["coefs_n_layers"]
        )
        coefficients = linear(params, "decoder.coefs.output", coefficients)
        if decoder_kind == "pod":
            return jnp.einsum("nk,bk->bn", basis, coefficients)
        return jnp.einsum("bnk,bk->bn", basis, coefficients)
    if decoder_kind == "conditional_mlp":
        return conditional_residual_mlp(params, position, code, config)
    if decoder_kind == "modulated_siren":
        return modulated_siren(params, xy, code, config)
    if position.ndim == 2:
        position = jnp.broadcast_to(position[None], (len(code), *position.shape))
    condition = layer_norm(params, "decoder.condition.0", code)
    condition = gelu(linear(params, "decoder.condition.1", condition))
    condition = gelu(linear(params, "decoder.condition.3", condition))
    values = linear(
        params,
        "decoder.coordinate_lift.2",
        gelu(linear(params, "decoder.coordinate_lift.0", position)),
    )
    for i in range(config["decoder_blocks"]):
        name = f"decoder.blocks.{i}"
        hidden = linear(
            params, name + ".linear_in", layer_norm(params, name + ".norm", values)
        )
        scale, shift = jnp.split(linear(params, name + ".film", condition), 2, -1)
        hidden = hidden * (1 + 0.1 * jnp.tanh(scale[:, None])) + shift[:, None]
        values = values + linear(
            params, name + ".linear_out", gelu(hidden)
        ) / math.sqrt(2)
    return linear(
        params, "decoder.output.1", layer_norm(params, "decoder.output.0", values)
    )[..., 0]


def apply(params, field, config, output_shape=None, observation_mask=None):
    code = encode(params, field, config, observation_mask)
    shape = output_shape or field.shape[-2:]
    prediction = decode_coordinates(params, code, coordinates(shape), config)
    return prediction.reshape(len(field), *shape), code
