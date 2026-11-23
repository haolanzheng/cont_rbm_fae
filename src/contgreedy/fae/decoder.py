"""Coordinate-conditioned decoders for controlled FAE ablations."""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn

from .core import Decoder


class CoordinateEncoding(nn.Module):
    """Fixed Fourier features shared by every decoder in the ablation."""

    frequencies: int = 6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        xx, yy = x[..., 0], x[..., 1]
        base = jnp.stack((xx, yy, 1 - xx**2, 1 - yy**2), -1)
        f = jnp.pi * 2.0 ** jnp.arange(self.frequencies)
        return jnp.concatenate(
            (
                base,
                jnp.sin(xx[..., None] * f),
                jnp.cos(xx[..., None] * f),
                jnp.sin(yy[..., None] * f),
                jnp.cos(yy[..., None] * f),
            ),
            -1,
        )


class ResidualMLPBlock(nn.Module):
    """Dense--GELU--Dense residual block without normalization."""

    width: int
    hidden_dim: int = 128
    act: Callable = nn.gelu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.glorot_normal())(x)
        h = self.act(h)
        h = nn.Dense(self.width, kernel_init=nn.initializers.glorot_normal())(h)
        return x + h


class ResidualMLP(nn.Module):
    """Residual MLP trunk with a fixed feature width."""

    width: int
    n_layers: int = 2
    hidden_dim: int = 128
    act: Callable = nn.gelu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.shape[-1] != self.width:
            x = nn.Dense(self.width, kernel_init=nn.initializers.glorot_normal())(x)
        for _ in range(int(self.n_layers)):
            x = ResidualMLPBlock(
                width=self.width,
                hidden_dim=self.hidden_dim,
                act=self.act,
            )(x)
        return x


def _broadcast_latent(z: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    return jnp.broadcast_to(z[:, None, :], (*x.shape[:2], z.shape[-1]))


class ConcatResidualDecoder(Decoder):
    """Baseline: concatenate the latent vector to every coordinate token."""

    width: int = 256
    blocks: int = 6
    frequencies: int = 6

    def _forward(self, z, x, train=False):
        position = CoordinateEncoding(self.frequencies)(x)
        values = jnp.concatenate((position, _broadcast_latent(z, x)), -1)
        values = ResidualMLP(
            width=self.width,
            hidden_dim=self.width,
            n_layers=self.blocks,
        )(values)
        return nn.Dense(1, kernel_init=nn.initializers.glorot_normal())(values)


class FiLMResidualBlock(nn.Module):
    """Residual block modulated by a sample-level latent vector."""

    width: int
    scale_and_shift: bool

    @nn.compact
    def __call__(self, values: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        hidden = nn.Dense(self.width, kernel_init=nn.initializers.glorot_normal())(values)
        outputs = 2 * self.width if self.scale_and_shift else self.width
        modulation = nn.Dense(outputs, kernel_init=nn.initializers.zeros)(z)
        if self.scale_and_shift:
            scale, shift = jnp.split(modulation, 2, axis=-1)
            hidden = hidden * (1.0 + scale[:, None, :]) + shift[:, None, :]
        else:
            hidden = hidden + modulation[:, None, :]
        hidden = nn.gelu(hidden)
        hidden = nn.Dense(self.width, kernel_init=nn.initializers.glorot_normal())(hidden)
        return values + hidden


class FiLMResidualDecoder(Decoder):
    """Residual coordinate MLP with shift-only or affine FiLM at each block."""

    width: int = 256
    blocks: int = 6
    frequencies: int = 6
    scale_and_shift: bool = True

    def _forward(self, z, x, train=False):
        position = CoordinateEncoding(self.frequencies)(x)
        values = nn.Dense(self.width, kernel_init=nn.initializers.glorot_normal())(
            jnp.concatenate((position, _broadcast_latent(z, x)), -1)
        )
        for _ in range(int(self.blocks)):
            values = FiLMResidualBlock(
                width=self.width,
                scale_and_shift=self.scale_and_shift,
            )(values, z)
        return nn.Dense(1, kernel_init=nn.initializers.glorot_normal())(values)


class LowRankModulatedBlock(nn.Module):
    """Residual block with a rank-r latent-dependent weight correction."""

    width: int
    rank: int = 16

    @nn.compact
    def __call__(self, values: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        base = nn.Dense(self.width, kernel_init=nn.initializers.glorot_normal())(values)
        left = nn.Dense(
            self.rank,
            use_bias=False,
            kernel_init=nn.initializers.glorot_normal(),
        )(values)
        coefficients = nn.Dense(
            self.rank,
            kernel_init=nn.initializers.zeros,
        )(z)
        correction = nn.Dense(
            self.width,
            use_bias=False,
            kernel_init=nn.initializers.glorot_normal(),
        )(left * coefficients[:, None, :])
        hidden = nn.gelu(base + correction)
        hidden = nn.Dense(self.width, kernel_init=nn.initializers.glorot_normal())(hidden)
        return values + hidden


class LowRankModulatedDecoder(Decoder):
    """Coordinate MLP with low-rank latent-dependent block weights."""

    width: int = 256
    blocks: int = 6
    frequencies: int = 6
    rank: int = 16

    def _forward(self, z, x, train=False):
        position = CoordinateEncoding(self.frequencies)(x)
        values = nn.Dense(self.width, kernel_init=nn.initializers.glorot_normal())(
            jnp.concatenate((position, _broadcast_latent(z, x)), -1)
        )
        for _ in range(int(self.blocks)):
            values = LowRankModulatedBlock(self.width, self.rank)(values, z)
        return nn.Dense(1, kernel_init=nn.initializers.glorot_normal())(values)


class LatentCrossAttentionDecoder(Decoder):
    """Coordinate queries attending to a fixed-budget set of latent tokens."""

    width: int = 256
    blocks: int = 6
    frequencies: int = 6
    latent_tokens: int = 8
    attention_width: int = 128
    attention_heads: int = 4

    def _forward(self, z, x, train=False):
        if z.shape[-1] % self.latent_tokens:
            raise ValueError("latent_dimension must be divisible by latent_tokens")
        token_width = z.shape[-1] // self.latent_tokens
        raw_tokens = z.reshape(z.shape[0], self.latent_tokens, token_width)
        tokens = nn.Dense(
            self.attention_width,
            kernel_init=nn.initializers.glorot_normal(),
        )(raw_tokens)
        token_embedding = self.param(
            "token_embedding",
            nn.initializers.normal(0.02),
            (1, self.latent_tokens, self.attention_width),
        )
        tokens = tokens + token_embedding
        position = CoordinateEncoding(self.frequencies)(x)
        queries = nn.Dense(
            self.attention_width,
            kernel_init=nn.initializers.glorot_normal(),
        )(position)
        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.attention_heads,
            qkv_features=self.attention_width,
            out_features=self.width,
            dropout_rate=0.0,
        )(queries, tokens, deterministic=True)
        values = attended + nn.Dense(
            self.width,
            kernel_init=nn.initializers.glorot_normal(),
        )(position)
        values = ResidualMLP(
            width=self.width,
            hidden_dim=self.width,
            n_layers=self.blocks,
        )(values)
        return nn.Dense(1, kernel_init=nn.initializers.glorot_normal())(values)


def bilinear_sample(features: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Sample a [B,H,W,C] feature field at coordinates in [-1,1]^2."""

    batch, height, width, channels = features.shape
    gx = jnp.clip((x[..., 0] + 1.0) * (width - 1) / 2.0, 0.0, width - 1)
    gy = jnp.clip((x[..., 1] + 1.0) * (height - 1) / 2.0, 0.0, height - 1)
    x0, y0 = jnp.floor(gx).astype(jnp.int32), jnp.floor(gy).astype(jnp.int32)
    x1, y1 = jnp.minimum(x0 + 1, width - 1), jnp.minimum(y0 + 1, height - 1)
    flat = features.reshape(batch, height * width, channels)

    def gather(ix, iy):
        return jnp.take_along_axis(flat, (iy * width + ix)[..., None], axis=1)

    f00, f10 = gather(x0, y0), gather(x1, y0)
    f01, f11 = gather(x0, y1), gather(x1, y1)
    wx, wy = (gx - x0)[..., None], (gy - y0)[..., None]
    return (
        (1.0 - wx) * (1.0 - wy) * f00
        + wx * (1.0 - wy) * f10
        + (1.0 - wx) * wy * f01
        + wx * wy * f11
    )


class MultiScaleLocalDecoder(Decoder):
    """Decode a 4x4 latent feature grid through continuous multiscale queries."""

    width: int = 256
    blocks: int = 6
    frequencies: int = 6
    latent_grid_size: int = 4
    grid_channels: int = 32

    def _forward(self, z, x, train=False):
        cells = self.latent_grid_size**2
        if z.shape[-1] % cells:
            raise ValueError("latent_dimension must be divisible by latent_grid_size**2")
        channels = z.shape[-1] // cells
        grid = z.reshape(z.shape[0], self.latent_grid_size, self.latent_grid_size, channels)
        grid4 = nn.gelu(nn.Conv(self.grid_channels, (3, 3), padding="SAME")(grid))
        grid8 = jax.image.resize(
            grid4,
            (z.shape[0], 2 * self.latent_grid_size, 2 * self.latent_grid_size,
             self.grid_channels),
            method="linear",
        )
        grid8 = nn.gelu(nn.Conv(self.grid_channels, (3, 3), padding="SAME")(grid8))
        grid16 = jax.image.resize(
            grid8,
            (z.shape[0], 4 * self.latent_grid_size, 4 * self.latent_grid_size,
             self.grid_channels),
            method="linear",
        )
        grid16 = nn.gelu(nn.Conv(self.grid_channels, (3, 3), padding="SAME")(grid16))
        position = CoordinateEncoding(self.frequencies)(x)
        values = jnp.concatenate(
            (
                position,
                bilinear_sample(grid4, x),
                bilinear_sample(grid8, x),
                bilinear_sample(grid16, x),
            ),
            axis=-1,
        )
        values = ResidualMLP(
            width=self.width,
            hidden_dim=self.width,
            n_layers=self.blocks,
        )(values)
        return nn.Dense(1, kernel_init=nn.initializers.glorot_normal())(values)
