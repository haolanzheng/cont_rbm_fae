from __future__ import annotations

from typing import Callable, Sequence

import jax.image
import jax.numpy as jnp
from flax import linen as nn

from .core import Encoder
from .core import IdentityEncoding, PositionalEncoding


def _infer_n_from_q(q: int, d: int) -> int:
    n = int(round(q ** (1.0 / d)))
    if n ** d != q:
        raise ValueError(f"Q={q} is not a perfect {d}-th power.")
    return n


def _num_groups(channels: int, max_groups: int = 8) -> int:
    channels = int(channels)
    for groups in range(min(int(max_groups), channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class UNOResidualBlock(nn.Module):
    channels: int
    act: Callable = nn.gelu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.GroupNorm(num_groups=_num_groups(int(x.shape[-1])))(x)
        h = self.act(h)
        h = nn.Conv(
            int(self.channels),
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=nn.initializers.glorot_normal(),
        )(h)
        h = nn.GroupNorm(num_groups=_num_groups(int(self.channels)))(h)
        h = self.act(h)
        h = nn.Conv(
            int(self.channels),
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=nn.initializers.glorot_normal(),
        )(h)

        if x.shape[-1] != int(self.channels):
            x = nn.Conv(
                int(self.channels),
                kernel_size=(1, 1),
                padding="SAME",
                kernel_init=nn.initializers.glorot_normal(),
            )(x)
        return x + h


class UNOStage(nn.Module):
    channels: int
    n_blocks: int = 1
    act: Callable = nn.gelu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(
            int(self.channels),
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=nn.initializers.glorot_normal(),
        )(x)
        for _ in range(int(self.n_blocks)):
            x = UNOResidualBlock(
                channels=int(self.channels),
                act=self.act,
            )(x)
        return x


class UNO(nn.Module):
    channels: Sequence[int] = (32, 64, 128)
    n_blocks: int = 1
    out_channels: int = 16
    act: Callable = nn.gelu

    @nn.compact
    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        if u.ndim != 4:
            raise ValueError(f"UNO expects [B,H,W,C], got {u.shape}")

        channels = tuple(int(c) for c in self.channels)
        if len(channels) == 0:
            raise ValueError("UNO requires at least one channel level.")

        h = u
        skips = []
        sizes = []
        for i, ch in enumerate(channels):
            h = UNOStage(
                channels=ch,
                n_blocks=int(self.n_blocks),
                act=self.act,
                name=f"down_{i}",
            )(h)
            skips.append(h)
            sizes.append((int(h.shape[1]), int(h.shape[2])))
            if i < len(channels) - 1:
                h = jax.image.resize(
                    h,
                    shape=(
                        h.shape[0],
                        max(int(h.shape[1]) // 2, 1),
                        max(int(h.shape[2]) // 2, 1),
                        h.shape[-1],
                    ),
                    method="linear",
                )

        for i, ch in reversed(list(enumerate(channels[:-1]))):
            h = jax.image.resize(
                h,
                shape=(h.shape[0], sizes[i][0], sizes[i][1], h.shape[-1]),
                method="linear",
            )
            h = jnp.concatenate([h, skips[i]], axis=-1)
            h = UNOStage(
                channels=ch,
                n_blocks=int(self.n_blocks),
                act=self.act,
                name=f"up_{i}",
            )(h)

        h = nn.GroupNorm(num_groups=_num_groups(int(h.shape[-1])))(h)
        h = self.act(h)
        return nn.Conv(
            int(self.out_channels),
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=nn.initializers.glorot_normal(),
        )(h)


class UNOEncoder(Encoder):
    latent_dim: int

    positional_encoding: PositionalEncoding = IdentityEncoding()
    pooling_fn: nn.Module = None  # caller supplies the experiment pooling
    pooling_concat_x: bool = True

    uno_channels: Sequence[int] = (32, 64, 128)
    uno_blocks_per_level: int = 1
    uno_projection_dim: int = 16
    use_token_skip: bool = True
    token_skip_init_scale: float = 0.0
    token_skip_use_x: bool = False

    @nn.compact
    def __call__(self, u: jnp.ndarray, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        if u.ndim != 3 or x.ndim != 3:
            raise ValueError(f"Expected u,x as [B,Q,*]. Got u={u.shape}, x={x.shape}")

        b, q, c_in = u.shape
        d = int(x.shape[-1])
        if d != 2:
            raise ValueError(f"UNOEncoder currently expects 2D coordinates, got d={d}.")

        x_pos = self.positional_encoding(x)

        n = _infer_n_from_q(int(q), d)
        x_grid = x.reshape(b, n, n, d)
        u_grid = u.reshape(b, n, n, c_in)
        u_trunk_in = jnp.concatenate([u_grid, x_grid], axis=-1) if self.pooling_concat_x else u_grid

        y_grid = UNO(
            channels=tuple(int(c) for c in self.uno_channels),
            n_blocks=int(self.uno_blocks_per_level),
            out_channels=int(self.uno_projection_dim),
            act=nn.gelu,
        )(u_trunk_in)

        y_flat = y_grid.reshape(b, q, int(y_grid.shape[-1]))

        if self.use_token_skip:
            skip_src = jnp.concatenate([u, x_pos], axis=-1) if self.token_skip_use_x else u
            c_out = int(y_flat.shape[-1])
            skip = nn.Dense(c_out, use_bias=False, name="token_skip_proj")(skip_src)
            alpha = self.param(
                "token_skip_alpha",
                lambda key: jnp.array(self.token_skip_init_scale, dtype=y_flat.dtype),
            )
            y_flat = y_flat + alpha * skip

        pooled = self.pooling_fn(y_flat, x_pos)
        out_dim = int(self.latent_dim) * 2 if self.is_variational else int(self.latent_dim)
        return nn.Dense(out_dim, use_bias=False)(pooled)
