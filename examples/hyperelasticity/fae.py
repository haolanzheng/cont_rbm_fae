"""Hyperelastic input FAE assembly, paired GRF data, quadrature, and metrics."""
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

FAE_SOURCE = Path(__file__).resolve().parents[2] / "src" / "contgreedy" / "fae"

from contgreedy.fae.core import Autoencoder, Domain
from contgreedy.fae.decoder import (
    ConcatResidualDecoder,
    CoordinateEncoding,
    FiLMResidualDecoder,
    LatentCrossAttentionDecoder,
    LowRankModulatedDecoder,
    MultiScaleLocalDecoder,
)
from contgreedy.fae.uno import UNOEncoder


def grid(n):
    yy, xx = np.meshgrid(np.linspace(-1, 1, n, dtype=np.float32),
                         np.linspace(-1, 1, n, dtype=np.float32), indexing="ij")
    return np.stack((xx, yy), -1).reshape(-1, 2)


def weights(q):
    n = math.isqrt(q)
    if n * n != q or n < 2:
        raise ValueError("Expected a square endpoint-inclusive grid")
    w = jnp.ones(n).at[0].set(0.5).at[-1].set(0.5)
    return (w[:, None] * w[None, :] / (n - 1)**2).reshape(q)


def resample_fields(fields, n):
    """Endpoint-aligned bilinear interpolation of the SAME source functions."""
    old = math.isqrt(fields.shape[1])
    if old * old != fields.shape[1] or n < 2:
        raise ValueError("Invalid square grid")
    t = np.linspace(0, old - 1, n)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, old - 1)
    frac = (t - lo).astype(np.float32)
    output = np.empty((len(fields), n * n, 1), np.float32)
    for start in range(0, len(fields), 32):
        u = np.asarray(fields[start:start + 32], np.float32).reshape(-1, old, old)
        rows = u[:, lo, :] * (1 - frac)[None, :, None] + u[:, hi, :] * frac[None, :, None]
        v = rows[:, :, lo] * (1 - frac)[None, None, :] + rows[:, :, hi] * frac[None, None, :]
        output[start:start + len(v)] = v.reshape(len(v), -1, 1)
    return output


class FunctionalLoader:
    """Functional (u_dec, x_dec, u_enc, x_enc) batches without torch.

    Paired mode presents clean and noisy copies of each logical batch in
    microbatches. Noise and sample order change each epoch.
    """
    def __init__(self, observed, target, ids, *, batch_size, microbatch_size=None,
                 noise_std=0.0, seed=0, paired=False):
        self.observed, self.target = observed, target
        self.ids = np.asarray(ids)
        self.batch_size = batch_size
        self.micro = microbatch_size or batch_size
        self.noise_std, self.seed, self.paired = noise_std, seed, paired
        self.epoch = 0
        self.x_enc = grid(math.isqrt(observed.shape[1]))
        self.x_dec = grid(math.isqrt(target.shape[1]))
        if paired and (len(ids) % batch_size or batch_size % self.micro):
            raise ValueError("Training batches and accumulation must be complete")

    def __len__(self):
        if self.paired:
            return len(self.ids) // self.micro * 2
        return math.ceil(len(self.ids) / self.micro)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        ids = rng.permutation(self.ids) if self.paired else self.ids
        if self.paired:
            self.epoch += 1
        for start in range(0, len(ids), self.batch_size):
            selected = ids[start:start + self.batch_size]
            clean = self.observed[selected]
            observations = [clean]
            if self.paired:
                observations.append(clean + self.noise_std * rng.standard_normal(
                    clean.shape, dtype=np.float32))
            target = self.target[selected]
            for observed in observations:
                for i in range(0, len(selected), self.micro):
                    u_enc, u_dec = observed[i:i + self.micro], target[i:i + self.micro]
                    b = len(u_enc)
                    yield tuple(jnp.asarray(v) for v in (
                        u_dec, np.broadcast_to(self.x_dec, (b, *self.x_dec.shape)),
                        u_enc, np.broadcast_to(self.x_enc, (b, *self.x_enc.shape)),
                    ))


class TrapezoidL2(Domain):
    def __init__(self):
        super().__init__("unit_square_trapezoid_L2")

    def squared_norm(self, u, x):
        return jnp.sum(u**2 * weights(u.shape[1])[None, :, None], axis=(1, 2))

    def inner_product(self, u, v, x):
        return jnp.sum(u * v * weights(u.shape[1])[None, :, None], axis=(1, 2))


class OrthogonalIntegral(nn.Module):
    latent_dim: int = 128
    hidden: int = 64
    epsilon: float = 1e-3

    @nn.compact
    def __call__(self, u, x):
        # All functions use one common grid; do not replicate its kernel by B.
        h = nn.gelu(nn.Dense(self.hidden)(x[0]))
        h = nn.gelu(nn.Dense(self.hidden)(h))
        kernel = nn.Dense(self.latent_dim * u.shape[-1])(h)
        kernel = kernel.reshape(u.shape[1], self.latent_dim, u.shape[-1])
        w = weights(u.shape[1])
        gram = jnp.einsum("nlc,nmc,n->lm", kernel, kernel, w)
        factor = jnp.linalg.cholesky(gram + self.epsilon * jnp.eye(self.latent_dim))
        whitened = jax.scipy.linalg.solve_triangular(
            factor, kernel.transpose(1, 0, 2).reshape(self.latent_dim, -1), lower=True
        ).reshape(self.latent_dim, u.shape[1], u.shape[-1]).transpose(1, 0, 2)
        return jnp.einsum("bnc,nlc,n->bl", u, whitened, w)


def build_autoencoder(config):
    m = config["model"]
    encoder = UNOEncoder(
        is_variational=False, latent_dim=m["latent_dimension"],
        uno_channels=tuple(m["encoder_channels"]), uno_blocks_per_level=1,
        uno_projection_dim=m["pool_channels"],
        positional_encoding=CoordinateEncoding(m["coordinate_frequencies"]),
        pooling_fn=OrthogonalIntegral(m["latent_dimension"], m["kernel_hidden"],
                                     m["orthogonal_epsilon"]),
    )
    common = dict(
        width=m["decoder_width"],
        blocks=m["decoder_blocks"],
        frequencies=m["coordinate_frequencies"],
    )
    kind = m.get("decoder_kind", "concat")
    if kind == "concat":
        decoder = ConcatResidualDecoder(**common)
    elif kind == "film_shift":
        decoder = FiLMResidualDecoder(**common, scale_and_shift=False)
    elif kind == "film_scale_shift":
        decoder = FiLMResidualDecoder(**common, scale_and_shift=True)
    elif kind == "low_rank":
        decoder = LowRankModulatedDecoder(**common, rank=m.get("modulation_rank", 16))
    elif kind == "cross_attention":
        decoder = LatentCrossAttentionDecoder(
            **common,
            latent_tokens=m.get("latent_tokens", 8),
            attention_width=m.get("attention_width", 128),
            attention_heads=m.get("attention_heads", 4),
        )
    elif kind == "multiscale_local":
        decoder = MultiScaleLocalDecoder(
            **common,
            latent_grid_size=m.get("latent_grid_size", 4),
            grid_channels=m.get("grid_channels", 32),
        )
    else:
        raise ValueError(f"Unknown decoder_kind: {kind}")
    return Autoencoder(encoder=encoder, decoder=decoder)


def statistics(values):
    a = np.asarray(values)
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.quantile(a, 0.95)), max=float(a.max()))


class RelativeL2Metric:
    name = "relative_L2"

    def __init__(self, autoencoder, mean, std):
        self.last = None

        @jax.jit
        def batch_error(params, batch_stats, batch):
            u_dec, x_dec, u_enc, x_enc = batch
            predicted = autoencoder.apply(dict(params=params, batch_stats=batch_stats or {}),
                                          u_enc, x_enc, x_dec)
            error = (predicted - u_dec) * std
            raw = u_dec * std + mean
            w = weights(u_dec.shape[1])[None, :, None]
            norm2 = jnp.sum(raw**2 * w, axis=(1, 2))
            errors = jnp.sqrt(jnp.sum(error**2 * w, axis=(1, 2)) / jnp.maximum(norm2, 1e-12))
            return errors
        self.batch_error = batch_error

    def __call__(self, state, key, loader):
        errors = []
        for batch in loader:
            errors.extend(np.asarray(self.batch_error(state.params, state.batch_stats, batch)).tolist())
        self.last = dict(n=len(errors), relative_L2=statistics(errors))
        return self.last["relative_L2"]["mean"]
