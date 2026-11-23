"""Shared FAE interfaces and Flax state, adapted from fae-rebano."""
from typing import Callable, Tuple
import jax
from jax.typing import ArrayLike
import flax
from flax import linen as nn
from flax.training import train_state

NonlocalTransform = Tuple[Callable[[ArrayLike], jax.Array], Callable[[ArrayLike], jax.Array]]

class Encoder(nn.Module):
    """
    For general comments, see `Autoencoder` documentation.
    """

    is_variational: bool

    def __call__(self, u, x):
        raise NotImplementedError()

    def get_latent_dim(self):
        return self.latent_dim

class Decoder(nn.Module):
    """
    For general comments, see `Autoencoder` documentation.
    """

    @nn.compact
    def __call__(self, z, x, train=False):
        u = self._forward(z, x, train)
        return u

    def _forward(self, z, x):
        raise NotImplementedError()

class Domain:
    """Function-space norms and inner products used by the reconstruction loss."""

    name: str

    def __init__(self, name: str):
        self.name = name

    def squared_norm(self, u: ArrayLike, x: ArrayLike) -> jax.Array:
        raise NotImplementedError()

    def inner_product(self, u: ArrayLike, v: ArrayLike, x: ArrayLike) -> jax.Array:
        raise NotImplementedError()

    def nonlocal_transform(self) -> NonlocalTransform:
        raise NotImplementedError()

class PositionalEncoding:
    r"""Maps co-ordinates :math:`x` to a function :math:`\gamma(x)`.

    A positional encoding is a map :math:`\gamma(x)` mapping an `in_dim`-dimensional co-ordinate :math:`x` to an `encoding_dim`-dimensional
    positional encoding of that co-ordinate.
    Examples of positional encodings include Fourier features.

    All positional encodings should take as input:

    x : jnp.array of shape [n_evals, in_dim]

    and should return an another array of shape [n_evals, encoding_dim].
    """

    def __call__(self, x):
        raise NotImplementedError("`PositionalEncoding` must implement `__call__`.")

class IdentityEncoding(PositionalEncoding):
    def __call__(self, x):
        return x

class Autoencoder(nn.Module):
    """Deterministic or variational FAE; batched coordinates may differ between encoder and decoder."""

    encoder: Encoder
    decoder: Decoder

    def _submodule_variables(self, state, name):
        variables = {"params": state.params[name]}
        if state.batch_stats is not None and name in state.batch_stats:
            variables["batch_stats"] = state.batch_stats[name]
        return variables

    @nn.compact
    def __call__(self, u, x_enc, x_dec, train=False):
        z = self.encoder(u, x_enc, train)

        if self.encoder.is_variational:
            latent_dim = self.get_latent_dim()
            mean, _ = (
                z[:, :latent_dim],
                z[:, latent_dim:],
            )
            return self.decoder(mean, x_dec, train)
        else:
            return self.decoder(z, x_dec, train)

    def encode(self, state, u, x, train=False):
        return self.encoder.apply(
            self._submodule_variables(state, "encoder"),
            u,
            x,
            train,
        )

    def decode(self, state, z, x, train=False):
        return self.decoder.apply(
            self._submodule_variables(state, "decoder"),
            z,
            x,
            train,
        )

    def get_latent_dim(self):
        return self.encoder.get_latent_dim()

class TrainNanError(Exception):
    pass

class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict
    key: jax.Array
