# Historical PyTorch FAE reference

This document describes the earlier UNO/kernel-pooling/FiLM model used for
migration checks. It is distinct from the paired PyTorch/MPS and JAX diffusion
example documented in
[README_FAE.txt](../examples/fae_diffusion_grf/README_FAE.txt).
Historical data and checkpoints are local artifacts and are not distributed
with this repository.

## Model

The encoder replaces a featurewise DeepSets average with a coordinate-dependent
matrix kernel. It produces one 128-dimensional latent vector and uses a FiLM
coordinate decoder. This architecture choice does not establish superiority
over attention pooling or other encoders.

Let `H(x)` denote the eight-channel UNO feature field and `p(x)` the shared
positional encoding:

```text
K_theta(p(x)) : R^28 -> R^(128 x 8)
z = W * sum_i w_i K_theta(p(x_i)) H(x_i) in R^128
prediction(x) = FiLM_coordinate_MLP(z, p(x))
```

- Kernel MLP: 28 -> 64 -> 64 -> 1024, with GELU and signed weights.
- Integration: two-dimensional trapezoidal quadrature on the physical domain,
  with weights summing to one.
- UNO padding is internal. Features are resampled to the physical grid before
  pooling.
- The encoder, pooling operator, and decoder share the trainable positional
  encoding.
- The model has 496,463 real scalar degrees of freedom when each complex
  coefficient is counted as two real numbers.
- The encoder accepts regular grids at different resolutions. The decoder
  accepts arbitrary query coordinates; this does not imply support for
  arbitrary irregular encoder grids.

The Fourier normalization is chosen consistently with the number of nodes when
transferring spectral coefficients between grids.

## Reference configuration

| Setting | Value |
|---|---|
| Grid | 80 x 80 |
| Training / validation / test fields | 1024 / 256 / 256 |
| Latent dimension | 128 |
| Batch size | 32 |
| Full training budget | 200 epochs, 6400 updates |
| Optimizer | Adam; five-epoch warmup to 5e-4, cosine decay to 1e-5 |
| Loss | Half the standardized-field quadrature reconstruction error squared, plus 1e-4 times the squared latent norm |

No additional rank, attention, or spectral penalty is used. The saved training
state includes model parameters, optimizer state, and random sampling state.
Validation selects the checkpoint; test evaluation follows completed training.
An interruption does not shorten the configured learning-rate schedule.

## Reference checkpoints and tests

The historical local run stopped after epoch 4 and 128 updates. Its best
previously evaluated validation checkpoint was epoch 2; epochs 3 and 4 did not
include new validation evaluations. These checkpoint files are not required to
read the source and are not part of the Git repository.

The reference model is retained under `tests/fixtures/` for numerical comparison.
Tests that load historical checkpoints require those local files. Current
training entry points and their checkpoint formats are documented with the
corresponding example; the earlier PyTorch checkpoint should not be passed to a
JAX training driver as an interchangeable optimizer state.

## Implementation context

The kernel-pooling construction follows the matrix-kernel integral approach in
*Autoencoders in Function Space* and its reference implementation:

- https://jmlr.org/papers/v26/25-0035.html
- https://github.com/htlambley/functional_autoencoders

The implementation is an independent model variant, not an exact reproduction
of the published experiments. Reconstruction quality and the reliability of
latent-space optimization require separate assessment.
