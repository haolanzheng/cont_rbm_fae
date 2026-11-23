# Continuous Greedy Reduced Basis Methods and Functional Autoencoders

Research code for greedy construction of reduced basis spaces using continuous
optimization, together with functional autoencoders (FAEs) for finite-dimensional
parameterization of function inputs.

The repository contains source code, configurations, tests, and usage notes.
Datasets, trained models, experiment outputs, manuscripts, and figures are kept
locally and excluded from version control.

## Repository layout

| Path | Purpose |
|---|---|
| `src/contgreedy/data/` | Data loading, statistics, and Gaussian random field assembly |
| `src/contgreedy/domain/` | Box, ball, and sphere constraints |
| `src/contgreedy/hf/` | Shared conventions for high-fidelity models |
| `src/contgreedy/rom/` | Weighted basis construction and the outer greedy iteration |
| `src/contgreedy/sampling/` | Sampling distributions, random pools, Adam, and local optimization |
| `src/contgreedy/fae/` | FAE components, training, checkpoints, and differentiation through the decoder |
| `examples/` | PDE models, reduced models, error indicators, experiment drivers, and configurations |
| `configs/` | Shared configurations and command definitions |
| `jobs/` | Cluster submission scripts |
| `tests/` | Numerical checks and regression tests |
| `docs/` | Architecture and usage notes |

Start with [the baseline examples](examples/baseline/README.md), followed by
[thermal-fin](examples/thermal_fin/README.md). See
[the architecture guide](docs/ARCHITECTURE.md) for the division between shared
components and example-specific implementations.

## Installation and quick checks

Use Python 3.10 or later. From the repository root:

```bash
python -m pip install -e '.[test,fem]'
python run.py list --all
python -m pytest tests/test_structure.py -q
```

FAE dependencies depend on the implementation: PyTorch for the local MPS model,
and JAX, Flax, and Optax for the native JAX model. Install the appropriate
accelerator-enabled framework in a separate environment; see the
[FAE training instructions](examples/fae_diffusion_grf/README_FAE.txt).

Small PDE runs:

```bash
python run.py spectral --config examples/baseline/configs/smoke.json --output runs/spectral_smoke
python run.py thermal-fin --config examples/thermal_fin/configs/smoke.json --output runs/fin_smoke
```

These entry points require a new output directory. Historical drivers may use
existing result files as caches. Some regression tests also require local data
or manuscript files that are not distributed in this repository.

## PDE examples

The baseline examples include controlled source families for elliptic problems,
with sphere, ball, and cube parameter domains. The continuous finite-element
formulation is implemented in `examples/baseline/continuous_fem.py`; the older
grid-calibrated formulation is retained for historical comparisons.

Other examples include piecewise constant forcing, thermal-fin, thermal-block,
elasticity, nonlinear elliptic problems, and BGK models. Model definitions,
search settings, and evaluation procedures are specific to each example. See
[the experiment index](docs/EXPERIMENTS.md) for entry points and data requirements.

## Functional autoencoders

The diffusion-coefficient example provides two implementations with a shared
model and training configuration:

- `examples/fae_diffusion_grf/train_torch.py`: local PyTorch/MPS training.
- `examples/fae_diffusion_grf/train_jax.py`: JAX training on a server GPU.
- `examples/fae_diffusion_grf/configs/fae_128.json`: shared parameters.

Both implementations use the same data splits, normalization, initial parameter
archive, loss, optimizer settings, and learning-rate schedule. Their numerical
agreement is checked separately from hardware-dependent implementation details.
Instructions cover data generation, initialization, training, checkpoint
recovery, and weight transfer in
[README_FAE.txt](examples/fae_diffusion_grf/README_FAE.txt).

`examples/hyperelasticity/` contains additional input-generation and FAE studies.
These implementations do not yet provide a complete nonlinear hyperelasticity
PDE/ROM pipeline coupled to latent-space greedy search.

## Running experiments

Run commands from the repository root. List available experiment and report
commands with:

```bash
python run.py list --all
```

Use a separate directory under `runs/` for each new experiment. Reporting scripts
read previously generated results; they do not rerun the underlying experiments.
Cluster scripts under `jobs/` require an appropriate account, partition, and
Python environment for the target system.
