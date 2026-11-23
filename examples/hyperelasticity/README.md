# Hyperelasticity input representation

## Native FAE implementation (2026-09-22)

The new entry point is `train_fae_pair.py`. Its case-specific FAE
assembly, square-grid quadrature, loader, and metric are in `fae.py`;
the reusable model and trainer components are in `src/contgreedy/fae`.
It directly uses the original UNO, residual MLP, Autoencoder, FAE loss,
Trainer, Optax and serializer. See
[the provenance and migration notes](../../src/contgreedy/fae/README.md).
It has no runtime dependency on the sibling fae-rebano checkout or PyTorch.

The proposed 100x100 configuration uses the same original 128x128 realizations,
64 clean/noisy pairs per optimizer update, microbatch 16, latent 128, initial
LR 1e-3 and 300 epochs. Both original-grid and training-grid test errors are
reported; validation selection uses original-grid targets. This is a model
migration and resolution change, not a noise-only controlled comparison.

```bash
sbatch jobs/fae100_pair_n2048_b64_e300.sbatch
# Compile and execute a memory/speed preflight without starting training:
sbatch --time=00:10:00 --export=ALL,FAE_PREFLIGHT_ONLY=1 jobs/fae100_pair_n2048_b64_e300.sbatch
```

The older experiments and checkpoint formats described below remain available
for reproduction; the failed full-batch clean/noisy script is historical.

Only input generation and FAE training are implemented. The nonlinear
hyperelastic truth solver, ROM and full greedy experiment remain pending.

- `data.py`: case-specific GRF parameters and dataset generation.
- `fae.py`: case-specific FAE assembly, data loader, and metric.
- `check_fae.py`: FAE checks used by `python run.py fae-check`.
- `train_fae.py`: current JAX training entry point.
- `configs/fae128_jax_w32.json`: widened model, initial LR 1e-3, no warmup.
- `train_fae.py` and `configs/fae128.json`: historical PyTorch/epoch-4 protocol.
- `contgreedy.data.grf`: reusable P1 GRF assembly.
- `contgreedy.fae`: reusable model, trainer, reconstruction checks and latent VJP.

The current FAE is pure JAX and starts from random initialization in a fresh run.
Array-based experiments import `contgreedy.fae.model` and
`contgreedy.fae.training` directly; the Flax common-base implementation uses
the modules listed above.
UNO widths are 32/48/96/192; latent 128, batch 32, 200 epochs/6400 updates.
Learning rate decays from 1e-3 at update 1 to 1e-5 at update 6400.
The former PyTorch files and checkpoint remain as a numerical reference.

```bash
sbatch jobs/fae_gpu_fresh.sbatch
# On an allocated GPU, after activating jaxnn-arm:
python run.py fae-train --run-dir runs/my_fresh_jax_fae
```

The Slurm script selects GH200 and jaxnn-arm. Test data are evaluated only
after all 200 epochs, using the checkpoint selected on validation data.
See [the JAX GPU protocol](../../docs/FAE_JAX_GPU.md) for the full configuration,
checkpoint format and explicit resume command.
