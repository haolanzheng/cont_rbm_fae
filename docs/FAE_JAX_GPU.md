# JAX FAE on GH200

The selected input FAE is a pure-JAX UNO/kernel-pooling encoder with a direct
nonlinear conditional coordinate decoder. The frozen 4096-field experiment is
documented in [FAE_NONLINEAR_DECODERS.md](FAE_NONLINEAR_DECODERS.md). The older
FiLM/LayerNorm JAX and PyTorch checkpoints are reference artifacts and are not
loaded.

## Reproducible configuration

Source: `examples/hyperelasticity/configs/fae64_jax_conditional_mlp_n4096.json`.

- Data: 80 x 80 log-modulus fields; 4096/256/256 train/validation/test.
- UNO channels: 32/48/96/192, eight Fourier modes per direction, spectral rank
  capped at 32, output features 8.
- Coordinate kernel: 28 -> 64 -> 64 -> 512, signed trapezoidal integration.
- Latent dimension: 64.
- Decoder: width 256, four ordinary GELU residual blocks. Raw latent is injected
  at the input and each block. There is no decoder LayerNorm, POD-like basis, or
  Pirate block.
- Shared coordinate encoding: six frequency levels.
- Parameters: 1,137,743 real float32 scalars. Complex spectral weights are pairs
  of real arrays; spectral arithmetic is complex64.
- Loss: 0.5 * standardized-field quadrature reconstruction error squared
  + 1e-4 * latent squared Euclidean norm.
- Batch 32, 200 epochs, 25,600 updates, seed 310519.
- Adam: beta1=.9, beta2=.999, epsilon=1e-8, weight decay=0.
- No warmup. Update s uses lr = 1e-5 + (1e-3-1e-5)/2 *
  (1 + cos(pi*(s-1)/25599)). First lr=1e-3; final lr=1e-5.
- Global gradient clipping at norm 1. No mixed precision; highest matmul precision.
- Validation at epoch 1, every ten epochs, and completion. Only validation selects
  the best checkpoint. Held-out test evaluation runs after all 200 epochs.

The completed selected model uses epoch 200: validation/test mean relative L2
errors are 0.151978/0.153882. Its test median/p95/maximum are
0.151741/0.242164/0.312092. These are one-seed architecture-selection results,
not an uncertainty estimate.

## Submission and outputs

From the repository root:

```bash
sbatch --parsable jobs/fae_nonlinear_decoder_comparison.sbatch
```

The script loads `conda/latest`, activates `jaxnn-arm`, and requests one
GH200 on `arm-gpu`, 4 CPUs, 32 GB RAM and 8 hours. It requires CUDA.
The existing ARM environment has JAX 0.10.2 and NumPy 2.4.6.
It uses the repository via PYTHONPATH without changing that environment.

Each submission creates `runs/fae_nonlinear_decoder_comparison_<jobid>` and
trains the conditional MLP and SIREN control sequentially from scratch. The log
is `runs/fae-nonlinear-decoders-<jobid>.log`. The aggregate `experiment.json`
records the frozen configurations, source hashes, stages, and latent-rank
diagnostics. In each stage, `last.npz` contains parameters, Adam moments/update
count, configuration, RNG, and history; `best.npz` contains the validation-
selected model. `metrics.json` appears only after complete training and test
evaluation.

To explicitly resume an interrupted stage, use an allocated GPU and its frozen
configuration (the paired submission intentionally never auto-resumes):

```bash
python -m examples.hyperelasticity.train_fae_jax \
  --config examples/hyperelasticity/configs/fae64_jax_conditional_mlp_n4096.json \
  --run-dir runs/my_conditional_mlp --resume
```

The model and trainer are in `src/contgreedy/fae/jax_model.py` and
`jax_training.py`; neither imports PyTorch. Tests cover parameter/input
gradients, nonlinear dependence on the latent code, SIREN initialization
bounds, parameter counts, learning-rate endpoints, and exact interrupted-versus-
continuous training state.

The 1024-field and data/capacity history remains in
[FAE_SCALING.md](FAE_SCALING.md); the fixed-versus-FiLM POD-like decoder
ablation is in [FAE_DECODER_ABLATION.md](FAE_DECODER_ABLATION.md). Neither is a
backend-only comparison. Cross-resolution reconstruction and decoder-gradient
quality still require trained-model audits before latent-space greedy search.
