# Native FAE implementation

See [the memory diagnosis](MEMORY.md) for the batch-64 Darcy comparison and
the distinction between activation-memory mitigation and native integration.

The common base was adapted from the user's fae-rebano/src on 2026-09-22.
It is now maintained directly as contgreedy.fae, with package-relative imports.
There is no vendor directory, sys.path injection, or runtime dependency on
the original repository; that repository was not modified.

Included: Autoencoder, Encoder/Decoder/Domain interfaces, deterministic FAE loss,
TrainState, Trainer, UNOEncoder, ordinary ResidualMLP blocks,
positional encodings, and the model-saving interface.

Pruning changes:
- trainer.py: one model-agnostic path using Adam, cosine decay, gradient
  clipping, optional accumulation, validation, and Flax checkpoints. It has no
  classifier/logit schedule, parameter grouping, or warmup.
- uno.py: no default PirateKernelPooling import; pooling must be supplied.
- decoder.py: only ResidualMLPBlock and ResidualMLP are retained; the Darcy
  classifier and its discrete material output are excluded.
- checkpoint.py: retain parameter counting and the saving interface only;
  use Flax msgpack plus JSON instead of dill-pickling Python closures. The ARM
  environment has no dill. Numerical model/training code is unaffected.

The hyperelasticity-specific integration is in
`examples/hyperelasticity/fae.py`:
- NumPy/JAX loader with the same (u_dec, x_dec, u_enc, x_enc) interface (ARM has no
  PyTorch); train-only normalization and paired clean/noisy inputs.
- Trapezoidal L2 on the endpoint-inclusive square.
- Orthogonal integral pooling and a continuous output head on the imported
  residual MLP, with decoder LayerNorm disabled.
- Optax MultiSteps: eight microbatches of 16 views per effective batch of 64
  functions and their noisy copies. Average gradients, clip once, Adam once.
- Training, validation, history, and checkpoint handling use the single Trainer.

This is a framework/model migration, not a strictly controlled noise-only
comparison with the old conditional decoder: the imported residual MLP
concatenates z and coordinate features at its input, rather than reinjecting z
at each block. The positional encoding is fixed Fourier features. No Pirate
blocks, POD initialization, consistency loss, or Darcy classification loss
are used by this experiment.

Training resolution is 100x100, obtained by endpoint-aligned interpolation of
the existing 128x128 realizations (same train/validation/test identities).
Validation selection and test_reference use the original 128x128 target grid;
test_training_grid is reported separately. This does not claim a native
100x100 FEM discretization. Hyperparameters are in the experiment JSON.

Runtime dependencies: JAX, Flax, Optax, NumPy.
Training: python -m examples.hyperelasticity.train_fae_pair --config
examples/hyperelasticity/configs/fae100_pair_n2048_b64_e300.json
--run-dir runs/UNIQUE_RUN.
Every run records source hashes, data hash, normalization, selected epoch, and
a compiled/runtime memory preflight. Last/best and selected model state use
Flax msgpack; model_results.json records the history and restoration procedure.

Native modules:
- core.py: model interfaces, Autoencoder, TrainState.
- uno.py / decoder.py: UNO encoder and ordinary residual MLP.
- losses.py / trainer.py / checkpoint.py: FAE loss, trainer, artifacts.

The square-grid loader, quadrature, orthogonal pooling assembly, and
reconstruction metric live with the hyperelasticity example. Other examples
can use the shared interfaces and components without importing that case.

The array-based JAX model and training utilities are available directly as
`contgreedy.fae.model` and `contgreedy.fae.training`. Existing runs retain their
recorded source hashes; moving the modules into the public package does not
change their numerical implementation.
