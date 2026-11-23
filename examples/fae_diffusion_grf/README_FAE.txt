FAE for bounded Matérn diffusion coefficients
=============================================

Two supported implementations
-----------------------------
train_torch.py: local PyTorch/MPS, float32.
train_jax.py:   server JAX/CUDA, float32; imports contgreedy.fae.
Both read configs/fae_128.json, the same data directory and the same initial
parameter archive. No separate backend-specific architecture or training
hyperparameters are maintained. Hardware kernels, tensor layout, and the
order of equivalent linear operations may differ. Bitwise equality of long
training trajectories is not expected.

Shared model and training
-------------------------
Input: a(x)=exp(log(10)*tanh(g(x))), where g is a unit-variance Matérn field
with nu=1 and correlation length 0.2. The encoder reconstructs a, not g,
log(a), or the PDE solution. The physical domain is (0,1)^2; network
coordinates map this domain affinely to [-1,1]^2.

The encoder is the native convolutional U-shaped UNOEncoder:
channels [32,64,128], one residual block per level, 16 output channels,
GroupNorm and GELU. This is not the older spectral UNO implementation.
Orthogonal integral pooling has hidden width 64, Gram regularization 1e-3,
and latent dimension 128. The coordinate decoder has width 256 and six
residual blocks; both coordinate encodings use six frequency levels.
The complete model has 1,702,914 trainable parameters.

The scalar head maps every decoded value into (0.1,10):
    a_reconstructed = 0.1 + 9.9*sigmoid(raw_decoder_output).
This enforces conductivity bounds. It does not constrain latent vectors to
the training distribution or guarantee an accurate decoder outside it.

Data: 2048/256/256 train/validation/test fields on 129x129 nodes. The initial
256/64/64 generation-pilot samples are unchanged prefixes. Additional FAE
samples require no PDE solves. One scalar mean and standard deviation,
fitted on training data only, standardize the inputs and reconstruction loss.
The encoder always receives the full grid. Each update samples 1024 decoder
query nodes with replacement according to normalized trapezoidal weights.
The expected reconstruction loss is
    0.5 * integral[((a_reconstructed-a)/training_std)^2].
The complete loss adds 1e-4 * mean(||latent||_2^2).

Training: batch 32, 100 epochs, Adam (beta1=.9, beta2=.999, eps=1e-8), no
weight decay, global gradient clipping at 1. The cosine learning rate is
1e-3 on the first update and 1e-5 on the final update. Both backends use the
same NumPy PCG64 seed for epoch permutations and query sampling. Validation
uses all nodes at epoch 1, every five epochs, and epoch 100. The lowest
validation mean relative L2 error selects the checkpoint. Test evaluation
follows checkpoint selection. Test data must not guide subsequent tuning.

Running
-------
All commands below run from the project root. Use a tested Python environment
for each backend. The local run uses PyTorch 2.15.0, NumPy 2.3.1 and Matplotlib
3.11.2. The JAX CPU reference checks use JAX 0.6.2, Flax 0.10.7 and Optax 0.2.5;
CUDA requires a compatible GPU-enabled JAX installation on the server.

    python examples/fae_diffusion_grf/train_torch.py \
        --data runs/fae_matern_20261122_training/data \
        --initial runs/fae_matern_20261122_training/initial_128.npz \
        --output runs/fae_matern_20261122_training/local_torch_float32_latent128_b32 \
        --device mps

Inside an allocated server GPU session (no job is submitted by this script):

    bash examples/fae_diffusion_grf/run_jax_gpu.sh /path/to/python \
        runs/fae_matern_20261122_training/data \
        runs/fae_matern_20261122_training/initial_128.npz \
        runs/fae_matern_20261122_training/server_jax_float32_latent128_b32

Both support --resume, restoring model, optimizer, random sampling state and
learning-rate position. Resume requires unchanged configuration and data.
Use --stop-after-epochs K for a controlled interruption without changing the
configured schedule. Existing checkpoints are never silently overwritten.
--config PATH selects another shared configuration; changing latent/model
sizes also requires a matching initial parameter archive.

Generate a new shared initial archive once, using JAX, then copy that same
archive to both machines (do not independently initialize each framework):

    PYTHONPATH=src JAX_PLATFORMS=cpu python \
        examples/fae_diffusion_grf/initialize_fae.py \
        --data DATA --config CONFIG --output INITIAL.npz

Data generation: prepare_fae.py creates all splits, training-only
normalization and POD controls. The default generated data match
configs/fae_128.json. train_fae.py is a compatibility entry point forwarding
to train_jax.py; it does not maintain another training implementation.

Backend alignment checks
------------------------
First export a JAX reference, then check Torch with the same configuration:

    PYTHONPATH=src JAX_PLATFORMS=cpu python \
        examples/fae_diffusion_grf/export_jax_reference.py \
        --data DATA --initial INITIAL.npz --output CHECK_DIRECTORY

    python examples/fae_diffusion_grf/check_alignment_torch.py \
        --initial INITIAL.npz --reference CHECK_DIRECTORY --device mps

Checks cover parameter names/shapes, forward values, loss, input and parameter
gradients, and Adam. The optimizer check additionally uses identical supplied
gradients to separate optimizer agreement from rounding of near-zero gradients.
Each run records the configuration and SHA-256 hashes of all data splits and
initial weights. Re-run alignment checks after changing framework versions or
model implementations. Matching seeds alone do not establish alignment.

export_torch_weights.py --checkpoint BEST.pt --initial INITIAL.npz
--output TRAINED.npz exports a trained Torch model in native Flax parameter
layout. The adjacent JSON records its normalization and configuration. This
supports the same trained encoder/decoder in either backend; optimizer state
conversion is not provided. Exact training continuation uses --resume within
the original backend.

MPS uses CPU Cholesky and triangular solves only for the small latent matrix;
autograd remains connected across devices. Integration before whitening is
algebraically equivalent to the native JAX pooling, with a smaller solve.
Do not use the old local Torch 2.5.0.dev20240707 environment: its MPS backward
failed the gradient comparison even when its forward values agreed.

Batch throughput can be rechecked with benchmark_torch.py --data DATA
--initial INITIAL.npz --output RESULTS.json --batches 16 32 64. This is a
separate benchmark; do not run it concurrently with a timing experiment.
On the local M2 Max, batch 32 was faster per sample than batches 16 and 64.

Artifacts
---------
All datasets, logs, checkpoints and checks belong under runs/, not examples/
or paper/. The current local run preserves source_at_launch/ and its hashes.
The 128-dimensional POD validation mean relative L2 error is 19.78%; this is
a linear control, not a lower bound on the achievable nonlinear FAE error.
The current FAE is an initial reconstruction experiment. Its latent search
region and coupling to continuous greedy have not yet been validated.
