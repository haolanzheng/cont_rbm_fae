#!/usr/bin/env bash
#SBATCH --job-name=fae128-multires
#SBATCH --partition=arm-gpu
#SBATCH --account=pi_ychen5_umassd_edu
#SBATCH --gpus=gh200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the continuous_greedy project root}"
source "$PWD/jobs/slurm_logging.sh"
module load conda/latest
conda activate jaxnn-arm
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=0
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false

experiment_config="$PWD/examples/hyperelasticity/configs/fae128_raw_multires_n2048_b64_e300.json"
experiment_run="$PWD/runs/fae128_raw_multires_n2048_b64_e300_${SLURM_JOB_ID}"
preflight_run="${experiment_run}_preflight"

python -c 'import sys,jax; print("python",sys.executable,"jax",jax.__version__,"devices",jax.devices(),flush=True)'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# One exact epoch in a disposable run catches shape, gradient, memory, and
# four-resolution JIT errors. Formal training then starts afresh from the seed.
srun --kill-on-bad-exit=1 python -u -m examples.hyperelasticity.train_fae \
    --config "$experiment_config" \
    --run-dir "$preflight_run" \
    --stop-after-epochs 1
srun --kill-on-bad-exit=1 python -u -m examples.hyperelasticity.train_fae \
    --config "$experiment_config" \
    --run-dir "$experiment_run"
