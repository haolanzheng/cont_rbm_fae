#!/usr/bin/env bash
#SBATCH --job-name=fae-pool-smoke
#SBATCH --partition=arm-gpu
#SBATCH --account=pi_ychen5_umassd_edu
#SBATCH --gpus=gh200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the continuous_greedy project root}"
source "$PWD/jobs/slurm_logging.sh"
module load conda/latest
conda activate jaxnn-arm
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=0
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -c 'import sys,jax; print("python",sys.executable,"jax",jax.__version__,"devices",jax.devices(),flush=True)'
srun --kill-on-bad-exit=1 python -m pytest -q \
    tests/test_fae_pooling.py tests/test_fae.py tests/test_fae_nonlinear_decoders.py
