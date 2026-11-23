#!/usr/bin/env bash
#SBATCH --job-name=fae128-orth-deep
#SBATCH --partition=arm-gpu
#SBATCH --account=pi_ychen5_umassd_edu
#SBATCH --gpus=gh200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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
python -c 'import sys,jax; print("python",sys.executable,"jax",jax.__version__,"devices",jax.devices(),flush=True)'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
srun --kill-on-bad-exit=1 python -u -m examples.hyperelasticity.train_fae \
    --config "$PWD/examples/hyperelasticity/configs/fae128_pool_orthogonal_w32_d256_b6_n2048_b64_e300.json" \
    --run-dir "$PWD/runs/fae128_pool_orthogonal_w32_d256_b6_n2048_b64_e300_${SLURM_JOB_ID}"
