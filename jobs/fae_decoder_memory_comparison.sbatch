#!/usr/bin/env bash
#SBATCH --job-name=fae-darcy-memory
#SBATCH --partition=arm-gpu
#SBATCH --account=pi_ychen5_umassd_edu
#SBATCH --gpus=gh200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source "$PWD/jobs/slurm_logging.sh"
module load conda/latest
conda activate jaxnn-arm
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD/src:$PWD"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=0 JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
srun --kill-on-bad-exit=1 python tools/fae_decoder_memory_comparison.py \
  --reference-root "$PWD/../../fae-rebano" \
  --output "runs/fae_decoder_memory_comparison_${SLURM_JOB_ID}.json"
