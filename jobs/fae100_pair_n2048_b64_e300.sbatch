#!/usr/bin/env bash
#SBATCH --job-name=fae100-pair
#SBATCH --partition=arm-gpu
#SBATCH --account=pi_ychen5_umassd_edu
#SBATCH --gpus=gh200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
source "$PWD/jobs/slurm_logging.sh"
module load conda/latest
conda activate jaxnn-arm
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$PWD/src:$PWD"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=0 JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
extra_args=()
if [[ "${FAE_PREFLIGHT_ONLY:-0}" == 1 ]]; then
    extra_args+=(--preflight-only)
fi
srun --kill-on-bad-exit=1 python -u -m examples.hyperelasticity.train_fae_pair \
  --config examples/hyperelasticity/configs/fae100_pair_n2048_b64_e300.json \
  --run-dir "runs/fae100_pair_n2048_b64_e300_${SLURM_JOB_ID}" "${extra_args[@]}"
