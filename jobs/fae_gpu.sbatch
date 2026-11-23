#!/usr/bin/env bash
#SBATCH --job-name=fae-kernel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source "$PWD/jobs/slurm_logging.sh"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
srun "${FAE_PYTHON:-python}" -u run.py fae-train
