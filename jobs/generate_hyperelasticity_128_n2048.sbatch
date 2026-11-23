#!/usr/bin/env bash
#SBATCH --job-name=fae-data128
#SBATCH --partition=arm
#SBATCH --account=pi_ychen5_umassd_edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the continuous_greedy project root}"
source "$PWD/jobs/slurm_logging.sh"
module load conda/latest
conda activate jaxnn-arm
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"

srun --kill-on-bad-exit=1 python -u -m examples.hyperelasticity.data \
    --resolution 128 \
    --train-count 2048 \
    --validation-count 256 \
    --test-count 256 \
    --output-dir data_128_formal2048

python - <<'PY'
import json
from pathlib import Path

import numpy as np

root = Path("data/data_128_formal2048")
metadata = json.loads((root / "metadata.json").read_text())
fields = np.load(root / "fields.npy", mmap_mode="r", allow_pickle=False)
assert fields.shape == (2560, 128 * 128), fields.shape
assert fields.dtype == np.float32, fields.dtype
assert metadata["shape"] == [2560, 128 * 128]
assert metadata["grid_resolution"] == 128
assert metadata["splits"] == {
    "train": [0, 2048],
    "validation": [2048, 2304],
    "test": [2304, 2560],
}
assert np.isfinite(fields).all()
assert np.isfinite(metadata["mean"]) and metadata["std"] > 0
print("DATA_VALIDATION_PASSED", metadata, flush=True)
PY
