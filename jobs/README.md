# Slurm jobs

Submit every batch script from the project root. Standard output and standard
error are written to

```text
logs/<job-name>-<UTC timestamp>-<job-id>.log
```

For example, `fae128-darcy-enc-20260922T143015Z-64670513.log`. Each log also
contains UTC `START` and `END` records, including the final exit code. Training
artifacts and metrics remain under `runs/`; `logs/` is only for scheduler and
process output.

New `.sbatch` files must use this initialization pattern:

```bash
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source "$PWD/jobs/slurm_logging.sh"
```

Keep the batch shell alive by invoking `srun` normally rather than with
`exec srun`; the shell's exit trap writes the timestamped completion record.
