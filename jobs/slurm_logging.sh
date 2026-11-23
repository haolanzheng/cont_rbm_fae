#!/usr/bin/env bash
# Shared logging setup for Slurm batch scripts. Source this after changing to
# SLURM_SUBMIT_DIR and do not replace the batch shell with `exec srun`.

if [[ -n "${CONTINUOUS_GREEDY_SLURM_LOGGING_INITIALIZED:-}" ]]; then
    return 0
fi
CONTINUOUS_GREEDY_SLURM_LOGGING_INITIALIZED=1

slurm_log_dir="$PWD/logs"
slurm_log_job_name="${SLURM_JOB_NAME:-slurm}"
slurm_log_job_id="${SLURM_JOB_ID:-manual-$$}"
slurm_log_file_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
slurm_log_bootstrap="$slurm_log_dir/${slurm_log_job_name}-${slurm_log_job_id}.log"
SLURM_LOG_FILE="$slurm_log_dir/${slurm_log_job_name}-${slurm_log_file_stamp}-${slurm_log_job_id}.log"

mkdir -p "$slurm_log_dir"
if [[ -e "$slurm_log_bootstrap" && "$slurm_log_bootstrap" != "$SLURM_LOG_FILE" ]]; then
    mv -- "$slurm_log_bootstrap" "$SLURM_LOG_FILE"
fi
exec >>"$SLURM_LOG_FILE" 2>&1
export SLURM_LOG_FILE

printf '[%s] START job_name=%s job_id=%s node=%s cwd=%s log=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$slurm_log_job_name" \
    "$slurm_log_job_id" \
    "${SLURMD_NODENAME:-${HOSTNAME:-unknown}}" \
    "$PWD" \
    "$SLURM_LOG_FILE"

_continuous_greedy_slurm_log_finish() {
    local exit_code=$?
    trap - EXIT
    printf '[%s] END job_name=%s job_id=%s exit_code=%d\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$slurm_log_job_name" \
        "$slurm_log_job_id" \
        "$exit_code"
    exit "$exit_code"
}
trap _continuous_greedy_slurm_log_finish EXIT
