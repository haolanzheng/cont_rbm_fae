#!/usr/bin/env bash
# Run inside an allocated GPU session; this script does not request resources.
# Usage: run_jax_gpu.sh /path/to/python DATA INITIAL OUTPUT [CONFIG] [--resume]
set -euo pipefail
if (( $# < 4 )); then
    echo 'Usage: run_jax_gpu.sh PYTHON DATA INITIAL OUTPUT [CONFIG] [--resume]' >&2
    exit 2
fi
fae_python=$1
fae_data=$2
fae_initial=$3
fae_output=$4
shift 4
fae_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fae_project=$(cd -- "$fae_directory/../.." && pwd)
fae_config="$fae_directory/configs/fae_128.json"
if (( $# > 0 )) && [[ $1 != --* ]]; then
    fae_config=$1
    shift
fi
export PYTHONPATH="$fae_project/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "$fae_python" -u "$fae_directory/train_jax.py" \
    --config "$fae_config" --data "$fae_data" --initial "$fae_initial" \
    --output "$fae_output" --final-test "$@"
