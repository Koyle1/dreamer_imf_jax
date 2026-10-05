#!/bin/bash
set -euo pipefail
PARALLEL_SOURCE="$1"
module purge
module load Python/3.12.3-GCCcore-13.3.0
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PARALLEL_SOURCE/dreamer_imf_comparison:$PARALLEL_SOURCE/imf_dreamer_jax/src"
exec /work2/ci72buri-dreamer_imf_neurips/venv-dreamer-ablation-jax0433/bin/python -u "$PARALLEL_SOURCE/dreamer_imf_comparison/scripts/advance_parallel_study.py" --root "$2" --completed-stage "$3"
