#!/bin/bash
set -euo pipefail
PARALLEL_SOURCE="$1"
PARALLEL_ROOT="$2"
PARALLEL_STAGE="$3"
module purge
module load Python/3.12.3-GCCcore-13.3.0
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
export PYTHONPATH="$PARALLEL_SOURCE/dreamer_imf_comparison:$PARALLEL_SOURCE/imf_dreamer_jax/src:/work2/ci72buri-dreamer_imf_neurips/dreamerv3-upstream-e3f0224"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$PARALLEL_SOURCE"
PARALLEL_PYTHON=/work2/ci72buri-dreamer_imf_neurips/venv-dreamer-ablation-jax0433/bin/python
"$PARALLEL_PYTHON" -u -m dreamer_imf_compare.parallel_study "$PARALLEL_STAGE" --root "$PARALLEL_ROOT" --seed "${SLURM_ARRAY_TASK_ID:-0}"
