#!/bin/bash
set -euo pipefail
READOUT_SOURCE="$1"
READOUT_PARENT="$2"
READOUT_OUTPUT="$3"
module purge
module load Python/3.12.3-GCCcore-13.3.0
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$READOUT_SOURCE/dreamer_imf_comparison:$READOUT_SOURCE/imf_dreamer_jax/src:/work2/ci72buri-dreamer_imf_neurips/dreamerv3-upstream-e3f0224"
READOUT_PYTHON=/work2/ci72buri-dreamer_imf_neurips/venv-dreamer-ablation-jax0433/bin/python
cd "$READOUT_SOURCE"
"$READOUT_PYTHON" -u -m dreamer_imf_compare.reward_readout_study run --parent "$READOUT_PARENT" --output "$READOUT_OUTPUT"
"$READOUT_PYTHON" -u -m dreamer_imf_compare.reward_readout_study verify --output "$READOUT_OUTPUT"
