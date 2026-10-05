#!/bin/bash
set -euo pipefail
PARALLEL_SOURCE="$1"
module purge
module load Python/3.12.3-GCCcore-13.3.0
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu
export PYTHONPATH="$PARALLEL_SOURCE/dreamer_imf_comparison:$PARALLEL_SOURCE/imf_dreamer_jax/src:/work2/ci72buri-dreamer_imf_neurips/dreamerv3-upstream-e3f0224"
cd "$PARALLEL_SOURCE"
PARALLEL_PYTHON=/work2/ci72buri-dreamer_imf_neurips/venv-dreamer-ablation-jax0433/bin/python
"$PARALLEL_PYTHON" -m unittest discover -s imf_dreamer_jax/tests -p 'test_parallel_*.py' -v
"$PARALLEL_PYTHON" -m unittest discover -s dreamer_imf_comparison/tests -p 'test_parallel_*.py' -v
"$PARALLEL_PYTHON" -m unittest discover -s dreamer_imf_comparison/tests -p 'test_conditional_repair.py' -v
"$PARALLEL_PYTHON" -m unittest discover -s dreamer_imf_comparison/tests -p 'test_joint_repair.py' -v
"$PARALLEL_PYTHON" -m unittest discover -s dreamer_imf_comparison/tests -p 'test_staged_*.py' -v
"$PARALLEL_PYTHON" -m unittest discover -s dreamer_imf_comparison/tests -p 'test_dreamer_ablation_*.py' -v
"$PARALLEL_PYTHON" -u dreamer_imf_comparison/scripts/parallel_frozen_smoke.py
git diff --check
echo PARALLEL_TRAJECTORY_REGRESSIONS_VERIFIED
