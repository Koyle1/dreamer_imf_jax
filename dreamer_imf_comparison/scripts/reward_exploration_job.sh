#!/bin/bash
set -euo pipefail
if [[ "$#" != 5 ]]; then
  echo "usage: reward_exploration_job.sh SOURCE OUTPUT STAGE PARENT_CELL DATASET" >&2
  exit 2
fi
EXPLORE_SOURCE="$1"
EXPLORE_OUTPUT="$2"
EXPLORE_STAGE="$3"
EXPLORE_PARENT="$4"
EXPLORE_DATASET="$5"
case "$EXPLORE_STAGE" in
  match|preflight|cell|finalize|advance-match|advance-preflight|advance-cells) ;;
  *) echo "unregistered reward-exploration stage" >&2; exit 2 ;;
esac
module purge
module load Python/3.12.3-GCCcore-13.3.0
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYTHONNOUSERSITE=1
export PYTHONPATH="$EXPLORE_SOURCE/dreamer_imf_comparison:$EXPLORE_SOURCE/imf_dreamer_jax/src:/work2/ci72buri-dreamer_imf_neurips/dreamerv3-upstream-e3f0224"
EXPLORE_PYTHON=/work2/ci72buri-dreamer_imf_neurips/venv-dreamer-ablation-jax0433/bin/python
# A fresh private cache per scheduler job, stage and array index; no cache reuse
# between source deployments, cells, retries, fitting or verification processes.
EXPLORE_CACHE=$(mktemp -d "$EXPLORE_OUTPUT/cache/${SLURM_JOB_ID:?}-${SLURM_ARRAY_TASK_ID:-none}-${EXPLORE_STAGE}.XXXXXX")
export XDG_CACHE_HOME="$EXPLORE_CACHE/xdg"
export JAX_COMPILATION_CACHE_DIR="$EXPLORE_CACHE/jax"
export CUDA_CACHE_PATH="$EXPLORE_CACHE/cuda"
export TMPDIR="$EXPLORE_CACHE/tmp"
mkdir -p "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"
cd "$EXPLORE_SOURCE"
if [[ "$EXPLORE_STAGE" == advance-* ]]; then
  "$EXPLORE_PYTHON" -u "$EXPLORE_SOURCE/dreamer_imf_comparison/scripts/submit_reward_exploration.py" "$EXPLORE_STAGE" --output "$EXPLORE_OUTPUT"
  exit 0
fi
EXPLORE_ARGS=(--output "$EXPLORE_OUTPUT" --parent-cell "$EXPLORE_PARENT" --dataset "$EXPLORE_DATASET")
if [[ "$EXPLORE_STAGE" == cell ]]; then
  EXPLORE_INDEX="${SLURM_ARRAY_TASK_ID:?array index is required}"
  if [[ ! "$EXPLORE_INDEX" =~ ^([0-9]|1[01])$ ]]; then
    echo "unregistered cell index" >&2
    exit 2
  fi
  EXPLORE_ARGS+=(--index "$EXPLORE_INDEX")
fi
"$EXPLORE_PYTHON" -u -m dreamer_imf_compare.reward_exploration_runner "$EXPLORE_STAGE" "${EXPLORE_ARGS[@]}"
# Verification is a new process that reconstructs evidence, not learner state.
"$EXPLORE_PYTHON" -u -m dreamer_imf_compare.reward_exploration_runner "verify-$EXPLORE_STAGE" "${EXPLORE_ARGS[@]}"
