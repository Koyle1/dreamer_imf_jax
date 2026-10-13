#!/bin/bash
set -euo pipefail
if [[ "$#" != 4 ]]; then
  echo "usage: posterior_reward_job.sh SOURCE OUTPUT STAGE UPSTREAM" >&2
  exit 2
fi
POSTERIOR_SOURCE="$1"
POSTERIOR_OUTPUT="$2"
POSTERIOR_STAGE="$3"
POSTERIOR_UPSTREAM="$4"
case "$POSTERIOR_STAGE" in
  preflight|cell|probes|finalize|advance-preflight|advance-cells|advance-probes|advance-finalize) ;;
  *) echo "unregistered posterior-reward stage" >&2; exit 2 ;;
esac
module purge
module load Python/3.12.3-GCCcore-13.3.0
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl
export PYTHONPATH="$POSTERIOR_SOURCE/dreamer_imf_comparison:$POSTERIOR_SOURCE/imf_dreamer_jax/src:$POSTERIOR_UPSTREAM"
POSTERIOR_PYTHON=/work2/ci72buri-dreamer_imf_neurips/venv-dreamer-ablation-jax0433/bin/python
POSTERIOR_CACHE=$(mktemp -d "$POSTERIOR_OUTPUT/cache/${SLURM_JOB_ID:?}-${SLURM_ARRAY_TASK_ID:-none}-${POSTERIOR_STAGE}.XXXXXX")
export XDG_CACHE_HOME="$POSTERIOR_CACHE/xdg"
export JAX_COMPILATION_CACHE_DIR="$POSTERIOR_CACHE/jax"
export CUDA_CACHE_PATH="$POSTERIOR_CACHE/cuda"
export TMPDIR="$POSTERIOR_CACHE/tmp"
mkdir -p "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"
cd "$POSTERIOR_SOURCE"
if [[ "$POSTERIOR_STAGE" == advance-* ]]; then
  export JAX_PLATFORMS=cpu
  "$POSTERIOR_PYTHON" -u "$POSTERIOR_SOURCE/dreamer_imf_comparison/scripts/submit_posterior_reward.py" "$POSTERIOR_STAGE" --output "$POSTERIOR_OUTPUT"
  exit 0
fi
POSTERIOR_ARGS=(--output "$POSTERIOR_OUTPUT")
if [[ "$POSTERIOR_STAGE" == cell ]]; then
  POSTERIOR_INDEX="${SLURM_ARRAY_TASK_ID:?array index required}"
  if [[ ! "$POSTERIOR_INDEX" =~ ^[0-8]$ ]]; then
    echo "unregistered cell index" >&2
    exit 2
  fi
  POSTERIOR_ARGS+=(--index "$POSTERIOR_INDEX")
fi
"$POSTERIOR_PYTHON" -u -m dreamer_imf_compare.posterior_reward_runner "$POSTERIOR_STAGE" "${POSTERIOR_ARGS[@]}"
"$POSTERIOR_PYTHON" -u -m dreamer_imf_compare.posterior_reward_runner "verify-$POSTERIOR_STAGE" "${POSTERIOR_ARGS[@]}"
