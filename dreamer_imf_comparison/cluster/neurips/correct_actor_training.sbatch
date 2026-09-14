#!/usr/bin/env bash
#SBATCH --job-name=imf-correct-actor
#SBATCH --account=dep_inin_dat
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.out
#SBATCH --error=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.err

set -euo pipefail
: "${CAT_SOURCE_ROOT:?CAT_SOURCE_ROOT is required}"
: "${CAT_PILOT_ROOT:?CAT_PILOT_ROOT is required}"
: "${CAT_OUTPUT_ROOT:?CAT_OUTPUT_ROOT is required}"
: "${CAT_SOURCE_COMMIT:?CAT_SOURCE_COMMIT is required}"
: "${CAT_STAGE:?CAT_STAGE is required}"

WORK_ROOT=/work2/ci72buri-dreamer_imf_neurips
module purge
module load Python/3.12.3-GCCcore-13.3.0
source "${WORK_ROOT}/venv-cuda12/bin/activate"
export PYTHONPATH="${CAT_SOURCE_ROOT}:${CAT_SOURCE_ROOT}/imf_dreamer_jax/src:${CAT_SOURCE_ROOT}/dreamer_imf_comparison"
export JAX_ENABLE_X64=0
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="${WORK_ROOT}/jax-compilation-cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_RAISE_PERSISTENT_CACHE_ERRORS=true
export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1

test "$(git -C "${CAT_SOURCE_ROOT}" rev-parse HEAD)" = "${CAT_SOURCE_COMMIT}"
test -z "$(git -C "${CAT_SOURCE_ROOT}" status --porcelain)"

RUNNER=${CAT_SOURCE_ROOT}/dreamer_imf_comparison/scripts/run_correct_actor_training.py
VERIFIER=${CAT_SOURCE_ROOT}/dreamer_imf_comparison/scripts/verify_correct_actor_training.py
COMMON=(--pilot-root "${CAT_PILOT_ROOT}" --output-root "${CAT_OUTPUT_ROOT}")

case "${CAT_STAGE}" in
  preflight)
    python "${RUNNER}" manifest "${COMMON[@]}"
    python "${VERIFIER}" contract "${COMMON[@]}"
    python "${VERIFIER}" implementation "${COMMON[@]}"
    python "${VERIFIER}" selection
    python "${VERIFIER}" self-test
    python "${RUNNER}" cell "${COMMON[@]}" --index 0
    python "${VERIFIER}" preflight --output-root "${CAT_OUTPUT_ROOT}" --index 0
    ;;
  cell)
    : "${SLURM_ARRAY_TASK_ID:?cell stage requires a cell index}"
    python "${RUNNER}" cell "${COMMON[@]}" --index "${SLURM_ARRAY_TASK_ID}"
    python "${RUNNER}" verify-cell --output-root "${CAT_OUTPUT_ROOT}" --index "${SLURM_ARRAY_TASK_ID}"
    ;;
  select)
    python "${VERIFIER}" cells --output-root "${CAT_OUTPUT_ROOT}"
    python "${RUNNER}" select --output-root "${CAT_OUTPUT_ROOT}"
    ;;
  finalize)
    python "${RUNNER}" finalize --output-root "${CAT_OUTPUT_ROOT}"
    python "${VERIFIER}" final --output-root "${CAT_OUTPUT_ROOT}"
    ;;
  *)
    echo "unknown CAT_STAGE=${CAT_STAGE}" >&2
    exit 2
    ;;
esac
