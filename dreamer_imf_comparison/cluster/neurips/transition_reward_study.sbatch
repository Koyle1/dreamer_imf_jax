#!/usr/bin/env bash
#SBATCH --job-name=imf-itpo-reward
#SBATCH --account=dep_inin_dat
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.out
#SBATCH --error=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.err

set -euo pipefail
: "${TR_SOURCE_ROOT:?TR_SOURCE_ROOT is required}"
: "${TR_CORRECTED_ROOT:?TR_CORRECTED_ROOT is required}"
: "${TR_OUTPUT_ROOT:?TR_OUTPUT_ROOT is required}"
: "${TR_SOURCE_COMMIT:?TR_SOURCE_COMMIT is required}"
: "${TR_STAGE:?TR_STAGE is required}"

WORK_ROOT=/work2/ci72buri-dreamer_imf_neurips
module purge
module load Python/3.12.3-GCCcore-13.3.0
source "${WORK_ROOT}/venv-cuda12/bin/activate"
export PYTHONPATH="${TR_SOURCE_ROOT}:${TR_SOURCE_ROOT}/imf_dreamer_jax/src:${TR_SOURCE_ROOT}/dreamer_imf_comparison"
export JAX_ENABLE_X64=0
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="${WORK_ROOT}/jax-compilation-cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_RAISE_PERSISTENT_CACHE_ERRORS=true
export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1

test "$(git -C "${TR_SOURCE_ROOT}" rev-parse HEAD)" = "${TR_SOURCE_COMMIT}"
test -z "$(git -C "${TR_SOURCE_ROOT}" status --porcelain)"

RUNNER=${TR_SOURCE_ROOT}/dreamer_imf_comparison/scripts/run_transition_reward_study.py
VERIFIER=${TR_SOURCE_ROOT}/dreamer_imf_comparison/scripts/verify_transition_reward_study.py
COMMON=(
  --corrected-root "${TR_CORRECTED_ROOT}"
  --output-root "${TR_OUTPUT_ROOT}"
  --reward-updates "${TR_REWARD_UPDATES:-10000}"
  --heldout-batches "${TR_HELDOUT_BATCHES:-32}"
)

case "${TR_STAGE}" in
  preflight)
    python "${RUNNER}" manifest "${COMMON[@]}"
    python "${VERIFIER}" --self-test
    python "${VERIFIER}" --implementation
    python "${VERIFIER}" --gpu
    python "${TR_SOURCE_ROOT}/imf_dreamer_jax/scripts/verify_transition_reward.py"
    echo IMF_TRANSITION_REWARD_PREFLIGHT_VERIFIED
    ;;
  reward)
    : "${SLURM_ARRAY_TASK_ID:?reward stage requires index 0..2}"
    python "${RUNNER}" reward "${COMMON[@]}" --index "${SLURM_ARRAY_TASK_ID}"
    python "${RUNNER}" verify-reward --output-root "${TR_OUTPUT_ROOT}" \
      --index "${SLURM_ARRAY_TASK_ID}"
    ;;
  actor)
    : "${SLURM_ARRAY_TASK_ID:?actor stage requires index 0..5}"
    python "${RUNNER}" actor "${COMMON[@]}" --index "${SLURM_ARRAY_TASK_ID}"
    python "${RUNNER}" verify-actor --output-root "${TR_OUTPUT_ROOT}" \
      --index "${SLURM_ARRAY_TASK_ID}"
    ;;
  finalize)
    python "${RUNNER}" finalize --output-root "${TR_OUTPUT_ROOT}"
    python "${VERIFIER}" --final --output-root "${TR_OUTPUT_ROOT}"
    ;;
  *)
    echo "unknown TR_STAGE=${TR_STAGE}" >&2
    exit 2
    ;;
esac
