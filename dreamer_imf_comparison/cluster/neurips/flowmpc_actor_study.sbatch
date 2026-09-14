#!/usr/bin/env bash
#SBATCH --job-name=imf-flowmpc-actor
#SBATCH --account=dep_inin_dat
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.out
#SBATCH --error=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.err

set -euo pipefail
: "${FM_SOURCE_ROOT:?FM_SOURCE_ROOT is required}"
: "${FM_REWARD_ROOT:?FM_REWARD_ROOT is required}"
: "${FM_OUTPUT_ROOT:?FM_OUTPUT_ROOT is required}"
: "${FM_SOURCE_COMMIT:?FM_SOURCE_COMMIT is required}"
: "${FM_STAGE:?FM_STAGE is required}"

WORK_ROOT=/work2/ci72buri-dreamer_imf_neurips
module purge
module load Python/3.12.3-GCCcore-13.3.0
source "${WORK_ROOT}/venv-cuda12/bin/activate"
export PYTHONPATH="${FM_SOURCE_ROOT}:${FM_SOURCE_ROOT}/imf_dreamer_jax/src:${FM_SOURCE_ROOT}/dreamer_imf_comparison"
export JAX_ENABLE_X64=0
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="${WORK_ROOT}/jax-compilation-cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_RAISE_PERSISTENT_CACHE_ERRORS=true
export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1

test "$(git -C "${FM_SOURCE_ROOT}" rev-parse HEAD)" = "${FM_SOURCE_COMMIT}"
test -z "$(git -C "${FM_SOURCE_ROOT}" status --porcelain)"

RUNNER=${FM_SOURCE_ROOT}/dreamer_imf_comparison/scripts/run_flowmpc_actor_study.py
VERIFIER=${FM_SOURCE_ROOT}/dreamer_imf_comparison/scripts/verify_flowmpc_actor_study.py
LIBRARY_VERIFIER=${FM_SOURCE_ROOT}/imf_dreamer_jax/scripts/verify_flowmpc.py
COMMON=(
  --reward-root "${FM_REWARD_ROOT}"
  --output-root "${FM_OUTPUT_ROOT}"
  --rebrac-updates "${FM_REBRAC_UPDATES:-1000000}"
)

case "${FM_STAGE}" in
  preflight)
    python "${RUNNER}" manifest "${COMMON[@]}"
    python "${VERIFIER}" --self-test
    python "${VERIFIER}" --implementation
    python "${VERIFIER}" --gpu
    python "${LIBRARY_VERIFIER}" --rebrac
    python "${LIBRARY_VERIFIER}" --controller
    echo FLOWMPC_ACTOR_PREFLIGHT_VERIFIED
    ;;
  rebrac)
    : "${SLURM_ARRAY_TASK_ID:?ReBRAC stage requires index 0..5}"
    python "${RUNNER}" rebrac "${COMMON[@]}" --index "${SLURM_ARRAY_TASK_ID}"
    python "${RUNNER}" verify-rebrac --output-root "${FM_OUTPUT_ROOT}" \
      --index "${SLURM_ARRAY_TASK_ID}"
    ;;
  tune)
    python "${RUNNER}" tune "${COMMON[@]}"
    python "${RUNNER}" verify-tune --output-root "${FM_OUTPUT_ROOT}"
    ;;
  evaluate)
    : "${SLURM_ARRAY_TASK_ID:?evaluation stage requires index 0..5}"
    python "${RUNNER}" evaluate "${COMMON[@]}" --index "${SLURM_ARRAY_TASK_ID}"
    python "${RUNNER}" verify-evaluate --output-root "${FM_OUTPUT_ROOT}" \
      --index "${SLURM_ARRAY_TASK_ID}"
    ;;
  finalize)
    python "${RUNNER}" finalize --output-root "${FM_OUTPUT_ROOT}"
    python "${VERIFIER}" --final --output-root "${FM_OUTPUT_ROOT}"
    echo FLOWMPC_ACTOR_STUDY_FINAL_VERIFIED
    ;;
  *)
    echo "unknown FM_STAGE=${FM_STAGE}" >&2
    exit 2
    ;;
esac
