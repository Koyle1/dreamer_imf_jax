#!/usr/bin/env bash
#SBATCH --job-name=imf-action-residual
#SBATCH --account=dep_inin_dat
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.out
#SBATCH --error=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.err

set -euo pipefail
: "${AR_SOURCE_ROOT:?AR_SOURCE_ROOT is required}"
: "${AR_OUTPUT_ROOT:?AR_OUTPUT_ROOT is required}"
: "${AR_MTP_ROOT:?AR_MTP_ROOT is required}"
: "${AR_SOURCE_COMMIT:?AR_SOURCE_COMMIT is required}"
: "${AR_STAGE:?AR_STAGE is required}"

WORK_ROOT=/work2/ci72buri-dreamer_imf_neurips
module purge
module load Python/3.12.3-GCCcore-13.3.0
source "${WORK_ROOT}/venv-cuda12/bin/activate"
export PYTHONPATH="${AR_SOURCE_ROOT}:${AR_SOURCE_ROOT}/imf_dreamer_jax/src:${AR_SOURCE_ROOT}/dreamer_imf_comparison"
export JAX_ENABLE_X64=0
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="${WORK_ROOT}/jax-compilation-cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_RAISE_PERSISTENT_CACHE_ERRORS=true
export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1

test "$(git -C "${AR_SOURCE_ROOT}" rev-parse HEAD)" = "${AR_SOURCE_COMMIT}"
test -z "$(git -C "${AR_SOURCE_ROOT}" status --porcelain)"

RUNNER=${AR_SOURCE_ROOT}/dreamer_imf_comparison/scripts/run_action_reward_residual_study.py
COMMON=(
  --mtp-root "${AR_MTP_ROOT}"
  --output-root "${AR_OUTPUT_ROOT}"
  --residual-updates "${AR_RESIDUAL_UPDATES:-1000}"
  --actor-updates "${AR_ACTOR_UPDATES:-3000}"
  --preparation-updates "${AR_PREPARATION_UPDATES:-500}"
  --evaluation-episodes "${AR_EVALUATION_EPISODES:-5}"
)
SEEDS=(211 223)
HORIZONS=(5 15)

case "${AR_STAGE}" in
  preflight)
    python "${RUNNER}" manifest "${COMMON[@]}"
    python -m unittest discover -s "${AR_SOURCE_ROOT}/imf_dreamer_jax/tests" -p 'test_*.py'
    echo FINAL_ACTION_RESIDUAL_LIBRARY_SUITE_VERIFIED
    python -m unittest discover -s "${AR_SOURCE_ROOT}/dreamer_imf_comparison/tests" -p 'test_*.py'
    echo FINAL_ACTION_RESIDUAL_COMPARISON_SUITE_VERIFIED
    python "${RUNNER}" preflight "${COMMON[@]}"
    echo ACTION_REWARD_RESIDUAL_PREFLIGHT_COMPLETE
    ;;
  residual|evaluation)
    : "${SLURM_ARRAY_TASK_ID:?residual/evaluation requires index 0..1}"
    python "${RUNNER}" "${AR_STAGE}" "${COMMON[@]}" \
      --seed "${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
    ;;
  actor)
    : "${SLURM_ARRAY_TASK_ID:?actor requires index 0..3}"
    SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
    HORIZON_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
    python "${RUNNER}" actor "${COMMON[@]}" \
      --seed "${SEEDS[${SEED_INDEX}]}" \
      --horizon "${HORIZONS[${HORIZON_INDEX}]}"
    ;;
  finalize)
    python "${RUNNER}" finalize "${COMMON[@]}"
    ;;
  *)
    echo "unknown AR_STAGE=${AR_STAGE}" >&2
    exit 2
    ;;
esac
