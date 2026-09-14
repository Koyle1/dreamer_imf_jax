#!/usr/bin/env bash
#SBATCH --job-name=imf-actor-diagnostic
#SBATCH --account=dep_inin_dat
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.out
#SBATCH --error=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.err

set -euo pipefail
: "${AFD_SOURCE_ROOT:?AFD_SOURCE_ROOT is required}"
: "${AFD_OUTPUT_ROOT:?AFD_OUTPUT_ROOT is required}"
: "${AFD_MTP_ROOT:?AFD_MTP_ROOT is required}"
: "${AFD_RESIDUAL_ROOT:?AFD_RESIDUAL_ROOT is required}"
: "${AFD_SOURCE_COMMIT:?AFD_SOURCE_COMMIT is required}"
: "${AFD_STAGE:?AFD_STAGE is required}"

WORK_ROOT=/work2/ci72buri-dreamer_imf_neurips
module purge
module load Python/3.12.3-GCCcore-13.3.0
source "${WORK_ROOT}/venv-cuda12/bin/activate"
export PYTHONPATH="${AFD_SOURCE_ROOT}:${AFD_SOURCE_ROOT}/imf_dreamer_jax/src:${AFD_SOURCE_ROOT}/dreamer_imf_comparison"
export JAX_ENABLE_X64=0
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="${WORK_ROOT}/jax-compilation-cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_RAISE_PERSISTENT_CACHE_ERRORS=true
export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1

test "$(git -C "${AFD_SOURCE_ROOT}" rev-parse HEAD)" = "${AFD_SOURCE_COMMIT}"
test -z "$(git -C "${AFD_SOURCE_ROOT}" status --porcelain)"

RUNNER=${AFD_SOURCE_ROOT}/dreamer_imf_comparison/scripts/run_actor_failure_diagnostic.py
VERIFIER=${AFD_SOURCE_ROOT}/dreamer_imf_comparison/scripts/verify_actor_failure_diagnostic.py
COMMON=(
  --mtp-root "${AFD_MTP_ROOT}"
  --residual-root "${AFD_RESIDUAL_ROOT}"
  --output-root "${AFD_OUTPUT_ROOT}"
  --actor-updates "${AFD_ACTOR_UPDATES:-3000}"
  --preparation-updates "${AFD_PREPARATION_UPDATES:-500}"
  --evaluation-episodes "${AFD_EVALUATION_EPISODES:-50}"
)

case "${AFD_STAGE}" in
  preflight)
    python "${VERIFIER}" library
    python "${VERIFIER}" contract
    python "${VERIFIER}" self-test
    python "${RUNNER}" preflight "${COMMON[@]}"
    python - <<'PY'
import jax
assert len(jax.devices()) == 1 and jax.devices()[0].platform == "gpu"
print("ACTOR_FAILURE_GPU_PREFLIGHT_VERIFIED")
PY
    ;;
  random)
    python "${RUNNER}" random "${COMMON[@]}"
    ;;
  actor)
    : "${SLURM_ARRAY_TASK_ID:?actor requires array index 1..48}"
    python "${RUNNER}" actor "${COMMON[@]}" --index "${SLURM_ARRAY_TASK_ID}"
    ;;
  finalize)
    python "${RUNNER}" finalize "${COMMON[@]}"
    python "${VERIFIER}" final --evidence-root "${AFD_OUTPUT_ROOT}"
    python "${VERIFIER}" decision --evidence-root "${AFD_OUTPUT_ROOT}"
    ;;
  *)
    echo "unknown AFD_STAGE=${AFD_STAGE}" >&2
    exit 2
    ;;
esac
