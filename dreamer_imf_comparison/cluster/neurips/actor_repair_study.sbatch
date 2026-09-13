#!/usr/bin/env bash
#SBATCH --job-name=imf-actor-repair
#SBATCH --account=dep_inin_dat
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.out
#SBATCH --error=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.err

set -euo pipefail
: "${ARS_SOURCE_ROOT:?ARS_SOURCE_ROOT is required}"
: "${ARS_OUTPUT_ROOT:?ARS_OUTPUT_ROOT is required}"
: "${ARS_MTP_ROOT:?ARS_MTP_ROOT is required}"
: "${ARS_SOURCE_COMMIT:?ARS_SOURCE_COMMIT is required}"
: "${ARS_STAGE:?ARS_STAGE is required}"

WORK_ROOT=/work2/ci72buri-dreamer_imf_neurips
module purge
module load Python/3.12.3-GCCcore-13.3.0
source "${WORK_ROOT}/venv-cuda12/bin/activate"
export PYTHONPATH="${ARS_SOURCE_ROOT}:${ARS_SOURCE_ROOT}/imf_dreamer_jax/src:${ARS_SOURCE_ROOT}/dreamer_imf_comparison"
export JAX_ENABLE_X64=0
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="${WORK_ROOT}/jax-compilation-cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_RAISE_PERSISTENT_CACHE_ERRORS=true
export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1

test "$(git -C "${ARS_SOURCE_ROOT}" rev-parse HEAD)" = "${ARS_SOURCE_COMMIT}"
test -z "$(git -C "${ARS_SOURCE_ROOT}" status --porcelain)"

RUNNER=${ARS_SOURCE_ROOT}/dreamer_imf_comparison/scripts/run_actor_repair_study.py
VERIFIER=${ARS_SOURCE_ROOT}/dreamer_imf_comparison/scripts/verify_actor_repair_study.py
COMMON=(
  --mtp-root "${ARS_MTP_ROOT}"
  --output-root "${ARS_OUTPUT_ROOT}"
  --actor-updates "${ARS_ACTOR_UPDATES:-1500}"
  --preparation-updates "${ARS_PREPARATION_UPDATES:-500}"
  --evaluation-episodes "${ARS_EVALUATION_EPISODES:-25}"
  --exact-batch-size "${ARS_EXACT_BATCH_SIZE:-256}"
)

case "${ARS_STAGE}" in
  preflight)
    python "${VERIFIER}" contract
    python "${VERIFIER}" self-test
    python "${VERIFIER}" regression
    python "${RUNNER}" manifest "${COMMON[@]}"
    python - <<'PY'
import jax
assert len(jax.devices()) == 1 and jax.devices()[0].platform == "gpu"
print("ACTOR_REPAIR_GPU_PREFLIGHT_VERIFIED")
PY
    ;;
  smoke)
    python "${RUNNER}" smoke --output-root "${ARS_OUTPUT_ROOT}/engineering-smoke"
    python "${VERIFIER}" gradient --output-root "${ARS_OUTPUT_ROOT}/engineering-smoke"
    ;;
  random)
    python "${RUNNER}" random "${COMMON[@]}"
    ;;
  pure)
    : "${SLURM_ARRAY_TASK_ID:?pure stage requires a cell index}"
    python "${RUNNER}" pure "${COMMON[@]}" --index "${SLURM_ARRAY_TASK_ID}"
    ;;
  world)
    : "${SLURM_ARRAY_TASK_ID:?world stage requires a cell index}"
    python "${RUNNER}" world "${COMMON[@]}" --index "${SLURM_ARRAY_TASK_ID}"
    ;;
  finalize)
    python "${RUNNER}" finalize "${COMMON[@]}"
    python "${VERIFIER}" final --output-root "${ARS_OUTPUT_ROOT}"
    ;;
  *)
    echo "unknown ARS_STAGE=${ARS_STAGE}" >&2
    exit 2
    ;;
esac
