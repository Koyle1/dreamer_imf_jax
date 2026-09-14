#!/usr/bin/env bash
#SBATCH --job-name=imf-policy-vnext
#SBATCH --account=dep_inin_dat
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.out
#SBATCH --error=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.err

set -euo pipefail
: "${VNEXT_SOURCE_ROOT:?VNEXT_SOURCE_ROOT is required}"
: "${VNEXT_OUTPUT_ROOT:?VNEXT_OUTPUT_ROOT is required}"
: "${VNEXT_PILOT_ROOT:?VNEXT_PILOT_ROOT is required}"
: "${VNEXT_CAUSAL_ROOT:?VNEXT_CAUSAL_ROOT is required}"
: "${VNEXT_SOURCE_COMMIT:?VNEXT_SOURCE_COMMIT is required}"
: "${VNEXT_STAGE:?VNEXT_STAGE is required}"

WORK_ROOT=/work2/ci72buri-dreamer_imf_neurips
ENV_ROOT=${WORK_ROOT}/venv-cuda12
module purge
module load Python/3.12.3-GCCcore-13.3.0
source "${ENV_ROOT}/bin/activate"

export PYTHONPATH="${VNEXT_SOURCE_ROOT}:${VNEXT_SOURCE_ROOT}/imf_dreamer_jax/src:${VNEXT_SOURCE_ROOT}/dreamer_imf_comparison"
export JAX_ENABLE_X64=0
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="${WORK_ROOT}/jax-compilation-cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=xla_gpu_per_fusion_autotune_cache_dir
export JAX_RAISE_PERSISTENT_CACHE_ERRORS=true
export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1

ACTUAL_COMMIT=$(git -C "${VNEXT_SOURCE_ROOT}" rev-parse HEAD)
test "${ACTUAL_COMMIT}" = "${VNEXT_SOURCE_COMMIT}"
test -z "$(git -C "${VNEXT_SOURCE_ROOT}" status --porcelain)"

ACTOR_UPDATES=${VNEXT_ACTOR_UPDATES:-3000}
PREPARATION_UPDATES=${VNEXT_PREPARATION_UPDATES:-500}
EPISTEMIC_SCALES=${VNEXT_EPISTEMIC_SCALES:-0,1}
RUNNER=${VNEXT_SOURCE_ROOT}/dreamer_imf_comparison/scripts/run_policy_consistency_vnext.py
COMMON=(
  --pilot-root "${VNEXT_PILOT_ROOT}"
  --causal-root "${VNEXT_CAUSAL_ROOT}"
  --output-root "${VNEXT_OUTPUT_ROOT}"
  --actor-updates "${ACTOR_UPDATES}"
  --preparation-updates "${PREPARATION_UPDATES}"
  --epistemic-scales "${EPISTEMIC_SCALES}"
)

case "${VNEXT_STAGE}" in
  preflight)
    python "${RUNNER}" manifest "${COMMON[@]}"
    python "${VNEXT_SOURCE_ROOT}/dreamer_imf_comparison/scripts/verify_policy_consistency_vnext.py"
    python "${VNEXT_SOURCE_ROOT}/dreamer_imf_comparison/scripts/verify_policy_consistency_vnext_docs.py"
    python - <<'PY'
import jax
assert len(jax.devices()) == 1
assert jax.devices()[0].platform == "gpu"
print("VNEXT_PREFLIGHT_VERIFIED")
PY
    ;;
  probe-bank)
    : "${SLURM_ARRAY_TASK_ID:?probe-bank requires array index 0..1}"
    SEEDS=(211 223)
    python "${RUNNER}" probe-bank "${COMMON[@]}" --seed "${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
    ;;
  probe)
    : "${SLURM_ARRAY_TASK_ID:?probe requires array index 0..5}"
    SEEDS=(211 223)
    ARMS=(shortcut_forcing trajectory_imf causal_trajectory_imf)
    SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
    ARM_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
    python "${RUNNER}" probe "${COMMON[@]}" \
      --seed "${SEEDS[${SEED_INDEX}]}" --arm "${ARMS[${ARM_INDEX}]}"
    ;;
  actor)
    : "${SLURM_ARRAY_TASK_ID:?actor requires array index 0..11}"
    : "${VNEXT_EPISTEMIC_SCALE:?actor requires VNEXT_EPISTEMIC_SCALE}"
    SEEDS=(211 223)
    ARMS=(shortcut_forcing trajectory_imf causal_trajectory_imf)
    HORIZONS=(5 15)
    SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 6))
    WITHIN=$((SLURM_ARRAY_TASK_ID % 6))
    ARM_INDEX=$((WITHIN / 2))
    HORIZON_INDEX=$((WITHIN % 2))
    python "${RUNNER}" actor "${COMMON[@]}" \
      --seed "${SEEDS[${SEED_INDEX}]}" \
      --arm "${ARMS[${ARM_INDEX}]}" \
      --horizon "${HORIZONS[${HORIZON_INDEX}]}" \
      --epistemic-scale "${VNEXT_EPISTEMIC_SCALE}"
    ;;
  finalize)
    python "${RUNNER}" finalize --output-root "${VNEXT_OUTPUT_ROOT}"
    ;;
  *)
    echo "unknown VNEXT_STAGE=${VNEXT_STAGE}" >&2
    exit 2
    ;;
esac
