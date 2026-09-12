#!/usr/bin/env bash
#SBATCH --job-name=imf-adv-repair
#SBATCH --account=dep_inin_dat
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.out
#SBATCH --error=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.err

set -euo pipefail
: "${REPAIR_SOURCE_ROOT:?REPAIR_SOURCE_ROOT is required}"
: "${REPAIR_OUTPUT_ROOT:?REPAIR_OUTPUT_ROOT is required}"
: "${REPAIR_BASELINE_ROOT:?REPAIR_BASELINE_ROOT is required}"
: "${REPAIR_SOURCE_COMMIT:?REPAIR_SOURCE_COMMIT is required}"
: "${REPAIR_STAGE:?REPAIR_STAGE is required}"

WORK_ROOT=/work2/ci72buri-dreamer_imf_neurips
module purge
module load Python/3.12.3-GCCcore-13.3.0
source "${WORK_ROOT}/venv-cuda12/bin/activate"
export PYTHONPATH="${REPAIR_SOURCE_ROOT}:${REPAIR_SOURCE_ROOT}/imf_dreamer_jax/src:${REPAIR_SOURCE_ROOT}/dreamer_imf_comparison"
export JAX_ENABLE_X64=0
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="${WORK_ROOT}/jax-compilation-cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_RAISE_PERSISTENT_CACHE_ERRORS=true
export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1

test "$(git -C "${REPAIR_SOURCE_ROOT}" rev-parse HEAD)" = "${REPAIR_SOURCE_COMMIT}"
test -z "$(git -C "${REPAIR_SOURCE_ROOT}" status --porcelain)"

VARIANT=${REPAIR_VARIANT:-advantage_reward}
WORLD_UPDATES=${REPAIR_WORLD_UPDATES:-1000}
ACTOR_UPDATES=${REPAIR_ACTOR_UPDATES:-3000}
PREPARATION_UPDATES=${REPAIR_PREPARATION_UPDATES:-500}
EPISTEMIC_SCALES=${REPAIR_EPISTEMIC_SCALES:-0,1}
RUNNER=${REPAIR_SOURCE_ROOT}/dreamer_imf_comparison/scripts/run_advantage_repair_study.py
COMMON=(
  --baseline-root "${REPAIR_BASELINE_ROOT}"
  --output-root "${REPAIR_OUTPUT_ROOT}"
  --variant "${VARIANT}"
  --world-updates "${WORLD_UPDATES}"
  --actor-updates "${ACTOR_UPDATES}"
  --preparation-updates "${PREPARATION_UPDATES}"
  --epistemic-scales "${EPISTEMIC_SCALES}"
)

case "${REPAIR_STAGE}" in
  preflight)
    python "${RUNNER}" manifest "${COMMON[@]}"
    python "${REPAIR_SOURCE_ROOT}/dreamer_imf_comparison/scripts/verify_policy_consistency_vnext.py"
    python - <<'PY'
import jax
assert len(jax.devices()) == 1 and jax.devices()[0].platform == "gpu"
print("ADVANTAGE_REPAIR_PREFLIGHT_VERIFIED")
PY
    ;;
  train-probe-bank)
    : "${SLURM_ARRAY_TASK_ID:?train-probe-bank requires index 0..1}"
    SEEDS=(211 223)
    python "${RUNNER}" train-probe-bank "${COMMON[@]}" --seed "${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
    ;;
  world)
    : "${SLURM_ARRAY_TASK_ID:?world requires index 0..3}"
    SEEDS=(211 223)
    ARMS=(trajectory_imf causal_trajectory_imf)
    SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
    ARM_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
    python "${RUNNER}" world "${COMMON[@]}" --seed "${SEEDS[${SEED_INDEX}]}" --arm "${ARMS[${ARM_INDEX}]}"
    ;;
  probe)
    : "${SLURM_ARRAY_TASK_ID:?probe requires index 0..3}"
    SEEDS=(211 223)
    ARMS=(trajectory_imf causal_trajectory_imf)
    SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
    ARM_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
    python "${RUNNER}" probe "${COMMON[@]}" --seed "${SEEDS[${SEED_INDEX}]}" --arm "${ARMS[${ARM_INDEX}]}"
    ;;
  actor)
    : "${SLURM_ARRAY_TASK_ID:?actor requires index 0..3}"
    SEEDS=(211 223)
    HORIZONS=(5 15)
    SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
    HORIZON_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
    python "${RUNNER}" actor "${COMMON[@]}" \
      --seed "${SEEDS[${SEED_INDEX}]}" \
      --arm trajectory_imf \
      --horizon "${HORIZONS[${HORIZON_INDEX}]}" \
      --epistemic-scale 0
    ;;
  finalize)
    python "${RUNNER}" finalize "${COMMON[@]}"
    ;;
  *)
    echo "unknown REPAIR_STAGE=${REPAIR_STAGE}" >&2
    exit 2
    ;;
esac
