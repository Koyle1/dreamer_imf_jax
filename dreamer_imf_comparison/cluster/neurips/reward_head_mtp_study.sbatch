#!/usr/bin/env bash
#SBATCH --job-name=imf-reward-mtp
#SBATCH --account=dep_inin_dat
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.out
#SBATCH --error=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.err

set -euo pipefail
: "${MTP_SOURCE_ROOT:?MTP_SOURCE_ROOT is required}"
: "${MTP_OUTPUT_ROOT:?MTP_OUTPUT_ROOT is required}"
: "${MTP_BASELINE_ROOT:?MTP_BASELINE_ROOT is required}"
: "${MTP_TRAIN_PROBE_ROOT:?MTP_TRAIN_PROBE_ROOT is required}"
: "${MTP_SOURCE_COMMIT:?MTP_SOURCE_COMMIT is required}"
: "${MTP_STAGE:?MTP_STAGE is required}"

WORK_ROOT=/work2/ci72buri-dreamer_imf_neurips
module purge
module load Python/3.12.3-GCCcore-13.3.0
source "${WORK_ROOT}/venv-cuda12/bin/activate"
export PYTHONPATH="${MTP_SOURCE_ROOT}:${MTP_SOURCE_ROOT}/imf_dreamer_jax/src:${MTP_SOURCE_ROOT}/dreamer_imf_comparison"
export JAX_ENABLE_X64=0
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="${WORK_ROOT}/jax-compilation-cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_RAISE_PERSISTENT_CACHE_ERRORS=true
export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1

test "$(git -C "${MTP_SOURCE_ROOT}" rev-parse HEAD)" = "${MTP_SOURCE_COMMIT}"
test -z "$(git -C "${MTP_SOURCE_ROOT}" status --porcelain)"

RUNNER=${MTP_SOURCE_ROOT}/dreamer_imf_comparison/scripts/run_reward_head_mtp_study.py
COMMON=(
  --baseline-root "${MTP_BASELINE_ROOT}"
  --train-probe-root "${MTP_TRAIN_PROBE_ROOT}"
  --output-root "${MTP_OUTPUT_ROOT}"
  --reward-updates "${MTP_REWARD_UPDATES:-1000}"
  --actor-updates "${MTP_ACTOR_UPDATES:-3000}"
  --preparation-updates "${MTP_PREPARATION_UPDATES:-500}"
  --reward-bins "${MTP_REWARD_BINS:-255}"
  --evaluation-episodes "${MTP_EVALUATION_EPISODES:-5}"
)
SEEDS=(211 223)
ARMS=(one_step_mse dreamer4_twohot_mtp counterfactual_advantage twohot_mtp_advantage)
HORIZONS=(5 15)

case "${MTP_STAGE}" in
  preflight)
    python "${RUNNER}" manifest "${COMMON[@]}"
    python "${MTP_SOURCE_ROOT}/imf_dreamer_jax/tests/test_reward_head_mtp.py"
    python "${MTP_SOURCE_ROOT}/dreamer_imf_comparison/tests/test_reward_head_mtp_study.py"
    python - <<'PY'
import jax
assert len(jax.devices()) == 1 and jax.devices()[0].platform == "gpu"
print("REWARD_HEAD_MTP_PREFLIGHT_VERIFIED")
PY
    ;;
  reward|evaluation)
    : "${SLURM_ARRAY_TASK_ID:?reward/evaluation requires index 0..7}"
    SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
    ARM_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
    python "${RUNNER}" "${MTP_STAGE}" "${COMMON[@]}" \
      --seed "${SEEDS[${SEED_INDEX}]}" --arm "${ARMS[${ARM_INDEX}]}"
    ;;
  actor)
    : "${SLURM_ARRAY_TASK_ID:?actor requires index 0..15}"
    SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 8))
    REMAINDER=$((SLURM_ARRAY_TASK_ID % 8))
    ARM_INDEX=$((REMAINDER / 2))
    HORIZON_INDEX=$((REMAINDER % 2))
    python "${RUNNER}" actor "${COMMON[@]}" \
      --seed "${SEEDS[${SEED_INDEX}]}" \
      --arm "${ARMS[${ARM_INDEX}]}" \
      --horizon "${HORIZONS[${HORIZON_INDEX}]}"
    ;;
  finalize)
    python "${RUNNER}" finalize "${COMMON[@]}"
    ;;
  *)
    echo "unknown MTP_STAGE=${MTP_STAGE}" >&2
    exit 2
    ;;
esac
