#!/usr/bin/env bash
#SBATCH --job-name=imf-causal-preflight
#SBATCH --account=dep_inin_dat
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%j.out
#SBATCH --error=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%j.err

set -euo pipefail
: "${CAUSAL_SOURCE_ROOT:?CAUSAL_SOURCE_ROOT is required}"
: "${CAUSAL_OUTPUT_ROOT:?CAUSAL_OUTPUT_ROOT is required}"
: "${CAUSAL_PILOT_ROOT:?CAUSAL_PILOT_ROOT is required}"
: "${CAUSAL_SOURCE_COMMIT:?CAUSAL_SOURCE_COMMIT is required}"

WORK_ROOT=/work2/ci72buri-dreamer_imf_neurips
ENV_ROOT=${WORK_ROOT}/venv-cuda12
module purge
module load Python/3.12.3-GCCcore-13.3.0
source "${ENV_ROOT}/bin/activate"

export PYTHONPATH="${CAUSAL_SOURCE_ROOT}:${CAUSAL_SOURCE_ROOT}/imf_dreamer_jax/src:${CAUSAL_SOURCE_ROOT}/dreamer_imf_comparison"
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

ACTUAL_COMMIT=$(git -C "${CAUSAL_SOURCE_ROOT}" rev-parse HEAD)
test "${ACTUAL_COMMIT}" = "${CAUSAL_SOURCE_COMMIT}"
test -z "$(git -C "${CAUSAL_SOURCE_ROOT}" status --porcelain)"

RUNNER=${CAUSAL_SOURCE_ROOT}/dreamer_imf_comparison/scripts/run_causal_reacher_study.py
python "${RUNNER}" manifest \
  --pilot-root "${CAUSAL_PILOT_ROOT}" \
  --output "${CAUSAL_OUTPUT_ROOT}"
python "${RUNNER}" verify-manifest \
  --pilot-root "${CAUSAL_PILOT_ROOT}" \
  --output "${CAUSAL_OUTPUT_ROOT}"

python -m unittest discover -q \
  -s "${CAUSAL_SOURCE_ROOT}/imf_dreamer_jax/tests" \
  -p "test_trajectory_world_model.py"
python -m unittest discover -q \
  -s "${CAUSAL_SOURCE_ROOT}/dreamer_imf_comparison/tests" \
  -p "test_policy_alignment_diagnostics.py"
python -m unittest discover -q \
  -s "${CAUSAL_SOURCE_ROOT}/dreamer_imf_comparison/tests" \
  -p "test_causal_reacher_study.py"
python -m unittest discover -q -s "${CAUSAL_SOURCE_ROOT}/imf_dreamer_jax/tests"
python -m unittest discover -q -s "${CAUSAL_SOURCE_ROOT}/dreamer_imf_comparison/tests"

python "${RUNNER}" preflight \
  --pilot-root "${CAUSAL_PILOT_ROOT}" \
  --output "${CAUSAL_OUTPUT_ROOT}"
