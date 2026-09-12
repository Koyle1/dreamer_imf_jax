#!/usr/bin/env bash
#SBATCH --job-name=imf-causal-reacher
#SBATCH --account=dep_inin_dat
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.out
#SBATCH --error=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.err

set -euo pipefail
: "${CAUSAL_SOURCE_ROOT:?CAUSAL_SOURCE_ROOT is required}"
: "${CAUSAL_OUTPUT_ROOT:?CAUSAL_OUTPUT_ROOT is required}"
: "${CAUSAL_PILOT_ROOT:?CAUSAL_PILOT_ROOT is required}"
: "${CAUSAL_STAGE:?CAUSAL_STAGE is required}"

WORK_ROOT=/work2/ci72buri-dreamer_imf_neurips
ENV_ROOT=${WORK_ROOT}/venv-cuda12
module purge
module load Python/3.12.3-GCCcore-13.3.0
source "${ENV_ROOT}/bin/activate"

export PYTHONPATH="${CAUSAL_SOURCE_ROOT}/imf_dreamer_jax/src:${CAUSAL_SOURCE_ROOT}/dreamer_imf_comparison"
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

RUNNER=${CAUSAL_SOURCE_ROOT}/dreamer_imf_comparison/scripts/run_causal_reacher_study.py
case "${CAUSAL_STAGE}" in
  prepare|train|diagnose)
    : "${SLURM_ARRAY_TASK_ID:?array stage requires an index}"
    SEEDS=(211 223)
    SEED=${SEEDS[${SLURM_ARRAY_TASK_ID}]}
    python "${RUNNER}" "${CAUSAL_STAGE}" \
      --pilot-root "${CAUSAL_PILOT_ROOT}" \
      --output "${CAUSAL_OUTPUT_ROOT}" \
      --seed "${SEED}"
    ;;
  finalize)
    : "${CAUSAL_BASELINE_DIAGNOSTICS:?baseline diagnostic report is required}"
    python "${RUNNER}" finalize \
      --pilot-root "${CAUSAL_PILOT_ROOT}" \
      --output "${CAUSAL_OUTPUT_ROOT}" \
      --baseline-diagnostics "${CAUSAL_BASELINE_DIAGNOSTICS}"
    python "${RUNNER}" verify \
      --pilot-root "${CAUSAL_PILOT_ROOT}" \
      --output "${CAUSAL_OUTPUT_ROOT}" \
      --baseline-diagnostics "${CAUSAL_BASELINE_DIAGNOSTICS}"
    ;;
  *)
    echo "unknown CAUSAL_STAGE=${CAUSAL_STAGE}" >&2
    exit 2
    ;;
esac
