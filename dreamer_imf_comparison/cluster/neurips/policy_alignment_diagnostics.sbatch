#!/usr/bin/env bash
#SBATCH --job-name=imf-policy-alignment
#SBATCH --account=dep_inin_dat
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work2/ci72buri-dreamer_imf_neurips/diagnostic-logs/%x-%j.out
#SBATCH --error=/work2/ci72buri-dreamer_imf_neurips/diagnostic-logs/%x-%j.err

set -euo pipefail

WORK_ROOT=/work2/ci72buri-dreamer_imf_neurips
SOURCE_ROOT=${WORK_ROOT}/diagnostic-source-456d449f
ENV_ROOT=${WORK_ROOT}/venv-cuda12
DIAGNOSTIC_COMMIT=$(git -C "${SOURCE_ROOT}" rev-parse HEAD)
PILOT_COMMIT=456d449f440456254e2927211c2f3966d4181e2e
OUTPUT_ROOT=${WORK_ROOT}/policy_alignment_diagnostics/${DIAGNOSTIC_COMMIT}

[[ -z "$(git -C "${SOURCE_ROOT}" status --porcelain)" ]]
[[ -f "${WORK_ROOT}/results/matched_objective_pilot/hpo_selection.json" ]]

module purge
module load Python/3.12.3-GCCcore-13.3.0
source "${ENV_ROOT}/bin/activate"

export PYTHONPATH="${SOURCE_ROOT}/imf_dreamer_jax/src:${SOURCE_ROOT}/dreamer_imf_comparison"
export JAX_ENABLE_X64=0
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="${OUTPUT_ROOT}/jax-compilation-cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=xla_gpu_per_fusion_autotune_cache_dir
export JAX_RAISE_PERSISTENT_CACHE_ERRORS=true
export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "${WORK_ROOT}/diagnostic-logs" "${OUTPUT_ROOT}"

python "${SOURCE_ROOT}/dreamer_imf_comparison/scripts/run_policy_alignment_diagnostics.py" run \
  --pilot-root "${WORK_ROOT}/results/matched_objective_pilot" \
  --pilot-source-commit "${PILOT_COMMIT}" \
  --output "${OUTPUT_ROOT}/report.json" \
  --cells-per-group 6 \
  --states-per-cell 4 \
  --episode -1 \
  --horizon 30 \
  --action-persistence 30 \
  --draws 8 \
  --action-delta 0.25 \
  --finite-difference-epsilon 0.05 \
  --discount 0.99
