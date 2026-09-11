#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=/work2/ci72buri-dreamer_imf_neurips
SOURCE_ROOT=${WORK_ROOT}/source
ENV_ROOT=${WORK_ROOT}/venv-cuda12

module purge
module load Python/3.12.3-GCCcore-13.3.0
source "${ENV_ROOT}/bin/activate"

export PYTHONPATH="${SOURCE_ROOT}/imf_dreamer_jax/src:${SOURCE_ROOT}/dreamer_imf_comparison"
export JAX_ENABLE_X64=0
export JAX_PLATFORM_NAME=gpu
export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1

[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]
[[ "${CUDA_VISIBLE_DEVICES}" != *,* ]]

