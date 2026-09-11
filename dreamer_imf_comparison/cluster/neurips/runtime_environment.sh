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
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR="${WORK_ROOT}/jax-compilation-cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=xla_gpu_per_fusion_autotune_cache_dir
export JAX_RAISE_PERSISTENT_CACHE_ERRORS=true
export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "${JAX_COMPILATION_CACHE_DIR}"
[[ -d "${JAX_COMPILATION_CACHE_DIR}" ]]
[[ -w "${JAX_COMPILATION_CACHE_DIR}" ]]

[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]
[[ "${CUDA_VISIBLE_DEVICES}" != *,* ]]
