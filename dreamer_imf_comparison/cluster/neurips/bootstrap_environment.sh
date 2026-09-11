#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=/work2/ci72buri-dreamer_imf_neurips
SOURCE_ROOT=${WORK_ROOT}/source
ENV_ROOT=${WORK_ROOT}/venv-cuda12
LOCK_FILE=${SOURCE_ROOT}/dreamer_imf_comparison/requirements-neurips-cuda12-lock.txt
LOCK_MARKER=${ENV_ROOT}/.neurips-lock-sha256

module purge
module load Python/3.12.3-GCCcore-13.3.0

[[ "$(python -c 'import platform; print(platform.python_version())')" == "3.12.3" ]]
[[ -f "${LOCK_FILE}" ]]
LOCK_SHA256=$(sha256sum "${LOCK_FILE}" | cut -d' ' -f1)
[[ "${LOCK_SHA256}" =~ ^[0-9a-f]{64}$ ]]

if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
  [[ ! -e "${ENV_ROOT}" ]]
  python -m venv "${ENV_ROOT}"
  "${ENV_ROOT}/bin/python" -m pip install \
    --disable-pip-version-check \
    --require-hashes \
    --only-binary=:all: \
    --requirement "${LOCK_FILE}"
  "${ENV_ROOT}/bin/python" -m pip check
  printf '%s\n' "${LOCK_SHA256}" > "${LOCK_MARKER}"
else
  [[ -f "${LOCK_MARKER}" ]]
  [[ "$(<"${LOCK_MARKER}")" == "${LOCK_SHA256}" ]]
fi

"${ENV_ROOT}/bin/python" -m pip check
