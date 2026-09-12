#!/usr/bin/env bash
set -euo pipefail

COMMIT=$(git rev-parse HEAD)
SHORT_COMMIT=${COMMIT:0:12}
REMOTE_SOURCE=/work2/ci72buri-dreamer_imf_neurips/causal-source-${SHORT_COMMIT}
REMOTE_OUTPUT=/work2/ci72buri-dreamer_imf_neurips/causal-reacher-small/${COMMIT}
PILOT=/work2/ci72buri-dreamer_imf_neurips/results/matched_objective_pilot
ssh login01 env \
  PYTHONPATH="${REMOTE_SOURCE}/imf_dreamer_jax/src:${REMOTE_SOURCE}/dreamer_imf_comparison" \
  /work2/ci72buri-dreamer_imf_neurips/venv-cuda12/bin/python \
  "${REMOTE_SOURCE}/dreamer_imf_comparison/scripts/run_causal_reacher_study.py" \
  verify-manifest --pilot-root "${PILOT}" --output "${REMOTE_OUTPUT}"
