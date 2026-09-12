#!/usr/bin/env bash
set -euo pipefail

COMMIT=$(git rev-parse HEAD)
REMOTE_OUTPUT=/work2/ci72buri-dreamer_imf_neurips/causal-reacher-small/${COMMIT}
ssh login01 test -f "${REMOTE_OUTPUT}/preflight.json"
ssh login01 grep -q '"focused_and_full_tests_completed": true' "${REMOTE_OUTPUT}/preflight.json"
ssh login01 grep -q '"jax_backend": "gpu"' "${REMOTE_OUTPUT}/preflight.json"
echo CAUSAL_CONSISTENCY_VERIFIED
