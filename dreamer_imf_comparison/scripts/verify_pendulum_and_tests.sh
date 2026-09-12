#!/usr/bin/env bash
set -euo pipefail

COMMIT=${CAUSAL_RESULT_COMMIT:-$(git rev-parse HEAD)}
REMOTE_OUTPUT=/work2/ci72buri-dreamer_imf_neurips/causal-reacher-small/${COMMIT}
ssh login01 test -f "${REMOTE_OUTPUT}/preflight.json"
ssh login01 "grep -q '\"focused_and_full_tests_completed\": true' '${REMOTE_OUTPUT}/preflight.json'"
git merge-base --is-ancestor f6dccd3 HEAD
echo PENDULUM_AND_TESTS_VERIFIED
