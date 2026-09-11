#!/bin/bash
# Read-only operational verifier. This script is intentionally invoked via SSH.
set -euo pipefail

MODE="${1:-}"
case "$MODE" in
  preflight|pilot|confirmatory|controls|pixel) ;;
  *)
    echo "usage: verify_remote_gate.sh {preflight|pilot|confirmatory|controls|pixel}" >&2
    exit 2
    ;;
esac

WORKSPACE_ROOT=/work2/ci72buri-dreamer_imf_neurips
SOURCE_ROOT="$WORKSPACE_ROOT/source"
PYTHON="$WORKSPACE_ROOT/venv-cuda12/bin/python"
CONTRACT="$WORKSPACE_ROOT/cluster_state/run_contract.json"
VERIFIER="$SOURCE_ROOT/dreamer_imf_comparison/cluster/neurips/verify_cluster_evidence.py"

test -x "$PYTHON"
test -f "$CONTRACT"
test ! -L "$CONTRACT"
test -f "$VERIFIER"
test ! -L "$VERIFIER"

SOURCE_COMMIT="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
test -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=normal)"
CONTRACT_COMMIT="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["source_commit"])' "$CONTRACT")"
test "$SOURCE_COMMIT" = "$CONTRACT_COMMIT"

exec "$PYTHON" "$VERIFIER" "--$MODE"
