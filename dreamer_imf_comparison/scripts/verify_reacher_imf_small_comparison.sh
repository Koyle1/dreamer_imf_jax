#!/usr/bin/env bash
set -euo pipefail

bash dreamer_imf_comparison/scripts/verify_remote_reacher_imf_small.sh >/dev/null
echo REACHER_IMF_SMALL_COMPARISON_VERIFIED
