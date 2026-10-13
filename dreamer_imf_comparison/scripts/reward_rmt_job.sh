#!/bin/bash
set -euo pipefail
source_root="$1"
output_root="$2"
dependency="$3"
expected_commit="$4"
test "$(git -C "$source_root" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$source_root" status --porcelain)"
test -f "$output_root/intent.json"
test ! -e "$output_root/preflight"
test ! -e "$output_root/full"
module purge
module load Python/3.12.3-GCCcore-13.3.0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$source_root/dreamer_imf_comparison:$source_root/imf_dreamer_jax/src:/work2/ci72buri-dreamer_imf_neurips/dreamerv3-upstream-e3f0224"
python_bin=/work2/ci72buri-dreamer_imf_neurips/venv-dreamer-ablation-jax0433/bin/python
"$python_bin" -u -m dreamer_imf_compare.reward_rmt_study prepare --preflight --dependency "$dependency" --output "$output_root/preflight"
"$python_bin" -u -m dreamer_imf_compare.reward_rmt_study fit --output "$output_root/preflight"
"$python_bin" -u -m dreamer_imf_compare.reward_rmt_study evaluate --output "$output_root/preflight"
"$python_bin" -u -m dreamer_imf_compare.reward_rmt_study verify --output "$output_root/preflight"
echo REWARD_RMT_PREFLIGHT_VERIFIED
"$python_bin" -u -m dreamer_imf_compare.reward_rmt_study prepare --dependency "$dependency" --output "$output_root/full"
"$python_bin" -u -m dreamer_imf_compare.reward_rmt_study fit --output "$output_root/full"
"$python_bin" -u -m dreamer_imf_compare.reward_rmt_study evaluate --output "$output_root/full"
"$python_bin" -u -m dreamer_imf_compare.reward_rmt_study verify --output "$output_root/full"
echo REWARD_RMT_ALL_STAGES_VERIFIED
