#!/bin/bash
# One frozen-parent diagnostic; fail closed before advancing beyond preflight.
set -euo pipefail
source_root="$1"
output_root="$2"
parent_root="$3"
expected_commit="$4"
test "$(git -C "$source_root" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$source_root" status --porcelain)"
test ! -e "$output_root"
module purge
module load Python/3.12.3-GCCcore-13.3.0
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$source_root/dreamer_imf_comparison:$source_root/imf_dreamer_jax/src:/work2/ci72buri-dreamer_imf_neurips/dreamerv3-upstream-e3f0224"
python_bin=/work2/ci72buri-dreamer_imf_neurips/venv-dreamer-ablation-jax0433/bin/python
"$python_bin" -u -m dreamer_imf_compare.prior_correction_study preflight --parent "$parent_root" --output "$output_root/preflight"
"$python_bin" -u -m dreamer_imf_compare.prior_correction_study verify --output "$output_root/preflight"
"$python_bin" -u -m dreamer_imf_compare.prior_correction_study run --parent "$parent_root" --output "$output_root/fit"
"$python_bin" -u -m dreamer_imf_compare.prior_correction_study verify --output "$output_root/fit"
echo PRIOR_CORRECTION_ALL_STAGES_VERIFIED
