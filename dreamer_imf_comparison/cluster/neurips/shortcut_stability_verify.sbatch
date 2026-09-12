#!/usr/bin/env bash
#SBATCH --job-name=shortcut-stability-verify
#SBATCH --account=dep_inin_dat
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%j.out
#SBATCH --error=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%j.err

set -euo pipefail
WORK_ROOT=/work2/ci72buri-dreamer_imf_neurips
SOURCE_ROOT=${WORK_ROOT}/source
source "${SOURCE_ROOT}/dreamer_imf_comparison/cluster/neurips/runtime_environment.sh"
python "${SOURCE_ROOT}/dreamer_imf_comparison/scripts/run_shortcut_stability_study.py" aggregate \
  --output "${SHORTCUT_STABILITY_OUTPUT:?SHORTCUT_STABILITY_OUTPUT is required}"
python "${SOURCE_ROOT}/dreamer_imf_comparison/scripts/verify_shortcut_stability_study.py" \
  --input "${SHORTCUT_STABILITY_OUTPUT}/summary.json"
