#!/usr/bin/env bash
#SBATCH --job-name=imf-neurips-diagnostics
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
CLUSTER_DIR=${SOURCE_ROOT}/dreamer_imf_comparison/cluster/neurips
OUTPUT=${WORK_ROOT}/results/matched_objective_confirmatory
source "${CLUSTER_DIR}/runtime_environment.sh"

python "${SOURCE_ROOT}/dreamer_imf_comparison/scripts/run_matched_objective_diagnostics.py" \
  --output-root "${OUTPUT}" \
  --workspace "${SOURCE_ROOT}"

