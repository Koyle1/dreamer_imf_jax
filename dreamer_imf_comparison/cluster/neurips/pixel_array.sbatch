#!/usr/bin/env bash
#SBATCH --job-name=imf-pixel-cell
#SBATCH --account=dep_inin_dat
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.out
#SBATCH --error=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.err

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?pixel worker requires an array index}"

WORK_ROOT=/work2/ci72buri-dreamer_imf_neurips
CLUSTER_DIR=${WORK_ROOT}/source/dreamer_imf_comparison/cluster/neurips
source "${CLUSTER_DIR}/runtime_environment.sh"

COMMAND=(
  python "${CLUSTER_DIR}/supplementary.py" run-pixel-cell
  --array-index "${SLURM_ARRAY_TASK_ID}"
)
if [[ -n "${NEURIPS_RETRY_MAP:-}" ]]; then
  COMMAND+=(--retry-map "${NEURIPS_RETRY_MAP}")
fi
"${COMMAND[@]}"
