#!/usr/bin/env bash
#SBATCH --job-name=imf-neurips-preflight
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

source "${CLUSTER_DIR}/bootstrap_environment.sh"
source "${CLUSTER_DIR}/runtime_environment.sh"

python "${CLUSTER_DIR}/workflow.py" initialize
python "${CLUSTER_DIR}/gpu_preflight.py"
python "${CLUSTER_DIR}/workflow.py" freeze --profile pilot

