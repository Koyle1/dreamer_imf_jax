#!/usr/bin/env bash
#SBATCH --job-name=imf-neurips-finalize
#SBATCH --account=dep_inin_dat
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%j.out
#SBATCH --error=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%j.err

set -euo pipefail
: "${NEURIPS_PROFILE:?NEURIPS_PROFILE must be exported by submit.py}"

WORK_ROOT=/work2/ci72buri-dreamer_imf_neurips
CLUSTER_DIR=${WORK_ROOT}/source/dreamer_imf_comparison/cluster/neurips
source "${CLUSTER_DIR}/runtime_environment.sh"

python "${CLUSTER_DIR}/workflow.py" finalize --profile "${NEURIPS_PROFILE}"

