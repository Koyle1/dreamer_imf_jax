#!/usr/bin/env bash
#SBATCH --job-name=imf-actor-gap-roadmap
#SBATCH --account=dep_inin_dat
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.out
#SBATCH --error=/work2/ci72buri-dreamer_imf_neurips/logs/%x-%A_%a.err

set -euo pipefail
: "${AGR_SOURCE_ROOT:?AGR_SOURCE_ROOT is required}"
: "${AGR_DEPENDENCY_ROOT:?AGR_DEPENDENCY_ROOT is required}"
: "${AGR_OUTPUT_ROOT:?AGR_OUTPUT_ROOT is required}"
: "${AGR_SOURCE_COMMIT:?AGR_SOURCE_COMMIT is required}"
: "${AGR_STAGE:?AGR_STAGE is required}"

WORK_ROOT=/work2/ci72buri-dreamer_imf_neurips
module purge
module load Python/3.12.3-GCCcore-13.3.0
source "${WORK_ROOT}/venv-cuda12/bin/activate"
export PYTHONPATH="${AGR_SOURCE_ROOT}:${AGR_SOURCE_ROOT}/imf_dreamer_jax/src:${AGR_SOURCE_ROOT}/dreamer_imf_comparison"
export JAX_ENABLE_X64=0
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_COMPILATION_CACHE=true
ACTOR_GAP_CACHE_ROOT="${WORK_ROOT}/jax-compilation-cache/actor-gap-roadmap/${AGR_SOURCE_COMMIT}"
if [[ "${AGR_STAGE}" == "evaluation-cell" ]]; then
  : "${SLURM_JOB_ID:?evaluation-cell requires SLURM_JOB_ID}"
  : "${SLURM_ARRAY_TASK_ID:?evaluation-cell requires SLURM_ARRAY_TASK_ID}"
  JAX_COMPILATION_CACHE_DIR="${ACTOR_GAP_CACHE_ROOT}/job-${SLURM_JOB_ID}-task-${SLURM_ARRAY_TASK_ID}"
  if [[ -e "${JAX_COMPILATION_CACHE_DIR}" ]]; then
    echo "refusing reused evaluation cache ${JAX_COMPILATION_CACHE_DIR}" >&2
    exit 2
  fi
  mkdir -p "${JAX_COMPILATION_CACHE_DIR}"
else
  JAX_COMPILATION_CACHE_DIR="${ACTOR_GAP_CACHE_ROOT}/shared"
  mkdir -p "${JAX_COMPILATION_CACHE_DIR}"
fi
export JAX_COMPILATION_CACHE_DIR
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_RAISE_PERSISTENT_CACHE_ERRORS=true
export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1

test "$(git -C "${AGR_SOURCE_ROOT}" rev-parse HEAD)" = "${AGR_SOURCE_COMMIT}"
test -z "$(git -C "${AGR_SOURCE_ROOT}" status --porcelain)"

RUNNER=${AGR_SOURCE_ROOT}/dreamer_imf_comparison/scripts/run_actor_gap_roadmap_study.py
VERIFIER=${AGR_SOURCE_ROOT}/dreamer_imf_comparison/scripts/verify_actor_gap_roadmap_study.py
CACHE_FINGERPRINTER=${AGR_SOURCE_ROOT}/dreamer_imf_comparison/scripts/fingerprint_jax_cache.py
COMMON=(
  --dependency-root "${AGR_DEPENDENCY_ROOT}"
  --output-root "${AGR_OUTPUT_ROOT}"
  --model-updates "${AGR_MODEL_UPDATES:-5000}"
  --evaluation-episodes "${AGR_EVALUATION_EPISODES:-2}"
)

require_array_index() {
  local upper_bound=$1
  : "${SLURM_ARRAY_TASK_ID:?${AGR_STAGE} requires a Slurm array index}"
  if ((SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= upper_bound)); then
    echo "invalid ${AGR_STAGE} array index ${SLURM_ARRAY_TASK_ID}; expected 0..$((upper_bound - 1))" >&2
    exit 2
  fi
}

submission_arguments() {
  : "${AGR_SUBMISSION_MAP:?${AGR_STAGE} requires AGR_SUBMISSION_MAP}"
  : "${AGR_SUBMISSION_MAP_FILE_SHA256:?${AGR_STAGE} requires AGR_SUBMISSION_MAP_FILE_SHA256}"
}

cache_fingerprint() {
  python "${CACHE_FINGERPRINTER}" --cache-root "${JAX_COMPILATION_CACHE_DIR}"
}

case "${AGR_STAGE}" in
  preflight)
    python "${RUNNER}" preflight "${COMMON[@]}"
    python "${VERIFIER}" self-test --output-root "${AGR_OUTPUT_ROOT}"
    python "${VERIFIER}" implementation --output-root "${AGR_OUTPUT_ROOT}"
    python "${VERIFIER}" gpu --output-root "${AGR_OUTPUT_ROOT}"
    python "${VERIFIER}" library-suite --output-root "${AGR_OUTPUT_ROOT}"
    python "${VERIFIER}" comparison-suite --output-root "${AGR_OUTPUT_ROOT}"
    python "${RUNNER}" seal-preflight "${COMMON[@]}"
    python "${VERIFIER}" preflight --output-root "${AGR_OUTPUT_ROOT}"
    echo ACTOR_GAP_ROADMAP_PREFLIGHT_VERIFIED
    ;;
  verify-preflight)
    python "${VERIFIER}" preflight --output-root "${AGR_OUTPUT_ROOT}"
    echo ACTOR_GAP_ROADMAP_PREFLIGHT_REVERIFIED
    ;;
  calibration)
    python "${RUNNER}" calibration "${COMMON[@]}"
    python "${VERIFIER}" calibration --output-root "${AGR_OUTPUT_ROOT}"
    ;;
  verify-calibration)
    python "${VERIFIER}" calibration --output-root "${AGR_OUTPUT_ROOT}"
    echo ACTOR_GAP_ROADMAP_CALIBRATION_REVERIFIED
    ;;
  diagnostic-cell)
    require_array_index 3
    submission_arguments
    python "${RUNNER}" diagnostic-cell "${COMMON[@]}" \
      --index "${SLURM_ARRAY_TASK_ID}" \
      --submission-map "${AGR_SUBMISSION_MAP}" \
      --submission-stage diagnostic \
      --submission-map-file-sha256 "${AGR_SUBMISSION_MAP_FILE_SHA256}"
    # A fresh Python interpreter is a required part of strict replay: it
    # prevents first-process compiler state from authenticating its own output.
    python "${RUNNER}" verify-diagnostic-cell --output-root "${AGR_OUTPUT_ROOT}" \
      --index "${SLURM_ARRAY_TASK_ID}" \
      --submission-map "${AGR_SUBMISSION_MAP}" \
      --submission-stage diagnostic \
      --submission-map-file-sha256 "${AGR_SUBMISSION_MAP_FILE_SHA256}"
    ;;
  verify-diagnostics)
    python "${RUNNER}" verify-diagnostics --output-root "${AGR_OUTPUT_ROOT}"
    python "${VERIFIER}" diagnostics --output-root "${AGR_OUTPUT_ROOT}"
    ;;
  model-cell)
    require_array_index 21
    submission_arguments
    python "${RUNNER}" model-cell "${COMMON[@]}" \
      --index "${SLURM_ARRAY_TASK_ID}" \
      --submission-map "${AGR_SUBMISSION_MAP}" \
      --submission-stage model \
      --submission-map-file-sha256 "${AGR_SUBMISSION_MAP_FILE_SHA256}"
    python "${RUNNER}" verify-model-cell --output-root "${AGR_OUTPUT_ROOT}" \
      --index "${SLURM_ARRAY_TASK_ID}" \
      --submission-map "${AGR_SUBMISSION_MAP}" \
      --submission-stage model \
      --submission-map-file-sha256 "${AGR_SUBMISSION_MAP_FILE_SHA256}"
    ;;
  verify-models)
    python "${RUNNER}" verify-models --output-root "${AGR_OUTPUT_ROOT}"
    python "${VERIFIER}" models --output-root "${AGR_OUTPUT_ROOT}"
    ;;
  evaluation-cell)
    require_array_index 90
    submission_arguments
    # The first process owns compilation and executes the exact full cell, but
    # publishes nothing. Creation and strict replay are separate cache-reader
    # processes, avoiding a compiler-writer/reader numerical regime mismatch.
    python "${RUNNER}" prime-evaluation-cell --output-root "${AGR_OUTPUT_ROOT}" \
      --index "${SLURM_ARRAY_TASK_ID}" \
      --submission-map "${AGR_SUBMISSION_MAP}" \
      --submission-stage evaluation \
      --submission-map-file-sha256 "${AGR_SUBMISSION_MAP_FILE_SHA256}"
    AGR_EVALUATION_CACHE_FINGERPRINT=$(cache_fingerprint)
    export AGR_EVALUATION_CACHE_FINGERPRINT
    echo "ACTOR_GAP_ROADMAP_PRIMED_CACHE_SHA256=${AGR_EVALUATION_CACHE_FINGERPRINT}"
    python "${RUNNER}" evaluation-cell "${COMMON[@]}" \
      --index "${SLURM_ARRAY_TASK_ID}" \
      --submission-map "${AGR_SUBMISSION_MAP}" \
      --submission-stage evaluation \
      --submission-map-file-sha256 "${AGR_SUBMISSION_MAP_FILE_SHA256}"
    test "$(cache_fingerprint)" = "${AGR_EVALUATION_CACHE_FINGERPRINT}"
    python "${RUNNER}" seal-evaluation-cache-reader --output-root "${AGR_OUTPUT_ROOT}" \
      --index "${SLURM_ARRAY_TASK_ID}" \
      --submission-map "${AGR_SUBMISSION_MAP}" \
      --submission-stage evaluation \
      --submission-map-file-sha256 "${AGR_SUBMISSION_MAP_FILE_SHA256}" \
      --cache-seal-phase creation
    python "${RUNNER}" verify-evaluation-cell --output-root "${AGR_OUTPUT_ROOT}" \
      --index "${SLURM_ARRAY_TASK_ID}" \
      --submission-map "${AGR_SUBMISSION_MAP}" \
      --submission-stage evaluation \
      --submission-map-file-sha256 "${AGR_SUBMISSION_MAP_FILE_SHA256}"
    test "$(cache_fingerprint)" = "${AGR_EVALUATION_CACHE_FINGERPRINT}"
    python "${RUNNER}" seal-evaluation-cache-reader --output-root "${AGR_OUTPUT_ROOT}" \
      --index "${SLURM_ARRAY_TASK_ID}" \
      --submission-map "${AGR_SUBMISSION_MAP}" \
      --submission-stage evaluation \
      --submission-map-file-sha256 "${AGR_SUBMISSION_MAP_FILE_SHA256}" \
      --cache-seal-phase verification
    ;;
  verify-evaluations)
    python "${RUNNER}" verify-evaluations --output-root "${AGR_OUTPUT_ROOT}"
    python "${VERIFIER}" evaluations --output-root "${AGR_OUTPUT_ROOT}"
    ;;
  finalize)
    python "${RUNNER}" finalize --output-root "${AGR_OUTPUT_ROOT}"
    python "${VERIFIER}" final --output-root "${AGR_OUTPUT_ROOT}"
    echo ACTOR_GAP_ROADMAP_STUDY_FINAL_VERIFIED
    ;;
  verify-final)
    python "${VERIFIER}" final --output-root "${AGR_OUTPUT_ROOT}"
    echo ACTOR_GAP_ROADMAP_STUDY_FINAL_REVERIFIED
    ;;
  *)
    echo "unknown AGR_STAGE=${AGR_STAGE}" >&2
    exit 2
    ;;
esac
