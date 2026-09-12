"""Resumable matched-objective benchmark for shortcut forcing and trajectory iMF.

The harness executes the two frozen objective arms inside the shared
``imf_dreamer_jax`` world model.  It deliberately keeps datasets, minibatch
indices, evaluation starts, predictive noise, and environment seeds paired.
World-model seeds are the independent units; actor seeds and episodes remain
nested.  Every persistent identity is bound to both protocol and source
digests, and every result is validated before it may be resumed or analysed.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from functools import partial
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .artifacts import read_json, write_json_atomic
from .dmc import DMCAdapter, split_task
from .matched_objective_protocol import (
    ACTOR_SEEDS,
    ARM_ORDER,
    BUDGET_TRACK_ORDER,
    NFE_FRONTIER,
    PRIMARY_METRICS,
    PROFILE_ORDER,
    TASKS,
    WORLD_MODEL_SEEDS,
    evaluate_superiority,
    protocol_digest,
    validate_matched_objective_protocol,
)


MATRIX_SCHEMA = "matched-objective-matrix-v1"
HPO_MATRIX_SCHEMA = "matched-objective-hpo-matrix-v1"
COMPUTE_SCHEMA = "matched-objective-compute-plan-v1"
DATASET_SCHEMA = "matched-objective-dataset-v1"
WORLD_SCHEMA = "matched-objective-world-model-v1"
ROLLOUT_SCHEMA = "matched-objective-rollout-v1"
ACTOR_SCHEMA = "matched-objective-actor-v1"
ANALYSIS_SCHEMA = "matched-objective-analysis-v1"
ARTIFACT_SCHEMA = "matched-objective-artifacts-v1"
ROLLOUT_REPLAY_ABSOLUTE_TOLERANCE = 1e-6
ROLLOUT_REPLAY_RELATIVE_TOLERANCE = 1e-6
SHORTCUT_PRIOR_LOSS_CEILING = 1_000.0

STAGE_ORDER = ("dataset", "compute_plan", "world_model", "rollout", "actor")
# Execution details absent from the statistical protocol live in one hashed
# harness specification.  Changing any value changes every matrix cell id.
EXECUTION_SPEC: dict[str, Any] = {
    "schema_version": "matched-objective-execution-v1",
    "profiles": {
        "smoke": {
            "batch_size": 2,
            "sequence_length": 32,
            "burn_in": 8,
            "checkpoint_every_updates": 1,
            "schedule_chunk_updates": 1,
        },
        "pilot": {
            "batch_size": 32,
            "sequence_length": 32,
            "burn_in": 8,
            "checkpoint_every_updates": 100,
            "schedule_chunk_updates": 100,
        },
        "confirmatory": {
            "batch_size": 32,
            "sequence_length": 32,
            "burn_in": 8,
            "checkpoint_every_updates": 500,
            "schedule_chunk_updates": 500,
        },
    },
    "dataset": {
        "smoke_episodes": 5,
        "smoke_native_steps_per_episode": 40,
    },
    "world_model_health": {
        "finite_parameters_and_metrics_required": True,
        "shortcut_max_checkpoint_prior_loss": SHORTCUT_PRIOR_LOSS_CEILING,
    },
    "smoke_overrides_are_engineering_only": True,
}

# A run is only source-addressed if every executable and certifying dependency
# exists in the checkout. Globs below may add future tests, but they may not
# silently turn a partial checkout into a valid benchmark source tree.
_LIBRARY_SOURCE_BASENAMES = (
    "__init__.py",
    "agent.py",
    "checkpoint.py",
    "config.py",
    "fidelity.py",
    "imf.py",
    "nn.py",
    "optim.py",
    "planner.py",
    "replay.py",
    "shortcut.py",
    "trajectory.py",
    "types.py",
    "uncertainty.py",
    "world_model.py",
)
_LIBRARY_TEST_BASENAMES = (
    "__init__.py",
    "test_actor_repairs.py",
    "test_agent.py",
    "test_dreamer4_actor_ablations.py",
    "test_dreamer4_dynamics_ablations.py",
    "test_fidelity.py",
    "test_imf.py",
    "test_imf_repairs.py",
    "test_shortcut_forcing.py",
    "test_shortcut_world_model.py",
    "test_trajectory_imf.py",
    "test_trajectory_world_model.py",
    "test_world_model.py",
    "test_world_model_repairs.py",
    "test_z_full_marker.py",
)
_COMPARISON_SOURCE_BASENAMES = (
    "__init__.py",
    "artifacts.py",
    "causal_reacher_study.py",
    "dmc.py",
    "matched_objective_benchmark.py",
    "matched_objective_diagnostics.py",
    "matched_objective_protocol.py",
    "neurips_controls.py",
    "pixel_benchmark.py",
    "policy_alignment_diagnostics.py",
    "protocol.py",
    "shortcut_stability_study.py",
)
_COMPARISON_SCRIPT_BASENAMES = (
    "generate_matched_objective_matrix.py",
    "run_causal_reacher_study.py",
    "run_matched_objective_benchmark.py",
    "run_matched_objective_diagnostics.py",
    "run_neurips_controls.py",
    "run_pixel_benchmark.py",
    "run_policy_alignment_diagnostics.py",
    "run_shortcut_stability_study.py",
    "verify_matched_objective_benchmark.py",
    "verify_matched_objective_protocol.py",
    "verify_causal_consistency.sh",
    "verify_neurips_controls.py",
    "verify_neurips_core.py",
    "verify_neurips_report.py",
    "verify_pixel_benchmark.py",
    "verify_pendulum_and_tests.sh",
    "verify_policy_alignment_diagnostics.py",
    "verify_reacher_imf_small_comparison.sh",
    "verify_reacher_imf_small_manifest.sh",
    "verify_remote_reacher_imf_small.sh",
    "verify_shortcut_stability_study.py",
    "verify_smoke_idempotence.py",
    "verify_trajectory_imf_novelty.py",
    "verify_trajectory_imf_theory.py",
)
_COMPARISON_TEST_BASENAMES = (
    "__init__.py",
    "test_causal_reacher_study.py",
    "test_cluster_workflow.py",
    "test_matched_objective_artifact_manifest.py",
    "test_matched_objective_benchmark.py",
    "test_matched_objective_diagnostics.py",
    "test_matched_objective_protocol.py",
    "test_matched_objective_validation_cache.py",
    "test_neurips_controls.py",
    "test_neurips_report.py",
    "test_neurips_verifiers.py",
    "test_pixel_benchmark.py",
    "test_policy_alignment_diagnostics.py",
    "test_rollout_replay_validation.py",
    "test_shortcut_stability_study.py",
    "test_trajectory_imf_theory.py",
)
_CLUSTER_SOURCE_BASENAMES = (
    "bootstrap_environment.sh",
    "causal_reacher_preflight.sbatch",
    "causal_reacher_small.sbatch",
    "cluster_spec.json",
    "controls_array.sbatch",
    "controls_finalize.sbatch",
    "controls_freeze.sbatch",
    "controls_retry_audit.sbatch",
    "controls_stage_verify.sbatch",
    "diagnostics.sbatch",
    "feasibility.py",
    "freeze_confirmatory.sbatch",
    "gpu_preflight.py",
    "pixel_array.sbatch",
    "pixel_finalize.sbatch",
    "pixel_freeze.sbatch",
    "pixel_retry_audit.sbatch",
    "policy_alignment_diagnostics.sbatch",
    "preflight.sbatch",
    "profile_finalize.sbatch",
    "retry_audit.sbatch",
    "runtime_environment.sh",
    "stage_array.sbatch",
    "stage_verify.sbatch",
    "shortcut_stability_array.sbatch",
    "shortcut_stability_verify.sbatch",
    "submit.py",
    "supplementary.py",
    "supplementary_plan.json",
    "verify_cluster_evidence.py",
    "verify_local_contract.py",
    "verify_remote_gate.sh",
    "workflow.py",
)
_REQUIRED_SOURCE_PATHS = tuple(
    sorted(
        [
            "imf_dreamer_jax/pyproject.toml",
            "dreamer_imf_comparison/requirements-neurips-cuda12-lock.txt",
            "dreamer_imf_comparison/matched_objective_protocol.json",
            "dreamer_imf_comparison/neurips_controls_protocol.json",
            "dreamer_imf_comparison/pixel_benchmark_protocol.json",
            "dreamer_imf_comparison/TRAJECTORY_IMF_THEORY.md",
            "dreamer_imf_comparison/TRAJECTORY_IMF_NOVELTY_AUDIT.md",
            "dreamer_imf_comparison/NEURIPS_READINESS_REPORT.md",
            "dreamer_imf_comparison/trajectory_imf_novelty_sources.json",
        ]
        + [
            f"imf_dreamer_jax/src/imf_dreamer_jax/{name}"
            for name in _LIBRARY_SOURCE_BASENAMES
        ]
        + [f"imf_dreamer_jax/tests/{name}" for name in _LIBRARY_TEST_BASENAMES]
        + [
            f"dreamer_imf_comparison/dreamer_imf_compare/{name}"
            for name in _COMPARISON_SOURCE_BASENAMES
        ]
        + [
            f"dreamer_imf_comparison/scripts/{name}"
            for name in _COMPARISON_SCRIPT_BASENAMES
        ]
        + [
            f"dreamer_imf_comparison/tests/{name}"
            for name in _COMPARISON_TEST_BASENAMES
        ]
        + [
            f"dreamer_imf_comparison/cluster/neurips/{name}"
            for name in _CLUSTER_SOURCE_BASENAMES
        ]
    )
)

_ACTUAL_ARTIFACTS = {
    "frozen_protocol",
    "source_commit_and_dirty_patch",
    "dependency_lock",
    "hardware_and_software_environment",
    "dataset_payload_and_digest",
    "episode_split_indices",
    "minibatch_indices",
    "sampled_objective_noise_times_and_step_sizes",
    "evaluation_start_indices",
    "evaluation_episode_seeds",
    "compiler_ir_and_cost_analysis",
    "world_model_and_actor_checkpoints",
    "parameter_tree_hashes_and_counts",
    "raw_predictive_draws",
    "raw_action_traces",
    "raw_per_seed_primary_and_secondary_metrics",
    "bootstrap_draw_indices_and_intervals",
    "machine_readable_summary_and_human_report",
}
_SMOKE_PLACEHOLDER_ARTIFACTS = {"hpo_trial_metrics_and_selection_manifest"}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        descriptor = canonical_bytes([name, value.dtype.str, list(value.shape)])
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _write_npz_atomic(path: str | Path, arrays: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=destination.name + ".", delete=False
        ) as handle:
            temporary = handle.name
            np.savez_compressed(
                handle, **{name: np.asarray(value) for name, value in arrays.items()}
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def derive_seed(*parts: Any) -> int:
    """Portable namespaced 32-bit seed derivation from canonical JSON."""

    return int.from_bytes(hashlib.sha256(canonical_bytes(parts)).digest()[:4], "big")


def derive_jax_key(*parts: Any) -> Any:
    """Derive a namespaced 64-bit Threefry key without truncating to one seed word."""

    import jax
    import jax.numpy as jnp

    digest = hashlib.sha256(canonical_bytes(parts)).digest()
    words = [
        int.from_bytes(digest[0:4], "big"),
        int.from_bytes(digest[4:8], "big"),
    ]
    return jax.random.wrap_key_data(jnp.asarray(words, dtype=jnp.uint32))


def _folded_key_rows(base_key: Any, count: int) -> np.ndarray:
    """Materialize a fold-in schedule with one vectorized device dispatch.

    Confirmatory world-model cells can contain 100,000 updates.  Keeping this
    derivation vectorized makes exact schedule verification practical without
    weakening the stored per-update RNG contract.
    """

    import jax
    import jax.numpy as jnp

    if count < 0:
        raise ValueError("fold-in schedule count must be nonnegative")
    indices = jnp.arange(count, dtype=jnp.uint32)
    key_data = jax.vmap(
        lambda index: jax.random.key_data(jax.random.fold_in(base_key, index))
    )(indices)
    return np.asarray(jax.device_get(key_data), dtype=np.uint32)


def canonical_task_id(task: str) -> str:
    if not isinstance(task, str) or not task:
        raise ValueError("task id must be a nonempty string")
    canonical = task if task.startswith("dmc_") else f"dmc_{task}"
    split_task(canonical)
    return canonical


def resolved_task(task: str) -> dict[str, str]:
    canonical = canonical_task_id(task)
    domain, task_name = split_task(canonical)
    return {"canonical_id": canonical, "domain": domain, "task": task_name}


def _workspace_root(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).resolve()
    return Path(__file__).resolve().parents[2]


GIT_COMMAND_TIMEOUT_SECONDS = 5


def _git_identity(workspace: Path, source_files: Sequence[str]) -> dict[str, Any]:
    # Provenance is fail-closed for pilot/confirmatory runs, so a short timeout
    # can only reject an unhealthy checkout; it cannot admit an unidentified
    # source tree.  Keeping this bounded also prevents cloud-backed/stale Git
    # metadata from hanging local smoke verification for minutes.
    timeout_seconds = GIT_COMMAND_TIMEOUT_SECONDS

    def run(*arguments: str) -> tuple[str | None, str, str]:
        try:
            completed = subprocess.run(
                ["git", "-C", str(workspace), *arguments],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return None, f"timeout_{timeout_seconds}_seconds", ""
        except OSError as error:
            return None, "unavailable", str(error)[:1000]
        if completed.returncode != 0:
            return None, f"git_exit_{completed.returncode}", completed.stderr.strip()[:1000]
        return completed.stdout.strip(), "complete", completed.stderr.strip()[:1000]

    commit, commit_status, commit_stderr = run("rev-parse", "HEAD")
    if commit_status == "complete":
        # Diff against HEAD, not only the index/worktree, so staged edits are archived too.
        diff, diff_status, diff_stderr = run(
            "diff", "--binary", "HEAD", "--", *source_files
        )
        untracked, untracked_status, untracked_stderr = run(
            "ls-files", "--others", "--exclude-standard", "--", *source_files
        )
    else:
        diff, diff_status, diff_stderr = None, "skipped_without_commit", ""
        untracked, untracked_status, untracked_stderr = (
            None,
            "skipped_without_commit",
            "",
        )
    untracked_files = sorted(line for line in (untracked or "").splitlines() if line)
    return {
        "commit": commit or "unavailable",
        "dirty_patch_sha256": hashlib.sha256((diff or "").encode("utf-8")).hexdigest(),
        "dirty_patch_bytes": len((diff or "").encode("utf-8")),
        "commit_status": commit_status,
        "commit_stderr": commit_stderr,
        "dirty_patch_status": diff_status,
        "dirty_patch_stderr": diff_stderr,
        "untracked_files": untracked_files,
        "untracked_status": untracked_status,
        "untracked_stderr": untracked_stderr,
        "command_timeout_seconds": timeout_seconds,
        "fallback": "complete_relevant_source_file_size_and_sha256_manifest",
    }


def _normalized_source_root(root: str | Path) -> Path:
    lexical = Path(os.path.abspath(os.fspath(root)))
    if lexical.is_symlink():
        raise ValueError("benchmark source root must not be a symlink")
    if not lexical.is_dir():
        raise FileNotFoundError(f"benchmark source root is absent: {lexical}")
    return lexical.resolve(strict=True)


def _require_regular_source_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or PureWindowsPath(relative).is_absolute()
        or "\\" in relative
        or ".." in pure.parts
        or pure.as_posix() != relative
    ):
        raise ValueError(f"source dependency path is not canonical: {relative!r}")
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"source dependency contains a symlink: {relative}")
    if not current.is_file():
        raise FileNotFoundError(f"source dependency is absent: {relative}")
    return current


def _source_files(root: Path) -> list[str]:
    root = _normalized_source_root(root)
    candidates: set[Path] = set()
    for pattern in (
        "imf_dreamer_jax/src/**/*.py",
        # Certification code is part of the frozen source identity too.  A
        # verifier or regression-test change must invalidate an older run just
        # as an executable-model change does.
        "imf_dreamer_jax/tests/test_*.py",
        "dreamer_imf_comparison/dreamer_imf_compare/__init__.py",
        "dreamer_imf_comparison/dreamer_imf_compare/artifacts.py",
        "dreamer_imf_comparison/dreamer_imf_compare/dmc.py",
        "dreamer_imf_comparison/dreamer_imf_compare/protocol.py",
        "dreamer_imf_comparison/dreamer_imf_compare/matched_objective*.py",
        "dreamer_imf_comparison/dreamer_imf_compare/neurips_controls.py",
        "dreamer_imf_comparison/dreamer_imf_compare/causal_reacher_study.py",
        "dreamer_imf_comparison/dreamer_imf_compare/pixel_benchmark.py",
        "dreamer_imf_comparison/dreamer_imf_compare/policy_alignment_diagnostics.py",
        "dreamer_imf_comparison/scripts/*causal*.py",
        "dreamer_imf_comparison/scripts/verify_causal_consistency.sh",
        "dreamer_imf_comparison/scripts/*matched_objective*.py",
        "dreamer_imf_comparison/scripts/*neurips*.py",
        "dreamer_imf_comparison/scripts/*pixel*.py",
        "dreamer_imf_comparison/scripts/*pendulum*.sh",
        "dreamer_imf_comparison/scripts/*policy_alignment*.py",
        "dreamer_imf_comparison/scripts/*reacher_imf_small*.sh",
        "dreamer_imf_comparison/scripts/verify_smoke_idempotence.py",
        "dreamer_imf_comparison/scripts/*trajectory_imf*.py",
        "dreamer_imf_comparison/dreamer_imf_compare/shortcut_stability_study.py",
        "dreamer_imf_comparison/scripts/*shortcut_stability*.py",
        "dreamer_imf_comparison/cluster/neurips/**/*.py",
        "dreamer_imf_comparison/cluster/neurips/**/*.sh",
        "dreamer_imf_comparison/cluster/neurips/**/*.sbatch",
        "dreamer_imf_comparison/cluster/neurips/**/*.json",
        "dreamer_imf_comparison/tests/test_matched_objective*.py",
        "dreamer_imf_comparison/tests/test_neurips_*.py",
        "dreamer_imf_comparison/tests/test_pixel*.py",
        "dreamer_imf_comparison/tests/test_policy_alignment*.py",
        "dreamer_imf_comparison/tests/test_causal_reacher*.py",
        "dreamer_imf_comparison/tests/test_trajectory_imf*.py",
        "dreamer_imf_comparison/tests/test_rollout_replay_validation.py",
        "dreamer_imf_comparison/tests/test_cluster_*.py",
        "dreamer_imf_comparison/tests/test_shortcut_stability_study.py",
        "imf_dreamer_jax/tests/__init__.py",
        "dreamer_imf_comparison/tests/__init__.py",
        "imf_dreamer_jax/pyproject.toml",
        "dreamer_imf_comparison/requirements-neurips-cuda12-lock.txt",
        "dreamer_imf_comparison/matched_objective_protocol.json",
        "dreamer_imf_comparison/neurips_controls_protocol.json",
        "dreamer_imf_comparison/pixel_benchmark_protocol.json",
        "dreamer_imf_comparison/TRAJECTORY_IMF_THEORY.md",
        "dreamer_imf_comparison/TRAJECTORY_IMF_NOVELTY_AUDIT.md",
        "dreamer_imf_comparison/NEURIPS_READINESS_REPORT.md",
        "dreamer_imf_comparison/trajectory_imf_novelty_sources.json",
    ):
        for path in root.glob(pattern):
            if path.is_symlink():
                raise ValueError(
                    f"source dependency contains a symlink: {path.relative_to(root)}"
                )
            if path.is_file():
                candidates.add(path)
    if not candidates:
        raise FileNotFoundError("no benchmark source files were discovered")
    discovered = sorted(path.relative_to(root).as_posix() for path in candidates)
    missing = sorted(set(_REQUIRED_SOURCE_PATHS) - set(discovered))
    if missing:
        raise FileNotFoundError(
            "benchmark source closure is incomplete: " + ", ".join(missing)
        )
    for relative in discovered:
        _require_regular_source_file(root, relative)
    return discovered


def _source_file_records(root: Path, source_files: Sequence[str]) -> list[dict[str, Any]]:
    root = _normalized_source_root(root)
    files: list[dict[str, Any]] = []
    for relative in source_files:
        path = _require_regular_source_file(root, relative)
        files.append(
            {"path": relative, "size": path.stat().st_size, "sha256": file_sha256(path)}
        )
    return files


def build_source_manifest(workspace: str | Path | None = None) -> dict[str, Any]:
    root = _workspace_root(workspace)
    source_files = _source_files(root)
    payload = {
        "schema_version": "matched-objective-source-v1",
        "files": _source_file_records(root, source_files),
        "execution_spec": EXECUTION_SPEC,
        "git": _git_identity(root, source_files),
    }
    payload["source_sha256"] = object_sha256(payload)
    return payload


def validate_source_manifest(
    manifest: Mapping[str, Any], workspace: str | Path | None = None
) -> None:
    root = _workspace_root(workspace)
    source_files = _source_files(root)
    payload = _without_digest(manifest, "source_sha256")
    if (
        set(manifest) != {"schema_version", "files", "execution_spec", "git", "source_sha256"}
        or manifest.get("schema_version") != "matched-objective-source-v1"
        or manifest.get("source_sha256") != object_sha256(payload)
        or manifest.get("execution_spec") != EXECUTION_SPEC
        or manifest.get("files") != _source_file_records(root, source_files)
    ):
        raise ValueError("source manifest does not match the current benchmark source tree")
    git = manifest.get("git")
    required_git = {
        "commit",
        "dirty_patch_sha256",
        "dirty_patch_bytes",
        "commit_status",
        "commit_stderr",
        "dirty_patch_status",
        "dirty_patch_stderr",
        "untracked_files",
        "untracked_status",
        "untracked_stderr",
        "command_timeout_seconds",
        "fallback",
    }
    if not isinstance(git, Mapping) or set(git) != required_git:
        raise ValueError("source Git identity is incomplete")
    hexadecimal = set("0123456789abcdef")
    if (
        git["command_timeout_seconds"] != GIT_COMMAND_TIMEOUT_SECONDS
        or not isinstance(git["commit_stderr"], str)
        or not isinstance(git["dirty_patch_stderr"], str)
        or not isinstance(git["untracked_stderr"], str)
        or not isinstance(git["untracked_files"], list)
        or any(not isinstance(path, str) or not path for path in git["untracked_files"])
        or not isinstance(git["dirty_patch_bytes"], int)
        or isinstance(git["dirty_patch_bytes"], bool)
        or git["dirty_patch_bytes"] < 0
        or not isinstance(git["dirty_patch_sha256"], str)
        or len(git["dirty_patch_sha256"]) != 64
        or any(character not in hexadecimal for character in git["dirty_patch_sha256"])
    ):
        raise ValueError("source Git identity values are malformed")
    if git["commit_status"] == "complete" and (
        not isinstance(git["commit"], str)
        or len(git["commit"]) not in (40, 64)
        or any(character not in hexadecimal for character in git["commit"])
    ):
        raise ValueError("source Git commit hash is malformed")
    if git["commit_status"] == "complete" and git["dirty_patch_status"] == "complete":
        if git != _git_identity(root, source_files):
            raise ValueError("source Git identity no longer matches the current worktree")
    elif git["fallback"] != "complete_relevant_source_file_size_and_sha256_manifest":
        raise ValueError("incomplete Git identity lacks the full source-file fallback")


def _matrix_digest(matrix: Mapping[str, Any]) -> str:
    payload = json.loads(json.dumps(matrix))
    payload.pop("matrix_sha256", None)
    return object_sha256(payload)


def _cell(
    *,
    stage: str,
    profile: str,
    evidence_class: str,
    protocol_sha256: str,
    source_sha256: str,
    task: str,
    world_model_seed: int | None = None,
    budget_track: str | None = None,
    arm: str | None = None,
    actor_seed: int | None = None,
    nfe: int | None = None,
    candidate_id: str | None = None,
    selection_sha256: str | None = None,
    config_template_sha256: str | None = None,
    dependencies: Sequence[str] = (),
) -> dict[str, Any]:
    identity = {
        "stage": stage,
        "profile": profile,
        "evidence_class": evidence_class,
        "protocol_sha256": protocol_sha256,
        "source_sha256": source_sha256,
        "task": task,
        "resolved_task": resolved_task(task),
        "world_model_seed": world_model_seed,
        "budget_track": budget_track,
        "arm": arm,
        "actor_seed": actor_seed,
        "nfe": nfe,
        "candidate_id": candidate_id,
        "selection_sha256": selection_sha256,
        "config_template_sha256": config_template_sha256,
        "dependencies": list(dependencies),
    }
    identity_sha256 = object_sha256(identity)
    return {
        **identity,
        "cell_id": f"{stage}-{identity_sha256[:24]}",
        "identity_sha256": identity_sha256,
    }


def build_matrix(
    protocol: Mapping[str, Any],
    profile: str,
    *,
    source_manifest: Mapping[str, Any],
    selection_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_matched_objective_protocol(protocol)
    if profile not in PROFILE_ORDER:
        raise ValueError(f"unknown benchmark profile {profile!r}")
    if profile == "pilot":
        raise ValueError("pilot HPO uses build_pilot_hpo_matrix, not the confirmatory matrix")
    if source_manifest.get("source_sha256") != object_sha256(
        {key: value for key, value in source_manifest.items() if key != "source_sha256"}
    ):
        raise ValueError("source manifest digest is invalid")
    selected = protocol["profiles"][profile]
    protocol_sha = protocol_digest(protocol)
    source_sha = str(source_manifest["source_sha256"])
    if profile == "smoke":
        if selection_manifest is not None:
            raise ValueError("smoke uses the registered neutral engineering configuration")
        selected_candidates = {
            arm: next(
                candidate
                for candidate in pilot_candidates(protocol, arm)
                if candidate["overrides"]
                == (
                    {"training_k_max": 4, "objective_loss_scale": 1.0}
                    if arm == "shortcut_forcing"
                    else {"imf_boundary_fraction": 0.5, "objective_loss_scale": 1.0}
                )
            )
            for arm in ARM_ORDER
        }
        selection = {
            "schema_version": "matched-objective-smoke-selection-v1",
            "status": "engineering_only",
            "protocol_sha256": protocol_sha,
            "selected": selected_candidates,
        }
    else:
        selection = validate_hpo_selection_manifest(
            selection_manifest, protocol, source_sha256=source_sha
        )
        selected_candidates = {
            arm: {
                "candidate_id": selection["selected"][arm]["candidate_id"],
                "overrides": selection["selected"][arm]["overrides"],
            }
            for arm in ARM_ORDER
        }
    selection_sha = (
        str(selection["selection_sha256"])
        if profile == "confirmatory"
        else object_sha256(selection)
    )
    templates = {
        arm: object_sha256(
            {
                "common": protocol["canonical_executable_config"]["dreamer_config_non_task_shape_common"],
                "arm": protocol["canonical_executable_config"]["dreamer_config_arm_values"][arm],
                "overrides": selected_candidates[arm]["overrides"],
                "primary_nfe": protocol["evaluation"]["primary_nfe"][arm],
            }
        )
        for arm in ARM_ORDER
    }
    cells: list[dict[str, Any]] = []
    for raw_task in selected["tasks"]:
        task = canonical_task_id(raw_task)
        compute = _cell(
            stage="compute_plan",
            profile=profile,
            evidence_class=selected["evidence_class"],
            protocol_sha256=protocol_sha,
            source_sha256=source_sha,
            task=task,
            selection_sha256=selection_sha,
        )
        cells.append(compute)
        for world_seed in selected["world_model_seeds"]:
            dataset = _cell(
                stage="dataset",
                profile=profile,
                evidence_class=selected["evidence_class"],
                protocol_sha256=protocol_sha,
                source_sha256=source_sha,
                task=task,
                world_model_seed=int(world_seed),
            )
            cells.append(dataset)
            for track in BUDGET_TRACK_ORDER:
                for arm in ARM_ORDER:
                    world = _cell(
                        stage="world_model",
                        profile=profile,
                        evidence_class=selected["evidence_class"],
                        protocol_sha256=protocol_sha,
                        source_sha256=source_sha,
                        task=task,
                        world_model_seed=int(world_seed),
                        budget_track=track,
                        arm=arm,
                        candidate_id=selected_candidates[arm]["candidate_id"],
                        selection_sha256=selection_sha,
                        config_template_sha256=templates[arm],
                        dependencies=(dataset["cell_id"], compute["cell_id"]),
                    )
                    cells.append(world)
                    for nfe in NFE_FRONTIER:
                        cells.append(
                            _cell(
                                stage="rollout",
                                profile=profile,
                                evidence_class=selected["evidence_class"],
                                protocol_sha256=protocol_sha,
                                source_sha256=source_sha,
                                task=task,
                                world_model_seed=int(world_seed),
                                budget_track=track,
                                arm=arm,
                                nfe=int(nfe),
                                candidate_id=selected_candidates[arm]["candidate_id"],
                                selection_sha256=selection_sha,
                                config_template_sha256=templates[arm],
                                dependencies=(
                                    dataset["cell_id"],
                                    compute["cell_id"],
                                    world["cell_id"],
                                ),
                            )
                        )
                    for actor_seed in selected[
                        "actor_seeds_nested_within_world_model_seed"
                    ]:
                        cells.append(
                            _cell(
                                stage="actor",
                                profile=profile,
                                evidence_class=selected["evidence_class"],
                                protocol_sha256=protocol_sha,
                                source_sha256=source_sha,
                                task=task,
                                world_model_seed=int(world_seed),
                                budget_track=track,
                                arm=arm,
                                actor_seed=int(actor_seed),
                                candidate_id=selected_candidates[arm]["candidate_id"],
                                selection_sha256=selection_sha,
                                config_template_sha256=templates[arm],
                                dependencies=(
                                    dataset["cell_id"],
                                    compute["cell_id"],
                                    world["cell_id"],
                                ),
                            )
                        )
    stage_rank = {stage: index for index, stage in enumerate(STAGE_ORDER)}
    cells.sort(key=lambda row: (stage_rank[row["stage"]], row["cell_id"]))
    matrix = {
        "schema_version": MATRIX_SCHEMA,
        "status": "frozen_before_execution",
        "profile": profile,
        "evidence_class": selected["evidence_class"],
        "claim_eligible": bool(selected["claim_eligible"]),
        "protocol_sha256": protocol_sha,
        "source_sha256": source_sha,
        "execution_spec_sha256": object_sha256(EXECUTION_SPEC),
        "selection_manifest": selection,
        "selection_sha256": selection_sha,
        "arm_order": list(ARM_ORDER),
        "budget_track_order": list(BUDGET_TRACK_ORDER),
        "nfe_frontier": list(NFE_FRONTIER),
        "cells": cells,
    }
    matrix["matrix_sha256"] = _matrix_digest(matrix)
    return matrix


def validate_matrix(
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> None:
    validate_matched_objective_protocol(protocol)
    required = {
        "schema_version",
        "status",
        "profile",
        "evidence_class",
        "claim_eligible",
        "protocol_sha256",
        "source_sha256",
        "execution_spec_sha256",
        "selection_manifest",
        "selection_sha256",
        "arm_order",
        "budget_track_order",
        "nfe_frontier",
        "cells",
        "matrix_sha256",
    }
    if set(matrix) != required:
        raise ValueError("matrix keys are incomplete or contain unregistered values")
    if matrix.get("schema_version") != MATRIX_SCHEMA:
        raise ValueError("matrix schema mismatch")
    if matrix.get("matrix_sha256") != _matrix_digest(matrix):
        raise ValueError("matrix digest mismatch")
    expected = build_matrix(
        protocol,
        str(matrix.get("profile")),
        source_manifest=source_manifest,
        selection_manifest=(
            None
            if matrix.get("profile") == "smoke"
            else matrix.get("selection_manifest")
        ),
    )
    if matrix != expected:
        raise ValueError("matrix differs from its protocol/source-derived canonical form")
    cells = matrix["cells"]
    identifiers = [cell["cell_id"] for cell in cells]
    identities = [cell["identity_sha256"] for cell in cells]
    if len(set(identifiers)) != len(identifiers) or len(set(identities)) != len(identities):
        raise ValueError("matrix contains a cell identity collision")
    for cell in cells:
        identity = {
            key: cell[key]
            for key in (
                "stage",
                "profile",
                "evidence_class",
                "protocol_sha256",
                "source_sha256",
                "task",
                "resolved_task",
                "world_model_seed",
                "budget_track",
                "arm",
                "actor_seed",
                "nfe",
                "candidate_id",
                "selection_sha256",
                "config_template_sha256",
                "dependencies",
            )
        }
        digest = object_sha256(identity)
        if digest != cell["identity_sha256"] or cell["cell_id"] != f"{cell['stage']}-{digest[:24]}":
            raise ValueError("matrix cell identity digest mismatch")
    by_id = {cell["cell_id"]: cell for cell in cells}
    for cell in cells:
        if any(dependency not in by_id for dependency in cell["dependencies"]):
            raise ValueError("matrix dependency refers to an absent cell")
        if any(
            STAGE_ORDER.index(by_id[dependency]["stage"])
            >= STAGE_ORDER.index(cell["stage"])
            for dependency in cell["dependencies"]
        ):
            raise ValueError("matrix dependency does not precede its consumer")


def matrix_cells(
    matrix: Mapping[str, Any], stage: str | None = None
) -> list[Mapping[str, Any]]:
    cells = list(matrix["cells"])
    return cells if stage is None else [cell for cell in cells if cell["stage"] == stage]


def expected_matrix_counts(protocol: Mapping[str, Any], profile: str) -> dict[str, int]:
    selected = protocol["profiles"][profile]
    tasks = len(selected["tasks"])
    worlds = tasks * len(selected["world_model_seeds"])
    trained = worlds * len(BUDGET_TRACK_ORDER) * len(ARM_ORDER)
    actors = trained * len(selected["actor_seeds_nested_within_world_model_seed"])
    return {
        "dataset": worlds,
        "compute_plan": tasks,
        "world_model": trained,
        "rollout": trained * len(NFE_FRONTIER),
        "actor": actors,
    }


def build_pilot_hpo_matrix(
    protocol: Mapping[str, Any], *, source_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Build all equal-update pilot cells for 12 candidates per objective."""

    validate_matched_objective_protocol(protocol)
    if source_manifest.get("source_sha256") != object_sha256(
        {key: value for key, value in source_manifest.items() if key != "source_sha256"}
    ):
        raise ValueError("source manifest digest is invalid")
    profile = "pilot"
    selected = protocol["profiles"][profile]
    protocol_sha = protocol_digest(protocol)
    source_sha = str(source_manifest["source_sha256"])
    per_arm = {arm: pilot_candidates(protocol, arm) for arm in ARM_ORDER}
    pairs = [
        {
            "pair_index": index,
            "pair_id": f"pilot-pair-{index:02d}",
            "candidates": {arm: per_arm[arm][index] for arm in ARM_ORDER},
        }
        for index in range(len(per_arm[ARM_ORDER[0]]))
    ]
    grid_sha = object_sha256(pairs)
    cells: list[dict[str, Any]] = []
    datasets: dict[tuple[str, int], dict[str, Any]] = {}
    computes: dict[tuple[str, int], dict[str, Any]] = {}
    for task in selected["tasks"]:
        task = canonical_task_id(task)
        for pair in pairs:
            compute = _cell(
                stage="compute_plan",
                profile=profile,
                evidence_class=selected["evidence_class"],
                protocol_sha256=protocol_sha,
                source_sha256=source_sha,
                task=task,
                candidate_id=pair["pair_id"],
                selection_sha256=grid_sha,
                config_template_sha256=object_sha256(pair["candidates"]),
            )
            computes[(task, pair["pair_index"])] = compute
            cells.append(compute)
        for world_seed in selected["world_model_seeds"]:
            dataset = _cell(
                stage="dataset",
                profile=profile,
                evidence_class=selected["evidence_class"],
                protocol_sha256=protocol_sha,
                source_sha256=source_sha,
                task=task,
                world_model_seed=int(world_seed),
            )
            datasets[(task, int(world_seed))] = dataset
            cells.append(dataset)
            for pair in pairs:
                compute = computes[(task, pair["pair_index"])]
                for arm in ARM_ORDER:
                    candidate = pair["candidates"][arm]
                    template = object_sha256(
                        {
                            "common": protocol["canonical_executable_config"]["dreamer_config_non_task_shape_common"],
                            "arm": protocol["canonical_executable_config"]["dreamer_config_arm_values"][arm],
                            "overrides": candidate["overrides"],
                            "primary_nfe": protocol["evaluation"]["primary_nfe"][arm],
                        }
                    )
                    world = _cell(
                        stage="world_model",
                        profile=profile,
                        evidence_class=selected["evidence_class"],
                        protocol_sha256=protocol_sha,
                        source_sha256=source_sha,
                        task=task,
                        world_model_seed=int(world_seed),
                        budget_track="equal_updates",
                        arm=arm,
                        candidate_id=candidate["candidate_id"],
                        selection_sha256=grid_sha,
                        config_template_sha256=template,
                        dependencies=(dataset["cell_id"], compute["cell_id"]),
                    )
                    cells.append(world)
                    cells.append(
                        _cell(
                            stage="rollout",
                            profile=profile,
                            evidence_class=selected["evidence_class"],
                            protocol_sha256=protocol_sha,
                            source_sha256=source_sha,
                            task=task,
                            world_model_seed=int(world_seed),
                            budget_track="equal_updates",
                            arm=arm,
                            nfe=int(protocol["evaluation"]["primary_nfe"][arm]),
                            candidate_id=candidate["candidate_id"],
                            selection_sha256=grid_sha,
                            config_template_sha256=template,
                            dependencies=(dataset["cell_id"], compute["cell_id"], world["cell_id"]),
                        )
                    )
                    for actor_seed in selected["actor_seeds_nested_within_world_model_seed"]:
                        cells.append(
                            _cell(
                                stage="actor",
                                profile=profile,
                                evidence_class=selected["evidence_class"],
                                protocol_sha256=protocol_sha,
                                source_sha256=source_sha,
                                task=task,
                                world_model_seed=int(world_seed),
                                budget_track="equal_updates",
                                arm=arm,
                                actor_seed=int(actor_seed),
                                candidate_id=candidate["candidate_id"],
                                selection_sha256=grid_sha,
                                config_template_sha256=template,
                                dependencies=(dataset["cell_id"], compute["cell_id"], world["cell_id"]),
                            )
                        )
    rank = {stage: index for index, stage in enumerate(STAGE_ORDER)}
    cells.sort(key=lambda row: (rank[row["stage"]], row["cell_id"]))
    matrix = {
        "schema_version": HPO_MATRIX_SCHEMA,
        "status": "frozen_before_execution",
        "profile": profile,
        "evidence_class": selected["evidence_class"],
        "claim_eligible": False,
        "protocol_sha256": protocol_sha,
        "source_sha256": source_sha,
        "execution_spec_sha256": object_sha256(EXECUTION_SPEC),
        "selection_sha256": grid_sha,
        "candidate_pairs": pairs,
        "arm_order": list(ARM_ORDER),
        "budget_track_order": ["equal_updates"],
        "nfe_frontier": protocol["evaluation"]["primary_nfe"],
        "cells": cells,
    }
    matrix["matrix_sha256"] = _matrix_digest(matrix)
    return matrix


def expected_hpo_matrix_counts(protocol: Mapping[str, Any]) -> dict[str, int]:
    profile = protocol["profiles"]["pilot"]
    candidates = int(protocol["hyperparameter_policy"]["candidate_evaluations_per_arm"])
    tasks = len(profile["tasks"])
    worlds = tasks * len(profile["world_model_seeds"])
    trained = worlds * len(ARM_ORDER) * candidates
    return {
        "dataset": worlds,
        "compute_plan": tasks * candidates,
        "world_model": trained,
        "rollout": trained,
        "actor": trained * len(profile["actor_seeds_nested_within_world_model_seed"]),
    }


def validate_pilot_hpo_matrix(
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> None:
    expected = build_pilot_hpo_matrix(protocol, source_manifest=source_manifest)
    if matrix != expected or matrix.get("matrix_sha256") != _matrix_digest(matrix):
        raise ValueError("pilot HPO matrix differs from the canonical frozen grid")
    counts = expected_hpo_matrix_counts(protocol)
    for stage, expected_count in counts.items():
        if len(matrix_cells(matrix, stage)) != expected_count:
            raise ValueError(f"pilot HPO {stage} cell count mismatch")


def _candidate_for_cell(
    matrix: Mapping[str, Any], cell: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    arm = cell.get("arm")
    if arm not in ARM_ORDER:
        return None
    if matrix.get("schema_version") == HPO_MATRIX_SCHEMA:
        matching = [
            pair["candidates"][arm]
            for pair in matrix["candidate_pairs"]
            if pair["candidates"][arm]["candidate_id"] == cell["candidate_id"]
        ]
        if len(matching) != 1:
            raise ValueError("pilot cell candidate is absent or duplicated")
        candidate = matching[0]
    else:
        selected = matrix["selection_manifest"]["selected"][arm]
        candidate = {
            "candidate_id": selected["candidate_id"],
            "overrides": selected["overrides"],
        }
    if (
        cell.get("candidate_id") != candidate["candidate_id"]
        or cell.get("selection_sha256") != matrix.get("selection_sha256")
    ):
        raise ValueError("cell candidate differs from its frozen matrix selection")
    return candidate


def _compute_candidates(
    matrix: Mapping[str, Any], cell: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    if matrix.get("schema_version") == HPO_MATRIX_SCHEMA:
        matching = [
            pair for pair in matrix["candidate_pairs"] if pair["pair_id"] == cell["candidate_id"]
        ]
        if len(matching) != 1:
            raise ValueError("pilot compute candidate pair is absent or duplicated")
        return dict(matching[0]["candidates"])
    return {
        arm: {
            "candidate_id": matrix["selection_manifest"]["selected"][arm]["candidate_id"],
            "overrides": matrix["selection_manifest"]["selected"][arm]["overrides"],
        }
        for arm in ARM_ORDER
    }


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def allocate_equal_flops(
    costs: Mapping[str, float],
    *,
    reference_arm: str,
    reference_updates: int,
    tolerance: float,
    enforce_tolerance: bool = True,
) -> tuple[float, dict[str, dict[str, float | int]]]:
    """Apply the frozen nearest-integer compiler-FLOP allocation exactly."""

    if set(costs) != set(ARM_ORDER) or reference_arm not in ARM_ORDER:
        raise ValueError("FLOP costs must contain exactly the two confirmatory arms")
    if isinstance(reference_updates, bool) or not isinstance(reference_updates, int) or reference_updates <= 0:
        raise ValueError("reference_updates must be a positive integer")
    if not 0.0 < float(tolerance) < 1.0:
        raise ValueError("FLOP tolerance must lie in (0, 1)")
    normalized = {arm: _finite_positive(costs[arm], f"{arm} cost") for arm in ARM_ORDER}
    target = normalized[reference_arm] * reference_updates
    allocation: dict[str, dict[str, float | int]] = {}
    for arm in ARM_ORDER:
        cost = normalized[arm]
        ratio = target / cost
        lower = max(1, math.floor(ratio))
        upper = max(1, math.ceil(ratio))
        candidates = (lower, upper)
        # On an exact distance tie, the lower (non-exceeding) allocation wins.
        updates = min(candidates, key=lambda value: (abs(value * cost - target), value))
        cumulative = cost * updates
        relative = abs(cumulative - target) / target
        if enforce_tolerance and relative > float(tolerance):
            raise ValueError(
                f"{arm} nearest integer allocation misses target by {relative:.6g}, "
                f"above tolerance {tolerance:.6g}"
            )
        allocation[arm] = {
            "flops_per_update": cost,
            "updates": int(updates),
            "cumulative_flops": cumulative,
            "relative_target_error": relative,
            "within_registered_tolerance": relative <= float(tolerance),
        }
    return target, allocation


def interquartile_mean(values: Sequence[float]) -> float:
    """Exact empirical IQM as the integral of the quantile function on [0.25, 0.75]."""

    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("IQM values must be a nonempty finite vector")
    n = array.size
    lower = 0.25 * n
    upper = 0.75 * n
    total = 0.0
    for index, value in enumerate(array):
        overlap = max(0.0, min(index + 1.0, upper) - max(float(index), lower))
        total += overlap * float(value)
    return total / (upper - lower)


def _validate_complete_units(
    units: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], profile: str
) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    selected = protocol["profiles"][profile]
    expected = {
        (track, canonical_task_id(task), int(seed))
        for track in BUDGET_TRACK_ORDER
        for task in selected["tasks"]
        for seed in selected["world_model_seeds"]
    }
    keyed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    actor_seeds = set(selected["actor_seeds_nested_within_world_model_seed"])
    for unit in units:
        key = (
            str(unit.get("budget_track")),
            canonical_task_id(str(unit.get("task"))),
            int(unit.get("world_model_seed")),
        )
        if key in keyed:
            raise ValueError("duplicate task/world-model unit")
        if set(unit.get("rollout_auc", {})) != set(ARM_ORDER):
            raise ValueError("unit does not contain both paired rollout arms")
        actors = unit.get("actor_episode_returns")
        if not isinstance(actors, Mapping) or set(actors) != set(ARM_ORDER):
            raise ValueError("unit does not contain both paired actor arms")
        for arm in ARM_ORDER:
            nested = actors[arm]
            if not isinstance(nested, Mapping) or {int(seed) for seed in nested} != actor_seeds:
                raise ValueError("unit actor seeds do not match the frozen nested seed set")
            for returns in nested.values():
                values = np.asarray(returns, dtype=np.float64)
                if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
                    raise ValueError("nested actor episode returns must be finite nonempty vectors")
        numeric = [float(unit["rollout_auc"][arm]) for arm in ARM_ORDER]
        if not np.isfinite(np.asarray(numeric)).all():
            raise ValueError("unit rollout AUC is non-finite")
        keyed[key] = unit
    if set(keyed) != expected:
        missing = sorted(expected - set(keyed))
        extra = sorted(set(keyed) - expected)
        raise ValueError(f"paired statistical units differ from protocol; missing={missing}, extra={extra}")
    return keyed


def _nested_actor_mean(unit: Mapping[str, Any], arm: str) -> float:
    # Episodes are averaged first inside each actor seed, then actor-seed means
    # are averaged inside the task x world-model seed independent unit.
    actor_means = [
        float(np.mean(np.asarray(returns, dtype=np.float64)))
        for _, returns in sorted(
            unit["actor_episode_returns"][arm].items(), key=lambda item: int(item[0])
        )
    ]
    return float(np.mean(actor_means))


def task_stratified_bootstrap(
    units: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    profile: str,
    *,
    resamples: int | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Bootstrap paired world-model indices within each task and recompute IQMs."""

    validate_matched_objective_protocol(protocol)
    keyed = _validate_complete_units(units, protocol, profile)
    selected = protocol["profiles"][profile]
    tasks = [canonical_task_id(task) for task in selected["tasks"]]
    seeds = [int(seed) for seed in selected["world_model_seeds"]]
    configured = int(protocol["statistics"]["bootstrap"]["resamples"])
    count = configured if resamples is None else int(resamples)
    if count <= 0 or count > configured:
        raise ValueError("bootstrap resamples must be positive and cannot exceed the protocol")
    if protocol["profiles"][profile]["claim_eligible"] and count != configured:
        raise ValueError("confirmatory analysis requires exactly the registered bootstrap count")
    random = np.random.default_rng(int(protocol["statistics"]["bootstrap"]["seed"]))
    indices = {
        track: random.integers(0, len(seeds), size=(count, len(tasks), len(seeds)), dtype=np.int32)
        for track in BUDGET_TRACK_ORDER
    }
    interval_confidence = float(protocol["statistics"]["multiplicity"]["interval_confidence"])
    tail = 0.5 * (1.0 - interval_confidence)
    metrics: dict[str, Any] = {}
    for track in BUDGET_TRACK_ORDER:
        by_metric: dict[str, Any] = {}
        for metric in PRIMARY_METRICS:
            arm_values: dict[str, np.ndarray] = {}
            for arm in ARM_ORDER:
                arm_values[arm] = np.asarray(
                    [
                        [
                            float(keyed[(track, task, seed)]["rollout_auc"][arm])
                            if metric == PRIMARY_METRICS[0]
                            else _nested_actor_mean(keyed[(track, task, seed)], arm)
                            for seed in seeds
                        ]
                        for task in tasks
                    ],
                    dtype=np.float64,
                )
            if metric == PRIMARY_METRICS[0]:
                point = interquartile_mean(arm_values["shortcut_forcing"].ravel()) - interquartile_mean(
                    arm_values["trajectory_imf"].ravel()
                )
            else:
                point = interquartile_mean(arm_values["trajectory_imf"].ravel()) - interquartile_mean(
                    arm_values["shortcut_forcing"].ravel()
                )
            draws = np.empty(count, dtype=np.float64)
            for draw in range(count):
                sampled: dict[str, list[float]] = {arm: [] for arm in ARM_ORDER}
                for task_index in range(len(tasks)):
                    chosen = indices[track][draw, task_index]
                    for arm in ARM_ORDER:
                        sampled[arm].extend(arm_values[arm][task_index, chosen].tolist())
                if metric == PRIMARY_METRICS[0]:
                    draws[draw] = interquartile_mean(sampled["shortcut_forcing"]) - interquartile_mean(
                        sampled["trajectory_imf"]
                    )
                else:
                    draws[draw] = interquartile_mean(sampled["trajectory_imf"]) - interquartile_mean(
                        sampled["shortcut_forcing"]
                    )
            lower, upper = np.quantile(draws, (tail, 1.0 - tail))
            by_metric[metric] = {
                "arm_iqm": {
                    arm: interquartile_mean(arm_values[arm].ravel()) for arm in ARM_ORDER
                },
                "contrast": float(point),
                "interval": {"lower": float(lower), "upper": float(upper)},
                "bootstrap_resamples": count,
                "independent_units": len(tasks) * len(seeds),
                "actor_aggregation": (
                    "episodes_then_nested_actor_seeds_within_task_x_world_model_seed"
                    if metric == PRIMARY_METRICS[1]
                    else "one_rollout_auc_per_task_x_world_model_seed"
                ),
            }
        metrics[track] = by_metric
    return metrics, indices


def analyze_units(
    units: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    profile: str,
    *,
    resamples: int | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metrics, indices = task_stratified_bootstrap(
        units, protocol, profile, resamples=resamples
    )
    intervals = {
        track: {
            metric: metrics[track][metric]["interval"] for metric in PRIMARY_METRICS
        }
        for track in BUDGET_TRACK_ORDER
    }
    evidence_class = protocol["profiles"][profile]["evidence_class"]
    decision = evaluate_superiority(
        protocol, intervals, evidence_class=evidence_class
    )
    practical = _evaluate_practical_significance(
        metrics,
        protocol,
        evidence_class=evidence_class,
        statistical_superiority_passed=decision.passed,
    )
    practical.update(_practical_heterogeneity(units, protocol, profile))
    analysis = {
        "schema_version": ANALYSIS_SCHEMA,
        "status": "complete",
        "profile": profile,
        "evidence_class": evidence_class,
        "protocol_sha256": protocol_digest(protocol),
        "final_checkpoint_only": True,
        "independent_unit": "task_x_world_model_seed",
        "actor_seed_treatment": "nested_and_averaged_after_episode_means",
        "primary_nfe": protocol["evaluation"]["primary_nfe"],
        "rollout_horizons": protocol["evaluation"]["rollout_horizons"],
        "bootstrap": {
            "method": "paired_world_model_indices_resampled_within_each_task",
            "resamples": next(iter(metrics.values()))[PRIMARY_METRICS[0]][
                "bootstrap_resamples"
            ],
            "interval_confidence": protocol["statistics"]["multiplicity"][
                "interval_confidence"
            ],
        },
        "tracks": metrics,
        "superiority": decision.to_dict(),
        "practical_significance": practical,
        "raw_units": json.loads(json.dumps(units)),
    }
    return analysis, indices


def _evaluate_practical_significance(
    metrics: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    evidence_class: str,
    statistical_superiority_passed: bool,
) -> dict[str, Any]:
    """Apply the outcome-blind practical-effect thresholds frozen in the protocol."""

    specification = protocol["statistics"]["practical_significance"]
    thresholds = specification["thresholds"]
    tracks: dict[str, Any] = {}
    all_thresholds_met = True
    for track in BUDGET_TRACK_ORDER:
        rollout = metrics[track][PRIMARY_METRICS[0]]
        rollout_absolute = float(rollout["contrast"])
        shortcut_auc = float(rollout["arm_iqm"]["shortcut_forcing"])
        rollout_relative = rollout_absolute / max(shortcut_auc, 1e-6)
        rollout_spec = thresholds[PRIMARY_METRICS[0]]
        rollout_passed = (
            rollout_absolute
            >= float(rollout_spec["minimum_absolute_iqm_contrast"])
            and rollout_relative
            >= float(rollout_spec["minimum_relative_iqm_reduction"])
        )

        actor = metrics[track][PRIMARY_METRICS[1]]
        actor_absolute = float(actor["contrast"])
        actor_spec = thresholds[PRIMARY_METRICS[1]]
        actor_passed = actor_absolute >= float(
            actor_spec["minimum_absolute_iqm_contrast"]
        )
        all_thresholds_met = all_thresholds_met and rollout_passed and actor_passed
        tracks[track] = {
            PRIMARY_METRICS[0]: {
                "absolute_iqm_contrast": rollout_absolute,
                "relative_iqm_reduction": rollout_relative,
                "minimum_absolute_iqm_contrast": float(
                    rollout_spec["minimum_absolute_iqm_contrast"]
                ),
                "minimum_relative_iqm_reduction": float(
                    rollout_spec["minimum_relative_iqm_reduction"]
                ),
                "passed": bool(rollout_passed),
            },
            PRIMARY_METRICS[1]: {
                "absolute_iqm_contrast": actor_absolute,
                "minimum_absolute_iqm_contrast": float(
                    actor_spec["minimum_absolute_iqm_contrast"]
                ),
                "raw_dmc_return_equivalent": float(
                    actor_spec["raw_dmc_return_equivalent"]
                ),
                "passed": bool(actor_passed),
            },
        }
    eligible = evidence_class == "confirmatory"
    return {
        "status": "complete",
        "claim_role": specification["claim_role"],
        "evidence_class": evidence_class,
        "eligible_evidence_class": "confirmatory",
        "all_thresholds_met": bool(all_thresholds_met),
        "statistical_superiority_also_required": True,
        "statistical_superiority_passed": bool(statistical_superiority_passed),
        "practically_meaningful_superiority_claim_allowed": bool(
            eligible and all_thresholds_met and statistical_superiority_passed
        ),
        "tracks": tracks,
    }


def _practical_heterogeneity(
    units: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    profile: str,
) -> dict[str, Any]:
    """Report paired task/seed effects without treating nested actors as replicates."""

    keyed = _validate_complete_units(units, protocol, profile)
    selected = protocol["profiles"][profile]
    tasks = [canonical_task_id(task) for task in selected["tasks"]]
    seeds = [int(seed) for seed in selected["world_model_seeds"]]
    per_task: dict[str, Any] = {}
    across_task: dict[str, Any] = {}
    for track in BUDGET_TRACK_ORDER:
        per_task[track] = {}
        task_rollout: list[float] = []
        task_actor: list[float] = []
        for task in tasks:
            rollout_differences = np.asarray(
                [
                    float(keyed[(track, task, seed)]["rollout_auc"]["shortcut_forcing"])
                    - float(keyed[(track, task, seed)]["rollout_auc"]["trajectory_imf"])
                    for seed in seeds
                ],
                dtype=np.float64,
            )
            actor_differences = np.asarray(
                [
                    _nested_actor_mean(keyed[(track, task, seed)], "trajectory_imf")
                    - _nested_actor_mean(keyed[(track, task, seed)], "shortcut_forcing")
                    for seed in seeds
                ],
                dtype=np.float64,
            )
            rollout_contrast = interquartile_mean(rollout_differences)
            actor_contrast = interquartile_mean(actor_differences)
            task_rollout.append(rollout_contrast)
            task_actor.append(actor_contrast)
            per_task[track][task] = {
                PRIMARY_METRICS[0]: {
                    "paired_seed_contrasts": rollout_differences.tolist(),
                    "paired_seed_contrast_iqm": rollout_contrast,
                    "paired_seed_contrast_mean": float(np.mean(rollout_differences)),
                    "paired_seed_contrast_standard_deviation": float(
                        np.std(
                            rollout_differences,
                            ddof=1 if rollout_differences.size > 1 else 0,
                        )
                    ),
                    "favorable_seed_fraction": float(np.mean(rollout_differences > 0.0)),
                },
                PRIMARY_METRICS[1]: {
                    "paired_seed_contrasts": actor_differences.tolist(),
                    "paired_seed_contrast_iqm": actor_contrast,
                    "paired_seed_contrast_mean": float(np.mean(actor_differences)),
                    "paired_seed_contrast_standard_deviation": float(
                        np.std(
                            actor_differences,
                            ddof=1 if actor_differences.size > 1 else 0,
                        )
                    ),
                    "favorable_seed_fraction": float(np.mean(actor_differences > 0.0)),
                },
            }
        across_task[track] = {}
        for metric, values in (
            (PRIMARY_METRICS[0], task_rollout),
            (PRIMARY_METRICS[1], task_actor),
        ):
            array = np.asarray(values, dtype=np.float64)
            across_task[track][metric] = {
                "task_contrast_minimum": float(np.min(array)),
                "task_contrast_maximum": float(np.max(array)),
                "task_contrast_standard_deviation": float(
                    np.std(array, ddof=1 if array.size > 1 else 0)
                ),
                "favorable_task_fraction": float(np.mean(array > 0.0)),
            }
    return {
        "per_task": per_task,
        "across_task_heterogeneity": across_task,
    }


def synthetic_units(
    protocol: Mapping[str, Any], profile: str, *, effect: float
) -> list[dict[str, Any]]:
    """Deterministic positive/adverse/boundary control without benchmark artifacts."""

    selected = protocol["profiles"][profile]
    units = []
    for track_index, track in enumerate(BUDGET_TRACK_ORDER):
        for task_index, task in enumerate(selected["tasks"]):
            for seed_index, world_seed in enumerate(selected["world_model_seeds"]):
                base = 0.2 + 0.01 * task_index + 0.001 * seed_index + 0.0001 * track_index
                actor_returns: dict[str, dict[str, list[float]]] = {}
                for arm in ARM_ORDER:
                    direction = effect if arm == "trajectory_imf" else 0.0
                    actor_returns[arm] = {
                        str(actor_seed): [
                            base + direction + episode * 1e-5 for episode in range(3)
                        ]
                        for actor_seed in selected[
                            "actor_seeds_nested_within_world_model_seed"
                        ]
                    }
                units.append(
                    {
                        "budget_track": track,
                        "task": canonical_task_id(task),
                        "world_model_seed": int(world_seed),
                        "rollout_auc": {
                            "shortcut_forcing": base + max(effect, 0.0),
                            "trajectory_imf": base + max(-effect, 0.0),
                        },
                        "actor_episode_returns": actor_returns,
                    }
                )
    return units


def _profile_execution(profile: str) -> Mapping[str, Any]:
    try:
        return EXECUTION_SPEC["profiles"][profile]
    except KeyError as error:
        raise ValueError(f"unknown execution profile {profile!r}") from error


def pilot_candidates(protocol: Mapping[str, Any], arm: str) -> list[dict[str, Any]]:
    """Enumerate the frozen equal-size objective-specific Cartesian grids."""

    validate_matched_objective_protocol(protocol)
    if arm not in ARM_ORDER:
        raise ValueError(f"unknown arm {arm!r}")
    search = protocol["hyperparameter_policy"]["search"][arm]
    if arm == "shortcut_forcing":
        candidates = [
            {"training_k_max": int(k_max), "objective_loss_scale": float(scale)}
            for k_max in search["training_k_max"]
            for scale in search["objective_loss_scale"]
        ]
    else:
        candidates = [
            {"imf_boundary_fraction": float(boundary), "objective_loss_scale": float(scale)}
            for boundary in search["imf_boundary_fraction"]
            for scale in search["objective_loss_scale"]
        ]
    expected = int(protocol["hyperparameter_policy"]["candidate_evaluations_per_arm"])
    if len(candidates) != expected:
        raise ValueError("pilot candidate grid does not match the frozen equal budget")
    return [
        {"candidate_id": f"{arm}-{object_sha256(value)[:12]}", "overrides": value}
        for value in candidates
    ]


def _validate_candidate(
    protocol: Mapping[str, Any], arm: str, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    canonical = {
        entry["candidate_id"]: entry for entry in pilot_candidates(protocol, arm)
    }
    if set(candidate) != {"candidate_id", "overrides"}:
        raise ValueError("candidate must contain exactly candidate_id and overrides")
    expected = canonical.get(str(candidate["candidate_id"]))
    if expected is None or candidate != expected:
        raise ValueError("candidate is outside the frozen pilot grid")
    return dict(expected["overrides"])


def _without_digest(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    payload = json.loads(json.dumps(value))
    payload.pop(field, None)
    return payload


def validate_hpo_trial_manifest(
    manifest: Mapping[str, Any] | None,
    protocol: Mapping[str, Any],
    *,
    source_sha256: str,
) -> dict[str, Any]:
    """Validate complete, artifact-bound results for all 24 frozen candidates."""

    if not isinstance(manifest, Mapping):
        raise ValueError("HPO trial manifest is required")
    required = {
        "schema_version",
        "status",
        "protocol_sha256",
        "source_sha256",
        "selection_profile",
        "selection_budget_track",
        "confirmatory_outcomes_accessed",
        "trials",
        "trial_manifest_sha256",
    }
    if set(manifest) != required:
        raise ValueError("HPO trial manifest keys are incomplete or contain extras")
    if (
        manifest["schema_version"] != "matched-objective-hpo-trials-v1"
        or manifest["status"] != "complete"
        or manifest["protocol_sha256"] != protocol_digest(protocol)
        or manifest["source_sha256"] != source_sha256
        or manifest["selection_profile"] != "pilot"
        or manifest["selection_budget_track"] != "equal_updates"
        or manifest["confirmatory_outcomes_accessed"] is not False
    ):
        raise ValueError("HPO trial manifest identity or leakage boundary is invalid")
    if manifest["trial_manifest_sha256"] != object_sha256(
        _without_digest(manifest, "trial_manifest_sha256")
    ):
        raise ValueError("HPO trial manifest digest mismatch")
    trials = manifest["trials"]
    if not isinstance(trials, list):
        raise ValueError("HPO trials must be a list")
    expected_candidates = {
        (arm, candidate["candidate_id"]): candidate
        for arm in ARM_ORDER
        for candidate in pilot_candidates(protocol, arm)
    }
    expected_units = {
        (canonical_task_id(task), int(seed))
        for task in protocol["profiles"]["pilot"]["tasks"]
        for seed in protocol["profiles"]["pilot"]["world_model_seeds"]
    }
    observed: set[tuple[str, str]] = set()
    for trial in trials:
        if not isinstance(trial, Mapping) or set(trial) != {
            "arm",
            "candidate_id",
            "overrides",
            "units",
        }:
            raise ValueError("HPO trial row has an invalid schema")
        key = (str(trial["arm"]), str(trial["candidate_id"]))
        if key in observed or key not in expected_candidates:
            raise ValueError("HPO trial candidate is duplicated or unregistered")
        observed.add(key)
        expected_candidate = expected_candidates[key]
        if trial["overrides"] != expected_candidate["overrides"]:
            raise ValueError("HPO trial overrides do not match its candidate id")
        units = trial["units"]
        if not isinstance(units, list):
            raise ValueError("HPO trial units must be a list")
        unit_keys: set[tuple[str, int]] = set()
        for unit in units:
            if not isinstance(unit, Mapping) or set(unit) != {
                "task",
                "world_model_seed",
                "rollout_auc",
                "nested_actor_return",
                "full_forward_backward_train_flops_per_update",
                "artifact_sha256s",
            }:
                raise ValueError("HPO unit has an invalid schema")
            unit_key = (canonical_task_id(str(unit["task"])), int(unit["world_model_seed"]))
            if unit_key in unit_keys:
                raise ValueError("HPO trial duplicates a task/world-model unit")
            unit_keys.add(unit_key)
            for metric in (
                "rollout_auc",
                "nested_actor_return",
                "full_forward_backward_train_flops_per_update",
            ):
                value = float(unit[metric])
                if not math.isfinite(value) or (metric.endswith("flops_per_update") and value <= 0.0):
                    raise ValueError("HPO unit metric is non-finite or invalid")
            hashes = unit["artifact_sha256s"]
            if (
                not isinstance(hashes, list)
                or len(hashes) != 4
                or any(
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                    for value in hashes
                )
            ):
                raise ValueError("HPO unit must bind world, rollout, and two actor artifacts")
        if unit_keys != expected_units:
            raise ValueError("HPO trial does not cover every frozen pilot task/world seed")
    if observed != set(expected_candidates):
        raise ValueError("HPO trial manifest does not contain all 12 candidates per arm")
    return json.loads(json.dumps(manifest))


def _selection_from_trials(
    trials: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for arm in ARM_ORDER:
        rows = [trial for trial in trials["trials"] if trial["arm"] == arm]
        summaries = []
        for trial in rows:
            rollout = interquartile_mean([float(unit["rollout_auc"]) for unit in trial["units"]])
            actor_return = interquartile_mean(
                [float(unit["nested_actor_return"]) for unit in trial["units"]]
            )
            flops = float(
                np.mean(
                    [
                        float(unit["full_forward_backward_train_flops_per_update"])
                        for unit in trial["units"]
                    ]
                )
            )
            summaries.append(
                {
                    "candidate_id": trial["candidate_id"],
                    "overrides": trial["overrides"],
                    "rollout_auc_iqm": rollout,
                    "negative_nested_actor_return_iqm": -actor_return,
                    "full_forward_backward_train_flops_per_update_mean": flops,
                    "config_hash": object_sha256(trial["overrides"]),
                }
            )
        rollout_order = sorted(
            summaries, key=lambda row: (row["rollout_auc_iqm"], row["config_hash"])
        )
        actor_order = sorted(
            summaries,
            key=lambda row: (row["negative_nested_actor_return_iqm"], row["config_hash"]),
        )
        rollout_rank = {row["candidate_id"]: index + 1 for index, row in enumerate(rollout_order)}
        actor_rank = {row["candidate_id"]: index + 1 for index, row in enumerate(actor_order)}
        for row in summaries:
            row["rollout_rank"] = rollout_rank[row["candidate_id"]]
            row["negative_actor_return_rank"] = actor_rank[row["candidate_id"]]
            row["rank_sum"] = row["rollout_rank"] + row["negative_actor_return_rank"]
        winner = min(
            summaries,
            key=lambda row: (
                row["rank_sum"],
                row["full_forward_backward_train_flops_per_update_mean"],
                row["config_hash"],
            ),
        )
        selected[arm] = winner
    return selected


def select_hpo_candidates(
    trial_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    source_sha256: str,
) -> dict[str, Any]:
    trials = validate_hpo_trial_manifest(
        trial_manifest, protocol, source_sha256=source_sha256
    )
    selection = {
        "schema_version": "matched-objective-hpo-selection-v1",
        "status": "complete",
        "protocol_sha256": protocol_digest(protocol),
        "source_sha256": source_sha256,
        "selection_profile": "pilot",
        "selection_budget_track": "equal_updates",
        "confirmatory_outcomes_accessed": False,
        "selection_rule": protocol["hyperparameter_policy"]["selection_rule"],
        "ranking_semantics": protocol["hyperparameter_policy"]["ranking_semantics"],
        "selection_aggregation": protocol["hyperparameter_policy"]["selection_aggregation"],
        "trial_manifest_sha256": trials["trial_manifest_sha256"],
        "trial_manifest": trials,
        "selected": _selection_from_trials(trials, protocol),
    }
    selection["selection_sha256"] = object_sha256(selection)
    return selection


def validate_hpo_selection_manifest(
    manifest: Mapping[str, Any] | None,
    protocol: Mapping[str, Any],
    *,
    source_sha256: str,
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise ValueError("confirmatory matrix requires a complete HPO selection manifest")
    required = {
        "schema_version",
        "status",
        "protocol_sha256",
        "source_sha256",
        "selection_profile",
        "selection_budget_track",
        "confirmatory_outcomes_accessed",
        "selection_rule",
        "ranking_semantics",
        "selection_aggregation",
        "trial_manifest_sha256",
        "trial_manifest",
        "selected",
        "selection_sha256",
    }
    if set(manifest) != required:
        raise ValueError("HPO selection manifest keys are incomplete or contain extras")
    trials = validate_hpo_trial_manifest(
        manifest["trial_manifest"], protocol, source_sha256=source_sha256
    )
    expected = select_hpo_candidates(trials, protocol, source_sha256=source_sha256)
    if manifest != expected:
        raise ValueError("HPO selection is not the deterministic result of all frozen trials")
    return json.loads(json.dumps(manifest))


def _selected_overrides(
    protocol: Mapping[str, Any],
    profile: str,
    arm: str,
    output_root: str | Path | None,
    candidate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if candidate is not None:
        return _validate_candidate(protocol, arm, candidate)
    if profile == "pilot":
        raise ValueError("pilot execution requires one explicit frozen-grid candidate")
    if profile == "smoke":
        return {
            "objective_loss_scale": 1.0,
            **(
                {"training_k_max": 4}
                if arm == "shortcut_forcing"
                else {"imf_boundary_fraction": 0.5}
            ),
        }
    if output_root is None:
        raise ValueError("confirmatory configuration requires an HPO selection artifact")
    selection = read_json(Path(output_root) / "hpo_selection.json")
    if (
        selection.get("schema_version") != "matched-objective-hpo-selection-v1"
        or selection.get("status") != "complete"
        or selection.get("protocol_sha256") != protocol_digest(protocol)
        or selection.get("selection_profile") != "pilot"
        or selection.get("selection_budget_track") != "equal_updates"
        or selection.get("confirmatory_outcomes_accessed") is not False
        or set(selection.get("selected", {})) != set(ARM_ORDER)
    ):
        raise ValueError("HPO selection artifact identity or leakage boundary is invalid")
    selected = selection["selected"][arm]
    candidate_value = {
        "candidate_id": selected.get("candidate_id"),
        "overrides": selected.get("overrides"),
    }
    return _validate_candidate(protocol, arm, candidate_value)


def make_config(
    protocol: Mapping[str, Any],
    profile: str,
    arm: str,
    observation_shape: Sequence[int],
    action_dim: int,
    *,
    nfe: int | None = None,
    output_root: str | Path | None = None,
    candidate: Mapping[str, Any] | None = None,
) -> Any:
    """Build one shared-trunk config; only objective-required fields differ."""

    validate_matched_objective_protocol(protocol)
    if arm not in ARM_ORDER:
        raise ValueError(f"unknown matched-objective arm {arm!r}")
    if nfe is not None and int(nfe) not in NFE_FRONTIER:
        raise ValueError("requested NFE is outside the frozen 1/2/4 frontier")
    from imf_dreamer_jax import DreamerConfig

    from dataclasses import fields

    common = dict(
        protocol["canonical_executable_config"]["dreamer_config_non_task_shape_common"]
    )
    # These fields were added after the matched-objective protocol was frozen.
    # Resolve them explicitly to a no-op rather than changing that protocol's
    # identity or silently relying on future library defaults.
    common.update(
        imf_causal_consistency_scale=0.0,
        imf_causal_reward_scale=1.0,
        imf_causal_huber_delta=1.0,
        imf_causal_normalization_epsilon=1e-3,
    )
    common["overshooting_distances"] = tuple(common["overshooting_distances"])
    common["observation_shape"] = tuple(int(value) for value in observation_shape)
    common["action_dim"] = int(action_dim)
    overrides = _selected_overrides(
        protocol, profile, arm, output_root, candidate
    )
    common["prior_scale"] = float(overrides["objective_loss_scale"])
    requested_nfe = int(nfe) if nfe is not None else int(
        protocol["evaluation"]["primary_nfe"][arm]
    )
    if arm == "shortcut_forcing":
        common.update(
            prior="shortcut",
            shortcut_training_k_max=int(overrides["training_k_max"]),
            shortcut_sampling_steps=requested_nfe,
            imf_sampling_steps=1,
            imf_trajectory_enabled=False,
        )
    else:
        common.update(
            prior="imf",
            shortcut_training_k_max=None,
            shortcut_sampling_steps=4,
            imf_sampling_steps=requested_nfe,
            imf_trajectory_enabled=True,
            imf_boundary_fraction=float(overrides["imf_boundary_fraction"]),
        )
    expected_fields = {field.name for field in fields(DreamerConfig)}
    if set(common) != expected_fields:
        raise ValueError(
            "resolved DreamerConfig is incomplete; "
            f"missing={sorted(expected_fields - set(common))}, "
            f"extra={sorted(set(common) - expected_fields)}"
        )
    return DreamerConfig(**common)


def _dummy_batch(config: Any, profile: str) -> dict[str, Any]:
    import jax.numpy as jnp

    del profile
    # FLOP accounting always uses the registered static signature.  The smoke
    # runner may use a smaller runtime batch, but that non-claim override is
    # never used to allocate the confirmatory equal-FLOP track.
    batch = 32
    steps = 32
    mask = np.ones((batch, steps), dtype=np.float32)
    mask[:, : config.burn_in] = 0.0
    is_first = np.zeros((batch, steps), dtype=np.bool_)
    is_first[:, 0] = True
    return {
        "observations": jnp.zeros((batch, steps, *config.observation_shape), jnp.float32),
        "actions": jnp.zeros((batch, steps, config.action_dim), jnp.float32),
        "rewards": jnp.zeros((batch, steps), jnp.float32),
        "continuations": jnp.ones((batch, steps), jnp.float32),
        "is_first": jnp.asarray(is_first),
        "loss_mask": jnp.asarray(mask),
    }


def actor_update_from_batch(state: Any, batch: Mapping[str, Any], key: Any, config: Any):
    """One complete frozen-world actor/critic update used for execution and costing."""

    import jax
    from imf_dreamer_jax import (
        diverse_imagination_starts,
        observe_sequence,
        train_actor_critic,
    )

    observe_key, start_key, actor_key = jax.random.split(key, 3)
    sequence = observe_sequence(
        state.params.world_model,
        batch["observations"],
        batch["actions"],
        observe_key,
        config,
        is_first=batch["is_first"],
    )
    starts = diverse_imagination_starts(sequence.states, config.burn_in, start_key)
    return train_actor_critic(state, starts, actor_key, config)


def inference_transition(
    params: Any, state: Any, action: Any, noise: Any, config: Any
) -> tuple[Any, Any, Any, Any]:
    """One generated recurrent transition for compiler FLOP and structural NFE evidence."""

    from imf_dreamer_jax import (
        RSSMState,
        decode,
        predict_continuation_logits,
        predict_reward,
        sample_prior_with_nfe,
        transition_deterministic,
    )

    deterministic = transition_deterministic(params, state, action, config)
    sampled = sample_prior_with_nfe(params, deterministic, None, config, noise=noise)
    next_state = RSSMState(deterministic, sampled.stochastic)
    feature = next_state.feature
    return (
        next_state,
        decode(params, feature, config),
        predict_reward(params, feature, config),
        predict_continuation_logits(params, feature),
    )


@partial(__import__("jax").jit, static_argnames=("config",))
def posterior_mean_filter(
    params: Any, observations: Any, actions: Any, config: Any
) -> Any:
    """Filter context windows without evaluation-time posterior noise."""

    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import (
        RSSMState,
        encode,
        initial_state,
        posterior,
        transition_deterministic,
    )

    start = initial_state(config, observations.shape[0])

    def step(state: Any, inputs: tuple[Any, Any]):
        observation, action = inputs
        deterministic = transition_deterministic(params, state, action, config)
        distribution = posterior(
            params, deterministic, encode(params, observation, config), config
        )
        next_state = RSSMState(deterministic, distribution.mean)
        return next_state, jnp.asarray(0.0)

    state, _ = jax.lax.scan(
        step,
        start,
        (jnp.swapaxes(observations, 0, 1), jnp.swapaxes(actions, 0, 1)),
    )
    return state


def open_loop_samples_with_continuation(
    params: Any, start: Any, actions: Any, noise: Any, config: Any
) -> tuple[Any, Any, Any]:
    """Free-run paired draws and retain observation, reward, and continuation heads."""

    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import (
        RSSMState,
        decode,
        predict_continuation_logits,
        predict_reward,
        sample_prior,
        transition_deterministic,
    )

    draws = noise.shape[0]
    initial = RSSMState(
        jnp.broadcast_to(start.deterministic[None], (draws, *start.deterministic.shape)),
        jnp.broadcast_to(start.stochastic[None], (draws, *start.stochastic.shape)),
    )

    def step(state: Any, inputs: tuple[Any, Any]):
        action, step_noise = inputs
        action = jnp.broadcast_to(action[None], (draws, *action.shape))
        deterministic = transition_deterministic(params, state, action, config)
        flat_stochastic, _ = sample_prior(
            params,
            deterministic.reshape((-1, config.deterministic_dim)),
            None,
            config,
            noise=step_noise.reshape((-1, config.stochastic_dim)),
        )
        stochastic = flat_stochastic.reshape(
            (*deterministic.shape[:-1], config.stochastic_dim)
        )
        next_state = RSSMState(deterministic, stochastic)
        feature = next_state.feature
        return next_state, (
            decode(params, feature, config),
            predict_reward(params, feature, config),
            jax.nn.sigmoid(predict_continuation_logits(params, feature)),
        )

    _, (observations, rewards, continuations) = jax.lax.scan(
        step,
        initial,
        (jnp.swapaxes(actions, 0, 1), jnp.swapaxes(noise, 0, 1)),
    )
    observation_axes = (1, 2, 0, *range(3, observations.ndim))
    return (
        jnp.transpose(observations, observation_axes),
        jnp.transpose(rewards, (1, 2, 0)),
        jnp.transpose(continuations, (1, 2, 0)),
    )


def _compiled_flops(compiled: Any, name: str) -> float:
    analysis = compiled.cost_analysis()
    if analysis is None:
        raise RuntimeError(f"compiler cost analysis unavailable for {name}")
    rows = analysis if isinstance(analysis, (tuple, list)) else [analysis]
    value = sum(float(row.get("flops", 0.0)) for row in rows)
    return _finite_positive(value, f"{name} compiler FLOPs")


def _compiler_cost_analysis(compiled: Any, name: str) -> list[dict[str, float]]:
    analysis = compiled.cost_analysis()
    if analysis is None:
        raise RuntimeError(f"compiler cost analysis unavailable for {name}")
    rows = analysis if isinstance(analysis, (tuple, list)) else [analysis]
    result: list[dict[str, float]] = []
    for row in rows:
        converted = {str(key): float(value) for key, value in sorted(row.items())}
        if not all(math.isfinite(value) for value in converted.values()):
            raise FloatingPointError(f"non-finite compiler cost analysis for {name}")
        result.append(converted)
    return result


def _time_compiled_update(
    executable: Any,
    state: Any,
    second_argument: Any,
    key: Any,
    *,
    warmup_updates: int,
    timed_updates: int,
) -> dict[str, Any]:
    """Synchronously time a compiled state-update executable, excluding compile time."""

    import jax

    current = state
    for _ in range(warmup_updates):
        output = executable(current, second_argument, key)
        jax.block_until_ready(output)
        current = output[0]
    durations: list[float] = []
    for _ in range(timed_updates):
        jax.block_until_ready(current)
        started = time.perf_counter()
        output = executable(current, second_argument, key)
        jax.block_until_ready(output)
        durations.append(time.perf_counter() - started)
        current = output[0]
    values = np.asarray(durations, dtype=np.float64)
    return {
        "status": "complete",
        "warmup_updates": warmup_updates,
        "timed_updates": timed_updates,
        "seconds_per_update": durations,
        "median_seconds_per_update": float(np.median(values)),
        "interquartile_range_seconds_per_update": float(
            np.quantile(values, 0.75) - np.quantile(values, 0.25)
        ),
        "synchronization": "block_until_ready_before_and_after_each_timed_update",
        "compile_time_excluded": True,
    }


def _count_parameters(tree: Any) -> int:
    import jax

    return sum(
        int(np.asarray(leaf).size) for leaf in jax.tree_util.tree_leaves(tree)
    )


def _tree_digest(tree: Any) -> str:
    import jax

    arrays = {
        f"leaf_{index:06d}": np.asarray(value)
        for index, value in enumerate(jax.tree_util.tree_leaves(tree))
    }
    return array_sha256(arrays)


def runtime_fingerprint() -> dict[str, Any]:
    """Return the worker/compiler identity used to enforce homogeneous cells."""

    import jax
    import jaxlib

    devices = jax.devices()
    client = devices[0].client if devices else None
    return {
        "python": platform.python_version(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "backend": jax.default_backend(),
        "device_platforms": [str(device.platform) for device in devices],
        "device_kinds": [str(device.device_kind) for device in devices],
        "devices": [str(device) for device in devices],
        "visible_device_count": len(devices),
        "xla_platform_version": getattr(client, "platform_version", "unavailable"),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jax_enable_compilation_cache": os.environ.get(
            "JAX_ENABLE_COMPILATION_CACHE"
        ),
        "jax_compilation_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
        "jax_persistent_cache_min_compile_time_secs": os.environ.get(
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"
        ),
        "jax_persistent_cache_min_entry_size_bytes": os.environ.get(
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"
        ),
        "jax_persistent_cache_enable_xla_caches": os.environ.get(
            "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"
        ),
        "jax_raise_persistent_cache_errors": os.environ.get(
            "JAX_RAISE_PERSISTENT_CACHE_ERRORS"
        ),
    }


def runtime_homogeneity_identity(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Fields that must match across cells, excluding scheduler-assigned ordinals."""

    fields = (
        "python",
        "jax_version",
        "jaxlib_version",
        "backend",
        "device_platforms",
        "device_kinds",
        "visible_device_count",
        "xla_platform_version",
        "jax_enable_x64",
        "jax_enable_compilation_cache",
        "jax_compilation_cache_dir",
        "jax_persistent_cache_min_compile_time_secs",
        "jax_persistent_cache_min_entry_size_bytes",
        "jax_persistent_cache_enable_xla_caches",
        "jax_raise_persistent_cache_errors",
    )
    if any(field not in runtime for field in fields):
        raise ValueError("runtime homogeneity fingerprint is incomplete")
    return {field: runtime[field] for field in fields}


def _compute_plan_digest(plan: Mapping[str, Any]) -> str:
    payload = json.loads(json.dumps(plan))
    payload.pop("plan_sha256", None)
    for arm in payload.get("arms", {}).values():
        arm.pop("compile_wall_seconds", None)
    return object_sha256(payload)


def build_compute_plan(
    protocol: Mapping[str, Any],
    profile: str,
    task: str,
    observation_shape: Sequence[int],
    action_dim: int,
    *,
    source_sha256: str,
    output_root: str | Path | None = None,
    candidate_by_arm: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, str]]:
    """Compile full losses, freeze both track allocations, and retain compiler IR."""

    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import (
        create_agent,
        initial_state,
        jit_train_actor_critic,
        jit_train_world_model,
        world_model_loss,
        world_model_parameter_counts,
    )

    selected = protocol["profiles"][profile]
    executables: dict[str, dict[str, Any]] = {}
    hlo_text: dict[str, str] = {}
    rows: dict[str, dict[str, Any]] = {}
    world_costs: dict[str, float] = {}
    actor_costs: dict[str, float] = {}
    for arm_index, arm in enumerate(ARM_ORDER):
        config = make_config(
            protocol,
            profile,
            arm,
            observation_shape,
            action_dim,
            output_root=output_root,
            candidate=(candidate_by_arm or {}).get(arm),
        )
        dummy = _dummy_batch(config, profile)
        state = create_agent(config, derive_jax_key("compile", profile, task, arm_index))
        started = time.perf_counter()
        world_lowered = jit_train_world_model.lower(
            state,
            dummy,
            derive_jax_key("compile-world", profile, task, arm_index),
            config,
        )
        world_unoptimized = world_lowered.as_text()
        world_train = world_lowered.compile()
        world_train_seconds = time.perf_counter() - started
        forward_function = jax.jit(
            lambda params, batch, key: world_model_loss(params, batch, key, config).total
        )
        started = time.perf_counter()
        forward_lowered = forward_function.lower(
            state.params.world_model,
            dummy,
            derive_jax_key("compile-forward", profile, task, arm_index),
        )
        forward_unoptimized = forward_lowered.as_text()
        world_forward = forward_lowered.compile()
        forward_seconds = time.perf_counter() - started
        started = time.perf_counter()
        actor_lowered = jit_train_actor_critic.lower(
            state,
            initial_state(config, 32),
            derive_jax_key("compile-actor", profile, task, arm_index),
            config,
        )
        actor_unoptimized = actor_lowered.as_text()
        actor_train = actor_lowered.compile()
        actor_seconds = time.perf_counter() - started
        world_costs[arm] = _compiled_flops(world_train, f"{arm} world train")
        actor_costs[arm] = _compiled_flops(actor_train, f"{arm} actor train")
        counts = world_model_parameter_counts(state.params.world_model, config)
        actor_parameters = _count_parameters(state.params.actor)
        critic_parameters = _count_parameters(state.params.critic)
        inference: dict[str, Any] = {}
        inference_cost_analysis: dict[str, Any] = {}
        inference_executables: dict[int, Any] = {}
        for nfe in NFE_FRONTIER:
            inference_config = make_config(
                protocol,
                profile,
                arm,
                observation_shape,
                action_dim,
                nfe=nfe,
                output_root=output_root,
                candidate=(candidate_by_arm or {}).get(arm),
            )
            transition_function = jax.jit(
                lambda parameters, latent, action, noise: inference_transition(
                    parameters, latent, action, noise, inference_config
                )
            )
            inference_lowered = transition_function.lower(
                state.params.world_model,
                initial_state(inference_config, 1),
                jnp.zeros((1, action_dim), jnp.float32),
                jnp.zeros((1, inference_config.stochastic_dim), jnp.float32),
            )
            inference_unoptimized = inference_lowered.as_text()
            compiled = inference_lowered.compile()
            inference_executables[int(nfe)] = compiled
            key = f"{arm}.inference_nfe_{nfe}"
            text = compiled.as_text()
            hlo_text[key] = text
            hlo_text[f"{key}.unoptimized"] = inference_unoptimized
            inference[str(nfe)] = {
                "structural_nfe_per_transition": int(nfe),
                "compiler_forward_flops_per_transition": _compiled_flops(
                    compiled, key
                ),
                "compiler_ir_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "compiler_ir_unoptimized_sha256": hashlib.sha256(
                    inference_unoptimized.encode("utf-8")
                ).hexdigest(),
            }
            inference_cost_analysis[str(nfe)] = _compiler_cost_analysis(compiled, key)
        world_text = world_train.as_text()
        forward_text = world_forward.as_text()
        actor_text = actor_train.as_text()
        hlo_text[f"{arm}.world_train"] = world_text
        hlo_text[f"{arm}.world_train.unoptimized"] = world_unoptimized
        hlo_text[f"{arm}.world_forward"] = forward_text
        hlo_text[f"{arm}.world_forward.unoptimized"] = forward_unoptimized
        hlo_text[f"{arm}.actor_train"] = actor_text
        hlo_text[f"{arm}.actor_train.unoptimized"] = actor_unoptimized
        if profile == "smoke":
            synchronized_walltime = {
                "world_model": {
                    "status": "not_run_engineering_smoke",
                    "warmup_updates": 0,
                    "timed_updates": 0,
                },
                "actor": {
                    "status": "not_run_engineering_smoke",
                    "warmup_updates": 0,
                    "timed_updates": 0,
                },
            }
        else:
            timing = protocol["compute_accounting"]["walltime_protocol"]
            warmup_updates = int(timing["warmup_updates"])
            timed_updates = int(timing["timed_updates"])
            synchronized_walltime = {
                "world_model": _time_compiled_update(
                    world_train,
                    state,
                    dummy,
                    derive_jax_key("timing-world", profile, task, arm_index),
                    warmup_updates=warmup_updates,
                    timed_updates=timed_updates,
                ),
                "actor": _time_compiled_update(
                    actor_train,
                    state,
                    initial_state(config, 32),
                    derive_jax_key("timing-actor", profile, task, arm_index),
                    warmup_updates=warmup_updates,
                    timed_updates=timed_updates,
                ),
            }
        compiler_cost_analysis = {
            "world_train": _compiler_cost_analysis(world_train, f"{arm} world train"),
            "world_forward": _compiler_cost_analysis(world_forward, f"{arm} world forward"),
            "actor_train": _compiler_cost_analysis(actor_train, f"{arm} actor train"),
            "inference": inference_cost_analysis,
        }
        rows[arm] = {
            "runtime_config": asdict(config),
            "runtime_config_sha256": object_sha256(asdict(config)),
            "parameters": {
                "world_model_total": int(counts.total),
                "world_model_active": int(counts.active),
                "prior_total": int(counts.prior_total),
                "prior_active": int(counts.prior_active),
                "actor": actor_parameters,
                "critic": critic_parameters,
                "agent_total": int(counts.total) + actor_parameters + critic_parameters,
                "agent_active": int(counts.active) + actor_parameters + critic_parameters,
            },
            "compiler_flops": {
                "world_forward": _compiled_flops(world_forward, f"{arm} world forward"),
                "world_forward_and_backward_train": world_costs[arm],
                "actor_forward_and_backward_train": actor_costs[arm],
            },
            "compiler_ir_sha256": {
                "world_train": hashlib.sha256(world_text.encode("utf-8")).hexdigest(),
                "world_train_unoptimized": hashlib.sha256(world_unoptimized.encode("utf-8")).hexdigest(),
                "world_forward": hashlib.sha256(forward_text.encode("utf-8")).hexdigest(),
                "world_forward_unoptimized": hashlib.sha256(forward_unoptimized.encode("utf-8")).hexdigest(),
                "actor_train": hashlib.sha256(actor_text.encode("utf-8")).hexdigest(),
                "actor_train_unoptimized": hashlib.sha256(actor_unoptimized.encode("utf-8")).hexdigest(),
            },
            "compiler_cost_analysis": compiler_cost_analysis,
            "compiler_cost_analysis_sha256": object_sha256(compiler_cost_analysis),
            "inference": inference,
            "synchronized_walltime": synchronized_walltime,
            "compile_wall_seconds": {
                "world_train": world_train_seconds,
                "world_forward": forward_seconds,
                "actor_train": actor_seconds,
            },
        }
        executables[arm] = {
            "world_model": world_train,
            "actor": actor_train,
            "inference": inference_executables,
        }

    budgets = selected["budgets"]
    equal_world_updates = int(budgets["equal_updates"]["world_model_updates"])
    equal_actor_updates = int(budgets["equal_updates"]["actor_updates"])
    equal_updates = {
        arm: {
            "world_model": {
                "flops_per_update": world_costs[arm],
                "updates": equal_world_updates,
                "cumulative_flops": world_costs[arm] * equal_world_updates,
            },
            "actor": {
                "flops_per_update": actor_costs[arm],
                "updates": equal_actor_updates,
                "cumulative_flops": actor_costs[arm] * equal_actor_updates,
            },
        }
        for arm in ARM_ORDER
    }
    flop_budget = budgets["equal_compiler_flops"]
    tolerance = float(protocol["budget_tracks"]["equal_compiler_flops"]["relative_tolerance"])
    world_target, world_allocation = allocate_equal_flops(
        world_costs,
        reference_arm="shortcut_forcing",
        reference_updates=int(flop_budget["reference_world_model_updates"]),
        tolerance=tolerance,
        enforce_tolerance=profile == "confirmatory",
    )
    actor_target, actor_allocation = allocate_equal_flops(
        actor_costs,
        reference_arm="shortcut_forcing",
        reference_updates=int(flop_budget["reference_actor_updates"]),
        tolerance=tolerance,
        enforce_tolerance=profile == "confirmatory",
    )
    equal_flops = {
        arm: {"world_model": world_allocation[arm], "actor": actor_allocation[arm]}
        for arm in ARM_ORDER
    }
    plan = {
        "schema_version": COMPUTE_SCHEMA,
        "status": "frozen_before_outcomes",
        "profile": profile,
        "evidence_class": selected["evidence_class"],
        "protocol_sha256": protocol_digest(protocol),
        "source_sha256": source_sha256,
        "task": canonical_task_id(task),
        "resolved_task": resolved_task(task),
        "cost_source": protocol["budget_tracks"]["equal_compiler_flops"]["cost_source"],
        "relative_flop_tolerance": tolerance,
        "arms": rows,
        "tracks": {
            "equal_updates": {
                "claim_role": "primary_objective_isolation_track",
                "allocations": equal_updates,
            },
            "equal_compiler_flops": {
                "claim_role": "required_compute_robustness_track",
                "world_model_target_flops": world_target,
                "actor_target_flops": actor_target,
                "allocations": equal_flops,
            },
        },
        "runtime": runtime_fingerprint(),
    }
    active = {arm: rows[arm]["parameters"]["world_model_active"] for arm in ARM_ORDER}
    active_gap = abs(active["shortcut_forcing"] - active["trajectory_imf"]) / max(
        active.values()
    )
    trigger = float(protocol["parameter_matching"]["relative_active_parameter_gap_trigger"])
    plan["active_parameter_gap"] = {
        "scope": "world_model_active_parameters",
        "relative_gap": active_gap,
        "trigger": trigger,
        "supporting_width_sensitivity_required": active_gap > trigger,
        "claim_role": protocol["parameter_matching"]["triggered_sensitivity_claim_role"],
    }
    plan["plan_sha256"] = _compute_plan_digest(plan)
    return plan, executables, hlo_text


def validate_compute_plan(
    plan: Mapping[str, Any],
    protocol: Mapping[str, Any],
    profile: str,
    task: str,
    source_sha256: str,
) -> None:
    expected_identity = {
        "schema_version": COMPUTE_SCHEMA,
        "status": "frozen_before_outcomes",
        "profile": profile,
        "evidence_class": protocol["profiles"][profile]["evidence_class"],
        "protocol_sha256": protocol_digest(protocol),
        "source_sha256": source_sha256,
        "task": canonical_task_id(task),
        "resolved_task": resolved_task(task),
        "cost_source": protocol["budget_tracks"]["equal_compiler_flops"]["cost_source"],
    }
    if any(plan.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("compute plan identity mismatch")
    if plan.get("plan_sha256") != _compute_plan_digest(plan):
        raise ValueError("compute plan digest mismatch")
    if set(plan.get("arms", {})) != set(ARM_ORDER) or set(plan.get("tracks", {})) != set(
        BUDGET_TRACK_ORDER
    ):
        raise ValueError("compute plan arm or track set is incomplete")
    runtime = plan.get("runtime", {})
    if set(runtime) != {
        "python",
        "jax_version",
        "jaxlib_version",
        "backend",
        "device_platforms",
        "device_kinds",
        "devices",
        "visible_device_count",
        "xla_platform_version",
        "jax_enable_x64",
        "cuda_visible_devices",
        "jax_enable_compilation_cache",
        "jax_compilation_cache_dir",
        "jax_persistent_cache_min_compile_time_secs",
        "jax_persistent_cache_min_entry_size_bytes",
        "jax_persistent_cache_enable_xla_caches",
        "jax_raise_persistent_cache_errors",
    } or (
        runtime["jax_enable_x64"] is not False
        or runtime["visible_device_count"] != len(runtime["devices"])
        or runtime["visible_device_count"] != len(runtime["device_platforms"])
        or runtime["visible_device_count"] != len(runtime["device_kinds"])
        or runtime["visible_device_count"] <= 0
    ):
        raise ValueError("compute plan runtime fingerprint is incomplete or invalid")
    runtime_homogeneity_identity(runtime)
    if profile != "smoke" and runtime["visible_device_count"] != 1:
        raise ValueError("pilot/confirmatory timing requires one exclusively visible device")
    if profile != "smoke" and (
        runtime["jax_enable_compilation_cache"] != "true"
        or not runtime["jax_compilation_cache_dir"]
        or runtime["jax_persistent_cache_min_compile_time_secs"] != "0"
        or runtime["jax_persistent_cache_min_entry_size_bytes"] != "-1"
        or runtime["jax_persistent_cache_enable_xla_caches"]
        != "xla_gpu_per_fusion_autotune_cache_dir"
        or runtime["jax_raise_persistent_cache_errors"] != "true"
    ):
        raise ValueError(
            "pilot/confirmatory compiler evidence requires the fail-closed "
            "persistent compilation cache"
        )
    tolerance = float(protocol["budget_tracks"]["equal_compiler_flops"]["relative_tolerance"])
    if float(plan.get("relative_flop_tolerance", -1.0)) != tolerance:
        raise ValueError("compute plan tolerance drifted")
    for arm in ARM_ORDER:
        row = plan["arms"][arm]
        if row.get("runtime_config_sha256") != object_sha256(row.get("runtime_config")):
            raise ValueError("compute plan runtime config digest mismatch")
        if set(row.get("compiler_ir_sha256", {})) != {
            "world_train",
            "world_train_unoptimized",
            "world_forward",
            "world_forward_unoptimized",
            "actor_train",
            "actor_train_unoptimized",
        }:
            raise ValueError("compute plan optimized/unoptimized HLO digests are incomplete")
        cost_analysis = row.get("compiler_cost_analysis")
        if (
            not isinstance(cost_analysis, Mapping)
            or row.get("compiler_cost_analysis_sha256")
            != object_sha256(cost_analysis)
            or set(cost_analysis) != {
                "world_train",
                "world_forward",
                "actor_train",
                "inference",
            }
            or set(cost_analysis["inference"]) != {str(nfe) for nfe in NFE_FRONTIER}
        ):
            raise ValueError("compute plan raw compiler cost analysis is incomplete")
        parameters = row["parameters"]
        for key in (
            "world_model_total",
            "world_model_active",
            "prior_total",
            "prior_active",
            "actor",
            "critic",
            "agent_total",
            "agent_active",
        ):
            if not isinstance(parameters.get(key), int) or parameters[key] <= 0:
                raise ValueError("compute plan parameter count is invalid")
        if parameters["world_model_active"] > parameters["world_model_total"]:
            raise ValueError("active world parameters exceed total")
        if parameters["agent_active"] > parameters["agent_total"]:
            raise ValueError("active agent parameters exceed total")
        for value in row["compiler_flops"].values():
            _finite_positive(value, "compiler FLOP field")
        if set(row["inference"]) != {str(nfe) for nfe in NFE_FRONTIER}:
            raise ValueError("compute plan inference frontier is incomplete")
        for nfe in NFE_FRONTIER:
            inference = row["inference"][str(nfe)]
            if inference["structural_nfe_per_transition"] != nfe:
                raise ValueError("structural NFE mismatch")
            for digest_key in (
                "compiler_ir_sha256",
                "compiler_ir_unoptimized_sha256",
            ):
                digest = inference.get(digest_key)
                if not isinstance(digest, str) or len(digest) != 64:
                    raise ValueError("inference optimized/unoptimized HLO digest is invalid")
            _finite_positive(
                inference["compiler_forward_flops_per_transition"],
                "inference compiler FLOPs",
            )
        timing = row.get("synchronized_walltime", {})
        if set(timing) != {"world_model", "actor"}:
            raise ValueError("synchronized timing record is incomplete")
        for stage in ("world_model", "actor"):
            record = timing[stage]
            if profile == "smoke":
                if record != {
                    "status": "not_run_engineering_smoke",
                    "warmup_updates": 0,
                    "timed_updates": 0,
                }:
                    raise ValueError("smoke timing record is not explicitly non-claim")
            else:
                timing_protocol = protocol["compute_accounting"]["walltime_protocol"]
                if (
                    record.get("status") != "complete"
                    or record.get("warmup_updates") != timing_protocol["warmup_updates"]
                    or record.get("timed_updates") != timing_protocol["timed_updates"]
                    or len(record.get("seconds_per_update", []))
                    != timing_protocol["timed_updates"]
                    or record.get("synchronization")
                    != "block_until_ready_before_and_after_each_timed_update"
                    or record.get("compile_time_excluded") is not True
                ):
                    raise ValueError("synchronized update timing differs from the protocol")
                values = np.asarray(record["seconds_per_update"], dtype=np.float64)
                if values.ndim != 1 or not np.isfinite(values).all() or np.any(values <= 0.0):
                    raise ValueError("synchronized update timing contains invalid durations")
    equal_updates = plan["tracks"]["equal_updates"]["allocations"]
    expected_budgets = protocol["profiles"][profile]["budgets"]["equal_updates"]
    for arm in ARM_ORDER:
        for stage, budget_key in (
            ("world_model", "world_model_updates"),
            ("actor", "actor_updates"),
        ):
            allocation = equal_updates[arm][stage]
            if allocation["updates"] != expected_budgets[budget_key]:
                raise ValueError("equal-update allocation drifted")
            if allocation["cumulative_flops"] != allocation["flops_per_update"] * allocation["updates"]:
                raise ValueError("equal-update cumulative FLOPs are inconsistent")
    equal_flops = plan["tracks"]["equal_compiler_flops"]
    for stage, target_key in (
        ("world_model", "world_model_target_flops"),
        ("actor", "actor_target_flops"),
    ):
        target = _finite_positive(equal_flops[target_key], f"{stage} target")
        costs = {
            arm: float(equal_flops["allocations"][arm][stage]["flops_per_update"])
            for arm in ARM_ORDER
        }
        reference_updates = int(
            protocol["profiles"][profile]["budgets"]["equal_compiler_flops"][
                "reference_world_model_updates"
                if stage == "world_model"
                else "reference_actor_updates"
            ]
        )
        expected_target, expected = allocate_equal_flops(
            costs,
            reference_arm="shortcut_forcing",
            reference_updates=reference_updates,
            tolerance=tolerance,
            enforce_tolerance=profile == "confirmatory",
        )
        if target != expected_target:
            raise ValueError("equal-FLOP target differs from compiler-derived reference")
        for arm in ARM_ORDER:
            if equal_flops["allocations"][arm][stage] != expected[arm]:
                raise ValueError("equal-FLOP allocation is not the frozen nearest integer")
    active = {
        arm: plan["arms"][arm]["parameters"]["world_model_active"]
        for arm in ARM_ORDER
    }
    expected_gap = abs(active["shortcut_forcing"] - active["trajectory_imf"]) / max(
        active.values()
    )
    gap = plan.get("active_parameter_gap", {})
    trigger = float(protocol["parameter_matching"]["relative_active_parameter_gap_trigger"])
    if gap != {
        "scope": "world_model_active_parameters",
        "relative_gap": expected_gap,
        "trigger": trigger,
        "supporting_width_sensitivity_required": expected_gap > trigger,
        "claim_role": protocol["parameter_matching"]["triggered_sensitivity_claim_role"],
    }:
        raise ValueError("active-parameter-gap trigger record is inconsistent")


def validate_compute_plan_cell(
    plan: Mapping[str, Any],
    protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    cell: Mapping[str, Any],
    output_root: str | Path,
    *,
    rederive_compiler_evidence: bool = False,
) -> None:
    """Bind compiler evidence to its exact matrix cell and resolved candidates."""

    _cell_identity(cell, "compute_plan")
    validate_compute_plan(
        plan,
        protocol,
        cell["profile"],
        cell["task"],
        cell["source_sha256"],
    )
    if (
        plan.get("cell_id") != cell["cell_id"]
        or plan.get("matrix_sha256") != matrix["matrix_sha256"]
        or plan.get("candidate_id") != cell.get("candidate_id")
        or plan.get("selection_sha256") != cell.get("selection_sha256")
        or plan.get("config_template_sha256") != cell.get("config_template_sha256")
    ):
        raise ValueError("compute-plan matrix/candidate identity mismatch")
    candidates = _compute_candidates(matrix, cell)
    for arm in ARM_ORDER:
        runtime = plan["arms"][arm]["runtime_config"]
        expected = make_config(
            protocol,
            cell["profile"],
            arm,
            runtime["observation_shape"],
            int(runtime["action_dim"]),
            output_root=output_root,
            candidate=candidates.get(arm),
        )
        if object_sha256(runtime) != object_sha256(asdict(expected)):
            raise ValueError("compute-plan resolved runtime config differs from the frozen arm")
    if not rederive_compiler_evidence:
        return
    reference_runtime = plan["arms"][ARM_ORDER[0]]["runtime_config"]
    observation_shape = tuple(int(value) for value in reference_runtime["observation_shape"])
    action_dim = int(reference_runtime["action_dim"])
    if any(
        tuple(plan["arms"][arm]["runtime_config"]["observation_shape"])
        != observation_shape
        or int(plan["arms"][arm]["runtime_config"]["action_dim"]) != action_dim
        for arm in ARM_ORDER
    ):
        raise ValueError("compute-plan arms use different task observation/action shapes")
    rederived, _, rederived_hlo = build_compute_plan(
        protocol,
        str(cell["profile"]),
        str(cell["task"]),
        observation_shape,
        action_dim,
        source_sha256=str(cell["source_sha256"]),
        output_root=output_root,
        candidate_by_arm=candidates,
    )
    if runtime_homogeneity_identity(plan["runtime"]) != runtime_homogeneity_identity(
        rederived["runtime"]
    ):
        raise ValueError("compute-plan runtime differs from the current compiler runtime")
    compiler_fields = (
        "runtime_config",
        "runtime_config_sha256",
        "parameters",
        "compiler_flops",
        "compiler_ir_sha256",
        "compiler_cost_analysis",
        "compiler_cost_analysis_sha256",
        "inference",
    )
    for arm in ARM_ORDER:
        stored = {field: plan["arms"][arm][field] for field in compiler_fields}
        expected = {field: rederived["arms"][arm][field] for field in compiler_fields}
        if canonical_bytes(stored) != canonical_bytes(expected):
            raise ValueError(
                f"compute-plan compiler/parameter evidence does not rederive for {arm}"
            )
    directory = stage_directory(output_root, cell)
    for name, text in rederived_hlo.items():
        path = directory / ("hlo_" + name.replace(".", "_") + ".txt")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError("retained HLO does not match an independent recompilation")


def _rederive_compute_plan_worker(
    connection: Any,
    plan: Mapping[str, Any],
    protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    cell: Mapping[str, Any],
    output_root: str,
) -> None:
    """Run exact compiler rederivation in a fresh cache-reading process."""

    try:
        validate_compute_plan_cell(
            plan,
            protocol,
            matrix,
            cell,
            output_root,
            rederive_compiler_evidence=True,
        )
    except BaseException as error:
        connection.send(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    else:
        connection.send({"status": "verified"})
    finally:
        connection.close()


def _build_compute_plan_worker(
    connection: Any,
    protocol: Mapping[str, Any],
    profile: str,
    task: str,
    observation_shape: Sequence[int],
    action_dim: int,
    source_sha256: str,
    output_root: str,
    candidate_by_arm: Mapping[str, Mapping[str, Any]],
) -> None:
    """Build compiler evidence in a process that exits before rederivation."""

    try:
        plan, _, hlo_text = build_compute_plan(
            protocol,
            profile,
            task,
            tuple(int(value) for value in observation_shape),
            action_dim,
            source_sha256=source_sha256,
            output_root=output_root,
            candidate_by_arm=candidate_by_arm,
        )
    except BaseException as error:
        connection.send(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    else:
        connection.send(
            {
                "status": "built",
                "plan": plan,
                "hlo_text": hlo_text,
            }
        )
    finally:
        connection.close()


def build_compute_plan_in_fresh_process(
    protocol: Mapping[str, Any],
    profile: str,
    task: str,
    observation_shape: Sequence[int],
    action_dim: int,
    *,
    source_sha256: str,
    output_root: str | Path,
    candidate_by_arm: Mapping[str, Mapping[str, Any]],
    timeout_seconds: int = 900,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Build a plan, then close its compiler process so cache writes are durable."""

    context = multiprocessing.get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(
        target=_build_compute_plan_worker,
        args=(
            sending,
            dict(protocol),
            profile,
            task,
            tuple(int(value) for value in observation_shape),
            int(action_dim),
            source_sha256,
            str(Path(output_root).resolve()),
            {arm: dict(candidate) for arm, candidate in candidate_by_arm.items()},
        ),
        name="compute-plan-cache-writer",
    )
    process.start()
    sending.close()
    if not receiving.poll(timeout_seconds):
        process.terminate()
        process.join()
        receiving.close()
        raise RuntimeError("fresh-process compute-plan build exceeded its bounded timeout")
    try:
        result = receiving.recv()
    except EOFError as error:
        raise RuntimeError(
            "fresh-process compute-plan build exited without an authenticated result"
        ) from error
    finally:
        receiving.close()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        raise RuntimeError("fresh-process compute-plan build did not exit cleanly")
    if process.exitcode != 0 or result.get("status") != "built":
        detail = result.get("error")
        if not isinstance(detail, str) or not detail:
            detail = f"exit code {process.exitcode}"
        raise RuntimeError(
            "fresh-process compute-plan build failed: "
            f"{result.get('error_type', 'ProcessError')}: {detail}"
        )
    plan = result.get("plan")
    hlo_text = result.get("hlo_text")
    if not isinstance(plan, dict) or not isinstance(hlo_text, dict):
        raise RuntimeError("fresh-process compute-plan build returned malformed evidence")
    return plan, hlo_text


def validate_compute_plan_cell_in_fresh_process(
    plan: Mapping[str, Any],
    protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    cell: Mapping[str, Any],
    output_root: str | Path,
    *,
    timeout_seconds: int = 900,
) -> None:
    """Require exact rederivation after reopening the persistent JAX cache."""

    context = multiprocessing.get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(
        target=_rederive_compute_plan_worker,
        args=(
            sending,
            dict(plan),
            dict(protocol),
            dict(matrix),
            dict(cell),
            str(Path(output_root).resolve()),
        ),
        name="compute-plan-independent-rederivation",
    )
    process.start()
    sending.close()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        receiving.close()
        raise RuntimeError(
            "fresh-process compute-plan rederivation exceeded its bounded timeout"
        )
    try:
        result = receiving.recv()
    except EOFError as error:
        raise RuntimeError(
            "fresh-process compute-plan rederivation exited without an authenticated result"
        ) from error
    finally:
        receiving.close()
    if process.exitcode != 0 or result.get("status") != "verified":
        detail = result.get("error")
        if not isinstance(detail, str) or not detail:
            detail = f"exit code {process.exitcode}"
        raise RuntimeError(
            "fresh-process compute-plan rederivation failed: "
            f"{result.get('error_type', 'ProcessError')}: {detail}"
        )


def _expected_hlo_file_records(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        row = plan["arms"][arm]
        named_digests = {
            f"{arm}.world_train": row["compiler_ir_sha256"]["world_train"],
            f"{arm}.world_train.unoptimized": row["compiler_ir_sha256"][
                "world_train_unoptimized"
            ],
            f"{arm}.world_forward": row["compiler_ir_sha256"]["world_forward"],
            f"{arm}.world_forward.unoptimized": row["compiler_ir_sha256"][
                "world_forward_unoptimized"
            ],
            f"{arm}.actor_train": row["compiler_ir_sha256"]["actor_train"],
            f"{arm}.actor_train.unoptimized": row["compiler_ir_sha256"][
                "actor_train_unoptimized"
            ],
        }
        for nfe in NFE_FRONTIER:
            inference = row["inference"][str(nfe)]
            named_digests[f"{arm}.inference_nfe_{nfe}"] = inference[
                "compiler_ir_sha256"
            ]
            named_digests[f"{arm}.inference_nfe_{nfe}.unoptimized"] = inference[
                "compiler_ir_unoptimized_sha256"
            ]
        for name, digest in named_digests.items():
            records.append(
                {
                    "name": name,
                    "path": "hlo_" + name.replace(".", "_") + ".txt",
                    "sha256": digest,
                }
            )
    return sorted(records, key=lambda row: str(row["name"]))


def _validate_compute_files(plan: Mapping[str, Any], directory: str | Path) -> None:
    root = Path(directory)
    expected_hlo = _expected_hlo_file_records(plan)
    if plan.get("hlo_files") != expected_hlo:
        raise ValueError("compute-plan HLO file manifest is not canonical")
    actual_hlo_names = sorted(path.name for path in root.glob("hlo_*.txt"))
    if actual_hlo_names != sorted(str(entry["path"]) for entry in expected_hlo):
        raise ValueError("compute-plan retained HLO file set is not exact")
    cost_path = root / "compiler_cost_analysis.json"
    expected_cost = {
        arm: plan["arms"][arm]["compiler_cost_analysis"] for arm in ARM_ORDER
    }
    if not cost_path.is_file() or read_json(cost_path) != expected_cost:
        raise ValueError("retained compiler cost analysis differs from the plan")
    expected_entries = [
        *expected_hlo,
        {
            "name": "compiler_cost_analysis",
            "path": "compiler_cost_analysis.json",
            "sha256": file_sha256(cost_path),
        },
    ]
    entries = plan.get("compiler_artifact_files", [])
    if entries != expected_entries:
        raise ValueError("compute-plan compiler artifact manifest is not canonical")
    for entry in entries:
        if set(entry) != {"name", "path", "sha256"}:
            raise ValueError("compute-plan compiler artifact entry is malformed")
        path = root / str(entry["path"])
        if not path.is_file() or entry["sha256"] != file_sha256(path):
            raise ValueError("retained compiler artifact digest mismatch")


def stage_directory(output_root: str | Path, cell: Mapping[str, Any]) -> Path:
    return Path(output_root) / str(cell["stage"]) / str(cell["cell_id"])


def _cell_result_path(output_root: str | Path, cell: Mapping[str, Any]) -> Path:
    return stage_directory(output_root, cell) / "result.json"


def _cell_identity(cell: Mapping[str, Any], stage: str) -> dict[str, Any]:
    if cell.get("stage") != stage:
        raise ValueError(f"expected {stage!r} cell, got {cell.get('stage')!r}")
    identity = {
        key: cell.get(key)
        for key in (
            "stage",
            "profile",
            "evidence_class",
            "protocol_sha256",
            "source_sha256",
            "task",
            "resolved_task",
            "world_model_seed",
            "budget_track",
            "arm",
            "actor_seed",
            "nfe",
            "candidate_id",
            "selection_sha256",
            "config_template_sha256",
            "dependencies",
        )
    }
    digest = object_sha256(identity)
    if cell.get("identity_sha256") != digest or cell.get("cell_id") != f"{stage}-{digest[:24]}":
        raise ValueError("cell identity digest mismatch")
    return identity


def _result_identity(cell: Mapping[str, Any], schema: str) -> dict[str, Any]:
    return {
        "schema_version": schema,
        "status": "complete",
        "stage": cell["stage"],
        "cell_id": cell["cell_id"],
        "profile": cell["profile"],
        "evidence_class": cell["evidence_class"],
        "protocol_sha256": cell["protocol_sha256"],
        "source_sha256": cell["source_sha256"],
        "task": cell["task"],
        "world_model_seed": cell["world_model_seed"],
        "budget_track": cell["budget_track"],
        "arm": cell["arm"],
        "actor_seed": cell["actor_seed"],
        "nfe": cell["nfe"],
        "candidate_id": cell["candidate_id"],
        "selection_sha256": cell["selection_sha256"],
        "config_template_sha256": cell["config_template_sha256"],
    }


def _dependency_cell(
    matrix: Mapping[str, Any], cell: Mapping[str, Any], stage: str
) -> Mapping[str, Any]:
    matching = [
        candidate
        for candidate in matrix["cells"]
        if candidate["cell_id"] in cell["dependencies"] and candidate["stage"] == stage
    ]
    if len(matching) != 1:
        raise ValueError(f"cell must have exactly one {stage} dependency")
    return matching[0]


def _load_dependency_result(
    output_root: str | Path,
    matrix: Mapping[str, Any],
    cell: Mapping[str, Any],
    stage: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Path]:
    dependency = _dependency_cell(matrix, cell, stage)
    path = _cell_result_path(output_root, dependency)
    if not path.is_file():
        raise FileNotFoundError(f"missing {stage} dependency {dependency['cell_id']}")
    return dependency, read_json(path), stage_directory(output_root, dependency)


def _cell_directory_fingerprint(directory: str | Path) -> tuple[tuple[str, int, str], ...]:
    """Hash the exact regular-file set retained by one completed matrix cell."""

    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("validated cell directory is absent, invalid, or a symlink")
    root = root.resolve(strict=True)
    candidates = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    entries: list[tuple[str, int, str]] = []
    for path in candidates:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"validated cell directory contains a symlink: {relative}")
        if not path.is_file():
            continue
        before = path.stat()
        digest = file_sha256(path)
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"validated cell file changed while it was hashed: {relative}")
        entries.append((relative, int(after.st_size), digest))
    observed_after = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    )
    if observed_after != [relative for relative, _, _ in entries]:
        raise ValueError("validated cell file set changed while it was fingerprinted")
    if not entries:
        raise ValueError("validated cell directory contains no retained files")
    return tuple(entries)


_VALIDATION_CACHE_CONSTRUCTION_TOKEN = object()


class _ValidatedResultCache:
    """Verifier-local semantic-validation cache bound to one immutable run root.

    Only :func:`_validate_all_cells` constructs and seals instances.  A cache hit
    re-authenticates the canonical result JSON and the path, size, and SHA-256 of
    every retained file in that cell directory.  This preserves one expensive
    semantic validation per cell without trusting state carried across roots or
    filesystem mutations.
    """

    __slots__ = (
        "_root",
        "_matrix_fingerprint",
        "_protocol_fingerprint",
        "_expected_cell_ids",
        "_entries",
        "_sealed",
    )

    def __init__(
        self,
        output_root: str | Path,
        matrix: Mapping[str, Any],
        protocol: Mapping[str, Any],
        *,
        _token: object,
    ) -> None:
        if _token is not _VALIDATION_CACHE_CONSTRUCTION_TOKEN:
            raise TypeError("validation caches are verifier-private")
        self._root = Path(output_root).resolve(strict=True)
        self._matrix_fingerprint = object_sha256(matrix)
        self._protocol_fingerprint = object_sha256(protocol)
        self._expected_cell_ids = frozenset(
            str(cell["cell_id"]) for cell in matrix["cells"]
        )
        self._entries: dict[
            str, tuple[bytes, Path, tuple[tuple[str, int, str], ...]]
        ] = {}
        self._sealed = False

    def _assert_context(
        self,
        output_root: str | Path,
        matrix: Mapping[str, Any],
        protocol: Mapping[str, Any],
    ) -> None:
        root = Path(output_root).resolve(strict=True)
        if root != self._root:
            raise ValueError("validated result cache belongs to another verification context")
        cell_ids = frozenset(str(cell["cell_id"]) for cell in matrix["cells"])
        if (
            object_sha256(matrix) != self._matrix_fingerprint
            or object_sha256(protocol) != self._protocol_fingerprint
            or cell_ids != self._expected_cell_ids
        ):
            raise ValueError("validated result cache belongs to another verification context")

    def contains(
        self,
        output_root: str | Path,
        matrix: Mapping[str, Any],
        protocol: Mapping[str, Any],
        cell: Mapping[str, Any],
    ) -> bool:
        self._assert_context(output_root, matrix, protocol)
        return str(cell["cell_id"]) in self._entries

    def _cell_directory(
        self, output_root: str | Path, cell: Mapping[str, Any]
    ) -> Path:
        directory = stage_directory(output_root, cell)
        if directory.is_symlink() or directory.parent.is_symlink():
            raise ValueError("validated cell path contains a symlink")
        resolved = directory.resolve(strict=True)
        try:
            resolved.relative_to(self._root)
        except ValueError as error:
            raise ValueError("validated cell directory escapes its verification root") from error
        return directory

    def record(
        self,
        output_root: str | Path,
        matrix: Mapping[str, Any],
        protocol: Mapping[str, Any],
        cell: Mapping[str, Any],
        result: Mapping[str, Any],
        result_path: Path,
        *,
        validated_fingerprint: tuple[tuple[str, int, str], ...],
    ) -> None:
        self._assert_context(output_root, matrix, protocol)
        if self._sealed:
            raise ValueError("a sealed validation cache cannot be extended")
        cell_id = str(cell["cell_id"])
        if cell_id not in self._expected_cell_ids or cell_id in self._entries:
            raise ValueError("validated result cache received an invalid or duplicate cell")
        directory = self._cell_directory(output_root, cell)
        expected_path = _cell_result_path(output_root, cell).resolve(strict=True)
        observed_path = result_path.resolve(strict=True)
        if observed_path != expected_path:
            raise ValueError("validated result cache result path is not canonical")
        observed_result = read_json(expected_path)
        result_bytes = canonical_bytes(result)
        if canonical_bytes(observed_result) != result_bytes:
            raise ValueError("validated result cache result differs from its canonical JSON")
        fingerprint = _cell_directory_fingerprint(directory)
        if fingerprint != validated_fingerprint:
            raise ValueError(
                "validated cell artifacts changed during semantic validation"
            )
        self._entries[cell_id] = (result_bytes, expected_path, fingerprint)

    def lookup(
        self,
        output_root: str | Path,
        matrix: Mapping[str, Any],
        protocol: Mapping[str, Any],
        cell: Mapping[str, Any],
    ) -> tuple[dict[str, Any], Path]:
        self._assert_context(output_root, matrix, protocol)
        cell_id = str(cell["cell_id"])
        entry = self._entries.get(cell_id)
        if entry is None:
            raise ValueError("validated dependency cache is not topologically complete")
        result_bytes, recorded_path, recorded_fingerprint = entry
        canonical_path = _cell_result_path(output_root, cell)
        expected_path = canonical_path.resolve(strict=True)
        if recorded_path != expected_path:
            raise ValueError("validated result cache path differs from the canonical result")
        observed_result = read_json(expected_path)
        if canonical_bytes(observed_result) != result_bytes:
            raise ValueError("validated result cache result changed after validation")
        current_fingerprint = _cell_directory_fingerprint(
            self._cell_directory(output_root, cell)
        )
        if current_fingerprint != recorded_fingerprint:
            raise ValueError("validated cell artifacts changed after semantic validation")
        return observed_result, canonical_path

    def seal(
        self,
        output_root: str | Path,
        matrix: Mapping[str, Any],
        protocol: Mapping[str, Any],
    ) -> None:
        self._assert_context(output_root, matrix, protocol)
        if set(self._entries) != set(self._expected_cell_ids):
            raise ValueError("validated result cache does not cover the exact matrix")
        self._sealed = True

    def require_complete(
        self,
        output_root: str | Path,
        matrix: Mapping[str, Any],
        protocol: Mapping[str, Any],
    ) -> None:
        self._assert_context(output_root, matrix, protocol)
        if not self._sealed or set(self._entries) != set(self._expected_cell_ids):
            raise ValueError("validated result cache is not a complete sealed verification pass")
        for cell in matrix["cells"]:
            self.lookup(output_root, matrix, protocol, cell)


def _new_validation_cache(
    output_root: str | Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> _ValidatedResultCache:
    return _ValidatedResultCache(
        output_root,
        matrix,
        protocol,
        _token=_VALIDATION_CACHE_CONSTRUCTION_TOKEN,
    )


def _require_validation_cache(
    cache: _ValidatedResultCache | None,
) -> _ValidatedResultCache | None:
    if cache is not None and type(cache) is not _ValidatedResultCache:
        raise TypeError("validation caches are verifier-private")
    return cache


def _dependency_was_prevalidated(
    validated_results: _ValidatedResultCache | None,
    output_root: str | Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
    dependency: Mapping[str, Any],
    result: Mapping[str, Any],
    result_path: Path,
) -> bool:
    """Authenticate one dependency against this verification pass's cache."""

    cache = _require_validation_cache(validated_results)
    if cache is None:
        return False
    cached_result, cached_path = cache.lookup(
        output_root, matrix, protocol, dependency
    )
    if (
        cached_path.resolve() != result_path.resolve()
        or canonical_bytes(cached_result) != canonical_bytes(result)
    ):
        raise ValueError("validated dependency cache differs from the canonical result")
    return True


def _validate_finite_arrays(arrays: Mapping[str, np.ndarray], names: Iterable[str]) -> None:
    for name in names:
        if name not in arrays or not np.isfinite(np.asarray(arrays[name])).all():
            raise ValueError(f"artifact array {name!r} is absent or non-finite")


def _expected_dataset_layout(
    protocol: Mapping[str, Any], profile: str
) -> tuple[int, int, int, bool]:
    """Return replay steps, episodes, steps/episode, and smoke truncation."""

    replay_steps = int(protocol["profiles"][profile]["replay_steps_per_task_world_seed"])
    if profile == "smoke":
        episodes = int(EXECUTION_SPEC["dataset"]["smoke_episodes"])
        steps = int(EXECUTION_SPEC["dataset"]["smoke_native_steps_per_episode"])
        if episodes * steps != replay_steps:
            raise ValueError(
                "smoke replay override must preserve the registered replay-step count"
            )
        return replay_steps, episodes, steps, True
    steps = int(protocol["data"]["native_episode_limit"])
    if replay_steps % steps:
        raise ValueError(
            "registered replay steps are not divisible by the native episode limit"
        )
    return replay_steps, replay_steps // steps, steps, False


def validate_dataset_result(
    result: Mapping[str, Any],
    cell: Mapping[str, Any],
    data_path: str | Path,
    *,
    protocol: Mapping[str, Any] | None = None,
    matrix: Mapping[str, Any] | None = None,
) -> None:
    _cell_identity(cell, "dataset")
    expected = _result_identity(cell, DATASET_SCHEMA)
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("dataset result identity mismatch")
    path = Path(data_path)
    if result.get("dataset_file_sha256") != file_sha256(path):
        raise ValueError("dataset file digest mismatch")
    arrays = load_npz(path)
    required = {
        "observations",
        "actions",
        "rewards",
        "continuations",
        "is_first",
        "episode_ids",
        "train_episode_ids",
        "test_episode_ids",
    }
    if set(arrays) != required or result.get("dataset_sha256") != array_sha256(arrays):
        raise ValueError("dataset payload or content digest mismatch")
    prefix = arrays["observations"].shape[:2]
    for name in ("actions", "rewards", "continuations", "is_first"):
        if arrays[name].shape[:2] != prefix:
            raise ValueError("dataset transition prefixes differ")
    if arrays["is_first"].dtype != np.bool_ or not arrays["is_first"][:, 0].all():
        raise ValueError("dataset reset mask is invalid")
    if arrays["is_first"][:, 1:].any():
        raise ValueError("dataset contains an unexpected mid-episode reset")
    episodes, transitions = prefix
    if (
        result.get("episodes") != episodes
        or result.get("steps_per_episode") != transitions - 1
        or result.get("native_action_steps") != episodes * (transitions - 1)
        or result.get("observation_shape") != list(arrays["observations"].shape[2:])
        or result.get("action_dim") != arrays["actions"].shape[-1]
    ):
        raise ValueError("dataset result shapes/counts differ from its payload")
    if (
        arrays["observations"].ndim < 3
        or arrays["actions"].ndim != 3
        or arrays["rewards"].ndim != 2
        or arrays["continuations"].ndim != 2
        or arrays["observations"].dtype != np.float32
        or arrays["actions"].dtype != np.float32
        or arrays["rewards"].dtype != np.float32
        or arrays["continuations"].dtype != np.float32
        or np.any(arrays["actions"] < -1.0)
        or np.any(arrays["actions"] > 1.0)
        or np.any(arrays["continuations"] < 0.0)
        or np.any(arrays["continuations"] > 1.0)
        or np.any(arrays["actions"][:, 0] != 0.0)
        or np.any(arrays["rewards"][:, 0] != 0.0)
        or np.any(arrays["continuations"][:, 0] != 1.0)
    ):
        raise ValueError("dataset tensor schema or initial dummy transitions are invalid")
    train = set(np.asarray(arrays["train_episode_ids"], dtype=np.int64).tolist())
    test = set(np.asarray(arrays["test_episode_ids"], dtype=np.int64).tolist())
    episodes = set(np.asarray(arrays["episode_ids"], dtype=np.int64).tolist())
    if not train or not test or train & test or train | test != episodes:
        raise ValueError("whole-episode train/test split is invalid")
    if (
        result.get("train_episode_ids") != arrays["train_episode_ids"].tolist()
        or result.get("test_episode_ids") != arrays["test_episode_ids"].tolist()
    ):
        raise ValueError("dataset result split ids differ from its payload")
    _validate_finite_arrays(arrays, ("observations", "actions", "rewards", "continuations"))
    if protocol is not None or matrix is not None:
        if protocol is None or matrix is None:
            raise ValueError("strong dataset validation requires protocol and matrix")
        replay_steps, expected_episodes, expected_steps, truncated = (
            _expected_dataset_layout(protocol, str(cell["profile"]))
        )
        expected_ids = np.arange(expected_episodes, dtype=np.int32)
        fraction = float(protocol["data"]["episode_split"]["train_fraction"])
        train_count = max(
            1,
            min(expected_episodes - 1, int(math.floor(expected_episodes * fraction))),
        )
        if (
            result.get("matrix_sha256") != matrix["matrix_sha256"]
            or result.get("resolved_task") != resolved_task(str(cell["task"]))
            or result.get("collection_policy") != protocol["data"]["collection_policy"]
            or result.get("native_action_steps") != replay_steps
            or result.get("episodes") != expected_episodes
            or result.get("steps_per_episode") != expected_steps
            or result.get("smoke_truncated_episodes") is not truncated
            or not math.isfinite(float(result.get("wall_seconds", math.nan)))
            or not np.array_equal(arrays["episode_ids"], expected_ids)
            or not np.array_equal(arrays["train_episode_ids"], expected_ids[:train_count])
            or not np.array_equal(arrays["test_episode_ids"], expected_ids[train_count:])
        ):
            raise ValueError("dataset differs from the frozen profile or canonical split")
        environment = DMCAdapter(
            str(cell["task"]),
            seed=derive_seed(
                "dataset-environment", cell["task"], cell["world_model_seed"]
            ),
            action_repeat=int(protocol["data"]["action_repeat"]),
        )
        try:
            if (
                result.get("observation_shape") != list(environment.observation_shape)
                or tuple(arrays["observations"].shape[2:])
                != tuple(environment.observation_shape)
                or result.get("action_dim") != int(environment.action_dim)
                or arrays["actions"].shape[-1] != int(environment.action_dim)
            ):
                raise ValueError("dataset shapes differ from the resolved environment spec")
        finally:
            environment.close()


def run_dataset_cell(
    cell: Mapping[str, Any],
    protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    output_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Collect one content-addressed replay shared by every paired arm/track."""

    _cell_identity(cell, "dataset")
    validate_matched_objective_protocol(protocol)
    if cell["protocol_sha256"] != protocol_digest(protocol):
        raise ValueError("dataset cell belongs to another protocol")
    directory = stage_directory(output_root, cell)
    data_path = directory / "dataset.npz"
    result_path = directory / "result.json"
    if not force and data_path.is_file() and result_path.is_file():
        result = read_json(result_path)
        validate_dataset_result(
            result, cell, data_path, protocol=protocol, matrix=matrix
        )
        return result

    replay_steps, episodes, steps, truncated = _expected_dataset_layout(
        protocol, str(cell["profile"])
    )
    environment = DMCAdapter(
        cell["task"],
        seed=derive_seed("dataset-environment", cell["task"], cell["world_model_seed"]),
        action_repeat=int(protocol["data"]["action_repeat"]),
    )
    observations = np.empty((episodes, steps + 1, *environment.observation_shape), np.float32)
    actions = np.zeros((episodes, steps + 1, environment.action_dim), np.float32)
    rewards = np.zeros((episodes, steps + 1), np.float32)
    continuations = np.ones((episodes, steps + 1), np.float32)
    is_first = np.zeros((episodes, steps + 1), np.bool_)
    random = np.random.default_rng(
        derive_seed("dataset-actions", cell["task"], cell["world_model_seed"])
    )
    uniform_probability = float(protocol["data"]["collection_policy"]["uniform_probability"])
    started = time.perf_counter()
    try:
        for episode in range(episodes):
            observations[episode, 0] = environment.reset()
            is_first[episode, 0] = True
            smooth = np.zeros(environment.action_dim, np.float32)
            uniform_episode = bool(random.random() < uniform_probability)
            for index in range(1, steps + 1):
                if uniform_episode:
                    action = random.uniform(-1.0, 1.0, environment.action_dim).astype(np.float32)
                else:
                    smooth = np.clip(
                        0.92 * smooth + random.normal(0.0, 0.35, environment.action_dim),
                        -1.0,
                        1.0,
                    ).astype(np.float32)
                    action = smooth
                transition = environment.step(action)
                observations[episode, index] = transition.observation
                actions[episode, index] = action
                rewards[episode, index] = transition.reward
                continuations[episode, index] = transition.continuation
                if transition.is_last and index < steps:
                    # Native time limits should coincide with the registered
                    # horizon; task termination before that is a real boundary.
                    observations[episode, index + 1 :] = transition.observation
                    continuations[episode, index + 1 :] = 0.0
                    break
    finally:
        environment.close()
    episode_ids = np.arange(episodes, dtype=np.int32)
    train_fraction = float(protocol["data"]["episode_split"]["train_fraction"])
    train_count = max(
        1, min(episodes - 1, int(math.floor(episodes * train_fraction)))
    )
    arrays = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "continuations": continuations,
        "is_first": is_first,
        "episode_ids": episode_ids,
        "train_episode_ids": episode_ids[:train_count],
        "test_episode_ids": episode_ids[train_count:],
    }
    _write_npz_atomic(data_path, arrays)
    result = {
        **_result_identity(cell, DATASET_SCHEMA),
        "matrix_sha256": matrix["matrix_sha256"],
        "resolved_task": resolved_task(cell["task"]),
        "collection_policy": protocol["data"]["collection_policy"],
        "native_action_steps": replay_steps,
        "episodes": episodes,
        "steps_per_episode": steps,
        "smoke_truncated_episodes": truncated,
        "observation_shape": list(environment.observation_shape),
        "action_dim": int(environment.action_dim),
        "train_episode_ids": arrays["train_episode_ids"].tolist(),
        "test_episode_ids": arrays["test_episode_ids"].tolist(),
        "dataset_sha256": array_sha256(arrays),
        "dataset_file_sha256": file_sha256(data_path),
        "wall_seconds": time.perf_counter() - started,
    }
    validate_dataset_result(result, cell, data_path, protocol=protocol, matrix=matrix)
    write_json_atomic(result_path, result)
    return result


def run_compute_plan_cell(
    cell: Mapping[str, Any],
    protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    output_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Compile both complete update paths and freeze integer budget allocation."""

    _cell_identity(cell, "compute_plan")
    directory = stage_directory(output_root, cell)
    result_path = directory / "result.json"
    if not force and result_path.is_file():
        result = read_json(result_path)
        validate_compute_plan_cell(
            result,
            protocol,
            matrix,
            cell,
            output_root,
            rederive_compiler_evidence=True,
        )
        _validate_compute_files(result, directory)
        return result
    dataset_cells = [
        candidate
        for candidate in matrix["cells"]
        if candidate["stage"] == "dataset" and candidate["task"] == cell["task"]
    ]
    if not dataset_cells:
        raise ValueError("compute plan has no task dataset from which to resolve shapes")
    exemplar_path = _cell_result_path(output_root, dataset_cells[0])
    if not exemplar_path.is_file():
        raise FileNotFoundError("compute plan requires one completed task dataset")
    exemplar = read_json(exemplar_path)
    plan, hlo_text = build_compute_plan_in_fresh_process(
        protocol,
        cell["profile"],
        cell["task"],
        exemplar["observation_shape"],
        int(exemplar["action_dim"]),
        source_sha256=cell["source_sha256"],
        output_root=output_root,
        candidate_by_arm=_compute_candidates(matrix, cell),
    )
    directory.mkdir(parents=True, exist_ok=True)
    hlo_files = []
    for name, payload in sorted(hlo_text.items()):
        filename = "hlo_" + name.replace(".", "_") + ".txt"
        path = directory / filename
        path.write_text(payload, encoding="utf-8")
        hlo_files.append({"name": name, "path": filename, "sha256": file_sha256(path)})
    cost_path = directory / "compiler_cost_analysis.json"
    write_json_atomic(
        cost_path,
        {arm: plan["arms"][arm]["compiler_cost_analysis"] for arm in ARM_ORDER},
    )
    compiler_files = [
        *hlo_files,
        {
            "name": "compiler_cost_analysis",
            "path": cost_path.name,
            "sha256": file_sha256(cost_path),
        },
    ]
    plan["hlo_files"] = hlo_files
    plan["compiler_artifact_files"] = compiler_files
    plan["cell_id"] = cell["cell_id"]
    plan["stage"] = "compute_plan"
    plan["matrix_sha256"] = matrix["matrix_sha256"]
    plan["candidate_id"] = cell.get("candidate_id")
    plan["selection_sha256"] = cell.get("selection_sha256")
    plan["config_template_sha256"] = cell.get("config_template_sha256")
    plan["plan_sha256"] = _compute_plan_digest(plan)
    validate_compute_plan_cell(
        plan,
        protocol,
        matrix,
        cell,
        output_root,
        rederive_compiler_evidence=False,
    )
    _validate_compute_files(plan, directory)
    validate_compute_plan_cell_in_fresh_process(
        plan,
        protocol,
        matrix,
        cell,
        output_root,
    )
    write_json_atomic(result_path, plan)
    return plan


def _batch_schedule(
    arrays: Mapping[str, np.ndarray],
    *,
    task: str,
    world_model_seed: int,
    updates: int,
    batch_size: int,
    sequence_length: int,
) -> dict[str, np.ndarray]:
    train_ids = np.asarray(arrays["train_episode_ids"], dtype=np.int32)
    transitions = int(arrays["observations"].shape[1])
    if transitions < sequence_length:
        raise ValueError("dataset episodes are shorter than the training sequence")
    random = np.random.default_rng(
        derive_seed("minibatch-schedule", task, world_model_seed)
    )
    episodes = random.choice(train_ids, size=(updates, batch_size), replace=True).astype(np.int32)
    starts = random.integers(
        0,
        transitions - sequence_length + 1,
        size=(updates, batch_size),
        dtype=np.int32,
    )
    return {"episode_ids": episodes, "starts": starts}


def _materialize_batch(
    arrays: Mapping[str, np.ndarray],
    schedule: Mapping[str, np.ndarray],
    update: int,
    *,
    sequence_length: int,
    burn_in: int,
) -> dict[str, Any]:
    import jax.numpy as jnp

    episodes = schedule["episode_ids"][update]
    starts = schedule["starts"][update]
    batch: dict[str, np.ndarray] = {}
    base_names = ("observations", "actions", "rewards", "continuations", "is_first")
    causal_names = (
        "causal_actions_lower",
        "causal_actions_upper",
        "causal_observations_lower",
        "causal_observations_upper",
        "causal_rewards_lower",
        "causal_rewards_upper",
        "causal_mask",
    )
    for name in base_names + tuple(name for name in causal_names if name in arrays):
        batch[name] = np.stack(
            [
                arrays[name][int(episode), int(start) : int(start) + sequence_length]
                for episode, start in zip(episodes, starts, strict=True)
            ]
        )
    loss_mask = np.ones((len(episodes), sequence_length), np.float32)
    loss_mask[:, :burn_in] = 0.0
    batch["loss_mask"] = loss_mask
    return {name: jnp.asarray(value) for name, value in batch.items()}


def _metrics_dict(value: Any) -> dict[str, float]:
    mapping = value if isinstance(value, Mapping) else value._asdict()
    result = {name: float(np.asarray(item)) for name, item in mapping.items()}
    if not np.isfinite(np.asarray(list(result.values()), dtype=np.float64)).all():
        raise FloatingPointError("training produced non-finite metrics")
    return result


def _tree_max_abs_difference(left: Any, right: Any) -> float:
    import jax

    left_values, left_tree = jax.tree_util.tree_flatten(left)
    right_values, right_tree = jax.tree_util.tree_flatten(right)
    if left_tree != right_tree:
        raise ValueError("parameter tree structures differ")
    return max(
        (
            float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            for a, b in zip(left_values, right_values, strict=True)
            if np.asarray(a).size
        ),
        default=0.0,
    )


def validate_world_result(
    result: Mapping[str, Any],
    cell: Mapping[str, Any],
    directory: str | Path,
    *,
    protocol: Mapping[str, Any] | None = None,
    matrix: Mapping[str, Any] | None = None,
    output_root: str | Path | None = None,
) -> None:
    _cell_identity(cell, "world_model")
    if any(result.get(key) != value for key, value in _result_identity(cell, WORLD_SCHEMA).items()):
        raise ValueError("world-model result identity mismatch")
    root = Path(directory)
    for field, filename in (
        ("checkpoint_sha256", "checkpoint.pkl"),
        ("batch_schedule_file_sha256", "batch_schedule.npz"),
        ("objective_rng_keys_file_sha256", "objective_rng_keys.npz"),
    ):
        path = root / filename
        if not path.is_file() or result.get(field) != file_sha256(path):
            raise ValueError(f"world-model {filename} digest mismatch")
    schedule = load_npz(root / "batch_schedule.npz")
    if result.get("batch_schedule_sha256") != array_sha256(schedule):
        raise ValueError("world-model minibatch schedule content digest mismatch")
    objective_keys = load_npz(root / "objective_rng_keys.npz")
    if set(objective_keys) != {
        "world_model_loss_keys",
        "base_noise_fold_in",
        "canonical_uniform_fold_in",
    }:
        raise ValueError("world-model objective key schedule is incomplete")
    key_rows = np.asarray(objective_keys["world_model_loss_keys"], dtype=np.uint32)
    if (
        key_rows.shape != (int(result.get("updates", -1)), 2)
        or np.unique(key_rows, axis=0).shape[0] != key_rows.shape[0]
        or not np.array_equal(objective_keys["base_noise_fold_in"], np.asarray([0], np.int32))
        or not np.array_equal(
            objective_keys["canonical_uniform_fold_in"], np.asarray([1], np.int32)
        )
    ):
        raise ValueError("world-model objective keys collide or violate the fold-in contract")
    if result.get("runtime_config_sha256") != object_sha256(result.get("runtime_config")):
        raise ValueError("world-model runtime config digest mismatch")
    if not math.isfinite(float(result.get("wall_seconds", math.nan))):
        raise ValueError("world-model wall time is non-finite")
    final_metrics = result.get("final_metrics")
    if not isinstance(final_metrics, Mapping) or not final_metrics:
        raise ValueError("world-model final metrics are missing")
    if not all(math.isfinite(float(value)) for value in final_metrics.values()):
        raise ValueError("world-model final metrics are non-finite")
    maximum_prior = float(result.get("maximum_checkpoint_prior_loss", math.nan))
    if not math.isfinite(maximum_prior) or maximum_prior < 0.0:
        raise ValueError("world-model checkpoint prior-loss maximum is invalid")
    if result.get("world_model_parameters_finite") is not True:
        raise ValueError("world-model parameters are non-finite")
    if cell.get("arm") == "shortcut_forcing" and (
        float(final_metrics.get("prior", math.inf)) > SHORTCUT_PRIOR_LOSS_CEILING
        or maximum_prior > SHORTCUT_PRIOR_LOSS_CEILING
    ):
        raise ValueError("shortcut world model exceeded the registered divergence ceiling")
    if protocol is not None or matrix is not None or output_root is not None:
        if protocol is None or matrix is None or output_root is None:
            raise ValueError("strong world validation requires protocol, matrix, and output root")
        runtime = result["runtime_config"]
        expected = make_config(
            protocol,
            cell["profile"],
            cell["arm"],
            runtime["observation_shape"],
            int(runtime["action_dim"]),
            output_root=output_root,
            candidate=_candidate_for_cell(matrix, cell),
        )
        if object_sha256(runtime) != object_sha256(asdict(expected)):
            raise ValueError("world-model runtime config differs from its frozen matrix arm")
        import jax
        from imf_dreamer_jax import create_agent, load_checkpoint

        initial = create_agent(
            expected,
            derive_jax_key("world-init", cell["task"], cell["world_model_seed"]),
        )
        shared_names = [
            "encoder",
            "decoder",
            "recurrence",
            "posterior",
            "reward",
            "continuation",
        ]
        shared = {name: initial.params.world_model[name] for name in shared_names}
        if (
            result.get("shared_initial_world_subtrees") != shared_names
            or result.get("shared_initial_world_parameter_sha256") != _tree_digest(shared)
        ):
            raise ValueError("world-model shared initialization evidence mismatch")

        state, stored_config, metadata = load_checkpoint(root / "checkpoint.pkl")
        if object_sha256(asdict(stored_config)) != object_sha256(asdict(expected)):
            raise ValueError("world-model checkpoint config mismatch")
        if (
            metadata.get("stage") != "world_model"
            or metadata.get("cell_id") != cell["cell_id"]
            or metadata.get("completed_updates") != result.get("updates")
            or metadata.get("dataset_sha256") != result.get("dataset_sha256")
        ):
            raise ValueError("world-model checkpoint metadata mismatch")
        optimizer_step = np.asarray(state.model_optimizer.step)
        if (
            optimizer_step.shape != ()
            or not np.issubdtype(optimizer_step.dtype, np.integer)
            or int(optimizer_step) != result.get("updates")
        ):
            raise ValueError("world-model checkpoint optimizer step mismatch")
        if _tree_digest(state.params.world_model) != result.get(
            "world_model_parameter_sha256"
        ):
            raise ValueError("world-model checkpoint parameter digest mismatch")
        teacher_digest = (
            None
            if state.world_model_teacher is None
            else _tree_digest(state.world_model_teacher)
        )
        if teacher_digest != result.get("world_model_teacher_parameter_sha256"):
            raise ValueError("world-model EMA teacher checkpoint digest mismatch")
        compute_cell = _dependency_cell(matrix, cell, "compute_plan")
        compute = read_json(_cell_result_path(output_root, compute_cell))
        allocation = compute["tracks"][cell["budget_track"]]["allocations"][
            cell["arm"]
        ]["world_model"]
        execution = _profile_execution(cell["profile"])
        expected_updates = int(allocation["updates"])
        if (
            result.get("updates") != expected_updates
            or result.get("batch_size") != int(execution["batch_size"])
            or result.get("sequence_length") != int(execution["sequence_length"])
            or result.get("smoke_runtime_override") is not (cell["profile"] == "smoke")
            or float(result.get("compiler_flops_per_update", math.nan))
            != float(allocation["flops_per_update"])
            or float(result.get("realized_compiler_flops", math.nan))
            != float(allocation["flops_per_update"]) * expected_updates
        ):
            raise ValueError("world-model execution differs from its frozen allocation")
        dataset_cell = _dependency_cell(matrix, cell, "dataset")
        dataset_directory = stage_directory(output_root, dataset_cell)
        dataset_result = read_json(_cell_result_path(output_root, dataset_cell))
        dataset_arrays = load_npz(dataset_directory / "dataset.npz")
        expected_schedule = _batch_schedule(
            dataset_arrays,
            task=cell["task"],
            world_model_seed=int(cell["world_model_seed"]),
            updates=expected_updates,
            batch_size=int(execution["batch_size"]),
            sequence_length=int(execution["sequence_length"]),
        )
        if (
            result.get("dataset_sha256") != dataset_result["dataset_sha256"]
            or set(schedule) != set(expected_schedule)
            or any(
                not np.array_equal(schedule[name], expected_schedule[name])
                for name in expected_schedule
            )
        ):
            raise ValueError("world-model dataset or minibatch schedule is not canonical")
        objective_base_key = derive_jax_key(
            "world-objective", cell["task"], cell["world_model_seed"]
        )
        expected_key_rows = _folded_key_rows(objective_base_key, expected_updates)
        if not np.array_equal(key_rows, expected_key_rows):
            raise ValueError("world-model objective keys differ from the frozen derivation")
        if result.get("parameter_counts") != compute["arms"][cell["arm"]]["parameters"]:
            raise ValueError("world-model parameter counts differ from compiler evidence")
        if runtime_homogeneity_identity(result.get("runtime", {})) != runtime_homogeneity_identity(
            compute.get("runtime", {})
        ):
            raise ValueError("world-model worker differs from its compiler/timing device")


def run_world_model_cell(
    cell: Mapping[str, Any],
    protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    output_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Train/resume one world model with a paired immutable minibatch prefix."""

    _cell_identity(cell, "world_model")
    _, dataset_result, dataset_directory = _load_dependency_result(
        output_root, matrix, cell, "dataset"
    )
    _, compute, _ = _load_dependency_result(output_root, matrix, cell, "compute_plan")
    validate_compute_plan_cell(
        compute,
        protocol,
        matrix,
        _dependency_cell(matrix, cell, "compute_plan"),
        output_root,
    )
    _validate_compute_files(
        compute,
        stage_directory(output_root, _dependency_cell(matrix, cell, "compute_plan")),
    )
    directory = stage_directory(output_root, cell)
    result_path = directory / "result.json"
    if not force and result_path.is_file():
        result = read_json(result_path)
        validate_world_result(
            result,
            cell,
            directory,
            protocol=protocol,
            matrix=matrix,
            output_root=output_root,
        )
        return result
    if force and result_path.is_file():
        raise ValueError("force does not overwrite a complete world-model cell; use a new output root")

    import jax
    from imf_dreamer_jax import create_agent, jit_train_world_model, load_checkpoint, save_checkpoint

    arrays = load_npz(dataset_directory / "dataset.npz")
    config = make_config(
        protocol,
        cell["profile"],
        cell["arm"],
        dataset_result["observation_shape"],
        int(dataset_result["action_dim"]),
        output_root=output_root,
        candidate=_candidate_for_cell(matrix, cell),
    )
    allocation = compute["tracks"][cell["budget_track"]]["allocations"][cell["arm"]]["world_model"]
    updates = int(allocation["updates"])
    execution = _profile_execution(cell["profile"])
    batch_size = int(execution["batch_size"])
    sequence_length = int(execution["sequence_length"])
    schedule = _batch_schedule(
        arrays,
        task=cell["task"],
        world_model_seed=int(cell["world_model_seed"]),
        updates=updates,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    directory.mkdir(parents=True, exist_ok=True)
    schedule_path = _write_npz_atomic(directory / "batch_schedule.npz", schedule)
    objective_base_key = derive_jax_key(
        "world-objective", cell["task"], cell["world_model_seed"]
    )
    training_keys = _folded_key_rows(objective_base_key, updates)
    key_path = _write_npz_atomic(
        directory / "objective_rng_keys.npz",
        {
            "world_model_loss_keys": training_keys,
            "base_noise_fold_in": np.asarray([0], np.int32),
            "canonical_uniform_fold_in": np.asarray([1], np.int32),
        },
    )
    checkpoint_path = directory / "checkpoint.pkl"
    start_update = 0
    accumulated_wall = 0.0
    latest = None
    maximum_checkpoint_prior_loss = 0.0
    initialization_key = derive_jax_key(
        "world-init", cell["task"], cell["world_model_seed"]
    )
    initial_agent = create_agent(config, initialization_key)
    shared_initial_world = {
        name: initial_agent.params.world_model[name]
        for name in (
            "encoder",
            "decoder",
            "recurrence",
            "posterior",
            "reward",
            "continuation",
        )
    }
    shared_initial_sha256 = _tree_digest(shared_initial_world)
    if checkpoint_path.is_file():
        state, stored_config, metadata = load_checkpoint(checkpoint_path)
        if stored_config != config or metadata.get("cell_id") != cell["cell_id"]:
            raise ValueError("partial world-model checkpoint identity mismatch")
        if metadata.get("dataset_sha256") != dataset_result["dataset_sha256"]:
            raise ValueError("partial world-model checkpoint dataset mismatch")
        start_update = int(metadata.get("completed_updates", -1))
        accumulated_wall = float(metadata.get("wall_seconds", 0.0))
        latest = metadata.get("last_metrics")
        maximum_checkpoint_prior_loss = float(
            metadata.get(
                "maximum_checkpoint_prior_loss",
                (latest or {}).get("prior", 0.0),
            )
        )
    else:
        state = initial_agent
    if not 0 <= start_update <= updates:
        raise ValueError("partial world-model update count is invalid")
    started = time.perf_counter()
    checkpoint_every = int(execution["checkpoint_every_updates"])
    for update in range(start_update, updates):
        batch = _materialize_batch(
            arrays,
            schedule,
            update,
            sequence_length=sequence_length,
            burn_in=config.burn_in,
        )
        key = jax.random.fold_in(objective_base_key, update)
        state, metrics = jit_train_world_model(state, batch, key, config)
        latest = _metrics_dict(metrics)
        completed = update + 1
        if completed % checkpoint_every == 0 or completed == updates:
            prior_loss = float(latest["prior"])
            maximum_checkpoint_prior_loss = max(
                maximum_checkpoint_prior_loss, prior_loss
            )
            prior_parameters_finite = all(
                np.isfinite(np.asarray(leaf)).all()
                for leaf in jax.tree_util.tree_leaves(state.params.world_model["prior"])
            )
            if not prior_parameters_finite:
                raise FloatingPointError(
                    "world-model prior parameters became non-finite"
                )
            if (
                config.prior == "shortcut"
                and maximum_checkpoint_prior_loss > SHORTCUT_PRIOR_LOSS_CEILING
            ):
                raise FloatingPointError(
                    "shortcut prior loss exceeded the registered divergence ceiling"
                )
            save_checkpoint(
                checkpoint_path,
                state,
                config,
                metadata={
                    "stage": "world_model",
                    "cell_id": cell["cell_id"],
                    "dataset_sha256": dataset_result["dataset_sha256"],
                    "completed_updates": completed,
                    "last_metrics": latest,
                    "maximum_checkpoint_prior_loss": maximum_checkpoint_prior_loss,
                    "wall_seconds": accumulated_wall + time.perf_counter() - started,
                },
            )
    if latest is None:
        raise RuntimeError("world-model cell completed no optimizer update")
    total_wall = accumulated_wall + time.perf_counter() - started
    counts = compute["arms"][cell["arm"]]["parameters"]
    world_model_parameters_finite = all(
        np.isfinite(np.asarray(leaf)).all()
        for leaf in jax.tree_util.tree_leaves(state.params.world_model)
    )
    if not world_model_parameters_finite:
        raise FloatingPointError("world-model parameters are non-finite")
    result = {
        **_result_identity(cell, WORLD_SCHEMA),
        "matrix_sha256": matrix["matrix_sha256"],
        "dataset_cell_id": _dependency_cell(matrix, cell, "dataset")["cell_id"],
        "compute_plan_cell_id": _dependency_cell(matrix, cell, "compute_plan")["cell_id"],
        "dataset_sha256": dataset_result["dataset_sha256"],
        "updates": updates,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "smoke_runtime_override": cell["profile"] == "smoke",
        "runtime_config": asdict(config),
        "runtime_config_sha256": object_sha256(asdict(config)),
        "parameter_counts": counts,
        "shared_initial_world_subtrees": [
            "encoder",
            "decoder",
            "recurrence",
            "posterior",
            "reward",
            "continuation",
        ],
        "shared_initial_world_parameter_sha256": shared_initial_sha256,
        "world_model_parameter_sha256": _tree_digest(state.params.world_model),
        "world_model_teacher_parameter_sha256": (
            None
            if state.world_model_teacher is None
            else _tree_digest(state.world_model_teacher)
        ),
        "world_model_parameters_finite": world_model_parameters_finite,
        "maximum_checkpoint_prior_loss": maximum_checkpoint_prior_loss,
        "final_metrics": latest,
        "compiler_flops_per_update": float(allocation["flops_per_update"]),
        "realized_compiler_flops": float(allocation["flops_per_update"]) * updates,
        "batch_schedule_sha256": array_sha256(schedule),
        "batch_schedule_file_sha256": file_sha256(schedule_path),
        "objective_rng_contract": "64_bit_namespaced_base_key_folded_with_update_then_fold_in_0_base_Gaussian_and_fold_in_1_canonical_uniforms_for_both_arms",
        "objective_rng_keys_file_sha256": file_sha256(key_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "wall_seconds": total_wall,
        "runtime": runtime_fingerprint(),
    }
    validate_world_result(
        result,
        cell,
        directory,
        protocol=protocol,
        matrix=matrix,
        output_root=output_root,
    )
    write_json_atomic(result_path, result)
    return result


def _rollout_windows(
    arrays: Mapping[str, np.ndarray],
    protocol: Mapping[str, Any],
    profile: str,
    *,
    task: str,
    world_model_seed: int,
    stochastic_dim: int,
) -> dict[str, np.ndarray]:
    context = int(protocol["evaluation"]["rollout_context_length"])
    maximum_horizon = max(int(value) for value in protocol["evaluation"]["rollout_horizons"])
    transitions = int(arrays["observations"].shape[1])
    if transitions < context + maximum_horizon:
        raise ValueError("held-out episodes are too short for registered rollout evaluation")
    test_ids = np.asarray(arrays["test_episode_ids"], dtype=np.int32)
    count = int(protocol["profiles"][profile]["rollout_windows_per_task_world_seed"])
    draws = int(protocol["profiles"][profile]["predictive_draws_per_window"])
    random = np.random.default_rng(derive_seed("rollout-windows", task, world_model_seed))
    episode_ids = random.choice(test_ids, size=count, replace=True).astype(np.int32)
    anchors = random.integers(
        context - 1,
        transitions - maximum_horizon,
        size=count,
        dtype=np.int32,
    )
    def gather(name: str, start_offset: int, stop_offset: int) -> np.ndarray:
        return np.stack(
            [
                arrays[name][int(episode), int(anchor) + start_offset : int(anchor) + stop_offset]
                for episode, anchor in zip(episode_ids, anchors, strict=True)
            ]
        )

    noise = np.random.default_rng(
        derive_seed("rollout-noise", task, world_model_seed)
    ).normal(size=(draws, maximum_horizon, count, stochastic_dim)).astype(np.float32)
    train_observations = arrays["observations"][
        np.asarray(arrays["train_episode_ids"], dtype=np.int32)
    ]
    flattened = train_observations.reshape((-1, *train_observations.shape[2:]))
    standard_deviation = np.std(flattened, axis=0, ddof=0).astype(np.float32)
    standard_deviation = np.maximum(standard_deviation, 1e-6)
    return {
        "episode_ids": episode_ids,
        "anchors": anchors,
        "context_observations": gather("observations", -context + 1, 1),
        "context_actions": gather("actions", -context + 1, 1),
        "future_actions": gather("actions", 1, maximum_horizon + 1),
        "target_observations": gather("observations", 1, maximum_horizon + 1),
        "target_rewards": gather("rewards", 1, maximum_horizon + 1),
        "target_continuations": gather("continuations", 1, maximum_horizon + 1),
        "noise": noise,
        "training_observation_std": standard_deviation,
    }


def normalized_rollout_statistics(
    observation_samples: np.ndarray,
    reward_samples: np.ndarray,
    windows: Mapping[str, np.ndarray],
    horizons: Sequence[int],
    continuation_samples: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute the registered expected standardized square error and AUC."""

    samples = np.asarray(observation_samples, dtype=np.float64)
    rewards = np.asarray(reward_samples, dtype=np.float64)
    targets = np.asarray(windows["target_observations"], dtype=np.float64)
    reward_targets = np.asarray(windows["target_rewards"], dtype=np.float64)
    continuations = (
        None
        if continuation_samples is None
        else np.asarray(continuation_samples, dtype=np.float64)
    )
    continuation_targets = (
        None
        if continuations is None
        else np.asarray(windows["target_continuations"], dtype=np.float64)
    )
    std = np.asarray(windows["training_observation_std"], dtype=np.float64)
    if samples.ndim < 4 or samples.shape[1:3] != targets.shape[:2]:
        raise ValueError("rollout sample shape does not match held-out targets")
    if rewards.shape[1:3] != reward_targets.shape[:2]:
        raise ValueError("reward sample shape does not match held-out targets")
    if continuations is not None and continuations.shape[1:3] != continuation_targets.shape[:2]:
        raise ValueError("continuation sample shape does not match held-out targets")
    per_horizon: dict[str, Any] = {}
    errors: list[float] = []
    coordinates = [int(value) for value in horizons]
    for horizon in coordinates:
        index = horizon - 1
        standardized = (samples[:, :, index] - targets[None, :, index]) / std
        expected_squared_error = float(np.mean(np.square(standardized)))
        predictive_mean_rmse = float(
            np.sqrt(np.mean(np.square(np.mean(samples[:, :, index], axis=0) - targets[:, index])))
        )
        reward_expected_mse = float(
            np.mean(np.square(rewards[:, :, index] - reward_targets[None, :, index]))
        )
        flattened_samples = samples[:, :, index].reshape((samples.shape[0], samples.shape[1], -1))
        flattened_targets = targets[:, index].reshape((targets.shape[0], -1))
        first = np.mean(np.linalg.norm(flattened_samples - flattened_targets[None], axis=-1))
        pairwise = flattened_samples[:, None] - flattened_samples[None, :]
        second = 0.5 * np.mean(np.linalg.norm(pairwise, axis=-1))
        lower = np.quantile(samples[:, :, index], 0.05, axis=0)
        upper = np.quantile(samples[:, :, index], 0.95, axis=0)
        coverage = float(np.mean((targets[:, index] >= lower) & (targets[:, index] <= upper)))
        width = float(np.mean(upper - lower))
        row = {
            "expected_standardized_squared_error": expected_squared_error,
            "predictive_mean_observation_rmse": predictive_mean_rmse,
            "reward_expected_squared_error": reward_expected_mse,
            "observation_energy_score": float(first - second),
            "central_90_percent_interval_coverage": coverage,
            "central_90_percent_interval_width": width,
            "central_90_percent_interval_calibration_absolute_error": abs(
                coverage - 0.9
            ),
        }
        if continuations is not None:
            continuation_draws = continuations[:, :, index]
            continuation_target = continuation_targets[:, index]
            row.update(
                {
                    "continuation_expected_squared_error": float(
                        np.mean(
                            np.square(
                                continuation_draws - continuation_target[None]
                            )
                        )
                    ),
                    "continuation_predictive_mean_brier": float(
                        np.mean(
                            np.square(
                                np.mean(continuation_draws, axis=0)
                                - continuation_target
                            )
                        )
                    ),
                }
            )
        per_horizon[str(horizon)] = row
        errors.append(expected_squared_error)
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:  # NumPy 1.26 compatibility.
        trapezoid = getattr(np, "trapz")
    auc = float(
        trapezoid(np.asarray(errors), np.asarray(coordinates))
        / (coordinates[-1] - coordinates[0])
    )
    if not math.isfinite(auc) or not all(
        math.isfinite(float(value))
        for row in per_horizon.values()
        for value in row.values()
    ):
        raise FloatingPointError("rollout evaluation produced non-finite metrics")
    return {
        "per_horizon": per_horizon,
        "normalized_free_running_rollout_error_auc": auc,
        "auc_coordinates": coordinates,
        "auc_rule": "normalized_trapezoid_over_registered_horizons",
    }


def _rollout_replay_comparison_contract() -> dict[str, Any]:
    return {
        "rule": "exact_else_elementwise_allclose",
        "absolute_tolerance": ROLLOUT_REPLAY_ABSOLUTE_TOLERANCE,
        "relative_tolerance": ROLLOUT_REPLAY_RELATIVE_TOLERANCE,
        "equal_nan": False,
    }


def _replay_rollout_inference(
    world_directory: str | Path,
    world_cell: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    config: Any,
) -> dict[str, np.ndarray]:
    """Rerun one rollout from its checkpoint and retained canonical inputs."""

    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import load_checkpoint
    checkpoint_path = Path(world_directory) / "checkpoint.pkl"
    state, _, metadata = load_checkpoint(checkpoint_path)
    if metadata.get("cell_id") != world_cell["cell_id"]:
        raise ValueError("rollout replay checkpoint identity mismatch")
    start = posterior_mean_filter(
        state.params.world_model,
        jnp.asarray(arrays["context_observations"]),
        jnp.asarray(arrays["context_actions"]),
        config,
    )
    sampler = jax.jit(open_loop_samples_with_continuation, static_argnames=("config",))
    observation_samples, reward_samples, continuation_samples = sampler(
        state.params.world_model,
        start,
        jnp.asarray(arrays["future_actions"]),
        jnp.asarray(arrays["noise"]),
        config,
    )
    return {
        "observation_samples": np.asarray(observation_samples),
        "reward_samples": np.asarray(reward_samples),
        "continuation_samples": np.asarray(continuation_samples),
    }


def _validate_rollout_replay_samples(
    observed: Mapping[str, np.ndarray],
    replayed: Mapping[str, np.ndarray],
) -> None:
    """Require exact replay, allowing only explicit roundoff-sized drift."""

    names = (
        "observation_samples",
        "reward_samples",
        "continuation_samples",
    )
    if set(replayed) != set(names):
        raise ValueError("rollout checkpoint replay fields are incomplete")
    for name in names:
        retained = np.asarray(observed[name])
        regenerated = np.asarray(replayed[name])
        if retained.shape != regenerated.shape:
            raise ValueError(f"rollout {name} shape differs from checkpoint replay")
        if retained.dtype != regenerated.dtype:
            raise ValueError(f"rollout {name} dtype differs from checkpoint replay")
        if not np.isfinite(regenerated).all():
            raise ValueError(f"rollout {name} checkpoint replay is non-finite")
        if np.array_equal(retained, regenerated):
            continue
        if not np.allclose(
            retained,
            regenerated,
            rtol=ROLLOUT_REPLAY_RELATIVE_TOLERANCE,
            atol=ROLLOUT_REPLAY_ABSOLUTE_TOLERANCE,
            equal_nan=False,
        ):
            maximum = float(np.max(np.abs(retained.astype(np.float64) - regenerated)))
            raise ValueError(
                f"rollout {name} does not match checkpoint replay "
                f"(maximum absolute difference {maximum:.9g})"
            )


def validate_rollout_result(
    result: Mapping[str, Any],
    cell: Mapping[str, Any],
    raw_path: str | Path,
    protocol: Mapping[str, Any] | None = None,
    matrix: Mapping[str, Any] | None = None,
    output_root: str | Path | None = None,
    *,
    validated_results: _ValidatedResultCache | None = None,
) -> None:
    _cell_identity(cell, "rollout")
    if any(result.get(key) != value for key, value in _result_identity(cell, ROLLOUT_SCHEMA).items()):
        raise ValueError("rollout result identity mismatch")
    path = Path(raw_path)
    if not path.is_file() or result.get("raw_predictive_draws_sha256") != file_sha256(path):
        raise ValueError("rollout predictive-draw artifact digest mismatch")
    arrays = load_npz(path)
    required = {
        "observation_samples",
        "reward_samples",
        "continuation_samples",
        "target_observations",
        "target_rewards",
        "target_continuations",
        "context_observations",
        "context_actions",
        "future_actions",
        "noise",
        "episode_ids",
        "anchors",
        "training_observation_std",
    }
    if set(arrays) != required:
        raise ValueError("rollout predictive-draw artifact is incomplete")
    _validate_finite_arrays(arrays, required)
    value = float(result.get("normalized_free_running_rollout_error_auc", math.nan))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("rollout AUC is invalid")
    if result.get("structural_nfe_per_transition") != cell["nfe"]:
        raise ValueError("rollout structural NFE mismatch")
    if result.get("windows") != arrays["target_observations"].shape[0]:
        raise ValueError("rollout window count differs from the raw artifact")
    if result.get("predictive_draws_per_window") != arrays["observation_samples"].shape[0]:
        raise ValueError("rollout draw count differs from the raw artifact")
    horizons = (
        protocol["evaluation"]["rollout_horizons"]
        if protocol is not None
        else result.get("auc_coordinates", [])
    )
    recomputed = normalized_rollout_statistics(
        arrays["observation_samples"],
        arrays["reward_samples"],
        {
            "target_observations": arrays["target_observations"],
            "target_rewards": arrays["target_rewards"],
            "target_continuations": arrays["target_continuations"],
            "training_observation_std": arrays["training_observation_std"],
        },
        horizons,
        arrays["continuation_samples"],
    )
    saved = {
        key: result.get(key)
        for key in (
            "per_horizon",
            "normalized_free_running_rollout_error_auc",
            "auc_coordinates",
            "auc_rule",
        )
    }
    if canonical_bytes(saved) != canonical_bytes(recomputed):
        raise ValueError("saved rollout metrics do not recompute from retained draws")
    if matrix is not None or output_root is not None:
        if protocol is None or matrix is None or output_root is None:
            raise ValueError("strong rollout validation requires protocol, matrix, and output root")
        root = Path(output_root)
        canonical_raw_path = stage_directory(root, cell) / "predictive_draws.npz"
        if path.resolve() != canonical_raw_path.resolve():
            raise ValueError("rollout predictive-draw path is not canonical")
        required_result_fields = set(_result_identity(cell, ROLLOUT_SCHEMA)) | {
            "matrix_sha256",
            "world_model_cell_id",
            "dataset_cell_id",
            "world_model_checkpoint_sha256",
            "world_model_result_file_sha256",
            "world_model_runtime_config_sha256",
            "dataset_sha256",
            "dataset_result_file_sha256",
            "runtime_config",
            "runtime_config_sha256",
            "structural_nfe_per_transition",
            "windows",
            "predictive_draws_per_window",
            "per_horizon",
            "normalized_free_running_rollout_error_auc",
            "auc_coordinates",
            "auc_rule",
            "raw_predictive_draws_sha256",
            "inference_replay_comparison",
            "runtime",
            "wall_seconds",
        }
        if set(result) != required_result_fields:
            raise ValueError("rollout result schema is incomplete or contains extras")
        wall_seconds = float(result.get("wall_seconds", math.nan))
        if not math.isfinite(wall_seconds) or wall_seconds < 0.0:
            raise ValueError("rollout wall time is invalid")
        dataset_cell = _dependency_cell(matrix, cell, "dataset")
        dataset_directory = stage_directory(root, dataset_cell)
        dataset_path = dataset_directory / "dataset.npz"
        dataset_result_path = dataset_directory / "result.json"
        if not dataset_result_path.is_file():
            raise FileNotFoundError("rollout canonical dataset result is missing")
        dataset_result = read_json(dataset_result_path)
        if not _dependency_was_prevalidated(
            validated_results,
            root,
            matrix,
            protocol,
            dataset_cell,
            dataset_result,
            dataset_result_path,
        ):
            validate_dataset_result(
                dataset_result,
                dataset_cell,
                dataset_path,
                protocol=protocol,
                matrix=matrix,
            )
        dataset = load_npz(dataset_path)
        world_cell = _dependency_cell(matrix, cell, "world_model")
        world_directory = stage_directory(root, world_cell)
        world_result_path = world_directory / "result.json"
        if not world_result_path.is_file():
            raise FileNotFoundError("rollout canonical world-model result is missing")
        world_result = read_json(world_result_path)
        if not _dependency_was_prevalidated(
            validated_results,
            root,
            matrix,
            protocol,
            world_cell,
            world_result,
            world_result_path,
        ):
            validate_world_result(
                world_result,
                world_cell,
                world_directory,
                protocol=protocol,
                matrix=matrix,
                output_root=root,
            )
        compute_cell = _dependency_cell(matrix, cell, "compute_plan")
        compute_directory = stage_directory(root, compute_cell)
        compute_result_path = compute_directory / "result.json"
        if not compute_result_path.is_file():
            raise FileNotFoundError("rollout canonical compute-plan result is missing")
        compute_result = read_json(compute_result_path)
        if not _dependency_was_prevalidated(
            validated_results,
            root,
            matrix,
            protocol,
            compute_cell,
            compute_result,
            compute_result_path,
        ):
            validate_compute_plan_cell(
                compute_result, protocol, matrix, compute_cell, root
            )
            _validate_compute_files(compute_result, compute_directory)
        if (
            _dependency_cell(matrix, world_cell, "dataset")["cell_id"]
            != dataset_cell["cell_id"]
            or _dependency_cell(matrix, world_cell, "compute_plan")["cell_id"]
            != compute_cell["cell_id"]
        ):
            raise ValueError("rollout and world-model canonical dependencies differ")
        checkpoint_path = world_directory / "checkpoint.pkl"
        if (
            result.get("matrix_sha256") != matrix["matrix_sha256"]
            or dataset_result.get("matrix_sha256") != matrix["matrix_sha256"]
            or world_result.get("matrix_sha256") != matrix["matrix_sha256"]
            or result.get("dataset_cell_id") != dataset_cell["cell_id"]
            or result.get("world_model_cell_id") != world_cell["cell_id"]
            or result.get("dataset_sha256") != dataset_result.get("dataset_sha256")
            or result.get("world_model_checkpoint_sha256")
            != world_result.get("checkpoint_sha256")
            or result.get("world_model_checkpoint_sha256")
            != file_sha256(checkpoint_path)
            or result.get("dataset_result_file_sha256")
            != file_sha256(dataset_result_path)
            or result.get("world_model_result_file_sha256")
            != file_sha256(world_result_path)
            or result.get("world_model_runtime_config_sha256")
            != world_result.get("runtime_config_sha256")
        ):
            raise ValueError("rollout dependency identities or digests are invalid")
        config = make_config(
            protocol,
            cell["profile"],
            cell["arm"],
            dataset_result["observation_shape"],
            int(dataset_result["action_dim"]),
            nfe=int(cell["nfe"]),
            output_root=root,
            candidate=_candidate_for_cell(matrix, cell),
        )
        config_payload = asdict(config)
        if (
            result.get("runtime_config_sha256") != object_sha256(config_payload)
            or canonical_bytes(result.get("runtime_config"))
            != canonical_bytes(config_payload)
        ):
            raise ValueError("rollout runtime config differs from its canonical NFE cell")
        if result.get("inference_replay_comparison") != _rollout_replay_comparison_contract():
            raise ValueError("rollout replay comparison contract differs")
        if runtime_homogeneity_identity(result.get("runtime", {})) != runtime_homogeneity_identity(
            compute_result.get("runtime", {})
        ):
            raise ValueError("rollout worker differs from its compiler/timing device")
        expected_windows = _rollout_windows(
            dataset,
            protocol,
            cell["profile"],
            task=cell["task"],
            world_model_seed=int(cell["world_model_seed"]),
            stochastic_dim=int(config.stochastic_dim),
        )
        for name, expected_value in expected_windows.items():
            if name not in arrays or not np.array_equal(arrays[name], expected_value):
                raise ValueError(f"rollout raw field {name!r} differs from dataset/seed derivation")
        replayed = _replay_rollout_inference(
            world_directory,
            world_cell,
            arrays,
            config,
        )
        _validate_rollout_replay_samples(arrays, replayed)


def run_rollout_cell(
    cell: Mapping[str, Any],
    protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    output_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate held-out free-running samples at one registered NFE point."""

    _cell_identity(cell, "rollout")
    _, world_result, world_directory = _load_dependency_result(
        output_root, matrix, cell, "world_model"
    )
    _, dataset_result, dataset_directory = _load_dependency_result(
        output_root, matrix, cell, "dataset"
    )
    directory = stage_directory(output_root, cell)
    raw_path = directory / "predictive_draws.npz"
    result_path = directory / "result.json"
    if not force and raw_path.is_file() and result_path.is_file():
        result = read_json(result_path)
        validate_rollout_result(
            result,
            cell,
            raw_path,
            protocol,
            matrix,
            output_root,
        )
        return result

    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import load_checkpoint
    state, _, metadata = load_checkpoint(world_directory / "checkpoint.pkl")
    if metadata.get("cell_id") != _dependency_cell(matrix, cell, "world_model")["cell_id"]:
        raise ValueError("rollout world-model checkpoint identity mismatch")
    config = make_config(
        protocol,
        cell["profile"],
        cell["arm"],
        dataset_result["observation_shape"],
        int(dataset_result["action_dim"]),
        nfe=int(cell["nfe"]),
        output_root=output_root,
        candidate=_candidate_for_cell(matrix, cell),
    )
    arrays = load_npz(dataset_directory / "dataset.npz")
    windows = _rollout_windows(
        arrays,
        protocol,
        cell["profile"],
        task=cell["task"],
        world_model_seed=int(cell["world_model_seed"]),
        stochastic_dim=config.stochastic_dim,
    )
    started = time.perf_counter()
    start = posterior_mean_filter(
        state.params.world_model,
        jnp.asarray(windows["context_observations"]),
        jnp.asarray(windows["context_actions"]),
        config,
    )
    sampler = jax.jit(open_loop_samples_with_continuation, static_argnames=("config",))
    observation_samples, reward_samples, continuation_samples = sampler(
        state.params.world_model,
        start,
        jnp.asarray(windows["future_actions"]),
        jnp.asarray(windows["noise"]),
        config,
    )
    observation_samples = np.asarray(observation_samples)
    reward_samples = np.asarray(reward_samples)
    continuation_samples = np.asarray(continuation_samples)
    metrics = normalized_rollout_statistics(
        observation_samples,
        reward_samples,
        windows,
        protocol["evaluation"]["rollout_horizons"],
        continuation_samples,
    )
    raw = {
        "observation_samples": observation_samples,
        "reward_samples": reward_samples,
        "continuation_samples": continuation_samples,
        "target_observations": windows["target_observations"],
        "target_rewards": windows["target_rewards"],
        "target_continuations": windows["target_continuations"],
        "context_observations": windows["context_observations"],
        "context_actions": windows["context_actions"],
        "future_actions": windows["future_actions"],
        "noise": windows["noise"],
        "episode_ids": windows["episode_ids"],
        "anchors": windows["anchors"],
        "training_observation_std": windows["training_observation_std"],
    }
    _write_npz_atomic(raw_path, raw)
    runtime_config = asdict(config)
    result = {
        **_result_identity(cell, ROLLOUT_SCHEMA),
        "matrix_sha256": matrix["matrix_sha256"],
        "world_model_cell_id": _dependency_cell(matrix, cell, "world_model")["cell_id"],
        "dataset_cell_id": _dependency_cell(matrix, cell, "dataset")["cell_id"],
        "world_model_checkpoint_sha256": world_result["checkpoint_sha256"],
        "world_model_result_file_sha256": file_sha256(
            world_directory / "result.json"
        ),
        "world_model_runtime_config_sha256": world_result["runtime_config_sha256"],
        "dataset_sha256": dataset_result["dataset_sha256"],
        "dataset_result_file_sha256": file_sha256(
            dataset_directory / "result.json"
        ),
        "runtime_config": runtime_config,
        "runtime_config_sha256": object_sha256(runtime_config),
        "structural_nfe_per_transition": int(cell["nfe"]),
        "windows": int(len(windows["episode_ids"])),
        "predictive_draws_per_window": int(windows["noise"].shape[0]),
        **metrics,
        "raw_predictive_draws_sha256": file_sha256(raw_path),
        "inference_replay_comparison": _rollout_replay_comparison_contract(),
        "runtime": runtime_fingerprint(),
        "wall_seconds": time.perf_counter() - started,
    }
    validate_rollout_result(
        result,
        cell,
        raw_path,
        protocol,
        matrix,
        output_root,
    )
    write_json_atomic(result_path, result)
    return result


def _evaluate_actor_policy(
    state: Any,
    config: Any,
    *,
    task: str,
    world_model_seed: int,
    actor_seed: int,
    episodes: int,
    maximum_steps: int,
) -> tuple[list[float], dict[str, np.ndarray]]:
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import initial_state, jit_act

    returns: list[float] = []
    traces: list[np.ndarray] = []
    reward_traces: list[np.ndarray] = []
    continuation_traces: list[np.ndarray] = []
    terminal_traces: list[np.ndarray] = []
    evaluation_seeds: list[int] = []
    for episode in range(episodes):
        evaluation_seed = derive_seed(
            "actor-evaluation", task, world_model_seed, actor_seed, episode
        )
        evaluation_seeds.append(evaluation_seed)
        environment = DMCAdapter(
            task,
            seed=evaluation_seed,
            action_repeat=1,
        )
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        continuations: list[float] = []
        terminals: list[bool] = []
        try:
            observation = environment.reset()
            belief = initial_state(config, 1)
            previous_action = jnp.zeros((1, config.action_dim), jnp.float32)
            evaluation_key = derive_jax_key(
                "actor-evaluation-action",
                task,
                world_model_seed,
                actor_seed,
                episode,
            )
            for step in range(maximum_steps):
                action, belief = jit_act(
                    state.params,
                    jnp.asarray(observation[None]),
                    previous_action,
                    belief,
                    jax.random.fold_in(evaluation_key, step),
                    config,
                    deterministic=True,
                )
                host_action = np.asarray(action[0], dtype=np.float32)
                transition = environment.step(host_action)
                actions.append(host_action)
                rewards.append(float(transition.reward))
                continuations.append(float(transition.continuation))
                terminals.append(bool(transition.is_last))
                observation = transition.observation
                previous_action = action
                if transition.is_last:
                    break
            returns.append(float(np.sum(np.asarray(rewards, dtype=np.float64))))
            traces.append(np.stack(actions) if actions else np.empty((0, config.action_dim), np.float32))
            reward_traces.append(np.asarray(rewards, dtype=np.float64))
            continuation_traces.append(np.asarray(continuations, dtype=np.float64))
            terminal_traces.append(np.asarray(terminals, dtype=np.bool_))
        finally:
            environment.close()
    lengths = np.asarray([len(trace) for trace in traces], dtype=np.int32)
    padded = np.zeros((episodes, int(lengths.max(initial=0)), config.action_dim), np.float32)
    padded_rewards = np.zeros((episodes, padded.shape[1]), dtype=np.float64)
    padded_continuations = np.zeros((episodes, padded.shape[1]), dtype=np.float64)
    padded_terminals = np.zeros((episodes, padded.shape[1]), dtype=np.bool_)
    for index, trace in enumerate(traces):
        padded[index, : len(trace)] = trace
        padded_rewards[index, : len(trace)] = reward_traces[index]
        padded_continuations[index, : len(trace)] = continuation_traces[index]
        padded_terminals[index, : len(trace)] = terminal_traces[index]
    return returns, {
        "actions": padded,
        "rewards": padded_rewards,
        "continuations": padded_continuations,
        "is_last": padded_terminals,
        "lengths": lengths,
        "evaluation_seeds": np.asarray(evaluation_seeds, dtype=np.uint32),
    }


def _validate_actor_environment_replay(
    traces: Mapping[str, np.ndarray],
    episode_returns: np.ndarray,
    *,
    state: Any,
    config: Any,
    task: str,
    world_model_seed: int,
    actor_seed: int,
    maximum_steps: int,
    action_repeat: int,
) -> None:
    """Regenerate policy actions and authenticate the full actor trajectory."""

    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import initial_state, jit_act

    actions = np.asarray(traces["actions"])
    rewards = np.asarray(traces["rewards"], dtype=np.float64)
    continuations = np.asarray(traces["continuations"], dtype=np.float64)
    terminals = np.asarray(traces["is_last"], dtype=np.bool_)
    lengths = np.asarray(traces["lengths"], dtype=np.int64)
    evaluation_seeds = np.asarray(traces["evaluation_seeds"], dtype=np.uint32)
    if maximum_steps <= 0 or action_repeat <= 0 or np.any(lengths > maximum_steps):
        raise ValueError("actor replay length or action repeat differs from the protocol")
    for episode, (length, evaluation_seed) in enumerate(
        zip(lengths.tolist(), evaluation_seeds.tolist(), strict=True)
    ):
        environment = DMCAdapter(
            task,
            seed=int(evaluation_seed),
            action_repeat=int(action_repeat),
        )
        replay_rewards: list[float] = []
        replay_continuations: list[float] = []
        replay_terminals: list[bool] = []
        try:
            observation = environment.reset()
            if environment.action_dim != actions.shape[-1]:
                raise ValueError("actor replay action dimension differs from DMC")
            belief = initial_state(config, 1)
            previous_action = jnp.zeros((1, config.action_dim), jnp.float32)
            evaluation_key = derive_jax_key(
                "actor-evaluation-action",
                task,
                world_model_seed,
                actor_seed,
                episode,
            )
            for step in range(int(length)):
                policy_action, belief = jit_act(
                    state.params,
                    jnp.asarray(observation[None]),
                    previous_action,
                    belief,
                    jax.random.fold_in(evaluation_key, step),
                    config,
                    deterministic=True,
                )
                regenerated_action = np.asarray(policy_action[0], dtype=np.float32)
                retained_action = np.asarray(actions[episode, step], dtype=np.float32)
                if not np.array_equal(regenerated_action, retained_action) and not np.allclose(
                    regenerated_action,
                    retained_action,
                    rtol=1e-6,
                    atol=1e-6,
                    equal_nan=False,
                ):
                    raise ValueError("actor actions differ from checkpoint policy replay")
                transition = environment.step(retained_action)
                replay_rewards.append(float(transition.reward))
                replay_continuations.append(float(transition.continuation))
                replay_terminals.append(bool(transition.is_last))
                observation = transition.observation
                previous_action = policy_action
                if transition.is_last and step + 1 != int(length):
                    raise ValueError("actor trace continued after replayed DMC termination")
        finally:
            environment.close()

        if not replay_terminals[-1] and int(length) != int(maximum_steps):
            raise ValueError("actor trace ended before DMC termination or the step limit")
        if not np.array_equal(
            rewards[episode, :length], np.asarray(replay_rewards, dtype=np.float64)
        ):
            raise ValueError("actor rewards differ from replayed DMC transitions")
        if not np.array_equal(
            continuations[episode, :length],
            np.asarray(replay_continuations, dtype=np.float64),
        ):
            raise ValueError("actor continuations differ from replayed DMC transitions")
        if not np.array_equal(
            terminals[episode, :length], np.asarray(replay_terminals, dtype=np.bool_)
        ):
            raise ValueError("actor terminal flags differ from replayed DMC transitions")
        replay_return = float(np.sum(np.asarray(replay_rewards, dtype=np.float64)))
        if replay_return != float(episode_returns[episode]):
            raise ValueError("actor return differs from replayed DMC rewards")


def validate_actor_result(
    result: Mapping[str, Any],
    cell: Mapping[str, Any],
    directory: str | Path,
    *,
    protocol: Mapping[str, Any] | None = None,
    matrix: Mapping[str, Any] | None = None,
    output_root: str | Path | None = None,
) -> None:
    _cell_identity(cell, "actor")
    if any(result.get(key) != value for key, value in _result_identity(cell, ACTOR_SCHEMA).items()):
        raise ValueError("actor result identity mismatch")
    root = Path(directory)
    for field, filename in (
        ("checkpoint_sha256", "checkpoint.pkl"),
        ("raw_action_traces_sha256", "action_traces.npz"),
        ("batch_schedule_file_sha256", "batch_schedule.npz"),
    ):
        path = root / filename
        if not path.is_file() or result.get(field) != file_sha256(path):
            raise ValueError(f"actor {filename} digest mismatch")
    if result.get("world_model_frozen") is not True or result.get("world_model_parameter_delta") != 0.0:
        raise ValueError("actor cell did not preserve the frozen world model")
    returns = np.asarray(result.get("episode_returns", []), dtype=np.float64)
    if returns.ndim != 1 or returns.size == 0 or not np.isfinite(returns).all():
        raise ValueError("actor episode returns are invalid")
    normalized = returns / 1000.0
    if not np.array_equal(
        normalized,
        np.asarray(result.get("normalized_episode_returns", []), dtype=np.float64),
    ):
        raise ValueError("actor normalized episode returns are inconsistent")
    if not math.isclose(float(result.get("normalized_episode_return_mean")), float(np.mean(normalized)), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("actor normalized return is inconsistent")
    schedule = load_npz(root / "batch_schedule.npz")
    if result.get("batch_schedule_sha256") != array_sha256(schedule):
        raise ValueError("actor minibatch schedule content digest mismatch")
    traces = load_npz(root / "action_traces.npz")
    if set(traces) != {
        "actions",
        "rewards",
        "continuations",
        "is_last",
        "lengths",
        "evaluation_seeds",
    }:
        raise ValueError("actor action-trace artifact is incomplete")
    lengths = np.asarray(traces["lengths"], dtype=np.int64)
    steps = np.arange(traces["actions"].shape[1])[None, :]
    valid = steps < lengths[:, None]
    if (
        traces["actions"].shape[0] != returns.size
        or traces["rewards"].shape != traces["actions"].shape[:2]
        or traces["continuations"].shape != traces["actions"].shape[:2]
        or traces["is_last"].shape != traces["actions"].shape[:2]
        or traces["evaluation_seeds"].shape != (returns.size,)
        or traces["is_last"].dtype != np.bool_
        or lengths.shape != (returns.size,)
        or np.any(lengths <= 0)
        or np.any(lengths > traces["actions"].shape[1])
        or not np.isfinite(traces["actions"]).all()
        or not np.isfinite(traces["rewards"]).all()
        or not np.isfinite(traces["continuations"]).all()
        or np.any(traces["continuations"][valid] < 0.0)
        or np.any(traces["continuations"][valid] > 1.0)
        or np.any(traces["rewards"][~valid] != 0.0)
        or np.any(traces["continuations"][~valid] != 0.0)
        or np.any(traces["is_last"][~valid])
    ):
        raise ValueError("actor action-trace shapes or values are invalid")
    recomputed_returns = np.sum(
        np.where(valid, traces["rewards"], 0.0), axis=1, dtype=np.float64
    )
    if not np.array_equal(returns, recomputed_returns):
        raise ValueError("actor episode returns do not recompute from retained rewards")
    for episode, length in enumerate(lengths):
        terminal = np.asarray(traces["is_last"][episode, :length], dtype=np.bool_)
        if np.any(terminal[:-1]):
            raise ValueError("actor trace continued after a terminal transition")
    if protocol is not None or matrix is not None or output_root is not None:
        if protocol is None or matrix is None or output_root is None:
            raise ValueError("strong actor validation requires protocol, matrix, and output root")
        expected_episodes = int(
            protocol["profiles"][cell["profile"]][
                "real_environment_evaluation_episodes"
            ]
        )
        if returns.size != expected_episodes:
            raise ValueError("actor evaluation episode count differs from the frozen profile")

        world_cell = _dependency_cell(matrix, cell, "world_model")
        expected_evaluation_seeds = np.asarray(
            [
                derive_seed(
                    "actor-evaluation",
                    cell["task"],
                    cell["world_model_seed"],
                    cell["actor_seed"],
                    episode,
                )
                for episode in range(returns.size)
            ],
            dtype=np.uint32,
        )
        if not np.array_equal(traces["evaluation_seeds"], expected_evaluation_seeds):
            raise ValueError("actor evaluation reset seeds differ from the frozen derivation")

        import jax
        from imf_dreamer_jax import create_agent, load_checkpoint

        world_result = read_json(_cell_result_path(output_root, world_cell))
        compute_cell = _dependency_cell(matrix, cell, "compute_plan")
        compute_result = read_json(_cell_result_path(output_root, compute_cell))
        allocation = compute_result["tracks"][cell["budget_track"]]["allocations"][
            cell["arm"]
        ]["actor"]
        if runtime_homogeneity_identity(result.get("runtime", {})) != runtime_homogeneity_identity(
            compute_result.get("runtime", {})
        ):
            raise ValueError("actor worker differs from its compiler/timing device")
        world_state, _, _ = load_checkpoint(
            stage_directory(output_root, world_cell) / "checkpoint.pkl"
        )
        actor_state, actor_config, metadata = load_checkpoint(root / "checkpoint.pkl")
        expected = make_config(
            protocol,
            cell["profile"],
            cell["arm"],
            world_result["runtime_config"]["observation_shape"],
            int(world_result["runtime_config"]["action_dim"]),
            output_root=output_root,
            candidate=_candidate_for_cell(matrix, cell),
        )
        execution = _profile_execution(cell["profile"])
        expected_updates = int(allocation["updates"])
        expected_nfe = int(protocol["evaluation"]["primary_nfe"][cell["arm"]])
        expected_imagined = (
            2 * int(execution["batch_size"]) * int(expected.imagination_horizon)
        )
        if (
            result.get("updates") != expected_updates
            or result.get("batch_size") != int(execution["batch_size"])
            or result.get("sequence_length") != int(execution["sequence_length"])
            or result.get("smoke_runtime_override") is not (cell["profile"] == "smoke")
            or result.get("structural_nfe_per_imagined_transition") != expected_nfe
            or result.get("imagined_transitions_per_update") != expected_imagined
            or result.get("prior_field_evaluations_per_update")
            != expected_imagined * expected_nfe
            or float(result.get("compiler_flops_per_update", math.nan))
            != float(allocation["flops_per_update"])
            or float(result.get("realized_compiler_flops", math.nan))
            != float(allocation["flops_per_update"]) * expected_updates
        ):
            raise ValueError("actor execution differs from its frozen allocation")
        dataset_cell = _dependency_cell(matrix, cell, "dataset")
        dataset_arrays = load_npz(
            stage_directory(output_root, dataset_cell) / "dataset.npz"
        )
        actor_schedule_seed = derive_seed(
            "actor-batches", cell["world_model_seed"], cell["actor_seed"]
        )
        expected_schedule = _batch_schedule(
            dataset_arrays,
            task=cell["task"],
            world_model_seed=actor_schedule_seed,
            updates=expected_updates,
            batch_size=int(execution["batch_size"]),
            sequence_length=int(execution["sequence_length"]),
        )
        if set(schedule) != set(expected_schedule) or any(
            not np.array_equal(schedule[name], expected_schedule[name])
            for name in expected_schedule
        ):
            raise ValueError("actor minibatch schedule is not canonical")
        fresh = create_agent(
            expected,
            derive_jax_key(
                "actor-init",
                cell["task"],
                cell["world_model_seed"],
                cell["actor_seed"],
            ),
        )
        if (
            object_sha256(asdict(actor_config)) != object_sha256(asdict(expected))
            or metadata.get("cell_id") != cell["cell_id"]
            or metadata.get("completed_updates") != result.get("updates")
            or int(np.asarray(actor_state.actor_optimizer.step)) != expected_updates
            or int(np.asarray(actor_state.critic_optimizer.step)) != expected_updates
            or metadata.get("world_model_checkpoint_sha256")
            != result.get("world_model_checkpoint_sha256")
            or _tree_max_abs_difference(
                actor_state.params.world_model, world_state.params.world_model
            )
            != 0.0
            or result.get("initial_actor_parameter_sha256")
            != _tree_digest(fresh.params.actor)
            or result.get("initial_critic_parameter_sha256")
            != _tree_digest(fresh.params.critic)
        ):
            raise ValueError("actor checkpoint identity or frozen world model mismatch")
        _validate_actor_environment_replay(
            traces,
            returns,
            state=actor_state,
            config=expected,
            task=cell["task"],
            world_model_seed=int(cell["world_model_seed"]),
            actor_seed=int(cell["actor_seed"]),
            maximum_steps=int(protocol["data"]["native_episode_limit"]),
            action_repeat=int(protocol["data"]["action_repeat"]),
        )


def run_actor_cell(
    cell: Mapping[str, Any],
    protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    output_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Train a nested actor seed while preserving the selected world model exactly."""

    _cell_identity(cell, "actor")
    _, world_result, world_directory = _load_dependency_result(
        output_root, matrix, cell, "world_model"
    )
    _, dataset_result, dataset_directory = _load_dependency_result(
        output_root, matrix, cell, "dataset"
    )
    _, compute, _ = _load_dependency_result(output_root, matrix, cell, "compute_plan")
    directory = stage_directory(output_root, cell)
    result_path = directory / "result.json"
    if not force and result_path.is_file():
        result = read_json(result_path)
        validate_actor_result(
            result,
            cell,
            directory,
            protocol=protocol,
            matrix=matrix,
            output_root=output_root,
        )
        return result
    if force and result_path.is_file():
        raise ValueError("force does not overwrite a complete actor cell; use a new output root")

    compute_cell = _dependency_cell(matrix, cell, "compute_plan")
    validate_compute_plan_cell(compute, protocol, matrix, compute_cell, output_root)
    _validate_compute_files(compute, stage_directory(output_root, compute_cell))

    import jax
    from imf_dreamer_jax import (
        AgentParams,
        AgentState,
        create_agent,
        diverse_imagination_starts,
        jit_observe_sequence,
        jit_train_actor_critic,
        load_checkpoint,
        save_checkpoint,
    )

    world_state, _, world_metadata = load_checkpoint(world_directory / "checkpoint.pkl")
    if world_metadata.get("cell_id") != _dependency_cell(matrix, cell, "world_model")["cell_id"]:
        raise ValueError("actor world-model checkpoint identity mismatch")
    config = make_config(
        protocol,
        cell["profile"],
        cell["arm"],
        dataset_result["observation_shape"],
        int(dataset_result["action_dim"]),
        output_root=output_root,
        candidate=_candidate_for_cell(matrix, cell),
    )
    fresh = create_agent(
        config,
        derive_jax_key(
            "actor-init", cell["task"], cell["world_model_seed"], cell["actor_seed"]
        ),
    )
    source_world = world_state.params.world_model
    state = AgentState(
        AgentParams(source_world, fresh.params.actor, fresh.params.critic),
        world_state.model_optimizer,
        fresh.actor_optimizer,
        fresh.critic_optimizer,
        fresh.slow_critic,
        world_state.world_model_teacher,
    )
    allocation = compute["tracks"][cell["budget_track"]]["allocations"][cell["arm"]]["actor"]
    updates = int(allocation["updates"])
    execution = _profile_execution(cell["profile"])
    batch_size = int(execution["batch_size"])
    sequence_length = int(execution["sequence_length"])
    arrays = load_npz(dataset_directory / "dataset.npz")
    actor_schedule_seed = derive_seed("actor-batches", cell["world_model_seed"], cell["actor_seed"])
    schedule = _batch_schedule(
        arrays,
        task=cell["task"],
        world_model_seed=actor_schedule_seed,
        updates=updates,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    directory.mkdir(parents=True, exist_ok=True)
    schedule_path = _write_npz_atomic(directory / "batch_schedule.npz", schedule)
    checkpoint_path = directory / "checkpoint.pkl"
    start_update = 0
    accumulated_wall = 0.0
    latest = None
    if checkpoint_path.is_file():
        state, stored_config, metadata = load_checkpoint(checkpoint_path)
        if stored_config != config or metadata.get("cell_id") != cell["cell_id"]:
            raise ValueError("partial actor checkpoint identity mismatch")
        if metadata.get("world_model_checkpoint_sha256") != world_result["checkpoint_sha256"]:
            raise ValueError("partial actor checkpoint world-model mismatch")
        start_update = int(metadata.get("completed_updates", -1))
        accumulated_wall = float(metadata.get("wall_seconds", 0.0))
        latest = metadata.get("last_metrics")
    if not 0 <= start_update <= updates:
        raise ValueError("partial actor update count is invalid")
    started = time.perf_counter()
    checkpoint_every = int(execution["checkpoint_every_updates"])
    posterior_base_key = derive_jax_key(
        "actor-posterior",
        cell["task"],
        cell["world_model_seed"],
        cell["actor_seed"],
    )
    start_base_key = derive_jax_key(
        "actor-start",
        cell["task"],
        cell["world_model_seed"],
        cell["actor_seed"],
    )
    objective_base_key = derive_jax_key(
        "actor-objective",
        cell["task"],
        cell["world_model_seed"],
        cell["actor_seed"],
    )
    for update in range(start_update, updates):
        batch = _materialize_batch(
            arrays,
            schedule,
            update,
            sequence_length=sequence_length,
            burn_in=config.burn_in,
        )
        sequence = jit_observe_sequence(
            state.params.world_model,
            batch["observations"],
            batch["actions"],
            jax.random.fold_in(posterior_base_key, update),
            config,
            is_first=batch["is_first"],
        )
        starts = diverse_imagination_starts(
            sequence.states,
            config.burn_in,
            jax.random.fold_in(start_base_key, update),
        )
        state, metrics = jit_train_actor_critic(
            state,
            starts,
            jax.random.fold_in(objective_base_key, update),
            config,
        )
        latest = _metrics_dict(metrics)
        completed = update + 1
        if completed % checkpoint_every == 0 or completed == updates:
            save_checkpoint(
                checkpoint_path,
                state,
                config,
                metadata={
                    "stage": "actor",
                    "cell_id": cell["cell_id"],
                    "world_model_checkpoint_sha256": world_result["checkpoint_sha256"],
                    "completed_updates": completed,
                    "last_metrics": latest,
                    "wall_seconds": accumulated_wall + time.perf_counter() - started,
                },
            )
    if latest is None:
        raise RuntimeError("actor cell completed no optimizer update")
    world_delta = _tree_max_abs_difference(state.params.world_model, source_world)
    if world_delta != 0.0:
        raise RuntimeError("actor training changed the frozen world model")
    episodes = int(protocol["profiles"][cell["profile"]]["real_environment_evaluation_episodes"])
    maximum_steps = int(protocol["data"]["native_episode_limit"])
    episode_returns, action_traces = _evaluate_actor_policy(
        state,
        config,
        task=cell["task"],
        world_model_seed=int(cell["world_model_seed"]),
        actor_seed=int(cell["actor_seed"]),
        episodes=episodes,
        maximum_steps=maximum_steps,
    )
    action_path = _write_npz_atomic(directory / "action_traces.npz", action_traces)
    save_checkpoint(
        checkpoint_path,
        state,
        config,
        metadata={
            "stage": "actor",
            "cell_id": cell["cell_id"],
            "world_model_checkpoint_sha256": world_result["checkpoint_sha256"],
            "completed_updates": updates,
            "last_metrics": latest,
            "wall_seconds": accumulated_wall + time.perf_counter() - started,
        },
    )
    total_wall = accumulated_wall + time.perf_counter() - started
    result = {
        **_result_identity(cell, ACTOR_SCHEMA),
        "matrix_sha256": matrix["matrix_sha256"],
        "world_model_cell_id": _dependency_cell(matrix, cell, "world_model")["cell_id"],
        "dataset_cell_id": _dependency_cell(matrix, cell, "dataset")["cell_id"],
        "world_model_checkpoint_sha256": world_result["checkpoint_sha256"],
        "updates": updates,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "smoke_runtime_override": cell["profile"] == "smoke",
        "structural_nfe_per_imagined_transition": int(protocol["evaluation"]["primary_nfe"][cell["arm"]]),
        "imagined_transitions_per_update": 2 * batch_size * config.imagination_horizon,
        "prior_field_evaluations_per_update": 2 * batch_size * config.imagination_horizon * int(protocol["evaluation"]["primary_nfe"][cell["arm"]]),
        "compiler_flops_per_update": float(allocation["flops_per_update"]),
        "realized_compiler_flops": float(allocation["flops_per_update"]) * updates,
        "world_model_frozen": True,
        "world_model_parameter_delta": world_delta,
        "initial_actor_parameter_sha256": _tree_digest(fresh.params.actor),
        "initial_critic_parameter_sha256": _tree_digest(fresh.params.critic),
        "final_metrics": latest,
        "episode_returns": [float(value) for value in episode_returns],
        "normalized_episode_returns": [float(value) / 1000.0 for value in episode_returns],
        "normalized_episode_return_mean": float(np.mean(episode_returns) / 1000.0),
        "evaluation_seed_derivation": "sha256(actor-evaluation,task,world_model_seed,actor_seed,episode)",
        "batch_schedule_sha256": array_sha256(schedule),
        "batch_schedule_file_sha256": file_sha256(schedule_path),
        "raw_action_traces_sha256": file_sha256(action_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "runtime": runtime_fingerprint(),
        "wall_seconds": total_wall,
    }
    validate_actor_result(
        result,
        cell,
        directory,
        protocol=protocol,
        matrix=matrix,
        output_root=output_root,
    )
    write_json_atomic(result_path, result)
    return result


def collect_verified_units(
    output_root: str | Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    validated_results: _ValidatedResultCache | None = None,
) -> list[dict[str, Any]]:
    """Load only canonical, digest-validated final cell artifacts for inference."""

    profile = str(matrix["profile"])
    selected = protocol["profiles"][profile]
    units: list[dict[str, Any]] = []
    for track in BUDGET_TRACK_ORDER:
        for task in selected["tasks"]:
            task = canonical_task_id(task)
            for world_seed in selected["world_model_seeds"]:
                rollout_auc: dict[str, float] = {}
                actor_returns: dict[str, dict[str, list[float]]] = {}
                artifact_hashes: dict[str, list[str]] = {}
                for arm in ARM_ORDER:
                    primary_nfe = int(protocol["evaluation"]["primary_nfe"][arm])
                    rollout_cells = [
                        cell
                        for cell in matrix["cells"]
                        if cell["stage"] == "rollout"
                        and cell["budget_track"] == track
                        and cell["task"] == task
                        and cell["world_model_seed"] == world_seed
                        and cell["arm"] == arm
                        and cell["nfe"] == primary_nfe
                    ]
                    if len(rollout_cells) != 1:
                        raise ValueError("primary rollout cell is missing or duplicated")
                    rollout_cell = rollout_cells[0]
                    rollout_result, rollout_path = _completed_result_for_cell(
                        output_root,
                        matrix,
                        protocol,
                        rollout_cell,
                        cache=validated_results,
                    )
                    if rollout_result.get("matrix_sha256") != matrix["matrix_sha256"]:
                        raise ValueError("rollout result belongs to another matrix")
                    rollout_auc[arm] = float(
                        rollout_result["normalized_free_running_rollout_error_auc"]
                    )
                    actor_returns[arm] = {}
                    artifact_hashes[arm] = [file_sha256(rollout_path)]
                    for actor_seed in selected["actor_seeds_nested_within_world_model_seed"]:
                        actor_cells = [
                            cell
                            for cell in matrix["cells"]
                            if cell["stage"] == "actor"
                            and cell["budget_track"] == track
                            and cell["task"] == task
                            and cell["world_model_seed"] == world_seed
                            and cell["arm"] == arm
                            and cell["actor_seed"] == actor_seed
                        ]
                        if len(actor_cells) != 1:
                            raise ValueError("nested actor cell is missing or duplicated")
                        actor_cell = actor_cells[0]
                        actor_result, actor_path = _completed_result_for_cell(
                            output_root,
                            matrix,
                            protocol,
                            actor_cell,
                            cache=validated_results,
                        )
                        if actor_result.get("matrix_sha256") != matrix["matrix_sha256"]:
                            raise ValueError("actor result belongs to another matrix")
                        actor_returns[arm][str(actor_seed)] = [
                            float(value) / 1000.0 for value in actor_result["episode_returns"]
                        ]
                        artifact_hashes[arm].append(file_sha256(actor_path))
                units.append(
                    {
                        "budget_track": track,
                        "task": task,
                        "world_model_seed": int(world_seed),
                        "rollout_auc": rollout_auc,
                        "actor_episode_returns": actor_returns,
                        "source_artifact_sha256s": artifact_hashes,
                    }
                )
    # The statistics routine consumes only registered estimand fields.
    statistical_units = [
        {key: value for key, value in unit.items() if key != "source_artifact_sha256s"}
        for unit in units
    ]
    _validate_complete_units(statistical_units, protocol, profile)
    return units


def _analysis_digest(analysis: Mapping[str, Any]) -> str:
    return object_sha256(_without_digest(analysis, "analysis_sha256"))


def collect_secondary_rollout_summary(
    output_root: str | Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    validated_results: _ValidatedResultCache | None = None,
) -> dict[str, Any]:
    """Aggregate descriptive rollout diagnostics without adding claim-bearing tests."""

    expected_units = (
        len(protocol["profiles"][matrix["profile"]]["tasks"])
        * len(protocol["profiles"][matrix["profile"]]["world_model_seeds"])
    )
    fields = (
        "expected_standardized_squared_error",
        "predictive_mean_observation_rmse",
        "reward_expected_squared_error",
        "observation_energy_score",
        "central_90_percent_interval_coverage",
        "central_90_percent_interval_width",
        "central_90_percent_interval_calibration_absolute_error",
        "continuation_expected_squared_error",
        "continuation_predictive_mean_brier",
    )
    tracks: dict[str, Any] = {}
    root = Path(output_root)
    for track in BUDGET_TRACK_ORDER:
        tracks[track] = {}
        for arm in ARM_ORDER:
            tracks[track][arm] = {}
            for nfe in NFE_FRONTIER:
                cells = [
                    cell
                    for cell in matrix_cells(matrix, "rollout")
                    if cell["budget_track"] == track
                    and cell["arm"] == arm
                    and int(cell["nfe"]) == nfe
                ]
                if len(cells) != expected_units:
                    raise ValueError("secondary rollout summary has incomplete task-seed units")
                results = []
                hashes = []
                for cell in cells:
                    result, path = _completed_result_for_cell(
                        root, matrix, protocol, cell, cache=validated_results
                    )
                    results.append(result)
                    hashes.append(file_sha256(path))
                per_horizon: dict[str, Any] = {}
                for horizon in protocol["evaluation"]["rollout_horizons"]:
                    horizon_rows = [result["per_horizon"][str(horizon)] for result in results]
                    if any(set(row) != set(fields) for row in horizon_rows):
                        raise ValueError("secondary rollout metric fields are incomplete")
                    per_horizon[str(horizon)] = {
                        f"{field}_iqm": interquartile_mean(
                            [float(row[field]) for row in horizon_rows]
                        )
                        for field in fields
                    }
                tracks[track][arm][str(nfe)] = {
                    "independent_task_world_seed_units": expected_units,
                    "normalized_rollout_error_auc_iqm": interquartile_mean(
                        [
                            float(result["normalized_free_running_rollout_error_auc"])
                            for result in results
                        ]
                    ),
                    "per_horizon": per_horizon,
                    "source_result_sha256s": hashes,
                }
    return {
        "status": "descriptive_non_claim_bearing",
        "aggregation": "IQM_over_task_x_world_model_seed_units_separately_by_track_arm_and_nfe",
        "tracks": tracks,
    }


def _analysis_auxiliary_evidence(
    output_root: str | Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    validated_results: _ValidatedResultCache | None = None,
) -> dict[str, Any]:
    validation = validate_confirmatory_auxiliary_evidence(
        output_root,
        matrix,
        protocol,
        validated_results=validated_results,
    )
    if not matrix["claim_eligible"]:
        return {
            "status": "not_required_for_nonclaim_profile",
            "validation": validation,
        }
    root = Path(output_root)
    validation_path = root / "auxiliary_validation.json"
    if not validation_path.is_file() or read_json(validation_path) != validation:
        raise ValueError("claim-bearing analysis requires stored auxiliary validation")
    diagnostic = read_json(root / "diagnostics" / "diagnostic_summary.json")
    passes = {
        name: {
            arm: bool(row["passed"])
            for arm, row in diagnostic["diagnostics"][name]["arms"].items()
        }
        for name in diagnostic["diagnostics"]
    }
    return {
        "status": "complete_interpretation_only_non_substitutive",
        "validation": validation,
        "validation_sha256": object_sha256(validation),
        "diagnostic_summary_sha256": diagnostic["diagnostic_summary_sha256"],
        "diagnostic_threshold_passes": passes,
        "all_diagnostic_thresholds_passed": all(
            passed for rows in passes.values() for passed in rows.values()
        ),
    }


def collect_compute_resource_summary(
    output_root: str | Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    validated_results: _ValidatedResultCache | None = None,
) -> dict[str, Any]:
    """Aggregate compiler/parameter evidence separately from outcome inference."""

    plans = [
        _completed_result_for_cell(
            output_root, matrix, protocol, cell, cache=validated_results
        )[0]
        for cell in matrix_cells(matrix, "compute_plan")
    ]
    if not plans:
        raise ValueError("resource summary requires compute-plan cells")
    arms: dict[str, Any] = {}
    for arm in ARM_ORDER:
        parameter_keys = tuple(plans[0]["arms"][arm]["parameters"])
        parameters = {
            key: sorted(
                {
                    int(plan["arms"][arm]["parameters"][key])
                    for plan in plans
                }
            )
            for key in parameter_keys
        }
        compiler_keys = tuple(plans[0]["arms"][arm]["compiler_flops"])
        compiler_flops = {
            f"{key}_mean": float(
                np.mean(
                    [
                        float(plan["arms"][arm]["compiler_flops"][key])
                        for plan in plans
                    ]
                )
            )
            for key in compiler_keys
        }
        inference = {
            str(nfe): {
                "compiler_forward_flops_per_transition_mean": float(
                    np.mean(
                        [
                            float(
                                plan["arms"][arm]["inference"][str(nfe)][
                                    "compiler_forward_flops_per_transition"
                                ]
                            )
                            for plan in plans
                        ]
                    )
                ),
                "structural_nfe_per_transition": nfe,
            }
            for nfe in NFE_FRONTIER
        }
        timing_rows = [plan["arms"][arm]["synchronized_walltime"] for plan in plans]
        synchronized_walltime: dict[str, Any] = {}
        for stage in ("world_model", "actor"):
            rows = [row[stage] for row in timing_rows]
            if all(row.get("status") == "complete" for row in rows):
                synchronized_walltime[stage] = {
                    "status": "complete",
                    "median_seconds_per_update_across_compute_cells": float(
                        np.median(
                            [float(row["median_seconds_per_update"]) for row in rows]
                        )
                    ),
                    "compute_cells": len(rows),
                }
            else:
                synchronized_walltime[stage] = {
                    "status": "not_run_engineering_smoke",
                    "compute_cells": len(rows),
                }
        arms[arm] = {
            "parameters_unique_values": parameters,
            "compiler_flops": compiler_flops,
            "inference": inference,
            "synchronized_walltime": synchronized_walltime,
        }
    tracks: dict[str, Any] = {}
    for track in BUDGET_TRACK_ORDER:
        tracks[track] = {}
        for arm in ARM_ORDER:
            allocation_rows = [plan["tracks"][track]["allocations"][arm] for plan in plans]
            stage_allocations: dict[str, Any] = {}
            for stage in ("world_model", "actor"):
                rows = [row[stage] for row in allocation_rows]
                stage_allocations[stage] = {
                    "updates_unique_values": sorted({int(row["updates"]) for row in rows}),
                    "flops_per_update_mean": float(
                        np.mean([float(row["flops_per_update"]) for row in rows])
                    ),
                    "cumulative_flops_mean": float(
                        np.mean([float(row["cumulative_flops"]) for row in rows])
                    ),
                    "maximum_relative_target_error": (
                        float(
                            max(
                                float(row["relative_target_error"])
                                for row in rows
                                if "relative_target_error" in row
                            )
                        )
                        if any("relative_target_error" in row for row in rows)
                        else None
                    ),
                }
            world_results = [
                _completed_result_for_cell(
                    output_root, matrix, protocol, cell, cache=validated_results
                )[0]
                for cell in matrix_cells(matrix, "world_model")
                if cell["budget_track"] == track and cell["arm"] == arm
            ]
            actor_results = [
                _completed_result_for_cell(
                    output_root, matrix, protocol, cell, cache=validated_results
                )[0]
                for cell in matrix_cells(matrix, "actor")
                if cell["budget_track"] == track and cell["arm"] == arm
            ]
            if not world_results or not actor_results:
                raise ValueError("resource summary lacks world or actor result cells")
            tracks[track][arm] = {
                "allocations": stage_allocations,
                "world_model_wall_seconds_iqm": interquartile_mean(
                    [float(result["wall_seconds"]) for result in world_results]
                ),
                "actor_wall_seconds_iqm": interquartile_mean(
                    [float(result["wall_seconds"]) for result in actor_results]
                ),
                "actor_imagined_transitions_per_update_unique_values": sorted(
                    {
                        int(result["imagined_transitions_per_update"])
                        for result in actor_results
                    }
                ),
                "actor_prior_field_evaluations_per_update_unique_values": sorted(
                    {
                        int(result["prior_field_evaluations_per_update"])
                        for result in actor_results
                    }
                ),
            }
    return {
        "status": "descriptive_non_claim_bearing",
        "compute_cells": len(plans),
        "homogeneous_runtime_identity": runtime_homogeneity_identity(plans[0]["runtime"]),
        "arms": arms,
        "tracks": tracks,
    }


def render_analysis_report(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# Matched objective benchmark result",
        "",
        f"Profile: `{analysis['profile']}` ({analysis['evidence_class']})",
        "",
    ]
    if analysis["profile"] != "confirmatory":
        lines.extend(
            [
                "> Engineering-only result. It is incapable of supporting the registered superiority claim.",
                "",
            ]
        )
    lines.extend(
        [
            "| Budget track | Metric | Favorable contrast | Adjusted interval |",
            "|---|---|---:|---:|",
        ]
    )
    for track in BUDGET_TRACK_ORDER:
        for metric in PRIMARY_METRICS:
            row = analysis["tracks"][track][metric]
            interval = row["interval"]
            lines.append(
                f"| {track} | {metric} | {row['contrast']:.6g} | "
                f"[{interval['lower']:.6g}, {interval['upper']:.6g}] |"
            )
    decision = analysis["superiority"]
    practical = analysis["practical_significance"]
    lines.extend(
        [
            "",
            "## Preregistered practical-effect interpretation",
            "",
            "These point-estimate thresholds cannot substitute for the adjusted confidence-interval gate.",
            "",
            "| Budget track | Rollout absolute gain | Rollout relative reduction | Rollout practical | Actor normalized gain | Actor practical |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for track in BUDGET_TRACK_ORDER:
        rollout = practical["tracks"][track][PRIMARY_METRICS[0]]
        actor = practical["tracks"][track][PRIMARY_METRICS[1]]
        lines.append(
            f"| {track} | {rollout['absolute_iqm_contrast']:.6g} | "
            f"{rollout['relative_iqm_reduction']:.2%} | "
            f"{'PASS' if rollout['passed'] else 'FAIL'} | "
            f"{actor['absolute_iqm_contrast']:.6g} | "
            f"{'PASS' if actor['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "Practically meaningful superiority wording: "
            f"**{'ALLOWED' if practical['practically_meaningful_superiority_claim_allowed'] else 'NOT ALLOWED'}**.",
            "",
            "### Per-task paired-effect heterogeneity",
            "",
            "| Budget track | Task | Rollout contrast IQM | Rollout favorable seeds | Actor contrast IQM | Actor favorable seeds |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for track in BUDGET_TRACK_ORDER:
        for task, row in practical["per_task"][track].items():
            rollout = row[PRIMARY_METRICS[0]]
            actor = row[PRIMARY_METRICS[1]]
            lines.append(
                f"| {track} | {task} | "
                f"{rollout['paired_seed_contrast_iqm']:.6g} | "
                f"{rollout['favorable_seed_fraction']:.2%} | "
                f"{actor['paired_seed_contrast_iqm']:.6g} | "
                f"{actor['favorable_seed_fraction']:.2%} |"
            )
    lines.extend(
        [
            "",
            "## Descriptive rollout diagnostics across the NFE frontier",
            "",
            "These IQMs are supporting summaries; they do not enter the superiority gate.",
            "",
            "| Budget track | Arm | NFE | Rollout AUC IQM | H30 calibration error IQM | H30 energy score IQM |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for track in BUDGET_TRACK_ORDER:
        for arm in ARM_ORDER:
            for nfe in NFE_FRONTIER:
                row = analysis["secondary_rollout"]["tracks"][track][arm][str(nfe)]
                horizon = row["per_horizon"]["30"]
                lines.append(
                    f"| {track} | {arm} | {nfe} | "
                    f"{row['normalized_rollout_error_auc_iqm']:.6g} | "
                    f"{horizon['central_90_percent_interval_calibration_absolute_error_iqm']:.6g} | "
                    f"{horizon['observation_energy_score_iqm']:.6g} |"
                )
    lines.extend(
        [
            "",
            "### Per-horizon descriptive IQMs",
            "",
            "| Track | Arm | NFE | Horizon | Std. squared error | Reward MSE | Continuation MSE | Continuation Brier | Coverage | Calibration error | Energy score |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for track in BUDGET_TRACK_ORDER:
        for arm in ARM_ORDER:
            for nfe in NFE_FRONTIER:
                per_horizon = analysis["secondary_rollout"]["tracks"][track][arm][
                    str(nfe)
                ]["per_horizon"]
                for horizon in analysis["rollout_horizons"]:
                    row = per_horizon[str(horizon)]
                    lines.append(
                        f"| {track} | {arm} | {nfe} | {horizon} | "
                        f"{row['expected_standardized_squared_error_iqm']:.6g} | "
                        f"{row['reward_expected_squared_error_iqm']:.6g} | "
                        f"{row['continuation_expected_squared_error_iqm']:.6g} | "
                        f"{row['continuation_predictive_mean_brier_iqm']:.6g} | "
                        f"{row['central_90_percent_interval_coverage_iqm']:.6g} | "
                        f"{row['central_90_percent_interval_calibration_absolute_error_iqm']:.6g} | "
                        f"{row['observation_energy_score_iqm']:.6g} |"
                    )
    lines.extend(
        [
            "",
            "## Parameters and compiler cost",
            "",
            "| Arm | Active world parameters | Total world parameters | World train FLOPs/update | Actor train FLOPs/update |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in ARM_ORDER:
        resource = analysis["compute_resources"]["arms"][arm]
        parameters = resource["parameters_unique_values"]
        flops = resource["compiler_flops"]
        lines.append(
            f"| {arm} | {parameters['world_model_active']} | "
            f"{parameters['world_model_total']} | "
            f"{flops['world_forward_and_backward_train_mean']:.6g} | "
            f"{flops['actor_forward_and_backward_train_mean']:.6g} |"
        )
    lines.extend(
        [
            "",
            "| Arm | NFE | Inference FLOPs/transition | World seconds/update | Actor seconds/update |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in ARM_ORDER:
        resource = analysis["compute_resources"]["arms"][arm]
        world_timing = resource["synchronized_walltime"]["world_model"]
        actor_timing = resource["synchronized_walltime"]["actor"]
        world_seconds = world_timing.get(
            "median_seconds_per_update_across_compute_cells", "n/a"
        )
        actor_seconds = actor_timing.get(
            "median_seconds_per_update_across_compute_cells", "n/a"
        )
        for nfe in NFE_FRONTIER:
            inference_flops = resource["inference"][str(nfe)][
                "compiler_forward_flops_per_transition_mean"
            ]
            lines.append(
                f"| {arm} | {nfe} | {inference_flops:.6g} | "
                f"{world_seconds} | {actor_seconds} |"
            )
    lines.extend(
        [
            "",
            "### Realized budget and actor-imagination accounting",
            "",
            "| Track | Arm | World updates | Actor updates | World cumulative FLOPs | Actor cumulative FLOPs | Max allocation error | Imagined transitions/update | Prior field evals/update | World wall-time IQM (s) | Actor wall-time IQM (s) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for track in BUDGET_TRACK_ORDER:
        for arm in ARM_ORDER:
            row = analysis["compute_resources"]["tracks"][track][arm]
            world = row["allocations"]["world_model"]
            actor = row["allocations"]["actor"]
            allocation_errors = [
                value
                for value in (
                    world["maximum_relative_target_error"],
                    actor["maximum_relative_target_error"],
                )
                if value is not None
            ]
            maximum_error = max(allocation_errors) if allocation_errors else None
            maximum_error_text = "n/a" if maximum_error is None else f"{maximum_error:.6g}"
            lines.append(
                f"| {track} | {arm} | {world['updates_unique_values']} | "
                f"{actor['updates_unique_values']} | {world['cumulative_flops_mean']:.6g} | "
                f"{actor['cumulative_flops_mean']:.6g} | {maximum_error_text} | "
                f"{row['actor_imagined_transitions_per_update_unique_values']} | "
                f"{row['actor_prior_field_evaluations_per_update_unique_values']} | "
                f"{row['world_model_wall_seconds_iqm']:.6g} | "
                f"{row['actor_wall_seconds_iqm']:.6g} |"
            )
    auxiliary = analysis["auxiliary_evidence"]
    if analysis["profile"] == "confirmatory":
        lines.extend(
            [
                "",
                "## Required interpretation diagnostics",
                "",
                "Diagnostic thresholds are engineering checks only: they cannot pass or fail the primary superiority gate.",
                "",
                "| Diagnostic | Arm | Threshold check |",
                "|---|---|---:|",
            ]
        )
        for name, rows in auxiliary["diagnostic_threshold_passes"].items():
            for arm, passed in rows.items():
                lines.append(f"| {name} | {arm} | {'PASS' if passed else 'FAIL'} |")
        if not auxiliary["all_diagnostic_thresholds_passed"]:
            lines.extend(
                [
                    "",
                    "> Warning: at least one mandatory interpretation diagnostic missed its engineering threshold. The primary statistical gate is unchanged, but a superiority result must be interpreted with this failure visible.",
                ]
            )
    lines.extend(
        [
            "",
            f"Registered superiority gate: **{'PASS' if decision['passed'] else 'FAIL'}**.",
            "",
            "The shortcut arm is a paper-derived Dreamer 4 Equation-(7)-style reimplementation with explicitly declared choices; this is not the unreleased Dreamer 4 system.",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(
    output_root: str | Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    resamples: int | None = None,
    validated_results: _ValidatedResultCache | None = None,
) -> dict[str, Any]:
    units_with_provenance = collect_verified_units(
        output_root,
        matrix,
        protocol,
        validated_results=validated_results,
    )
    units = [
        {key: value for key, value in unit.items() if key != "source_artifact_sha256s"}
        for unit in units_with_provenance
    ]
    if resamples is None and matrix["profile"] == "smoke":
        resamples = 1000
    analysis, indices = analyze_units(
        units, protocol, matrix["profile"], resamples=resamples
    )
    analysis.update(
        {
            "matrix_sha256": matrix["matrix_sha256"],
            "source_sha256": matrix["source_sha256"],
            "selection_sha256": matrix["selection_sha256"],
            "source_artifact_sha256s": [
                unit["source_artifact_sha256s"] for unit in units_with_provenance
            ],
            "secondary_rollout": collect_secondary_rollout_summary(
                output_root,
                matrix,
                protocol,
                validated_results=validated_results,
            ),
            "auxiliary_evidence": _analysis_auxiliary_evidence(
                output_root,
                matrix,
                protocol,
                validated_results=validated_results,
            ),
            "compute_resources": collect_compute_resource_summary(
                output_root,
                matrix,
                protocol,
                validated_results=validated_results,
            ),
        }
    )
    analysis["analysis_sha256"] = _analysis_digest(analysis)
    root = Path(output_root)
    _write_npz_atomic(
        root / "bootstrap_indices.npz",
        {f"{track}_world_model_indices": value for track, value in indices.items()},
    )
    write_json_atomic(root / "analysis.json", analysis)
    (root / "REPORT.md").write_text(render_analysis_report(analysis), encoding="utf-8")
    return analysis


def validate_analysis(
    analysis: Mapping[str, Any],
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    if (
        analysis.get("schema_version") != ANALYSIS_SCHEMA
        or analysis.get("status") != "complete"
        or analysis.get("profile") != matrix["profile"]
        or analysis.get("evidence_class") != matrix["evidence_class"]
        or analysis.get("protocol_sha256") != matrix["protocol_sha256"]
        or analysis.get("source_sha256") != matrix["source_sha256"]
        or analysis.get("matrix_sha256") != matrix["matrix_sha256"]
        or analysis.get("selection_sha256") != matrix["selection_sha256"]
        or analysis.get("analysis_sha256") != _analysis_digest(analysis)
    ):
        raise ValueError("analysis identity or digest mismatch")
    count = int(analysis["bootstrap"]["resamples"])
    configured = int(protocol["statistics"]["bootstrap"]["resamples"])
    if matrix["claim_eligible"] and count != configured:
        raise ValueError("claim-eligible analysis does not use the frozen bootstrap count")
    decision = evaluate_superiority(
        protocol,
        {
            track: {
                metric: analysis["tracks"][track][metric]["interval"]
                for metric in PRIMARY_METRICS
            }
            for track in BUDGET_TRACK_ORDER
        },
        evidence_class=analysis["evidence_class"],
    )
    if object_sha256(analysis.get("superiority")) != object_sha256(decision.to_dict()):
        raise ValueError("saved superiority decision is not reproducible")
    practical = _evaluate_practical_significance(
        analysis["tracks"],
        protocol,
        evidence_class=analysis["evidence_class"],
        statistical_superiority_passed=decision.passed,
    )
    practical.update(
        _practical_heterogeneity(
            analysis["raw_units"], protocol, analysis["profile"]
        )
    )
    if canonical_bytes(analysis.get("practical_significance")) != canonical_bytes(
        practical
    ):
        raise ValueError("saved practical-significance decision is not reproducible")


def validate_analysis_artifacts(
    analysis: Mapping[str, Any],
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
    output_root: str | Path,
    *,
    validated_results: _ValidatedResultCache | None = None,
) -> None:
    """Recompute the registered estimands and bootstrap draw indices from cell artifacts."""

    validate_analysis(analysis, matrix, protocol)
    units_with_provenance = collect_verified_units(
        output_root,
        matrix,
        protocol,
        validated_results=validated_results,
    )
    units = [
        {key: value for key, value in unit.items() if key != "source_artifact_sha256s"}
        for unit in units_with_provenance
    ]
    expected, indices = analyze_units(
        units,
        protocol,
        matrix["profile"],
        resamples=int(analysis["bootstrap"]["resamples"]),
    )
    expected.update(
        {
            "matrix_sha256": matrix["matrix_sha256"],
            "source_sha256": matrix["source_sha256"],
            "selection_sha256": matrix["selection_sha256"],
            "source_artifact_sha256s": [
                unit["source_artifact_sha256s"] for unit in units_with_provenance
            ],
            "secondary_rollout": collect_secondary_rollout_summary(
                output_root,
                matrix,
                protocol,
                validated_results=validated_results,
            ),
            "auxiliary_evidence": _analysis_auxiliary_evidence(
                output_root,
                matrix,
                protocol,
                validated_results=validated_results,
            ),
            "compute_resources": collect_compute_resource_summary(
                output_root,
                matrix,
                protocol,
                validated_results=validated_results,
            ),
        }
    )
    expected["analysis_sha256"] = _analysis_digest(expected)
    if canonical_bytes(analysis) != canonical_bytes(expected):
        raise ValueError("saved analysis does not recompute from verified cell artifacts")
    stored = load_npz(Path(output_root) / "bootstrap_indices.npz")
    expected_keys = {f"{track}_world_model_indices" for track in BUDGET_TRACK_ORDER}
    if set(stored) != expected_keys or any(
        not np.array_equal(stored[f"{track}_world_model_indices"], indices[track])
        for track in BUDGET_TRACK_ORDER
    ):
        raise ValueError("retained bootstrap indices do not reproduce the analysis")
    report_path = Path(output_root) / "REPORT.md"
    expected_report = render_analysis_report(analysis)
    if (
        not report_path.is_file()
        or report_path.read_text(encoding="utf-8") != expected_report
    ):
        raise ValueError("claim-facing REPORT.md does not reproduce the analysis")


def _write_environment_and_dependencies(root: Path, *, strict: bool = False) -> None:
    dependency_status = "complete"
    try:
        dependency_text = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        dependency_status = "failed"
        dependency_text = f"dependency capture failed: {type(error).__name__}: {error}\n"
    (root / "dependency_lock.txt").write_text(dependency_text, encoding="utf-8")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "JAX_PLATFORM_NAME",
                "XLA_FLAGS",
                "CUDA_VISIBLE_DEVICES",
                "JAX_ENABLE_COMPILATION_CACHE",
                "JAX_COMPILATION_CACHE_DIR",
                "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
                "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
                "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES",
                "JAX_RAISE_PERSISTENT_CACHE_ERRORS",
            )
        },
        "dependency_capture_status": dependency_status,
    }
    try:
        import jax
        import jaxlib

        devices = jax.devices()
        client = devices[0].client if devices else None

        environment["jax"] = {
            "version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in devices],
            "device_platforms": [str(device.platform) for device in devices],
            "device_kinds": [str(device.device_kind) for device in devices],
            "visible_device_count": len(devices),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "xla_platform_version": getattr(client, "platform_version", "unavailable"),
        }
    except Exception as error:  # pragma: no cover - useful on scheduler login nodes
        environment["jax_error"] = f"{type(error).__name__}: {error}"
    write_json_atomic(root / "environment.json", environment)
    if strict and dependency_status != "complete":
        raise RuntimeError("claim-bearing freeze requires a complete dependency lock")
    if strict and (
        "jax" not in environment or environment["jax"].get("jax_enable_x64") is not False
    ):
        raise RuntimeError("claim-bearing freeze requires JAX with x64 disabled")


def validate_frozen_runtime_provenance(
    output_root: str | Path,
    source: Mapping[str, Any],
    *,
    strict: bool,
    matrix: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate the frozen environment and bind it to compute workers."""

    root = Path(output_root)
    identity_path = root / "source_identity.txt"
    dependency_path = root / "dependency_lock.txt"
    environment_path = root / "environment.json"
    if not identity_path.is_file() or not dependency_path.is_file() or not environment_path.is_file():
        raise FileNotFoundError("frozen runtime provenance files are incomplete")
    git = source["git"]
    expected_identity = (
        f"commit={git['commit']}\n"
        f"dirty_patch_sha256={git['dirty_patch_sha256']}\n"
    )
    if identity_path.read_text(encoding="utf-8") != expected_identity:
        raise ValueError("source_identity.txt differs from the source manifest")
    dependency_text = dependency_path.read_text(encoding="utf-8")
    if not dependency_text.strip():
        raise ValueError("dependency lock is empty")
    environment = read_json(environment_path)
    required_environment = {
        "python",
        "platform",
        "executable",
        "environment",
        "dependency_capture_status",
    }
    if not isinstance(environment, Mapping) or not required_environment <= set(environment):
        raise ValueError("environment provenance schema is incomplete")
    if strict:
        if (
            environment["dependency_capture_status"] != "complete"
            or dependency_text.startswith("dependency capture failed:")
            or set(environment) != required_environment | {"jax"}
            or git["commit_status"] != "complete"
            or git["dirty_patch_status"] != "complete"
            or git["untracked_status"] != "complete"
            or git["dirty_patch_bytes"] != 0
            or git["untracked_files"]
        ):
            raise ValueError(
                "claim-bearing provenance requires a complete lock and clean committed source"
            )
    if "jax" in environment:
        jax_environment = environment["jax"]
        required_jax = {
            "version",
            "jaxlib_version",
            "backend",
            "devices",
            "device_platforms",
            "device_kinds",
            "visible_device_count",
            "jax_enable_x64",
            "xla_platform_version",
        }
        if (
            not isinstance(jax_environment, Mapping)
            or set(jax_environment) != required_jax
            or jax_environment["visible_device_count"]
            != len(jax_environment["devices"])
            or jax_environment["visible_device_count"]
            != len(jax_environment["device_platforms"])
            or jax_environment["visible_device_count"]
            != len(jax_environment["device_kinds"])
        ):
            raise ValueError("frozen JAX environment is malformed")
    elif strict:
        raise ValueError("claim-bearing environment lacks JAX provenance")
    if matrix is not None:
        compute_cells = matrix_cells(matrix, "compute_plan")
        if not compute_cells:
            raise ValueError("runtime provenance requires compute-plan cells")
        runtimes = [
            read_json(_cell_result_path(root, cell))["runtime"]
            for cell in compute_cells
        ]
        identity = runtime_homogeneity_identity(runtimes[0])
        if any(runtime_homogeneity_identity(row) != identity for row in runtimes[1:]):
            raise ValueError("compute-plan runtime provenance is heterogeneous")
        if "jax" in environment:
            frozen_identity = {
                "python": environment["python"],
                "jax_version": environment["jax"]["version"],
                "jaxlib_version": environment["jax"]["jaxlib_version"],
                "backend": environment["jax"]["backend"],
                "device_platforms": environment["jax"]["device_platforms"],
                "device_kinds": environment["jax"]["device_kinds"],
                "visible_device_count": environment["jax"]["visible_device_count"],
                "xla_platform_version": environment["jax"]["xla_platform_version"],
                "jax_enable_x64": environment["jax"]["jax_enable_x64"],
                "jax_enable_compilation_cache": environment["environment"][
                    "JAX_ENABLE_COMPILATION_CACHE"
                ],
                "jax_compilation_cache_dir": environment["environment"][
                    "JAX_COMPILATION_CACHE_DIR"
                ],
                "jax_persistent_cache_min_compile_time_secs": environment[
                    "environment"
                ]["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"],
                "jax_persistent_cache_min_entry_size_bytes": environment[
                    "environment"
                ]["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"],
                "jax_persistent_cache_enable_xla_caches": environment[
                    "environment"
                ]["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"],
                "jax_raise_persistent_cache_errors": environment["environment"][
                    "JAX_RAISE_PERSISTENT_CACHE_ERRORS"
                ],
            }
            if frozen_identity != identity:
                raise ValueError("freeze environment differs from compute-plan hardware/compiler")
    return {
        "status": "complete",
        "strict": strict,
        "dependency_lock_sha256": file_sha256(dependency_path),
        "environment_sha256": file_sha256(environment_path),
        "source_identity_sha256": file_sha256(identity_path),
    }


def freeze_run(
    protocol: Mapping[str, Any],
    profile: str,
    output_root: str | Path,
    *,
    workspace: str | Path | None = None,
    selection_manifest: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze protocol, source, selection, and all cell identities before outcomes."""

    root = Path(output_root)
    if profile == "confirmatory" and root.exists():
        outcome_paths = [
            root / stage for stage in STAGE_ORDER
        ] + [root / "analysis.json", root / "REPORT.md"]
        if any(path.exists() for path in outcome_paths):
            raise ValueError(
                "confirmatory outcomes already exist before the pilot selection was frozen"
            )
    root.mkdir(parents=True, exist_ok=True)
    source = build_source_manifest(workspace)
    if profile == "confirmatory" and (
        source["git"]["commit_status"] != "complete"
        or source["git"]["dirty_patch_status"] != "complete"
        or source["git"]["untracked_status"] != "complete"
        or source["git"]["dirty_patch_bytes"] != 0
        or source["git"]["untracked_files"]
    ):
        raise RuntimeError(
            "confirmatory freeze requires a clean committed relevant source tree"
        )
    matrix = build_matrix(
        protocol,
        profile,
        source_manifest=source,
        selection_manifest=selection_manifest,
    )
    frozen_paths = {
        "frozen_protocol.json": protocol,
        "source_manifest.json": source,
        "matrix.json": matrix,
    }
    for filename, payload in frozen_paths.items():
        path = root / filename
        if path.is_file() and read_json(path) != payload:
            raise ValueError(f"existing frozen artifact differs: {filename}")
        write_json_atomic(path, payload)
    if profile == "confirmatory":
        selection_path = root / "hpo_selection.json"
        if (
            selection_path.is_file()
            and read_json(selection_path) != matrix["selection_manifest"]
        ):
            raise ValueError("existing frozen artifact differs: hpo_selection.json")
        write_json_atomic(selection_path, matrix["selection_manifest"])
    _write_environment_and_dependencies(root, strict=profile == "confirmatory")
    git = source["git"]
    (root / "source_identity.txt").write_text(
        f"commit={git['commit']}\ndirty_patch_sha256={git['dirty_patch_sha256']}\n",
        encoding="utf-8",
    )
    return source, matrix


def freeze_pilot_hpo_run(
    protocol: Mapping[str, Any],
    output_root: str | Path,
    *,
    workspace: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze the complete 24-candidate pilot grid before pilot outcomes."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    source = build_source_manifest(workspace)
    if (
        source["git"]["commit_status"] != "complete"
        or source["git"]["dirty_patch_status"] != "complete"
        or source["git"]["untracked_status"] != "complete"
        or source["git"]["dirty_patch_bytes"] != 0
        or source["git"]["untracked_files"]
    ):
        raise RuntimeError(
            "pilot HPO freeze requires a clean committed relevant source tree"
        )
    matrix = build_pilot_hpo_matrix(protocol, source_manifest=source)
    frozen_paths = {
        "frozen_protocol.json": protocol,
        "source_manifest.json": source,
        "matrix.json": matrix,
    }
    for filename, payload in frozen_paths.items():
        path = root / filename
        if path.is_file() and read_json(path) != payload:
            raise ValueError(f"existing frozen artifact differs: {filename}")
        write_json_atomic(path, payload)
    _write_environment_and_dependencies(root, strict=True)
    git = source["git"]
    (root / "source_identity.txt").write_text(
        f"commit={git['commit']}\ndirty_patch_sha256={git['dirty_patch_sha256']}\n",
        encoding="utf-8",
    )
    return source, matrix


def run_cell(
    cell: Mapping[str, Any],
    protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    stage = cell["stage"]
    if stage == "dataset":
        return run_dataset_cell(cell, protocol, matrix, output_root)
    if stage == "compute_plan":
        return run_compute_plan_cell(cell, protocol, matrix, output_root)
    if stage == "world_model":
        return run_world_model_cell(cell, protocol, matrix, output_root)
    if stage == "rollout":
        return run_rollout_cell(cell, protocol, matrix, output_root)
    if stage == "actor":
        return run_actor_cell(cell, protocol, matrix, output_root)
    raise ValueError(f"unsupported benchmark stage {stage!r}")


def run_pending_cells(
    protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    output_root: str | Path,
    *,
    stages: Sequence[str] = STAGE_ORDER,
) -> dict[str, int]:
    """Run every pending canonical cell in dependency order; completed cells validate and skip."""

    source = read_json(Path(output_root) / "source_manifest.json")
    validate_source_manifest(source)
    if matrix.get("schema_version") == HPO_MATRIX_SCHEMA:
        validate_pilot_hpo_matrix(matrix, protocol, source)
    else:
        validate_matrix(matrix, protocol, source)
    counts = {stage: 0 for stage in STAGE_ORDER}
    selected_stages = set(stages)
    if not selected_stages <= set(STAGE_ORDER):
        raise ValueError("unknown requested execution stage")
    for stage in STAGE_ORDER:
        if stage not in selected_stages:
            continue
        for cell in matrix_cells(matrix, stage):
            run_cell(cell, protocol, matrix, output_root)
            counts[stage] += 1
    return counts


def _require_common_prefix(
    payloads: Sequence[Mapping[str, np.ndarray]], label: str
) -> None:
    if not payloads:
        raise ValueError(f"no payloads supplied for paired {label}")
    names = set(payloads[0])
    if any(set(payload) != names for payload in payloads[1:]):
        raise ValueError(f"paired {label} payload fields differ")
    for name in names:
        arrays = [np.asarray(payload[name]) for payload in payloads]
        if any(array.ndim != arrays[0].ndim for array in arrays[1:]):
            raise ValueError(f"paired {label}.{name} ranks differ")
        if arrays[0].ndim == 0:
            if any(not np.array_equal(arrays[0], array) for array in arrays[1:]):
                raise ValueError(f"paired {label}.{name} scalar differs")
            continue
        prefix = min(array.shape[0] for array in arrays)
        reference = arrays[0][:prefix]
        if any(
            array.shape[1:] != reference.shape[1:]
            or not np.array_equal(reference, array[:prefix])
            for array in arrays[1:]
        ):
            raise ValueError(f"paired {label}.{name} common prefix differs")


def _require_npz_fields_equal(
    paths: Sequence[Path], fields: Sequence[str], label: str
) -> None:
    """Compare selected NPZ arrays one field/file at a time to cap verifier memory."""

    if not paths:
        raise ValueError(f"no payloads supplied for paired {label}")
    for name in fields:
        with np.load(paths[0], allow_pickle=False) as archive:
            if name not in archive.files:
                raise ValueError(f"paired {label} reference lacks {name!r}")
            reference = np.asarray(archive[name])
        for path in paths[1:]:
            with np.load(path, allow_pickle=False) as archive:
                if name not in archive.files:
                    raise ValueError(f"paired {label} payload lacks {name!r}")
                candidate = np.asarray(archive[name])
            if not np.array_equal(reference, candidate):
                raise ValueError(f"paired {label}.{name} differs")


def validate_pairing_invariants(
    output_root: str | Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    validated_results: _ValidatedResultCache | None = None,
) -> dict[str, Any]:
    """Prove the common-random-number contract across every paired result cell."""

    root = Path(output_root)
    selected = protocol["profiles"][matrix["profile"]]
    compute_results = [
        _completed_result_for_cell(
            root, matrix, protocol, cell, cache=validated_results
        )[0]
        for cell in matrix_cells(matrix, "compute_plan")
    ]
    if not compute_results:
        raise ValueError("matrix contains no compute-plan runtime evidence")
    reference_runtime = runtime_homogeneity_identity(compute_results[0]["runtime"])
    if any(
        runtime_homogeneity_identity(result["runtime"]) != reference_runtime
        for result in compute_results[1:]
    ):
        raise ValueError(
            "compute-plan cells used heterogeneous hardware/compiler runtimes"
        )
    world_groups = 0
    rollout_groups = 0
    actor_groups = 0
    for raw_task in selected["tasks"]:
        task = canonical_task_id(raw_task)
        for world_seed in selected["world_model_seeds"]:
            worlds = [
                cell
                for cell in matrix["cells"]
                if cell["stage"] == "world_model"
                and cell["task"] == task
                and cell["world_model_seed"] == int(world_seed)
            ]
            if not worlds:
                raise ValueError("paired world-model group is empty")
            world_results = [
                _completed_result_for_cell(
                    root, matrix, protocol, cell, cache=validated_results
                )[0]
                for cell in worlds
            ]
            if len(
                {result["shared_initial_world_parameter_sha256"] for result in world_results}
            ) != 1:
                raise ValueError("paired world-model shared initialization differs")
            _require_common_prefix(
                [load_npz(stage_directory(root, cell) / "batch_schedule.npz") for cell in worlds],
                "world-model minibatch schedule",
            )
            _require_common_prefix(
                [load_npz(stage_directory(root, cell) / "objective_rng_keys.npz") for cell in worlds],
                "world-model objective RNG schedule",
            )
            world_groups += 1

            rollouts = [
                cell
                for cell in matrix["cells"]
                if cell["stage"] == "rollout"
                and cell["task"] == task
                and cell["world_model_seed"] == int(world_seed)
            ]
            if not rollouts:
                raise ValueError("paired rollout group is empty")
            paired_fields = (
                "target_observations",
                "target_rewards",
                "target_continuations",
                "context_observations",
                "context_actions",
                "future_actions",
                "noise",
                "episode_ids",
                "anchors",
                "training_observation_std",
            )
            _require_npz_fields_equal(
                [
                    stage_directory(root, cell) / "predictive_draws.npz"
                    for cell in rollouts
                ],
                paired_fields,
                "rollout starts, targets, actions, and noise",
            )
            rollout_groups += 1

            for actor_seed in selected["actor_seeds_nested_within_world_model_seed"]:
                actors = [
                    cell
                    for cell in matrix["cells"]
                    if cell["stage"] == "actor"
                    and cell["task"] == task
                    and cell["world_model_seed"] == int(world_seed)
                    and cell["actor_seed"] == int(actor_seed)
                ]
                if not actors:
                    raise ValueError("paired actor group is empty")
                actor_results = [
                    _completed_result_for_cell(
                        root, matrix, protocol, cell, cache=validated_results
                    )[0]
                    for cell in actors
                ]
                if (
                    len({result["initial_actor_parameter_sha256"] for result in actor_results})
                    != 1
                    or len(
                        {result["initial_critic_parameter_sha256"] for result in actor_results}
                    )
                    != 1
                ):
                    raise ValueError("paired actor or critic initialization differs")
                _require_common_prefix(
                    [
                        load_npz(stage_directory(root, cell) / "batch_schedule.npz")
                        for cell in actors
                    ],
                    "actor minibatch schedule",
                )
                actor_groups += 1
    evidence = {
        "schema_version": "matched-objective-pairing-v1",
        "status": "complete",
        "profile": matrix["profile"],
        "protocol_sha256": matrix["protocol_sha256"],
        "source_sha256": matrix["source_sha256"],
        "matrix_sha256": matrix["matrix_sha256"],
        "compute_runtime_identity": reference_runtime,
        "world_model_groups": world_groups,
        "rollout_groups": rollout_groups,
        "actor_groups": actor_groups,
        "checks": [
            "homogeneous_compute_hardware_and_compiler_runtime",
            "shared_world_initialization",
            "world_minibatch_common_prefix",
            "objective_rng_common_prefix",
            "rollout_starts_targets_actions_and_noise",
            "shared_actor_and_critic_initialization",
            "actor_minibatch_common_prefix",
        ],
    }
    evidence["pairing_sha256"] = object_sha256(evidence)
    return evidence


def _safe_artifact_path(
    root: Path,
    relative: Any,
    *,
    required_parent: str | None = None,
) -> Path:
    """Resolve an artifact path while rejecting traversal and escaping symlinks."""

    if not isinstance(relative, str) or not relative:
        raise ValueError("artifact path must be a nonempty relative string")
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("artifact path must not be absolute or contain parent traversal")
    root_resolved = root.resolve()
    path = (root / value).resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError("artifact path escapes the output root") from error
    if required_parent is not None:
        parent = (root / required_parent).resolve()
        try:
            path.relative_to(parent)
        except ValueError as error:
            raise ValueError(
                f"artifact path must remain below {required_parent!r}"
            ) from error
    return path


def _diagnostic_threshold_pass(
    name: str,
    estimands: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> bool:
    values = {key: float(value) for key, value in estimands.items()}
    limits = {key: float(value) for key, value in thresholds.items()}
    if name == "deterministic_transition_diagnostic":
        return (
            values["one_step_normalized_RMSE"]
            <= limits["one_step_normalized_RMSE_max"]
            and values["horizon_30_normalized_RMSE"]
            <= limits["horizon_30_normalized_RMSE_max"]
            and values["horizon_30_to_horizon_1_error_ratio"]
            <= limits["horizon_30_to_horizon_1_error_ratio_max"]
        )
    if name == "genuinely_stochastic_transition_diagnostic":
        energy_ok = all(
            values[f"energy_score_over_oracle_ratio_horizon_{horizon}"]
            <= limits[f"energy_score_over_oracle_ratio_horizon_{horizon}_max"]
            for horizon in (1, 8, 30)
        )
        coverage_ok = all(
            limits[f"central_90_percent_interval_coverage_horizon_{horizon}_min"]
            <= values[f"central_90_percent_interval_coverage_horizon_{horizon}"]
            <= limits[f"central_90_percent_interval_coverage_horizon_{horizon}_max"]
            for horizon in (1, 8, 30)
        )
        return (
            energy_ok
            and coverage_ok
            and values["branch_probability_expected_calibration_error"]
            <= limits["branch_probability_expected_calibration_error_max"]
        )
    raise ValueError(f"unknown registered diagnostic {name!r}")


def validate_confirmatory_auxiliary_evidence(
    output_root: str | Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    validated_results: _ValidatedResultCache | None = None,
) -> dict[str, Any]:
    """Reject claim interpretation without authenticated diagnostics/sensitivity."""

    if not matrix.get("claim_eligible"):
        return {
            "status": "not_required_for_nonclaim_profile",
            "diagnostics": False,
            "active_parameter_width_sensitivity": False,
        }
    root = Path(output_root)
    diagnostic_path = root / "diagnostics" / "diagnostic_summary.json"
    if not diagnostic_path.is_file():
        raise FileNotFoundError(
            "confirmatory interpretation requires diagnostics/diagnostic_summary.json"
        )
    summary = read_json(diagnostic_path)
    required = {
        "schema_version",
        "status",
        "protocol_sha256",
        "source_sha256",
        "matrix_sha256",
        "selection_sha256",
        "diagnostics",
        "diagnostic_summary_sha256",
    }
    if set(summary) != required or (
        summary.get("schema_version") != "matched-objective-diagnostics-v1"
        or summary.get("status") != "complete"
        or summary.get("protocol_sha256") != matrix["protocol_sha256"]
        or summary.get("source_sha256") != matrix["source_sha256"]
        or summary.get("matrix_sha256") != matrix["matrix_sha256"]
        or summary.get("selection_sha256") != matrix["selection_sha256"]
        or summary.get("diagnostic_summary_sha256")
        != object_sha256(_without_digest(summary, "diagnostic_summary_sha256"))
    ):
        raise ValueError("confirmatory diagnostic summary identity or digest is invalid")
    # Import lazily because the standalone diagnostic runner reuses benchmark helpers.
    # This is the authoritative raw-data recomputation gate; the checks below add
    # benchmark-level protocol/category constraints but never trust summary numbers.
    from .matched_objective_diagnostics import (
        validate_diagnostic_summary_from_raw,
    )

    validate_diagnostic_summary_from_raw(
        summary,
        root,
        matrix,
        protocol,
        smoke_nonclaim=False,
    )
    required_diagnostics = set(
        protocol["diagnostics"]["required_before_confirmatory_interpretation"]
    )
    if set(summary["diagnostics"]) != required_diagnostics:
        raise ValueError("confirmatory diagnostic set is incomplete")
    for name in required_diagnostics:
        row = summary["diagnostics"][name]
        spec = protocol["diagnostics"][name]
        if set(row) != {
            "status",
            "registered_spec_sha256",
            "interpretation_only",
            "arms",
        } or (
            row["status"] != "complete"
            or row["registered_spec_sha256"] != object_sha256(spec)
            or row["interpretation_only"] is not True
            or set(row["arms"]) != set(ARM_ORDER)
        ):
            raise ValueError(f"diagnostic {name!r} identity is invalid")
        for arm in ARM_ORDER:
            arm_row = row["arms"][arm]
            if set(arm_row) != {"estimands", "thresholds", "passed", "raw_files"}:
                raise ValueError(f"diagnostic {name!r} arm schema is invalid")
            if set(arm_row["estimands"]) != set(spec["estimands"]):
                raise ValueError(f"diagnostic {name!r} estimands are incomplete")
            if not all(
                math.isfinite(float(value)) for value in arm_row["estimands"].values()
            ):
                raise ValueError(f"diagnostic {name!r} estimands are non-finite")
            if (
                arm_row["thresholds"] != spec["interpretation_thresholds"]
                or not isinstance(arm_row["passed"], bool)
                or arm_row["passed"]
                != _diagnostic_threshold_pass(
                    name, arm_row["estimands"], arm_row["thresholds"]
                )
            ):
                raise ValueError(f"diagnostic {name!r} threshold record differs from protocol")
            files = arm_row["raw_files"]
            if not isinstance(files, list) or not files:
                raise ValueError(f"diagnostic {name!r} has no raw evidence")
            for entry in files:
                if set(entry) != {"path", "size", "sha256"}:
                    raise ValueError("diagnostic raw-file entry is malformed")
                path = _safe_artifact_path(
                    root, entry["path"], required_parent="diagnostics"
                )
                if (
                    path.suffix != ".npz"
                    or not path.is_file()
                    or path.stat().st_size != entry["size"]
                    or file_sha256(path) != entry["sha256"]
                ):
                    raise ValueError("diagnostic raw-file digest mismatch")
                arrays = load_npz(path)
                if not arrays or any(
                    value.size == 0 or not np.isfinite(value).all()
                    for value in arrays.values()
                ):
                    raise ValueError("diagnostic raw-file arrays are empty or non-finite")

    compute_results = [
        _completed_result_for_cell(
            root, matrix, protocol, cell, cache=validated_results
        )[0]
        for cell in matrix_cells(matrix, "compute_plan")
    ]
    sensitivity_required = any(
        result["active_parameter_gap"]["supporting_width_sensitivity_required"]
        for result in compute_results
    )
    if sensitivity_required:
        raise RuntimeError(
            "active-parameter gap exceeds the registered 2% trigger; this source is "
            "claim-ineligible until a matrix-bound width-control runner and verifier "
            "are implemented"
        )
    return {
        "status": "complete",
        "diagnostics": True,
        "active_parameter_width_sensitivity": sensitivity_required,
    }


def _completed_result_for_cell(
    output_root: str | Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
    cell: Mapping[str, Any],
    *,
    cache: _ValidatedResultCache | None = None,
) -> tuple[dict[str, Any], Path]:
    """Load and validate one canonical cell result plus its retained payloads."""

    validation_cache = _require_validation_cache(cache)
    if validation_cache is not None and validation_cache.contains(
        output_root, matrix, protocol, cell
    ):
        return validation_cache.lookup(output_root, matrix, protocol, cell)
    path = _cell_result_path(output_root, cell)
    if not path.is_file():
        raise FileNotFoundError(f"missing completed cell result {cell['cell_id']}")
    directory = stage_directory(output_root, cell)
    # Bind semantic validation to the exact bytes it inspected.  Without this
    # pre/post fingerprint, a mutation between the validator's final read and
    # cache insertion could become the cache's trusted baseline.
    validated_fingerprint = _cell_directory_fingerprint(directory)
    result = read_json(path)
    if result.get("matrix_sha256") != matrix["matrix_sha256"]:
        raise ValueError("cell result belongs to another matrix")
    if cell["stage"] == "dataset":
        validate_dataset_result(
            result,
            cell,
            directory / "dataset.npz",
            protocol=protocol,
            matrix=matrix,
        )
    elif cell["stage"] == "compute_plan":
        validate_compute_plan_cell(
            result,
            protocol,
            matrix,
            cell,
            output_root,
            rederive_compiler_evidence=True,
        )
        _validate_compute_files(result, directory)
    elif cell["stage"] == "world_model":
        validate_world_result(
            result,
            cell,
            directory,
            protocol=protocol,
            matrix=matrix,
            output_root=output_root,
        )
    elif cell["stage"] == "rollout":
        validate_rollout_result(
            result,
            cell,
            directory / "predictive_draws.npz",
            protocol,
            matrix,
            output_root,
            validated_results=validation_cache,
        )
    elif cell["stage"] == "actor":
        validate_actor_result(
            result,
            cell,
            directory,
            protocol=protocol,
            matrix=matrix,
            output_root=output_root,
        )
    else:  # pragma: no cover - matrices are validated before this helper
        raise ValueError(f"unsupported benchmark stage {cell['stage']!r}")
    if validation_cache is not None:
        validation_cache.record(
            output_root,
            matrix,
            protocol,
            cell,
            result,
            path,
            validated_fingerprint=validated_fingerprint,
        )
    return result, path


def _validate_all_cells(
    output_root: str | Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> _ValidatedResultCache:
    """Strongly validate each matrix cell exactly once for one finalize/verify pass."""

    _validate_output_semantic_layout(Path(output_root), matrix, protocol)
    cache = _new_validation_cache(output_root, matrix, protocol)
    for cell in matrix["cells"]:
        _completed_result_for_cell(
            output_root,
            matrix,
            protocol,
            cell,
            cache=cache,
        )
    cache.seal(output_root, matrix, protocol)
    return cache


def _validate_output_semantic_layout(
    output_root: Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    """Reject result-like artifacts that are not members of the frozen matrix.

    Arbitrary cluster metadata remains hash-inventoried, but a forged cell
    directory or a profile-inapplicable HPO artifact cannot be legitimized by
    merely rebuilding the artifact manifest around it.
    """

    root = Path(output_root)
    profile = matrix.get("profile")
    prohibited_root_files = {
        "smoke": {
            "hpo_selection.json",
            "hpo_trials.json",
            "auxiliary_validation.json",
        },
        "confirmatory": {"hpo_trials.json"},
        "pilot": {
            "analysis.json",
            "bootstrap_indices.npz",
            "REPORT.md",
            "artifact_manifest.json",
            "auxiliary_validation.json",
        },
    }
    if profile in prohibited_root_files:
        present = sorted(
            name for name in prohibited_root_files[profile] if (root / name).exists()
        )
        if present:
            raise ValueError(
                f"profile-inapplicable root artifacts are present: {present}"
            )

    cells = matrix.get("cells")
    if not isinstance(cells, list):
        return
    diagnostics_root = root / "diagnostics"
    if profile != "confirmatory":
        if diagnostics_root.exists():
            raise ValueError(
                "profile-inapplicable diagnostics tree is present"
            )
    else:
        expected_diagnostics = {
            path.resolve()
            for path in _expected_diagnostic_artifact_paths(
                root, matrix, protocol
            )
        }
        observed_diagnostics: set[Path] = set()
        if diagnostics_root.is_symlink():
            raise ValueError("diagnostics tree must not be a symlink")
        if diagnostics_root.is_dir():
            for path in diagnostics_root.rglob("*"):
                if path.is_symlink():
                    raise ValueError("diagnostics tree contains a symlink")
                if path.is_file():
                    observed_diagnostics.add(path.resolve())
        if observed_diagnostics != expected_diagnostics:
            raise ValueError(
                "diagnostic artifact paths differ from the registered layout"
            )
    for stage in STAGE_ORDER:
        expected = {
            str(cell["cell_id"])
            for cell in cells
            if cell.get("stage") == stage
        }
        stage_root = root / stage
        observed: set[str] = set()
        unexpected_files: list[str] = []
        if stage_root.exists():
            if stage_root.is_symlink() or not stage_root.is_dir():
                raise ValueError(f"canonical stage root is invalid: {stage}")
            for child in stage_root.iterdir():
                if child.is_symlink() or not child.is_dir():
                    unexpected_files.append(child.name)
                else:
                    observed.add(child.name)
        if observed != expected or unexpected_files:
            raise ValueError(
                f"{stage} stage directories differ from the frozen matrix; "
                f"missing={sorted(expected - observed)}, "
                f"unexpected={sorted(observed - expected)}, "
                f"non_directories={sorted(unexpected_files)}"
            )
        expected_results = {
            (stage_root / cell_id / "result.json").resolve()
            for cell_id in expected
        }
        observed_results = (
            {
                path.resolve()
                for path in stage_root.rglob("result.json")
                if path.is_file() and not path.is_symlink()
            }
            if stage_root.is_dir()
            else set()
        )
        if observed_results != expected_results:
            raise ValueError(
                f"{stage} result.json paths differ from the frozen matrix"
            )


def build_hpo_trial_manifest(
    output_root: str | Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    validated_results: _ValidatedResultCache | None = None,
) -> dict[str, Any]:
    """Materialize all pilot estimands from validated, source-bound cell artifacts."""

    if matrix.get("schema_version") != HPO_MATRIX_SCHEMA:
        raise ValueError("HPO trials require the frozen pilot HPO matrix")
    source = read_json(Path(output_root) / "source_manifest.json")
    validate_pilot_hpo_matrix(matrix, protocol, source)
    profile = protocol["profiles"]["pilot"]
    trials: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        for candidate in pilot_candidates(protocol, arm):
            units: list[dict[str, Any]] = []
            for raw_task in profile["tasks"]:
                task = canonical_task_id(raw_task)
                for world_seed in profile["world_model_seeds"]:
                    worlds = [
                        cell
                        for cell in matrix["cells"]
                        if cell["stage"] == "world_model"
                        and cell["task"] == task
                        and cell["world_model_seed"] == int(world_seed)
                        and cell["arm"] == arm
                        and cell["candidate_id"] == candidate["candidate_id"]
                    ]
                    if len(worlds) != 1:
                        raise ValueError("pilot world-model result is missing or duplicated")
                    world = worlds[0]
                    world_result, world_path = _completed_result_for_cell(
                        output_root,
                        matrix,
                        protocol,
                        world,
                        cache=validated_results,
                    )
                    rollouts = [
                        cell
                        for cell in matrix["cells"]
                        if cell["stage"] == "rollout"
                        and cell["task"] == task
                        and cell["world_model_seed"] == int(world_seed)
                        and cell["arm"] == arm
                        and cell["candidate_id"] == candidate["candidate_id"]
                    ]
                    if len(rollouts) != 1:
                        raise ValueError("pilot rollout result is missing or duplicated")
                    rollout_result, rollout_path = _completed_result_for_cell(
                        output_root,
                        matrix,
                        protocol,
                        rollouts[0],
                        cache=validated_results,
                    )
                    actor_means: list[float] = []
                    actor_hashes: list[str] = []
                    for actor_seed in profile["actor_seeds_nested_within_world_model_seed"]:
                        actors = [
                            cell
                            for cell in matrix["cells"]
                            if cell["stage"] == "actor"
                            and cell["task"] == task
                            and cell["world_model_seed"] == int(world_seed)
                            and cell["arm"] == arm
                            and cell["actor_seed"] == int(actor_seed)
                            and cell["candidate_id"] == candidate["candidate_id"]
                        ]
                        if len(actors) != 1:
                            raise ValueError("pilot nested actor result is missing or duplicated")
                        actor_result, actor_path = _completed_result_for_cell(
                            output_root,
                            matrix,
                            protocol,
                            actors[0],
                            cache=validated_results,
                        )
                        actor_means.append(float(actor_result["normalized_episode_return_mean"]))
                        actor_hashes.append(file_sha256(actor_path))
                    compute = _dependency_cell(matrix, world, "compute_plan")
                    compute_result, _ = _completed_result_for_cell(
                        output_root,
                        matrix,
                        protocol,
                        compute,
                        cache=validated_results,
                    )
                    units.append(
                        {
                            "task": task,
                            "world_model_seed": int(world_seed),
                            "rollout_auc": float(
                                rollout_result["normalized_free_running_rollout_error_auc"]
                            ),
                            "nested_actor_return": float(np.mean(actor_means)),
                            "full_forward_backward_train_flops_per_update": float(
                                compute_result["tracks"]["equal_updates"]["allocations"][arm][
                                    "world_model"
                                ]["flops_per_update"]
                            ),
                            "artifact_sha256s": [
                                file_sha256(world_path),
                                file_sha256(rollout_path),
                                *actor_hashes,
                            ],
                        }
                    )
            trials.append(
                {
                    "arm": arm,
                    "candidate_id": candidate["candidate_id"],
                    "overrides": candidate["overrides"],
                    "units": units,
                }
            )
    manifest = {
        "schema_version": "matched-objective-hpo-trials-v1",
        "status": "complete",
        "protocol_sha256": matrix["protocol_sha256"],
        "source_sha256": matrix["source_sha256"],
        "selection_profile": "pilot",
        "selection_budget_track": "equal_updates",
        "confirmatory_outcomes_accessed": False,
        "trials": trials,
    }
    manifest["trial_manifest_sha256"] = object_sha256(manifest)
    validate_hpo_trial_manifest(
        manifest, protocol, source_sha256=matrix["source_sha256"]
    )
    return manifest


def _authenticate_frozen_arguments(
    root: Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    workspace: str | Path | None,
) -> dict[str, Any]:
    """Bind library finalizers to the immutable protocol/matrix stored at the root."""

    stored_protocol = read_json(root / "frozen_protocol.json")
    if canonical_bytes(stored_protocol) != canonical_bytes(protocol):
        raise ValueError("supplied protocol differs from the output root's frozen protocol")
    validate_matched_objective_protocol(stored_protocol)
    source = read_json(root / "source_manifest.json")
    validate_source_manifest(source, workspace)
    stored_matrix = read_json(root / "matrix.json")
    if canonical_bytes(stored_matrix) != canonical_bytes(matrix):
        raise ValueError("supplied matrix differs from the output root's frozen matrix")
    if matrix.get("schema_version") == HPO_MATRIX_SCHEMA:
        validate_pilot_hpo_matrix(matrix, protocol, source)
    else:
        validate_matrix(matrix, protocol, source)
    return source


def finalize_pilot_hpo_run(
    output_root: str | Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze deterministic pilot rankings and the selected config per arm."""

    root = Path(output_root)
    source = _authenticate_frozen_arguments(
        root, matrix, protocol, workspace=workspace
    )
    validate_frozen_runtime_provenance(
        root, source, strict=True, matrix=matrix
    )
    validated_results = _validate_all_cells(root, matrix, protocol)
    pairing = validate_pairing_invariants(
        output_root,
        matrix,
        protocol,
        validated_results=validated_results,
    )
    write_json_atomic(Path(output_root) / "pairing_validation.json", pairing)
    trials = build_hpo_trial_manifest(
        output_root,
        matrix,
        protocol,
        validated_results=validated_results,
    )
    selection = select_hpo_candidates(
        trials, protocol, source_sha256=matrix["source_sha256"]
    )
    for filename, payload in (
        ("hpo_trials.json", trials),
        ("hpo_selection.json", selection),
    ):
        path = root / filename
        if path.is_file() and read_json(path) != payload:
            raise ValueError(f"existing pilot artifact differs: {filename}")
        write_json_atomic(path, payload)
    verified = _verify_pilot_hpo_output_root(
        root,
        workspace=workspace,
        _validation_cache=validated_results,
    )
    if verified["selection_sha256"] != selection["selection_sha256"]:
        raise ValueError("pilot finalization did not verify its selected configuration")
    return selection


def verify_pilot_hpo_output_root(
    output_root: str | Path,
    *,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless every pilot cell and the deterministic selection validate."""

    return _verify_pilot_hpo_output_root(
        output_root, workspace=workspace, _validation_cache=None
    )


def _verify_pilot_hpo_output_root(
    output_root: str | Path,
    *,
    workspace: str | Path | None,
    _validation_cache: _ValidatedResultCache | None,
) -> dict[str, Any]:
    """Internal pilot verifier that may reuse this pass's sealed validation cache."""

    root = Path(output_root)
    protocol = read_json(root / "frozen_protocol.json")
    validate_matched_objective_protocol(protocol)
    source = read_json(root / "source_manifest.json")
    validate_source_manifest(source, workspace)
    matrix = read_json(root / "matrix.json")
    validate_pilot_hpo_matrix(matrix, protocol, source)
    validated_results = (
        _validate_all_cells(root, matrix, protocol)
        if _validation_cache is None
        else _require_validation_cache(_validation_cache)
    )
    if validated_results is None:  # pragma: no cover - guarded by construction above
        raise RuntimeError("pilot verifier did not construct a validation cache")
    validated_results.require_complete(root, matrix, protocol)
    validate_frozen_runtime_provenance(root, source, strict=True, matrix=matrix)
    auxiliary = validate_confirmatory_auxiliary_evidence(
        root,
        matrix,
        protocol,
        validated_results=validated_results,
    )
    auxiliary_path = root / "auxiliary_validation.json"
    if matrix["claim_eligible"]:
        if not auxiliary_path.is_file() or read_json(auxiliary_path) != auxiliary:
            raise ValueError("stored confirmatory auxiliary-evidence gate does not recompute")
    pairing = validate_pairing_invariants(
        root,
        matrix,
        protocol,
        validated_results=validated_results,
    )
    if read_json(root / "pairing_validation.json") != pairing:
        raise ValueError("stored pilot paired-randomness evidence does not recompute")
    expected_trials = build_hpo_trial_manifest(
        root,
        matrix,
        protocol,
        validated_results=validated_results,
    )
    stored_trials = read_json(root / "hpo_trials.json")
    if stored_trials != expected_trials:
        raise ValueError("stored pilot trial manifest differs from validated results")
    selection = read_json(root / "hpo_selection.json")
    validate_hpo_selection_manifest(
        selection, protocol, source_sha256=matrix["source_sha256"]
    )
    expected_selection = select_hpo_candidates(
        expected_trials, protocol, source_sha256=matrix["source_sha256"]
    )
    if selection != expected_selection:
        raise ValueError("stored pilot selection differs from deterministic ranking")
    return {
        "profile": "pilot",
        "cells": len(matrix["cells"]),
        "protocol_sha256": matrix["protocol_sha256"],
        "source_sha256": matrix["source_sha256"],
        "matrix_sha256": matrix["matrix_sha256"],
        "trial_manifest_sha256": stored_trials["trial_manifest_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "selected": selection["selected"],
    }


def _expected_diagnostic_artifact_paths(
    root: Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> list[Path]:
    if matrix.get("profile") != "confirmatory" or not isinstance(
        matrix.get("cells"), list
    ):
        return []
    diagnostics = protocol.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or not diagnostics:
        raise ValueError("confirmatory protocol has no registered diagnostics")
    registered = diagnostics.get("required_before_confirmatory_interpretation")
    supported = {
        "deterministic_transition_diagnostic",
        "genuinely_stochastic_transition_diagnostic",
    }
    if (
        not isinstance(registered, list)
        or len(registered) != len(set(registered))
        or set(registered) != supported
    ):
        raise ValueError("confirmatory diagnostic registry is unsupported or incomplete")
    paths = [root / "diagnostics" / "diagnostic_summary.json"]
    for diagnostic_name in sorted(registered):
        diagnostic_root = root / "diagnostics" / str(diagnostic_name)
        paths.extend(
            (
                diagnostic_root / "training_dataset.npz",
                diagnostic_root / "minibatch_schedule.npz",
            )
        )
        for arm in ARM_ORDER:
            arm_root = diagnostic_root / arm
            paths.extend(
                (
                    arm_root / "checkpoint.pkl",
                    arm_root / "training_manifest.json",
                    arm_root / "result_manifest.json",
                    arm_root / "evaluation_raw.npz",
                )
            )
    return paths


def _artifact_category_files(
    root: Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, list[Path]]:
    all_results = [
        _cell_result_path(root, cell)
        for cell in matrix.get("cells", [])
        if cell.get("stage") in STAGE_ORDER
    ]
    return {
        "frozen_protocol": [root / "frozen_protocol.json"],
        "source_commit_and_dirty_patch": [root / "source_manifest.json", root / "source_identity.txt"],
        "dependency_lock": [root / "dependency_lock.txt"],
        "hardware_and_software_environment": [root / "environment.json"],
        "dataset_payload_and_digest": list(root.glob("dataset/*/dataset.npz")) + list(root.glob("dataset/*/result.json")),
        "episode_split_indices": list(root.glob("dataset/*/dataset.npz")),
        "minibatch_indices": list(root.glob("world_model/*/batch_schedule.npz")) + list(root.glob("actor/*/batch_schedule.npz")) + [root / "pairing_validation.json"],
        "sampled_objective_noise_times_and_step_sizes": list(root.glob("world_model/*/objective_rng_keys.npz")),
        "evaluation_start_indices": list(root.glob("rollout/*/predictive_draws.npz")),
        "evaluation_episode_seeds": list(root.glob("actor/*/result.json")),
        "compiler_ir_and_cost_analysis": list(root.glob("compute_plan/*/result.json")) + list(root.glob("compute_plan/*/hlo_*.txt")) + list(root.glob("compute_plan/*/compiler_cost_analysis.json")),
        "world_model_and_actor_checkpoints": list(root.glob("world_model/*/checkpoint.pkl")) + list(root.glob("actor/*/checkpoint.pkl")),
        "parameter_tree_hashes_and_counts": list(root.glob("world_model/*/result.json")) + list(root.glob("compute_plan/*/result.json")),
        "hpo_trial_metrics_and_selection_manifest": [root / "hpo_selection.json"],
        "resolved_per_arm_DreamerConfig_and_sha256": list(root.glob("world_model/*/result.json")) + list(root.glob("compute_plan/*/result.json")),
        "unoptimized_and_optimized_HLO_with_sha256": list(root.glob("compute_plan/*/hlo_*.txt")),
        "full_update_compute_and_synchronized_walltime_records": list(root.glob("compute_plan/*/result.json")) + all_results,
        "diagnostic_raw_samples_estimands_and_threshold_interpretations": _expected_diagnostic_artifact_paths(
            root, matrix, protocol
        ),
        "raw_predictive_draws": list(root.glob("rollout/*/predictive_draws.npz")),
        "raw_action_traces": list(root.glob("actor/*/action_traces.npz")),
        "raw_per_seed_primary_and_secondary_metrics": all_results,
        "bootstrap_draw_indices_and_intervals": [root / "bootstrap_indices.npz", root / "analysis.json"],
        "machine_readable_summary_and_human_report": [root / "analysis.json", root / "REPORT.md"],
    }


def _artifact_identity(matrix: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact matrix-bound identity of an artifact manifest."""

    return {
        "schema_version": ARTIFACT_SCHEMA,
        "status": "complete",
        "profile": matrix["profile"],
        "evidence_class": matrix["evidence_class"],
        "claim_eligible": matrix["claim_eligible"],
        "protocol_sha256": matrix["protocol_sha256"],
        "source_sha256": matrix["source_sha256"],
        "matrix_sha256": matrix["matrix_sha256"],
    }


def _safe_artifact_relative_path(value: Any) -> PurePosixPath:
    """Parse one canonical relative artifact path without filesystem access."""

    if not isinstance(value, str) or not value:
        raise ValueError("artifact path must be a nonempty relative POSIX path")
    relative = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        value == "."
        or value.endswith("/")
        or relative.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or "\\" in value
        or relative.as_posix() != value
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise ValueError(f"artifact path is not canonical and relative: {value!r}")
    return relative


def _resolved_artifact_file(root: Path, relative: PurePosixPath) -> Path:
    """Resolve a retained file and reject symlinks or root escapes."""

    lexical = root.joinpath(*relative.parts)
    if lexical.is_symlink():
        raise ValueError(f"artifact path must not be a symlink: {relative.as_posix()}")
    try:
        resolved = lexical.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"artifact file is absent: {relative.as_posix()}") from error
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(
            f"artifact path escapes the output root: {relative.as_posix()}"
        ) from error
    if not resolved.is_file():
        raise ValueError(f"artifact path is not a file: {relative.as_posix()}")
    return resolved


def _retained_artifact_paths(root: Path) -> list[Path]:
    """Enumerate exactly the regular files retained below an output root."""

    if not root.is_dir():
        raise FileNotFoundError(f"artifact output root is absent: {root}")
    root_resolved = root.resolve(strict=True)
    excluded_root_paths = {
        root_resolved / "artifact_manifest.json",
        # Written only after core finalization succeeds and authenticated by
        # the cluster verifier.  Excluding this single post-seal marker avoids
        # making a successful cluster run unverifiable on its next core pass.
        root_resolved / "cluster_profile_verified.json",
    }
    retained: list[Path] = []
    for candidate in root_resolved.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(
                "artifact output root contains a symlink: "
                f"{candidate.relative_to(root_resolved).as_posix()}"
            )
        if candidate in excluded_root_paths:
            continue
        if candidate.is_file():
            retained.append(candidate)
    return sorted(
        retained, key=lambda path: path.relative_to(root_resolved).as_posix()
    )


def _artifact_file_entries(root: Path) -> list[dict[str, Any]]:
    root_resolved = root.resolve(strict=True)
    return [
        {
            "path": path.relative_to(root_resolved).as_posix(),
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in _retained_artifact_paths(root_resolved)
    ]


def _artifact_categories(
    root: Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    allowed_placeholders = (
        {
            "hpo_trial_metrics_and_selection_manifest",
            "diagnostic_raw_samples_estimands_and_threshold_interpretations",
        }
        if matrix["profile"] == "smoke"
        else set()
    )
    available = _artifact_category_files(root, matrix, protocol)
    categories: dict[str, Any] = {}
    for category in protocol["provenance"]["required_artifacts"]:
        if category not in available:
            raise ValueError(f"protocol names an unknown artifact category: {category}")
        files = sorted(
            {
                path.relative_to(root).as_posix()
                for path in available[category]
                if path.is_file()
            }
        )
        if files:
            categories[category] = {"status": "complete", "files": files}
        elif category in allowed_placeholders:
            categories[category] = {
                "status": "not_applicable_to_engineering_smoke",
                "files": [],
            }
        else:
            raise FileNotFoundError(f"required artifact category is empty: {category}")
    return categories


def _validate_confirmatory_hpo_selection(
    root: Path, matrix: Mapping[str, Any]
) -> None:
    if matrix.get("profile") != "confirmatory":
        return
    selection = matrix.get("selection_manifest")
    if not isinstance(selection, Mapping):
        raise ValueError("confirmatory matrix has no HPO selection manifest")
    selection_path = root / "hpo_selection.json"
    if (
        not selection_path.is_file()
        or canonical_bytes(read_json(selection_path)) != canonical_bytes(selection)
    ):
        raise ValueError(
            "confirmatory hpo_selection.json does not match matrix selection_manifest"
        )


def build_artifact_manifest(
    output_root: str | Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(output_root)
    _validate_output_semantic_layout(root, matrix, protocol)
    _validate_confirmatory_hpo_selection(root, matrix)
    manifest = {
        **_artifact_identity(matrix),
        "categories": _artifact_categories(root, matrix, protocol),
        "files": _artifact_file_entries(root),
    }
    manifest["artifact_manifest_sha256"] = object_sha256(manifest)
    return manifest


def validate_artifact_manifest(
    manifest: Mapping[str, Any],
    output_root: str | Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    root = Path(output_root)
    _validate_output_semantic_layout(root, matrix, protocol)
    required_keys = {
        *_artifact_identity(matrix),
        "categories",
        "files",
        "artifact_manifest_sha256",
    }
    if set(manifest) != required_keys:
        raise ValueError("artifact manifest keys are incomplete or contain extras")
    if manifest.get("artifact_manifest_sha256") != object_sha256(
        _without_digest(manifest, "artifact_manifest_sha256")
    ):
        raise ValueError("artifact manifest digest mismatch")
    observed_identity = {key: manifest[key] for key in _artifact_identity(matrix)}
    if canonical_bytes(observed_identity) != canonical_bytes(_artifact_identity(matrix)):
        raise ValueError("artifact manifest identity does not match its frozen matrix")
    if not isinstance(manifest["files"], list):
        raise ValueError("artifact manifest files must be a list")
    for entry in manifest["files"]:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "size", "sha256"}:
            raise ValueError("artifact file entry keys are invalid")
        relative = _safe_artifact_relative_path(entry["path"])
        _resolved_artifact_file(root, relative)
    listed_paths = [entry["path"] for entry in manifest["files"]]
    if len(set(listed_paths)) != len(listed_paths):
        raise ValueError("artifact manifest contains duplicate file paths")
    if manifest["files"] != _artifact_file_entries(root):
        raise ValueError(
            "artifact manifest files do not exactly match the output-root file set"
        )
    if not isinstance(manifest["categories"], Mapping):
        raise ValueError("artifact manifest categories must be an object")
    for category in manifest["categories"].values():
        if not isinstance(category, Mapping) or set(category) != {"status", "files"}:
            raise ValueError("artifact category entry keys are invalid")
        if not isinstance(category["files"], list):
            raise ValueError("artifact category files must be a list")
        for relative in category["files"]:
            _safe_artifact_relative_path(relative)
    expected_categories = _artifact_categories(root, matrix, protocol)
    if canonical_bytes(manifest["categories"]) != canonical_bytes(expected_categories):
        raise ValueError("artifact category assignments do not recompute")
    _validate_confirmatory_hpo_selection(root, matrix)


def _verified_run_summary(
    matrix: Mapping[str, Any], analysis: Mapping[str, Any]
) -> dict[str, Any]:
    """Expose both statistical and practical claim gates to automation."""

    practical = analysis["practical_significance"]
    return {
        "profile": matrix["profile"],
        "cells": len(matrix["cells"]),
        "protocol_sha256": matrix["protocol_sha256"],
        "source_sha256": matrix["source_sha256"],
        "matrix_sha256": matrix["matrix_sha256"],
        "analysis_sha256": analysis["analysis_sha256"],
        "superiority": analysis["superiority"],
        "practical_significance": practical,
        "practically_meaningful_superiority_claim_allowed": bool(
            practical["practically_meaningful_superiority_claim_allowed"]
        ),
    }


def verify_output_root(
    output_root: str | Path,
    *,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless every expected result and retained payload validates."""

    return _verify_output_root(
        output_root, workspace=workspace, _validation_cache=None
    )


def _verify_output_root(
    output_root: str | Path,
    *,
    workspace: str | Path | None,
    _validation_cache: _ValidatedResultCache | None,
) -> dict[str, Any]:
    """Internal verifier that may reuse this pass's sealed validation cache."""

    root = Path(output_root)
    protocol = read_json(root / "frozen_protocol.json")
    validate_matched_objective_protocol(protocol)
    source = read_json(root / "source_manifest.json")
    validate_source_manifest(source, workspace)
    matrix = read_json(root / "matrix.json")
    if matrix.get("schema_version") == HPO_MATRIX_SCHEMA:
        return _verify_pilot_hpo_output_root(
            root,
            workspace=workspace,
            _validation_cache=_validation_cache,
        )
    validate_matrix(matrix, protocol, source)
    validated_results = (
        _validate_all_cells(root, matrix, protocol)
        if _validation_cache is None
        else _require_validation_cache(_validation_cache)
    )
    if validated_results is None:  # pragma: no cover - guarded by construction above
        raise RuntimeError("verifier did not construct a validation cache")
    validated_results.require_complete(root, matrix, protocol)
    validate_frozen_runtime_provenance(
        root, source, strict=bool(matrix["claim_eligible"]), matrix=matrix
    )
    auxiliary = validate_confirmatory_auxiliary_evidence(
        root,
        matrix,
        protocol,
        validated_results=validated_results,
    )
    if matrix["claim_eligible"]:
        auxiliary_path = root / "auxiliary_validation.json"
        if not auxiliary_path.is_file() or read_json(auxiliary_path) != auxiliary:
            raise ValueError(
                "stored confirmatory auxiliary-evidence gate does not recompute"
            )
    pairing = validate_pairing_invariants(
        root,
        matrix,
        protocol,
        validated_results=validated_results,
    )
    if read_json(root / "pairing_validation.json") != pairing:
        raise ValueError("stored paired-randomness evidence does not recompute")
    analysis = read_json(root / "analysis.json")
    validate_analysis_artifacts(
        analysis,
        matrix,
        protocol,
        root,
        validated_results=validated_results,
    )
    manifest = read_json(root / "artifact_manifest.json")
    validate_artifact_manifest(manifest, root, matrix, protocol)
    return _verified_run_summary(matrix, analysis)


def finalize_run(
    output_root: str | Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    source = _authenticate_frozen_arguments(
        root, matrix, protocol, workspace=workspace
    )
    validate_frozen_runtime_provenance(
        root, source, strict=bool(matrix["claim_eligible"]), matrix=matrix
    )
    validated_results = _validate_all_cells(root, matrix, protocol)
    auxiliary = validate_confirmatory_auxiliary_evidence(
        output_root,
        matrix,
        protocol,
        validated_results=validated_results,
    )
    if matrix["claim_eligible"]:
        write_json_atomic(root / "auxiliary_validation.json", auxiliary)
    pairing = validate_pairing_invariants(
        output_root,
        matrix,
        protocol,
        validated_results=validated_results,
    )
    write_json_atomic(Path(output_root) / "pairing_validation.json", pairing)
    analysis = run_analysis(
        output_root,
        matrix,
        protocol,
        validated_results=validated_results,
    )
    validate_analysis_artifacts(
        analysis,
        matrix,
        protocol,
        output_root,
        validated_results=validated_results,
    )
    manifest = build_artifact_manifest(output_root, matrix, protocol)
    write_json_atomic(Path(output_root) / "artifact_manifest.json", manifest)
    validate_artifact_manifest(manifest, output_root, matrix, protocol)
    verified = _verify_output_root(
        root,
        workspace=workspace,
        _validation_cache=validated_results,
    )
    if verified["analysis_sha256"] != analysis["analysis_sha256"]:
        raise ValueError("finalization did not verify the analysis it just wrote")
    return analysis


__all__ = [
    "ACTOR_SCHEMA",
    "ANALYSIS_SCHEMA",
    "ARTIFACT_SCHEMA",
    "COMPUTE_SCHEMA",
    "DATASET_SCHEMA",
    "EXECUTION_SPEC",
    "MATRIX_SCHEMA",
    "WORLD_SCHEMA",
    "allocate_equal_flops",
    "analyze_units",
    "build_artifact_manifest",
    "build_hpo_trial_manifest",
    "build_matrix",
    "build_pilot_hpo_matrix",
    "build_source_manifest",
    "collect_verified_units",
    "expected_matrix_counts",
    "expected_hpo_matrix_counts",
    "finalize_run",
    "finalize_pilot_hpo_run",
    "freeze_run",
    "freeze_pilot_hpo_run",
    "interquartile_mean",
    "make_config",
    "normalized_rollout_statistics",
    "posterior_mean_filter",
    "pilot_candidates",
    "run_cell",
    "run_pending_cells",
    "select_hpo_candidates",
    "synthetic_units",
    "task_stratified_bootstrap",
    "validate_artifact_manifest",
    "validate_hpo_selection_manifest",
    "validate_hpo_trial_manifest",
    "validate_matrix",
    "validate_pilot_hpo_matrix",
    "validate_source_manifest",
    "verify_pilot_hpo_output_root",
    "verify_output_root",
]
