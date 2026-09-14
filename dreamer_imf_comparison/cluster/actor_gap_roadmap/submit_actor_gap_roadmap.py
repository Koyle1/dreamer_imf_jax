#!/usr/bin/env python3
"""Canonical fail-closed Slurm submitter for the actor-gap roadmap study."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT.parent
for path in (PROJECT, WORKSPACE / "imf_dreamer_jax" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dreamer_imf_compare import actor_gap_roadmap_study as study  # noqa: E402

STAGES = (
    "preflight",
    "calibration",
    "diagnostics",
    "models",
    "evaluations",
    "finalize",
)
ARRAY_STAGES = {
    "diagnostics": ("diagnostic", "diagnostic-cell", "verify-diagnostics", 3),
    "models": ("model", "model-cell", "verify-models", 4),
    "evaluations": ("evaluation", "evaluation-cell", "verify-evaluations", 4),
}
SUBMISSION_RECORD_SCHEMA = "trajectory-imf-actor-gap-slurm-submission-record-v1"
RELEASE_RECORD_SCHEMA = "trajectory-imf-actor-gap-slurm-release-record-v1"
CANCELLATION_RECORD_SCHEMA = "trajectory-imf-actor-gap-slurm-cancellation-record-v1"
COMPLETION_ATTESTATION_SCHEMA = (
    "trajectory-imf-actor-gap-scheduler-completion-attestation-v1"
)
COMPLETION_RECOVERY = {
    "preflight": ("preflight-verifier", "verify-preflight"),
    "calibration": ("calibration-verifier", "verify-calibration"),
    "diagnostic": ("diagnostic-verifier", "verify-diagnostics"),
    "model": ("model-verifier", "verify-models"),
    "evaluation": ("evaluation-verifier", "verify-evaluations"),
    "final": ("final-verifier", "verify-final"),
}
ACTIVE_SLURM_STATES = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "REQUEUED",
    "RESIZING",
    "RUNNING",
    "SUSPENDED",
}
TERMINAL_SLURM_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}


class SlurmAccountingAbsent(RuntimeError):
    """The controller and accounting database no longer know a receipt job."""


def _git(source_root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source_root), *arguments], text=True
    ).strip()


def _validate_source(source_root: Path, source_commit: str) -> None:
    if _git(source_root, "rev-parse", "HEAD") != source_commit:
        raise ValueError("source checkout does not match --source-commit")
    if _git(source_root, "status", "--porcelain"):
        raise ValueError("source checkout is not clean")
    expected = (source_root / "dreamer_imf_comparison").resolve(strict=True)
    if expected != PROJECT.resolve(strict=True):
        raise ValueError("submitter must execute from the exact --source-root checkout")


@contextmanager
def _stage_submission_lock(root: Path, label: str):
    """Serialize check-to-release for one stage without stale crash locks."""

    directory = root / "submissions"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f".{label}.submission.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another canonical submitter owns the {label} stage lock"
            ) from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _array_spec(indices: Sequence[int], concurrency: int) -> str:
    ordered = sorted(set(int(value) for value in indices))
    if not ordered or concurrency <= 0:
        raise ValueError("array indices and concurrency must be positive")
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return f"{','.join(ranges)}%{concurrency}"


def _sbatch(
    sbatch_path: Path,
    exports: Mapping[str, str],
    *,
    array: str | None = None,
    dependency_job: str | None = None,
    hold: bool = False,
) -> str:
    for key, value in exports.items():
        if any(character in value for character in (",", "\n", "\r")):
            raise ValueError(f"unsafe Slurm export value for {key}")
    command = [
        "sbatch",
        "--parsable",
        "--export=ALL," + ",".join(f"{key}={value}" for key, value in exports.items()),
    ]
    if array is not None:
        command.append(f"--array={array}")
    if hold:
        command.append("--hold")
    if dependency_job is not None:
        command.append(f"--dependency=afterok:{dependency_job}")
        # A failed array must terminalize its verifier instead of leaving a
        # DependencyNeverSatisfied job pending forever and blocking audited
        # missing-index retries.
        command.append("--kill-on-invalid-dep=yes")
    command.append(str(sbatch_path))
    try:
        output = subprocess.check_output(command, text=True).strip()
    except subprocess.CalledProcessError as error:
        # Slurm can accept a job even when the client loses a clean reply.  An
        # exact parsable stdout is sufficient to reconcile that accepted job;
        # every other non-zero outcome remains ambiguous and fails closed.
        recovered = _parse_sbatch_job_id(error.output)
        if recovered is not None:
            return recovered
        raise
    job_id = _parse_sbatch_job_id(output)
    if job_id is None:
        raise RuntimeError(f"sbatch returned an invalid job id: {output!r}")
    return job_id


def _parse_sbatch_job_id(output: str | bytes | None) -> str | None:
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    # This protocol intentionally targets one cluster.  Silently discarding the
    # `;cluster` suffix from federated --parsable output would make every later
    # squeue/sacct/scontrol query ambiguous.
    match = re.fullmatch(r"\s*([0-9]+)\s*", output or "")
    return match.group(1) if match is not None else None


def _record_digest(payload: Mapping[str, Any], field: str) -> str:
    unsigned = json.loads(json.dumps(payload))
    unsigned.pop(field, None)
    return study.benchmark.object_sha256(unsigned)


def _squeue_output(job_id: str, output_format: str) -> str:
    """Return queued rows, treating only Slurm's absent-job reply as empty."""

    try:
        return subprocess.check_output(
            ["squeue", "-h", "-j", job_id, "-o", output_format],
            text=True,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        message = "\n".join(
            part
            for part in (_bounded_text(error.stdout), _bounded_text(error.stderr))
            if part
        ).lower()
        if "invalid job id" in message or "invalid job specification" in message:
            return ""
        raise


def _accounting_observation(job_id: str) -> dict[str, Any]:
    accounting = subprocess.check_output(
        [
            "sacct",
            "-n",
            "-X",
            "--array",
            "-j",
            job_id,
            "--format=JobIDRaw,JobID,State,ExitCode",
            "--parsable2",
        ],
        text=True,
    )
    records: list[dict[str, str]] = []
    for line in accounting.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 4:
            continue
        display_id = fields[1].strip()
        if display_id != job_id and not display_id.startswith(job_id + "_"):
            continue
        state_fields = fields[2].strip().split()
        if not state_fields:
            raise RuntimeError(f"Slurm accounting state for job {job_id} is empty")
        records.append(
            {
                "job_id_raw": fields[0].strip(),
                "job_id": display_id,
                "state": state_fields[0].split("+")[0],
                "exit_code": fields[3].strip(),
            }
        )
    if not records:
        raise SlurmAccountingAbsent(f"Slurm accounting for job {job_id} is absent")
    if any(record["state"] not in TERMINAL_SLURM_STATES for record in records):
        state = "RUNNING"
    elif all(
        record["state"] == "COMPLETED" and record["exit_code"].startswith("0:")
        for record in records
    ):
        state = "COMPLETED"
    else:
        state = "FAILED"
    return {
        "job_id": job_id,
        "state": state,
        "records": records,
        "raw_output": accounting,
        "raw_output_sha256": study.benchmark.object_sha256(accounting),
    }


def _validate_accounting_observation(
    observation: Mapping[str, Any], job_id: str, *, require_completed: bool = False
) -> dict[str, Any]:
    canonical = dict(observation)
    raw = canonical.get("raw_output")
    if (
        set(canonical)
        != {"job_id", "state", "records", "raw_output", "raw_output_sha256"}
        or canonical.get("job_id") != job_id
        or not isinstance(raw, str)
        or canonical.get("raw_output_sha256") != study.benchmark.object_sha256(raw)
    ):
        raise ValueError("retained Slurm accounting observation is invalid")
    # Reparse the retained scheduler response instead of trusting its summary.
    reparsed_records: list[dict[str, str]] = []
    for line in raw.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 4:
            continue
        display_id = fields[1].strip()
        if display_id != job_id and not display_id.startswith(job_id + "_"):
            continue
        state_fields = fields[2].strip().split()
        if not state_fields:
            raise ValueError("retained Slurm accounting state is empty")
        reparsed_records.append(
            {
                "job_id_raw": fields[0].strip(),
                "job_id": display_id,
                "state": state_fields[0].split("+")[0],
                "exit_code": fields[3].strip(),
            }
        )
    if not reparsed_records or canonical.get("records") != reparsed_records:
        raise ValueError("retained Slurm accounting records differ")
    if any(record["state"] not in TERMINAL_SLURM_STATES for record in reparsed_records):
        expected_state = "RUNNING"
    elif all(
        record["state"] == "COMPLETED" and record["exit_code"].startswith("0:")
        for record in reparsed_records
    ):
        expected_state = "COMPLETED"
    else:
        expected_state = "FAILED"
    if canonical.get("state") != expected_state or (
        require_completed and expected_state != "COMPLETED"
    ):
        raise ValueError("retained Slurm accounting outcome differs")
    return canonical


def _slurm_observation(job_id: str) -> dict[str, Any]:
    queued = _squeue_output(job_id, "%T").strip()
    if queued:
        # Array tasks can legitimately be a mixture of PENDING and RUNNING.
        # Any squeue record means that the logical job is still active.
        return {"job_id": job_id, "state": "RUNNING", "queued_output": queued}
    return _accounting_observation(job_id)


def _slurm_state(job_id: str) -> str:
    return str(_slurm_observation(job_id)["state"])


def _marker_scheduler(marker: Mapping[str, Any]) -> Mapping[str, Any]:
    scheduler = marker.get("scheduler_provenance")
    if not isinstance(scheduler, Mapping):
        scheduler = marker.get("verification_scheduler_provenance")
    if not isinstance(scheduler, Mapping):
        raise ValueError("upstream marker lacks verifier-job provenance")
    return scheduler


def _marker_receipt(marker: Mapping[str, Any]) -> Mapping[str, Any]:
    receipt = marker.get("submission_receipt")
    if not isinstance(receipt, Mapping):
        receipt = marker.get("verification_submission_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("upstream marker lacks verifier submission receipt")
    return receipt


def _receipt_completion_binding(
    root: Path,
    receipt_path: Path,
    job_id: str,
    *,
    retained: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_root = root.resolve(strict=True)
    resolved_submissions = (resolved_root / "submissions").resolve(strict=True)
    resolved_receipt = receipt_path.resolve(strict=True)
    if not resolved_receipt.is_relative_to(resolved_submissions):
        raise ValueError("marker completion receipt is outside the output root")
    receipt_path = resolved_receipt
    receipt = study.read_json(receipt_path)
    file_sha256 = study.benchmark.file_sha256(receipt_path)
    jobs = receipt.get("jobs", {})
    if set(jobs) == {"job"}:
        job_binding_valid = str(jobs["job"]) == job_id
        dependency_valid = receipt.get("dependency_policies") == {}
    elif set(jobs) == {"array", "verifier"}:
        job_binding_valid = str(jobs["verifier"]) == job_id
        dependency_valid = receipt.get("dependency_policies") == {
            "verifier": "afterok_array_kill_on_invalid_dependency"
        }
    else:
        job_binding_valid = False
        dependency_valid = False
    relative_intent = str(receipt.get("intent_path", ""))
    intent_path = (resolved_root / relative_intent).resolve(strict=True)
    if not intent_path.is_relative_to(resolved_submissions):
        raise ValueError("marker completion intent is outside the output root")
    intent = study.read_json(intent_path)
    expected_receipt = _receipt_path(intent_path)
    manifest_path = resolved_root / "manifest.json"
    manifest = study.read_json(manifest_path) if manifest_path.is_file() else None
    if intent_path.name.endswith(".intent.json"):
        intent_name = re.fullmatch(
            rf"{re.escape(str(receipt.get('label', '')))}-attempt-([0-9]+)\.intent\.json",
            intent_path.name,
        )
        manifest_binding_valid = (
            intent.get("manifest_sha256")
            in {
                None,
                None if manifest is None else manifest.get("manifest_sha256"),
            }
            if intent.get("label") == "preflight"
            else manifest is not None
            and intent.get("manifest_sha256") == manifest.get("manifest_sha256")
        )
        intent_valid = (
            set(intent)
            == {
                "schema_version",
                "status",
                "label",
                "attempt",
                "source_commit",
                "output_root",
                "manifest_sha256",
                "payload",
                "intent_sha256",
            }
            and intent.get("schema_version") == SUBMISSION_RECORD_SCHEMA
            and intent.get("status") == "frozen_before_submission"
            and intent.get("label") == receipt.get("label")
            and intent.get("source_commit") == receipt.get("source_commit")
            and (
                manifest is None
                or intent.get("source_commit") == manifest.get("source_commit")
            )
            and intent.get("output_root") == str(resolved_root)
            and intent_name is not None
            and isinstance(intent.get("attempt"), int)
            and not isinstance(intent.get("attempt"), bool)
            and int(intent["attempt"]) > 0
            and int(intent_name.group(1)) == int(intent["attempt"])
            and intent_name.group(1) == f"{int(intent['attempt']):03d}"
            and isinstance(intent.get("payload"), Mapping)
            and manifest_binding_valid
            and intent.get("intent_sha256") == _record_digest(intent, "intent_sha256")
        )
    else:
        intent_valid = manifest is not None
        if intent_valid:
            try:
                _validate_retained_submission_map(
                    resolved_root,
                    str(receipt.get("label", "")),
                    intent_path,
                    intent,
                    manifest,
                )
            except (KeyError, TypeError, ValueError):
                intent_valid = False
    if (
        set(receipt)
        != {
            "schema_version",
            "status",
            "label",
            "source_commit",
            "output_root",
            "intent_path",
            "intent_file_sha256",
            "jobs",
            "dependency_policies",
            "failure",
            "launch_policy",
            "receipt_sha256",
        }
        or receipt_path != expected_receipt
        or receipt.get("schema_version") != SUBMISSION_RECORD_SCHEMA
        or receipt.get("status") != "submitted"
        or receipt.get("output_root") != str(resolved_root)
        or receipt.get("intent_path") != str(intent_path.relative_to(resolved_root))
        or receipt.get("intent_file_sha256") != study.benchmark.file_sha256(intent_path)
        or receipt.get("failure") is not None
        or receipt.get("launch_policy") != "held_until_receipt_persisted_before_release"
        or not job_binding_valid
        or not dependency_valid
        or not intent_valid
        or receipt.get("receipt_sha256") != _record_digest(receipt, "receipt_sha256")
        or (
            retained is not None
            and (
                retained.get("receipt_path") != str(receipt_path.relative_to(root))
                or retained.get("receipt_file_sha256") != file_sha256
                or retained.get("receipt_sha256") != receipt.get("receipt_sha256")
            )
        )
    ):
        raise ValueError("marker completion receipt differs")
    return {
        "receipt_path": str(receipt_path.relative_to(resolved_root)),
        "receipt_file_sha256": file_sha256,
        "receipt_sha256": receipt["receipt_sha256"],
        "receipt_label": receipt["label"],
        "intent_payload": dict(intent.get("payload", {})),
        "job_id": job_id,
    }


def _recovery_binding_matches_marker(
    root: Path,
    label: str,
    marker_path: Path,
    marker: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> bool:
    recovery_label, slurm_stage = COMPLETION_RECOVERY[label]
    return binding.get("receipt_label") == recovery_label and binding.get(
        "intent_payload"
    ) == {
        "slurm_stage": slurm_stage,
        "reason": "immutable_marker_scheduler_completion_recovery",
        "mode": "verification_only",
        "marker_path": str(marker_path.relative_to(root)),
        "marker_file_sha256": study.benchmark.file_sha256(marker_path),
        "marker_sha256": marker.get("marker_sha256"),
    }


def _owner_binding_matches_marker(label: str, binding: Mapping[str, Any]) -> bool:
    receipt_label = binding.get("receipt_label")
    payload = binding.get("intent_payload")
    if label == "preflight":
        return receipt_label == "preflight" and payload == {"slurm_stage": "preflight"}
    if label == "calibration":
        return (
            receipt_label == "calibration"
            and isinstance(payload, Mapping)
            and set(payload) == {"slurm_stage", "mode"}
            and payload.get("slurm_stage") == "calibration"
            and payload.get("mode") in {"new", "verification_only"}
        )
    if label in {"diagnostic", "model", "evaluation"}:
        verifier_stage = COMPLETION_RECOVERY[label][1]
        return (receipt_label == label and payload == {}) or (
            receipt_label == f"{label}-verifier"
            and payload
            == {
                "slurm_stage": verifier_stage,
                "reason": "all_cells_have_markers",
            }
        )
    if label == "final":
        return (
            receipt_label == "finalize"
            and isinstance(payload, Mapping)
            and set(payload) == {"slurm_stage", "mode"}
            and payload.get("slurm_stage") == "finalize"
            and payload.get("mode") in {"new", "report_recovery"}
        )
    return False


def _completion_directory(root: Path) -> Path:
    return root / "verified/scheduler-completions"


def _validate_completion_attestation(
    root: Path,
    label: str,
    marker_path: Path,
    marker: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    body = study.read_json(path)
    receipt_path = root / str(body.get("receipt_path", ""))
    binding = _receipt_completion_binding(
        root, receipt_path, str(body.get("job_id", ""))
    )
    accounting = _validate_accounting_observation(
        body.get("accounting_observation", {}),
        binding["job_id"],
        require_completed=True,
    )
    binding_role = body.get("binding_role")
    if binding_role == "owner":
        owner_receipt = _marker_receipt(marker)
        owner = _receipt_completion_binding(
            root,
            root / str(owner_receipt.get("receipt_path", "")),
            str(_marker_scheduler(marker).get("job_id", "")),
            retained=owner_receipt,
        )
        binding_role_valid = binding == owner and _owner_binding_matches_marker(
            label, binding
        )
    elif binding_role == "recovery":
        binding_role_valid = _recovery_binding_matches_marker(
            root, label, marker_path, marker, binding
        )
    else:
        binding_role_valid = False
    expected_path = _completion_directory(root) / (
        f"{label}-{marker['marker_sha256'][:16]}-{binding['job_id']}.json"
    )
    if (
        set(body)
        != {
            "schema_version",
            "status",
            "label",
            "source_commit",
            "marker_path",
            "marker_file_sha256",
            "marker_sha256",
            "receipt_path",
            "receipt_file_sha256",
            "receipt_sha256",
            "job_id",
            "binding_role",
            "accounting_observation",
            "attestation_sha256",
        }
        or path != expected_path
        or body.get("schema_version") != COMPLETION_ATTESTATION_SCHEMA
        or body.get("status") != "scheduler_completed_successfully"
        or body.get("label") != label
        or body.get("source_commit") != marker.get("source_commit")
        or body.get("marker_path") != str(marker_path.relative_to(root))
        or body.get("marker_file_sha256") != study.benchmark.file_sha256(marker_path)
        or body.get("marker_sha256") != marker.get("marker_sha256")
        or body.get("receipt_path") != binding["receipt_path"]
        or body.get("receipt_file_sha256") != binding["receipt_file_sha256"]
        or body.get("receipt_sha256") != binding["receipt_sha256"]
        or body.get("job_id") != binding["job_id"]
        or body.get("accounting_observation") != accounting
        or not binding_role_valid
        or body.get("attestation_sha256") != _record_digest(body, "attestation_sha256")
    ):
        raise ValueError("scheduler completion attestation differs")
    return dict(body)


def _write_completion_attestation(
    root: Path,
    label: str,
    marker_path: Path,
    marker: Mapping[str, Any],
    binding: Mapping[str, Any],
    accounting: Mapping[str, Any],
) -> dict[str, Any]:
    job_id = str(binding["job_id"])
    canonical_accounting = _validate_accounting_observation(
        accounting, job_id, require_completed=True
    )
    body = {
        "schema_version": COMPLETION_ATTESTATION_SCHEMA,
        "status": "scheduler_completed_successfully",
        "label": label,
        "source_commit": marker["source_commit"],
        "marker_path": str(marker_path.relative_to(root)),
        "marker_file_sha256": study.benchmark.file_sha256(marker_path),
        "marker_sha256": marker["marker_sha256"],
        "receipt_path": binding["receipt_path"],
        "receipt_file_sha256": binding["receipt_file_sha256"],
        "receipt_sha256": binding["receipt_sha256"],
        "job_id": job_id,
        "binding_role": binding["binding_role"],
        "accounting_observation": canonical_accounting,
    }
    body["attestation_sha256"] = _record_digest(body, "attestation_sha256")
    path = _completion_directory(root) / (
        f"{label}-{marker['marker_sha256'][:16]}-{job_id}.json"
    )
    if path.is_file():
        return _validate_completion_attestation(root, label, marker_path, marker, path)
    try:
        study._write_json_exclusive(path, body)
    except FileExistsError:
        return _validate_completion_attestation(root, label, marker_path, marker, path)
    return body


def _try_attest_marker_completion(
    root: Path,
    label: str,
    marker_path: Path,
    marker: Mapping[str, Any],
) -> bool:
    """Attest a successful owner/recovery job after it has actually exited."""

    directory = _completion_directory(root)
    existing = (
        sorted(directory.glob(f"{label}-{marker['marker_sha256'][:16]}-*.json"))
        if directory.is_dir()
        else []
    )
    if existing:
        for path in existing:
            _validate_completion_attestation(root, label, marker_path, marker, path)
        return True

    scheduler = _marker_scheduler(marker)
    owner_job = str(scheduler.get("job_id", ""))
    owner_receipt = _marker_receipt(marker)
    owner_path = root / str(owner_receipt.get("receipt_path", ""))
    owner_binding = _receipt_completion_binding(
        root, owner_path, owner_job, retained=owner_receipt
    )
    if not _owner_binding_matches_marker(label, owner_binding):
        raise ValueError("marker owner submission intent differs")
    candidates: list[dict[str, Any]] = [{**owner_binding, "binding_role": "owner"}]
    recovery_label = COMPLETION_RECOVERY[label][0]
    for receipt_path in sorted(
        (root / "submissions").glob(f"{recovery_label}-attempt-*.receipt.json")
    ):
        receipt = study.read_json(receipt_path)
        if receipt.get("status") != "submitted" or set(receipt.get("jobs", {})) != {
            "job"
        }:
            continue
        job_id = str(receipt["jobs"]["job"])
        binding = _receipt_completion_binding(root, receipt_path, job_id)
        if _recovery_binding_matches_marker(root, label, marker_path, marker, binding):
            candidate = {**binding, "binding_role": "recovery"}
            if candidate not in candidates:
                candidates.append(candidate)

    active = False
    for binding in candidates:
        try:
            observation = _slurm_observation(binding["job_id"])
        except SlurmAccountingAbsent:
            # Ancient failed/unknown owners do not block a new read-only
            # verifier; successful completion must be backed by retained sacct.
            continue
        state = str(observation["state"])
        if state == "COMPLETED":
            _write_completion_attestation(
                root, label, marker_path, marker, binding, observation
            )
            return True
        if state not in TERMINAL_SLURM_STATES:
            active = True
    if active:
        raise RuntimeError("marker completion verifier is still active")
    return False


def _receipt_path(intent_path: Path) -> Path:
    suffix = ".intent.json"
    if intent_path.name.endswith(suffix):
        name = intent_path.name[: -len(suffix)] + ".receipt.json"
    else:
        name = intent_path.stem + ".receipt.json"
    return intent_path.with_name(name)


def _release_path(receipt_path: Path) -> Path:
    suffix = ".receipt.json"
    if not receipt_path.name.endswith(suffix):
        raise ValueError("release outcome requires a canonical receipt path")
    return receipt_path.with_name(receipt_path.name[: -len(suffix)] + ".release.json")


def _bounded_text(value: object, limit: int = 4096) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    return text[:limit]


def _text_digest(value: object) -> dict[str, Any] | None:
    text = _bounded_text(value)
    if text is None:
        return None
    encoded = text.encode("utf-8")
    return {
        "bounded_text": text,
        "bounded_length": len(text),
        "bounded_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _release_observation(job_id: str) -> dict[str, Any]:
    """Classify the exact receipt-bound job as held, active, or terminal."""

    output = _squeue_output(job_id, "%T|%r")
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        fields = line.strip().split("|", 1)
        if len(fields) != 2:
            raise RuntimeError("Slurm release-state output is malformed")
        rows.append({"state": fields[0].strip(), "reason": fields[1].strip()})
    if not rows:
        terminal = _slurm_state(job_id)
        if terminal not in TERMINAL_SLURM_STATES:
            raise RuntimeError(
                "receipt-bound release job is neither queued nor terminal"
            )
        return {"classification": "terminal", "rows": [], "terminal": terminal}
    if any(
        row["state"] == "PENDING" and row["reason"] == "JobHeldAdmin" for row in rows
    ):
        classification = "admin_held"
    elif any(
        row["state"] == "PENDING" and row["reason"] == "JobHeldUser" for row in rows
    ):
        # A partially materialized array can mix RUNNING and held rows.  It is
        # not fully released until *zero* receipt-bound rows remain user-held.
        classification = "user_held"
    else:
        classification = "active"
    return {"classification": classification, "rows": rows, "terminal": None}


def _release_record_body(
    receipt_path: Path,
    job_id: str,
    status: str,
    before: Mapping[str, Any],
    *,
    release_invoked: bool,
    failure: Mapping[str, Any] | None = None,
    after: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = study.read_json(receipt_path)
    body = {
        "schema_version": RELEASE_RECORD_SCHEMA,
        "status": status,
        "receipt_path": receipt_path.name,
        "receipt_file_sha256": study.benchmark.file_sha256(receipt_path),
        "receipt_sha256": receipt["receipt_sha256"],
        "job_id": job_id,
        "before": dict(before),
        "release_invoked": release_invoked,
        "failure": None if failure is None else dict(failure),
        "after": None if after is None else dict(after),
    }
    body["release_sha256"] = _record_digest(body, "release_sha256")
    return body


def _validate_release_record(
    receipt_path: Path, job_id: str, body: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = study.read_json(receipt_path)
    status = body.get("status")
    before = body.get("before")
    after = body.get("after")
    observation_valid = (
        (
            status == "released"
            and isinstance(after, Mapping)
            and after.get("classification") in {"active", "terminal"}
        )
        or (
            status == "reconciled_active"
            and (
                (
                    isinstance(before, Mapping)
                    and before.get("classification") == "active"
                )
                or (
                    isinstance(after, Mapping)
                    and after.get("classification") == "active"
                )
            )
        )
        or (
            status == "reconciled_terminal"
            and (
                (
                    isinstance(before, Mapping)
                    and before.get("classification") == "terminal"
                )
                or (
                    isinstance(after, Mapping)
                    and after.get("classification") == "terminal"
                )
            )
        )
    )
    if (
        body.get("schema_version") != RELEASE_RECORD_SCHEMA
        or status not in {"released", "reconciled_active", "reconciled_terminal"}
        or body.get("receipt_path") != receipt_path.name
        or body.get("receipt_file_sha256") != study.benchmark.file_sha256(receipt_path)
        or body.get("receipt_sha256") != receipt.get("receipt_sha256")
        or body.get("job_id") != job_id
        or body.get("release_sha256") != _record_digest(body, "release_sha256")
        or not observation_valid
    ):
        raise ValueError("release reconciliation record differs")
    return dict(body)


def _write_release_attempt(
    receipt_path: Path,
    job_id: str,
    before: Mapping[str, Any],
    error: BaseException,
    after: Mapping[str, Any] | None,
) -> Path:
    prefix = receipt_path.name[: -len(".receipt.json")]
    attempt = 1
    while (
        receipt_path.parent / f"{prefix}.release-attempt-{attempt:03d}.json"
    ).exists():
        attempt += 1
    path = receipt_path.parent / f"{prefix}.release-attempt-{attempt:03d}.json"
    body = _release_record_body(
        receipt_path,
        job_id,
        "release_attempt_failed",
        before,
        release_invoked=True,
        failure=_failure_record(error, "scontrol_release"),
        after=after,
    )
    # Attempt records use the same digest field even though they are not final
    # reconciliation records.
    study._write_json_exclusive(path, body)
    return path


def _release_or_reconcile(receipt_path: Path, job_id: str) -> dict[str, Any]:
    """Idempotently release exactly the held job bound by an immutable receipt."""

    receipt = study.read_json(receipt_path)
    if job_id not in {str(value) for value in receipt.get("jobs", {}).values()}:
        raise ValueError("release job is not bound by its receipt")
    outcome_path = _release_path(receipt_path)
    if outcome_path.is_file():
        return _validate_release_record(
            receipt_path, job_id, study.read_json(outcome_path)
        )
    before = _release_observation(job_id)
    classification = before["classification"]
    if classification == "admin_held":
        raise RuntimeError("receipt-bound job is administratively held")
    if classification in {"active", "terminal"}:
        status = (
            "reconciled_active" if classification == "active" else "reconciled_terminal"
        )
        body = _release_record_body(
            receipt_path, job_id, status, before, release_invoked=False
        )
        study._write_json_exclusive(outcome_path, body)
        return body
    try:
        subprocess.check_call(["scontrol", "release", job_id])
    except (subprocess.CalledProcessError, OSError) as error:
        try:
            after = _release_observation(job_id)
        except BaseException as observation_error:
            after = {
                "classification": "unknown",
                "observation_failure": _failure_record(
                    observation_error, "release_reconciliation_query"
                ),
            }
        _write_release_attempt(receipt_path, job_id, before, error, after)
        if after.get("classification") not in {"active", "terminal"}:
            raise
        status = (
            "reconciled_active"
            if after["classification"] == "active"
            else "reconciled_terminal"
        )
        body = _release_record_body(
            receipt_path,
            job_id,
            status,
            before,
            release_invoked=True,
            failure=_failure_record(error, "scontrol_release_reply_lost"),
            after=after,
        )
    else:
        after = None
        for _ in range(5):
            after = _release_observation(job_id)
            if after.get("classification") in {"active", "terminal"}:
                break
            time.sleep(0.05)
        if after is None or after.get("classification") not in {
            "active",
            "terminal",
        }:
            error = RuntimeError("receipt-bound job remained held after release")
            _write_release_attempt(receipt_path, job_id, before, error, after)
            raise error
        body = _release_record_body(
            receipt_path,
            job_id,
            "released",
            before,
            release_invoked=True,
            after=after,
        )
    study._write_json_exclusive(outcome_path, body)
    return body


def _cancellation_path(receipt_path: Path) -> Path:
    suffix = ".receipt.json"
    if not receipt_path.name.endswith(suffix):
        raise ValueError("cancellation outcome requires a canonical receipt path")
    return receipt_path.with_name(
        receipt_path.name[: -len(suffix)] + ".cancellation.json"
    )


def _cancellation_record_body(
    receipt_path: Path,
    job_id: str,
    status: str,
    before: Mapping[str, Any],
    *,
    cancel_invoked: bool,
    failure: Mapping[str, Any] | None = None,
    after: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = study.read_json(receipt_path)
    body = {
        "schema_version": CANCELLATION_RECORD_SCHEMA,
        "status": status,
        "receipt_path": receipt_path.name,
        "receipt_file_sha256": study.benchmark.file_sha256(receipt_path),
        "receipt_sha256": receipt["receipt_sha256"],
        "job_id": job_id,
        "before": dict(before),
        "cancel_invoked": cancel_invoked,
        "failure": None if failure is None else dict(failure),
        "after": None if after is None else dict(after),
    }
    body["cancellation_sha256"] = _record_digest(body, "cancellation_sha256")
    return body


def _validate_cancellation_record(
    receipt_path: Path, job_id: str, body: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = study.read_json(receipt_path)
    after = body.get("after")
    before = body.get("before")
    terminal_observations = [
        observation
        for observation in (after, before)
        if isinstance(observation, Mapping)
        and observation.get("state") in {"COMPLETED", "FAILED"}
    ]
    terminal = False
    for observation in terminal_observations:
        try:
            _validate_accounting_observation(observation, job_id)
        except ValueError:
            continue
        terminal = True
        break
    if (
        body.get("schema_version") != CANCELLATION_RECORD_SCHEMA
        or body.get("status") not in {"cancelled", "reconciled_terminal"}
        or body.get("receipt_path") != receipt_path.name
        or body.get("receipt_file_sha256") != study.benchmark.file_sha256(receipt_path)
        or body.get("receipt_sha256") != receipt.get("receipt_sha256")
        or body.get("job_id") != job_id
        or not terminal
        or body.get("cancellation_sha256")
        != _record_digest(body, "cancellation_sha256")
    ):
        raise ValueError("cancellation reconciliation record differs")
    return dict(body)


def _write_cancellation_attempt(
    receipt_path: Path,
    job_id: str,
    before: Mapping[str, Any],
    error: BaseException,
    after: Mapping[str, Any] | None,
) -> Path:
    prefix = receipt_path.name[: -len(".receipt.json")]
    attempt = 1
    while (
        receipt_path.parent / f"{prefix}.cancellation-attempt-{attempt:03d}.json"
    ).exists():
        attempt += 1
    path = receipt_path.parent / f"{prefix}.cancellation-attempt-{attempt:03d}.json"
    body = _cancellation_record_body(
        receipt_path,
        job_id,
        "cancellation_attempt_incomplete",
        before,
        cancel_invoked=True,
        failure=_failure_record(error, "scancel"),
        after=after,
    )
    study._write_json_exclusive(path, body)
    return path


def _cancel_or_reconcile(receipt_path: Path, job_id: str) -> dict[str, Any]:
    """Idempotently terminalize the known array when its verifier is not usable."""

    receipt = study.read_json(receipt_path)
    if receipt.get("status") not in {
        "partial_submission",
        "ambiguous_submission",
    } or receipt.get("jobs") != {"array": job_id}:
        raise ValueError("cancellation requires its array-only failure receipt")
    outcome_path = _cancellation_path(receipt_path)
    if outcome_path.is_file():
        return _validate_cancellation_record(
            receipt_path, job_id, study.read_json(outcome_path)
        )
    before = _slurm_observation(job_id)
    if before["state"] in TERMINAL_SLURM_STATES:
        body = _cancellation_record_body(
            receipt_path,
            job_id,
            "reconciled_terminal",
            before,
            cancel_invoked=False,
        )
        study._write_json_exclusive(outcome_path, body)
        return body
    try:
        subprocess.check_call(["scancel", job_id])
    except (subprocess.CalledProcessError, OSError) as error:
        try:
            after = _slurm_observation(job_id)
        except BaseException as observation_error:
            after = {
                "state": "UNKNOWN",
                "observation_failure": _failure_record(
                    observation_error, "cancellation_reconciliation_query"
                ),
            }
        _write_cancellation_attempt(receipt_path, job_id, before, error, after)
        if after.get("state") not in TERMINAL_SLURM_STATES:
            raise
        body = _cancellation_record_body(
            receipt_path,
            job_id,
            "reconciled_terminal",
            before,
            cancel_invoked=True,
            failure=_failure_record(error, "scancel_reply_lost"),
            after=after,
        )
    else:
        after = None
        for _ in range(10):
            try:
                after = _slurm_observation(job_id)
            except SlurmAccountingAbsent:
                after = None
                time.sleep(0.1)
                continue
            if after["state"] in TERMINAL_SLURM_STATES:
                break
            time.sleep(0.1)
        if after is None or after.get("state") not in TERMINAL_SLURM_STATES:
            error = RuntimeError("receipt-bound array remained active after scancel")
            _write_cancellation_attempt(receipt_path, job_id, before, error, after)
            raise error
        body = _cancellation_record_body(
            receipt_path,
            job_id,
            "cancelled",
            before,
            cancel_invoked=True,
            after=after,
        )
    try:
        study._write_json_exclusive(outcome_path, body)
    except FileExistsError:
        return _validate_cancellation_record(
            receipt_path, job_id, study.read_json(outcome_path)
        )
    return body


def _validate_retained_submission_map(
    root: Path,
    label: str,
    path: Path,
    body: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Reauthenticate every byte and index authorized by a retained array map."""

    resolved_root = root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    submission_root = (resolved_root / "submissions").resolve(strict=True)
    match = re.fullmatch(
        rf"{re.escape(label)}-attempt-([0-9]+)\.json", resolved_path.name
    )
    expected_keys = {
        "schema_version",
        "status",
        "stage",
        "attempt",
        "source_commit",
        "manifest_sha256",
        "upstream_stage",
        "upstream_marker_sha256",
        "entries",
        "map_path",
        "map_sha256",
    }
    if (
        not resolved_path.is_relative_to(submission_root)
        or match is None
        or set(body) != expected_keys
        or body.get("schema_version") != study.SUBMISSION_MAP_SCHEMA
        or body.get("status") != "frozen_before_submission"
        or body.get("stage") != label
        or not isinstance(body.get("attempt"), int)
        or isinstance(body.get("attempt"), bool)
        or int(body["attempt"]) <= 0
        or int(match.group(1)) != int(body["attempt"])
        or match.group(1) != f"{int(body['attempt']):03d}"
        or body.get("source_commit") != manifest.get("source_commit")
        or body.get("manifest_sha256") != manifest.get("manifest_sha256")
        or body.get("map_path") != str(resolved_path.relative_to(resolved_root))
        or body.get("map_sha256") != _record_digest(body, "map_sha256")
    ):
        raise ValueError(f"submission map is invalid: {path}")
    upstream = study._submission_upstream_marker(resolved_root, manifest, label)
    if body.get("upstream_stage") != upstream.get("stage") or body.get(
        "upstream_marker_sha256"
    ) != upstream.get("marker_sha256"):
        raise ValueError(f"submission map upstream binding is invalid: {path}")
    entries = body.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"submission map entries are invalid: {path}")
    seen: set[int] = set()
    for entry in entries:
        if (
            not isinstance(entry, Mapping)
            or set(entry)
            != {"index", "mode", "retained_data_files", "artifact_state_sha256"}
            or not isinstance(entry.get("index"), int)
            or isinstance(entry.get("index"), bool)
            or entry.get("mode") not in {"new", "verification_only"}
            or not isinstance(entry.get("retained_data_files"), list)
            or entry.get("artifact_state_sha256")
            != study._submission_entry_digest(entry)
            or int(entry["index"]) in seen
        ):
            raise ValueError(f"submission map entries are invalid: {path}")
        seen.add(int(entry["index"]))
        cell = study._cell(manifest, label, int(entry["index"]))
        expected_files = (
            []
            if entry["mode"] == "new"
            else study._retained_data_file_records(resolved_root, cell, label)
        )
        if entry["retained_data_files"] != expected_files:
            raise ValueError(
                f"submission map retained artifact digests changed: {path}"
            )
    return dict(body)


def _intent_paths(root: Path, label: str) -> list[Path]:
    directory = root / "submissions"
    if not directory.is_dir():
        return []
    explicit = list(directory.glob(f"{label}-attempt-*.intent.json"))
    maps = [
        path
        for path in directory.glob(f"{label}-attempt-*.json")
        if not path.name.endswith(
            (
                ".intent.json",
                ".receipt.json",
                ".release.json",
                ".cancellation.json",
            )
        )
        and ".release-attempt-" not in path.name
        and ".cancellation-attempt-" not in path.name
    ]
    return sorted({*explicit, *maps})


def _assert_no_unresolved_submission(root: Path, label: str) -> None:
    """Refuse duplicate work while any prior intent lacks terminal accounting."""

    for intent_path in _intent_paths(root, label):
        intent = study.read_json(intent_path)
        manifest_path = root / "manifest.json"
        retained_manifest = (
            study.read_json(manifest_path) if manifest_path.is_file() else None
        )
        if intent_path.name.endswith(".intent.json"):
            if (
                intent.get("schema_version") != SUBMISSION_RECORD_SCHEMA
                or intent.get("status") != "frozen_before_submission"
                or intent.get("label") != label
                or intent.get("output_root") != str(root)
                or not isinstance(intent.get("source_commit"), str)
                or (
                    retained_manifest is not None
                    and intent.get("source_commit")
                    != retained_manifest.get("source_commit")
                )
                or (
                    label == "preflight"
                    and intent.get("manifest_sha256")
                    not in {
                        None,
                        (
                            None
                            if retained_manifest is None
                            else retained_manifest.get("manifest_sha256")
                        ),
                    }
                )
                or (
                    label != "preflight"
                    and (
                        retained_manifest is None
                        or intent.get("manifest_sha256")
                        != retained_manifest.get("manifest_sha256")
                    )
                )
                or intent.get("intent_sha256")
                != _record_digest(intent, "intent_sha256")
            ):
                raise ValueError(f"submission intent is invalid: {intent_path}")
        else:
            if retained_manifest is None:
                raise ValueError(f"submission map lacks its manifest: {intent_path}")
            _validate_retained_submission_map(
                root, label, intent_path, intent, retained_manifest
            )
        receipt_path = _receipt_path(intent_path)
        if not receipt_path.is_file():
            raise RuntimeError(
                f"unresolved submission intent has no receipt: {intent_path}"
            )
        receipt = study.read_json(receipt_path)
        if (
            receipt.get("schema_version") != SUBMISSION_RECORD_SCHEMA
            or receipt.get("status")
            not in {
                "submitted",
                "submission_failed",
                "partial_submission",
                "ambiguous_submission",
            }
            or receipt.get("label") != label
            or receipt.get("output_root") != str(root)
            or receipt.get("source_commit") != intent.get("source_commit")
            or receipt.get("intent_path") != str(intent_path.relative_to(root))
            or receipt.get("intent_file_sha256")
            != study.benchmark.file_sha256(intent_path)
            or receipt.get("receipt_sha256")
            != _record_digest(receipt, "receipt_sha256")
        ):
            raise ValueError(f"submission receipt is invalid: {receipt_path}")
        jobs = receipt.get("jobs", {})
        status = str(receipt["status"])
        failure = receipt.get("failure")
        if (
            any(not str(job).isdigit() for job in jobs.values())
            or (status == "submitted" and (not jobs or failure is not None))
            or (
                status == "submission_failed"
                and (jobs or not isinstance(failure, Mapping))
            )
            or (
                status == "partial_submission"
                and (not jobs or not isinstance(failure, Mapping))
            )
            or (status == "ambiguous_submission" and not isinstance(failure, Mapping))
            or receipt.get("launch_policy")
            != (
                "submission_outcome_unknown_manual_resolution"
                if status == "ambiguous_submission" and not jobs
                else (
                    "not_submitted"
                    if not jobs
                    else "held_until_receipt_persisted_before_release"
                )
            )
        ):
            raise ValueError(f"submission receipt job ids are invalid: {receipt_path}")
        if not intent_path.name.endswith(".intent.json"):
            if status == "submitted" and (
                set(jobs) != {"array", "verifier"}
                or receipt.get("dependency_policies")
                != {"verifier": "afterok_array_kill_on_invalid_dependency"}
            ):
                raise ValueError(
                    f"array submission dependency policy is invalid: {receipt_path}"
                )
            if status == "partial_submission" and (
                set(jobs) != {"array"} or receipt.get("dependency_policies") != {}
            ):
                raise ValueError(f"partial array receipt is invalid: {receipt_path}")
            if (
                status == "submission_failed"
                and receipt.get("dependency_policies") != {}
            ):
                raise ValueError(f"failed array receipt is invalid: {receipt_path}")
            if status == "ambiguous_submission" and (
                set(jobs) not in (set(), {"array"})
                or receipt.get("dependency_policies") != {}
            ):
                raise ValueError(f"ambiguous array receipt is invalid: {receipt_path}")
        else:
            if status == "submitted" and (
                set(jobs) != {"job"} or receipt.get("dependency_policies") != {}
            ):
                raise ValueError(f"single-job receipt is invalid: {receipt_path}")
            if status == "submission_failed" and (
                jobs or receipt.get("dependency_policies") != {}
            ):
                raise ValueError(
                    f"failed single-job receipt is invalid: {receipt_path}"
                )
            if status == "ambiguous_submission" and (
                jobs or receipt.get("dependency_policies") != {}
            ):
                raise ValueError(
                    f"ambiguous single-job receipt is invalid: {receipt_path}"
                )
        if status == "submission_failed":
            continue
        if status == "ambiguous_submission":
            if set(jobs) == {"array"}:
                _cancel_or_reconcile(receipt_path, str(jobs["array"]))
            raise RuntimeError(
                f"ambiguous prior submission requires manual resolution: {receipt_path}"
            )
        if status == "partial_submission":
            _cancel_or_reconcile(receipt_path, str(jobs["array"]))
            continue
        if status == "submitted":
            release_job = str(jobs["array"] if "array" in jobs else jobs.get("job", ""))
            _release_or_reconcile(receipt_path, release_job)
        states = {name: _slurm_state(str(job)) for name, job in jobs.items()}
        unresolved = {
            name: state
            for name, state in states.items()
            if state not in TERMINAL_SLURM_STATES
        }
        if unresolved:
            raise RuntimeError(
                f"prior {label} submission is still active; refusing duplicate: {unresolved}"
            )


def _write_intent(
    root: Path, label: str, arguments: Any, payload: Mapping[str, Any]
) -> Path:
    directory = root / "submissions"
    attempt = 1
    while any(directory.glob(f"{label}-attempt-{attempt:03d}.*.json")):
        attempt += 1
    path = directory / f"{label}-attempt-{attempt:03d}.intent.json"
    manifest_path = root / "manifest.json"
    manifest_sha256 = (
        study.read_json(manifest_path).get("manifest_sha256")
        if manifest_path.is_file()
        else None
    )
    body = {
        "schema_version": SUBMISSION_RECORD_SCHEMA,
        "status": "frozen_before_submission",
        "label": label,
        "attempt": attempt,
        "source_commit": arguments.source_commit,
        "output_root": str(root),
        "manifest_sha256": manifest_sha256,
        "payload": dict(payload),
    }
    body["intent_sha256"] = _record_digest(body, "intent_sha256")
    study._write_json_exclusive(path, body)
    return path


def _write_receipt(
    intent_path: Path,
    label: str,
    arguments: Any,
    jobs: Mapping[str, str],
    *,
    dependency_policies: Mapping[str, str] | None = None,
    status: str = "submitted",
    failure: Mapping[str, Any] | None = None,
) -> Path:
    if status not in {
        "submitted",
        "submission_failed",
        "partial_submission",
        "ambiguous_submission",
    }:
        raise ValueError("unknown submission receipt status")
    body = {
        "schema_version": SUBMISSION_RECORD_SCHEMA,
        "status": status,
        "label": label,
        "source_commit": arguments.source_commit,
        "output_root": str(arguments.output_root),
        "intent_path": str(intent_path.relative_to(arguments.output_root)),
        "intent_file_sha256": study.benchmark.file_sha256(intent_path),
        "jobs": dict(jobs),
        "dependency_policies": dict(dependency_policies or {}),
        "failure": None if failure is None else dict(failure),
        "launch_policy": (
            "submission_outcome_unknown_manual_resolution"
            if status == "ambiguous_submission" and not jobs
            else (
                "not_submitted"
                if not jobs
                else "held_until_receipt_persisted_before_release"
            )
        ),
    }
    body["receipt_sha256"] = _record_digest(body, "receipt_sha256")
    path = _receipt_path(intent_path)
    study._write_json_exclusive(path, body)
    return path


def _failure_record(error: BaseException, boundary: str) -> dict[str, Any]:
    return {
        "boundary": boundary,
        "exception_type": type(error).__name__,
        "returncode": (
            int(error.returncode)
            if isinstance(error, subprocess.CalledProcessError)
            else None
        ),
        "errno": (
            int(error.errno)
            if isinstance(error, OSError) and error.errno is not None
            else None
        ),
        "message": _text_digest(error),
        "command": _text_digest(
            error.cmd if isinstance(error, subprocess.CalledProcessError) else None
        ),
        "stdout": _text_digest(
            error.stdout if isinstance(error, subprocess.CalledProcessError) else None
        ),
        "stderr": _text_digest(
            error.stderr if isinstance(error, subprocess.CalledProcessError) else None
        ),
    }


def _submit_single_job(
    intent_path: Path,
    label: str,
    arguments: Any,
    exports: Mapping[str, str],
) -> tuple[str, Path]:
    try:
        job_id = _sbatch(arguments.sbatch, exports, hold=True)
    except OSError as error:
        _write_receipt(
            intent_path,
            label,
            arguments,
            {},
            status="submission_failed",
            failure=_failure_record(error, "single_sbatch"),
        )
        raise
    except subprocess.CalledProcessError as error:
        _write_receipt(
            intent_path,
            label,
            arguments,
            {},
            status="ambiguous_submission",
            failure=_failure_record(error, "single_sbatch"),
        )
        raise
    except RuntimeError as error:
        _write_receipt(
            intent_path,
            label,
            arguments,
            {},
            status="ambiguous_submission",
            failure=_failure_record(error, "single_sbatch_output"),
        )
        raise
    receipt_path = _write_receipt(intent_path, label, arguments, {"job": job_id})
    _release_or_reconcile(receipt_path, job_id)
    return job_id, receipt_path


def _submit_marker_completion_recovery(
    arguments: Any,
    label: str,
    marker_path: Path,
    marker: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_label, slurm_stage = COMPLETION_RECOVERY[label]
    _assert_no_unresolved_submission(arguments.output_root, receipt_label)
    payload: dict[str, Any] = {
        "slurm_stage": slurm_stage,
        "reason": "immutable_marker_scheduler_completion_recovery",
        "mode": "verification_only",
        "marker_path": str(marker_path.relative_to(arguments.output_root)),
        "marker_file_sha256": study.benchmark.file_sha256(marker_path),
        "marker_sha256": marker["marker_sha256"],
    }
    intent_path = _write_intent(
        arguments.output_root, receipt_label, arguments, payload
    )
    job_id, receipt_path = _submit_single_job(
        intent_path,
        receipt_label,
        arguments,
        _base_exports(arguments, slurm_stage),
    )
    return {
        "status": "marker_completion_recovery_submitted",
        "label": label,
        "recovery_job": job_id,
        "submission_receipt": str(receipt_path),
    }


def _classify_cells(
    root: Path, manifest: Mapping[str, Any], stage: str
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for cell in manifest[f"{stage}_cells"]:
        paths = study._cell_artifact_paths(root, cell, stage)
        data_paths, marker_path = paths[:-1], paths[-1]
        if marker_path.is_file():
            study._validate_cell_marker(root, manifest, cell, stage)
            continue
        present = [path.is_file() for path in data_paths]
        if not any(present):
            if any(path.exists() for path in paths):
                raise ValueError(f"non-file artifact blocks {cell['cell_id']}")
            mode = "new"
        elif all(present):
            if _verification_only_was_submitted(root, stage, int(cell["index"])):
                raise ValueError(
                    f"strict verification already ran for {cell['cell_id']}; "
                    "preserve this output and use a fresh audited output root"
                )
            mode = "verification_only"
        else:
            raise ValueError(
                f"partial immutable artifacts for {cell['cell_id']}; "
                "preserve this output and use a fresh audited output root"
            )
        entries.append({"index": int(cell["index"]), "mode": mode})
    return entries


def _verification_only_was_submitted(root: Path, stage: str, index: int) -> bool:
    """Allow one replay retry, but never loop on known-invalid immutable bytes."""

    for map_path in _intent_paths(root, stage):
        if map_path.name.endswith(".intent.json"):
            continue
        body = study.read_json(map_path)
        matching = [
            entry
            for entry in body.get("entries", ())
            if int(entry.get("index", -1)) == int(index)
            and entry.get("mode") == "verification_only"
        ]
        if not matching:
            continue
        receipt_path = _receipt_path(map_path)
        if receipt_path.is_file() and study.read_json(receipt_path).get("status") == (
            "submitted"
        ):
            return True
    return False


def _base_exports(arguments: Any, stage: str) -> dict[str, str]:
    return {
        "AGR_SOURCE_ROOT": str(arguments.source_root),
        "AGR_DEPENDENCY_ROOT": str(arguments.dependency_root),
        "AGR_OUTPUT_ROOT": str(arguments.output_root),
        "AGR_SOURCE_COMMIT": arguments.source_commit,
        "AGR_STAGE": stage,
        "AGR_MODEL_UPDATES": str(arguments.model_updates),
        "AGR_EVALUATION_EPISODES": str(arguments.evaluation_episodes),
    }


def _submit_array_stage(arguments: Any, stage_name: str) -> dict[str, Any]:
    logical_stage, cell_stage, verifier_stage, default_concurrency = ARRAY_STAGES[
        stage_name
    ]
    root = arguments.output_root
    manifest = study.read_json(root / "manifest.json")
    study.validate_manifest(manifest)
    if logical_stage == "diagnostic":
        # Submission is deliberately read-only with respect to scientific
        # computation.  The calibration verifier already rederived the
        # retained result in its own GPU Slurm process; recomputing it here
        # would run on the login host and make certification backend-dependent.
        upstream_marker = study._validate_calibration_marker(root, manifest)
        upstream_label = "calibration"
        upstream_path = root / str(manifest["calibration"]["marker_path"])
    elif logical_stage == "model":
        upstream_marker = study._validate_stage_marker(root, manifest, "diagnostic")
        upstream_label = "diagnostic"
        upstream_path = root / study._STAGE_PATHS["diagnostic"]
    else:
        upstream_marker = study._validate_stage_marker(root, manifest, "model")
        upstream_label = "model"
        upstream_path = root / study._STAGE_PATHS["model"]
    if not _try_attest_marker_completion(
        root, upstream_label, upstream_path, upstream_marker
    ):
        raise RuntimeError(
            f"{upstream_label} marker needs its stage completion-recovery run"
        )

    stage_marker_path = root / study._STAGE_PATHS[logical_stage]
    if stage_marker_path.is_file():
        marker = study._validate_stage_marker(root, manifest, logical_stage)
        if not _try_attest_marker_completion(
            root, logical_stage, stage_marker_path, marker
        ):
            return _submit_marker_completion_recovery(
                arguments, logical_stage, stage_marker_path, marker
            )
        return {"status": "already_verified", "stage_marker": marker}

    _assert_no_unresolved_submission(root, logical_stage)
    entries = _classify_cells(root, manifest, logical_stage)
    exports = _base_exports(arguments, cell_stage)
    array_job: str | None = None
    submission_map: dict[str, Any] | None = None
    receipt_path: Path
    if entries:
        submission_map = study.write_submission_map(root, logical_stage, entries)
        map_path = root / submission_map["map_path"]
        exports.update(
            {
                "AGR_SUBMISSION_MAP": str(map_path),
                "AGR_SUBMISSION_MAP_FILE_SHA256": study.benchmark.file_sha256(map_path),
            }
        )
        concurrency = arguments.concurrency or default_concurrency
        try:
            array_job = _sbatch(
                arguments.sbatch,
                exports,
                array=_array_spec([entry["index"] for entry in entries], concurrency),
                hold=True,
            )
        except OSError as error:
            receipt_path = _write_receipt(
                map_path,
                logical_stage,
                arguments,
                {},
                status="submission_failed",
                failure=_failure_record(error, "array_sbatch"),
            )
            raise
        except subprocess.CalledProcessError as error:
            receipt_path = _write_receipt(
                map_path,
                logical_stage,
                arguments,
                {},
                status="ambiguous_submission",
                failure=_failure_record(error, "array_sbatch"),
            )
            raise
        except RuntimeError as error:
            _write_receipt(
                map_path,
                logical_stage,
                arguments,
                {},
                status="ambiguous_submission",
                failure=_failure_record(error, "array_sbatch_output"),
            )
            raise
        try:
            verifier_job = _sbatch(
                arguments.sbatch,
                _base_exports(arguments, verifier_stage),
                dependency_job=array_job,
            )
        except OSError as error:
            receipt_path = _write_receipt(
                map_path,
                logical_stage,
                arguments,
                {"array": array_job},
                status="partial_submission",
                failure=_failure_record(error, "verifier_sbatch"),
            )
            _cancel_or_reconcile(receipt_path, array_job)
            raise
        except subprocess.CalledProcessError as error:
            receipt_path = _write_receipt(
                map_path,
                logical_stage,
                arguments,
                {"array": array_job},
                status="ambiguous_submission",
                failure=_failure_record(error, "verifier_sbatch"),
            )
            _cancel_or_reconcile(receipt_path, array_job)
            raise
        except RuntimeError as error:
            receipt_path = _write_receipt(
                map_path,
                logical_stage,
                arguments,
                {"array": array_job},
                status="ambiguous_submission",
                failure=_failure_record(error, "verifier_sbatch_output"),
            )
            _cancel_or_reconcile(receipt_path, array_job)
            raise
        receipt_path = _write_receipt(
            map_path,
            logical_stage,
            arguments,
            {"array": array_job, "verifier": verifier_job},
            dependency_policies={
                "verifier": "afterok_array_kill_on_invalid_dependency"
            },
        )
        _release_or_reconcile(receipt_path, array_job)
    else:
        verifier_label = f"{logical_stage}-verifier"
        _assert_no_unresolved_submission(root, verifier_label)
        intent_path = _write_intent(
            root,
            verifier_label,
            arguments,
            {"slurm_stage": verifier_stage, "reason": "all_cells_have_markers"},
        )
        verifier_job, receipt_path = _submit_single_job(
            intent_path,
            verifier_label,
            arguments,
            _base_exports(arguments, verifier_stage),
        )
    return {
        "status": "submitted",
        "logical_stage": logical_stage,
        "array_job": array_job,
        "verifier_job": verifier_job,
        "submitted_entries": entries,
        "submission_map": submission_map,
        "submission_receipt": str(receipt_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--model-updates", type=int, default=study.MODEL_UPDATES)
    parser.add_argument(
        "--evaluation-episodes", type=int, default=study.EVALUATION_EPISODES
    )
    parser.add_argument("--concurrency", type=int)
    arguments = parser.parse_args()

    arguments.source_root = arguments.source_root.resolve(strict=True)
    arguments.dependency_root = arguments.dependency_root.resolve(strict=True)
    arguments.output_root = arguments.output_root.resolve(strict=False)
    arguments.sbatch = (
        arguments.source_root
        / "dreamer_imf_comparison/cluster/actor_gap_roadmap/actor_gap_roadmap.sbatch"
    ).resolve(strict=True)
    _validate_source(arguments.source_root, arguments.source_commit)
    if arguments.model_updates <= 0 or not 1 <= arguments.evaluation_episodes <= 5:
        raise ValueError("model/evaluation settings are outside the frozen bounds")

    lock_label = {
        "diagnostics": "diagnostic",
        "models": "model",
        "evaluations": "evaluation",
    }.get(arguments.stage, arguments.stage)
    with _stage_submission_lock(arguments.output_root, lock_label):
        return _dispatch(arguments)


def _dispatch(arguments: Any) -> int:

    if arguments.stage == "preflight":
        marker_path = arguments.output_root / "verified/preflight.json"
        if marker_path.is_file():
            marker = study.validate_preflight_marker(arguments.output_root)
            if _try_attest_marker_completion(
                arguments.output_root, "preflight", marker_path, marker
            ):
                payload = {"status": "already_verified", "preflight_marker": marker}
            else:
                payload = _submit_marker_completion_recovery(
                    arguments, "preflight", marker_path, marker
                )
        elif marker_path.exists():
            raise ValueError("preflight marker is not a regular file")
        else:
            if (
                arguments.output_root.exists()
                and not (arguments.output_root / "manifest.json").is_file()
            ):
                unexpected = [
                    path
                    for path in arguments.output_root.iterdir()
                    if path.name != "submissions"
                ]
                if unexpected:
                    raise ValueError("preflight output root is not fresh or resumable")
            _assert_no_unresolved_submission(arguments.output_root, "preflight")
            intent_path = _write_intent(
                arguments.output_root,
                "preflight",
                arguments,
                {"slurm_stage": "preflight"},
            )
            job_id, receipt_path = _submit_single_job(
                intent_path,
                "preflight",
                arguments,
                _base_exports(arguments, "preflight"),
            )
            payload = {
                "status": "submitted",
                "preflight_job": job_id,
                "submission_receipt": str(receipt_path),
            }
    elif arguments.stage == "calibration":
        preflight_marker = study.validate_preflight_marker(arguments.output_root)
        manifest = study.read_json(arguments.output_root / "manifest.json")
        preflight_path = arguments.output_root / str(
            manifest["preflight"]["marker_path"]
        )
        if not _try_attest_marker_completion(
            arguments.output_root,
            "preflight",
            preflight_path,
            preflight_marker,
        ):
            raise RuntimeError(
                "preflight marker needs completion recovery; rerun the preflight stage"
            )
        marker_path = arguments.output_root / str(
            manifest["calibration"]["marker_path"]
        )
        if marker_path.is_file():
            calibration_marker = study._validate_calibration_marker(
                arguments.output_root, manifest
            )
            if _try_attest_marker_completion(
                arguments.output_root,
                "calibration",
                marker_path,
                calibration_marker,
            ):
                payload = {
                    "status": "already_verified",
                    "calibration_marker": calibration_marker,
                }
            else:
                payload = _submit_marker_completion_recovery(
                    arguments, "calibration", marker_path, calibration_marker
                )
        else:
            result_path = arguments.output_root / str(
                manifest["calibration"]["result_path"]
            )
            arrays_path = arguments.output_root / str(
                manifest["calibration"]["arrays_path"]
            )
            if result_path.exists() != arrays_path.exists():
                raise ValueError(
                    "partial immutable calibration evidence requires a fresh output root"
                )
            _assert_no_unresolved_submission(arguments.output_root, "calibration")
            intent_path = _write_intent(
                arguments.output_root,
                "calibration",
                arguments,
                {
                    "slurm_stage": "calibration",
                    "mode": (
                        "verification_only"
                        if result_path.is_file() and arrays_path.is_file()
                        else "new"
                    ),
                },
            )
            job_id, receipt_path = _submit_single_job(
                intent_path,
                "calibration",
                arguments,
                _base_exports(arguments, "calibration"),
            )
            payload = {
                "status": "submitted",
                "calibration_job": job_id,
                "submission_receipt": str(receipt_path),
            }
    elif arguments.stage in ARRAY_STAGES:
        payload = _submit_array_stage(arguments, arguments.stage)
    else:
        manifest = study.read_json(arguments.output_root / "manifest.json")
        study.validate_manifest(manifest)
        evaluation_marker = study._validate_stage_marker(
            arguments.output_root, manifest, "evaluation"
        )
        evaluation_path = arguments.output_root / study._STAGE_PATHS["evaluation"]
        if not _try_attest_marker_completion(
            arguments.output_root,
            "evaluation",
            evaluation_path,
            evaluation_marker,
        ):
            raise RuntimeError(
                "evaluation marker needs completion recovery; rerun the evaluations stage"
            )
        report_path = arguments.output_root / "report.json"
        final_marker_path = arguments.output_root / "verified/final.json"
        if report_path.is_file() and final_marker_path.is_file():
            report = study.validate_final(arguments.output_root)
            final_marker = study.read_json(final_marker_path)
            if _try_attest_marker_completion(
                arguments.output_root,
                "final",
                final_marker_path,
                final_marker,
            ):
                payload = {
                    "status": "already_verified",
                    "report": report,
                }
            else:
                payload = _submit_marker_completion_recovery(
                    arguments, "final", final_marker_path, final_marker
                )
        elif final_marker_path.exists():
            raise ValueError("final marker exists without its immutable report")
        else:
            _assert_no_unresolved_submission(arguments.output_root, "finalize")
            mode = "report_recovery" if report_path.is_file() else "new"
            intent_path = _write_intent(
                arguments.output_root,
                "finalize",
                arguments,
                {"slurm_stage": "finalize", "mode": mode},
            )
            job_id, receipt_path = _submit_single_job(
                intent_path,
                "finalize",
                arguments,
                _base_exports(arguments, "finalize"),
            )
            payload = {
                "status": "submitted",
                "finalize_job": job_id,
                "submission_receipt": str(receipt_path),
            }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
