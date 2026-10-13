"""Frozen posterior-reward continuation identities and fail-closed evidence.

This module never creates an environment or mutates historical evidence. Generic
byte/hash/runtime helpers are reused without changing any legacy module globals.
Native-step reservations are durable before simulator calls and never refunded.
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from .reward_exploration_protocol import (
    BASE,
    DATASET_SHA256 as DATASET_SHA256,
    INTERPRETER_PROFILES,
    PARENT_SHA256 as PARENT_SHA256,
    ZERO_HASH,
    _inside,
    _parent_identity,
    _unique_pairs,
    artifact,
    canonical,
    clean_commit as clean_commit,
    object_sha256,
    read,
    runtime_identity,
    sha,
    source_identity,
    verify_artifact,
    verify_runtime_identity,
    write_bytes_exclusive,
    write_json_exclusive,
)

SOURCE = Path(__file__).resolve().parents[2]
PROTOCOL = SOURCE / "dreamer_imf_comparison/posterior_reward_protocol.json"
PROTOCOL_DIGEST = "73c4c893a2dc28f0a2ef347e8f54cdd3dbc771d6505675b6ca90d2f0c9b2d485"
ARMS = ("A", "B", "C")
SEEDS = (901, 902, 903)
STAGES = ("preflight", "cells", "probes", "finalize")
CELL_COUNT = 9
CELL_CEILING = 120000
TOTAL_CEILING = 1080000
__all__ = [
    "DATASET_SHA256",
    "PARENT_SHA256",
    "clean_commit",
    "read",
    "sha",
    "artifact",
    "write_json_exclusive",
    "write_bytes_exclusive",
    "load_protocol",
    "cell_for_index",
    "cell_directory",
    "collection_seed",
    "eval_seeds",
    "initialize_manifest",
    "authenticate",
    "NativeStepBudget",
    "write_marker",
    "require_marker",
    "object_sha256",
]


def validate_protocol(value):
    if object_sha256(value) != PROTOCOL_DIGEST:
        raise ValueError("protocol differs from the frozen posterior-reward experiment")
    return copy.deepcopy(value)


def load_protocol(path=None):
    return validate_protocol(read(PROTOCOL if path is None else path))


def cell_for_index(index):
    if type(index) is not int or not 0 <= index < CELL_COUNT:
        raise ValueError("cell index must be an integer from 0 to 8")
    arm, seed = ARMS[index // 3], SEEDS[index % 3]
    return dict(
        index=index,
        arm=arm,
        seed=seed,
        readout_seed=seed - 901,
        cell_id=f"{arm}-seed{seed}",
    )


def cell_directory(output, index):
    return Path(output) / "cells" / cell_for_index(index)["cell_id"]


def _seed(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("unregistered continuation seed")


def collection_seed(seed, worker):
    _seed(seed)
    if type(worker) is not int or not 0 <= worker < 16:
        raise ValueError("unregistered collection worker")
    return 100000000 + (seed - 901) * 100000 + worker


def eval_seeds(seed, milestone):
    _seed(seed)
    if type(milestone) is not int or milestone not in (0, 40000, 80000):
        raise ValueError("unregistered evaluation milestone")
    return [
        200000000 + (seed - 901) * 100000 + (milestone // 40000) * 100 + i
        for i in range(10)
    ]


def _rmt_identity(path, protocol):
    """Bind historical proof and selected weights, never historical Slurm jobs.

    The pinned report authenticates the selection table; the pinned final seal
    authenticates the evaluated inventory and hence selected weight bytes. All
    imported/provenance bytes are rechecked; unrelated rejected historical fits
    need not be rehashed in every online process.
    """
    path = Path(path).resolve()
    manifest = read(path / "manifest.json")
    verified = read(path / "verified.json")
    report = read(path / "report.json")
    selections = read(path / "selections.json")
    if (
        manifest.get("source_commit") != protocol["rmt_source_commit"]
        or manifest.get("preflight") is not False
        or verified.get("exact_replay") is not True
        or verified.get("source_commit") != protocol["rmt_source_commit"]
        or verified.get("report_sha256") != protocol["rmt_report_sha256"]
        or sha(path / "report.json") != protocol["rmt_report_sha256"]
        or verified.get("evaluated_sha256") != sha(path / "evaluated.json")
        or sha(path / "evaluated.json") != protocol["rmt_evaluated_sha256"]
        or verified.get("frozen_digest") != protocol["rmt_frozen_digest"]
        or report.get("selections") != selections
    ):
        raise ValueError("RMT dependency is not the pinned verified full study")
    seal = read(path / "evaluated.json")
    if seal.get("stage") != "evaluated" or not seal.get("artifacts"):
        raise ValueError("invalid RMT evaluated seal")
    for name in ("manifest.json", "report.json", "selections.json"):
        if seal["artifacts"].get(name) != sha(path / name):
            raise ValueError("RMT seal does not bind required provenance: " + name)
    selected = {}
    for group in ("hz", "hmean"):
        family = selections[group]
        if set(family["selections"]) != {"0", "1", "2"}:
            raise ValueError("RMT selected initialization seed set changed")
        selected[group] = {}
        for seed in range(3):
            directory = (
                path / "heads" / f"reward-{group}-a{float(family['alpha']):g}-s{seed}"
            )
            selection = read(directory / "selection.json")
            candidate = family["selections"][str(seed)]
            if (
                selection.get("seed") != seed
                or selection.get("alpha") != family["alpha"]
                or selection.get("width") != 128
                or selection.get("depth") != 2
                or selection.get("classes") != 3
                or selection.get("selected") != candidate
                or min(
                    selection["candidates"],
                    key=lambda row: (row["validation"]["nll"], row["update"]),
                )
                != candidate
            ):
                raise ValueError("RMT head is not the validation-selected candidate")
            files = {}
            for name in (
                "selection.json",
                "normalization.npz",
                f"checkpoint-{candidate['update']:05d}.pkl",
            ):
                value = artifact(directory / name)
                relative = _inside(path, directory / name)
                if seal["artifacts"].get(relative) != value["sha256"]:
                    raise ValueError(
                        "selected RMT head missing from authenticated inventory"
                    )
                files[name] = value
            selected[group][str(seed)] = dict(
                alpha=family["alpha"], update=candidate["update"], files=files
            )
    return dict(
        path=str(path),
        source_commit=protocol["rmt_source_commit"],
        report=artifact(path / "report.json"),
        manifest=artifact(path / "manifest.json"),
        verified=artifact(path / "verified.json"),
        evaluated=artifact(path / "evaluated.json"),
        selections=artifact(path / "selections.json"),
        selected_heads=selected,
    )


def bundle_identity(bundle, commit):
    if bundle is None:
        raise ValueError("immutable deployment requires a Git bundle")
    value = artifact(Path(bundle).resolve())
    subprocess.check_output(
        ["git", "-C", str(SOURCE), "bundle", "verify", value["path"]],
        stderr=subprocess.STDOUT,
    )
    heads = subprocess.check_output(
        ["git", "bundle", "list-heads", value["path"]], text=True
    )
    if not heads.strip() or any(
        line.split()[0] != commit for line in heads.splitlines()
    ):
        raise ValueError("deployment bundle does not bind only the exact source commit")
    return dict(value, source_commit=commit, heads=heads)


def initialize_manifest(
    output, source, parent_cell, dataset, rmt_full, *, upstream=None, bundle=None
):
    output, source = Path(output).resolve(), Path(source).resolve()
    if source != SOURCE.resolve():
        raise ValueError("registration must run from the deployed source")
    protocol = load_protocol()
    identity = source_identity(source)
    upstream = source_identity(Path(upstream or BASE / "dreamerv3-upstream-e3f0224"))
    if upstream["commit"] != protocol["upstream_commit"]:
        raise ValueError("upstream commit differs")
    inputs = _parent_identity(parent_cell, dataset, protocol)
    rmt = _rmt_identity(rmt_full, protocol)
    for dependency in (
        source,
        Path(inputs["parent_cell"]),
        Path(dataset).resolve().parent,
        Path(rmt["path"]),
    ):
        if (
            output == dependency
            or dependency in output.parents
            or output in dependency.parents
        ):
            raise ValueError(
                "fresh output must be disjoint from source and dependency trees"
            )
    runtime = runtime_identity()
    verify_runtime_identity(runtime, runtime)
    manifest = dict(
        schema=protocol["schema"],
        source=str(source),
        source_commit=identity["commit"],
        source_identity=identity,
        upstream=upstream,
        protocol_sha256=object_sha256(protocol),
        inputs=inputs,
        rmt=rmt,
        runtime=runtime,
        runtime_interpreter_profiles=copy.deepcopy(INTERPRETER_PROFILES),
        bundle=bundle_identity(bundle, identity["commit"]),
        cells=[cell_for_index(i) for i in range(CELL_COUNT)],
    )
    output.mkdir(parents=True, exist_ok=False)
    for name in ("markers", "submissions", "logs", "cells", "cache", "attestations"):
        (output / name).mkdir()
    write_json_exclusive(output / "protocol.json", protocol)
    genesis = {}
    for index in range(CELL_COUNT):
        directory = cell_directory(output, index)
        directory.mkdir()
        _initialize_budget(directory, index)
        genesis[str(index)] = artifact(directory / "budget-genesis.json")
    manifest["budget_genesis"] = genesis
    manifest["manifest_sha256"] = object_sha256(manifest)
    write_json_exclusive(output / "manifest.json", manifest)
    return manifest


def authenticate(output):
    output = Path(output).resolve()
    manifest = read(output / "manifest.json")
    body = dict(manifest)
    digest = body.pop("manifest_sha256")
    if object_sha256(body) != digest:
        raise ValueError("manifest digest changed")
    protocol = load_protocol(output / "protocol.json")
    if (
        manifest["protocol_sha256"] != object_sha256(protocol)
        or load_protocol() != protocol
    ):
        raise ValueError("registered protocol changed")
    if (
        manifest["source"] != str(SOURCE.resolve())
        or source_identity(SOURCE) != manifest["source_identity"]
        or manifest["source_commit"] != manifest["source_identity"]["commit"]
    ):
        raise ValueError("exact source checkout changed")
    if (
        source_identity(manifest["upstream"]["path"]) != manifest["upstream"]
        or manifest["upstream"]["commit"] != protocol["upstream_commit"]
    ):
        raise ValueError("upstream dependency changed")
    if manifest["runtime_interpreter_profiles"] != INTERPRETER_PROFILES:
        raise ValueError("runtime profile set changed")
    verify_runtime_identity(manifest["runtime"], runtime_identity())
    if (
        bundle_identity(manifest["bundle"]["path"], manifest["source_commit"])
        != manifest["bundle"]
    ):
        raise ValueError("deployment bundle changed")
    inputs = manifest["inputs"]
    if (
        _parent_identity(inputs["parent_cell"], inputs["dataset"]["path"], protocol)
        != inputs
    ):
        raise ValueError("parent or common retained dataset changed")
    if _rmt_identity(manifest["rmt"]["path"], protocol) != manifest["rmt"]:
        raise ValueError("RMT provenance or selected heads changed")
    if manifest["cells"] != [cell_for_index(i) for i in range(CELL_COUNT)] or set(
        manifest["budget_genesis"]
    ) != {str(i) for i in range(CELL_COUNT)}:
        raise ValueError("cell grid or budgets changed")
    total = 0
    for index in range(CELL_COUNT):
        directory = cell_directory(output, index)
        if manifest["budget_genesis"][str(index)]["path"] != str(
            directory / "budget-genesis.json"
        ):
            raise ValueError("budget path changed")
        verify_artifact(manifest["budget_genesis"][str(index)])
        with NativeStepBudget(directory, index) as budget:
            total += budget.total
    if total > TOTAL_CEILING:
        raise ValueError("aggregate experiment budget exceeded")
    return manifest, protocol


def _budget_genesis(index):
    return dict(
        schema="posterior-reward-budget-v1",
        cell=cell_for_index(index),
        ceiling=CELL_CEILING,
        experiment_ceiling=TOTAL_CEILING,
        action_repeat=2,
    )


def _initialize_budget(directory, index):
    directory = Path(directory)
    write_json_exclusive(directory / "budget-genesis.json", _budget_genesis(index))
    event = dict(
        seq=0,
        previous=ZERO_HASH,
        kind="genesis",
        native_steps=0,
        resets=0,
        total_native_steps=0,
        reset_count=0,
        genesis_sha256=sha(directory / "budget-genesis.json"),
    )
    event["hash"] = object_sha256(event)
    write_bytes_exclusive(directory / "native-budget.jsonl", canonical(event) + b"\n")


class NativeStepBudget:
    """A locked append-only charge ledger; no retry can recreate or refund it.

    Nine disjoint 120k ceilings imply the 1.08m total ceiling, even when cells
    reserve concurrently. One-step reservations permit charging each native call.
    """

    def __init__(self, directory, index):
        self.directory = Path(directory)
        if read(self.directory / "budget-genesis.json") != _budget_genesis(index):
            raise ValueError("budget genesis identity mismatch")
        self.genesis_sha256 = sha(self.directory / "budget-genesis.json")
        self.path = self.directory / "native-budget.jsonl"
        if self.path.is_symlink():
            raise ValueError("budget cannot be a symlink")
        self.stream = self.path.open("r+b")
        self._offset = self._records = self._total = self._resets = 0
        self._head, self._last_line = ZERO_HASH, b""
        self._inode = os.fstat(self.stream.fileno()).st_ino
        try:
            self.snapshot()
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.stream.close()

    def _sync(self):
        stat = os.fstat(self.stream.fileno())
        if (
            self.path.is_symlink()
            or stat.st_ino != self._inode
            or self.path.stat().st_ino != self._inode
            or stat.st_size < self._offset
        ):
            raise ValueError("budget ledger replaced or truncated")
        if self._last_line:
            self.stream.seek(self._offset - len(self._last_line))
            if self.stream.read(len(self._last_line)) != self._last_line:
                raise ValueError("budget tail changed")
        self.stream.seek(self._offset)
        for line in self.stream:
            if not line.endswith(b"\n"):
                raise ValueError("partial budget record")
            event = json.loads(line, object_pairs_hook=_unique_pairs)
            digest = event.pop("hash")
            if (
                event.get("seq") != self._records
                or event.get("previous") != self._head
                or object_sha256(event) != digest
            ):
                raise ValueError("broken budget hash chain")
            steps, resets = event["native_steps"], event["resets"]
            if (
                type(steps) is not int
                or type(resets) is not int
                or steps < 0
                or resets < 0
            ):
                raise ValueError("invalid budget increment")
            if self._records == 0:
                if event != dict(
                    seq=0,
                    previous=ZERO_HASH,
                    kind="genesis",
                    native_steps=0,
                    resets=0,
                    total_native_steps=0,
                    reset_count=0,
                    genesis_sha256=self.genesis_sha256,
                ):
                    raise ValueError("budget genesis was reset")
            elif (
                event["kind"] not in ("collect", "evaluate", "reset")
                or (event["kind"] == "reset" and (steps or not resets))
                or (event["kind"] != "reset" and (steps <= 0 or resets))
            ):
                raise ValueError("invalid budget event kind")
            total, reset_count = self._total + steps, self._resets + resets
            if (
                event["total_native_steps"] != total
                or event["reset_count"] != reset_count
                or total > CELL_CEILING
            ):
                raise ValueError("nonmonotonic or exceeded native-step budget")
            self._total, self._resets, self._head = total, reset_count, digest
            self._records += 1
            self._offset += len(line)
            self._last_line = line
        if not self._records:
            raise ValueError("empty budget cannot be reset")

    def _append(self, steps, resets, kind, detail):
        fcntl.flock(self.stream, fcntl.LOCK_EX)
        try:
            self._sync()
            if self._total + steps > CELL_CEILING:
                raise ValueError(
                    "native-step budget would exceed the 120000 hard ceiling"
                )
            event = dict(
                seq=self._records,
                previous=self._head,
                kind=kind,
                native_steps=steps,
                resets=resets,
                total_native_steps=self._total + steps,
                reset_count=self._resets + resets,
                detail=detail,
                timestamp_ns=time.time_ns(),
            )
            event["hash"] = object_sha256(event)
            self.stream.seek(0, os.SEEK_END)
            self.stream.write(canonical(event) + b"\n")
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self._sync()
            return event
        finally:
            fcntl.flock(self.stream, fcntl.LOCK_UN)

    def reserve(self, native_steps, kind="collect", detail=None):
        if (
            type(native_steps) is not int
            or native_steps <= 0
            or kind not in ("collect", "evaluate")
        ):
            raise ValueError(
                "reserve a positive integer native-step count and registered kind"
            )
        return self._append(native_steps, 0, kind, detail)

    def record_reset(self, count=1, detail=None):
        if type(count) is not int or count <= 0:
            raise ValueError("reset count must be a positive integer")
        return self._append(0, count, "reset", detail)

    def snapshot(self):
        fcntl.flock(self.stream, fcntl.LOCK_SH)
        try:
            self._sync()
            return dict(
                total_native_steps=self._total,
                reset_count=self._resets,
                records=self._records,
                head=self._head,
                ceiling=CELL_CEILING,
            )
        finally:
            fcntl.flock(self.stream, fcntl.LOCK_UN)

    @property
    def total(self):
        return self.snapshot()["total_native_steps"]


def _stage(stage):
    if stage not in ("preflight", "probes", "finalize") and not re.fullmatch(
        r"cell-[0-8]", stage
    ):
        raise ValueError("unregistered marker stage")
    return stage


def submission_reference(output, stage, manifest, *, runtime=False):
    output = Path(output)
    submission_stage = "cells" if stage.startswith("cell-") else stage
    receipt_path = output / "submissions" / f"{submission_stage}.json"
    intent_path = output / "submissions" / f"{submission_stage}-intent.json"
    receipt, intent = read(receipt_path), read(intent_path)
    if (
        receipt.get("stage") != submission_stage
        or receipt.get("status") != "submitted"
        or receipt.get("manifest_sha256") != manifest["manifest_sha256"]
        or receipt.get("intent_sha256") != sha(intent_path)
        or intent.get("stage") != submission_stage
        or intent.get("manifest_sha256") != manifest["manifest_sha256"]
        or receipt.get("command") != intent.get("command")
        or not re.fullmatch(r"[1-9][0-9]*", receipt.get("job", ""))
    ):
        raise ValueError("marker submission identity mismatch")
    if runtime:
        job = os.environ.get(
            "SLURM_ARRAY_JOB_ID" if stage.startswith("cell-") else "SLURM_JOB_ID"
        )
        if job != receipt["job"]:
            raise ValueError("running scheduler job differs from receipt")
        index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if stage.startswith("cell-") and index != stage[5:]:
            raise ValueError("running array index differs")
    return dict(
        stage=submission_stage,
        job=receipt["job"],
        receipt_sha256=sha(receipt_path),
        intent_sha256=sha(intent_path),
    )


def write_marker(output, stage, artifacts, details=None):
    output, stage = Path(output).resolve(), _stage(stage)
    manifest, _ = authenticate(output)
    files = {}
    for path in artifacts:
        relative = _inside(output, path)
        if relative in files:
            raise ValueError("duplicate marker artifact")
        files[relative] = dict(artifact(path), path=relative)
    if not files:
        raise ValueError("completion marker requires evidence")
    value = dict(
        stage=stage,
        manifest_sha256=manifest["manifest_sha256"],
        source_commit=manifest["source_commit"],
        artifacts=files,
        details=details or {},
        submission=submission_reference(output, stage, manifest, runtime=True),
    )
    if stage.startswith("cell-"):
        index = int(stage[5:])
        with NativeStepBudget(cell_directory(output, index), index) as budget:
            value["budget"] = budget.snapshot()
    value["marker_sha256"] = object_sha256(value)
    write_json_exclusive(output / "markers" / f"{stage}.json", value)
    return value


def require_marker(output, stage, *, authenticate_source=True):
    """Offline verification checks historical submission provenance, not current job."""
    output, stage = Path(output).resolve(), _stage(stage)
    manifest = (
        authenticate(output)[0]
        if authenticate_source
        else read(output / "manifest.json")
    )
    value = read(output / "markers" / f"{stage}.json")
    body = dict(value)
    digest = body.pop("marker_sha256")
    if (
        object_sha256(body) != digest
        or value["stage"] != stage
        or value["manifest_sha256"] != manifest["manifest_sha256"]
        or value["source_commit"] != manifest["source_commit"]
        or not value["artifacts"]
        or value["submission"] != submission_reference(output, stage, manifest)
    ):
        raise ValueError("marker identity or digest mismatch")
    for relative, record in value["artifacts"].items():
        path = output / relative
        if _inside(output, path) != relative or record["path"] != relative:
            raise ValueError("marker artifact escapes output root")
        verify_artifact(dict(record, path=str(path)))
    if stage.startswith("cell-"):
        index = int(stage[5:])
        with NativeStepBudget(cell_directory(output, index), index) as budget:
            if budget.snapshot() != value["budget"]:
                raise ValueError("native-step ledger changed after cell verification")
    return value
