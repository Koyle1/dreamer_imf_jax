"""Frozen pilot identity, immutable evidence and conservative native-step charges.

No environment is created here. A reservation is persisted *before* a native
environment call; failure never refunds it. Only registration creates ledgers.
Submission and completion records are write-once, including failed attempts.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

SOURCE = Path(__file__).resolve().parents[2]
BASE = Path("/work2/ci72buri-dreamer_imf_neurips")
PROTOCOL = SOURCE / "dreamer_imf_comparison/reward_exploration_protocol.json"
PROTOCOL_DIGEST = "7f22851257202d3bdbe039424c647edba3902fa09b2dd3f8ae385183323f35fe"
PARENT_SHA256 = "7469bc2dd4e9129b964f3d25ab8a0be0926ea2444b2917bfabc1e244a53ea717"
DATASET_SHA256 = "5707350751eb0ad60c04406f873f6ef49aadf1843eb9a9431df452979a742b9f"
ARMS = ("A", "B", "C", "D")
SEEDS = (701, 702, 703)
STAGES = ("match", "preflight", "cells", "finalize")
ZERO_HASH = "0" * 64
# Architecture-specific module builds share one path. Read-only probes 28277646
# (A30) and 28277653 (CPU handoff) bind the compute build explicitly. Installed
# Python/CUDA package bytes must still match; this is not version-only checking.
INTERPRETER_PROFILES = [
    {
        "python_version": "3.12.3 (main, Nov 17 2025, 14:11:43) [GCC 13.3.0]",
        "executable": {
            "path": "/software/all/Python/3.12.3-GCCcore-13.3.0/bin/python3.12",
            "sha256": "1fb797ac65a3820ee5c328e5d3b32843c89a09e9b027e3d469b5d21810ee5632",
            "bytes": 21136,
        },
    },
    {
        "python_version": "3.12.3 (main, Nov 17 2025, 12:47:01) [GCC 13.3.0]",
        "executable": {
            "path": "/software/all/Python/3.12.3-GCCcore-13.3.0/bin/python3.12",
            "sha256": "f3a23874360698d88c850604a1405aac16b19ce986ee1b2940e214d81309eae2",
            "bytes": 21096,
        },
    },
]


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite JSON evidence")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON keys must be strings")
            _finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite(item)


def canonical(value):
    _finite(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def object_sha256(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read(path):
    value = json.loads(Path(path).read_text(), object_pairs_hook=_unique_pairs)
    _finite(value)
    return value


def sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact must be a regular, non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_dir(directory):
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_bytes_exclusive(path, data):
    """Atomic publish with link(2), which cannot replace existing evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_dir(path.parent)
    finally:
        os.unlink(temporary)


def write_json_exclusive(path, value):
    write_bytes_exclusive(path, canonical(value) + b"\n")


def validate_protocol(value):
    if object_sha256(value) != PROTOCOL_DIGEST:
        raise ValueError("protocol differs from the frozen registered pilot")
    return copy.deepcopy(value)


def load_protocol(path=None):
    return validate_protocol(read(PROTOCOL if path is None else path))


def cell_for_index(index):
    if type(index) is not int or not 0 <= index < 12:
        raise ValueError("cell index must be an integer from 0 to 11")
    arm, seed = ARMS[index // 3], SEEDS[index % 3]
    return dict(index=index, arm=arm, seed=seed, cell_id=f"{arm}-seed{seed}")


def cell_directory(output, index):
    return Path(output) / "cells" / cell_for_index(index)["cell_id"]


def _seed(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("unregistered continuation seed")


def exploration_block(seed, cycle):
    _seed(seed)
    if type(cycle) is not int or cycle < 0:
        raise ValueError("invalid exploration cycle")
    # Hashing is deterministic across hosts/Python versions; no mutable RNG.
    digest = hashlib.sha256(f"reward-exploration:{seed}:{cycle}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 5


def collection_policy(arm, seed, decision_index):
    _seed(seed)
    if arm not in ARMS or type(decision_index) is not int or decision_index < 0:
        raise ValueError("unregistered arm or decision index")
    block = decision_index // 20
    explore = block % 5 == exploration_block(seed, block // 5)
    if arm == "C" and explore:
        return "random"
    if arm == "D" and explore:
        return "explore"
    return "task"


def eval_seeds(seed, milestone):
    _seed(seed)
    if type(milestone) is not int or milestone not in (40000, 80000):
        raise ValueError("unregistered evaluation milestone")
    return [1000000 + seed * 10000 + milestone + index for index in range(5)]


def _inside(root, path):
    root, path = Path(root).resolve(), Path(path)
    if path.is_symlink():
        raise ValueError("symlink evidence is not accepted")
    path = path.resolve()
    if root not in path.parents:
        raise ValueError("artifact escapes the fresh output root")
    return str(path.relative_to(root))


def artifact(path):
    path = Path(path).absolute()
    if path.suffix == ".json":
        read(path)
    elif path.suffix == ".npz":
        import numpy as np

        with np.load(path, allow_pickle=False) as arrays:
            for name in arrays.files:
                value = arrays[name]
                if value.dtype.kind in "fc" and not np.isfinite(value).all():
                    raise ValueError(f"nonfinite array artifact: {path}:{name}")
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def verify_artifact(value):
    if set(value) != {"path", "sha256", "bytes"} or artifact(value["path"]) != value:
        raise ValueError("artifact content, size or identity changed")


def clean_commit(path):
    path = Path(path).resolve()
    status = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    )
    if status.strip():
        raise ValueError(f"source checkout is dirty: {path}")
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("invalid full source commit")
    return commit


def source_identity(path):
    path = Path(path).resolve()
    commit = clean_commit(path)
    names = (
        subprocess.check_output(["git", "-C", str(path), "ls-files", "-z"])
        .decode()
        .split("\0")
    )
    ignored = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(path),
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "-z",
            ]
        )
        .decode()
        .split("\0")
    )
    if any(
        name and (Path(name).suffix in (".py", ".so", ".pth", ".pyd") or ".so." in name)
        for name in ignored
    ):
        raise ValueError("ignored executable source can bypass the exact checkout")
    files = {name: sha(path / name) for name in names if name}
    if not files:
        raise ValueError("empty source checkout")
    return dict(
        path=str(path), commit=commit, files=files, tree_sha256=object_sha256(files)
    )


def runtime_identity():
    packages = (
        "numpy",
        "jax",
        "jaxlib",
        "optax",
        "dm-control",
        "mujoco",
        "PyYAML",
        "elements",
        "ninjax",
        "portal",
        "ruamel.yaml",
        "einops",
        "chex",
        "scipy",
    )
    # CUDA wheel code is not part of jaxlib's file inventory. Include every
    # installed CUDA plugin/library distribution, without requiring GPUs on
    # the offline test host.
    gpu_packages = sorted(
        distribution.metadata["Name"]
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"].lower().startswith(("jax-cuda", "nvidia-"))
    )
    versions = {}
    for package in (*packages, *gpu_packages):
        distribution = importlib.metadata.distribution(package)
        metadata = distribution.read_text("METADATA")
        if not metadata:
            raise ValueError(f"missing installed dependency metadata: {package}")
        files = distribution.files
        if not files:
            raise ValueError(f"missing installed dependency file inventory: {package}")
        # Version strings/METADATA alone do not detect an edited installed .py
        # or shared library. Bind actual distribution bytes, ignoring only
        # interpreter-generated bytecode whose presence may change on import.
        hashes = {}
        for item in files:
            if (
                str(item).endswith((".pyc", ".pyo"))
                or "__pycache__" in Path(str(item)).parts
            ):
                continue
            installed = Path(distribution.locate_file(item)).resolve()
            hashes[str(item)] = sha(installed)
        versions[package] = dict(
            version=distribution.version,
            metadata_sha256=hashlib.sha256(metadata.encode()).hexdigest(),
            files_sha256=object_sha256(hashes),
            file_count=len(hashes),
        )
    return dict(
        python_version=sys.version,
        executable=artifact(Path(sys.executable).resolve()),
        packages=versions,
    )


def verify_runtime_identity(registered, current):
    required = {"python_version", "executable", "packages"}
    if set(registered) != required or set(current) != required:
        raise ValueError("unexpected runtime identity fields")
    for runtime in (registered, current):
        interpreter = {key: runtime[key] for key in required - {"packages"}}
        if interpreter not in INTERPRETER_PROFILES:
            raise ValueError("unregistered interpreter binary/build")
    if registered["packages"] != current["packages"]:
        raise ValueError("installed runtime dependency identity changed")


def _parent_identity(parent_cell, dataset, protocol):
    parent_cell, dataset = Path(parent_cell).resolve(), Path(dataset).resolve()
    checkpoint = parent_cell / protocol["parent"]["checkpoint_name"]
    if sha(checkpoint) != PARENT_SHA256 or sha(dataset) != DATASET_SHA256:
        raise ValueError(
            "parent checkpoint or common dataset is not the pinned artifact"
        )
    complete = read(parent_cell / "complete.json")
    if (
        complete.get("completed") is not True
        or complete.get("seed") != 431
        or complete.get("arm") != "imf"
    ):
        raise ValueError("parent completion identity mismatch")
    matches = [
        row
        for row in complete.get("checkpoints", [])
        if row.get("native_steps") == 200000
    ]
    if (
        len(matches) != 1
        or matches[0].get("sha256") != PARENT_SHA256
        or matches[0].get("path") != checkpoint.name
    ):
        raise ValueError(
            "parent completion does not authenticate the pinned checkpoint"
        )
    return dict(
        parent_cell=str(parent_cell),
        dataset=artifact(dataset),
        parent_files=[
            artifact(parent_cell / name)
            for name in (
                checkpoint.name,
                "config.yaml",
                "complete.json",
                "replay_sample_200000_48985.npz",
            )
        ],
    )


def initialize_manifest(
    output, source, parent_cell, dataset, dependencies=None, *, upstream=None
):
    """Register once; a partial registration stays visible and is never reset."""
    output, source = Path(output).resolve(), Path(source).resolve()
    if source != SOURCE.resolve():
        raise ValueError("registration must run from the exact deployed source")
    protocol = load_protocol()
    identity = source_identity(source)
    upstream = Path(upstream or BASE / "dreamerv3-upstream-e3f0224").resolve()
    upstream_identity = source_identity(upstream)
    if upstream_identity["commit"] != protocol["upstream_commit"]:
        raise ValueError("upstream source is not the frozen commit")
    inputs = _parent_identity(parent_cell, dataset, protocol)
    dependency_files = {
        name: artifact(path) for name, path in sorted((dependencies or {}).items())
    }
    runtime = runtime_identity()
    verify_runtime_identity(runtime, runtime)
    manifest = dict(
        schema=protocol["schema"],
        source=str(source),
        source_commit=identity["commit"],
        source_identity=identity,
        upstream=upstream_identity,
        protocol_sha256=object_sha256(protocol),
        inputs=inputs,
        dependencies=dependency_files,
        runtime=runtime,
        runtime_interpreter_profiles=copy.deepcopy(INTERPRETER_PROFILES),
        cells=[cell_for_index(i) for i in range(12)],
    )
    output.mkdir(parents=True, exist_ok=False)
    for name in ("markers", "submissions", "logs", "cells", "cache"):
        (output / name).mkdir()
    write_json_exclusive(output / "protocol.json", protocol)
    genesis = {}
    for index in range(12):
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
    claimed = body.pop("manifest_sha256")
    if object_sha256(body) != claimed:
        raise ValueError("manifest digest changed")
    protocol = load_protocol(output / "protocol.json")
    if (
        manifest["protocol_sha256"] != object_sha256(protocol)
        or load_protocol() != protocol
    ):
        raise ValueError("registered protocol differs")
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
    if manifest.get("runtime_interpreter_profiles") != INTERPRETER_PROFILES:
        raise ValueError("registered interpreter profile set changed")
    verify_runtime_identity(manifest["runtime"], runtime_identity())
    inputs = manifest["inputs"]
    if (
        _parent_identity(inputs["parent_cell"], inputs["dataset"]["path"], protocol)
        != inputs
    ):
        raise ValueError("parent or dataset evidence changed")
    for dependency in manifest["dependencies"].values():
        verify_artifact(dependency)
    if manifest["cells"] != [cell_for_index(i) for i in range(12)] or set(
        manifest["budget_genesis"]
    ) != {str(i) for i in range(12)}:
        raise ValueError("cell grid or budget registrations changed")
    for index in range(12):
        expected = cell_directory(output, index) / "budget-genesis.json"
        if manifest["budget_genesis"][str(index)]["path"] != str(expected):
            raise ValueError("budget path changed")
        verify_artifact(manifest["budget_genesis"][str(index)])
        with NativeStepBudget(cell_directory(output, index), index):
            pass
    return manifest, protocol


def _initialize_budget(directory, index):
    directory = Path(directory)
    cell = cell_for_index(index)
    genesis = dict(
        schema="reward-exploration-budget-v1",
        cell=cell,
        ceiling=100000,
        action_repeat=2,
    )
    write_json_exclusive(directory / "budget-genesis.json", genesis)
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
    """Locked append-only ledger. Reopening never recreates a missing ledger."""

    def __init__(self, cell_dir, index):
        self.directory = Path(cell_dir)
        self.index = index
        expected = dict(
            schema="reward-exploration-budget-v1",
            cell=cell_for_index(index),
            ceiling=100000,
            action_repeat=2,
        )
        if read(self.directory / "budget-genesis.json") != expected:
            raise ValueError("budget genesis identity mismatch")
        self.genesis_sha256 = sha(self.directory / "budget-genesis.json")
        path = self.directory / "native-budget.jsonl"
        if path.is_symlink():
            raise ValueError("budget cannot be a symlink")
        self.stream = path.open("r+b")  # no create, no truncate, including retries
        self._offset, self._records, self._total, self._resets = 0, 0, 0, 0
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
        path_stat = (self.directory / "native-budget.jsonl").stat()
        if (
            stat.st_ino != self._inode
            or path_stat.st_ino != self._inode
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
                raise ValueError("partial budget record; fail closed")
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
                or (event["kind"] != "reset" and (steps <= 0 or steps % 2 or resets))
            ):
                raise ValueError("invalid budget event kind or action-repeat increment")
            total, reset_count = self._total + steps, self._resets + resets
            if (
                event["total_native_steps"] != total
                or event["reset_count"] != reset_count
                or total > 100000
            ):
                raise ValueError("nonmonotonic or exceeded native-step budget")
            self._total, self._resets, self._head = total, reset_count, digest
            self._records += 1
            self._offset += len(line)
            self._last_line = line
        if not self._records:
            raise ValueError("empty budget ledger cannot be reset")

    def _append(self, steps, resets, kind, detail):
        fcntl.flock(self.stream, fcntl.LOCK_EX)
        try:
            self._sync()
            if self._total + steps > 100000:
                raise ValueError(
                    "native-step budget would exceed the 100000 hard ceiling"
                )
            value = dict(
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
            value["hash"] = object_sha256(value)
            self.stream.seek(0, os.SEEK_END)
            self.stream.write(canonical(value) + b"\n")
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self._sync()
            return value
        finally:
            fcntl.flock(self.stream, fcntl.LOCK_UN)

    def reserve(self, native_steps, kind="collect", detail=None):
        if (
            type(native_steps) is not int
            or native_steps <= 0
            or native_steps % 2
            or kind not in ("collect", "evaluate")
        ):
            raise ValueError(
                "reserve a positive even native-step count and registered kind"
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
                ceiling=100000,
            )
        finally:
            fcntl.flock(self.stream, fcntl.LOCK_UN)

    @property
    def total(self):
        return self.snapshot()["total_native_steps"]


def _stage(stage):
    if stage not in ("match", "preflight", "finalize") and not re.fullmatch(
        r"cell-(0[0-9]|1[01])", stage
    ):
        raise ValueError("unregistered marker stage")
    return stage


def _submission_reference(output, stage, manifest, *, runtime=False):
    submission_stage = "cells" if stage.startswith("cell-") else stage
    receipt_path = output / "submissions" / f"{submission_stage}.json"
    intent_path = output / "submissions" / f"{submission_stage}-intent.json"
    receipt, intent = read(receipt_path), read(intent_path)
    if (
        receipt.get("stage") != submission_stage
        or receipt.get("status") != "submitted"
        or receipt.get("manifest_sha256") != manifest["manifest_sha256"]
        or receipt.get("intent_sha256") != sha(intent_path)
    ):
        raise ValueError("marker lacks an authenticated successful submission receipt")
    if (
        intent.get("stage") != submission_stage
        or intent.get("manifest_sha256") != manifest["manifest_sha256"]
        or receipt.get("command") != intent.get("command")
        or not re.fullmatch(r"[1-9][0-9]*", receipt.get("job", ""))
    ):
        raise ValueError("marker submission identity changed")
    if runtime:
        job = (
            os.environ.get("SLURM_ARRAY_JOB_ID")
            if stage.startswith("cell-")
            else os.environ.get("SLURM_JOB_ID")
        )
        if job is not None and job != receipt["job"]:
            raise ValueError("running scheduler job differs from submission receipt")
        index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if (
            stage.startswith("cell-")
            and index is not None
            and index != str(int(stage[5:]))
        ):
            raise ValueError("running array index differs from verified cell")
    return dict(
        stage=submission_stage,
        job=receipt["job"],
        receipt_sha256=sha(receipt_path),
        intent_sha256=sha(intent_path),
    )


def write_marker(output, stage, artifacts, details=None):
    """Called only after separate-process independent verification succeeds."""
    output, stage = Path(output).resolve(), _stage(stage)
    manifest, _ = authenticate(output)
    files = {}
    for path in artifacts:
        relative = _inside(output, path)
        record = artifact(path)
        record["path"] = relative
        files[relative] = record
    if not files:
        raise ValueError("completion marker requires nonempty artifact evidence")
    submission = _submission_reference(output, stage, manifest, runtime=True)
    value = dict(
        stage=stage,
        manifest_sha256=manifest["manifest_sha256"],
        source_commit=manifest["source_commit"],
        artifacts=files,
        details=details or {},
        submission=submission,
    )
    if stage.startswith("cell-"):
        index = int(stage[5:])
        with NativeStepBudget(cell_directory(output, index), index) as budget:
            value["budget"] = budget.snapshot()
    value["marker_sha256"] = object_sha256(value)
    write_json_exclusive(output / "markers" / f"{stage}.json", value)
    return value


def require_marker(output, stage, *, authenticate_source=True):
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
    ):
        raise ValueError("marker identity or digest mismatch")
    if not value["artifacts"]:
        raise ValueError("empty completion evidence")
    if value.get("submission") != _submission_reference(output, stage, manifest):
        raise ValueError("marker and scheduler receipt provenance differ")
    for relative, record in value["artifacts"].items():
        path = output / relative
        if _inside(output, path) != relative or record["path"] != relative:
            raise ValueError("marker artifact escaped root")
        verify_artifact(dict(record, path=str(path)))
    if stage.startswith("cell-"):
        index = int(stage[5:])
        with NativeStepBudget(cell_directory(output, index), index) as budget:
            if budget.snapshot() != value["budget"]:
                raise ValueError("native-step ledger changed after cell completion")
    return value
