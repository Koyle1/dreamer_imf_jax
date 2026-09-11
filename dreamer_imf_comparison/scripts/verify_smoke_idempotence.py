#!/usr/bin/env python3
"""Prove that a completed smoke run is immutable and deterministically reusable."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
for path in (PROJECT, WORKSPACE / "imf_dreamer_jax" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dreamer_imf_compare.artifacts import read_json  # noqa: E402
from dreamer_imf_compare.matched_objective_benchmark import (  # noqa: E402
    stage_directory,
    verify_output_root,
)


RERUN_TIMEOUT_SECONDS = 900


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _cell_snapshot(root: Path, matrix: dict[str, Any]) -> dict[str, tuple[int, int, str]]:
    snapshot: dict[str, tuple[int, int, str]] = {}
    for cell in matrix["cells"]:
        directory = stage_directory(root, cell)
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                stat = path.stat()
                relative = str(path.relative_to(root))
                snapshot[relative] = (stat.st_size, stat.st_mtime_ns, _digest(path))
    if not snapshot:
        raise AssertionError("the smoke root contains no retained cell artifacts")
    return snapshot


def _retained_snapshot(root: Path) -> dict[str, tuple[int, str]]:
    """Bind the exact retained path set, sizes, and bytes (but not root mtimes)."""

    if not root.is_dir():
        raise FileNotFoundError("smoke output root does not exist")
    snapshot: dict[str, tuple[int, str]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AssertionError(f"retained smoke output contains symlink: {path}")
        if path.is_file():
            stat = path.stat()
            snapshot[path.relative_to(root).as_posix()] = (stat.st_size, _digest(path))
    if not snapshot:
        raise AssertionError("the smoke root contains no retained artifacts")
    return snapshot


def _assert_retained_snapshot_equal(
    before: dict[str, tuple[int, str]], after: dict[str, tuple[int, str]]
) -> None:
    if before == after:
        return
    before_paths = set(before)
    after_paths = set(after)
    added = sorted(after_paths - before_paths)
    removed = sorted(before_paths - after_paths)
    changed = sorted(
        path for path in before_paths & after_paths if before[path] != after[path]
    )
    raise AssertionError(
        "rerun changed the exact retained path/hash/size set; "
        f"added={added}, removed={removed}, changed={changed}"
    )


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    additions = [str(WORKSPACE / "imf_dreamer_jax" / "src"), str(PROJECT)]
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        additions + ([previous] if previous else [])
    )
    return environment


def _rerun_smoke(
    root: Path, workspace: Path, *, timeout_seconds: int = RERUN_TIMEOUT_SECONDS
) -> None:
    try:
        subprocess.run(
            [
                sys.executable,
                str(PROJECT / "scripts" / "run_matched_objective_benchmark.py"),
                "all",
                "--profile",
                "smoke",
                "--output",
                str(root),
                "--workspace",
                str(workspace),
            ],
            cwd=WORKSPACE,
            env=_environment(),
            check=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"smoke idempotence rerun timed out after {timeout_seconds} seconds"
        ) from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workspace", default=str(WORKSPACE))
    arguments = parser.parse_args()
    root = Path(arguments.output).resolve()
    workspace = Path(arguments.workspace).resolve()
    before_verification = verify_output_root(root, workspace=workspace)
    matrix_path = root / "matrix.json"
    before_matrix = read_json(matrix_path)
    if before_matrix.get("profile") != "smoke" or before_matrix.get("claim_eligible") is not False:
        raise AssertionError("idempotence verifier only accepts a non-claim smoke matrix")
    before_matrix_digest = _digest(matrix_path)
    before_retained = _retained_snapshot(root)
    before_cells = _cell_snapshot(root, before_matrix)
    before_analysis = before_verification["analysis_sha256"]

    _rerun_smoke(root, workspace)
    after_verification = verify_output_root(root, workspace=workspace)
    after_matrix = read_json(matrix_path)
    if after_matrix != before_matrix or _digest(matrix_path) != before_matrix_digest:
        raise AssertionError("rerun changed the frozen matrix")
    after_retained = _retained_snapshot(root)
    _assert_retained_snapshot_equal(before_retained, after_retained)
    after_cells = _cell_snapshot(root, after_matrix)
    if before_cells != after_cells:
        raise AssertionError(
            "rerun changed an immutable cell artifact hash, size, path, or mtime"
        )
    if after_verification["analysis_sha256"] != before_analysis:
        raise AssertionError("rerun changed the deterministic analysis digest")
    print("MATCHED_OBJECTIVE_SMOKE_IDEMPOTENCE_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
