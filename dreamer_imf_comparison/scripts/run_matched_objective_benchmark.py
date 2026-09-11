#!/usr/bin/env python3
"""Freeze, execute, finalize, and verify the matched-objective benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
for path in (PROJECT, WORKSPACE / "imf_dreamer_jax" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dreamer_imf_compare.artifacts import read_json  # noqa: E402
from dreamer_imf_compare.matched_objective_benchmark import (  # noqa: E402
    HPO_MATRIX_SCHEMA,
    STAGE_ORDER,
    finalize_pilot_hpo_run,
    finalize_run,
    freeze_pilot_hpo_run,
    freeze_run,
    run_cell,
    run_pending_cells,
    validate_matrix,
    validate_pilot_hpo_matrix,
    validate_source_manifest,
    verify_pilot_hpo_output_root,
    verify_output_root,
)
from dreamer_imf_compare.matched_objective_protocol import (  # noqa: E402
    read_matched_objective_protocol,
)


def _paths(arguments: argparse.Namespace) -> tuple[Path, Path, Path]:
    protocol = Path(arguments.protocol or PROJECT / "matched_objective_protocol.json").resolve()
    output = Path(
        arguments.output
        or PROJECT / "results" / f"matched_objective_{arguments.profile}"
    ).resolve()
    workspace = Path(arguments.workspace or WORKSPACE).resolve()
    return protocol, output, workspace


def _freeze(arguments: argparse.Namespace, protocol: dict, output: Path, workspace: Path):
    if arguments.profile == "pilot":
        return freeze_pilot_hpo_run(protocol, output, workspace=workspace)
    selection = None
    if arguments.profile == "confirmatory":
        if arguments.selection:
            raise ValueError(
                "a standalone --selection is insufficient for a claim; use --pilot-output"
            )
        if not arguments.pilot_output:
            raise ValueError("confirmatory freeze requires --pilot-output")
        pilot_output = Path(arguments.pilot_output).resolve()
        verify_pilot_hpo_output_root(pilot_output, workspace=workspace)
        selection = read_json(pilot_output / "hpo_selection.json")
    return freeze_run(
        protocol,
        arguments.profile,
        output,
        workspace=workspace,
        selection_manifest=selection,
    )


def _load_frozen(output: Path) -> tuple[dict, dict, dict]:
    return (
        read_json(output / "frozen_protocol.json"),
        read_json(output / "source_manifest.json"),
        read_json(output / "matrix.json"),
    )


def _validate_frozen(protocol: dict, source: dict, matrix: dict, workspace: Path) -> None:
    validate_source_manifest(source, workspace)
    if matrix.get("schema_version") == HPO_MATRIX_SCHEMA:
        validate_pilot_hpo_matrix(matrix, protocol, source)
    else:
        validate_matrix(matrix, protocol, source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "finalize", "verify", "all"))
    parser.add_argument("--profile", choices=("smoke", "pilot", "confirmatory"), default="smoke")
    parser.add_argument("--protocol")
    parser.add_argument("--output")
    parser.add_argument("--workspace")
    parser.add_argument("--selection", help="completed pilot hpo_selection.json")
    parser.add_argument("--pilot-output", help="verified completed pilot output root")
    parser.add_argument("--cell-id")
    parser.add_argument(
        "--stages",
        default=",".join(STAGE_ORDER),
        help="comma-separated dependency-ordered stages for the run command",
    )
    arguments = parser.parse_args()
    protocol_path, output, workspace = _paths(arguments)

    if arguments.command == "verify":
        print(json.dumps(verify_output_root(output, workspace=workspace), indent=2, sort_keys=True))
        print("MATCHED_OBJECTIVE_BENCHMARK_VERIFIED")
        return 0

    requested_protocol = read_matched_objective_protocol(protocol_path)
    if arguments.command == "freeze":
        _, matrix = _freeze(arguments, requested_protocol, output, workspace)
        print(json.dumps({"output": str(output), "cells": len(matrix["cells"]), "matrix_sha256": matrix["matrix_sha256"]}, indent=2, sort_keys=True))
        print("MATCHED_OBJECTIVE_BENCHMARK_FROZEN")
        return 0

    if arguments.command == "all" and not (output / "matrix.json").is_file():
        _freeze(arguments, requested_protocol, output, workspace)
    protocol, source, matrix = _load_frozen(output)
    if protocol != requested_protocol:
        raise ValueError("requested protocol differs from the output root's frozen protocol")
    if matrix["profile"] != arguments.profile:
        raise ValueError("requested profile differs from the frozen matrix")
    _validate_frozen(protocol, source, matrix, workspace)

    if arguments.command in ("run", "all"):
        if arguments.cell_id:
            matching = [cell for cell in matrix["cells"] if cell["cell_id"] == arguments.cell_id]
            if len(matching) != 1:
                raise ValueError("--cell-id is absent from or duplicated in the frozen matrix")
            result = run_cell(matching[0], protocol, matrix, output)
            print(json.dumps({"cell_id": result["cell_id"], "stage": result["stage"]}, sort_keys=True))
        else:
            stages = tuple(value.strip() for value in arguments.stages.split(",") if value.strip())
            counts = run_pending_cells(protocol, matrix, output, stages=stages)
            print(json.dumps({"executed_or_verified": counts}, indent=2, sort_keys=True))
        if arguments.command == "run":
            print("MATCHED_OBJECTIVE_CELLS_COMPLETE")
            return 0

    if arguments.command in ("finalize", "all"):
        if matrix.get("schema_version") == HPO_MATRIX_SCHEMA:
            summary = finalize_pilot_hpo_run(
                output, matrix, protocol, workspace=workspace
            )
        else:
            summary = finalize_run(
                output, matrix, protocol, workspace=workspace
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
        if arguments.command == "finalize":
            print("MATCHED_OBJECTIVE_FINALIZED")
            return 0

    print(json.dumps(verify_output_root(output, workspace=workspace), indent=2, sort_keys=True))
    print("MATCHED_OBJECTIVE_BENCHMARK_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
