#!/usr/bin/env python3
"""Generate a source-bound matched-objective matrix without executing cells."""

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

from dreamer_imf_compare.artifacts import read_json, write_json_atomic  # noqa: E402
from dreamer_imf_compare.matched_objective_benchmark import (  # noqa: E402
    build_matrix,
    build_pilot_hpo_matrix,
    build_source_manifest,
    expected_hpo_matrix_counts,
    expected_matrix_counts,
    verify_pilot_hpo_output_root,
)
from dreamer_imf_compare.matched_objective_protocol import (  # noqa: E402
    read_matched_objective_protocol,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "pilot", "confirmatory"), required=True)
    parser.add_argument("--protocol", default=str(PROJECT / "matched_objective_protocol.json"))
    parser.add_argument("--workspace", default=str(WORKSPACE))
    parser.add_argument("--selection")
    parser.add_argument("--pilot-output")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    protocol = read_matched_objective_protocol(Path(arguments.protocol).resolve())
    workspace = Path(arguments.workspace).resolve()
    source = build_source_manifest(workspace)
    if arguments.profile == "pilot":
        matrix = build_pilot_hpo_matrix(protocol, source_manifest=source)
        counts = expected_hpo_matrix_counts(protocol)
    else:
        if arguments.profile == "confirmatory":
            if arguments.selection:
                raise ValueError(
                    "a standalone --selection is insufficient for a claim; use --pilot-output"
                )
            if not arguments.pilot_output:
                raise ValueError("confirmatory matrix generation requires --pilot-output")
            pilot_output = Path(arguments.pilot_output).resolve()
            verify_pilot_hpo_output_root(pilot_output, workspace=workspace)
            selection = read_json(pilot_output / "hpo_selection.json")
        else:
            if arguments.selection or arguments.pilot_output:
                raise ValueError("selection inputs are only valid for confirmatory generation")
            selection = None
        matrix = build_matrix(
            protocol,
            arguments.profile,
            source_manifest=source,
            selection_manifest=selection,
        )
        counts = expected_matrix_counts(protocol, arguments.profile)
    if arguments.output:
        write_json_atomic(Path(arguments.output).resolve(), matrix)
    print(json.dumps({"profile": arguments.profile, "counts": counts, "cells": len(matrix["cells"]), "matrix_sha256": matrix["matrix_sha256"]}, indent=2, sort_keys=True))
    print("MATCHED_OBJECTIVE_MATRIX_GENERATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
