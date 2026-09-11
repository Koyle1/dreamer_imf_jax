#!/usr/bin/env python3
"""Freeze, execute, finalize, or verify the NeurIPS controls benchmark."""

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
from dreamer_imf_compare.neurips_controls import (  # noqa: E402
    STAGE_ORDER,
    finalize_controls_run,
    freeze_controls_run,
    load_frozen_controls,
    read_controls_protocol,
    run_canonical_datasets,
    run_canonical_dataset_unit,
    run_control_cell,
    run_controls_all,
    run_stage_cells,
    validate_controls_matrix,
    validate_controls_source_manifest,
    verify_controls_output,
)
from dreamer_imf_compare.matched_objective_protocol import (  # noqa: E402
    read_matched_objective_protocol,
)
from dreamer_imf_compare import matched_objective_benchmark as parent  # noqa: E402


def _selection(
    path: str | None, filename: str, *, label: str, workspace: Path
) -> dict | None:
    if path is None:
        return None
    source = Path(path).resolve()
    if source.is_dir():
        if label == "parent":
            verified = parent.verify_pilot_hpo_output_root(
                source, workspace=workspace
            )
            if verified.get("profile") != "pilot":
                raise ValueError("parent selection root is not a verified pilot")
        else:
            verified = verify_controls_output(source, workspace=workspace)
            if verified.get("profile") != "development":
                raise ValueError(
                    "controls selection root is not a verified development run"
                )
        source = source / filename
    return read_json(source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "finalize", "verify", "all"))
    parser.add_argument("--profile", choices=("smoke", "development", "confirmatory"), default="smoke")
    parser.add_argument("--protocol", default=str(PROJECT / "neurips_controls_protocol.json"))
    parser.add_argument("--parent-protocol", default=str(PROJECT / "matched_objective_protocol.json"))
    parser.add_argument("--output")
    parser.add_argument("--workspace", default=str(WORKSPACE))
    parser.add_argument("--parent-selection", help="parent pilot root or hpo_selection.json")
    parser.add_argument("--controls-selection", help="controls development root or controls_selection.json")
    parser.add_argument(
        "--stages",
        default=None,
        help="dependency-ordered comma-separated compute,world,rollout,actor",
    )
    parser.add_argument(
        "--cell-id",
        help="for `run`, execute exactly one frozen compute/world/rollout/actor cell",
    )
    parser.add_argument(
        "--dataset-cell-id",
        help="for `run`, execute exactly one frozen canonical parent dataset cell",
    )
    parser.add_argument(
        "--dataset-task",
        help="for `run`, select one canonical dataset by task (requires seed)",
    )
    parser.add_argument(
        "--dataset-world-model-seed",
        type=int,
        help="for `run`, select one canonical dataset by world-model seed (requires task)",
    )
    arguments = parser.parse_args()
    output = Path(
        arguments.output
        or PROJECT / "results" / f"neurips_controls_{arguments.profile}"
    ).resolve()
    workspace = Path(arguments.workspace).resolve()
    protocol = read_controls_protocol(arguments.protocol)
    parent_protocol = read_matched_objective_protocol(arguments.parent_protocol)
    parent_selection = _selection(
        arguments.parent_selection,
        "hpo_selection.json",
        label="parent",
        workspace=workspace,
    )
    controls_selection = _selection(
        arguments.controls_selection,
        "controls_selection.json",
        label="controls",
        workspace=workspace,
    )
    selectors = (
        arguments.cell_id,
        arguments.dataset_cell_id,
        arguments.dataset_task,
        arguments.dataset_world_model_seed,
    )
    if arguments.command != "run" and any(value is not None for value in selectors):
        parser.error("cell/dataset selectors are valid only with the `run` command")
    if arguments.cell_id is not None and any(
        value is not None
        for value in (
            arguments.dataset_cell_id,
            arguments.dataset_task,
            arguments.dataset_world_model_seed,
        )
    ):
        parser.error("select a controls cell or a dataset cell, not both")
    if arguments.dataset_cell_id is not None and any(
        value is not None
        for value in (arguments.dataset_task, arguments.dataset_world_model_seed)
    ):
        parser.error("select a dataset by cell id or task/seed, not both")
    if (arguments.dataset_task is None) != (
        arguments.dataset_world_model_seed is None
    ):
        parser.error("--dataset-task and --dataset-world-model-seed are a pair")
    if arguments.stages is not None and any(value is not None for value in selectors):
        parser.error("--stages cannot be combined with an exact cell selector")

    if arguments.command == "verify":
        result = verify_controls_output(output, workspace=workspace)
        print(json.dumps(result, indent=2, sort_keys=True))
        print("NEURIPS_CONTROLS_VERIFIED")
        return 0
    if arguments.command == "all":
        result = run_controls_all(
            protocol,
            parent_protocol,
            arguments.profile,
            output,
            workspace=workspace,
            parent_selection=parent_selection,
            controls_selection=controls_selection,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        print("NEURIPS_CONTROLS_COMPLETE")
        return 0
    if arguments.command == "freeze":
        result = freeze_controls_run(
            protocol,
            parent_protocol,
            arguments.profile,
            output,
            workspace=workspace,
            parent_selection=parent_selection,
            controls_selection=controls_selection,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        print("NEURIPS_CONTROLS_FROZEN")
        return 0
    if arguments.command == "finalize":
        result = finalize_controls_run(output, workspace=workspace)
        print(json.dumps(result, indent=2, sort_keys=True))
        print("NEURIPS_CONTROLS_FINALIZED")
        return 0

    frozen_protocol, frozen_parent, source, matrix = load_frozen_controls(output)
    if frozen_protocol != protocol or frozen_parent != parent_protocol:
        raise ValueError("requested protocols differ from the frozen controls root")
    validate_controls_source_manifest(source, workspace)
    validate_controls_matrix(matrix, protocol, parent_protocol, source)
    if arguments.dataset_cell_id is not None or arguments.dataset_task is not None:
        result = run_canonical_dataset_unit(
            output,
            protocol,
            parent_protocol,
            source,
            matrix,
            dataset_cell_id=arguments.dataset_cell_id,
            task=arguments.dataset_task,
            world_model_seed=arguments.dataset_world_model_seed,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        print("NEURIPS_CONTROLS_DATASET_CELL_COMPLETE")
        return 0
    if arguments.cell_id is not None:
        result = run_control_cell(
            arguments.cell_id, protocol, parent_protocol, matrix, output
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        print("NEURIPS_CONTROLS_CELL_COMPLETE")
        return 0
    run_canonical_datasets(output, protocol, parent_protocol, source, matrix)
    stages_text = arguments.stages or ",".join(STAGE_ORDER)
    stages = tuple(value.strip() for value in stages_text.split(",") if value.strip())
    result = run_stage_cells(protocol, parent_protocol, matrix, output, stages=stages)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("NEURIPS_CONTROLS_CELLS_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
