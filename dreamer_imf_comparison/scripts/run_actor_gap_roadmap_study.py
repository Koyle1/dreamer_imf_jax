#!/usr/bin/env python3
"""Run one frozen stage of the trajectory-iMF actor-gap roadmap study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
for path in (PROJECT, WORKSPACE / "imf_dreamer_jax" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dreamer_imf_compare import actor_gap_roadmap_study as study  # noqa: E402

COMMANDS = (
    "preflight",
    "seal-preflight",
    "validate-submission",
    "calibration",
    "diagnostic-cell",
    "verify-diagnostic-cell",
    "verify-diagnostics",
    "model-cell",
    "verify-model-cell",
    "verify-models",
    "evaluation-cell",
    "verify-evaluation-cell",
    "verify-evaluations",
    "finalize",
)


def _require_dependency(parser: argparse.ArgumentParser, arguments: Any) -> Path:
    if arguments.dependency_root is None:
        parser.error(f"{arguments.command} requires --dependency-root")
    return arguments.dependency_root


def _require_index(parser: argparse.ArgumentParser, arguments: Any) -> int:
    if arguments.index is None:
        parser.error(f"{arguments.command} requires --index")
    return int(arguments.index)


def _print_payload(payload: Any, token: str) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(token)


def _authorize_submission(
    parser: argparse.ArgumentParser, arguments: Any, stage: str, *, action: str
) -> dict[str, Any]:
    if (
        arguments.submission_map is None
        or arguments.submission_stage != stage
        or arguments.submission_map_file_sha256 is None
    ):
        parser.error(f"{stage}-cell requires its immutable submission map")
    body = study.validate_submission_map(
        arguments.output_root,
        arguments.submission_map,
        stage,
        _require_index(parser, arguments),
        expected_file_sha256=arguments.submission_map_file_sha256,
        action=action,
    )
    matches = [
        entry
        for entry in body["entries"]
        if int(entry["index"]) == int(arguments.index)
    ]
    if len(matches) != 1:
        raise ValueError("submission authorization entry is absent or duplicated")
    return {
        "map_path": body["map_path"],
        "map_file_sha256": arguments.submission_map_file_sha256,
        "map_sha256": body["map_sha256"],
        "mode": matches[0]["mode"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--dependency-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--submission-map", type=Path)
    parser.add_argument(
        "--submission-stage", choices=("diagnostic", "model", "evaluation")
    )
    parser.add_argument("--submission-map-file-sha256")
    parser.add_argument("--submission-action", choices=("create", "verify"))
    parser.add_argument("--model-updates", type=int, default=study.MODEL_UPDATES)
    parser.add_argument(
        "--evaluation-episodes", type=int, default=study.EVALUATION_EPISODES
    )
    arguments = parser.parse_args()
    settings = {
        "model_updates": arguments.model_updates,
        "evaluation_episodes": arguments.evaluation_episodes,
    }

    if arguments.command == "preflight":
        dependency_root = _require_dependency(parser, arguments)
        manifest = study.write_manifest(
            dependency_root, arguments.output_root, **settings
        )
        dependency = study.authenticate_dependency(dependency_root)
        payload = {
            "status": "frozen_before_execution",
            "source_commit": manifest["source_commit"],
            "manifest_sha256": manifest["manifest_sha256"],
            "dependency_root": dependency["root"],
            "dependency_source_commit": dependency["source_commit"],
            "diagnostic_cells": len(manifest["diagnostic_cells"]),
            "model_cells": len(manifest["model_cells"]),
            "evaluation_cells": len(manifest["evaluation_cells"]),
        }
        token = "ACTOR_GAP_ROADMAP_MANIFEST_FROZEN"
    elif arguments.command == "seal-preflight":
        marker = study.write_preflight_marker(arguments.output_root)
        payload = marker
        token = "ACTOR_GAP_ROADMAP_PREFLIGHT_SEALED"
    elif arguments.command == "validate-submission":
        if (
            arguments.submission_map is None
            or arguments.submission_stage is None
            or arguments.submission_map_file_sha256 is None
        ):
            parser.error(
                "validate-submission requires --submission-map, "
                "--submission-stage, and --submission-map-file-sha256"
            )
        payload = study.validate_submission_map(
            arguments.output_root,
            arguments.submission_map,
            arguments.submission_stage,
            _require_index(parser, arguments),
            expected_file_sha256=arguments.submission_map_file_sha256,
            action=arguments.submission_action or "create",
        )
        token = "ACTOR_GAP_ROADMAP_SUBMISSION_AUTHORIZED"
    elif arguments.command == "calibration":
        dependency_root = _require_dependency(parser, arguments)
        result = study.run_calibration(
            dependency_root, arguments.output_root, **settings
        )
        payload = result
        token = "ACTOR_GAP_ROADMAP_CALIBRATION_DATA_READY"
    elif arguments.command == "diagnostic-cell":
        dependency_root = _require_dependency(parser, arguments)
        index = _require_index(parser, arguments)
        authorization = _authorize_submission(
            parser, arguments, "diagnostic", action="create"
        )
        result = study.run_diagnostic_cell(
            dependency_root,
            arguments.output_root,
            index,
            submission_authorization=authorization,
            **settings,
        )
        payload = result
        token = "ACTOR_GAP_ROADMAP_DIAGNOSTIC_CELL_DATA_READY"
    elif arguments.command == "verify-diagnostic-cell":
        index = _require_index(parser, arguments)
        authorization = _authorize_submission(
            parser, arguments, "diagnostic", action="verify"
        )
        marker = study.verify_diagnostic_cell(
            arguments.output_root,
            index,
            strict_replay=True,
            submission_authorization=authorization,
        )
        payload = marker
        token = "ACTOR_GAP_ROADMAP_DIAGNOSTIC_CELL_VERIFIED"
    elif arguments.command == "verify-diagnostics":
        payload = study.verify_diagnostic_stage(arguments.output_root)
        token = "ACTOR_GAP_ROADMAP_DIAGNOSTICS_VERIFIED"
    elif arguments.command == "model-cell":
        dependency_root = _require_dependency(parser, arguments)
        index = _require_index(parser, arguments)
        authorization = _authorize_submission(
            parser, arguments, "model", action="create"
        )
        result = study.train_model_cell(
            dependency_root,
            arguments.output_root,
            index,
            submission_authorization=authorization,
            **settings,
        )
        payload = result
        token = "ACTOR_GAP_ROADMAP_MODEL_CELL_DATA_READY"
    elif arguments.command == "verify-model-cell":
        index = _require_index(parser, arguments)
        authorization = _authorize_submission(
            parser, arguments, "model", action="verify"
        )
        marker = study.verify_model_cell(
            arguments.output_root,
            index,
            strict_replay=True,
            submission_authorization=authorization,
        )
        payload = marker
        token = "ACTOR_GAP_ROADMAP_MODEL_CELL_VERIFIED"
    elif arguments.command == "verify-models":
        payload = study.verify_model_stage(arguments.output_root)
        token = "ACTOR_GAP_ROADMAP_MODELS_VERIFIED"
    elif arguments.command == "evaluation-cell":
        dependency_root = _require_dependency(parser, arguments)
        index = _require_index(parser, arguments)
        authorization = _authorize_submission(
            parser, arguments, "evaluation", action="create"
        )
        result = study.run_evaluation_cell(
            dependency_root,
            arguments.output_root,
            index,
            submission_authorization=authorization,
            **settings,
        )
        payload = result
        token = "ACTOR_GAP_ROADMAP_EVALUATION_CELL_DATA_READY"
    elif arguments.command == "verify-evaluation-cell":
        index = _require_index(parser, arguments)
        authorization = _authorize_submission(
            parser, arguments, "evaluation", action="verify"
        )
        marker = study.verify_evaluation_cell(
            arguments.output_root,
            index,
            strict_replay=True,
            submission_authorization=authorization,
        )
        payload = marker
        token = "ACTOR_GAP_ROADMAP_EVALUATION_CELL_VERIFIED"
    elif arguments.command == "verify-evaluations":
        payload = study.verify_evaluation_stage(arguments.output_root)
        token = "ACTOR_GAP_ROADMAP_EVALUATIONS_VERIFIED"
    else:
        report = study.finalize(arguments.output_root)
        validated = study.validate_final(arguments.output_root)
        if validated != report:
            raise ValueError("final report differs from independent validation")
        payload = report
        token = "ACTOR_GAP_ROADMAP_STUDY_COMPLETE"

    _print_payload(payload, token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
