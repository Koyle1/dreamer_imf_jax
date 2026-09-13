#!/usr/bin/env python3
"""Fail-closed gates for the corrected matched actor training study."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
for path in (PROJECT, WORKSPACE / "imf_dreamer_jax" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dreamer_imf_compare import correct_actor_training as study  # noqa: E402


def _require_paths(arguments: argparse.Namespace, *names: str) -> None:
    for name in names:
        if getattr(arguments, name) is None:
            raise ValueError(f"{arguments.mode} requires --{name.replace('_', '-')}")


def verify_contract(pilot_root: Path, output_root: Path) -> None:
    manifest = study.write_manifest(pilot_root, output_root)
    study.validate_manifest(manifest, pilot_root=pilot_root)
    if len(manifest["cells"]) != study.EXPECTED_CELLS:
        raise ValueError("corrected actor contract does not contain 288 cells")
    counts = {
        arm: sum(cell["arm"] == arm for cell in manifest["cells"])
        for arm in ("shortcut_forcing", "trajectory_imf")
    }
    if counts != {"shortcut_forcing": 144, "trajectory_imf": 144}:
        raise ValueError("corrected actor arms are not balanced")
    if len({cell["original_actor_cell_id"] for cell in manifest["cells"]}) != 288:
        raise ValueError("an original actor grid cell is absent or duplicated")
    print(json.dumps({"cells": 288, "arms": counts}, sort_keys=True))
    print("CORRECT_ACTOR_CONTRACT_VERIFIED")


def verify_implementation(pilot_root: Path, output_root: Path) -> None:
    manifest = study.write_manifest(pilot_root, output_root)
    source = inspect.getsource(study.run_cell)
    required = (
        "create_agent(",
        "jit_train_behavior_cloning(",
        "jit_train_replay_critic(",
        "jit_train_actor_critic_dreamer3(",
        "behavior_prior=None",
        "frozen_world_digest",
        "world_model_parameter_delta",
    )
    if any(token not in source for token in required):
        raise ValueError("corrected actor implementation lost a required operation")
    for arm in ("shortcut_forcing", "trajectory_imf"):
        cell = next(row for row in manifest["cells"] if row["arm"] == arm)
        config, *_ = study.corrected_config(manifest, cell)
        config_dict = asdict(config)
        if (
            config_dict["actor_gradient"] != "reinforce"
            or config_dict["behavior_kl_scale"] != 0.0
            or config_dict["critic_bins"] != 51
            or config_dict["critic_output_init_scale"] != 0.0
            or config_dict["return_scale_ema_decay"] != 0.99
        ):
            raise ValueError(f"corrected {arm} runtime config differs")
    if (
        manifest["preparation_updates"] != 500
        or manifest["actor_updates"] != 10_000
    ):
        raise ValueError("corrected actor update counts differ")
    print("CORRECT_ACTOR_IMPLEMENTATION_VERIFIED")


def verify_selection_implementation() -> None:
    source = inspect.getsource(study.build_corrected_trials)
    if (
        "_corrected_result_for" not in source
        or "normalized_episode_return_mean" not in source
        or "select_hpo_candidates" not in inspect.getsource(study.select_candidates)
    ):
        raise ValueError("corrected HPO evidence or frozen selector is missing")
    forbidden = (
        'stage_directory(pilot_root, originals[0])',
        '_completed_result_for_cell(pilot_root, matrix, protocol, originals[0])',
    )
    if any(token in source for token in forbidden):
        raise ValueError("corrected HPO selector reads an old actor result")
    print("CORRECT_ACTOR_SELECTION_VERIFIED")


def verify_preflight(output_root: Path, index: int) -> None:
    import jax

    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("corrected actor preflight did not run on exactly one GPU")
    marker = study.verify_cell(output_root, index, strict_replay=True)
    if marker["strict_policy_and_environment_replay"] is not True:
        raise ValueError("corrected actor preflight omitted strict replay")
    print(json.dumps({"device": str(devices[0]), "cell_index": index}, sort_keys=True))
    print("CORRECT_ACTOR_GPU_PREFLIGHT_VERIFIED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "contract",
            "implementation",
            "selection",
            "self-test",
            "preflight",
            "cells",
            "selection-result",
            "final",
        ),
    )
    parser.add_argument("--pilot-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--index", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.mode == "contract":
        _require_paths(arguments, "pilot_root", "output_root")
        verify_contract(arguments.pilot_root, arguments.output_root)
    elif arguments.mode == "implementation":
        _require_paths(arguments, "pilot_root", "output_root")
        verify_implementation(arguments.pilot_root, arguments.output_root)
    elif arguments.mode == "selection":
        verify_selection_implementation()
    elif arguments.mode == "self-test":
        study.self_test()
        print("CORRECT_ACTOR_SELF_TEST_VERIFIED")
    elif arguments.mode == "preflight":
        _require_paths(arguments, "output_root")
        verify_preflight(arguments.output_root, arguments.index)
    elif arguments.mode == "cells":
        _require_paths(arguments, "output_root")
        rows = study.validate_all_cells(arguments.output_root)
        print(json.dumps({"verified_cells": len(rows)}, sort_keys=True))
        print("CORRECT_ACTOR_CELLS_VERIFIED")
    elif arguments.mode == "selection-result":
        _require_paths(arguments, "output_root")
        selected = study.validate_selection_result(arguments.output_root)
        print(json.dumps(selected["selected"], sort_keys=True))
        print("CORRECT_ACTOR_HPO_SELECTION_VERIFIED")
    else:
        _require_paths(arguments, "output_root")
        report = study.validate_final(arguments.output_root)
        print(json.dumps(report, indent=2, sort_keys=True))
        print("CORRECT_ACTOR_TRAINING_FINAL_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
