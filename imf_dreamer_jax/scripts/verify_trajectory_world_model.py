#!/usr/bin/env python3
"""Executable acceptance checks for trajectory-iMF world-model integration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = PACKAGE_ROOT / "tests"


def run_pytest(arguments: list[str]) -> None:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *arguments],
        cwd=PACKAGE_ROOT.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode:
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--contexts", action="store_true")
    mode.add_argument("--objective", action="store_true")
    mode.add_argument("--tests", action="store_true")
    arguments = parser.parse_args()

    trajectory_tests = TEST_ROOT / "test_trajectory_world_model.py"
    if arguments.contexts:
        run_pytest([f"{trajectory_tests}::TrajectoryConditionTests"])
        print("TRAJECTORY_WORLD_MODEL_CONTEXTS_VERIFIED")
    elif arguments.objective:
        run_pytest(
            [
                f"{trajectory_tests}::TrajectoryConfigurationTests",
                f"{trajectory_tests}::TrajectoryObjectiveTests",
            ]
        )
        print("TRAJECTORY_WORLD_MODEL_OBJECTIVE_VERIFIED")
    else:
        run_pytest(
            [
                str(trajectory_tests),
                str(TEST_ROOT / "test_world_model.py"),
                str(TEST_ROOT / "test_world_model_repairs.py"),
            ]
        )
        print("TRAJECTORY_WORLD_MODEL_TESTS_VERIFIED")


if __name__ == "__main__":
    main()
