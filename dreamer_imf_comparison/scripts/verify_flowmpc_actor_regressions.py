#!/usr/bin/env python3
"""Run focused and complete regressions for the FlowMPC actor study."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def run(command: list[str]) -> None:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main() -> int:
    run([sys.executable, "imf_dreamer_jax/scripts/verify_flowmpc.py", "--rebrac"])
    run([sys.executable, "imf_dreamer_jax/scripts/verify_flowmpc.py", "--controller"])
    run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "imf_dreamer_jax/tests",
            "-p",
            "test_*.py",
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "dreamer_imf_comparison/tests",
            "-p",
            "test_*.py",
        ]
    )
    print("FLOWMPC_ACTOR_REGRESSIONS_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
