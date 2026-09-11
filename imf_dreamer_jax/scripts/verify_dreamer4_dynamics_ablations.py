#!/usr/bin/env python3
"""Run focused behavioral gates for Dreamer-4 dynamics ablations."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    workspace = project.parent
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    source = str(project / "src")
    environment["PYTHONPATH"] = source if not existing else source + os.pathsep + existing
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(project / "tests"),
            "-p",
            "test_dreamer4_dynamics_ablations.py",
        ],
        cwd=workspace,
        env=environment,
        check=False,
    )
    if completed.returncode:
        return completed.returncode
    print("DREAMER4_DYNAMICS_ABLATIONS_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

