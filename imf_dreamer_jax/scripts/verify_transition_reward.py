#!/usr/bin/env python3
"""Run the focused direct transition-reward acceptance tests."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(PACKAGE_ROOT / "tests"),
            "-p",
            "test_transition_reward.py",
            "-v",
        ],
        cwd=PACKAGE_ROOT.parent,
        env=environment,
        check=False,
    )
    if result.returncode:
        raise SystemExit(result.returncode)
    print("IMF_TRANSITION_REWARD_VERIFIED")


if __name__ == "__main__":
    main()
