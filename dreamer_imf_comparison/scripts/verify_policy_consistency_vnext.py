#!/usr/bin/env python3
"""Run focused integration and legacy-regression verification."""

from pathlib import Path
import subprocess
import sys


root = Path(__file__).resolve().parents[2]
commands = (
    (
        root / "imf_dreamer_jax",
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_policy_consistency_vnext",
            "tests.test_dreamer4_actor_ablations",
            "tests.test_trajectory_world_model",
        ],
    ),
    (
        root / "dreamer_imf_comparison",
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_policy_consistency_vnext",
            "tests.test_policy_alignment_diagnostics",
        ],
    ),
)
for cwd, command in commands:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if completed.returncode:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit(completed.returncode)
print("POLICY_CONSISTENCY_VNEXT_VERIFIED")
