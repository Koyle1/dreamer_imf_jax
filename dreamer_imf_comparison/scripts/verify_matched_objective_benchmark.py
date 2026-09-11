#!/usr/bin/env python3
"""Verify a completed matched-objective output root and all retained digests."""

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

from dreamer_imf_compare.matched_objective_benchmark import verify_output_root  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workspace", default=str(WORKSPACE))
    arguments = parser.parse_args()
    summary = verify_output_root(
        Path(arguments.output).resolve(), workspace=Path(arguments.workspace).resolve()
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("MATCHED_OBJECTIVE_BENCHMARK_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
