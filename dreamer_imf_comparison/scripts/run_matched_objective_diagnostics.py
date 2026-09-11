#!/usr/bin/env python3
"""Run the registered scalar-transition diagnostics for a frozen matrix."""

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

from dreamer_imf_compare.matched_objective_diagnostics import (  # noqa: E402
    run_registered_diagnostics,
)


SMOKE_DEFAULTS = {
    "smoke_updates": 2,
    "smoke_training_episodes": 16,
    "smoke_test_conditions": 8,
    "smoke_repeated_futures": 8,
    "smoke_batch_size": 4,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        required=True,
        help="frozen claim-eligible confirmatory benchmark output root",
    )
    parser.add_argument("--workspace", default=str(WORKSPACE))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="reduced engineering run; writes validator-ineligible smoke_nonclaim evidence",
    )
    parser.add_argument("--smoke-updates", type=int, default=SMOKE_DEFAULTS["smoke_updates"])
    parser.add_argument(
        "--smoke-training-episodes",
        type=int,
        default=SMOKE_DEFAULTS["smoke_training_episodes"],
    )
    parser.add_argument(
        "--smoke-test-conditions",
        type=int,
        default=SMOKE_DEFAULTS["smoke_test_conditions"],
    )
    parser.add_argument(
        "--smoke-repeated-futures",
        type=int,
        default=SMOKE_DEFAULTS["smoke_repeated_futures"],
    )
    parser.add_argument(
        "--smoke-batch-size",
        type=int,
        default=SMOKE_DEFAULTS["smoke_batch_size"],
    )
    arguments = parser.parse_args()
    supplied = {
        name: getattr(arguments, name)
        for name in SMOKE_DEFAULTS
    }
    if not arguments.smoke and supplied != SMOKE_DEFAULTS:
        parser.error("smoke count overrides require --smoke")
    summary = run_registered_diagnostics(
        Path(arguments.output_root).resolve(),
        workspace=Path(arguments.workspace).resolve(),
        smoke=arguments.smoke,
        **supplied,
    )
    concise = {
        "status": summary["status"],
        "diagnostic_summary_sha256": summary["diagnostic_summary_sha256"],
        "passed": {
            diagnostic: {
                arm: arm_row["passed"]
                for arm, arm_row in details["arms"].items()
            }
            for diagnostic, details in summary["diagnostics"].items()
        },
    }
    print(json.dumps(concise, indent=2, sort_keys=True))
    print(
        "MATCHED_OBJECTIVE_DIAGNOSTICS_SMOKE_NONCLAIM"
        if arguments.smoke
        else "MATCHED_OBJECTIVE_DIAGNOSTICS_COMPLETE"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
