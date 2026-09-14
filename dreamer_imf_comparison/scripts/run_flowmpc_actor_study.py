#!/usr/bin/env python3
"""Run the paper-faithful ReBRAC plus ITPO controller study."""

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

from dreamer_imf_compare import flowmpc_actor_study as study  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "manifest",
            "rebrac",
            "verify-rebrac",
            "tune",
            "verify-tune",
            "evaluate",
            "verify-evaluate",
            "finalize",
        ),
    )
    parser.add_argument("--reward-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--rebrac-updates", type=int, default=study.REBRAC_UPDATES)
    parser.add_argument("--no-strict-replay", action="store_true")
    arguments = parser.parse_args()
    settings = {"rebrac_updates": arguments.rebrac_updates}
    if arguments.command == "manifest":
        if arguments.reward_root is None:
            parser.error("manifest requires --reward-root")
        payload = study.write_manifest(
            arguments.reward_root, arguments.output_root, **settings
        )
        token = "FLOWMPC_ACTOR_MANIFEST_FROZEN"
    elif arguments.command == "rebrac":
        if arguments.reward_root is None or arguments.index is None:
            parser.error("rebrac requires --reward-root and --index")
        payload = study.train_rebrac_cell(
            arguments.reward_root, arguments.output_root, arguments.index, **settings
        )
        token = "FLOWMPC_REBRAC_CELL_COMPLETE"
    elif arguments.command == "verify-rebrac":
        if arguments.index is None:
            parser.error("verify-rebrac requires --index")
        payload = study.verify_rebrac_cell(arguments.output_root, arguments.index)
        token = "FLOWMPC_REBRAC_CELL_VERIFIED"
    elif arguments.command == "tune":
        if arguments.reward_root is None:
            parser.error("tune requires --reward-root")
        payload = study.run_tuning(
            arguments.reward_root, arguments.output_root, **settings
        )
        token = "FLOWMPC_TUNING_COMPLETE"
    elif arguments.command == "verify-tune":
        payload = study.verify_tuning(
            arguments.output_root, strict_replay=not arguments.no_strict_replay
        )
        token = "FLOWMPC_TUNING_VERIFIED"
    elif arguments.command == "evaluate":
        if arguments.reward_root is None or arguments.index is None:
            parser.error("evaluate requires --reward-root and --index")
        payload = study.run_evaluation_cell(
            arguments.reward_root, arguments.output_root, arguments.index, **settings
        )
        token = "FLOWMPC_EVALUATION_CELL_COMPLETE"
    elif arguments.command == "verify-evaluate":
        if arguments.index is None:
            parser.error("verify-evaluate requires --index")
        payload = study.verify_evaluation_cell(
            arguments.output_root,
            arguments.index,
            strict_replay=not arguments.no_strict_replay,
        )
        token = "FLOWMPC_EVALUATION_CELL_VERIFIED"
    else:
        payload = study.finalize(arguments.output_root)
        study.validate_final(arguments.output_root)
        token = "FLOWMPC_ACTOR_STUDY_COMPLETE"
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
