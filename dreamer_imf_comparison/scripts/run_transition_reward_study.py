#!/usr/bin/env python3
"""Execute the iMF-only direct transition-reward study."""

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

from dreamer_imf_compare import transition_reward_study as study  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "manifest",
            "reward",
            "verify-reward",
            "actor",
            "verify-actor",
            "finalize",
        ),
    )
    parser.add_argument("--corrected-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--reward-updates", type=int, default=study.REWARD_UPDATES)
    parser.add_argument("--heldout-batches", type=int, default=study.HELDOUT_BATCHES)
    parser.add_argument("--no-strict-replay", action="store_true")
    arguments = parser.parse_args()
    settings = {
        "reward_updates": arguments.reward_updates,
        "heldout_batches": arguments.heldout_batches,
    }
    if arguments.command == "manifest":
        if arguments.corrected_root is None:
            parser.error("manifest requires --corrected-root")
        payload = study.write_manifest(
            arguments.corrected_root, arguments.output_root, **settings
        )
        token = "IMF_TRANSITION_REWARD_MANIFEST_FROZEN"
    elif arguments.command == "reward":
        if arguments.corrected_root is None or arguments.index is None:
            parser.error("reward requires --corrected-root and --index")
        payload = study.train_reward_cell(
            arguments.corrected_root,
            arguments.output_root,
            arguments.index,
            **settings,
        )
        token = "IMF_TRANSITION_REWARD_CELL_COMPLETE"
    elif arguments.command == "verify-reward":
        if arguments.index is None:
            parser.error("verify-reward requires --index")
        payload = study.verify_reward_cell(arguments.output_root, arguments.index)
        token = "IMF_TRANSITION_REWARD_CELL_VERIFIED"
    elif arguments.command == "actor":
        if arguments.corrected_root is None or arguments.index is None:
            parser.error("actor requires --corrected-root and --index")
        payload = study.train_actor_cell(
            arguments.corrected_root,
            arguments.output_root,
            arguments.index,
            **settings,
        )
        token = "IMF_TRANSITION_REWARD_ACTOR_COMPLETE"
    elif arguments.command == "verify-actor":
        if arguments.index is None:
            parser.error("verify-actor requires --index")
        payload = study.verify_actor_cell(
            arguments.output_root,
            arguments.index,
            strict_replay=not arguments.no_strict_replay,
        )
        token = "IMF_TRANSITION_REWARD_ACTOR_VERIFIED"
    else:
        payload = study.finalize(arguments.output_root)
        study.validate_final(arguments.output_root)
        token = "IMF_TRANSITION_REWARD_STUDY_COMPLETE"
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
