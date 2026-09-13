#!/usr/bin/env python3
"""Execute the fresh corrected actor training study."""

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

from dreamer_imf_compare import correct_actor_training as study  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("manifest", "cell", "verify-cell", "select", "finalize"),
    )
    parser.add_argument("--pilot-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument(
        "--preparation-updates", type=int, default=study.PREPARATION_UPDATES
    )
    parser.add_argument("--actor-updates", type=int, default=study.ACTOR_UPDATES)
    parser.add_argument("--no-strict-replay", action="store_true")
    arguments = parser.parse_args()
    settings = {
        "preparation_updates": arguments.preparation_updates,
        "actor_updates": arguments.actor_updates,
    }

    if arguments.command == "manifest":
        if arguments.pilot_root is None:
            parser.error("manifest requires --pilot-root")
        payload = study.write_manifest(
            arguments.pilot_root, arguments.output_root, **settings
        )
        token = "CORRECT_ACTOR_MANIFEST_FROZEN"
    elif arguments.command == "cell":
        if arguments.pilot_root is None or arguments.index is None:
            parser.error("cell requires --pilot-root and --index")
        payload = study.run_cell(
            arguments.pilot_root,
            arguments.output_root,
            arguments.index,
            **settings,
        )
        token = "CORRECT_ACTOR_CELL_COMPLETE"
    elif arguments.command == "verify-cell":
        if arguments.index is None:
            parser.error("verify-cell requires --index")
        payload = study.verify_cell(
            arguments.output_root,
            arguments.index,
            strict_replay=not arguments.no_strict_replay,
        )
        token = "CORRECT_ACTOR_CELL_VERIFIED"
    elif arguments.command == "select":
        payload = study.select_candidates(arguments.output_root)
        study.validate_selection_result(arguments.output_root)
        token = "CORRECT_ACTOR_HPO_SELECTION_COMPLETE"
    else:
        payload = study.finalize(arguments.output_root)
        study.validate_final(arguments.output_root)
        token = "CORRECT_ACTOR_TRAINING_COMPLETE"
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
