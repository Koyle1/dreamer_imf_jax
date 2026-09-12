#!/usr/bin/env python3
"""Prepare, execute, or aggregate the paired shortcut stability diagnostic."""

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

from dreamer_imf_compare.shortcut_stability_study import (  # noqa: E402
    aggregate,
    prepare_manifest,
    resolve_array_index,
    run_entry,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "aggregate"))
    parser.add_argument("--archived-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--updates", type=int, default=10_000)
    parser.add_argument("--index", type=int)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        if not arguments.archived_root:
            parser.error("prepare requires --archived-root")
        result = prepare_manifest(
            arguments.archived_root, arguments.output, updates=arguments.updates
        )
        marker = "SHORTCUT_STABILITY_STUDY_FROZEN"
    elif arguments.command == "run":
        result = run_entry(arguments.output, resolve_array_index(arguments.index))
        marker = "SHORTCUT_STABILITY_CELL_COMPLETE"
    else:
        result = aggregate(arguments.output)
        marker = "SHORTCUT_STABILITY_STUDY_AGGREGATED"
    print(json.dumps(result, indent=2, sort_keys=True))
    print(marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
