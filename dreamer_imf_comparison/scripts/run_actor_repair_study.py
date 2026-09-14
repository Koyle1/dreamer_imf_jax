#!/usr/bin/env python3
"""Run one append-only stage of the continuous actor-repair study."""

from __future__ import annotations

import argparse
import json

from dreamer_imf_compare import actor_repair_study as study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("manifest", "random", "pure", "world", "finalize", "smoke"),
    )
    parser.add_argument("--mtp-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--actor-updates", type=int, default=1500)
    parser.add_argument("--preparation-updates", type=int, default=500)
    parser.add_argument("--evaluation-episodes", type=int, default=25)
    parser.add_argument("--exact-batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.stage == "smoke":
        result = study.run_smoke(args.output_root)
        print(json.dumps(result, sort_keys=True))
        print("ACTOR_REPAIR_SMOKE_VERIFIED")
        return
    if not args.mtp_root:
        parser.error("--mtp-root is required outside smoke mode")
    settings = {
        "actor_updates": args.actor_updates,
        "preparation_updates": args.preparation_updates,
        "evaluation_episodes": args.evaluation_episodes,
        "exact_batch_size": args.exact_batch_size,
    }
    common = (args.mtp_root, args.output_root)
    if args.stage == "manifest":
        result = study.write_manifest(*common, **settings)
    elif args.stage == "random":
        result = study.run_random_cell(*common, **settings)
    elif args.stage in ("pure", "world"):
        if args.index is None:
            parser.error(f"--index is required for {args.stage}")
        function = (
            study.run_pure_cell if args.stage == "pure" else study.run_world_cell
        )
        result = function(*common, args.index, **settings)
    else:
        result = study.finalize(*common, **settings)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
