#!/usr/bin/env python3
"""Run one append-only stage of the actor-failure causal diagnostic."""

from __future__ import annotations

import argparse
import json

from dreamer_imf_compare import actor_failure_diagnostic as study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("manifest", "preflight", "random", "actor", "finalize")
    )
    parser.add_argument("--mtp-root", required=True)
    parser.add_argument("--residual-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--actor-updates", type=int, default=3000)
    parser.add_argument("--preparation-updates", type=int, default=500)
    parser.add_argument("--evaluation-episodes", type=int, default=50)
    args = parser.parse_args()
    settings = {
        "actor_updates": args.actor_updates,
        "preparation_updates": args.preparation_updates,
        "evaluation_episodes": args.evaluation_episodes,
    }
    common = (args.mtp_root, args.residual_root, args.output_root)
    if args.stage == "manifest":
        result = study.write_manifest(*common, **settings)
    elif args.stage == "preflight":
        result = study.record_preflight(*common, **settings)
    elif args.stage == "random":
        result = study.run_random_cell(*common, **settings)
    elif args.stage == "actor":
        if args.index is None:
            parser.error("--index is required for actor")
        result = study.run_actor_cell(*common, args.index, **settings)
    else:
        result = study.finalize(*common, **settings)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
