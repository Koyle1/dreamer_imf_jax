#!/usr/bin/env python3
"""Run one append-only cell of the frozen reward-head MTP study."""

from __future__ import annotations

import argparse
import json

from dreamer_imf_compare import reward_head_mtp_study as study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("manifest", "reward", "evaluation", "actor", "finalize"))
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--train-probe-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, choices=study.WORLD_MODEL_SEEDS)
    parser.add_argument("--arm", choices=study.ARMS)
    parser.add_argument("--horizon", type=int, choices=(5, 15))
    parser.add_argument("--reward-updates", type=int, default=1000)
    parser.add_argument("--actor-updates", type=int, default=3000)
    parser.add_argument("--preparation-updates", type=int, default=500)
    parser.add_argument("--reward-bins", type=int, default=255)
    parser.add_argument("--evaluation-episodes", type=int, default=5)
    args = parser.parse_args()
    settings = {
        "reward_updates": args.reward_updates,
        "actor_updates": args.actor_updates,
        "preparation_updates": args.preparation_updates,
        "reward_bins": args.reward_bins,
        "evaluation_episodes": args.evaluation_episodes,
    }
    common = (args.baseline_root, args.train_probe_root, args.output_root)
    if args.stage == "manifest":
        result = study.write_manifest(*common, **settings)
    elif args.stage == "reward":
        if args.seed is None or args.arm is None:
            parser.error("--seed and --arm are required")
        result = study.train_reward_cell(*common, args.seed, args.arm, **settings)
    elif args.stage == "evaluation":
        if args.seed is None or args.arm is None:
            parser.error("--seed and --arm are required")
        result = study.evaluate_reward_cell(*common, args.seed, args.arm, **settings)
    elif args.stage == "actor":
        if args.seed is None or args.arm is None or args.horizon is None:
            parser.error("--seed, --arm, and --horizon are required")
        result = study.train_actor_cell(
            *common, args.seed, args.arm, args.horizon, **settings
        )
    else:
        result = study.finalize(*common, **settings)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
