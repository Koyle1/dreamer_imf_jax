#!/usr/bin/env python3
"""Run one append-only cell of the gated advantage-repair study."""

from __future__ import annotations

import argparse
import json

from dreamer_imf_compare import advantage_repair_study as study


def scales(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item)
    if not result:
        raise argparse.ArgumentTypeError("at least one epistemic scale is required")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("manifest", "train-probe-bank", "world", "probe", "actor", "finalize"),
    )
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variant", choices=study.VARIANTS, default="advantage_reward")
    parser.add_argument("--seed", type=int, choices=study.WORLD_MODEL_SEEDS)
    parser.add_argument("--arm", choices=study.REPAIR_ARMS)
    parser.add_argument("--horizon", type=int, choices=(5, 15))
    parser.add_argument("--epistemic-scale", type=float, default=0.0)
    parser.add_argument("--world-updates", type=int, default=1000)
    parser.add_argument("--actor-updates", type=int, default=3000)
    parser.add_argument("--preparation-updates", type=int, default=500)
    parser.add_argument("--epistemic-scales", type=scales, default=(0.0, 1.0))
    args = parser.parse_args()
    settings = {
        "variant": args.variant,
        "world_updates": args.world_updates,
        "actor_updates": args.actor_updates,
        "preparation_updates": args.preparation_updates,
        "epistemic_scales": args.epistemic_scales,
    }
    if args.stage == "manifest":
        result = study.write_manifest(args.baseline_root, args.output_root, **settings)
    elif args.stage == "train-probe-bank":
        if args.seed is None:
            parser.error("--seed is required")
        result = study.prepare_train_probe_bank(
            args.baseline_root, args.output_root, args.seed, **settings
        )
    elif args.stage == "world":
        if args.seed is None or args.arm is None:
            parser.error("--seed and --arm are required")
        result = study.train_world_repair(
            args.baseline_root, args.output_root, args.seed, args.arm, **settings
        )
    elif args.stage == "probe":
        if args.seed is None or args.arm is None:
            parser.error("--seed and --arm are required")
        result = study.evaluate_world_repair(
            args.baseline_root, args.output_root, args.seed, args.arm, **settings
        )
    elif args.stage == "actor":
        if args.seed is None or args.arm is None or args.horizon is None:
            parser.error("--seed, --arm, and --horizon are required")
        result = study.train_repaired_actor(
            args.baseline_root,
            args.output_root,
            args.seed,
            args.arm,
            args.horizon,
            epistemic_scale=args.epistemic_scale,
            **settings,
        )
    else:
        result = study.finalize(args.baseline_root, args.output_root, **settings)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
