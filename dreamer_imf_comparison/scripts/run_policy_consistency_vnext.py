#!/usr/bin/env python3
"""Run one append-only cell of the policy-consistency vNext study."""

from __future__ import annotations

import argparse
import json

from dreamer_imf_compare import policy_consistency_vnext as study


def scales(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item)
    if not result:
        raise argparse.ArgumentTypeError("at least one epistemic scale is required")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("manifest", "probe-bank", "probe", "actor", "finalize"))
    parser.add_argument("--pilot-root", required=False)
    parser.add_argument("--causal-root", required=False)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, choices=study.WORLD_MODEL_SEEDS)
    parser.add_argument("--arm", choices=study.ARMS)
    parser.add_argument("--horizon", type=int, choices=study.HORIZONS)
    parser.add_argument("--actor-updates", type=int, default=3000)
    parser.add_argument("--preparation-updates", type=int, default=500)
    parser.add_argument("--epistemic-scale", type=float, default=0.0)
    parser.add_argument("--epistemic-scales", type=scales, default=(0.0,))
    parser.add_argument("--evaluation-episodes", type=int, default=5)
    args = parser.parse_args()
    if args.stage != "finalize" and (not args.pilot_root or not args.causal_root):
        parser.error("--pilot-root and --causal-root are required for this stage")
    common = {
        "actor_updates": args.actor_updates,
        "preparation_updates": args.preparation_updates,
        "epistemic_scales": args.epistemic_scales,
    }
    if args.stage == "manifest":
        result = study.write_manifest(
            args.pilot_root, args.causal_root, args.output_root, **common
        )
    elif args.stage == "probe-bank":
        if args.seed is None:
            parser.error("--seed is required")
        result = study.prepare_probe_bank(
            args.pilot_root,
            args.causal_root,
            args.output_root,
            args.seed,
            **common,
        )
    elif args.stage == "probe":
        if args.seed is None or args.arm is None:
            parser.error("--seed and --arm are required")
        result = study.evaluate_probe_arm(
            args.pilot_root,
            args.causal_root,
            args.output_root,
            args.seed,
            args.arm,
            **common,
        )
    elif args.stage == "actor":
        if args.seed is None or args.arm is None or args.horizon is None:
            parser.error("--seed, --arm, and --horizon are required")
        result = study.train_repaired_actor(
            args.pilot_root,
            args.causal_root,
            args.output_root,
            args.seed,
            args.arm,
            args.horizon,
            actor_updates=args.actor_updates,
            preparation_updates=args.preparation_updates,
            epistemic_scale=args.epistemic_scale,
            epistemic_scales=args.epistemic_scales,
            evaluation_episodes=args.evaluation_episodes,
        )
    else:
        result = study.finalize(args.output_root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
