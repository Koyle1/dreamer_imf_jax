#!/usr/bin/env python3
"""Run one append-only cell of the action-reward residual study."""

from __future__ import annotations

import argparse
import json

from dreamer_imf_compare import action_reward_residual_study as study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "manifest",
            "preflight",
            "dense_probe",
            "residual",
            "evaluation",
            "actor",
            "finalize",
        ),
    )
    parser.add_argument("--mtp-root", required=True)
    parser.add_argument("--centered-root", required=True)
    parser.add_argument("--sparse-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, choices=study.WORLD_MODEL_SEEDS)
    parser.add_argument("--horizon", type=int, choices=study.HORIZONS)
    parser.add_argument("--residual-updates", type=int, default=1000)
    parser.add_argument("--actor-updates", type=int, default=3000)
    parser.add_argument("--preparation-updates", type=int, default=500)
    parser.add_argument("--evaluation-episodes", type=int, default=5)
    parser.add_argument(
        "--reward-objective",
        choices=("uncentered_pseudo_huber", "dense_quadratic_control"),
        default="uncentered_pseudo_huber",
    )
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--dense-reference-root")
    args = parser.parse_args()
    settings = {
        "centered_root": args.centered_root,
        "sparse_root": args.sparse_root,
        "residual_updates": args.residual_updates,
        "actor_updates": args.actor_updates,
        "preparation_updates": args.preparation_updates,
        "evaluation_episodes": args.evaluation_episodes,
        "reward_objective": args.reward_objective,
        "control_scale": args.control_scale,
        "dense_reference_root": args.dense_reference_root,
    }
    common = (args.mtp_root, args.output_root)
    if args.stage == "manifest":
        result = study.write_manifest(*common, **settings)
    elif args.stage == "preflight":
        result = study.record_preflight(*common, **settings)
    elif args.stage == "dense_probe":
        if args.seed is None:
            parser.error("--seed is required")
        result = study.prepare_dense_probe_cell(*common, args.seed, **settings)
    elif args.stage == "residual":
        if args.seed is None:
            parser.error("--seed is required")
        result = study.train_residual_cell(*common, args.seed, **settings)
    elif args.stage == "evaluation":
        if args.seed is None:
            parser.error("--seed is required")
        result = study.evaluate_residual_cell(*common, args.seed, **settings)
    elif args.stage == "actor":
        if args.seed is None or args.horizon is None:
            parser.error("--seed and --horizon are required")
        result = study.train_actor_cell(*common, args.seed, args.horizon, **settings)
    else:
        result = study.finalize(*common, **settings)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
