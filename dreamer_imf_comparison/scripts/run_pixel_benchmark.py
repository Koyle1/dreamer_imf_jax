#!/usr/bin/env python3
"""Run or verify the secondary real-render DMC pixel benchmark."""

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

from dreamer_imf_compare.pixel_benchmark import (  # noqa: E402
    aggregate_artifacts,
    load_protocol,
    run_benchmark,
    verify_aggregate,
    verify_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("run", "verify", "contract", "aggregate", "verify-aggregate")
    )
    parser.add_argument("--output")
    parser.add_argument("--profile", choices=("smoke", "hard"), default="smoke")
    parser.add_argument("--task")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--track", choices=("equal_updates", "equal_compiler_flops"), default="equal_updates"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        help="complete hard artifact roots for the aggregate command",
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        choices=("equal_updates", "equal_compiler_flops"),
        help="exact hard tracks requested; omission means both registered tracks",
    )
    parser.add_argument(
        "--rederive-compute",
        action="store_true",
        help="recompile cost evidence; must run on the original homogeneous device runtime",
    )
    parser.add_argument(
        "--authentication-only",
        action="store_true",
        help="skip checkpoint and DMC replay (use only after worker verification)",
    )
    arguments = parser.parse_args()
    protocol = load_protocol()
    if arguments.command == "contract":
        print(json.dumps(protocol, indent=2, sort_keys=True))
        print("PIXEL_BENCHMARK_CONTRACT_VERIFIED")
        return 0
    if arguments.command == "aggregate":
        if not arguments.inputs or not arguments.output:
            parser.error("aggregate requires --inputs and an --output JSON path")
        summary = aggregate_artifacts(
            arguments.inputs,
            Path(arguments.output),
            tracks=arguments.tracks,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        print("PIXEL_BENCHMARK_AGGREGATE_COMPLETE")
        return 0
    if arguments.command == "verify-aggregate":
        if not arguments.output:
            parser.error("verify-aggregate requires --output pointing to the summary JSON")
        summary = verify_aggregate(Path(arguments.output), arguments.inputs)
        print(json.dumps(summary["tracks"], indent=2, sort_keys=True))
        print("PIXEL_BENCHMARK_AGGREGATE_VERIFIED")
        return 0
    profile = protocol["profiles"][arguments.profile]
    task = arguments.task or profile["tasks"][0]
    seed = int(arguments.seed if arguments.seed is not None else profile["seeds"][0])
    output = Path(
        arguments.output
        or (
            "/private/tmp/trajectory_imf_pixel_smoke_v1"
            if arguments.profile == "smoke"
            else PROJECT
            / "results"
            / "pixel_hard"
            / arguments.track
            / task
            / f"seed_{seed}"
        )
    ).expanduser().absolute()
    if arguments.command == "run":
        result = run_benchmark(
            output,
            profile=arguments.profile,
            task=task,
            seed=seed,
            track=arguments.track,
        )
        token = "PIXEL_BENCHMARK_RUN_COMPLETE"
    else:
        result = verify_artifact(
            output,
            recompute_predictions=not arguments.authentication_only,
            rederive_compute=arguments.rederive_compute,
        )
        token = "PIXEL_BENCHMARK_ARTIFACT_VERIFIED"
    concise = {
        "output": str(output),
        "profile": result["profile"],
        "task": result["task"],
        "seed": result["seed"],
        "track": result["track"],
        "claim_eligible_for_primary_gate": result["claim_eligible_for_primary_gate"],
        "primary_nfe_auc": {
            arm: result["metrics"]["primary_nfe"][arm][
                "normalized_visual_mse_auc"
            ]
            for arm in ("shortcut_forcing", "trajectory_imf")
        },
    }
    print(json.dumps(concise, indent=2, sort_keys=True))
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
