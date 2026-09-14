#!/usr/bin/env python3
"""Fail-closed checks for the ReBRAC plus ITPO controller study."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
for path in (PROJECT, WORKSPACE / "imf_dreamer_jax" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dreamer_imf_compare import flowmpc_actor_study as study  # noqa: E402


def verify_implementation() -> None:
    training = inspect.getsource(study.train_rebrac_cell)
    control = inspect.getsource(study._run_controller)
    manifest = inspect.getsource(study.build_manifest)
    required_training = (
        "init_rebrac_state(",
        "jit_train_rebrac_chunk(",
        '"flowmpc-rebrac-init"',
        '"flowmpc-rebrac-training"',
    )
    required_control = (
        "jit_flowmpc_adapt_actor(",
        "actor = update.actor",
        "actor = rebrac_state.actor",
        "belief = observe_function(",
        "jax.random.fold_in(noise_key, step)",
        "if adapted and not controller_warmed:",
        "jax.block_until_ready(warm_action)",
        "controller_warmed = True",
    )
    required_manifest = (
        '"old_learned_actors_reused": False',
        '"primary_comparison": "flowmpc_minus_same_rebrac_zero_shot"',
        '"existing trajectory-iMF dynamics were not trained with FlowMPC policy-tilted weighting"',
    )
    missing = [token for token in required_training if token not in training]
    missing += [token for token in required_control if token not in control]
    missing += [token for token in required_manifest if token not in manifest]
    if missing:
        raise ValueError(f"FlowMPC study fidelity tokens missing: {missing}")
    print("FLOWMPC_ACTOR_STUDY_IMPLEMENTATION_VERIFIED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--implementation", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--output-root", type=Path)
    arguments = parser.parse_args()
    if sum(
        map(
            bool,
            (
                arguments.self_test,
                arguments.implementation,
                arguments.gpu,
                arguments.final,
            ),
        )
    ) != 1:
        parser.error("select exactly one verification mode")
    if arguments.self_test:
        study.self_test()
        print("FLOWMPC_ACTOR_STUDY_CONTRACT_VERIFIED")
    elif arguments.implementation:
        verify_implementation()
    elif arguments.gpu:
        import jax

        devices = jax.devices()
        if len(devices) != 1 or devices[0].platform != "gpu":
            raise ValueError("FlowMPC study requires exactly one GPU per process")
        print("FLOWMPC_ACTOR_STUDY_GPU_VERIFIED")
    else:
        if arguments.output_root is None:
            parser.error("--final requires --output-root")
        report = study.validate_final(arguments.output_root)
        print(report["report_sha256"])
        print("FLOWMPC_ACTOR_STUDY_FINAL_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
