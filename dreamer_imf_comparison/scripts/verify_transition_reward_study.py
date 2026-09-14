#!/usr/bin/env python3
"""Fail-closed checks for the iMF ITPO-style state-action reward study."""

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

from dreamer_imf_compare import transition_reward_study as study  # noqa: E402


def verify_implementation() -> None:
    reward_source = inspect.getsource(study.train_reward_cell)
    actor_source = inspect.getsource(study.train_actor_cell)
    required_reward = (
        "init_transition_reward_state(",
        "jit_train_transition_reward_step(",
        "attach_transition_reward_head(",
        "_training_observation_statistics(arrays)",
        'hidden_dim=int(manifest["reward_hidden_dim"])',
        "source_world_model_parameter_delta",
    )
    required_actor = (
        "create_agent(",
        "jit_train_behavior_cloning(",
        "jit_train_replay_critic(",
        "jit_train_actor_critic_dreamer3(",
        "behavior_prior=None",
        "frozen_world_digest",
    )
    if any(token not in reward_source for token in required_reward):
        raise ValueError("state-action reward implementation lost an isolation operation")
    if any(token not in actor_source for token in required_actor):
        raise ValueError("state-action reward actor lost a corrected training operation")
    print("IMF_TRANSITION_REWARD_IMPLEMENTATION_VERIFIED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--implementation", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--corrected-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--final", action="store_true")
    arguments = parser.parse_args()
    selected = sum(
        bool(value)
        for value in (
            arguments.self_test,
            arguments.implementation,
            arguments.gpu,
            arguments.final,
        )
    )
    if selected != 1:
        parser.error("select exactly one verification mode")
    if arguments.self_test:
        study.self_test()
        print("IMF_TRANSITION_REWARD_STUDY_CONTRACT_VERIFIED")
    elif arguments.implementation:
        verify_implementation()
    elif arguments.gpu:
        import jax

        devices = jax.devices()
        if len(devices) != 1 or devices[0].platform != "gpu":
            raise ValueError("transition reward preflight requires exactly one GPU")
        print("IMF_TRANSITION_REWARD_GPU_VERIFIED")
    else:
        if arguments.output_root is None:
            parser.error("--final requires --output-root")
        report = study.validate_final(arguments.output_root)
        print(report["report_sha256"])
        print("IMF_TRANSITION_REWARD_FINAL_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
