#!/usr/bin/env python3
"""Fail-closed implementation checks for ReBRAC and FlowMPC."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "imf_dreamer_jax" / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from imf_dreamer_jax import flowmpc as implementation  # noqa: E402


def run_tests(pattern: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "imf_dreamer_jax/tests",
            "-p",
            pattern,
        ],
        cwd=ROOT,
        check=True,
    )


def verify_rebrac() -> None:
    source = inspect.getsource(implementation)
    required = (
        "hidden_layers: int = 3",
        "hidden_dim: int = 256",
        "batch_size: int = 1024",
        "target_update_rate: float = 5e-3",
        "policy_frequency: int = 2",
        "normalize_q: bool = True",
        "next_q = next_q - config.critic_bc_coefficient * critic_bc",
        "config.actor_bc_coefficient * bc_penalty - q_scale * q_values",
        "state.step % config.policy_frequency == 0",
        "state.target_actor, state.actor, config.target_update_rate",
        "state.target_critics, critics, config.target_update_rate",
    )
    missing = [token for token in required if token not in source]
    if missing:
        raise ValueError(f"ReBRAC fidelity tokens missing: {missing}")
    run_tests("test_flowmpc.py")
    print("FLOWMPC_REBRAC_VERIFIED")


def verify_controller() -> None:
    source = inspect.getsource(implementation.flowmpc_objective)
    update_source = inspect.getsource(implementation.flowmpc_adapt_actor)
    required_objective = (
        "predict_state_action_reward_from_observation(",
        "transition_deterministic(",
        "sample_prior(",
        "noise=noise",
        "terminal_q = jnp.min(",
        "stage_return + terminal_value",
    )
    required_update = (
        "jax.grad(objective)",
        "parameter + flowmpc_config.step_size * gradient",
        "actor_params",
    )
    missing = [
        token
        for token in required_objective
        if token not in source
    ] + [token for token in required_update if token not in update_source]
    forbidden = ("reinforce", "log_prob", "entropy", "adam_update")
    present = [token for token in forbidden if token in update_source.lower()]
    if missing or present:
        raise ValueError(
            f"FlowMPC fidelity check failed: missing={missing}, forbidden={present}"
        )
    run_tests("test_flowmpc.py")
    print("FLOWMPC_CONTROLLER_VERIFIED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebrac", action="store_true")
    parser.add_argument("--controller", action="store_true")
    arguments = parser.parse_args()
    if arguments.rebrac == arguments.controller:
        parser.error("select exactly one verification mode")
    if arguments.rebrac:
        verify_rebrac()
    else:
        verify_controller()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
