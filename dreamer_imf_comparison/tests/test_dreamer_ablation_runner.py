"""CPU integration is opt-in; it executes the actual upstream learner per arm."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dreamer_imf_compare.dreamer_ablation_runner import (
    _settings,
    _runtime_telemetry,
    run_cell,
)
from types import SimpleNamespace
import numpy as np


class RunnerContractTest(unittest.TestCase):
    def test_production_and_full_geometry_preflight(self):
        prod = _settings({}, False)
        smoke = _settings({}, True)
        self.assertEqual(prod["native_steps"], 500000)
        for field in ("model_size", "batch_size", "batch_length", "envs"):
            self.assertEqual(prod[field], smoke[field])
        self.assertEqual(smoke["max_updates"], 2)

    def test_reject_wrong_budget_and_arm(self):
        with self.assertRaises(ValueError):
            _settings({"native_steps": 32}, False)
        with self.assertRaises(ValueError):
            _settings(
                {"preflight": {"eval_at_native_steps": [3], "native_steps": 3}}, True
            )
        with self.assertRaises(ValueError):
            run_cell("/does/not/exist", "rebrac", 431)

    def test_cuda_assertion_and_model_parameter_count(self):
        cpu = SimpleNamespace(
            id=0,
            platform="cpu",
            device_kind="cpu",
            process_index=0,
            client=SimpleNamespace(platform_version="cpu"),
        )
        agent = SimpleNamespace(
            train_devices=[cpu],
            model=SimpleNamespace(modules=[SimpleNamespace(path="dyn")]),
            params={"dyn/kernel": np.zeros((2, 3)), "opt/state": np.zeros(12)},
        )
        result = _runtime_telemetry(agent, "cpu")
        self.assertEqual(result["model_parameter_count"], 6)
        with self.assertRaises(RuntimeError):
            _runtime_telemetry(agent, "cuda")

    @unittest.skipUnless(
        os.environ.get("DREAMER_RUN_INTEGRATION") == "1",
        "opt-in genuine Agent integration",
    )
    def test_actual_agent_all_arms(self):
        repo = Path(__file__).resolve().parents[2]
        for arm in ("categorical", "gaussian", "imf"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory(
                prefix="dreamer-runner-test-"
            ) as root:
                path = Path(root)
                protocol = {
                    "upstream_path": os.environ.get(
                        "DREAMER_UPSTREAM", "/private/tmp/dreamerv3-ablation-upstream"
                    ),
                    "preflight": {
                        "model_size": "debug",
                        "jax_platform": "cpu",
                        "batch_size": 1,
                        "batch_length": 4,
                        "envs": 1,
                        "native_steps": 32,
                        "eval_at_native_steps": [32],
                        "episode_native_steps": 10,
                        "eval_episodes": 2,
                        "train_ratio": 2,
                        "max_updates": 2,
                    },
                }
                if arm == "categorical":
                    # Deliberately non-batch-aligned checkpoints exercise carry
                    # selection/merge and continuation after held-out evaluation.
                    protocol["preflight"].update(envs=3, eval_at_native_steps=[14, 32])
                (path / "protocol.json").write_text(json.dumps(protocol))
                env = dict(
                    os.environ,
                    PYTHONPATH=os.pathsep.join(
                        [
                            str(repo / "dreamer_imf_comparison"),
                            str(repo / "imf_dreamer_jax/src"),
                        ]
                    ),
                    MUJOCO_GL="disable",
                )
                code = "from dreamer_imf_compare.dreamer_ablation_runner import run_cell; import sys; run_cell(sys.argv[1],sys.argv[2],431,True)"
                proc = subprocess.run(
                    [sys.executable, "-c", code, root, arm],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=240,
                )
                self.assertEqual(
                    proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-7000:]
                )
                cell = path / "preflight" / arm / "seed_431"
                result = json.loads((cell / "complete.json").read_text())
                self.assertEqual(result["native_steps"], 32)
                self.assertEqual(result["agent_transitions"], 16)
                self.assertEqual(result["learner_updates"], 2)
                self.assertEqual(
                    result["heldout_native_steps"], 40 if arm == "categorical" else 20
                )
                self.assertEqual(result["reset_rows"], 4)
                self.assertEqual(result["replay_rows"], 20)
                self.assertTrue(result["finite_parameters"])
                runtime = result["runtime"]
                self.assertEqual(runtime["devices"][0]["platform"], "cpu")
                self.assertGreater(runtime["model_parameter_count"], 0)
                self.assertLess(
                    runtime["model_parameter_count"],
                    result["parameter_elements_including_optimizer"],
                )
                self.assertEqual(
                    [x["update"] for x in result["update_timing_samples"]], [1, 2]
                )
                self.assertTrue(
                    all(
                        x["seconds"] > 0 and x["synchronized"]
                        for x in result["update_timing_samples"]
                    )
                )
                progress = json.loads((cell / "progress.json").read_text())
                self.assertEqual(progress["native_steps"], 32)
                self.assertEqual(
                    progress["model_parameter_count"], runtime["model_parameter_count"]
                )
                self.assertEqual(len(result["evaluations"][0]["returns"]), 2)
                # Even a complete result is not overwritten by accidental retry.
                with self.assertRaises(FileExistsError):
                    run_cell(path, arm, 431, True)


if __name__ == "__main__":
    unittest.main()
