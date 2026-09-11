from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
for path in (PROJECT, WORKSPACE / "imf_dreamer_jax" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dreamer_imf_compare.artifacts import read_json, write_json_atomic
import dreamer_imf_compare.matched_objective_benchmark as benchmark


class RolloutReplayValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = read_json(PROJECT / "matched_objective_protocol.json")
        source = {"source_sha256": benchmark.object_sha256({})}
        cls.matrix = benchmark.build_matrix(
            cls.protocol, "smoke", source_manifest=source
        )
        cls.cell = next(
            cell
            for cell in cls.matrix["cells"]
            if cell["stage"] == "rollout"
            and cell["arm"] == "shortcut_forcing"
            and cell["nfe"] == 1
        )
        cls.runtime = {
            "python": "3.12.0",
            "jax_version": "test",
            "jaxlib_version": "test",
            "backend": "cpu",
            "device_platforms": ["cpu"],
            "device_kinds": ["test-cpu"],
            "devices": ["TFRT_CPU_0"],
            "visible_device_count": 1,
            "xla_platform_version": "test",
            "jax_enable_x64": False,
            "cuda_visible_devices": None,
        }

    def _fixture(self, root: Path):
        cell = self.cell
        dataset_cell = benchmark._dependency_cell(self.matrix, cell, "dataset")
        world_cell = benchmark._dependency_cell(self.matrix, cell, "world_model")
        compute_cell = benchmark._dependency_cell(self.matrix, cell, "compute_plan")
        dataset_directory = benchmark.stage_directory(root, dataset_cell)
        world_directory = benchmark.stage_directory(root, world_cell)
        compute_directory = benchmark.stage_directory(root, compute_cell)
        dataset_directory.mkdir(parents=True)
        world_directory.mkdir(parents=True)
        compute_directory.mkdir(parents=True)

        benchmark._write_npz_atomic(
            dataset_directory / "dataset.npz",
            {"placeholder": np.asarray([1.0], dtype=np.float32)},
        )
        dataset_result = {
            "matrix_sha256": self.matrix["matrix_sha256"],
            "dataset_sha256": "d" * 64,
            "observation_shape": [1],
            "action_dim": 1,
        }
        write_json_atomic(dataset_directory / "result.json", dataset_result)
        checkpoint_path = world_directory / "checkpoint.pkl"
        checkpoint_path.write_bytes(b"test-world-checkpoint")
        world_result = {
            "matrix_sha256": self.matrix["matrix_sha256"],
            "checkpoint_sha256": benchmark.file_sha256(checkpoint_path),
            "runtime_config_sha256": "w" * 64,
        }
        write_json_atomic(world_directory / "result.json", world_result)
        compute_result = {"runtime": self.runtime}
        write_json_atomic(compute_directory / "result.json", compute_result)

        config = benchmark.make_config(
            self.protocol,
            cell["profile"],
            cell["arm"],
            (1,),
            1,
            nfe=int(cell["nfe"]),
            output_root=root,
            candidate=benchmark._candidate_for_cell(self.matrix, cell),
        )
        draws = int(
            self.protocol["profiles"][cell["profile"]][
                "predictive_draws_per_window"
            ]
        )
        windows_count = int(
            self.protocol["profiles"][cell["profile"]][
                "rollout_windows_per_task_world_seed"
            ]
        )
        horizon = max(self.protocol["evaluation"]["rollout_horizons"])
        context = int(self.protocol["evaluation"]["rollout_context_length"])
        windows = {
            "target_observations": np.zeros(
                (windows_count, horizon, 1), dtype=np.float32
            ),
            "target_rewards": np.zeros(
                (windows_count, horizon), dtype=np.float32
            ),
            "target_continuations": np.ones(
                (windows_count, horizon), dtype=np.float32
            ),
            "context_observations": np.zeros(
                (windows_count, context, 1), dtype=np.float32
            ),
            "context_actions": np.zeros(
                (windows_count, context, 1), dtype=np.float32
            ),
            "future_actions": np.zeros(
                (windows_count, horizon, 1), dtype=np.float32
            ),
            "noise": np.zeros(
                (draws, horizon, windows_count, config.stochastic_dim),
                dtype=np.float32,
            ),
            "episode_ids": np.arange(windows_count, dtype=np.int32),
            "anchors": np.full(windows_count, context - 1, dtype=np.int32),
            "training_observation_std": np.ones((1,), dtype=np.float32),
        }
        replayed = {
            "observation_samples": np.full(
                (draws, windows_count, horizon, 1), 0.5, dtype=np.float32
            ),
            "reward_samples": np.full(
                (draws, windows_count, horizon), 0.25, dtype=np.float32
            ),
            "continuation_samples": np.full(
                (draws, windows_count, horizon), 0.75, dtype=np.float32
            ),
        }
        return (
            dataset_cell,
            world_cell,
            compute_cell,
            dataset_result,
            world_result,
            config,
            windows,
            replayed,
        )

    def _raw(self, windows, samples):
        return {
            **samples,
            **windows,
        }

    def _result(
        self,
        root: Path,
        dataset_cell,
        world_cell,
        dataset_result,
        world_result,
        config,
        raw_path: Path,
        raw,
    ):
        metrics = benchmark.normalized_rollout_statistics(
            raw["observation_samples"],
            raw["reward_samples"],
            raw,
            self.protocol["evaluation"]["rollout_horizons"],
            raw["continuation_samples"],
        )
        runtime_config = asdict(config)
        return {
            **benchmark._result_identity(self.cell, benchmark.ROLLOUT_SCHEMA),
            "matrix_sha256": self.matrix["matrix_sha256"],
            "world_model_cell_id": world_cell["cell_id"],
            "dataset_cell_id": dataset_cell["cell_id"],
            "world_model_checkpoint_sha256": world_result["checkpoint_sha256"],
            "world_model_result_file_sha256": benchmark.file_sha256(
                benchmark.stage_directory(root, world_cell) / "result.json"
            ),
            "world_model_runtime_config_sha256": world_result[
                "runtime_config_sha256"
            ],
            "dataset_sha256": dataset_result["dataset_sha256"],
            "dataset_result_file_sha256": benchmark.file_sha256(
                benchmark.stage_directory(root, dataset_cell) / "result.json"
            ),
            "runtime_config": runtime_config,
            "runtime_config_sha256": benchmark.object_sha256(runtime_config),
            "structural_nfe_per_transition": int(self.cell["nfe"]),
            "windows": int(raw["target_observations"].shape[0]),
            "predictive_draws_per_window": int(
                raw["observation_samples"].shape[0]
            ),
            **metrics,
            "raw_predictive_draws_sha256": benchmark.file_sha256(raw_path),
            "inference_replay_comparison": (
                benchmark._rollout_replay_comparison_contract()
            ),
            "runtime": self.runtime,
            "wall_seconds": 1.0,
        }

    def test_favorable_rehashed_draws_fail_checkpoint_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                dataset_cell,
                world_cell,
                _,
                dataset_result,
                world_result,
                config,
                windows,
                replayed,
            ) = self._fixture(root)
            raw_path = (
                benchmark.stage_directory(root, self.cell) / "predictive_draws.npz"
            )
            canonical_raw = self._raw(windows, replayed)
            benchmark._write_npz_atomic(raw_path, canonical_raw)
            canonical_result = self._result(
                root,
                dataset_cell,
                world_cell,
                dataset_result,
                world_result,
                config,
                raw_path,
                canonical_raw,
            )
            patches = (
                mock.patch.object(benchmark, "validate_dataset_result"),
                mock.patch.object(benchmark, "validate_world_result"),
                mock.patch.object(benchmark, "validate_compute_plan_cell"),
                mock.patch.object(benchmark, "_validate_compute_files"),
                mock.patch.object(
                    benchmark, "_rollout_windows", return_value=windows
                ),
                mock.patch.object(
                    benchmark, "_replay_rollout_inference", return_value=replayed
                ),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                benchmark.validate_rollout_result(
                    canonical_result,
                    self.cell,
                    raw_path,
                    self.protocol,
                    self.matrix,
                    root,
                )

                fabricated_samples = {
                    "observation_samples": windows["target_observations"][
                        None
                    ].repeat(replayed["observation_samples"].shape[0], axis=0),
                    "reward_samples": windows["target_rewards"][None].repeat(
                        replayed["reward_samples"].shape[0], axis=0
                    ),
                    "continuation_samples": windows["target_continuations"][
                        None
                    ].repeat(replayed["continuation_samples"].shape[0], axis=0),
                }
                fabricated_raw = self._raw(windows, fabricated_samples)
                benchmark._write_npz_atomic(raw_path, fabricated_raw)
                fabricated_result = self._result(
                    root,
                    dataset_cell,
                    world_cell,
                    dataset_result,
                    world_result,
                    config,
                    raw_path,
                    fabricated_raw,
                )
                self.assertLess(
                    fabricated_result["normalized_free_running_rollout_error_auc"],
                    canonical_result["normalized_free_running_rollout_error_auc"],
                )
                with self.assertRaisesRegex(
                    ValueError, "does not match checkpoint replay"
                ):
                    benchmark.validate_rollout_result(
                        fabricated_result,
                        self.cell,
                        raw_path,
                        self.protocol,
                        self.matrix,
                        root,
                    )

    def test_real_jax_checkpoint_replay_returns_all_three_heads(self) -> None:
        import jax
        from imf_dreamer_jax import DreamerConfig, create_agent, save_checkpoint

        config = DreamerConfig(
            observation_shape=(1,),
            action_dim=1,
            deterministic_dim=8,
            stochastic_dim=4,
            embedding_dim=8,
            hidden_dim=8,
            prior="shortcut",
            shortcut_training_k_max=4,
            shortcut_sampling_steps=1,
            burn_in=1,
            overshooting_scale=0.0,
        )
        state = create_agent(config, jax.random.key(11))
        world_cell = {"cell_id": "world-model-test"}
        arrays = {
            "context_observations": np.zeros((2, 2, 1), dtype=np.float32),
            "context_actions": np.zeros((2, 2, 1), dtype=np.float32),
            "future_actions": np.zeros((2, 3, 1), dtype=np.float32),
            "noise": np.asarray(
                jax.random.normal(jax.random.key(12), (4, 3, 2, 4)),
                dtype=np.float32,
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_checkpoint(
                root / "checkpoint.pkl",
                state,
                config,
                metadata={"cell_id": world_cell["cell_id"]},
            )
            replayed = benchmark._replay_rollout_inference(
                root, world_cell, arrays, config
            )
        self.assertEqual(
            replayed["observation_samples"].shape, (4, 2, 3, 1)
        )
        self.assertEqual(replayed["reward_samples"].shape, (4, 2, 3))
        self.assertEqual(replayed["continuation_samples"].shape, (4, 2, 3))
        self.assertTrue(
            all(np.isfinite(value).all() for value in replayed.values())
        )


if __name__ == "__main__":
    unittest.main()
