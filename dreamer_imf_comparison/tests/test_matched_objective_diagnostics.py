from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
for path in (PROJECT, WORKSPACE / "imf_dreamer_jax" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dreamer_imf_compare.artifacts import read_json
import dreamer_imf_compare.matched_objective_diagnostics as diagnostics
from dreamer_imf_compare.matched_objective_benchmark import (
    object_sha256,
    pilot_candidates,
    validate_confirmatory_auxiliary_evidence,
)
from dreamer_imf_compare.matched_objective_protocol import ARM_ORDER, protocol_digest


class MatchedObjectiveDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = read_json(PROJECT / "matched_objective_protocol.json")
        cls.matrix = {
            "protocol_sha256": protocol_digest(cls.protocol),
            "source_sha256": "1" * 64,
            "matrix_sha256": "2" * 64,
            "selection_sha256": "3" * 64,
            "claim_eligible": True,
            "selection_manifest": {
                "selected": {
                    arm: pilot_candidates(cls.protocol, arm)[0] for arm in ARM_ORDER
                }
            },
            "cells": [],
        }
        cls.runtime = {
            "python": "3.12.0",
            "python_executable": "/frozen/venv/bin/python",
            "python_executable_sha256": "a" * 64,
            "platform_system": "Linux",
            "platform_release": "test-kernel",
            "platform_machine": "x86_64",
            "numpy_version": "test",
            "jax_version": "test",
            "jaxlib_version": "test",
            "dm_control_version": "test",
            "mujoco_version": "test",
            "environment_package_count": 5,
            "environment_packages_sha256": "b" * 64,
            "backend": "cpu",
            "device_platforms": ["cpu"],
            "device_kinds": ["test-cpu"],
            "devices": ["TFRT_CPU_0"],
            "visible_device_count": 1,
            "xla_platform_version": "test",
            "jax_enable_x64": False,
            "cuda_visible_devices": None,
            "jax_enable_compilation_cache": None,
            "jax_compilation_cache_dir": None,
            "jax_persistent_cache_min_compile_time_secs": None,
            "jax_persistent_cache_min_entry_size_bytes": None,
            "jax_persistent_cache_enable_xla_caches": None,
            "jax_raise_persistent_cache_errors": None,
        }
        cls.runtime_identity = diagnostics.runtime_homogeneity_identity(cls.runtime)
        cls.runtime_patcher = mock.patch.object(
            diagnostics,
            "runtime_fingerprint",
            side_effect=lambda: copy.deepcopy(cls.runtime),
        )
        cls.compute_runtime_patcher = mock.patch.object(
            diagnostics,
            "_frozen_compute_runtime_identity",
            side_effect=lambda *args: copy.deepcopy(cls.runtime_identity),
        )
        cls.runtime_patcher.start()
        cls.compute_runtime_patcher.start()
        cls.addClassCleanup(cls.compute_runtime_patcher.stop)
        cls.addClassCleanup(cls.runtime_patcher.stop)

    def deterministic_raw(self, error: float = 0.01) -> dict[str, np.ndarray]:
        evaluation = diagnostics.generate_scalar_evaluation_data(
            diagnostics.DETERMINISTIC_NAME,
            conditions=12,
            episode_length=64,
            repeated_futures=1,
            seed=101,
        )
        target = evaluation["target_futures"][:, :30]
        return {
            "horizons": np.arange(1, 31, dtype=np.int32),
            "initial_states": evaluation["initial_states"],
            "future_actions": evaluation["future_actions"][:, :30],
            "target_futures": target,
            "model_futures": target + np.float32(error),
            "training_observation_std": np.asarray([1.0], dtype=np.float64),
            "test_seed": np.asarray([101], dtype=np.uint32),
            "model_noise_key": np.asarray([11, 12], dtype=np.uint32),
            "ratio_epsilon": np.asarray([diagnostics.RATIO_EPSILON], dtype=np.float64),
        }

    @staticmethod
    def checkpoint_trace_row(update: int, value: float = 0.0) -> dict[str, float | int]:
        return {
            "update": update,
            **{
                name: float(value)
                for name in diagnostics.CHECKPOINT_TRACE_METRIC_FIELDS
            },
        }

    @staticmethod
    def checkpoint_state(
        model_step: int, *, actor_step: int = 0, critic_step: int = 0
    ) -> SimpleNamespace:
        def optimizer(step: int) -> SimpleNamespace:
            return SimpleNamespace(step=np.asarray(step, dtype=np.int32))

        return SimpleNamespace(
            model_optimizer=optimizer(model_step),
            actor_optimizer=optimizer(actor_step),
            critic_optimizer=optimizer(critic_step),
        )

    def stochastic_raw(self) -> dict[str, np.ndarray]:
        evaluation = diagnostics.generate_scalar_evaluation_data(
            diagnostics.STOCHASTIC_NAME,
            conditions=32,
            episode_length=64,
            repeated_futures=64,
            seed=202,
            oracle_reference_futures=64,
            evaluation_futures=64,
        )
        oracle = evaluation["oracle_reference_futures"][:, :, :30]
        # A deterministic draw permutation preserves the empirical predictive
        # reference distribution while ensuring the arrays are distinct.
        model = np.roll(oracle, shift=1, axis=1)
        return {
            "horizons": np.arange(1, 31, dtype=np.int32),
            "score_horizons": np.asarray((1, 8, 30), dtype=np.int32),
            "rollout_horizons": np.asarray((1, 2, 4, 8, 15, 30), dtype=np.int32),
            "initial_states": evaluation["initial_states"],
            "future_actions": evaluation["future_actions"][:, :30],
            "oracle_reference_futures": oracle,
            "evaluation_futures": evaluation["evaluation_futures"][:, :, :30],
            "evaluation_branches": evaluation["evaluation_branches"][:, :, :30],
            "model_futures": model,
            "training_observation_std": np.asarray([1.0], dtype=np.float64),
            "ece_bin_edges": np.linspace(0.0, 1.0, 11, dtype=np.float64),
            "test_seed": np.asarray([202], dtype=np.uint32),
            "model_noise_key": np.asarray([21, 22], dtype=np.uint32),
        }

    def test_deterministic_estimands_and_thresholds_recompute(self) -> None:
        estimands, decisions, passed = diagnostics.deterministic_estimands_from_raw(
            self.deterministic_raw()
        )
        spec = self.protocol["diagnostics"][diagnostics.DETERMINISTIC_NAME]
        self.assertEqual(set(estimands), set(spec["estimands"]))
        self.assertEqual(set(decisions), set(spec["interpretation_thresholds"]))
        self.assertAlmostEqual(estimands["one_step_normalized_RMSE"], 0.01, places=6)
        self.assertAlmostEqual(
            estimands["horizon_30_to_horizon_1_error_ratio"], 1.0, places=5
        )
        self.assertTrue(passed)

    def test_deterministic_compounding_failure_is_detected(self) -> None:
        raw = self.deterministic_raw(error=0.0)
        raw["model_futures"] = raw["target_futures"].copy()
        raw["model_futures"][:, 0] += 0.01
        raw["model_futures"][:, 29] += 0.3
        _, decisions, passed = diagnostics.deterministic_estimands_from_raw(raw)
        self.assertFalse(decisions["horizon_30_normalized_RMSE_max"])
        self.assertFalse(decisions["horizon_30_to_horizon_1_error_ratio_max"])
        self.assertFalse(passed)

    def test_stochastic_estimands_are_finite_and_complete(self) -> None:
        estimands, decisions, passed = diagnostics.stochastic_estimands_from_raw(
            self.stochastic_raw()
        )
        spec = self.protocol["diagnostics"][diagnostics.STOCHASTIC_NAME]
        self.assertEqual(set(estimands), set(spec["estimands"]))
        self.assertEqual(set(decisions), set(spec["interpretation_thresholds"]))
        self.assertTrue(all(np.isfinite(value) for value in estimands.values()))
        self.assertIsInstance(passed, bool)

    def test_raw_schema_rejects_missing_extra_empty_and_nonfinite_fields(self) -> None:
        cases = []
        missing = self.deterministic_raw()
        missing.pop("target_futures")
        cases.append(missing)
        extra = self.deterministic_raw()
        extra["unregistered"] = np.ones(1)
        cases.append(extra)
        empty = self.deterministic_raw()
        empty["model_futures"] = np.empty((0, 30))
        cases.append(empty)
        nonfinite = self.deterministic_raw()
        nonfinite["model_futures"][0, 0] = np.nan
        cases.append(nonfinite)
        for raw in cases:
            with self.subTest(fields=set(raw)):
                with self.assertRaises(ValueError):
                    diagnostics.deterministic_estimands_from_raw(raw)

    def test_generators_are_seed_deterministic_and_stochastic_repeats_are_real(self) -> None:
        first = diagnostics.generate_scalar_training_data(
            diagnostics.STOCHASTIC_NAME, episodes=8, episode_length=64, seed=303
        )
        second = diagnostics.generate_scalar_training_data(
            diagnostics.STOCHASTIC_NAME, episodes=8, episode_length=64, seed=303
        )
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        evaluation = diagnostics.generate_scalar_evaluation_data(
            diagnostics.STOCHASTIC_NAME,
            conditions=4,
            episode_length=64,
            repeated_futures=64,
            seed=304,
            oracle_reference_futures=64,
            evaluation_futures=64,
        )
        self.assertEqual(evaluation["evaluation_futures"].shape, (4, 64, 64))
        self.assertGreater(
            int(np.unique(evaluation["evaluation_futures"][0, :, 0]).size), 1
        )

    def test_registered_and_smoke_plans_preserve_claim_boundary(self) -> None:
        registered = diagnostics.resolve_diagnostic_plan(self.protocol)
        deterministic = self.protocol["diagnostics"][diagnostics.DETERMINISTIC_NAME]
        stochastic = self.protocol["diagnostics"][diagnostics.STOCHASTIC_NAME]
        self.assertFalse(registered.smoke_nonclaim)
        self.assertEqual(registered.deterministic_updates, deterministic["world_model_updates"])
        self.assertEqual(registered.stochastic_updates, stochastic["world_model_updates"])
        smoke = diagnostics.resolve_diagnostic_plan(
            self.protocol,
            smoke=True,
            smoke_updates=1,
            smoke_training_episodes=4,
            smoke_test_conditions=4,
            smoke_repeated_futures=4,
            smoke_batch_size=2,
        )
        self.assertTrue(smoke.smoke_nonclaim)
        self.assertEqual(smoke.status, "smoke_nonclaim")
        self.assertEqual(smoke.deterministic_repeated_futures, 1)
        self.assertEqual(smoke.stochastic_predictive_draws, 4)
        with self.assertRaises(ValueError):
            diagnostics.resolve_diagnostic_plan(
                self.protocol, smoke=True, smoke_repeated_futures=3
            )

    def _write_all_raw(self, root: Path) -> dict[str, dict[str, Path]]:
        paths: dict[str, dict[str, Path]] = {}
        payloads = {
            diagnostics.DETERMINISTIC_NAME: self.deterministic_raw(),
            diagnostics.STOCHASTIC_NAME: self.stochastic_raw(),
        }
        for diagnostic_name, raw in payloads.items():
            paths[diagnostic_name] = {}
            for arm in ARM_ORDER:
                path = root / "diagnostics" / diagnostic_name / arm / "evaluation_raw.npz"
                diagnostics._write_npz_once(path, raw)
                paths[diagnostic_name][arm] = path
        return paths

    def _write_registered_evidence(
        self, root: Path
    ) -> dict[str, dict[str, Path]]:
        """Create a full-count, internally authenticated fixture without training."""

        import jax

        plan = diagnostics.resolve_diagnostic_plan(self.protocol)
        raw_paths: dict[str, dict[str, Path]] = {
            name: {} for name in diagnostics.DIAGNOSTIC_NAMES
        }
        for diagnostic_name in diagnostics.DIAGNOSTIC_NAMES:
            episodes, conditions, draws, updates = diagnostics._plan_counts(
                plan, diagnostic_name
            )
            directory = root / "diagnostics" / diagnostic_name
            train_seed = diagnostics.derive_seed(
                "matched-objective-diagnostic-train",
                self.matrix["matrix_sha256"],
                diagnostic_name,
            )
            schedule_seed = diagnostics.derive_seed(
                "matched-objective-diagnostic-minibatches",
                self.matrix["matrix_sha256"],
                diagnostic_name,
            )
            test_seed = diagnostics.derive_seed(
                "matched-objective-diagnostic-test",
                self.matrix["matrix_sha256"],
                diagnostic_name,
            )
            dataset = diagnostics.generate_scalar_training_data(
                diagnostic_name,
                episodes=episodes,
                episode_length=plan.episode_length,
                seed=train_seed,
            )
            dataset_path = diagnostics._write_npz_once(
                directory / "training_dataset.npz", dataset
            )
            schedule_path = diagnostics._write_npz_once(
                directory / "minibatch_schedule.npz",
                diagnostics._batch_schedule(
                    schedule_seed, updates, plan.batch_size, episodes
                ),
            )
            evaluation = diagnostics.generate_scalar_evaluation_data(
                diagnostic_name,
                conditions=conditions,
                episode_length=plan.episode_length,
                repeated_futures=draws,
                seed=test_seed,
                oracle_reference_futures=(
                    plan.stochastic_oracle_reference_futures
                    if diagnostic_name == diagnostics.STOCHASTIC_NAME
                    else None
                ),
                evaluation_futures=(
                    plan.stochastic_evaluation_futures
                    if diagnostic_name == diagnostics.STOCHASTIC_NAME
                    else None
                ),
            )
            model_noise_key = diagnostics.derive_jax_key(
                "matched-objective-diagnostic-model-noise",
                self.matrix["matrix_sha256"],
                diagnostic_name,
            )
            common = {
                "horizons": np.arange(1, 31, dtype=np.int32),
                "initial_states": evaluation["initial_states"],
                "future_actions": evaluation["future_actions"][:, :30],
                "training_observation_std": np.asarray(
                    [float(np.std(dataset["observations"], ddof=0))],
                    dtype=np.float64,
                ),
                "test_seed": np.asarray([test_seed], dtype=np.uint32),
                "model_noise_key": np.asarray(
                    jax.random.key_data(model_noise_key), dtype=np.uint32
                ),
            }
            if diagnostic_name == diagnostics.DETERMINISTIC_NAME:
                target = evaluation["target_futures"][:, :30]
                raw = {
                    **common,
                    "target_futures": target,
                    "model_futures": target + np.float32(0.01),
                    "ratio_epsilon": np.asarray(
                        [diagnostics.RATIO_EPSILON], dtype=np.float64
                    ),
                }
            else:
                oracle = evaluation["oracle_reference_futures"][:, :, :30]
                raw = {
                    **common,
                    "score_horizons": np.asarray((1, 8, 30), dtype=np.int32),
                    "rollout_horizons": np.asarray(
                        (1, 2, 4, 8, 15, 30), dtype=np.int32
                    ),
                    "oracle_reference_futures": oracle,
                    "evaluation_futures": evaluation["evaluation_futures"][:, :, :30],
                    "evaluation_branches": evaluation["evaluation_branches"][:, :, :30],
                    "model_futures": np.roll(oracle, shift=1, axis=1),
                    "ece_bin_edges": np.linspace(
                        0.0, 1.0, 11, dtype=np.float64
                    ),
                }
            for arm in ARM_ORDER:
                arm_directory = directory / arm
                raw_path = diagnostics._write_npz_once(
                    arm_directory / "evaluation_raw.npz", raw
                )
                checkpoint_path = arm_directory / "checkpoint.pkl"
                checkpoint_path.write_bytes(
                    f"authenticated-test-checkpoint:{diagnostic_name}:{arm}".encode()
                )
                candidate = diagnostics._selected_candidate(self.matrix, arm)
                config = diagnostics.make_config(
                    self.protocol,
                    "confirmatory",
                    arm,
                    (1,),
                    1,
                    candidate=candidate,
                )
                config_payload = asdict(config)
                identity = {
                    "diagnostic_name": diagnostic_name,
                    "arm": arm,
                    "protocol_sha256": self.matrix["protocol_sha256"],
                    "source_sha256": self.matrix["source_sha256"],
                    "matrix_sha256": self.matrix["matrix_sha256"],
                    "selection_sha256": self.matrix["selection_sha256"],
                    "smoke_nonclaim": False,
                    "effective_updates": updates,
                    "effective_config_sha256": object_sha256(config_payload),
                    "dataset_file_sha256": diagnostics.file_sha256(dataset_path),
                    "schedule_file_sha256": diagnostics.file_sha256(schedule_path),
                }
                expected_updates = list(
                    range(plan.checkpoint_every, updates + 1, plan.checkpoint_every)
                )
                if not expected_updates or expected_updates[-1] != updates:
                    expected_updates.append(updates)
                manifest = {
                    "schema_version": diagnostics.TRAINING_MANIFEST_SCHEMA,
                    "status": "complete",
                    **identity,
                    "identity_sha256": object_sha256(identity),
                    "selected_candidate": candidate,
                    "effective_config": config_payload,
                    "checkpoint_trace": [
                        self.checkpoint_trace_row(update)
                        for update in expected_updates
                    ],
                    "checkpoint_sha256": diagnostics.file_sha256(checkpoint_path),
                    "wall_seconds": 1.0,
                    "runtime": copy.deepcopy(self.runtime),
                }
                manifest["training_manifest_sha256"] = object_sha256(manifest)
                diagnostics._write_json_once(
                    arm_directory / "training_manifest.json", manifest
                )
                result = diagnostics._arm_result_manifest(
                    root,
                    diagnostic_name,
                    arm,
                    raw_path,
                    manifest,
                    self.protocol,
                    plan,
                    self.runtime,
                )
                diagnostics._write_json_once(
                    arm_directory / "result_manifest.json", result
                )
                raw_paths[diagnostic_name][arm] = raw_path
        return raw_paths

    def test_summary_matches_current_schema_and_recomputes_from_raw(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._write_registered_evidence(root)
            summary = diagnostics.build_diagnostic_summary(
                root, self.matrix, self.protocol, paths, smoke_nonclaim=False
            )
            with mock.patch.object(
                diagnostics, "_validate_model_futures_from_checkpoint"
            ) as checkpoint_replay:
                diagnostics.validate_diagnostic_summary_from_raw(
                    summary,
                    root,
                    self.matrix,
                    self.protocol,
                    smoke_nonclaim=False,
                )
                self.assertEqual(checkpoint_replay.call_count, 4)
            self.assertEqual(
                set(summary),
                {
                    "schema_version",
                    "status",
                    "protocol_sha256",
                    "source_sha256",
                    "matrix_sha256",
                    "selection_sha256",
                    "diagnostics",
                    "diagnostic_summary_sha256",
                },
            )
            for row in summary["diagnostics"].values():
                for arm_row in row["arms"].values():
                    self.assertEqual(len(arm_row["raw_files"]), 1)
            diagnostics._write_json_once(
                root / "diagnostics" / "diagnostic_summary.json", summary
            )
            with mock.patch.object(
                diagnostics, "_validate_model_futures_from_checkpoint"
            ) as checkpoint_replay:
                current_validator = validate_confirmatory_auxiliary_evidence(
                    root, self.matrix, self.protocol
                )
                self.assertEqual(checkpoint_replay.call_count, 4)
            self.assertTrue(current_validator["diagnostics"])

    def test_checkpoint_replay_rejects_model_future_tampering(self) -> None:
        plan = diagnostics.resolve_diagnostic_plan(
            self.protocol,
            smoke=True,
            smoke_updates=1,
            smoke_training_episodes=4,
            smoke_test_conditions=4,
            smoke_repeated_futures=4,
            smoke_batch_size=2,
        )
        diagnostic_name = diagnostics.DETERMINISTIC_NAME
        arm = ARM_ORDER[0]
        candidate = diagnostics._selected_candidate(self.matrix, arm)
        config = diagnostics.make_config(
            self.protocol,
            "confirmatory",
            arm,
            (1,),
            1,
            candidate=candidate,
        )
        raw = self.deterministic_raw()
        regenerated = {
            "evaluation": object(),
            "training_observation_std": 1.0,
            "model_noise_key": object(),
            "test_seed": 101,
        }
        trace = [self.checkpoint_trace_row(1)]
        manifest = {
            "identity_sha256": "a" * 64,
            "effective_updates": 1,
            "checkpoint_trace": trace,
            "wall_seconds": 0.25,
        }
        metadata = {
            "stage": diagnostics.CHECKPOINT_STAGE,
            "identity_sha256": "a" * 64,
            "completed_updates": 1,
            "wall_seconds": 0.25,
            "checkpoint_trace": trace,
        }
        state = self.checkpoint_state(1)
        replayed = {name: np.asarray(value).copy() for name, value in raw.items()}
        with (
            mock.patch(
                "imf_dreamer_jax.load_checkpoint",
                return_value=(state, config, metadata),
            ),
            mock.patch.object(diagnostics, "_evaluate_model", return_value=replayed),
        ):
            diagnostics._validate_model_futures_from_checkpoint(
                Path("/unused"),
                diagnostic_name,
                arm,
                raw,
                manifest,
                self.protocol,
                self.matrix,
                plan,
                regenerated,
            )
        replayed["model_futures"][0, 0] += np.float32(0.25)
        with (
            mock.patch(
                "imf_dreamer_jax.load_checkpoint",
                return_value=(state, config, metadata),
            ),
            mock.patch.object(diagnostics, "_evaluate_model", return_value=replayed),
            self.assertRaisesRegex(ValueError, "checkpoint_replay.model_futures"),
        ):
            diagnostics._validate_model_futures_from_checkpoint(
                Path("/unused"),
                diagnostic_name,
                arm,
                raw,
                manifest,
                self.protocol,
                self.matrix,
                plan,
                regenerated,
            )

    def test_real_checkpoint_round_trip_replays_actual_evaluator(self) -> None:
        import jax
        import jax.numpy as jnp
        from imf_dreamer_jax import DreamerConfig, create_agent, save_checkpoint

        plan = diagnostics.resolve_diagnostic_plan(
            self.protocol,
            smoke=True,
            smoke_updates=1,
            smoke_training_episodes=4,
            smoke_test_conditions=4,
            smoke_repeated_futures=4,
            smoke_batch_size=2,
        )
        diagnostic_name = diagnostics.DETERMINISTIC_NAME
        arm = ARM_ORDER[0]
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
        state = create_agent(config, jax.random.key(501))
        state = state._replace(
            model_optimizer=state.model_optimizer._replace(
                step=jnp.asarray(1, dtype=state.model_optimizer.step.dtype)
            )
        )
        evaluation = diagnostics.generate_scalar_evaluation_data(
            diagnostic_name,
            conditions=2,
            episode_length=64,
            repeated_futures=1,
            seed=502,
        )
        model_noise_key = jax.random.key(503)
        raw = diagnostics._evaluate_model(
            state,
            config,
            diagnostic_name,
            evaluation,
            draws=1,
            training_observation_std=1.25,
            model_noise_key=model_noise_key,
            test_seed=502,
        )
        identity_sha256 = "a" * 64
        trace = [self.checkpoint_trace_row(1, 0.125)]
        metadata = {
            "stage": diagnostics.CHECKPOINT_STAGE,
            "identity_sha256": identity_sha256,
            "completed_updates": 1,
            "wall_seconds": 0.5,
            "checkpoint_trace": trace,
        }
        manifest = {
            "identity_sha256": identity_sha256,
            "effective_updates": 1,
            "checkpoint_trace": trace,
            "wall_seconds": 0.5,
        }
        regenerated = {
            "evaluation": evaluation,
            "training_observation_std": 1.25,
            "model_noise_key": model_noise_key,
            "test_seed": 502,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = (
                root
                / "diagnostics"
                / "smoke"
                / diagnostic_name
                / arm
                / "checkpoint.pkl"
            )
            save_checkpoint(checkpoint, state, config, metadata=metadata)
            with mock.patch.object(diagnostics, "make_config", return_value=config):
                diagnostics._validate_model_futures_from_checkpoint(
                    root,
                    diagnostic_name,
                    arm,
                    raw,
                    manifest,
                    self.protocol,
                    self.matrix,
                    plan,
                    regenerated,
                )

    def test_stale_zero_step_checkpoint_with_full_metadata_is_rejected(self) -> None:
        import jax
        from imf_dreamer_jax import DreamerConfig, create_agent, save_checkpoint

        plan = diagnostics.resolve_diagnostic_plan(
            self.protocol,
            smoke=True,
            smoke_updates=1,
            smoke_training_episodes=4,
            smoke_test_conditions=4,
            smoke_repeated_futures=4,
            smoke_batch_size=2,
        )
        diagnostic_name = diagnostics.DETERMINISTIC_NAME
        arm = ARM_ORDER[0]
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
        stale_state = create_agent(config, jax.random.key(511))
        trace = [self.checkpoint_trace_row(1)]
        metadata = {
            "stage": diagnostics.CHECKPOINT_STAGE,
            "identity_sha256": "b" * 64,
            "completed_updates": 1,
            "wall_seconds": 0.5,
            "checkpoint_trace": trace,
        }
        manifest = {
            "identity_sha256": "b" * 64,
            "effective_updates": 1,
            "checkpoint_trace": trace,
            "wall_seconds": 0.5,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = (
                root
                / "diagnostics"
                / "smoke"
                / diagnostic_name
                / arm
                / "checkpoint.pkl"
            )
            save_checkpoint(checkpoint, stale_state, config, metadata=metadata)
            with (
                mock.patch.object(diagnostics, "make_config", return_value=config),
                self.assertRaisesRegex(ValueError, "model optimizer step"),
            ):
                diagnostics._validate_model_futures_from_checkpoint(
                    root,
                    diagnostic_name,
                    arm,
                    {},
                    manifest,
                    self.protocol,
                    self.matrix,
                    plan,
                    {
                        "evaluation": {},
                        "training_observation_std": 1.0,
                        "model_noise_key": jax.random.key(512),
                        "test_seed": 513,
                    },
                )

    def test_checkpoint_metadata_state_trace_and_wall_are_fail_closed(self) -> None:
        plan = diagnostics.resolve_diagnostic_plan(
            self.protocol,
            smoke=True,
            smoke_updates=1,
            smoke_training_episodes=4,
            smoke_test_conditions=4,
            smoke_repeated_futures=4,
            smoke_batch_size=2,
        )
        trace = [self.checkpoint_trace_row(1)]
        metadata = {
            "stage": diagnostics.CHECKPOINT_STAGE,
            "identity_sha256": "c" * 64,
            "completed_updates": 1,
            "wall_seconds": 0.75,
            "checkpoint_trace": trace,
        }
        diagnostics._validate_loaded_diagnostic_checkpoint(
            self.checkpoint_state(1),
            metadata,
            identity_sha256="c" * 64,
            expected_completed_updates=1,
            total_updates=1,
            plan=plan,
            expected_trace=trace,
            expected_wall_seconds=0.75,
        )
        cases = (
            (
                "stage mismatch",
                {**metadata, "stage": "world_model"},
                self.checkpoint_state(1),
                trace,
                0.75,
            ),
            (
                "completed_updates must be an integer",
                {**metadata, "completed_updates": True},
                self.checkpoint_state(1),
                trace,
                0.75,
            ),
            (
                "actor optimizer step",
                metadata,
                self.checkpoint_state(1, actor_step=1),
                trace,
                0.75,
            ),
            (
                "critic optimizer step",
                metadata,
                self.checkpoint_state(1, critic_step=1),
                trace,
                0.75,
            ),
            (
                "canonical update prefix",
                {**metadata, "checkpoint_trace": [self.checkpoint_trace_row(1), self.checkpoint_trace_row(1)]},
                self.checkpoint_state(1),
                trace,
                0.75,
            ),
            (
                "wall seconds must be finite",
                {**metadata, "wall_seconds": float("nan")},
                self.checkpoint_state(1),
                trace,
                0.75,
            ),
            (
                "trace differs from the training manifest",
                metadata,
                self.checkpoint_state(1),
                [self.checkpoint_trace_row(1, 1.0)],
                0.75,
            ),
            (
                "wall time differs from the training manifest",
                metadata,
                self.checkpoint_state(1),
                trace,
                1.0,
            ),
        )
        for message, observed, state, expected_trace, expected_wall in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                diagnostics._validate_loaded_diagnostic_checkpoint(
                    state,
                    observed,
                    identity_sha256="c" * 64,
                    expected_completed_updates=1,
                    total_updates=1,
                    plan=plan,
                    expected_trace=expected_trace,
                    expected_wall_seconds=expected_wall,
                )

    def test_summary_tampering_and_path_traversal_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = diagnostics.build_diagnostic_summary(
                root,
                self.matrix,
                self.protocol,
                self._write_registered_evidence(root),
                smoke_nonclaim=False,
            )
            tampered = copy.deepcopy(summary)
            tampered["diagnostics"][diagnostics.DETERMINISTIC_NAME]["arms"][
                ARM_ORDER[0]
            ]["estimands"]["one_step_normalized_RMSE"] += 1.0
            tampered["diagnostic_summary_sha256"] = object_sha256(
                {
                    key: value
                    for key, value in tampered.items()
                    if key != "diagnostic_summary_sha256"
                }
            )
            with self.assertRaisesRegex(ValueError, "do not recompute"):
                diagnostics.validate_diagnostic_summary_from_raw(
                    tampered,
                    root,
                    self.matrix,
                    self.protocol,
                    smoke_nonclaim=False,
                )
            traversal = copy.deepcopy(summary)
            traversal["diagnostics"][diagnostics.DETERMINISTIC_NAME]["arms"][
                ARM_ORDER[0]
            ]["raw_files"][0]["path"] = "../outside.npz"
            traversal["diagnostic_summary_sha256"] = object_sha256(
                {
                    key: value
                    for key, value in traversal.items()
                    if key != "diagnostic_summary_sha256"
                }
            )
            with self.assertRaisesRegex(ValueError, "traversal"):
                diagnostics.validate_diagnostic_summary_from_raw(
                    traversal,
                    root,
                    self.matrix,
                    self.protocol,
                    smoke_nonclaim=False,
                )

    def test_main_confirmatory_gate_recomputes_estimands_from_raw(self) -> None:
        """A newly digested, favorable hand-authored summary remains ineligible."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = diagnostics.build_diagnostic_summary(
                root,
                self.matrix,
                self.protocol,
                self._write_registered_evidence(root),
                smoke_nonclaim=False,
            )
            arm_row = summary["diagnostics"][diagnostics.DETERMINISTIC_NAME]["arms"][
                ARM_ORDER[0]
            ]
            arm_row["estimands"]["one_step_normalized_RMSE"] = 0.0
            arm_row["passed"] = True
            summary["diagnostic_summary_sha256"] = object_sha256(
                {
                    key: value
                    for key, value in summary.items()
                    if key != "diagnostic_summary_sha256"
                }
            )
            diagnostics._write_json_once(
                root / "diagnostics" / "diagnostic_summary.json", summary
            )

            with self.assertRaisesRegex(ValueError, "estimands do not recompute"):
                validate_confirmatory_auxiliary_evidence(
                    root, self.matrix, self.protocol
                )

    def test_diagnostic_runtime_must_match_frozen_compute_identity(self) -> None:
        ordinal_only_change = copy.deepcopy(self.runtime)
        ordinal_only_change["devices"] = ["scheduler-assigned-device-7"]
        ordinal_only_change["cuda_visible_devices"] = "7"
        diagnostics._validate_runtime_against_frozen(
            ordinal_only_change,
            self.runtime_identity,
            "ordinal-only test runtime",
        )
        self.assertEqual(ordinal_only_change["cuda_visible_devices"], "7")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = diagnostics.build_diagnostic_summary(
                root,
                self.matrix,
                self.protocol,
                self._write_registered_evidence(root),
                smoke_nonclaim=False,
            )
            mismatched = copy.deepcopy(self.runtime_identity)
            mismatched["backend"] = "gpu"
            with (
                mock.patch.object(
                    diagnostics,
                    "_frozen_compute_runtime_identity",
                    return_value=mismatched,
                ),
                self.assertRaisesRegex(
                    ValueError, "runtime differs from the frozen compute-plan runtime"
                ),
            ):
                diagnostics.validate_diagnostic_summary_from_raw(
                    summary,
                    root,
                    self.matrix,
                    self.protocol,
                    smoke_nonclaim=False,
                )

    def test_smoke_summary_is_deliberately_rejected_by_confirmatory_validator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = diagnostics.build_diagnostic_summary(
                root,
                self.matrix,
                self.protocol,
                self._write_all_raw(root),
                smoke_nonclaim=True,
            )
            path = root / "diagnostics" / "diagnostic_summary.json"
            diagnostics._write_json_once(path, summary)
            with self.assertRaisesRegex(ValueError, "identity or digest"):
                validate_confirmatory_auxiliary_evidence(
                    root, self.matrix, self.protocol
                )

    def test_smoke_sized_artifacts_cannot_be_relabelled_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_paths = self._write_all_raw(root)
            diagnostic_name = diagnostics.DETERMINISTIC_NAME
            train_seed = diagnostics.derive_seed(
                "matched-objective-diagnostic-train",
                self.matrix["matrix_sha256"],
                diagnostic_name,
            )
            schedule_seed = diagnostics.derive_seed(
                "matched-objective-diagnostic-minibatches",
                self.matrix["matrix_sha256"],
                diagnostic_name,
            )
            diagnostic_root = root / "diagnostics" / diagnostic_name
            diagnostics._write_npz_once(
                diagnostic_root / "training_dataset.npz",
                diagnostics.generate_scalar_training_data(
                    diagnostic_name,
                    episodes=4,
                    episode_length=64,
                    seed=train_seed,
                ),
            )
            diagnostics._write_npz_once(
                diagnostic_root / "minibatch_schedule.npz",
                diagnostics._batch_schedule(schedule_seed, 1, 2, 4),
            )
            summary = diagnostics.build_diagnostic_summary(
                root,
                self.matrix,
                self.protocol,
                raw_paths,
                smoke_nonclaim=False,
            )
            diagnostics._write_json_once(
                root / "diagnostics" / "diagnostic_summary.json", summary
            )
            with self.assertRaisesRegex(ValueError, "training dataset"):
                validate_confirmatory_auxiliary_evidence(
                    root, self.matrix, self.protocol
                )

    def test_immutable_writers_and_lock_do_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "value.npz"
            diagnostics._write_npz_once(path, {"value": np.asarray([1.0])})
            diagnostics._write_npz_once(path, {"value": np.asarray([1.0])})
            with self.assertRaises(FileExistsError):
                diagnostics._write_npz_once(path, {"value": np.asarray([2.0])})
            lock = root / ".lock"
            lock.write_text("another runner\n")
            with self.assertRaises(RuntimeError):
                with diagnostics._exclusive_runner_lock(lock):
                    pass
            self.assertTrue(lock.is_file(), "a failed acquisition must not delete another lock")

    def test_preempted_raw_without_result_is_regenerated_and_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = (
                root
                / "diagnostics"
                / "smoke"
                / diagnostics.DETERMINISTIC_NAME
                / ARM_ORDER[0]
                / "evaluation_raw.npz"
            )
            deterministic_raw = self.deterministic_raw()
            diagnostics._write_npz_once(raw_path, deterministic_raw)
            original_digest = diagnostics.file_sha256(raw_path)

            def fake_evaluation(
                state,
                config,
                diagnostic_name,
                evaluation,
                **kwargs,
            ):
                del state, config, evaluation, kwargs
                if diagnostic_name == diagnostics.DETERMINISTIC_NAME:
                    return self.deterministic_raw()
                return self.stochastic_raw()

            def fake_result(
                output_root,
                diagnostic_name,
                arm,
                observed_raw_path,
                training_manifest,
                protocol,
                plan,
                runtime,
            ):
                del output_root, training_manifest, protocol, plan
                return {
                    "diagnostic_name": diagnostic_name,
                    "arm": arm,
                    "raw_sha256": diagnostics.file_sha256(observed_raw_path),
                    "runtime": dict(runtime),
                }

            with (
                mock.patch.object(
                    diagnostics,
                    "_validate_frozen_root",
                    return_value=(self.protocol, {}, self.matrix),
                ),
                mock.patch.object(
                    diagnostics,
                    "_train_world_model_arm",
                    return_value=(
                        object(),
                        object(),
                        {"training_manifest_sha256": "a" * 64},
                    ),
                ),
                mock.patch.object(
                    diagnostics, "_evaluate_model", side_effect=fake_evaluation
                ) as evaluate,
                mock.patch.object(
                    diagnostics,
                    "_arm_result_manifest",
                    side_effect=fake_result,
                ),
                mock.patch.object(
                    diagnostics,
                    "build_diagnostic_summary",
                    return_value={
                        "status": "smoke_nonclaim",
                        "diagnostic_summary_sha256": "b" * 64,
                        "diagnostics": {},
                    },
                ),
                mock.patch.object(
                    diagnostics, "validate_diagnostic_summary_from_raw"
                ),
            ):
                summary = diagnostics.run_registered_diagnostics(
                    root,
                    smoke=True,
                    smoke_updates=1,
                    smoke_training_episodes=4,
                    smoke_test_conditions=4,
                    smoke_repeated_futures=4,
                    smoke_batch_size=2,
                )

            self.assertEqual(summary["status"], "smoke_nonclaim")
            self.assertEqual(evaluate.call_count, 4)
            self.assertEqual(diagnostics.file_sha256(raw_path), original_digest)
            self.assertTrue(
                raw_path.with_name("result_manifest.json").is_file(),
                "resume must complete the missing result manifest",
            )

    def test_shared_jax_world_model_sampler_emits_canonical_raw_for_both_arms(self) -> None:
        import jax
        from imf_dreamer_jax import DreamerConfig, create_agent

        evaluation = diagnostics.generate_scalar_evaluation_data(
            diagnostics.STOCHASTIC_NAME,
            conditions=2,
            episode_length=64,
            repeated_futures=4,
            oracle_reference_futures=4,
            evaluation_futures=4,
            seed=401,
        )
        configurations = {
            "shortcut_forcing": DreamerConfig(
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
            ),
            "trajectory_imf": DreamerConfig(
                observation_shape=(1,),
                action_dim=1,
                deterministic_dim=8,
                stochastic_dim=4,
                embedding_dim=8,
                hidden_dim=8,
                prior="imf",
                imf_trajectory_enabled=True,
                imf_boundary_velocity_supervision=True,
                burn_in=1,
                overshooting_scale=0.0,
            ),
        }
        for index, arm in enumerate(ARM_ORDER):
            config = configurations[arm]
            state = create_agent(config, jax.random.key(410 + index))
            raw = diagnostics._evaluate_model(
                state,
                config,
                diagnostics.STOCHASTIC_NAME,
                evaluation,
                draws=4,
                training_observation_std=1.0,
                model_noise_key=jax.random.key(420),
                test_seed=401,
            )
            self.assertEqual(set(raw), set(diagnostics.STOCHASTIC_RAW_FIELDS))
            estimands, _, _ = diagnostics.stochastic_estimands_from_raw(raw)
            self.assertTrue(all(np.isfinite(value) for value in estimands.values()))


if __name__ == "__main__":
    unittest.main()
