from __future__ import annotations

import copy
import json
import os
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

import dreamer_imf_compare.pixel_benchmark as pixel_module  # noqa: E402
from dreamer_imf_compare.pixel_benchmark import (  # noqa: E402
    ARM_ORDER,
    RESULT_KEYS,
    SEALED_ARTIFACT_FILES,
    PixelDMCAdapter,
    _artifact_root,
    _expected_hard_identities,
    _independent_auc_by_cell,
    _independent_iqm,
    _json_load,
    _requested_tracks,
    _validate_aggregate_schema,
    _validate_exact_artifact_entries,
    _validate_checkpoint_state,
    _validate_result_schema,
    _validate_runtime_fingerprint,
    aggregate_artifacts,
    area_downsample_uint8,
    batch_from_starts,
    collect_pixel_dataset,
    eligible_window_starts,
    global_ssim,
    interquartile_mean,
    load_protocol,
    make_batch_schedule,
    object_sha256,
    prediction_metrics,
    preprocess_pixels_jax,
    replay_pixel_dataset,
    resolved_config,
    run_benchmark,
    select_windows,
    validate_dataset,
    validate_exploration_actions,
    validate_protocol,
    verify_aggregate,
    verify_artifact,
)


class _Spec:
    shape = (1,)
    minimum = np.asarray([-2.0], np.float32)
    maximum = np.asarray([2.0], np.float32)


class _TimeStep:
    def __init__(self, reward: float, discount: float, last: bool) -> None:
        self.reward = reward
        self.discount = discount
        self._last = last

    def last(self) -> bool:
        return self._last


class _Physics:
    def __init__(self, environment: "_Environment") -> None:
        self.environment = environment
        self.calls: list[dict[str, object]] = []

    def render(self, **kwargs):
        self.calls.append(dict(kwargs))
        height = int(kwargs["height"])
        width = int(kwargs["width"])
        rows = np.arange(height, dtype=np.uint16)[:, None]
        columns = np.arange(width, dtype=np.uint16)[None, :]
        value = (rows + columns + self.environment.native_step * 7) % 256
        return np.stack(
            (value, (value + 11) % 256, (value + 23) % 256), axis=-1
        ).astype(np.uint8)


class _Environment:
    def __init__(self, terminate_every: int = 3) -> None:
        self.native_step = 0
        self.episode_step = 0
        self.terminate_every = terminate_every
        self.actions: list[np.ndarray] = []
        self.physics = _Physics(self)
        self.closed = False

    def action_spec(self):
        return _Spec()

    def reset(self):
        self.episode_step = 0
        return _TimeStep(0.0, 1.0, False)

    def step(self, action):
        self.native_step += 1
        self.episode_step += 1
        self.actions.append(np.asarray(action).copy())
        last = self.episode_step >= self.terminate_every
        return _TimeStep(float(self.episode_step), 0.5, last)

    def close(self):
        self.closed = True


def _adapter(environment: _Environment, *, repeat: int = 2) -> PixelDMCAdapter:
    return PixelDMCAdapter(
        "dmc_pendulum_swingup",
        seed=7,
        action_repeat=repeat,
        raw_height=64,
        raw_width=64,
        model_height=32,
        model_width=32,
        frame_stack=3,
        camera_id=0,
        renderer_backend="test",
        environment=environment,
    )


def _hard_runtime_fixture(protocol):
    return {
        **protocol["runtime_contract"]["hard"]["versions"],
        "platform": "Linux-fixture",
        "backend": "gpu",
        "jax_enable_x64": False,
        "device_count": 1,
        "visible_device_count": 1,
        "device_platforms": ["gpu"],
        "device_kinds": ["NVIDIA L40S"],
        "mujoco_gl": "egl",
        "jax_enable_x64_environment": "0",
        "jax_platform_name": "gpu",
        "cuda_visible_devices": "GPU-fixture",
    }


class PixelProtocolTests(unittest.TestCase):
    def test_contract_is_frozen_and_secondary(self):
        protocol = load_protocol()
        validate_protocol(protocol)
        self.assertFalse(protocol["study_role"]["claim_eligible_for_primary_gate"])
        self.assertEqual(protocol["evaluation"]["nfe_frontier"], [1, 2, 4])
        self.assertEqual(protocol["rendering"]["raw_height"], 64)
        self.assertEqual(protocol["observation"]["model_shape"], [32, 32, 9])

    def test_hard_runtime_is_frozen_while_smoke_is_portable(self):
        protocol = load_protocol()
        hard = protocol["runtime_contract"]["hard"]
        self.assertEqual(hard["backend"], "gpu")
        self.assertEqual(hard["visible_device_count"], 1)
        self.assertEqual(hard["device_kind_substring"], "L40S")
        self.assertFalse(protocol["runtime_contract"]["smoke"]["claim_eligible"])
        self.assertEqual(
            protocol["runtime_contract"]["smoke"]["allowed_backends"],
            ["cpu", "gpu"],
        )

    def test_claim_boundary_tamper_is_rejected(self):
        protocol = copy.deepcopy(load_protocol())
        protocol["study_role"]["claim_eligible_for_primary_gate"] = True
        with self.assertRaises(ValueError):
            validate_protocol(protocol)

    def test_both_objective_configs_share_shapes(self):
        protocol = load_protocol()
        shortcut = resolved_config(protocol, "smoke", "shortcut_forcing", action_dim=1)
        trajectory = resolved_config(protocol, "smoke", "trajectory_imf", action_dim=1)
        self.assertEqual(shortcut.observation_shape, trajectory.observation_shape)
        self.assertEqual(shortcut.deterministic_dim, trajectory.deterministic_dim)
        self.assertEqual(shortcut.stochastic_dim, trajectory.stochastic_dim)
        self.assertEqual(shortcut.prior, "shortcut")
        self.assertEqual(trajectory.prior, "imf")
        self.assertFalse(shortcut.imf_trajectory_enabled)
        self.assertTrue(trajectory.imf_trajectory_enabled)

    def test_runner_rejects_cross_profile_task_and_seed_identities(self):
        with tempfile.TemporaryDirectory(prefix="pixel-profile-test-") as temporary:
            parent = Path(temporary).resolve()
            with self.assertRaisesRegex(ValueError, "outside the frozen profile"):
                run_benchmark(
                    parent / "smoke-as-hard-task",
                    profile="smoke",
                    task="dmc_walker_walk",
                    seed=7,
                    track="equal_updates",
                )
            with self.assertRaisesRegex(ValueError, "outside the frozen profile"):
                run_benchmark(
                    parent / "hard-as-smoke-task",
                    profile="hard",
                    task="dmc_pendulum_swingup",
                    seed=19,
                    track="equal_updates",
                )


class PixelEnvironmentTests(unittest.TestCase):
    def test_integer_area_downsampling_rounds_half_up(self):
        frame = np.zeros((4, 4, 3), np.uint8)
        frame[:2, :2, :] = np.asarray(
            [[[0], [0]], [[0], [2]]], np.uint8
        )
        down = area_downsample_uint8(frame, (2, 2))
        self.assertEqual(down.shape, (2, 2, 3))
        self.assertTrue(np.all(down[0, 0] == 1))

    def test_preprocess_is_jax_float32_unit_range(self):
        value = np.zeros((2, 32, 32, 9), np.uint8)
        value[1] = 255
        result = np.asarray(preprocess_pixels_jax(value))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(float(result.min()), 0.0)
        self.assertAlmostEqual(float(result.max()), 1.0, places=6)

    def test_reset_repeat_terminal_and_action_scaling(self):
        environment = _Environment(terminate_every=3)
        adapter = _adapter(environment, repeat=4)
        initial = adapter.reset()
        self.assertEqual(initial.shape, (32, 32, 9))
        self.assertTrue(np.array_equal(initial[..., :3], initial[..., 3:6]))
        self.assertTrue(np.array_equal(initial[..., 3:6], initial[..., 6:9]))
        step = adapter.step(np.asarray([0.5], np.float32))
        self.assertTrue(step.is_last)
        self.assertEqual(step.native_steps, 3)
        self.assertEqual(step.reward, 1.0 + 2.0 + 3.0)
        self.assertAlmostEqual(step.continuation, 0.5**3)
        self.assertEqual(len(environment.actions), 3)
        self.assertTrue(np.allclose(environment.actions[0], [1.0]))
        with self.assertRaises(RuntimeError):
            adapter.step(np.zeros((1,), np.float32))
        adapter.reset()
        self.assertEqual(environment.physics.calls[-1]["camera_id"], 0)
        self.assertFalse(environment.physics.calls[-1]["depth"])
        self.assertFalse(environment.physics.calls[-1]["segmentation"])
        adapter.close()
        self.assertTrue(environment.closed)

    def test_collector_inserts_reset_token_only_after_terminal(self):
        adapter = _adapter(_Environment(terminate_every=3), repeat=2)
        arrays = collect_pixel_dataset(
            adapter,
            9,
            exploration_seed=123,
            ar_coefficient=0.8,
            innovation_scale=0.6,
        )
        adapter.close()
        validate_dataset(arrays, 1, 9, 2)
        reset_indices = np.flatnonzero(arrays["is_first"])
        self.assertGreaterEqual(len(reset_indices), 2)
        for index in reset_indices[1:]:
            self.assertTrue(arrays["is_last"][index - 1])
            self.assertEqual(arrays["episode_step"][index], 0)
        corrupted = {name: value.copy() for name, value in arrays.items()}
        nonreset = int(np.flatnonzero(~corrupted["is_first"])[0])
        corrupted["observations"][nonreset, 0, 0, 0] ^= 1
        with self.assertRaises(ValueError):
            validate_dataset(corrupted, 1, 9, 2)

    def test_exploration_actions_rederive_and_tamper_fails(self):
        adapter = _adapter(_Environment(terminate_every=3), repeat=2)
        arrays = collect_pixel_dataset(
            adapter,
            9,
            exploration_seed=123,
            ar_coefficient=0.8,
            innovation_scale=0.6,
        )
        adapter.close()
        validate_exploration_actions(
            arrays,
            exploration_seed=123,
            ar_coefficient=0.8,
            innovation_scale=0.6,
        )
        corrupted = {name: value.copy() for name, value in arrays.items()}
        index = int(np.flatnonzero(~corrupted["is_first"])[0])
        corrupted["actions"][index, 0] = np.nextafter(
            corrupted["actions"][index, 0], np.float32(1.0)
        )
        with self.assertRaisesRegex(ValueError, "frozen AR\\(1\\) replay"):
            validate_exploration_actions(
                corrupted,
                exploration_seed=123,
                ar_coefficient=0.8,
                innovation_scale=0.6,
            )

    def test_retained_actions_replay_every_render_and_transition_directly(self):
        adapter = _adapter(_Environment(terminate_every=1000), repeat=2)
        arrays = collect_pixel_dataset(
            adapter,
            12,
            exploration_seed=123,
            ar_coefficient=0.8,
            innovation_scale=0.6,
        )
        adapter.close()
        protocol = load_protocol()
        with mock.patch.dict(os.environ, {"MUJOCO_GL": "glfw"}, clear=False), mock.patch(
            "dreamer_imf_compare.pixel_benchmark._make_environment",
            side_effect=lambda _task, _seed: _Environment(terminate_every=1000),
        ):
            replay_pixel_dataset(
                protocol,
                profile="smoke",
                task="dmc_pendulum_swingup",
                environment_seed=17,
                arrays=arrays,
            )
            corrupted = {name: value.copy() for name, value in arrays.items()}
            corrupted["observations"][5, 0, 0, 0] ^= np.uint8(1)
            with self.assertRaisesRegex(ValueError, "rendered pixels differ"):
                replay_pixel_dataset(
                    protocol,
                    profile="smoke",
                    task="dmc_pendulum_swingup",
                    environment_seed=17,
                    arrays=corrupted,
                )


class PixelArtifactContractTests(unittest.TestCase):
    def test_json_loader_rejects_duplicate_keys_and_nonfinite_constants(self):
        with tempfile.TemporaryDirectory(prefix="pixel-json-test-") as temporary:
            path = Path(temporary) / "value.json"
            path.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                _json_load(path)
            path.write_text('{"a": NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
                _json_load(path)

    def test_artifact_root_and_entry_symlinks_or_extras_fail(self):
        with tempfile.TemporaryDirectory(prefix="pixel-entry-test-") as temporary:
            parent = Path(temporary).resolve()
            root = parent / "artifact"
            root.mkdir()
            for name in SEALED_ARTIFACT_FILES:
                (root / name).write_bytes(b"fixture")
            _validate_exact_artifact_entries(root)

            alias = parent / "artifact-link"
            alias.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink component"):
                _artifact_root(alias)

            real_parent = parent / "real-parent"
            real_parent.mkdir()
            nested = real_parent / "nested-artifact"
            nested.mkdir()
            ancestor_alias = parent / "aliased-parent"
            ancestor_alias.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink component"):
                _artifact_root(ancestor_alias / "nested-artifact")

            extra = root / "unexpected"
            extra.mkdir()
            with self.assertRaisesRegex(ValueError, "entry set is not exact"):
                _validate_exact_artifact_entries(root)
            extra.rmdir()

            target = parent / "external-result"
            target.write_bytes(b"fixture")
            (root / "result.json").unlink()
            (root / "result.json").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "regular non-symlink"):
                _validate_exact_artifact_entries(root)

        with tempfile.TemporaryDirectory(prefix="pixel-source-link-test-") as temporary:
            workspace = Path(temporary).resolve()
            project = workspace / "dreamer_imf_comparison"
            protocol_path = project / "pixel_benchmark_protocol.json"
            base_protocol_path = project / "matched_objective_protocol.json"
            paths = [
                protocol_path,
                base_protocol_path,
                project / "dreamer_imf_compare" / "pixel_benchmark.py",
                project / "dreamer_imf_compare" / "dmc.py",
                project / "scripts" / "run_pixel_benchmark.py",
                project / "scripts" / "verify_pixel_benchmark.py",
                project / "tests" / "__init__.py",
                project / "tests" / "test_pixel_benchmark.py",
                project / "PIXEL_BENCHMARK.md",
            ]
            paths.extend(
                workspace / "imf_dreamer_jax" / "src" / "imf_dreamer_jax" / name
                for name in pixel_module.LIBRARY_SOURCE_BASENAMES
            )
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            linked = project / "dreamer_imf_compare" / "pixel_benchmark.py"
            linked.unlink()
            external = workspace / "external.py"
            external.write_text("outside = True\n", encoding="utf-8")
            linked.symlink_to(external)
            with mock.patch.object(
                pixel_module, "WORKSPACE_ROOT", workspace
            ), mock.patch.object(
                pixel_module, "PROJECT_ROOT", project
            ), mock.patch.object(
                pixel_module, "PROTOCOL_PATH", protocol_path
            ), mock.patch.object(
                pixel_module, "BASE_PROTOCOL_PATH", base_protocol_path
            ):
                with self.assertRaisesRegex(ValueError, "symlink component"):
                    pixel_module._source_paths()

    def test_result_top_level_schema_rejects_missing_and_extra_fields(self):
        protocol = load_protocol()
        extra = {name: None for name in RESULT_KEYS}
        extra["unregistered"] = None
        with self.assertRaisesRegex(ValueError, "extra=\\['unregistered'\\]"):
            _validate_result_schema(extra, protocol)
        missing = {name: None for name in RESULT_KEYS - {"runtime"}}
        with self.assertRaisesRegex(ValueError, "missing=\\['runtime'\\]"):
            _validate_result_schema(missing, protocol)

    def test_full_artifact_verifier_and_runner_are_directly_idempotent(self):
        with tempfile.TemporaryDirectory(prefix="pixel-direct-verifier-") as temporary:
            root = Path(temporary).resolve() / "artifact"
            with mock.patch.dict(
                os.environ, {"MUJOCO_GL": "glfw"}, clear=False
            ), mock.patch(
                "dreamer_imf_compare.pixel_benchmark._make_environment",
                side_effect=lambda _task, _seed: _Environment(terminate_every=1000),
            ):
                created = run_benchmark(
                    root,
                    profile="smoke",
                    task="dmc_pendulum_swingup",
                    seed=7,
                    track="equal_updates",
                )
                verified = verify_artifact(
                    root, recompute_predictions=True, rederive_compute=False
                )
                repeated = run_benchmark(
                    root,
                    profile="smoke",
                    task="dmc_pendulum_swingup",
                    seed=7,
                    track="equal_updates",
                )
            self.assertEqual(created, verified)
            self.assertEqual(verified, repeated)

    def test_hard_runtime_rejects_wrong_backend_count_version_and_kind(self):
        protocol = load_protocol()
        runtime = _hard_runtime_fixture(protocol)
        _validate_runtime_fingerprint(runtime, protocol, "hard")
        for key, bad in (
            ("backend", "cpu"),
            ("visible_device_count", 2),
            ("jax", "0.0.0"),
            ("device_kinds", ["NVIDIA A100"]),
            ("cuda_visible_devices", "0,1"),
        ):
            corrupted = copy.deepcopy(runtime)
            corrupted[key] = bad
            if key == "visible_device_count":
                corrupted["device_count"] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                _validate_runtime_fingerprint(corrupted, protocol, "hard")

    def test_checkpoint_step_dtype_shape_and_all_numeric_leaves_are_exact(self):
        import jax
        import jax.numpy as jnp
        from imf_dreamer_jax import create_agent

        protocol = load_protocol()
        config = resolved_config(
            protocol, "smoke", "shortcut_forcing", action_dim=1
        )
        state = create_agent(config, jax.random.PRNGKey(0))
        valid_optimizer = state.model_optimizer._replace(
            step=jnp.asarray(1, jnp.int32)
        )
        valid = state._replace(model_optimizer=valid_optimizer)
        _validate_checkpoint_state(valid, updates=1, arm="shortcut_forcing")

        fractional = valid._replace(
            model_optimizer=valid.model_optimizer._replace(
                step=jnp.asarray(1.9, jnp.float32)
            )
        )
        with self.assertRaisesRegex(ValueError, "exact scalar integer"):
            _validate_checkpoint_state(
                fractional, updates=1, arm="shortcut_forcing"
            )
        vector = valid._replace(
            model_optimizer=valid.model_optimizer._replace(
                step=jnp.asarray([1], jnp.int32)
            )
        )
        with self.assertRaisesRegex(ValueError, "exact scalar integer"):
            _validate_checkpoint_state(vector, updates=1, arm="shortcut_forcing")

        leaves, structure = jax.tree_util.tree_flatten(valid.params.world_model)
        bad_parameters = structure.unflatten(
            [jnp.full_like(leaves[0], jnp.nan), *leaves[1:]]
        )
        nonfinite_model = valid._replace(
            params=valid.params._replace(world_model=bad_parameters)
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _validate_checkpoint_state(
                nonfinite_model, updates=1, arm="shortcut_forcing"
            )

        moment_leaves, moment_structure = jax.tree_util.tree_flatten(
            valid.model_optimizer.first_moment
        )
        bad_first_moment = moment_structure.unflatten(
            [jnp.full_like(moment_leaves[0], jnp.inf), *moment_leaves[1:]]
        )
        nonfinite_optimizer = valid._replace(
            model_optimizer=valid.model_optimizer._replace(
                first_moment=bad_first_moment
            )
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _validate_checkpoint_state(
                nonfinite_optimizer, updates=1, arm="shortcut_forcing"
            )

    def test_requested_track_identity_matrix_is_exact(self):
        protocol = load_protocol()
        self.assertEqual(
            _requested_tracks(protocol, None),
            ["equal_updates", "equal_compiler_flops"],
        )
        one = _requested_tracks(protocol, ["equal_compiler_flops"])
        self.assertEqual(one, ["equal_compiler_flops"])
        self.assertEqual(len(_expected_hard_identities(protocol, one)), 15)
        self.assertEqual(
            len(_expected_hard_identities(protocol, _requested_tracks(protocol, None))),
            30,
        )
        with self.assertRaises(ValueError):
            _requested_tracks(protocol, ["equal_updates", "equal_updates"])

    def test_independent_raw_auc_and_iqm_rederivation(self):
        horizons = np.asarray([1, 2, 4], np.int16)
        targets = np.zeros((1, 3, 2, 2, 3), np.float32)
        predictions = np.empty((2, 3, 1, 2, 3, 2, 2, 3), np.float32)
        metrics = {"frontier": {arm: {} for arm in ARM_ORDER}}
        for arm_index, arm in enumerate(ARM_ORDER):
            for nfe_index, nfe in enumerate((1, 2, 4)):
                value = np.float32(0.1 * (1 + arm_index * 3 + nfe_index))
                predictions[arm_index, nfe_index] = value
                metrics["frontier"][arm][str(nfe)] = {
                    "normalized_visual_mse_auc": float(value) ** 2
                }
        with tempfile.TemporaryDirectory(prefix="pixel-auc-test-") as temporary:
            root = Path(temporary)
            np.savez_compressed(
                root / "predictions.npz",
                arm_names=np.asarray(ARM_ORDER, dtype="<U32"),
                nfe_values=np.asarray([1, 2, 4], np.int16),
                horizons=horizons,
                targets=targets,
                predictions=predictions,
                training_channel_variance=np.ones((3,), np.float64),
            )
            values = _independent_auc_by_cell(root, {"metrics": metrics})
        self.assertAlmostEqual(values[("shortcut_forcing", 1)], 0.01, places=6)
        self.assertAlmostEqual(
            _independent_iqm([0, 1, 2, 3, 4, 100]),
            interquartile_mean([0, 1, 2, 3, 4, 100]),
        )

    def test_aggregate_schema_rejects_an_extra_field(self):
        value = {
            "schema_version": "trajectory-imf-pixel-aggregate-v1",
            "status": "complete",
            "study_role": "descriptive_secondary_hard_visual_evaluation",
            "claim_eligible_for_primary_gate": False,
            "primary_gate_substitution": False,
            "protocol_sha256": "0" * 64,
            "requested_tracks": [],
            "task_order": [],
            "seed_order": [],
            "expected_input_count": 0,
            "runtime_identity": {},
            "tracks": {},
            "inputs": [],
            "analysis_sha256": "0" * 64,
            "extra": True,
        }
        with self.assertRaisesRegex(ValueError, "extra=\\['extra'\\]"):
            _validate_aggregate_schema(value)

    def test_aggregate_requires_exact_two_track_30_cell_matrix_and_rederives(self):
        protocol = load_protocol()
        hard = protocol["profiles"]["hard"]
        with tempfile.TemporaryDirectory(prefix="pixel-aggregate-test-") as temporary:
            parent = Path(temporary).resolve()
            roots = []
            results = {}
            for track_index, track in enumerate(hard["tracks"]):
                for task_index, task in enumerate(hard["tasks"]):
                    for seed_index, seed in enumerate(hard["seeds"]):
                        root = parent / track / task / f"seed_{seed}"
                        root.mkdir(parents=True)
                        predictions = np.empty(
                            (2, 3, 1, 2, 1, 1, 1, 3), np.float32
                        )
                        frontier = {arm: {} for arm in ARM_ORDER}
                        for arm_index, arm in enumerate(ARM_ORDER):
                            for nfe_index, nfe in enumerate((1, 2, 4)):
                                scalar = np.float32(
                                    0.01
                                    * (
                                        1
                                        + track_index
                                        + task_index
                                        + seed_index
                                        + arm_index * 3
                                        + nfe_index
                                    )
                                )
                                predictions[arm_index, nfe_index] = scalar
                                frontier[arm][str(nfe)] = {
                                    "normalized_visual_mse_auc": float(scalar) ** 2
                                }
                        np.savez_compressed(
                            root / "predictions.npz",
                            arm_names=np.asarray(ARM_ORDER, dtype="<U32"),
                            nfe_values=np.asarray([1, 2, 4], np.int16),
                            horizons=np.asarray([1], np.int16),
                            targets=np.zeros((1, 1, 1, 1, 3), np.float32),
                            predictions=predictions,
                            training_channel_variance=np.ones((3,), np.float64),
                        )
                        (root / "manifest.json").write_bytes(b"manifest")
                        (root / "seal.json").write_bytes(b"seal")
                        (root / "result.json").write_text("{}\n", encoding="utf-8")
                        result = {
                            "profile": "hard",
                            "track": track,
                            "task": task,
                            "seed": int(seed),
                            "runtime": _hard_runtime_fixture(protocol),
                            "metrics": {
                                "frontier": frontier,
                                "primary_nfe": {
                                    "shortcut_forcing": frontier["shortcut_forcing"]["4"],
                                    "trajectory_imf": frontier["trajectory_imf"]["1"],
                                },
                            },
                        }
                        roots.append(root)
                        results[root.resolve()] = result

            def verified(root, **_kwargs):
                return results[Path(root).resolve()]

            with mock.patch(
                "dreamer_imf_compare.pixel_benchmark.verify_artifact",
                side_effect=verified,
            ):
                summary = aggregate_artifacts(roots)
                self.assertEqual(summary["expected_input_count"], 30)
                self.assertEqual(summary["requested_tracks"], list(hard["tracks"]))
                self.assertEqual(set(summary["tracks"]), set(hard["tracks"]))
                summary_path = parent / "summary.json"
                summary_path.write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                verified_summary = verify_aggregate(summary_path, roots)
                self.assertEqual(verified_summary, summary)
                with self.assertRaisesRegex(ValueError, "input count"):
                    aggregate_artifacts(roots[:-1])
                with self.assertRaisesRegex(ValueError, "duplicate pixel artifact root"):
                    aggregate_artifacts([*roots[:-1], roots[0]])

                corrupted = copy.deepcopy(summary)
                corrupted["tracks"]["equal_updates"][
                    "primary_visual_delta_shortcut_minus_trajectory"
                ] += 1.0
                body = {
                    key: value
                    for key, value in corrupted.items()
                    if key != "analysis_sha256"
                }
                corrupted["analysis_sha256"] = object_sha256(body)
                summary_path.write_text(
                    json.dumps(corrupted, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "independently rederive"):
                    verify_aggregate(summary_path, roots)


class PixelBatchAndMetricTests(unittest.TestCase):
    def test_fractional_iqm_matches_direct_weighting(self):
        # For six values, 1.5 observations are trimmed from either side.
        self.assertAlmostEqual(
            interquartile_mean([0, 1, 2, 3, 4, 100]),
            (0.5 * 1 + 2 + 3 + 0.5 * 4) / 3.0,
        )

    def test_schedule_and_chunk_boundary_are_deterministic(self):
        first = make_batch_schedule(
            20, updates=3, batch_size=2, sequence_length=5, seed=11
        )
        second = make_batch_schedule(
            20, updates=3, batch_size=2, sequence_length=5, seed=11
        )
        self.assertTrue(np.array_equal(first, second))
        dataset = {
            "observations": np.zeros((20, 32, 32, 9), np.uint8),
            "actions": np.ones((20, 1), np.float32),
            "rewards": np.zeros((20,), np.float32),
            "continuations": np.ones((20,), np.float32),
            "is_first": np.zeros((20,), np.bool_),
        }
        batch = batch_from_starts(dataset, np.asarray([4, 8]), 5, 2)
        self.assertTrue(np.all(np.asarray(batch["is_first"])[:, 0]))
        self.assertTrue(np.all(np.asarray(batch["actions"])[:, 0] == 0.0))
        self.assertTrue(np.all(np.asarray(batch["loss_mask"])[:, :2] == 0.0))

    def test_windows_never_cross_episode_and_selection_is_paired(self):
        heldout = {"episode_id": np.asarray([0] * 8 + [1] * 9, np.int32)}
        eligible = eligible_window_starts(heldout, context_length=2, max_horizon=3)
        for start in eligible:
            self.assertEqual(
                len(set(heldout["episode_id"][start : start + 5].tolist())), 1
            )
        first = select_windows(eligible, 3, seed=8)
        second = select_windows(eligible, 3, seed=8)
        self.assertTrue(np.array_equal(first, second))

    def test_perfect_visual_predictions_have_zero_error(self):
        targets = np.linspace(0, 1, 2 * 3 * 4 * 4 * 3, dtype=np.float32).reshape(
            2, 3, 4, 4, 3
        )
        predictions = np.repeat(targets[:, None], 2, axis=1)
        summary, raw = prediction_metrics(
            predictions,
            targets,
            np.asarray([0.1, 0.2, 0.3]),
            [1, 2, 4],
        )
        self.assertAlmostEqual(summary["normalized_visual_mse_auc"], 0.0)
        self.assertTrue(np.allclose(raw["energy_score"], 0.0))
        self.assertTrue(np.allclose(raw["predictive_mean_global_ssim"], 1.0))
        self.assertTrue(np.allclose(summary["predictive_mean_psnr_db"], 120.0))

    def test_global_ssim_rejects_wrong_shapes(self):
        image = np.zeros((4, 4, 3), np.float32)
        self.assertAlmostEqual(global_ssim(image, image), 1.0)
        with self.assertRaises(ValueError):
            global_ssim(image[..., 0], image[..., 0])
        with self.assertRaises(ValueError):
            prediction_metrics(
                np.zeros((1, 2, 3, 4, 4), np.float32),
                np.zeros((1, 3, 4, 4, 3), np.float32),
                np.ones((3,), np.float64),
                [1, 2, 4],
            )

    def test_each_real_objective_executes_a_finite_update(self):
        import jax
        from imf_dreamer_jax import create_agent, train_world_model

        protocol = load_protocol()
        rng = np.random.default_rng(5)
        dataset = {
            "observations": rng.integers(0, 256, (8, 32, 32, 9), dtype=np.uint8),
            "actions": rng.uniform(-1, 1, (8, 1)).astype(np.float32),
            "rewards": np.zeros((8,), np.float32),
            "continuations": np.ones((8,), np.float32),
            "is_first": np.asarray([True] + [False] * 7, np.bool_),
        }
        batch = batch_from_starts(dataset, np.asarray([0]), 8, 2)
        losses = {}
        for index, arm in enumerate(ARM_ORDER):
            config = resolved_config(protocol, "smoke", arm, action_dim=1)
            state = create_agent(config, jax.random.PRNGKey(0))
            state, loss = train_world_model(
                state, batch, jax.random.PRNGKey(10 + index), config
            )
            del state
            losses[arm] = float(np.asarray(loss.total))
            self.assertTrue(np.isfinite(losses[arm]))
        self.assertEqual(set(losses), set(ARM_ORDER))


if __name__ == "__main__":
    unittest.main()
