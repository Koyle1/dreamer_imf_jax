from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from dreamer_imf_compare.artifacts import read_json
import dreamer_imf_compare.neurips_controls as controls


PROJECT = Path(__file__).resolve().parents[1]


class NeurIPSControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = read_json(PROJECT / "neurips_controls_protocol.json")
        cls.parent_protocol = read_json(PROJECT / "matched_objective_protocol.json")

    @classmethod
    def tearDownClass(cls) -> None:
        print("NEURIPS_CONTROLS_TESTS_OK")

    def test_contract_expands_exact_full_factorial_and_is_claim_isolated(self) -> None:
        controls.validate_controls_protocol(self.protocol, self.parent_protocol)
        arms = controls.expected_arm_specs(self.protocol)
        self.assertEqual(len(arms), 33)
        self.assertEqual(len({arm["arm_id"] for arm in arms}), 33)
        factorial = [arm for arm in arms if arm["family"] == "trajectory_imf"]
        self.assertEqual(len(factorial), 28)
        self.assertEqual(len(controls._profile_arm_specs(self.protocol, "smoke")), 33)
        self.assertEqual(len(controls._profile_arm_specs(self.protocol, "development")), 33)
        self.assertEqual(
            [
                arm["arm_id"]
                for arm in controls._profile_arm_specs(self.protocol, "confirmatory")
            ],
            self.protocol["profile_arm_policy"]["confirmatory_arm_ids"],
        )
        combinations = {
            (
                arm["query_history_noise_coupling"],
                arm["query_history_time_relation"],
                arm["context_mask"],
            )
            for arm in factorial
        }
        self.assertEqual(
            combinations,
            {
                (noise, relation, mask)
                for noise in ("shared", "independent")
                for relation in ("separate", "tied_on_corrupted_positions")
                for mask in range(1, 8)
            },
        )
        proposed = [arm for arm in arms if arm["arm_id"] == "trajectory_imf"]
        self.assertEqual(len(proposed), 1)
        self.assertEqual(proposed[0]["context_mask"], 7)
        self.assertEqual(proposed[0]["context_probabilities"], [1 / 3, 1 / 3, 1 / 3])
        certificate = next(
            arm for arm in arms if arm["arm_id"] == "trajectory_endpoint_certificate"
        )
        self.assertTrue(certificate["endpoint_certificate_enabled"])
        self.assertEqual(certificate["endpoint_certificate_mass"], 0.25)
        self.assertFalse(self.protocol["claim_boundary"]["may_support_primary_superiority_claim"])
        self.assertFalse(self.protocol["statistics"]["superiority_decision_field_permitted"])

    def test_contract_rejects_primary_gate_contamination_and_task_drift(self) -> None:
        changed = copy.deepcopy(self.protocol)
        changed["statistics"]["superiority_decision_field_permitted"] = True
        with self.assertRaisesRegex(ValueError, "contaminate"):
            controls.validate_controls_protocol(changed, self.parent_protocol)
        changed = copy.deepcopy(self.protocol)
        changed["profiles"]["confirmatory"]["tasks"] = changed["profiles"]["confirmatory"]["tasks"][:-1]
        with self.assertRaisesRegex(ValueError, "differs from parent"):
            controls.validate_controls_protocol(changed, self.parent_protocol)

    def test_dynamic_schedule_exactly_reduces_to_library_schedule(self) -> None:
        import jax
        import jax.numpy as jnp
        from imf_dreamer_jax import sample_trajectory_schedule

        uniforms = jax.random.uniform(jax.random.key(3), (3, 9, 6))
        mask = jnp.asarray(
            [[0, 0, 1, 1, 1, 1, 1, 1, 1]] * 3, dtype=jnp.float32
        )
        expected = sample_trajectory_schedule(
            jax.random.key(4),
            3,
            9,
            token_mask=mask,
            boundary_fraction=0.5,
            time_mean=-0.4,
            time_std=1.0,
            history_noise_max=1.0,
            pattern_probabilities=(1 / 3, 1 / 3, 1 / 3),
            canonical_uniforms=uniforms,
        )
        actual = controls.sample_control_schedule(
            uniforms,
            mask,
            jnp.asarray([1 / 3, 1 / 3, 1 / 3]),
            boundary_fraction=0.5,
            time_mean=-0.4,
            time_std=1.0,
            history_noise_max=1.0,
        )
        for observed, target in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(np.asarray(observed), np.asarray(target))

    def test_proposed_control_loss_and_gradients_reduce_to_canonical_trajectory(self) -> None:
        import jax
        import jax.numpy as jnp
        from imf_dreamer_jax import DreamerConfig, create_agent, world_model_loss

        config = DreamerConfig(
            observation_shape=(5,),
            action_dim=2,
            deterministic_dim=8,
            stochastic_dim=4,
            embedding_dim=8,
            hidden_dim=16,
            prior="imf",
            burn_in=1,
            overshooting_scale=0.0,
            imf_trajectory_enabled=True,
            imf_boundary_velocity_supervision=True,
        )
        state = create_agent(config, jax.random.key(11))
        key = jax.random.key(17)
        batch = {
            "observations": jax.random.normal(jax.random.key(1), (2, 5, 5)),
            "actions": jax.random.uniform(jax.random.key(2), (2, 5, 2), minval=-1, maxval=1),
            "rewards": jax.random.normal(jax.random.key(3), (2, 5)),
            "continuations": jnp.ones((2, 5), jnp.float32),
            "is_first": jnp.asarray([[True, False, False, False, False]] * 2),
            "loss_mask": jnp.asarray([[0, 1, 1, 1, 1]] * 2, jnp.float32),
        }
        vector = jnp.asarray(
            [1.0, 0.0, 1 / 3, 1 / 3, 1 / 3, 0.0, 0.0], jnp.float32
        )
        expected = world_model_loss(state.params.world_model, batch, key, config)
        actual = controls.trajectory_control_loss(
            state.params.world_model, batch, key, config, vector
        )
        for name in ("total", "reconstruction", "reward", "continuation", "prior", "representation", "imf_loss_u", "imf_loss_v"):
            np.testing.assert_allclose(
                np.asarray(getattr(actual, name)),
                np.asarray(getattr(expected, name)),
                rtol=2e-6,
                atol=2e-6,
            )
        expected_grad = jax.grad(
            lambda params: world_model_loss(params, batch, key, config).total
        )(state.params.world_model)
        actual_grad = jax.grad(
            lambda params: controls.trajectory_control_loss(
                params, batch, key, config, vector
            ).total
        )(state.params.world_model)
        expected_leaves = jax.tree_util.tree_leaves(expected_grad)
        actual_leaves = jax.tree_util.tree_leaves(actual_grad)
        for observed, target in zip(actual_leaves, expected_leaves, strict=True):
            np.testing.assert_allclose(np.asarray(observed), np.asarray(target), rtol=2e-5, atol=2e-5)

        certificate_vector = vector.at[5].set(1.0).at[6].set(0.25)
        certificate = controls.trajectory_control_loss(
            state.params.world_model,
            batch,
            key,
            config,
            certificate_vector,
            endpoint_certificate_enabled=True,
        )
        self.assertGreater(float(certificate.endpoint_certificate), 0.0)
        np.testing.assert_allclose(
            np.asarray(certificate.prior - actual.prior),
            np.asarray(0.25 * certificate.endpoint_certificate),
            rtol=2e-6,
            atol=2e-6,
        )

    def test_factor_vectors_change_only_registered_schedule_inputs(self) -> None:
        arms = controls.expected_arm_specs(self.protocol)
        proposed = next(arm for arm in arms if arm["arm_id"] == "trajectory_imf")
        run = {
            "family": "trajectory_imf",
            "arm_spec": proposed,
        }
        vector = controls.trajectory_control_vector(run)
        np.testing.assert_allclose(vector, [1, 0, 1 / 3, 1 / 3, 1 / 3, 0, 0])
        for arm in arms:
            if arm["family"] != "trajectory_imf":
                continue
            probabilities = np.asarray(arm["context_probabilities"])
            self.assertAlmostEqual(float(probabilities.sum()), 1.0)
            self.assertEqual(int(np.count_nonzero(probabilities)), int(arm["context_mask"]).bit_count())

    def test_native_baseline_configs_are_complete_and_have_registered_priors(self) -> None:
        # Matrix-free config behavior is covered through a minimal frozen-shape
        # shell because make_control_config only consumes the declared fields.
        matrix = {
            "profile": "smoke",
            "parent_candidates": controls._neutral_parent_candidates(self.parent_protocol),
        }
        runs = controls._arm_runs(self.protocol, "smoke", None)
        expected = {
            "shortcut_forcing": ("shortcut", False),
            "ordinary_imf": ("imf", False),
            "gaussian_rssm": ("gaussian", False),
            "temporal_increment_imf": ("imf", False),
            "trajectory_endpoint_certificate": ("imf", True),
            "trajectory_imf": ("imf", True),
        }
        for arm_id, pair in expected.items():
            run = next(value for value in runs if value["arm_id"] == arm_id)
            config = controls.make_control_config(
                self.parent_protocol, matrix, run, (7,), 2
            )
            self.assertEqual((config.prior, config.imf_trajectory_enabled), pair)
            self.assertEqual(set(asdict(config)), {field.name for field in __import__("dataclasses").fields(config)})

    def test_equal_flop_allocator_uses_nearest_integer_and_lower_tie(self) -> None:
        target, allocation = controls._general_equal_flop_allocation(
            {"shortcut_forcing": 10.0, "other": 4.0},
            reference_run="shortcut_forcing",
            reference_updates=1,
            tolerance=1.0,
            enforce=False,
        )
        self.assertEqual(target, 10.0)
        self.assertEqual(allocation["other"]["updates"], 2)
        self.assertEqual(allocation["other"]["cumulative_flops"], 8.0)

    def test_controls_selection_is_source_bound_and_grid_closed(self) -> None:
        selected = {
            family: copy.deepcopy(
                self.protocol["development_selection"]["candidate_grid"][family][0]
            )
            for family in self.protocol["development_selection"]["selected_families"]
        }
        selection = {
            "schema_version": controls.SELECTION_SCHEMA,
            "status": "complete",
            "controls_protocol_sha256": controls.controls_protocol_digest(self.protocol),
            "development_analysis_sha256": "1" * 64,
            "development_matrix_sha256": "2" * 64,
            "development_source_sha256": "3" * 64,
            "selection_profile": "development",
            "selection_budget_track": "equal_updates",
            "confirmatory_outcomes_accessed": False,
            "selected": selected,
            "candidate_rank_evidence": {
                family: [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "selection_score": float(index + 1),
                        "task_scores": [float(index + 1), float(index + 1)],
                    }
                    for index, candidate in enumerate(
                        self.protocol["development_selection"]["candidate_grid"][family]
                    )
                ]
                for family in self.protocol["development_selection"]["selected_families"]
            },
        }
        selection["selection_sha256"] = controls.object_sha256(selection)
        self.assertEqual(
            controls._validate_controls_selection(selection, self.protocol, "3" * 64),
            selection,
        )
        with self.assertRaisesRegex(ValueError, "development-only boundary"):
            controls._validate_controls_selection(selection, self.protocol, "4" * 64)
        changed = copy.deepcopy(selection)
        changed["selected"]["ordinary_imf"]["prior_scale"] = 99.0
        changed["selection_sha256"] = controls.object_sha256(
            controls._without_digest(changed, "selection_sha256")
        )
        with self.assertRaisesRegex(ValueError, "frozen grid"):
            controls._validate_controls_selection(changed, self.protocol, "3" * 64)

    def test_artifact_manifest_rejects_added_tampered_and_traversal_files(self) -> None:
        matrix = {
            "profile": "smoke",
            "controls_protocol_sha256": "1" * 64,
            "source_sha256": "2" * 64,
            "matrix_sha256": "3" * 64,
        }
        analysis = {"analysis_sha256": "4" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "value.bin").write_bytes(b"authentic")
            manifest = controls.build_controls_artifact_manifest(root, matrix, analysis)
            controls.validate_controls_artifact_manifest(manifest, root, matrix, analysis)
            (root / "extra.bin").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "exact retained file set"):
                controls.validate_controls_artifact_manifest(manifest, root, matrix, analysis)
            (root / "extra.bin").unlink()
            (root / "value.bin").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "digest/size"):
                controls.validate_controls_artifact_manifest(manifest, root, matrix, analysis)
            changed = copy.deepcopy(manifest)
            changed["files"][0]["path"] = "../escape"
            changed["artifact_manifest_sha256"] = controls.object_sha256(
                controls._without_digest(changed, "artifact_manifest_sha256")
            )
            with self.assertRaisesRegex(ValueError, "canonical and relative"):
                controls.validate_controls_artifact_manifest(changed, root, matrix, analysis)

    def test_report_is_independent_of_mapping_insertion_order(self) -> None:
        row_a = {
            "family": "shortcut_forcing",
            "rollout_auc_iqm": 2.0,
            "actor_return_iqm": 3.0,
        }
        row_b = {
            "family": "gaussian_rssm",
            "rollout_auc_iqm": 1.0,
            "actor_return_iqm": 4.0,
        }
        forward = {
            "profile": "smoke",
            "evidence_class": "engineering_smoke",
            "summaries": {
                "equal_updates": {"z_run": row_a, "a_run": row_b},
                "equal_compiler_flops": {"z_run": row_a, "a_run": row_b},
            },
        }
        reversed_order = {
            "profile": "smoke",
            "evidence_class": "engineering_smoke",
            "summaries": {
                "equal_compiler_flops": {"a_run": row_b, "z_run": row_a},
                "equal_updates": {"a_run": row_b, "z_run": row_a},
            },
        }
        report = controls.render_controls_report(forward)
        self.assertEqual(report, controls.render_controls_report(reversed_order))
        self.assertLess(report.index("## equal_updates"), report.index("## equal_compiler_flops"))
        self.assertLess(report.index("`a_run`"), report.index("`z_run`"))
        smoke_subset = copy.deepcopy(forward)
        smoke_subset["summaries"].pop("equal_compiler_flops")
        subset_report = controls.render_controls_report(smoke_subset)
        self.assertIn("## equal_updates", subset_report)
        self.assertNotIn("## equal_compiler_flops", subset_report)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analysis.json"
            for index, analysis in enumerate(
                (forward, reversed_order, smoke_subset)
            ):
                path = path.with_name(f"analysis-{index}.json")
                controls.write_json_atomic(path, analysis)
                self.assertEqual(
                    controls.render_controls_report(analysis),
                    controls.render_controls_report(read_json(path)),
                )

    def test_cell_directory_rejects_traversal_and_noncanonical_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for cell_id in ("../escape", "world-not-hex", "dataset-" + "a" * 24):
                with self.subTest(cell_id=cell_id), self.assertRaisesRegex(
                    ValueError, "path-safe"
                ):
                    controls._cell_directory(root, {"cell_id": cell_id})
            valid = "world-" + "a" * 24
            self.assertEqual(
                controls._cell_directory(root, {"cell_id": valid}),
                root / "cells" / valid,
            )

    def test_confirmatory_selection_files_must_equal_matrix_copies(self) -> None:
        parent_selection = {"selection_sha256": "1" * 64, "selected": {}}
        controls_selection = {"selection_sha256": "2" * 64, "selected": {}}
        matrix = {
            "profile": "confirmatory",
            "parent_selection": parent_selection,
            "controls_selection": controls_selection,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controls.write_json_atomic(root / "parent_selection.json", parent_selection)
            controls.write_json_atomic(root / "controls_selection.json", controls_selection)
            controls._validate_frozen_selection_files(root, matrix)
            changed = copy.deepcopy(controls_selection)
            changed["selection_sha256"] = "3" * 64
            controls.write_json_atomic(root / "controls_selection.json", changed)
            with self.assertRaisesRegex(ValueError, "embedded in the matrix"):
                controls._validate_frozen_selection_files(root, matrix)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controls.write_json_atomic(root / "parent_selection.json", parent_selection)
            with self.assertRaisesRegex(ValueError, "unexpected parent selection"):
                controls._validate_frozen_selection_files(
                    root,
                    {
                        "profile": "development",
                        "parent_selection": None,
                        "controls_selection": None,
                    },
                )

    def test_finalizer_validates_payload_before_publishing_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = {"profile": "smoke", "matrix_sha256": "1" * 64}
            analysis = {"analysis_sha256": "2" * 64}
            manifest = {"artifact_manifest_sha256": "3" * 64}
            with (
                mock.patch.object(
                    controls,
                    "load_frozen_controls",
                    return_value=({}, {}, {}, matrix),
                ),
                mock.patch.object(controls, "validate_controls_protocol"),
                mock.patch.object(controls, "validate_controls_source_manifest"),
                mock.patch.object(controls, "validate_controls_matrix"),
                mock.patch.object(
                    controls,
                    "validate_all_control_cells",
                    return_value={"compute": 1, "world": 1, "rollout": 1, "actor": 1},
                ),
                mock.patch.object(
                    controls, "build_controls_analysis", return_value=analysis
                ),
                mock.patch.object(controls, "render_controls_report", return_value="report\n"),
                mock.patch.object(
                    controls, "build_controls_artifact_manifest", return_value=manifest
                ),
                mock.patch.object(
                    controls,
                    "verify_controls_output",
                    side_effect=ValueError("pre-final payload rejection"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "pre-final payload rejection"):
                    controls.finalize_controls_run(root)
            self.assertFalse((root / "FINALIZED.json").exists())

    def test_run_all_validates_frozen_matrix_before_any_execution(self) -> None:
        protocol = {"protocol": "controls"}
        parent_protocol = {"protocol": "parent"}
        source = {"source": "manifest"}
        matrix = {"profile": "smoke"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "matrix.json").write_text("{}\n", encoding="utf-8")
            calls: list[str] = []
            with (
                mock.patch.object(
                    controls,
                    "load_frozen_controls",
                    return_value=(protocol, parent_protocol, source, matrix),
                ),
                mock.patch.object(
                    controls,
                    "validate_controls_source_manifest",
                    side_effect=lambda *_args, **_kwargs: calls.append("source"),
                ),
                mock.patch.object(
                    controls,
                    "validate_controls_matrix",
                    side_effect=lambda *_args, **_kwargs: calls.append("matrix"),
                ),
                mock.patch.object(
                    controls,
                    "run_canonical_datasets",
                    side_effect=lambda *_args, **_kwargs: calls.append("datasets"),
                ),
                mock.patch.object(
                    controls,
                    "run_stage_cells",
                    side_effect=lambda *_args, **_kwargs: calls.append("stages"),
                ),
                mock.patch.object(
                    controls,
                    "finalize_controls_run",
                    side_effect=lambda *_args, **_kwargs: calls.append("finalize") or {},
                ),
            ):
                controls.run_controls_all(
                    protocol, parent_protocol, "smoke", root
                )
            self.assertEqual(
                calls, ["source", "matrix", "datasets", "stages", "finalize"]
            )


if __name__ == "__main__":
    unittest.main()
