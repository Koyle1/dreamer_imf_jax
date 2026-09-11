from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from dreamer_imf_comparison.scripts import verify_neurips_report as report


class NeurIPSReportTests(unittest.TestCase):
    """Adversarial tests for the final report's trust boundary."""

    @classmethod
    def tearDownClass(cls) -> None:
        print("NEURIPS_REPORT_TESTS_OK")

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="neurips-report-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, *, positive: bool) -> tuple[Path, Path, dict]:
        bundle_path, report_path = report.create_synthetic_bundle(
            self.root / ("positive" if positive else "negative"),
            positive=positive,
        )
        return bundle_path, report_path, report.read_json(bundle_path)

    def rewrite_bundle(self, path: Path, bundle: dict) -> None:
        bundle["bundle_sha256"] = report.object_sha256(
            report._without_digest(bundle, "bundle_sha256")
        )
        report.write_json_atomic(path, bundle)

    def refresh_binding(self, bundle: dict, name: str) -> None:
        bundle["bindings"][name]["sha256"] = report.file_sha256(
            Path(bundle["bindings"][name]["path"])
        )

    def test_positive_and_valid_negative_rederive_without_relabeling(self) -> None:
        positive_path, positive_report, _ = self.fixture(positive=True)
        negative_path, negative_report, _ = self.fixture(positive=False)
        positive = report.verify_bundle(positive_path, positive_report)
        negative = report.verify_bundle(negative_path, negative_report)

        self.assertEqual(
            positive["decision"]["claim_state"], "synthetic_positive_control"
        )
        self.assertEqual(
            negative["decision"]["claim_state"],
            "synthetic_valid_negative_control",
        )
        self.assertTrue(positive["decision"]["gate_math_passed"])
        self.assertFalse(positive["decision"]["registered_profiles_complete"])
        self.assertFalse(positive["decision"]["registered_gate_passed"])
        self.assertFalse(positive["decision"]["empirical_claim_allowed"])
        self.assertFalse(negative["decision"]["registered_gate_passed"])
        self.assertTrue(all(row["passed"] for row in positive["statistical_intervals"]))
        self.assertTrue(all(not row["passed"] for row in negative["statistical_intervals"]))
        self.assertIn(
            "SYNTHETIC VALID-NEGATIVE CONTROL",
            negative_report.read_text(encoding="utf-8"),
        )

    def test_synthetic_positive_control_never_becomes_empirical_claim(self) -> None:
        bundle_path, report_path, _ = self.fixture(positive=True)
        evidence = report.verify_bundle(bundle_path, report_path)
        self.assertNotEqual(
            evidence["decision"]["claim_state"], "registered_claim_supported"
        )
        self.assertTrue(evidence["decision"]["gate_math_passed"])
        self.assertFalse(evidence["decision"]["registered_profiles_complete"])
        self.assertFalse(evidence["decision"]["registered_gate_passed"])
        self.assertFalse(evidence["decision"]["empirical_claim_allowed"])
        self.assertIn("support no empirical comparison", report_path.read_text())

    def test_report_is_deterministic_under_mapping_order_changes(self) -> None:
        _, _, bundle = self.fixture(positive=True)

        def reverse(value):
            if isinstance(value, dict):
                return {key: reverse(value[key]) for key in reversed(list(value))}
            if isinstance(value, list):
                return [reverse(item) for item in value]
            return value

        shuffled = reverse(bundle)
        self.assertEqual(report.render_report(bundle), report.render_report(shuffled))

    def test_report_byte_tampering_fails_closed(self) -> None:
        bundle_path, report_path, _ = self.fixture(positive=False)
        report_path.write_text(report_path.read_text() + "\nbenign-looking edit\n")
        with self.assertRaisesRegex(report.VerificationError, "exactly rederive"):
            report.verify_bundle(bundle_path, report_path)

    def test_every_required_file_digest_detects_drift(self) -> None:
        for name in report.BINDING_ORDER:
            with self.subTest(binding=name):
                case = self.root / name
                bundle_path, report_path = report.create_synthetic_bundle(
                    case, positive=False
                )
                bundle = report.read_json(bundle_path)
                bound = Path(bundle["bindings"][name]["path"])
                bound.write_bytes(bound.read_bytes() + b" ")
                with self.assertRaisesRegex(report.VerificationError, f"{name} digest drift"):
                    report.verify_bundle(bundle_path, report_path)

    def test_rehashed_bundle_evidence_cannot_promote_negative_result(self) -> None:
        bundle_path, report_path, bundle = self.fixture(positive=False)
        bundle["evidence"]["decision"]["claim_state"] = "registered_claim_supported"
        bundle["evidence"]["decision"]["registered_gate_passed"] = True
        self.rewrite_bundle(bundle_path, bundle)
        with self.assertRaisesRegex(report.VerificationError, "does not rederive"):
            report.verify_bundle(bundle_path, report_path)

    def test_rehashed_negative_analysis_cannot_flip_stored_gate(self) -> None:
        bundle_path, report_path, bundle = self.fixture(positive=False)
        analysis_path = Path(bundle["bindings"]["analysis"]["path"])
        analysis = report.read_json(analysis_path)
        analysis["superiority"]["passed"] = True
        analysis["superiority"]["satisfied_interval_count"] = 4
        analysis["analysis_sha256"] = report.object_sha256(
            report._without_digest(analysis, "analysis_sha256")
        )
        report.write_json_atomic(analysis_path, analysis)
        self.refresh_binding(bundle, "analysis")
        self.rewrite_bundle(bundle_path, bundle)
        with self.assertRaisesRegex(report.VerificationError, "interval counts"):
            report.verify_bundle(bundle_path, report_path)

    def test_missing_primary_evidence_sections_fail_closed(self) -> None:
        cases = tuple((key, "required evidence field") for key in (
            "tracks", "practical_significance", "secondary_rollout", "compute_resources"
        ))
        for index, (key, expected) in enumerate(cases):
            with self.subTest(section=key):
                bundle_path, report_path = report.create_synthetic_bundle(
                    self.root / f"missing-{index}", positive=True
                )
                bundle = report.read_json(bundle_path)
                analysis_path = Path(bundle["bindings"]["analysis"]["path"])
                analysis = report.read_json(analysis_path)
                del analysis[key]
                analysis["analysis_sha256"] = report.object_sha256(
                    report._without_digest(analysis, "analysis_sha256")
                )
                report.write_json_atomic(analysis_path, analysis)
                self.refresh_binding(bundle, "analysis")
                self.rewrite_bundle(bundle_path, bundle)
                with self.assertRaisesRegex(report.VerificationError, expected):
                    report.verify_bundle(bundle_path, report_path)

    def test_missing_actor_metric_fails_closed(self) -> None:
        bundle_path, report_path, bundle = self.fixture(positive=True)
        analysis_path = Path(bundle["bindings"]["analysis"]["path"])
        analysis = report.read_json(analysis_path)
        del analysis["tracks"]["equal_updates"]["real_environment_normalized_return_iqm"]
        analysis["analysis_sha256"] = report.object_sha256(
            report._without_digest(analysis, "analysis_sha256")
        )
        report.write_json_atomic(analysis_path, analysis)
        self.refresh_binding(bundle, "analysis")
        self.rewrite_bundle(bundle_path, bundle)
        with self.assertRaisesRegex(report.VerificationError, "metric family drifted"):
            report.verify_bundle(bundle_path, report_path)

    def test_missing_per_task_heterogeneity_fails_closed(self) -> None:
        bundle_path, report_path, bundle = self.fixture(positive=True)
        analysis_path = Path(bundle["bindings"]["analysis"]["path"])
        analysis = report.read_json(analysis_path)
        analysis["practical_significance"]["per_task"] = {}
        analysis["analysis_sha256"] = report.object_sha256(
            report._without_digest(analysis, "analysis_sha256")
        )
        report.write_json_atomic(analysis_path, analysis)
        self.refresh_binding(bundle, "analysis")
        self.rewrite_bundle(bundle_path, bundle)
        with self.assertRaisesRegex(report.VerificationError, "required evidence field"):
            report.verify_bundle(bundle_path, report_path)

    def test_missing_calibration_field_fails_closed(self) -> None:
        bundle_path, report_path, bundle = self.fixture(positive=True)
        analysis_path = Path(bundle["bindings"]["analysis"]["path"])
        analysis = report.read_json(analysis_path)
        del analysis["secondary_rollout"]["tracks"]["equal_updates"]["trajectory_imf"]["1"][
            "per_horizon"
        ]["30"]["central_90_percent_interval_coverage_iqm"]
        analysis["analysis_sha256"] = report.object_sha256(
            report._without_digest(analysis, "analysis_sha256")
        )
        report.write_json_atomic(analysis_path, analysis)
        self.refresh_binding(bundle, "analysis")
        self.rewrite_bundle(bundle_path, bundle)
        with self.assertRaisesRegex(report.VerificationError, "required evidence field"):
            report.verify_bundle(bundle_path, report_path)

    def test_controls_cannot_access_or_decide_primary_gate(self) -> None:
        for key, value, expected in (
            ("claim_eligible", True, "accessed or decided"),
            ("primary_gate_accessed", True, "accessed or decided"),
            ("superiority_decision", "pass", "accessed or decided"),
        ):
            with self.subTest(key=key):
                bundle_path, report_path = report.create_synthetic_bundle(
                    self.root / f"controls-{key}", positive=True
                )
                bundle = report.read_json(bundle_path)
                path = Path(bundle["bindings"]["control"]["path"])
                control = report.read_json(path)
                control[key] = value
                control["analysis_sha256"] = report.object_sha256(
                    report._without_digest(control, "analysis_sha256")
                )
                report.write_json_atomic(path, control)
                self.refresh_binding(bundle, "control")
                self.rewrite_bundle(bundle_path, bundle)
                with self.assertRaisesRegex(report.VerificationError, expected):
                    report.verify_bundle(bundle_path, report_path)

    def test_one_track_missing_strong_baseline_family_fails_closed(self) -> None:
        bundle_path, report_path, bundle = self.fixture(positive=True)
        path = Path(bundle["bindings"]["control"]["path"])
        control = report.read_json(path)
        del control["summaries"]["equal_updates"]["gaussian_rssm@fixed"]
        control["analysis_sha256"] = report.object_sha256(
            report._without_digest(control, "analysis_sha256")
        )
        report.write_json_atomic(path, control)
        self.refresh_binding(bundle, "control")
        self.rewrite_bundle(bundle_path, bundle)
        with self.assertRaisesRegex(report.VerificationError, "required control families"):
            report.verify_bundle(bundle_path, report_path)

    def test_synthetic_pixel_matrix_is_exactly_three_tasks_by_five_seeds(self) -> None:
        bundle_path, report_path, bundle = self.fixture(positive=True)
        evidence = report.verify_bundle(bundle_path, report_path)
        self.assertEqual(
            {(row["task_count"], row["seed_count"])
             for row in evidence["pixel"]["primary_contrasts"]},
            {(3, 5)},
        )
        path = Path(bundle["bindings"]["pixel"]["path"])
        pixel = report.read_json(path)
        pixel["tracks"]["equal_updates"]["task_order"] = list(
            report.SYNTHETIC_PIXEL_TASKS[:-1]
        )
        pixel["tracks"]["equal_updates"]["task_by_seed_primary_delta"] = pixel[
            "tracks"
        ]["equal_updates"]["task_by_seed_primary_delta"][:-1]
        pixel["analysis_sha256"] = report.object_sha256(
            report._without_digest(pixel, "analysis_sha256")
        )
        report.write_json_atomic(path, pixel)
        self.refresh_binding(bundle, "pixel")
        self.rewrite_bundle(bundle_path, bundle)
        with self.assertRaisesRegex(report.VerificationError, "registered matrix"):
            report.verify_bundle(bundle_path, report_path)

    def test_authenticated_pixel_identity_is_the_exact_registered_matrix(self) -> None:
        pixel = report._synthetic_documents(True)["pixel"]
        pixel["schema_version"] = "trajectory-imf-pixel-aggregate-v1"
        pixel["analysis_sha256"] = report.object_sha256(
            report._without_digest(pixel, "analysis_sha256")
        )
        evidence = report._pixel_evidence(pixel, "authenticated_runs")
        self.assertEqual(
            {(row["task_count"], row["seed_count"])
             for row in evidence["primary_contrasts"]},
            {(3, 5)},
        )
        pixel["tracks"]["equal_compiler_flops"]["seed_order"] = [19, 29, 39]
        pixel["tracks"]["equal_compiler_flops"]["task_by_seed_primary_delta"] = [
            row[:3] for row in pixel["tracks"]["equal_compiler_flops"][
                "task_by_seed_primary_delta"
            ]
        ]
        pixel["analysis_sha256"] = report.object_sha256(
            report._without_digest(pixel, "analysis_sha256")
        )
        with self.assertRaisesRegex(report.VerificationError, "registered matrix"):
            report._pixel_evidence(pixel, "authenticated_runs")

    def test_authenticated_path_uses_independent_pixel_verifier_exactly(self) -> None:
        roots = {name: self.root / name for name in ("core", "controls")}
        for path in roots.values():
            path.mkdir()
        pixel_path = self.root / "pixel.json"
        source_sha, protocol_sha = "1" * 64, "2" * 64
        report.write_json_atomic(
            roots["controls"] / "matrix.json",
            {"parent_source_sha256": source_sha, "parent_protocol_sha256": protocol_sha},
        )
        documents = {
            "source": {"source_sha256": source_sha},
            "matrix": {"protocol_sha256": protocol_sha},
            "analysis": {"analysis_sha256": "3" * 64},
            "control": {"analysis_sha256": "4" * 64},
            "pixel": {"inputs": [{"path": str(self.root / "cell")}], "sentinel": "bound"},
        }
        bundle = {
            "evidence_mode": "authenticated_runs",
            "roots": {"core": str(roots["core"]), "controls": str(roots["controls"]),
                      "pixel": str(pixel_path)},
        }
        with mock.patch(
            "dreamer_imf_compare.matched_objective_benchmark.verify_output_root",
            return_value={"analysis_sha256": "3" * 64},
        ), mock.patch(
            "dreamer_imf_compare.neurips_controls.verify_controls_output",
            return_value={"analysis_sha256": "4" * 64},
        ), mock.patch(
            "dreamer_imf_compare.pixel_benchmark.verify_aggregate",
            return_value=documents["pixel"],
        ) as verifier:
            report._authenticate(bundle, documents)
            verifier.return_value = {
                "inputs": documents["pixel"]["inputs"], "sentinel": "changed"
            }
            with self.assertRaisesRegex(report.VerificationError, "verifier differs"):
                report._authenticate(bundle, documents)
        self.assertEqual(verifier.call_count, 2)
        verifier.assert_called_with(pixel_path, [str(self.root / "cell")])

    def test_pixel_result_cannot_substitute_for_primary_gate(self) -> None:
        bundle_path, report_path, bundle = self.fixture(positive=True)
        path = Path(bundle["bindings"]["pixel"]["path"])
        pixel = report.read_json(path)
        pixel["primary_gate_substitution"] = True
        report.write_json_atomic(path, pixel)
        self.refresh_binding(bundle, "pixel")
        self.rewrite_bundle(bundle_path, bundle)
        with self.assertRaisesRegex(report.VerificationError, "claim boundary"):
            report.verify_bundle(bundle_path, report_path)

    def test_pixel_nfe_frontier_and_long_horizon_are_required(self) -> None:
        for mutation, expected in (("nfe", "frontier drifted"), ("horizon", "long horizon")):
            with self.subTest(mutation=mutation):
                bundle_path, report_path = report.create_synthetic_bundle(
                    self.root / f"pixel-{mutation}", positive=True
                )
                bundle = report.read_json(bundle_path)
                path = Path(bundle["bindings"]["pixel"]["path"])
                pixel = report.read_json(path)
                if mutation == "nfe":
                    del pixel["tracks"]["equal_updates"]["frontier_iqm"]["trajectory_imf"]["2"]
                else:
                    pixel["horizons"] = [1, 2, 4]
                pixel["analysis_sha256"] = report.object_sha256(
                    report._without_digest(pixel, "analysis_sha256")
                )
                report.write_json_atomic(path, pixel)
                self.refresh_binding(bundle, "pixel")
                self.rewrite_bundle(bundle_path, bundle)
                with self.assertRaisesRegex(report.VerificationError, expected):
                    report.verify_bundle(bundle_path, report_path)

    def test_theory_cannot_claim_trained_assumptions(self) -> None:
        bundle_path, report_path, bundle = self.fixture(positive=True)
        path = Path(bundle["bindings"]["theory"]["path"])
        theory = report.read_json(path)
        theory["trained_model_assumptions_established"] = True
        report.write_json_atomic(path, theory)
        self.refresh_binding(bundle, "theory")
        self.rewrite_bundle(bundle_path, bundle)
        with self.assertRaisesRegex(report.VerificationError, "conditional claim boundary"):
            report.verify_bundle(bundle_path, report_path)

    def test_unsupported_dreamer4_reproduction_claim_is_rejected(self) -> None:
        with self.assertRaisesRegex(report.VerificationError, "Dreamer 4"):
            report.validate_report_language(
                "We reproduce Dreamer 4 and compare against it.",
                "registered_claim_supported",
            )

    def test_broad_video_generation_claim_is_rejected(self) -> None:
        with self.assertRaisesRegex(report.VerificationError, "video-generation"):
            report.validate_report_language(
                "This is a stable video generation method.",
                "registered_claim_supported",
            )

    def test_positive_superiority_wording_is_rejected_for_negative_result(self) -> None:
        for phrase in report.FORBIDDEN_POSITIVE:
            with self.subTest(phrase=phrase):
                with self.assertRaisesRegex(report.VerificationError, "positive comparison"):
                    report.validate_report_language(
                        f"Trajectory iMF {phrase}.",
                        "synthetic_valid_negative_control",
                    )

    def test_binding_path_escape_and_symlink_are_rejected(self) -> None:
        bundle_path, report_path, bundle = self.fixture(positive=True)
        bundle["bindings"]["source"]["path"] = str(self.root.resolve() / "outside.json")
        self.rewrite_bundle(bundle_path, bundle)
        with self.assertRaisesRegex(report.VerificationError, "escaped"):
            report.verify_bundle(bundle_path, report_path)

        bundle_path, report_path = report.create_synthetic_bundle(
            self.root / "symlink-case", positive=True
        )
        bundle = report.read_json(bundle_path)
        source = Path(bundle["bindings"]["source"]["path"])
        linked = source.parent / "source-link.json"
        linked.symlink_to(source)
        bundle["bindings"]["source"]["path"] = str(linked)
        self.rewrite_bundle(bundle_path, bundle)
        with self.assertRaisesRegex(report.VerificationError, "not canonical"):
            report.verify_bundle(bundle_path, report_path)

    def test_cli_rejects_symlinked_verify_and_build_outputs_before_resolution(self) -> None:
        bundle_path, report_path, _ = self.fixture(positive=True)
        bundle_link = self.root / "bundle-link.json"
        report_link = self.root / "report-link.md"
        bundle_link.symlink_to(bundle_path)
        report_link.symlink_to(report_path)
        for bundle, rendered in ((bundle_link, report_path), (bundle_path, report_link)):
            with self.subTest(verify_link=bundle if bundle.is_symlink() else rendered):
                self.assertEqual(
                    report.main(["--verify", "--bundle", str(bundle), "--report", str(rendered)]),
                    1,
                )

        for name in ("bundle", "report"):
            target = self.root / f"build-{name}-target"
            target.write_text("unchanged", encoding="utf-8")
            linked = self.root / f"build-{name}-link"
            linked.symlink_to(target)
            outputs = {"bundle": self.root / "new-bundle.json",
                       "report": self.root / "new-report.md"}
            outputs[name] = linked
            with self.subTest(build_link=name):
                self.assertEqual(
                    report.main([
                        "--build", "--core-root", str(self.root / "absent-core"),
                        "--controls-root", str(self.root / "absent-controls"),
                        "--pixel-aggregate", str(self.root / "absent-pixel.json"),
                        "--bundle", str(outputs["bundle"]), "--report", str(outputs["report"]),
                    ]),
                    1,
                )
            self.assertEqual(target.read_text(encoding="utf-8"), "unchanged")

    def test_scaffold_is_exactly_nonfinal(self) -> None:
        self.assertEqual(
            report.DEFAULT_SCAFFOLD.read_text(encoding="utf-8"),
            report.render_scaffold(),
        )
        report.verify_scaffold()
        lower = report.render_scaffold().lower()
        self.assertIn("nonfinal", lower)
        self.assertNotIn("outperforms shortcut forcing", lower)

    def test_all_required_sections_and_digest_bindings_are_retained(self) -> None:
        bundle_path, report_path, _ = self.fixture(positive=True)
        evidence = report.verify_bundle(bundle_path, report_path)
        for key in (
            "statistical_intervals",
            "practical_effects",
            "per_task_heterogeneity",
            "calibration",
            "actor_returns",
            "compute_nfe",
            "controls",
            "pixel",
            "theory",
            "limitations",
        ):
            self.assertIn(key, evidence)
            self.assertTrue(evidence[key])
        self.assertEqual(set(evidence["digests"]), set(report.BINDING_ORDER))
        self.assertEqual(evidence["limitations"], list(report.LIMITATIONS))


if __name__ == "__main__":
    unittest.main()
