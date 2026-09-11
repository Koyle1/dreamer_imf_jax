from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import unittest

from dreamer_imf_compare.artifacts import read_json, write_json_atomic
import dreamer_imf_compare.matched_objective_benchmark as benchmark


class MatchedObjectiveArtifactManifestTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        print("MATCHED_OBJECTIVE_ARTIFACT_MANIFEST_OK")

    @staticmethod
    def _matrix(profile: str = "smoke") -> dict:
        return {
            "profile": profile,
            "evidence_class": (
                "claim_eligible_confirmatory"
                if profile == "confirmatory"
                else "engineering_smoke"
            ),
            "claim_eligible": profile == "confirmatory",
            "protocol_sha256": "1" * 64,
            "source_sha256": "2" * 64,
            "matrix_sha256": "3" * 64,
            "selection_manifest": (
                {"selection_sha256": "4" * 64, "selected": {}}
                if profile == "confirmatory"
                else None
            ),
        }

    @staticmethod
    def _protocol(*categories: str) -> dict:
        return {"provenance": {"required_artifacts": list(categories)}}

    @staticmethod
    def _resign(manifest: dict) -> dict:
        manifest.pop("artifact_manifest_sha256", None)
        manifest["artifact_manifest_sha256"] = benchmark.object_sha256(manifest)
        return manifest

    @staticmethod
    def _basic_root(root: Path) -> None:
        write_json_atomic(root / "frozen_protocol.json", {"frozen": True})
        (root / "dependency_lock.txt").write_text("package==1\n", encoding="utf-8")

    def test_build_and_validate_bind_every_exact_identity_field(self) -> None:
        matrix = self._matrix()
        protocol = self._protocol("frozen_protocol", "dependency_lock")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._basic_root(root)
            manifest = benchmark.build_artifact_manifest(root, matrix, protocol)
            benchmark.validate_artifact_manifest(manifest, root, matrix, protocol)

            mutations = {
                "schema_version": "wrong-schema",
                "status": "incomplete",
                "profile": "confirmatory",
                "evidence_class": "wrong-evidence",
                "claim_eligible": 0,
                "protocol_sha256": "a" * 64,
                "source_sha256": "b" * 64,
                "matrix_sha256": "c" * 64,
            }
            for field, value in mutations.items():
                changed = self._resign({**copy.deepcopy(manifest), field: value})
                with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, "identity does not match"
                ):
                    benchmark.validate_artifact_manifest(
                        changed, root, matrix, protocol
                    )

            extra = copy.deepcopy(manifest)
            extra["unregistered"] = True
            self._resign(extra)
            with self.assertRaisesRegex(
                ValueError, "keys are incomplete or contain extras"
            ):
                benchmark.validate_artifact_manifest(extra, root, matrix, protocol)

    def test_manifest_file_set_must_equal_every_retained_file(self) -> None:
        matrix = self._matrix()
        protocol = self._protocol("frozen_protocol", "dependency_lock")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._basic_root(root)
            write_json_atomic(root / "artifact_manifest.json", {"stale": True})
            nested = root / "nested" / "artifact_manifest.json"
            nested.parent.mkdir()
            write_json_atomic(nested, {"nested": True})
            manifest = benchmark.build_artifact_manifest(root, matrix, protocol)
            paths = [entry["path"] for entry in manifest["files"]]
            self.assertNotIn("artifact_manifest.json", paths)
            self.assertIn("nested/artifact_manifest.json", paths)
            benchmark.validate_artifact_manifest(manifest, root, matrix, protocol)

            (root / "unlisted.bin").write_bytes(b"not in the signed manifest")
            with self.assertRaisesRegex(ValueError, "exactly match"):
                benchmark.validate_artifact_manifest(manifest, root, matrix, protocol)

    def test_post_seal_cluster_profile_marker_is_the_only_explicit_exclusion(self) -> None:
        matrix = self._matrix()
        protocol = self._protocol("frozen_protocol", "dependency_lock")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._basic_root(root)
            marker = root / "cluster_profile_verified.json"
            marker.write_text('{"status":"verified_complete"}\n', encoding="utf-8")
            nested = root / "nested" / "cluster_profile_verified.json"
            nested.parent.mkdir()
            nested.write_text('{"status":"not_exempt"}\n', encoding="utf-8")
            manifest = benchmark.build_artifact_manifest(root, matrix, protocol)
            paths = {entry["path"] for entry in manifest["files"]}
            self.assertNotIn("cluster_profile_verified.json", paths)
            self.assertIn("nested/cluster_profile_verified.json", paths)
            benchmark.validate_artifact_manifest(manifest, root, matrix, protocol)

    def test_forged_cell_and_smoke_hpo_artifacts_cannot_be_resigned(self) -> None:
        protocol = self._protocol("frozen_protocol", "dependency_lock")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._basic_root(root)
            matrix = {
                **self._matrix(),
                "cells": [{"stage": "rollout", "cell_id": "canonical"}],
            }
            (root / "rollout" / "canonical").mkdir(parents=True)
            (root / "rollout" / "forged-cell").mkdir()
            with self.assertRaisesRegex(ValueError, "differ from the frozen matrix"):
                benchmark.build_artifact_manifest(root, matrix, protocol)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._basic_root(root)
            write_json_atomic(root / "hpo_selection.json", {"forged": True})
            with self.assertRaisesRegex(
                ValueError, "profile-inapplicable root artifacts"
            ):
                benchmark.build_artifact_manifest(root, self._matrix(), protocol)

    def test_nested_result_below_a_canonical_cell_cannot_be_resigned(self) -> None:
        protocol = self._protocol("frozen_protocol", "dependency_lock")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._basic_root(root)
            matrix = {
                **self._matrix(),
                "cells": [{"stage": "rollout", "cell_id": "canonical"}],
            }
            canonical = root / "rollout" / "canonical"
            canonical.mkdir(parents=True)
            write_json_atomic(canonical / "result.json", {"canonical": True})
            forged = canonical / "forged"
            forged.mkdir()
            write_json_atomic(forged / "result.json", {"forged": True})
            with self.assertRaisesRegex(
                ValueError, "result.json paths differ from the frozen matrix"
            ):
                benchmark.build_artifact_manifest(root, matrix, protocol)

    def test_smoke_cannot_promote_an_unvalidated_diagnostics_file(self) -> None:
        protocol = self._protocol(
            "frozen_protocol",
            "dependency_lock",
            "diagnostic_raw_samples_estimands_and_threshold_interpretations",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._basic_root(root)
            matrix = {**self._matrix(), "cells": []}
            diagnostics = root / "diagnostics"
            diagnostics.mkdir()
            (diagnostics / "forged.txt").write_text(
                "not diagnostic evidence\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "profile-inapplicable diagnostics tree"
            ):
                benchmark.build_artifact_manifest(root, matrix, protocol)

    def test_confirmatory_diagnostic_layout_uses_only_registered_diagnostics(self) -> None:
        protocol = read_json(
            Path(__file__).resolve().parents[1] / "matched_objective_protocol.json"
        )
        matrix = {**self._matrix("confirmatory"), "cells": []}
        paths = benchmark._expected_diagnostic_artifact_paths(
            Path("/tmp/diagnostic-layout-test"), matrix, protocol
        )
        self.assertEqual(len(paths), 21)
        relative = {
            path.relative_to("/tmp/diagnostic-layout-test").as_posix()
            for path in paths
        }
        registered = set(
            protocol["diagnostics"][
                "required_before_confirmatory_interpretation"
            ]
        )
        observed = {
            path.split("/")[1]
            for path in relative
            if path != "diagnostics/diagnostic_summary.json"
        }
        self.assertEqual(observed, registered)

    def test_absolute_parent_and_windows_paths_are_rejected(self) -> None:
        matrix = self._matrix()
        protocol = self._protocol("frozen_protocol", "dependency_lock")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._basic_root(root)
            manifest = benchmark.build_artifact_manifest(root, matrix, protocol)
            for unsafe in ("../escape", "/tmp/escape", "C:\\escape"):
                changed = copy.deepcopy(manifest)
                changed["files"][0]["path"] = unsafe
                self._resign(changed)
                with self.subTest(path=unsafe), self.assertRaisesRegex(
                    ValueError, "not canonical and relative"
                ):
                    benchmark.validate_artifact_manifest(
                        changed, root, matrix, protocol
                    )

            changed = copy.deepcopy(manifest)
            changed["categories"]["frozen_protocol"]["files"] = ["../escape"]
            self._resign(changed)
            with self.assertRaisesRegex(ValueError, "not canonical and relative"):
                benchmark.validate_artifact_manifest(changed, root, matrix, protocol)

    def test_category_assignments_are_recomputed_from_disk(self) -> None:
        matrix = self._matrix()
        protocol = self._protocol("frozen_protocol", "dependency_lock")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._basic_root(root)
            manifest = benchmark.build_artifact_manifest(root, matrix, protocol)
            changed = copy.deepcopy(manifest)
            changed["categories"]["frozen_protocol"]["files"] = [
                "dependency_lock.txt"
            ]
            self._resign(changed)
            with self.assertRaisesRegex(ValueError, "assignments do not recompute"):
                benchmark.validate_artifact_manifest(changed, root, matrix, protocol)

    def test_confirmatory_selection_is_exactly_matrix_selection(self) -> None:
        matrix = self._matrix("confirmatory")
        protocol = self._protocol("hpo_trial_metrics_and_selection_manifest")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json_atomic(root / "hpo_selection.json", matrix["selection_manifest"])
            manifest = benchmark.build_artifact_manifest(root, matrix, protocol)
            benchmark.validate_artifact_manifest(manifest, root, matrix, protocol)

            wrong = {"selection_sha256": "f" * 64, "selected": {}}
            write_json_atomic(root / "hpo_selection.json", wrong)
            with self.assertRaisesRegex(ValueError, "does not match matrix"):
                benchmark.build_artifact_manifest(root, matrix, protocol)

            changed = copy.deepcopy(manifest)
            entry = next(
                item for item in changed["files"] if item["path"] == "hpo_selection.json"
            )
            selection_path = root / "hpo_selection.json"
            entry["size"] = selection_path.stat().st_size
            entry["sha256"] = benchmark.file_sha256(selection_path)
            self._resign(changed)
            with self.assertRaisesRegex(ValueError, "does not match matrix"):
                benchmark.validate_artifact_manifest(changed, root, matrix, protocol)

    def test_symlinked_artifacts_are_not_accepted_as_root_files(self) -> None:
        matrix = self._matrix()
        protocol = self._protocol("frozen_protocol", "dependency_lock")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "run"
            root.mkdir()
            self._basic_root(root)
            external = base / "outside.bin"
            external.write_bytes(b"outside")
            try:
                os.symlink(external, root / "linked.bin")
            except (OSError, NotImplementedError) as error:  # pragma: no cover
                self.skipTest(f"symlinks unavailable: {error}")
            with self.assertRaisesRegex(ValueError, "contains a symlink"):
                benchmark.build_artifact_manifest(root, matrix, protocol)


if __name__ == "__main__":
    unittest.main()
