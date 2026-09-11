"""Focused fail-closed tests for the NeurIPS verification entry points."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = PROJECT / "scripts" / f"{name}.py"
    specification = importlib.util.spec_from_file_location(
        f"_neurips_test_{name}", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not import verifier script {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


core = _load_script("verify_neurips_core")
idempotence = _load_script("verify_smoke_idempotence")


class NeurIPSCoreVerifierTests(unittest.TestCase):
    def test_source_manifest_covers_verifiers_and_their_tests(self) -> None:
        source_files = set(core.benchmark._source_files(core.WORKSPACE))
        expected = {
            "dreamer_imf_comparison/scripts/verify_neurips_core.py",
            "dreamer_imf_comparison/scripts/verify_smoke_idempotence.py",
            "dreamer_imf_comparison/scripts/verify_trajectory_imf_theory.py",
            "dreamer_imf_comparison/TRAJECTORY_IMF_THEORY.md",
            "dreamer_imf_comparison/neurips_controls_protocol.json",
            "dreamer_imf_comparison/pixel_benchmark_protocol.json",
        }
        expected.update(
            module.replace(".", "/") + ".py"
            for module in core.COMPARISON_MODULES
        )
        self.assertTrue(expected.issubset(source_files), expected - source_files)

    def test_expected_rejection_requires_the_frozen_error_substring(self) -> None:
        core._expect_rejection(
            lambda: (_ for _ in ()).throw(ValueError("prefix exact cause suffix")),
            "exact cause",
            "rejection was not observed",
        )
        with self.assertRaisesRegex(AssertionError, "expected error containing"):
            core._expect_rejection(
                lambda: (_ for _ in ()).throw(ValueError("different cause")),
                "exact cause",
                "wrong rejection",
            )
        with self.assertRaisesRegex(AssertionError, "was accepted"):
            core._expect_rejection(lambda: None, "exact cause", "tamper was accepted")

    def test_regression_command_timeout_fails_closed(self) -> None:
        timeout = subprocess.TimeoutExpired(["python", "-m", "unittest"], 7)
        with mock.patch.object(core.subprocess, "run", side_effect=timeout):
            with self.assertRaisesRegex(
                RuntimeError, "verification command timed out after 7 seconds"
            ):
                core._run(["python", "-m", "unittest"], timeout_seconds=7)

    def test_frozen_identity_manifest_rejects_any_test_drift(self) -> None:
        frozen = {
            "library": {
                "count": 1,
                "identity_sha256": core.hashlib.sha256(
                    b"a.Test.test_one"
                ).hexdigest(),
            },
            "comparison": {
                "count": 1,
                "identity_sha256": core.hashlib.sha256(
                    b"b.Test.test_two"
                ).hexdigest(),
            },
        }
        core._assert_frozen_test_identities(
            {"library": ("a.Test.test_one",), "comparison": ("b.Test.test_two",)},
            frozen,
        )
        with self.assertRaisesRegex(AssertionError, "identity manifest mismatch"):
            core._assert_frozen_test_identities(
                {
                    "library": ("a.Test.test_one", "a.Test.test_three"),
                    "comparison": ("b.Test.test_two",),
                },
                frozen,
            )
        with self.assertRaisesRegex(AssertionError, "suite names differ"):
            core._assert_frozen_test_identities(
                {"library": ("a.Test.test_one",)}, frozen
            )

    def test_manual_iqm_rollout_auc_and_actor_returns(self) -> None:
        self.assertEqual(core._manual_iqm([0.0, 1.0, 2.0, 3.0]), 1.5)
        arrays = {
            "observation_samples": np.asarray(
                [[[[1.0], [4.0]]], [[[3.0], [6.0]]]], dtype=np.float32
            ),
            "target_observations": np.asarray([[[2.0], [2.0]]], dtype=np.float32),
            "training_observation_std": np.asarray([2.0], dtype=np.float32),
        }
        errors, auc = core._manual_rollout_metrics(arrays, [1, 2])
        self.assertEqual(errors, {"1": 0.25, "2": 2.5})
        self.assertEqual(auc, 1.375)
        returns = core._manual_actor_returns(
            {
                "rewards": np.asarray([[1.0, 2.0, 99.0], [4.0, 99.0, 99.0]]),
                "lengths": np.asarray([2, 1]),
            }
        )
        np.testing.assert_array_equal(returns, np.asarray([3.0, 4.0]))


class SmokeIdempotenceVerifierTests(unittest.TestCase):
    def test_retained_snapshot_binds_exact_paths_sizes_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stage" / "cell").mkdir(parents=True)
            artifact = root / "stage" / "cell" / "result.json"
            artifact.write_text("one", encoding="utf-8")
            before = idempotence._retained_snapshot(root)
            self.assertEqual(set(before), {"stage/cell/result.json"})
            same = idempotence._retained_snapshot(root)
            idempotence._assert_retained_snapshot_equal(before, same)
            artifact.write_text("two", encoding="utf-8")
            after = idempotence._retained_snapshot(root)
            with self.assertRaisesRegex(AssertionError, "changed=.*result.json"):
                idempotence._assert_retained_snapshot_equal(before, after)
            (root / "new.txt").write_text("new", encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "added=.*new.txt"):
                idempotence._assert_retained_snapshot_equal(
                    after, idempotence._retained_snapshot(root)
                )

    def test_cell_snapshot_includes_mtime_but_retained_snapshot_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell_dir = root / "actor" / "actor-test"
            cell_dir.mkdir(parents=True)
            artifact = cell_dir / "result.json"
            artifact.write_text("immutable", encoding="utf-8")
            matrix = {"cells": [{"stage": "actor", "cell_id": "actor-test"}]}
            retained_before = idempotence._retained_snapshot(root)
            cells_before = idempotence._cell_snapshot(root, matrix)
            stat = artifact.stat()
            os.utime(
                artifact,
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
            )
            retained_after = idempotence._retained_snapshot(root)
            cells_after = idempotence._cell_snapshot(root, matrix)
            self.assertEqual(retained_before, retained_after)
            self.assertNotEqual(cells_before, cells_after)
            self.assertGreaterEqual(artifact.stat().st_mtime_ns, stat.st_mtime_ns)

    def test_idempotence_rerun_timeout_fails_closed(self) -> None:
        timeout = subprocess.TimeoutExpired(["python", "benchmark"], 11)
        with mock.patch.object(idempotence.subprocess, "run", side_effect=timeout):
            with self.assertRaisesRegex(
                RuntimeError, "smoke idempotence rerun timed out after 11 seconds"
            ):
                idempotence._rerun_smoke(
                    Path("/tmp/output"), Path("/tmp/workspace"), timeout_seconds=11
                )


if __name__ == "__main__":
    unittest.main()
