import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from dreamer_imf_compare import staged_study as study


class EvidenceTests(unittest.TestCase):
    def fixture(self):
        root = Path(tempfile.mkdtemp(prefix="staged-evidence-"))
        cell = dict(index=0, arm="gaussian", seed=431)
        directory = study.evidence.cell_dir(root, "preflight", cell)
        directory.mkdir(parents=True)
        (directory / "diagnostic_batch.npz").write_bytes(b"fixture-batch")
        study.evidence.publish(
            directory / "diagnostic.json", dict(learner_unchanged=True, metric=1.0)
        )
        result = dict(
            native_steps=4096,
            learner_updates=8,
            staged=dict(
                protocol="staged-scratch-v1",
                native=4096,
                updates=8,
                batch_sha256=study.evidence.filehash(
                    directory / "diagnostic_batch.npz"
                ),
                diagnostics=[
                    dict(
                        path="diagnostic.json",
                        sha256=study.evidence.filehash(directory / "diagnostic.json"),
                    )
                ],
                freeze_checks=[dict(passed=True)],
                controls=[dict(values=dict(imag_horizon=h)) for h in (5, 15)],
            ),
        )
        study.evidence.publish(directory / "complete.json", result)
        return root, cell, directory, result

    def test_exact_cell_counts(self):
        p = study.evidence.read(study.evidence.PROTOCOL)
        self.assertEqual(len(study.evidence.cells(p, "preflight")), 2)
        self.assertEqual(len(study.evidence.cells(p, "training")), 6)

    def test_digest_checks_precede_marker_publication(self):
        root, cell, directory, result = self.fixture()
        with mock.patch.object(
            study, "_original_verify", return_value=result
        ) as verify:
            self.assertEqual(study.verify_cell(root, "preflight", cell), result)
            verify.assert_called_once()
        (directory / "diagnostic.json").write_text("{}")
        with mock.patch.object(study, "_original_verify") as verify:
            with self.assertRaisesRegex(ValueError, "digest"):
                study.verify_cell(root, "preflight", cell)
            verify.assert_not_called()

    def test_missing_phase_is_rejected(self):
        root, cell, directory, result = self.fixture()
        result["staged"]["controls"] = [dict(values=dict(imag_horizon=5))]
        (directory / "complete.json").write_text(json.dumps(result))
        with mock.patch.object(study, "_original_verify") as verify:
            with self.assertRaisesRegex(ValueError, "both reliance"):
                study.verify_cell(root, "preflight", cell)
            verify.assert_not_called()

    def test_uncertain_submission_never_duplicates(self):
        root = Path(tempfile.mkdtemp(prefix="staged-submit-"))
        (root / "submissions").mkdir()
        (root / "submissions/preflight-intent.json").write_text("{}")
        with mock.patch.object(
            study.evidence, "manifest", return_value=({}, {})
        ), mock.patch.object(study.subprocess, "check_output") as submit:
            with self.assertRaises(FileExistsError):
                study.submit(root, "preflight", "gpu-l40s")
            submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
