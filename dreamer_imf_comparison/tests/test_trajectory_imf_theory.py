"""Regression tests for the conditional-to-rollout theorem package."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "dreamer_imf_comparison" / "scripts" / "verify_trajectory_imf_theory.py"
THEORY_PATH = REPO_ROOT / "dreamer_imf_comparison" / "TRAJECTORY_IMF_THEORY.md"
PROTOCOL_PATH = REPO_ROOT / "dreamer_imf_comparison" / "matched_objective_protocol.json"

SPEC = importlib.util.spec_from_file_location("verify_trajectory_imf_theory", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
theory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = theory
SPEC.loader.exec_module(theory)


class TheoryDocumentTests(unittest.TestCase):
    def test_checked_in_document_verifies(self) -> None:
        evidence = theory.verify_document(THEORY_PATH, PROTOCOL_PATH)
        self.assertGreater(evidence["bytes"], 20_000)
        self.assertGreaterEqual(evidence["proof_blocks"], 8)
        self.assertGreaterEqual(evidence["proved_statuses"], 8)

    def test_missing_endpoint_assumption_fails_closed(self) -> None:
        original = THEORY_PATH.read_text(encoding="utf-8")
        damaged = original.replace(
            "**Assumption A3 (endpoint calibration).**",
            "**Deleted A3.**",
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "theory.md"
            path.write_text(damaged, encoding="utf-8")
            with self.assertRaisesRegex(theory.VerificationError, "A3"):
                theory.verify_document(path, PROTOCOL_PATH)

    def test_missing_euclidean_integrability_or_trace_bridge_fails_closed(self) -> None:
        original = THEORY_PATH.read_text(encoding="utf-8")
        mutations = {
            "euclidean specialization": ("\\mathsf X=\\mathbb R^d", "\\mathsf X=\\mathsf S"),
            "metric bridge": (
                "W_{1,\\widetilde d_X}(P,Q)\\le B_hW_{1,\\|\\cdot\\|_2}(P,Q)",
                "W_1(P,Q)\\le W_1(P,Q)",
            ),
            "square integrability": ("square-integrable under", "integrable under"),
            "endpoint trace": ("trace at \\(s=1\\)", "interior limit near \\(s=1\\)"),
        }
        with tempfile.TemporaryDirectory() as directory:
            for label, (needle, replacement) in mutations.items():
                with self.subTest(label=label):
                    self.assertIn(needle, original)
                    path = Path(directory) / (label.replace(" ", "_") + ".md")
                    path.write_text(original.replace(needle, replacement, 1), encoding="utf-8")
                    with self.assertRaises(theory.VerificationError):
                        theory.verify_document(path, PROTOCOL_PATH)

    def test_unqualified_overclaim_fails_closed(self) -> None:
        original = THEORY_PATH.read_text(encoding="utf-8")
        damaged = original + "\nThe current objective has a proved endpoint certificate.\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "theory.md"
            path.write_text(damaged, encoding="utf-8")
            with self.assertRaisesRegex(theory.VerificationError, "overclaim"):
                theory.verify_document(path, PROTOCOL_PATH)

    def test_protocol_drift_fails_closed(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        damaged = deepcopy(protocol)
        damaged["canonical_executable_config"]["dreamer_config_non_task_shape_common"][
            "imf_endpoint_scale"
        ] = 0.1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(damaged), encoding="utf-8")
            with self.assertRaisesRegex(theory.VerificationError, "endpoint-loss limitation"):
                theory.verify_document(THEORY_PATH, path)


class FiniteStateTheoryTests(unittest.TestCase):
    def test_finite_state_positive_control(self) -> None:
        result = theory.run_finite_state_control(horizons=12)
        self.assertLessEqual(result.final_exact_w1, result.final_recursive_bound + 1e-14)
        self.assertGreaterEqual(result.minimum_recursion_slack, -1e-14)
        self.assertGreaterEqual(result.maximum_change_of_measure_slack, 0.0)

    def test_coverage_counterexample(self) -> None:
        result = theory.run_finite_state_control(horizons=4)
        self.assertEqual(result.no_coverage_training_risk, 0.0)
        self.assertEqual(result.no_coverage_generated_error, 1.0)

    def test_bernoulli_transport_is_unmatched_mass(self) -> None:
        # Independently enumerate how much probability can stay on the diagonal.
        for first in np.linspace(0.0, 1.0, 11):
            for second in np.linspace(0.0, 1.0, 11):
                diagonal = min(first, second) + min(1.0 - first, 1.0 - second)
                enumerated_cost = 1.0 - diagonal
                self.assertAlmostEqual(
                    theory.bernoulli_w1(float(first), float(second)),
                    enumerated_cost,
                    places=14,
                )

    def test_finite_state_recursion_and_change_measure_property_grid(self) -> None:
        rng = np.random.default_rng(20260911)
        for _ in range(500):
            true_kernel = rng.uniform(0.0, 1.0, size=2)
            learned_kernel = rng.uniform(0.0, 1.0, size=2)
            true_state, learned_state = rng.uniform(0.0, 1.0, size=2)
            next_true = (1.0 - true_state) * true_kernel[0] + true_state * true_kernel[1]
            next_learned = (
                (1.0 - learned_state) * learned_kernel[0]
                + learned_state * learned_kernel[1]
            )
            rho = abs(true_kernel[1] - true_kernel[0])
            errors = np.abs(true_kernel - learned_kernel)
            delta = (1.0 - learned_state) * errors[0] + learned_state * errors[1]
            bound = rho * abs(true_state - learned_state) + delta
            self.assertLessEqual(abs(next_true - next_learned), bound + 2e-15)

            training_one = float(rng.uniform(0.02, 0.98))
            weights = np.array(
                [(1.0 - learned_state) / (1.0 - training_one), learned_state / training_one]
            )
            mu = np.array([1.0 - training_one, training_one])
            transferred = float(np.dot(np.array([1.0 - learned_state, learned_state]), errors))
            cs_bound = math.sqrt(float(np.dot(mu, weights * weights))) * math.sqrt(
                float(np.dot(mu, errors * errors))
            )
            self.assertLessEqual(transferred, cs_bound + 2e-15)


class LinearGaussianTheoryTests(unittest.TestCase):
    def test_linear_gaussian_positive_control(self) -> None:
        result = theory.run_linear_gaussian_control(horizons=12)
        self.assertLessEqual(result.final_exact_w1, result.final_recursive_bound + 2e-12)
        self.assertLessEqual(result.maximum_recursion_violation, 2e-12)
        self.assertLessEqual(result.maximum_change_of_measure_violation, 1e-12)
        self.assertLess(result.maximum_quantile_formula_error, 1.5e-2)

    def test_gaussian_density_ratio_special_cases(self) -> None:
        self.assertAlmostEqual(
            theory.gaussian_density_ratio_second_moment(0.0, 1.0, 0.0, 1.0),
            1.0,
            places=14,
        )
        shift = 0.7
        self.assertAlmostEqual(
            theory.gaussian_density_ratio_second_moment(shift, 1.0, 0.0, 1.0),
            math.exp(shift * shift),
            places=13,
        )
        self.assertTrue(
            math.isinf(theory.gaussian_density_ratio_second_moment(0.0, 2.0, 0.0, 1.0))
        )

    def test_normal_absolute_mean_against_gauss_hermite(self) -> None:
        roots, weights = np.polynomial.hermite.hermgauss(64)
        for mean, sd in ((0.0, 1.0), (0.3, 0.4), (-1.2, 0.2), (2.0, 0.0)):
            exact = theory.normal_absolute_mean(mean, sd)
            if sd == 0.0:
                quadrature = abs(mean)
            else:
                values = np.abs(mean + sd * math.sqrt(2.0) * roots)
                quadrature = float(np.sum(weights * values) / math.sqrt(math.pi))
            self.assertLess(abs(exact - quadrature), 1.5e-2)

    def test_regular_kernel_assumptions_have_numeric_negative_controls(self) -> None:
        result = theory.run_linear_gaussian_control(horizons=12)
        self.assertEqual(result.unstable_growth_ratio, 4096.0)
        self.assertGreaterEqual(result.discontinuous_kernel_ratio, 500_000.0)

    def test_linear_gaussian_recursion_property_grid(self) -> None:
        rng = np.random.default_rng(90210)
        for _ in range(300):
            true_slope = float(rng.uniform(-1.4, 1.4))
            learned_slope = float(rng.uniform(-1.4, 1.4))
            learned_bias = float(rng.uniform(-0.3, 0.3))
            noise_sd = float(rng.uniform(0.05, 0.7))
            true_mean, learned_mean = rng.normal(size=2)
            true_sd, learned_sd = rng.uniform(0.05, 1.5, size=2)
            current = theory.gaussian_w1(
                float(true_mean), float(true_sd), float(learned_mean), float(learned_sd)
            )
            gap = true_slope - learned_slope
            delta = theory.normal_absolute_mean(
                gap * float(learned_mean) - learned_bias,
                abs(gap) * float(learned_sd),
            )
            next_true_mean = true_slope * float(true_mean)
            next_learned_mean = learned_slope * float(learned_mean) + learned_bias
            next_true_sd = math.sqrt((true_slope * float(true_sd)) ** 2 + noise_sd ** 2)
            next_learned_sd = math.sqrt((learned_slope * float(learned_sd)) ** 2 + noise_sd ** 2)
            next_distance = theory.gaussian_w1(
                next_true_mean, next_true_sd, next_learned_mean, next_learned_sd
            )
            self.assertLessEqual(next_distance, abs(true_slope) * current + delta + 3e-14)


class ObjectiveFailureTests(unittest.TestCase):
    def test_differential_identity_requires_velocity_correction(self) -> None:
        result = theory.run_differential_identity_control()
        self.assertLess(result.corrected_identity_error, 2e-13)
        self.assertGreater(result.omitted_velocity_correction_error, 1e-2)

    def test_endpoint_and_adaptive_failures(self) -> None:
        result = theory.run_objective_counterexamples()
        self.assertEqual(result.endpoint_scheduled_risk, 0.0)
        self.assertEqual(result.endpoint_error_squared, 49.0)
        self.assertLess(result.endpoint_no_atom_spike_risk, 1e-6)
        self.assertEqual(result.endpoint_no_atom_error_squared, 1.0)
        self.assertGreaterEqual(result.endpoint_no_atom_error_to_risk_ratio, 3e6)
        self.assertLess(result.adaptive_large_residual_value, 1.000000000001)
        self.assertGreater(result.adaptive_raw_to_weighted_ratio, 1e11)

    def test_independent_pairing_is_not_wasserstein(self) -> None:
        result = theory.run_objective_counterexamples()
        self.assertEqual(result.identical_law_w1, 0.0)
        self.assertEqual(result.common_noise_pair_cost, 0.0)
        self.assertEqual(result.independent_pair_cost, 0.5)

    def test_cli_numeric_success_token(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--numerics"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(theory.NUMERIC_SUCCESS, completed.stdout)


if __name__ == "__main__":
    unittest.main()
