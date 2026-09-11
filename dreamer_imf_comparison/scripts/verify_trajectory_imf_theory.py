#!/usr/bin/env python3
"""Offline verification and numerical controls for the trajectory-iMF theory note.

The document checker is deliberately structural: it makes it difficult to delete a
required assumption, counterexample, or claim limitation accidentally.  The numerical
checks independently exercise the finite-state change-of-measure/rollout recursion and
the one-dimensional linear-Gaussian recursion.  They are controls, not empirical proof
about a trained neural network.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


SUCCESS = "TRAJECTORY_IMF_THEORY_VERIFIED"
NUMERIC_SUCCESS = "TRAJECTORY_IMF_THEORY_NUMERICS_VERIFIED"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THEORY = REPO_ROOT / "dreamer_imf_comparison" / "TRAJECTORY_IMF_THEORY.md"
DEFAULT_PROTOCOL = REPO_ROOT / "dreamer_imf_comparison" / "matched_objective_protocol.json"
DEFAULT_TRAJECTORY_SOURCE = (
    REPO_ROOT / "imf_dreamer_jax" / "src" / "imf_dreamer_jax" / "trajectory.py"
)

REQUIRED_HEADINGS = (
    "# Conditional-to-rollout theory for trajectory iMF",
    "## 1. Status and claim boundary",
    "## 2. Measurable setup",
    "## 3. The fixed-context, per-token iMF derivative",
    "## 4. From raw iMF regression to an endpoint certificate",
    "## 5. Generated-context transfer",
    "## 6. Multi-step rollout theorem",
    "## 7. Reductions",
    "## 8. What the present implementation does not satisfy",
    "## 9. Counterexamples",
    "## 10. Executable controls",
    "## 11. Relation to prior results",
    "## 12. Publication-safe theorem statement",
)

# These are semantic anchors, not a proof checker.  Each corresponds to an item that
# previously made the proposed argument unsound when omitted.
REQUIRED_ANCHORS = (
    "Status: **proved**",
    "Status: **assumption, not established for the current trained model**",
    "Ionescu--Tulcea",
    "standard Borel",
    "Borel probability kernel",
    "finite first moment",
    "common policy",
    "oracle characteristic",
    "same base-noise draw",
    "Euclidean metric",
    "metric domination bridge",
    "square-integrable under",
    "W^{1,1}([0,1];\\mathbb R^d)",
    "trace at \\(s=1\\)",
    "fundamental theorem of calculus",
    "Bayes excess risk",
    "Pythagorean regression identity",
    "endpoint-calibration assumption",
    "Radon--Nikodym",
    "absolute continuity",
    "chi-square",
    "Cauchy--Schwarz",
    "true closed-loop kernel",
    "fixed-policy result",
    "not a policy-improvement guarantee",
    "ordinary iMF",
    "conditional flow matching",
    "one-step dynamics",
    "no atom at `r=0`",
    "`imf_adaptive_power = 1.0`",
    "`imf_endpoint_scale = 0.0`",
    "shared-noise augmentation",
    "does not establish epistemic uncertainty",
    "coverage failure",
    "endpoint-support failure",
    "adaptive-weight failure",
    "kernel-regularity failure",
    "independent-pairing failure",
    "TRAJECTORY_IMF_THEORY_NUMERICS_VERIFIED",
)

REQUIRED_LABELS = (
    "**Lemma 0 (metric domination bridge).**",
    "**Assumption A1 (conditional flow regularity).**",
    "**Assumption A2 (raw endpoint-slice regression).**",
    "**Lemma 1 (regression projection).**",
    "**Lemma 2 (iMF differential certificate).**",
    "**Lemma 3 (common-noise endpoint coupling).**",
    "**Assumption A3 (endpoint calibration).**",
    "**Lemma 4 (generated-context change of measure).**",
    "**Theorem 1 (conditional residual to rollout error).**",
    "**Corollary 1 (fixed-policy return error).**",
)

REQUIRED_FORMULA_FRAGMENTS = (
    "\\mathsf X=\\mathbb R^d",
    "W_{1,\\widetilde d_X}(P,Q)\\le B_hW_{1,\\|\\cdot\\|_2}(P,Q)",
    "=\\partial_tu_\\theta+J_zu_\\theta\\,v_\\theta",
    "D_{h+1} \\le \\rho_hD_h+\\delta_h",
    "1+\\chi^2(\\nu_h^\\pi\\Vert\\mu_h)",
    "T_{\\theta,h}(c,a,\\xi)=\\xi-u_{\\theta,h}(\\xi,c,a,0,1)",
    "\\mathbb E_{\\mu_h}e_h^2\\le\\kappa_h\\mathcal E_h",
    "\\delta_h\\le L_hM_h\\sqrt{\\kappa_h\\mathcal E_h}",
)

REQUIRED_PRIMARY_LINKS = (
    "https://arxiv.org/abs/2210.02747",
    "https://arxiv.org/abs/2505.13447",
    "https://arxiv.org/abs/2512.02012",
    "https://arxiv.org/abs/2407.01392",
    "https://arxiv.org/abs/2606.25473",
)

FORBIDDEN_OVERCLAIMS = (
    "the current objective satisfies theorem 1",
    "the current objective has a proved endpoint certificate",
    "trajectory imf guarantees stable rollouts",
    "we prove population consistency of the current objective",
    "we prove all-subsequence consistency",
    "we prove planner safety",
    "outperforms dreamer 4",
)


class VerificationError(RuntimeError):
    """Expected verification failure with a concise message."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def verify_document(
    theory_path: Path,
    protocol_path: Path = DEFAULT_PROTOCOL,
    trajectory_source_path: Path = DEFAULT_TRAJECTORY_SOURCE,
) -> dict[str, Any]:
    """Check that the note retains every theorem dependency and limitation."""

    require(theory_path.is_file(), f"theory note does not exist: {theory_path}")
    text = theory_path.read_text(encoding="utf-8")
    lower = text.lower()
    for heading in REQUIRED_HEADINGS:
        require(heading in text, f"missing required heading: {heading}")
    for anchor in REQUIRED_ANCHORS:
        require(anchor.lower() in lower, f"missing required theory anchor: {anchor}")
    for label in REQUIRED_LABELS:
        require(label in text, f"missing required labeled result: {label}")
    for fragment in REQUIRED_FORMULA_FRAGMENTS:
        require(fragment in text, f"missing required formula fragment: {fragment}")
    for link in REQUIRED_PRIMARY_LINKS:
        require(link in text, f"missing primary-source link: {link}")
    for phrase in FORBIDDEN_OVERCLAIMS:
        require(phrase not in lower, f"forbidden unqualified overclaim is present: {phrase!r}")

    # A proof label must be present after each proved result, while assumptions must not
    # be silently relabeled as theorems.
    require(text.count("*Proof.*") >= 6, "expected at least six explicit proof blocks")
    require(text.count("Status: **proved**") >= 6, "expected at least six proved-status labels")
    require(
        text.count("Status: **assumption, not established for the current trained model**") >= 1,
        "endpoint calibration must remain explicitly an assumption",
    )
    require("Conjecture" not in text or "Status: **unproved conjecture**" in text,
            "every conjecture must carry an unproved status")

    require(protocol_path.is_file(), f"protocol does not exist: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    shared = (
        protocol.get("canonical_executable_config", {})
        .get("dreamer_config_non_task_shape_common", {})
    )
    require(shared.get("imf_adaptive_power") == 1.0,
            "the documented adaptive-weight limitation no longer matches the protocol")
    require(shared.get("imf_endpoint_scale") == 0.0,
            "the documented endpoint-loss limitation no longer matches the protocol")
    arm = protocol.get("arms", {}).get("trajectory_imf", {})
    require(arm.get("objective") == "fixed_context_per_token_time_trajectory_imf",
            "theory note is bound to the fixed-context trajectory-iMF arm")

    require(trajectory_source_path.is_file(),
            f"trajectory source does not exist: {trajectory_source_path}")
    trajectory_source = trajectory_source_path.read_text(encoding="utf-8")
    for snippet in (
        "def sample_trajectory_time_pairs(",
        "samples = jax.nn.sigmoid",
        "return jnp.where(on_boundary, t, r), t",
    ):
        require(snippet in trajectory_source,
                f"trajectory schedule implementation drifted; missing {snippet!r}")

    return {
        "bytes": len(text.encode("utf-8")),
        "headings": len(REQUIRED_HEADINGS),
        "anchors": len(REQUIRED_ANCHORS),
        "proved_statuses": text.count("Status: **proved**"),
        "proof_blocks": text.count("*Proof.*"),
        "primary_links": len(REQUIRED_PRIMARY_LINKS),
        "schedule_source_checked": True,
    }


def bernoulli_w1(probability_a: float, probability_b: float) -> float:
    """W1 between Bernoulli laws for the metric d(0,1)=1."""

    require(0.0 <= probability_a <= 1.0, "invalid Bernoulli probability")
    require(0.0 <= probability_b <= 1.0, "invalid Bernoulli probability")
    # On two points the unmatched mass is the unique transport cost.
    return abs(probability_a - probability_b)


def normal_absolute_mean(mean: float, standard_deviation: float) -> float:
    """Return E|mean + standard_deviation * Z| for standard-normal Z."""

    require(standard_deviation >= 0.0, "standard deviation must be nonnegative")
    if standard_deviation == 0.0:
        return abs(mean)
    ratio = abs(mean) / standard_deviation
    return (
        standard_deviation * math.sqrt(2.0 / math.pi) * math.exp(-0.5 * ratio * ratio)
        + abs(mean) * math.erf(ratio / math.sqrt(2.0))
    )


def gaussian_w1(mean_a: float, sd_a: float, mean_b: float, sd_b: float) -> float:
    """Exact one-dimensional W1 via the monotone Gaussian quantile coupling."""

    return normal_absolute_mean(mean_a - mean_b, abs(sd_a - sd_b))


def gaussian_density_ratio_second_moment(
    nu_mean: float,
    nu_sd: float,
    mu_mean: float,
    mu_sd: float,
) -> float:
    """Return integral (d nu / d mu)^2 d mu for univariate Gaussians.

    It is finite exactly when ``nu_sd**2 < 2 * mu_sd**2``.
    """

    require(nu_sd > 0.0 and mu_sd > 0.0, "Gaussian scales must be positive")
    denominator_variance = 2.0 * mu_sd * mu_sd - nu_sd * nu_sd
    if denominator_variance <= 0.0:
        return math.inf
    prefactor = (mu_sd * mu_sd) / (nu_sd * math.sqrt(denominator_variance))
    exponent = ((nu_mean - mu_mean) ** 2) / denominator_variance
    return prefactor * math.exp(exponent)


@dataclass(frozen=True)
class FiniteStateControl:
    horizons: int
    maximum_recursion_slack: float
    minimum_recursion_slack: float
    maximum_change_of_measure_slack: float
    final_exact_w1: float
    final_recursive_bound: float
    no_coverage_training_risk: float
    no_coverage_generated_error: float


def run_finite_state_control(horizons: int = 12) -> FiniteStateControl:
    """Exercise Theorem 1 exactly on a two-state Markov model."""

    require(horizons >= 2, "finite-state control needs at least two horizons")
    # Entries are Pr(next state = 1 | current state = 0 or 1).
    true_kernel = (0.20, 0.70)
    learned_kernel = (0.25, 0.60)
    rho = abs(true_kernel[1] - true_kernel[0])
    require(abs(rho - 0.5) < 1e-15, "positive-control Lipschitz constant drifted")
    mu = (0.5, 0.5)

    true_probability = 0.30
    learned_probability = 0.30
    recursive_bound = 0.0
    recursion_slacks: list[float] = []
    change_slacks: list[float] = []

    for _ in range(horizons):
        errors = (
            bernoulli_w1(true_kernel[0], learned_kernel[0]),
            bernoulli_w1(true_kernel[1], learned_kernel[1]),
        )
        delta = (1.0 - learned_probability) * errors[0] + learned_probability * errors[1]
        density = (
            (1.0 - learned_probability) / mu[0],
            learned_probability / mu[1],
        )
        ratio_second_moment = mu[0] * density[0] ** 2 + mu[1] * density[1] ** 2
        l2_training_error = math.sqrt(mu[0] * errors[0] ** 2 + mu[1] * errors[1] ** 2)
        change_bound = math.sqrt(ratio_second_moment) * l2_training_error
        require(delta <= change_bound + 1e-14, "finite-state change-of-measure bound failed")
        change_slacks.append(change_bound - delta)

        next_true = (1.0 - true_probability) * true_kernel[0] + true_probability * true_kernel[1]
        next_learned = (
            (1.0 - learned_probability) * learned_kernel[0]
            + learned_probability * learned_kernel[1]
        )
        exact_next = bernoulli_w1(next_true, next_learned)
        one_step_bound = rho * bernoulli_w1(true_probability, learned_probability) + delta
        require(exact_next <= one_step_bound + 1e-14, "finite-state rollout recursion failed")
        recursion_slacks.append(one_step_bound - exact_next)
        recursive_bound = rho * recursive_bound + delta
        true_probability, learned_probability = next_true, next_learned

    final_exact = bernoulli_w1(true_probability, learned_probability)
    require(final_exact <= recursive_bound + 1e-14, "iterated finite-state bound failed")

    # Coverage counterexample: training observes context 0 only; generated inference
    # starts at context 1.  The model is exact on context 0 and maximally wrong on 1.
    no_coverage_training_risk = 0.0 * 0.0
    no_coverage_generated_error = bernoulli_w1(0.0, 1.0)
    require(no_coverage_training_risk == 0.0, "coverage counterexample risk must be zero")
    require(no_coverage_generated_error == 1.0, "coverage counterexample must have unit error")

    return FiniteStateControl(
        horizons=horizons,
        maximum_recursion_slack=max(recursion_slacks),
        minimum_recursion_slack=min(recursion_slacks),
        maximum_change_of_measure_slack=max(change_slacks),
        final_exact_w1=final_exact,
        final_recursive_bound=recursive_bound,
        no_coverage_training_risk=no_coverage_training_risk,
        no_coverage_generated_error=no_coverage_generated_error,
    )


@dataclass(frozen=True)
class LinearGaussianControl:
    horizons: int
    maximum_recursion_violation: float
    maximum_change_of_measure_violation: float
    maximum_quantile_formula_error: float
    final_exact_w1: float
    final_recursive_bound: float
    unstable_growth_ratio: float
    discontinuous_kernel_ratio: float


def _gaussian_quantile_w1_quadrature(
    mean_a: float, sd_a: float, mean_b: float, sd_b: float
) -> float:
    """Independent deterministic quadrature for the Gaussian W1 formula."""

    # Gauss-Hermite integrates against exp(-x^2); scaling gives standard normal.
    roots, weights = np.polynomial.hermite.hermgauss(64)
    differences = (mean_a - mean_b) + (sd_a - sd_b) * math.sqrt(2.0) * roots
    return float(np.sum(weights * np.abs(differences)) / math.sqrt(math.pi))


def run_linear_gaussian_control(horizons: int = 12) -> LinearGaussianControl:
    """Exercise the recursion for stable scalar affine Gaussian transitions."""

    require(horizons >= 2, "linear-Gaussian control needs at least two horizons")
    true_slope = 0.75
    learned_slope = 0.68
    learned_bias = 0.04
    noise_sd = 0.30
    rho = abs(true_slope)
    training_mean = 0.0
    training_sd = 2.5

    true_mean = learned_mean = 0.40
    true_sd = learned_sd = 0.80
    recursive_bound = 0.0
    recursion_violations: list[float] = []
    change_violations: list[float] = []
    quadrature_errors: list[float] = []

    for _ in range(horizons):
        delta_mean = (true_slope - learned_slope) * learned_mean - learned_bias
        delta_sd = abs(true_slope - learned_slope) * learned_sd
        generated_one_step_error = normal_absolute_mean(delta_mean, delta_sd)

        ratio_second_moment = gaussian_density_ratio_second_moment(
            learned_mean, learned_sd, training_mean, training_sd
        )
        require(math.isfinite(ratio_second_moment), "Gaussian coverage control lost finite chi-square")
        slope_gap = true_slope - learned_slope
        training_l2_squared = (
            slope_gap * slope_gap * (training_sd * training_sd + training_mean * training_mean)
            - 2.0 * slope_gap * learned_bias * training_mean
            + learned_bias * learned_bias
        )
        change_bound = math.sqrt(ratio_second_moment * training_l2_squared)
        change_violations.append(generated_one_step_error - change_bound)
        require(generated_one_step_error <= change_bound + 1e-12,
                "linear-Gaussian change-of-measure bound failed")

        current_w1 = gaussian_w1(true_mean, true_sd, learned_mean, learned_sd)
        next_true_mean = true_slope * true_mean
        next_learned_mean = learned_slope * learned_mean + learned_bias
        next_true_sd = math.sqrt((true_slope * true_sd) ** 2 + noise_sd * noise_sd)
        next_learned_sd = math.sqrt((learned_slope * learned_sd) ** 2 + noise_sd * noise_sd)
        next_w1 = gaussian_w1(
            next_true_mean, next_true_sd, next_learned_mean, next_learned_sd
        )
        one_step_bound = rho * current_w1 + generated_one_step_error
        recursion_violations.append(next_w1 - one_step_bound)
        require(next_w1 <= one_step_bound + 2e-12, "linear-Gaussian recursion failed")
        recursive_bound = rho * recursive_bound + generated_one_step_error

        quadrature = _gaussian_quantile_w1_quadrature(
            next_true_mean, next_true_sd, next_learned_mean, next_learned_sd
        )
        quadrature_errors.append(abs(quadrature - next_w1))
        # Gauss-Hermite integrates a kinked absolute value, so this is a numerical
        # consistency tolerance rather than a floating-point identity.
        require(quadrature_errors[-1] < 1.5e-2, "Gaussian W1 formula/quadrature mismatch")

        true_mean, learned_mean = next_true_mean, next_learned_mean
        true_sd, learned_sd = next_true_sd, next_learned_sd

    final_w1 = gaussian_w1(true_mean, true_sd, learned_mean, learned_sd)
    require(final_w1 <= recursive_bound + 2e-12, "iterated Gaussian bound failed")

    # Removing a contraction premise: a perfect but expansive kernel x -> 2x
    # amplifies a small initial distribution mismatch by 2^H.
    initial_gap = 1e-4
    unstable_growth_ratio = (2.0 ** horizons * initial_gap) / initial_gap
    require(unstable_growth_ratio > 100.0, "expansive-kernel counterexample is too weak")

    # Removing any finite uniform Lipschitz control: a threshold kernel turns
    # +/-epsilon contexts into outputs distance one apart.
    epsilon = 1e-6
    discontinuous_kernel_ratio = 1.0 / (2.0 * epsilon)
    require(discontinuous_kernel_ratio >= 500_000.0,
            "discontinuous-kernel counterexample is too weak")

    return LinearGaussianControl(
        horizons=horizons,
        maximum_recursion_violation=max(recursion_violations),
        maximum_change_of_measure_violation=max(change_violations),
        maximum_quantile_formula_error=max(quadrature_errors),
        final_exact_w1=final_w1,
        final_recursive_bound=recursive_bound,
        unstable_growth_ratio=unstable_growth_ratio,
        discontinuous_kernel_ratio=discontinuous_kernel_ratio,
    )


@dataclass(frozen=True)
class ObjectiveCounterexamples:
    endpoint_scheduled_risk: float
    endpoint_error_squared: float
    endpoint_no_atom_spike_risk: float
    endpoint_no_atom_error_squared: float
    endpoint_no_atom_error_to_risk_ratio: float
    adaptive_small_residual_value: float
    adaptive_large_residual_value: float
    adaptive_raw_to_weighted_ratio: float
    identical_law_w1: float
    independent_pair_cost: float
    common_noise_pair_cost: float


@dataclass(frozen=True)
class DifferentialIdentityControl:
    endpoint_error: float
    corrected_integral: float
    corrected_identity_error: float
    omitted_velocity_correction_error: float


def run_differential_identity_control() -> DifferentialIdentityControl:
    """Numerically check the exact identity (4.8), including its correction."""

    # Oracle characteristic: z_s=z_0+s, v*=u*=F*=1.  The learned fields are
    # smooth but deliberately use v_theta != v* so the correction is nonzero.
    z0 = 0.3
    roots, weights = np.polynomial.legendre.leggauss(128)
    times = 0.5 * (roots + 1.0)
    quadrature_weights = 0.5 * weights
    z = z0 + times
    learned_u = z * z + times ** 3 + 0.5
    learned_v = 1.0 + 0.2 * z - 0.1 * times
    learned_f = learned_u + times * (3.0 * times ** 2 + 2.0 * z * learned_v)
    correction = times * 2.0 * z * (learned_v - 1.0)
    corrected_integral = float(np.sum(quadrature_weights * (learned_f - 1.0 - correction)))
    uncorrected_integral = float(np.sum(quadrature_weights * (learned_f - 1.0)))
    endpoint_error = float((z0 + 1.0) ** 2 + 1.0 ** 3 + 0.5 - 1.0)
    identity_error = abs(corrected_integral - endpoint_error)
    omitted_error = abs(uncorrected_integral - endpoint_error)
    require(identity_error < 2e-13, "differential endpoint identity failed")
    require(omitted_error > 1e-2, "velocity-correction negative control is too weak")
    return DifferentialIdentityControl(
        endpoint_error=endpoint_error,
        corrected_integral=corrected_integral,
        corrected_identity_error=identity_error,
        omitted_velocity_correction_error=omitted_error,
    )


def run_objective_counterexamples() -> ObjectiveCounterexamples:
    """Check three failures that prevent overreading the current objective."""

    # A continuous field can vanish on all sampled r >= 1/4 while differing at r=0.
    magnitude = 7.0
    sampled_r = np.linspace(0.25, 1.0, 4097)
    field = magnitude * np.maximum(1.0 - 4.0 * sampled_r, 0.0)
    endpoint_scheduled_risk = float(np.mean(field * field))
    endpoint_error_squared = magnitude * magnitude
    require(endpoint_scheduled_risk == 0.0 and endpoint_error_squared == 49.0,
            "endpoint-support counterexample failed")
    # More directly, under a Uniform(0,1) schedule (full interior support but no
    # endpoint atom), u_n(r)=max(1-nr,0) has exact risk 1/(3n) and endpoint 1.
    spike_rate = 1_000_000.0
    no_atom_risk = 1.0 / (3.0 * spike_rate)
    no_atom_endpoint = 1.0
    no_atom_ratio = no_atom_endpoint / no_atom_risk
    require(no_atom_ratio >= 3_000_000.0, "no-atom trace counterexample is too weak")

    epsilon = 0.01
    small_raw = 1e-8
    large_raw = 1e12
    adaptive_small = small_raw / (small_raw + epsilon)
    adaptive_large = large_raw / (large_raw + epsilon)
    ratio = large_raw / adaptive_large
    require(adaptive_large < 1.000000000001 and ratio > 1e11,
            "adaptive-weight counterexample failed")

    # P=Q=Bernoulli(1/2): W1 is zero.  Independent target/sample pairing has
    # positive cost, whereas the same base-bit coupling is exact.
    identical_w1 = bernoulli_w1(0.5, 0.5)
    independent_cost = 0.5
    common_cost = 0.0
    require(identical_w1 == common_cost == 0.0 and independent_cost == 0.5,
            "independent-pairing counterexample failed")

    return ObjectiveCounterexamples(
        endpoint_scheduled_risk=endpoint_scheduled_risk,
        endpoint_error_squared=endpoint_error_squared,
        endpoint_no_atom_spike_risk=no_atom_risk,
        endpoint_no_atom_error_squared=no_atom_endpoint,
        endpoint_no_atom_error_to_risk_ratio=no_atom_ratio,
        adaptive_small_residual_value=adaptive_small,
        adaptive_large_residual_value=adaptive_large,
        adaptive_raw_to_weighted_ratio=ratio,
        identical_law_w1=identical_w1,
        independent_pair_cost=independent_cost,
        common_noise_pair_cost=common_cost,
    )


def run_numerics() -> dict[str, Any]:
    """Run every positive and negative mathematical control."""

    return {
        "differential_identity": asdict(run_differential_identity_control()),
        "finite_state": asdict(run_finite_state_control()),
        "linear_gaussian": asdict(run_linear_gaussian_control()),
        "objective_counterexamples": asdict(run_objective_counterexamples()),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        require(math.isfinite(value), "numeric control returned a non-finite value")
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theory", type=Path, default=DEFAULT_THEORY)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--numerics", action="store_true")
    parser.add_argument("--json", action="store_true", help="print checked evidence as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        document = verify_document(args.theory, args.protocol)
        evidence: dict[str, Any] = {"document": document}
        if args.numerics:
            evidence["numerics"] = run_numerics()
        evidence = _json_safe(evidence)
    except (OSError, ValueError, json.JSONDecodeError, VerificationError) as error:
        print(f"TRAJECTORY_IMF_THEORY_VERIFICATION_FAILED: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    print(NUMERIC_SUCCESS if args.numerics else SUCCESS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
