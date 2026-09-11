#!/usr/bin/env python3
"""Deterministically verify the frozen trajectory-iMF novelty audit.

This verifier intentionally makes no network requests.  It checks the structure,
cross-references, dates, direct primary-source URLs, and bounded claim language of
the checked-in audit.  Remote availability and semantic correctness remain part
of the documented manual literature-review pass.
"""

from __future__ import annotations

import argparse
import calendar
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import re
import sys
from typing import Any
from urllib.parse import urlparse


SUCCESS = "TRAJECTORY_IMF_NOVELTY_AUDIT_VERIFIED"
CUTOFF = date(2026, 9, 10)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT = REPO_ROOT / "dreamer_imf_comparison" / "TRAJECTORY_IMF_NOVELTY_AUDIT.md"
DEFAULT_SOURCES = REPO_ROOT / "dreamer_imf_comparison" / "trajectory_imf_novelty_sources.json"

REQUIRED_TOP_LEVEL = {
    "schema_version",
    "audit_as_of",
    "checked_on",
    "scope",
    "search_boundary",
    "required_coverage",
    "sources",
    "claim_elements",
    "verdict",
}

REQUIRED_COVERAGE = {
    "flow_matching",
    "meanflow",
    "improved_meanflow",
    "shortcut_models",
    "diffusion_forcing",
    "dreamer4",
    "per_token_flow",
    "multi_time_flow",
    "causal_sequence_flow",
    "fixed_context_causal_jvp",
    "flow_world_models",
    "meanflow_world_models",
    "improved_meanflow_world_models",
    "meanflow_trajectory_control",
    "meanflow_shortcut_relation",
}

EXPECTED_PRIMARY_SOURCES = {
    "S01": ("Flow Matching for Generative Modeling", "https://arxiv.org/abs/2210.02747"),
    "S02": ("Mean Flows for One-step Generative Modeling", "https://arxiv.org/abs/2505.13447"),
    "S03": (
        "Improved Mean Flows: On the Challenges of Fastforward Generative Models",
        "https://arxiv.org/abs/2512.02012",
    ),
    "S04": ("One Step Diffusion via Shortcut Models", "https://arxiv.org/abs/2410.12557"),
    "S05": (
        "Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion",
        "https://arxiv.org/abs/2407.01392",
    ),
    "S06": ("Training Agents Inside of Scalable World Models", "https://arxiv.org/abs/2509.24527"),
    "S13": (
        "StreamFlow: Streaming Audio Generation from Discrete Tokens via Streaming Flow Matching",
        "https://proceedings.neurips.cc/paper_files/paper/2025/hash/0713495297dab18fabca4795cb9ef8ef-Abstract-Conference.html",
    ),
    "S14": (
        "Inference Time Policy Optimization for Offline RL with Differentiable World Models",
        "https://arxiv.org/abs/2603.22430",
    ),
    "S15": (
        "Flow Matching in Feature Space for Stochastic World Modeling",
        "https://arxiv.org/abs/2606.29059",
    ),
    "S16": (
        "Flow-JEPA: Flow Matching for Robust Latent Dynamics in JEPA World Models",
        "https://arxiv.org/abs/2608.29029",
    ),
    "S17": (
        "MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation",
        "https://ojs.aaai.org/index.php/AAAI/article/view/38919",
    ),
    "S18": (
        "Causal-rCM: A Unified Teacher-Forcing and Self-Forcing Open Recipe for Autoregressive Diffusion Distillation in Streaming Video Generation and Interactive World Models",
        "https://arxiv.org/abs/2606.25473",
    ),
    "S19": ("AlphaFlow: Understanding and Improving MeanFlow Models", "https://arxiv.org/abs/2510.20771"),
    "S20": (
        "SplitMeanFlow: Interval Splitting Consistency in Few-Step Generative Modeling",
        "https://arxiv.org/abs/2507.16884",
    ),
    "S21": (
        "Autoregressive One-Step Generative Modeling for Dynamical System Forecasting",
        "https://arxiv.org/abs/2605.05540",
    ),
}

ALLOWED_STATUSES = {
    "established_prior_art",
    "partly_prior_art",
    "exact_combination_not_found",
    "unresolved_empirically",
    "missing_theory",
}

PRIMARY_PAPER_HOSTS = {
    "arxiv.org",
    "proceedings.neurips.cc",
    "ojs.aaai.org",
    "openreview.net",
}

REQUIRED_HEADINGS = [
    "# Trajectory iMF novelty audit",
    "## Bounded verdict",
    "## Method under audit",
    "## Search boundary and evidence policy",
    "## Claim-element audit",
    "## Closest prior work",
    "## Likely reviewer objections",
    "## Falsifiable stronger-theory targets",
    "## Safe publication wording",
    "## Primary-source manifest",
    "## Verification",
]

REQUIRED_PHRASES = [
    "bounded combination novelty only",
    "no exact combination found",
    "not a priority claim",
    "does not establish empirical superiority",
    "aleatoric randomness does not provide epistemic uncertainty",
    "not a reproduction of dreamer 4",
    "fixed-context causal jvp itself is established prior art",
    "decoupled from history exposure times",
    "same per-token gaussian draw",
    "alphaflow",
    "splitmeanflow",
    "infinitesimal",
    "melisa",
]


class VerificationError(RuntimeError):
    """A concise expected verification failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def parse_date_upper(value: Any, field: str) -> date:
    """Parse YYYY, YYYY-MM, or YYYY-MM-DD as the interval's upper date."""

    require(isinstance(value, str), f"{field} must be a date string")
    if re.fullmatch(r"\d{4}", value):
        return date(int(value), 12, 31)
    if re.fullmatch(r"\d{4}-\d{2}", value):
        year, month = (int(part) for part in value.split("-"))
        return date(year, month, calendar.monthrange(year, month)[1])
    require(bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)), f"{field} has invalid date format: {value!r}")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise VerificationError(f"{field} is not a calendar date: {value!r}") from error


def parse_date_lower(value: Any, field: str) -> date:
    """Parse YYYY, YYYY-MM, or YYYY-MM-DD as the interval's lower date."""

    require(isinstance(value, str), f"{field} must be a date string")
    if re.fullmatch(r"\d{4}", value):
        return date(int(value), 1, 1)
    if re.fullmatch(r"\d{4}-\d{2}", value):
        year, month = (int(part) for part in value.split("-"))
        return date(year, month, 1)
    require(bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)), f"{field} has invalid date format: {value!r}")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise VerificationError(f"{field} is not a calendar date: {value!r}") from error


def verify_https_url(value: Any, field: str, *, paper: bool = False) -> None:
    require(isinstance(value, str) and value, f"{field} must be a nonempty URL")
    parsed = urlparse(value)
    require(parsed.scheme == "https", f"{field} must use https: {value!r}")
    require(bool(parsed.netloc) and bool(parsed.path), f"{field} must be a direct URL with a path: {value!r}")
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if paper:
        require(host in PRIMARY_PAPER_HOSTS, f"{field} is not on an allowed primary-paper host: {host}")
        require("/search" not in parsed.path.lower(), f"{field} must not be a search-results URL")
        if host == "arxiv.org":
            require(parsed.path.startswith("/abs/"), f"{field} must use the stable arXiv abstract URL")


def verify_string_list(value: Any, field: str, *, minimum: int = 1) -> list[str]:
    require(isinstance(value, list) and len(value) >= minimum, f"{field} must contain at least {minimum} item(s)")
    require(all(isinstance(item, str) and item.strip() for item in value), f"{field} must contain nonempty strings")
    return value


def verify_manifest(data: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(isinstance(data, dict), "source manifest must be a JSON object")
    missing = REQUIRED_TOP_LEVEL - set(data)
    require(not missing, f"source manifest is missing top-level keys: {sorted(missing)}")
    require(data["schema_version"] == 1, "unsupported schema_version")
    require(parse_date_lower(data["audit_as_of"], "audit_as_of") == CUTOFF, "audit_as_of must equal 2026-09-10")
    require(parse_date_lower(data["checked_on"], "checked_on") == CUTOFF, "checked_on must equal 2026-09-10")

    scope = data["scope"]
    require(isinstance(scope, dict), "scope must be an object")
    require(scope.get("proposed_method_label") == "fixed-context trajectory iMF", "scope must name fixed-context trajectory iMF")
    verify_string_list(scope.get("method_definition"), "scope.method_definition", minimum=6)
    verify_string_list(scope.get("excluded_claims"), "scope.excluded_claims", minimum=4)

    boundary = data["search_boundary"]
    require(isinstance(boundary, dict), "search_boundary must be an object")
    require(parse_date_lower(boundary.get("cutoff"), "search_boundary.cutoff") == CUTOFF, "search cutoff must equal audit cutoff")
    require("Primary papers" in boundary.get("evidence_policy", ""), "evidence policy must require primary papers")
    verify_string_list(boundary.get("families_checked"), "search_boundary.families_checked", minimum=6)
    verify_string_list(boundary.get("limitations"), "search_boundary.limitations", minimum=3)

    sources = data["sources"]
    require(isinstance(sources, list) and len(sources) >= 21, "manifest must contain at least 21 primary sources")
    ids: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(sources):
        field = f"sources[{index}]"
        require(isinstance(source, dict), f"{field} must be an object")
        source_id = source.get("id")
        require(isinstance(source_id, str) and re.fullmatch(r"S\d{2}", source_id) is not None, f"{field}.id must match SNN")
        require(source_id not in by_id, f"duplicate source id: {source_id}")
        ids.append(source_id)
        by_id[source_id] = source
        require(isinstance(source.get("title"), str) and len(source["title"]) >= 8, f"{source_id} needs a title")
        verify_string_list(source.get("authors"), f"{source_id}.authors")
        published_lower = parse_date_lower(source.get("publication_date"), f"{source_id}.publication_date")
        published_upper = parse_date_upper(source.get("publication_date"), f"{source_id}.publication_date")
        require(published_upper <= CUTOFF, f"{source_id} postdates the audit cutoff")
        revised = source.get("latest_revision_date")
        if revised is not None:
            revised_lower = parse_date_lower(revised, f"{source_id}.latest_revision_date")
            require(revised_lower >= published_lower, f"{source_id} revision predates publication")
            require(parse_date_upper(revised, f"{source_id}.latest_revision_date") <= CUTOFF, f"{source_id} revision postdates cutoff")
        require(isinstance(source.get("venue_or_status"), str) and source["venue_or_status"], f"{source_id} needs venue_or_status")
        verify_https_url(source.get("paper_url"), f"{source_id}.paper_url", paper=True)
        for optional_url in ("code_url", "project_url"):
            if source.get(optional_url) is not None:
                verify_https_url(source[optional_url], f"{source_id}.{optional_url}")
        verify_string_list(source.get("topics"), f"{source_id}.topics")
        require(isinstance(source.get("relevance"), str) and len(source["relevance"]) >= 40, f"{source_id} needs a substantive relevance statement")

    expected_ids = [f"S{number:02d}" for number in range(1, 22)]
    require(ids == expected_ids, "sources must be uniquely ordered and contiguous from S01 through S21")
    for source_id, (title, url) in EXPECTED_PRIMARY_SOURCES.items():
        require(by_id[source_id]["title"] == title, f"{source_id} title changed from the audited primary source")
        require(by_id[source_id]["paper_url"] == url, f"{source_id} URL changed from the audited primary source")
    melisa_relevance = by_id["S21"]["relevance"].lower()
    require("not the identical imf objective" in melisa_relevance, "S21 must distinguish MeLISA from the exact iMF objective")
    require("pathwise epsilon-x tangent" in melisa_relevance, "S21 must record MeLISA's displayed pathwise JVP tangent")
    require("predicted marginal-velocity tangent" in melisa_relevance, "S21 must record iMF's distinct predicted tangent")

    coverage = data["required_coverage"]
    require(isinstance(coverage, dict), "required_coverage must be an object")
    missing_coverage = REQUIRED_COVERAGE - set(coverage)
    require(not missing_coverage, f"missing required coverage families: {sorted(missing_coverage)}")
    for family in REQUIRED_COVERAGE:
        references = verify_string_list(coverage[family], f"required_coverage.{family}")
        unknown = set(references) - set(by_id)
        require(not unknown, f"coverage family {family} references unknown sources: {sorted(unknown)}")

    claims = data["claim_elements"]
    require(isinstance(claims, list) and len(claims) >= 19, "claim_elements must contain at least 19 entries")
    claim_ids: list[str] = []
    for index, claim in enumerate(claims):
        field = f"claim_elements[{index}]"
        require(isinstance(claim, dict), f"{field} must be an object")
        claim_id = claim.get("id")
        require(isinstance(claim_id, str) and re.fullmatch(r"E\d{2}", claim_id) is not None, f"{field}.id must match ENN")
        require(claim_id not in claim_ids, f"duplicate claim id: {claim_id}")
        claim_ids.append(claim_id)
        require(claim.get("status") in ALLOWED_STATUSES, f"{claim_id} has unsupported status")
        require(isinstance(claim.get("element"), str) and len(claim["element"]) >= 20, f"{claim_id} needs a precise element")
        references = verify_string_list(claim.get("prior_source_ids"), f"{claim_id}.prior_source_ids")
        unknown = set(references) - set(by_id)
        require(not unknown, f"{claim_id} references unknown sources: {sorted(unknown)}")
        require(isinstance(claim.get("audit_finding"), str) and len(claim["audit_finding"]) >= 30, f"{claim_id} needs an audit finding")
        require(isinstance(claim.get("residual_claim"), str) and claim["residual_claim"], f"{claim_id} needs a residual claim")
    require(claim_ids == [f"E{number:02d}" for number in range(1, 20)], "claim elements must be ordered E01 through E19")
    by_claim_id = {claim["id"]: claim for claim in claims}
    status_by_id = {claim_id: claim["status"] for claim_id, claim in by_claim_id.items()}
    require(status_by_id["E06"] == "established_prior_art", "fixed-context causal JVP must be marked prior art")
    require(status_by_id["E12"] == "exact_combination_not_found", "full combination must use bounded not-found status")
    require(status_by_id["E13"] == "unresolved_empirically", "superiority must remain unresolved")
    require(status_by_id["E14"] == status_by_id["E15"] == "missing_theory", "theory and uncertainty gaps must stay explicit")
    require(status_by_id["E17"] == status_by_id["E18"] == "established_prior_art", "MeanFlow/shortcut theory relationships must be prior art")
    require(status_by_id["E19"] == "established_prior_art", "MeanFlow/iMF-style one-step autoregressive dynamics must be prior art")
    e14_finding = by_claim_id["E14"]["audit_finding"].lower()
    require("normalized joint rollout law" in e14_finding, "E14 must acknowledge autoregressive normalization")
    require("no tractable likelihood" in e14_finding, "E14 must state the precise missing likelihood result")

    verdict = data["verdict"]
    require(isinstance(verdict, dict), "verdict must be an object")
    require(verdict.get("code") == "bounded_combination_novelty_only", "verdict must stay bounded_combination_novelty_only")
    summary = verdict.get("summary", "")
    for phrase in ("No exact combination was found", "Causal-rCM", "MeLISA", "AlphaFlow", "SplitMeanFlow", "not a priority claim"):
        require(phrase in summary, f"verdict summary must contain {phrase!r}")
    verify_string_list(verdict.get("allowed_wording"), "verdict.allowed_wording", minimum=3)
    prohibited = verify_string_list(verdict.get("prohibited_wording"), "verdict.prohibited_wording", minimum=9)
    required_prohibitions = {
        "the first MeanFlow world model",
        "the first fixed-context causal JVP for MeanFlow",
        "a new unification of trajectory flow matching, Shortcut Models, and MeanFlow",
        "the first improved-MeanFlow autoregressive dynamical world model",
        "outperforms Dreamer 4",
    }
    require(required_prohibitions <= set(prohibited), "verdict is missing required prohibited claims")
    verify_string_list(verdict.get("evidence_required_before_strengthening"), "verdict.evidence_required_before_strengthening", minimum=3)
    return sources, claims


def verify_audit(text: str, sources: list[dict[str, Any]], claims: list[dict[str, Any]]) -> None:
    require(len(text) >= 12_000, "audit is too short to contain the required publication-facing analysis")
    for heading in REQUIRED_HEADINGS:
        require(heading in text, f"audit is missing heading: {heading}")
    lowered = text.lower()
    for phrase in REQUIRED_PHRASES:
        require(phrase in lowered, f"audit is missing required bounded language: {phrase!r}")

    for source in sources:
        source_id = source["id"]
        require(f"[{source_id}]" in text, f"audit never cites {source_id}")
        require(f"| {source_id} |" in text, f"source manifest table is missing {source_id}")
        require(source["paper_url"] in text, f"audit is missing direct primary URL for {source_id}")
    for claim in claims:
        require(f"| {claim['id']} |" in text, f"claim-element table is missing {claim['id']}")

    objection_ids = set(re.findall(r"\| (O\d{2}) \|", text))
    require({f"O{number:02d}" for number in range(1, 18)} <= objection_ids, "audit must include reviewer objections O01 through O17")
    target_ids = set(re.findall(r"\| (T\d{2}) \|", text))
    require({f"T{number:02d}" for number in range(1, 11)} <= target_ids, "audit must include falsifiable targets T01 through T10")

    required_manual_details = [
        "Causal-rCM Equation (17)",
        "zero tangent",
        "separately sampled history times",
        "shared-noise two-view",
        "all-subsequence",
        "teacher-corrupted suffix",
        "normalized joint rollout law",
        "pathwise `epsilon - x` tangent",
        "predicted marginal velocity",
        "tractable likelihood",
        "policy-exploitation",
        "equal-compiler-FLOP",
    ]
    for detail in required_manual_details:
        require(detail.lower() in lowered, f"audit is missing required expert detail: {detail!r}")
    forbidden_semantic_claims = [
        "does not by construction provide a normalized joint density",
        "they still do not define a normalized joint density",
        "need not correspond to one compatible joint trajectory law",
        "melisa already applies imf to a 1-nfe",
        "self-generated suffix supervision",
    ]
    for claim in forbidden_semantic_claims:
        require(claim not in lowered, f"audit retains a rejected semantic claim: {claim!r}")


def expect_verification_failure(callback: Any, label: str) -> None:
    try:
        callback()
    except VerificationError:
        return
    raise VerificationError(f"negative control unexpectedly passed: {label}")


def run_negative_controls(data: dict[str, Any], text: str) -> None:
    wrong_status = deepcopy(data)
    wrong_status["claim_elements"][5]["status"] = "partly_prior_art"
    expect_verification_failure(
        lambda: verify_manifest(wrong_status),
        "fixed-context JVP downgraded from established prior art",
    )

    post_cutoff = deepcopy(data)
    post_cutoff["sources"][-1]["latest_revision_date"] = "2026-09-11"
    expect_verification_failure(
        lambda: verify_manifest(post_cutoff),
        "source revision after the frozen cutoff",
    )

    wrong_primary_url = deepcopy(data)
    wrong_primary_url["sources"][17]["paper_url"] = "https://arxiv.org/search/?query=Causal-rCM"
    expect_verification_failure(
        lambda: verify_manifest(wrong_primary_url),
        "search-results URL substituted for Causal-rCM primary source",
    )

    sources, claims = verify_manifest(data)
    unbounded_text = text.replace("bounded combination novelty only", "novel method", 1)
    expect_verification_failure(
        lambda: verify_audit(unbounded_text, sources, claims),
        "bounded verdict removed from the audit",
    )

    identical_melisa = deepcopy(data)
    identical_melisa["sources"][-1]["relevance"] = (
        "MeLISA uses the identical iMF objective with the same predicted marginal-velocity tangent "
        "for a one-evaluation autoregressive dynamical forecast."
    )
    expect_verification_failure(
        lambda: verify_manifest(identical_melisa),
        "MeLISA incorrectly described as the identical iMF objective",
    )

    denied_normalization = text.replace(
        "measurable and normalized, their autoregressive product defines a normalized joint rollout law",
        "measurable and normalized, they still do not define a normalized joint density",
        1,
    )
    expect_verification_failure(
        lambda: verify_audit(denied_normalization, sources, claims),
        "autoregressive normalization incorrectly denied",
    )

    mislabeled_suffix = text.replace("teacher-corrupted suffix", "self-generated suffix supervision")
    expect_verification_failure(
        lambda: verify_audit(mislabeled_suffix, sources, claims),
        "teacher-corrupted suffix mislabeled as self-generated",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT, help="Markdown audit path")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES, help="JSON source-manifest path")
    parser.add_argument("--self-test", action="store_true", help="also run in-memory negative controls")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(args.sources.is_file(), f"source manifest does not exist: {args.sources}")
        require(args.audit.is_file(), f"audit does not exist: {args.audit}")
        try:
            data = json.loads(args.sources.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise VerificationError(f"source manifest is invalid JSON: {error}") from error
        sources, claims = verify_manifest(data)
        audit_text = args.audit.read_text(encoding="utf-8")
        verify_audit(audit_text, sources, claims)
        if args.self_test:
            run_negative_controls(data, audit_text)
    except (OSError, VerificationError) as error:
        print(f"TRAJECTORY_IMF_NOVELTY_AUDIT_FAILED: {error}", file=sys.stderr)
        return 1
    if args.self_test:
        print("TRAJECTORY_IMF_NOVELTY_NEGATIVE_CONTROLS_VERIFIED")
    print(SUCCESS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
