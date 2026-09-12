#!/usr/bin/env python3
"""Fail-closed verifier for the paired shortcut stability diagnostic summary."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
for path in (PROJECT, WORKSPACE / "imf_dreamer_jax" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dreamer_imf_compare.artifacts import read_json  # noqa: E402
from dreamer_imf_compare.matched_objective_benchmark import object_sha256  # noqa: E402
from dreamer_imf_compare.shortcut_stability_study import SUMMARY_SCHEMA  # noqa: E402


def verify(path: str | Path) -> dict:
    summary = read_json(path)
    body = dict(summary)
    claimed = body.pop("summary_sha256", None)
    if claimed != object_sha256(body):
        raise ValueError("shortcut stability summary digest mismatch")
    if summary.get("schema_version") != SUMMARY_SCHEMA or summary.get("status") != "complete":
        raise ValueError("shortcut stability summary is incomplete")
    if summary.get("evidence_class") != "engineering_diagnostic_not_claim_evidence":
        raise ValueError("shortcut stability evidence class is invalid")
    if summary.get("cells") != 48:
        raise ValueError("shortcut stability study did not complete all 48 cells")
    if summary.get("archived_baseline_failures", 0) < 4:
        raise ValueError("selected archived controls did not establish the stress condition")
    if (
        summary.get("production_variant_total") != 6
        or summary.get("production_variant_passed") != 6
        or summary.get("accepted") is not True
    ):
        raise ValueError("the fully repaired shortcut variant did not pass all six paired cells")
    variants = summary.get("variant_summary", [])
    if len(variants) != 8 or any(row.get("total") != 6 for row in variants):
        raise ValueError("shortcut stability factorial coverage is incomplete")
    if len(summary.get("minimal_passing_variants", [])) != 6:
        raise ValueError("shortcut stability minimal-factor analysis is incomplete")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    arguments = parser.parse_args()
    summary = verify(arguments.input)
    print(
        "SHORTCUT_STABILITY_STUDY_VERIFIED "
        f"cells={summary['cells']} repaired={summary['production_variant_passed']}/"
        f"{summary['production_variant_total']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
