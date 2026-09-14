#!/usr/bin/env python3
"""Fail closed if the vNext method contract is incompletely documented."""

from pathlib import Path


root = Path(__file__).resolve().parents[1]
path = root / "POLICY_CONSISTENCY_VNEXT.md"
text = path.read_text(encoding="utf-8")
required = (
    "## 1. Frozen-world actor repair",
    "sign-only PMPO",
    "reverse KL coefficient 0.3",
    "## 2. Shared fixed probe bank",
    "policy selects",
    "## 3. Short-horizon advantage consistency",
    "persistent running RMS",
    "## 4. Epistemic pessimism",
    "aleatoric",
    "## 5. Exposure-drift controls",
    "endpoint prediction",
    "exploratory",
    "cannot support a NeurIPS-level superiority claim",
)
missing = [item for item in required if item not in text]
if missing:
    raise SystemExit(f"vNext documentation is missing: {missing}")
print("POLICY_CONSISTENCY_VNEXT_DOCS_VERIFIED")

