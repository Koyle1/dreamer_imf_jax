#!/usr/bin/env python3
"""Print a deterministic content fingerprint for a non-empty JAX cache tree."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "dreamer_imf_compare" / "cache_fingerprint.py"
)
SPEC = importlib.util.spec_from_file_location(
    "trajectory_imf_cache_fingerprint", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load cache fingerprint implementation")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
cache_tree_sha256 = MODULE.cache_tree_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    arguments = parser.parse_args()
    print(cache_tree_sha256(arguments.cache_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
