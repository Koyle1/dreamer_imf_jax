"""Read-only, source-bound acceptance oracles for the diagnostic ledger."""

import argparse
from pathlib import Path

from dreamer_imf_compare.parallel_study import authenticate, require, read, sha
from submit_parallel_study import terminal


def inspect(root, level):
    root = Path(root)
    manifest, _ = authenticate(root)
    if not manifest.get("deployment_bundle"):
        raise ValueError("missing exact deployment bundle")
    stages = []
    if level in ("preflight", "evaluated", "final"):
        stages.append("preflight")
    if level in ("evaluated", "final"):
        stages.extend(("collect", "fit", "evaluate"))
    if level == "final":
        stages.append("verify")
    for stage in stages:
        receipt = read(root / "submissions" / f"{stage}.json")
        terminal(receipt["job"], 3 if stage == "fit" else 1)
        for name in ([f"fit-{s}" for s in range(3)] if stage == "fit" else [stage]):
            require(root, name)
    if level == "final":
        completion = read(root / "completion.json")
        if (
            completion.get("completed") is not True
            or completion["source_commit"] != manifest["source_commit"]
            or completion["manifest_sha256"] != sha(root / "manifest.json")
            or completion["report_sha256"] != sha(root / "report.json")
        ):
            raise ValueError("terminal completion authentication differs")
    print("PARALLEL_INSPECTION_VERIFIED", level, manifest["source_commit"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--level",
        choices=("deployed", "preflight", "evaluated", "final"),
        required=True,
    )
    args = parser.parse_args()
    inspect(args.root, args.level)
