#!/usr/bin/env python3
"""Verification entry point for the secondary hard visual benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
for path in (PROJECT, WORKSPACE / "imf_dreamer_jax" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dreamer_imf_compare.pixel_benchmark import (  # noqa: E402
    ARM_ORDER,
    _load_npz,
    file_sha256,
    object_sha256,
    load_protocol,
    replay_pixel_dataset,
    run_benchmark,
    source_manifest,
    split_dataset_archive,
    verify_aggregate,
    verify_artifact,
)


def smoke_output() -> Path:
    identity = object_sha256(source_manifest())[:16]
    return Path(f"/private/tmp/trajectory_imf_pixel_smoke_{identity}")


def verify_contract() -> None:
    protocol = load_protocol()
    if protocol["study_role"]["claim_eligible_for_primary_gate"] is not False:
        raise AssertionError("pixel protocol escaped its secondary claim boundary")
    hard = protocol["runtime_contract"]["hard"]
    if (
        hard["backend"] != "gpu"
        or hard["visible_device_count"] != 1
        or hard["device_kind_substring"] != "L40S"
        or protocol["provenance"][
            "environment_replay_from_retained_actions_and_seeds_required"
        ]
        is not True
    ):
        raise AssertionError("pixel hard runtime or DMC replay contract is not frozen")
    print("PIXEL_BENCHMARK_CONTRACT_VERIFIED")


def verify_tests() -> None:
    environment = dict(os.environ)
    python_path = [str(PROJECT), str(WORKSPACE / "imf_dreamer_jax" / "src")]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "dreamer_imf_comparison.tests.test_pixel_benchmark",
        ],
        cwd=WORKSPACE,
        env=environment,
        text=True,
        capture_output=True,
        timeout=240,
        check=False,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    print("PIXEL_BENCHMARK_TESTS_VERIFIED")


def _assert_tamper_rejected(root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="pixel-artifact-tamper-") as temporary:
        copy = Path(temporary).resolve() / "artifact"
        shutil.copytree(root, copy)
        path = copy / "predictions.npz"
        with path.open("r+b") as handle:
            handle.seek(-1, os.SEEK_END)
            value = handle.read(1)
            handle.seek(-1, os.SEEK_END)
            handle.write(bytes([value[0] ^ 1]))
        try:
            verify_artifact(copy, recompute_predictions=False, rederive_compute=False)
        except ValueError as error:
            if "digest mismatch" not in str(error):
                raise AssertionError("tamper failed for an unexpected reason") from error
        else:
            raise AssertionError("mutated pixel prediction artifact was accepted")

    with tempfile.TemporaryDirectory(prefix="pixel-artifact-extra-entry-") as temporary:
        copy = Path(temporary).resolve() / "artifact"
        shutil.copytree(root, copy)
        (copy / "unexpected").mkdir()
        try:
            verify_artifact(copy, recompute_predictions=False, rederive_compute=False)
        except ValueError as error:
            if "entry set is not exact" not in str(error):
                raise AssertionError("extra-entry control failed for an unexpected reason") from error
        else:
            raise AssertionError("pixel artifact with an unexpected directory was accepted")

    with tempfile.TemporaryDirectory(prefix="pixel-artifact-root-link-") as temporary:
        alias = Path(temporary).resolve() / "artifact-link"
        alias.symlink_to(root, target_is_directory=True)
        try:
            verify_artifact(alias, recompute_predictions=False, rederive_compute=False)
        except ValueError as error:
            if "symlink component" not in str(error):
                raise AssertionError("root-symlink control failed for an unexpected reason") from error
        else:
            raise AssertionError("symlinked pixel artifact root was accepted")

    with tempfile.TemporaryDirectory(prefix="pixel-artifact-result-schema-") as temporary:
        copy = Path(temporary).resolve() / "artifact"
        shutil.copytree(root, copy)
        result_path = copy / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["unregistered_field"] = True
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        manifest_path = copy / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest["artifact_files"]:
            if record["path"] == "result.json":
                record["bytes"] = result_path.stat().st_size
                record["sha256"] = file_sha256(result_path)
        body = {key: value for key, value in manifest.items() if key != "body_sha256"}
        manifest["body_sha256"] = object_sha256(body)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        seal_path = copy / "seal.json"
        seal = {
            "schema_version": "trajectory-imf-pixel-seal-v1",
            "manifest_sha256": file_sha256(manifest_path),
            "manifest_body_sha256": manifest["body_sha256"],
        }
        seal_path.write_text(
            json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        try:
            verify_artifact(copy, recompute_predictions=False, rederive_compute=False)
        except ValueError as error:
            if "pixel result schema is not exact" not in str(error):
                raise AssertionError("result-schema control failed for an unexpected reason") from error
        else:
            raise AssertionError("self-resealed result with an extra field was accepted")


def _assert_render_replay_tamper_rejected(root: Path) -> None:
    protocol = load_protocol()
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    train, _ = split_dataset_archive(_load_npz(root / "dataset.npz"))
    corrupted = {name: value.copy() for name, value in train.items()}
    corrupted["observations"][0, 0, 0, 0] ^= np.uint8(1)
    try:
        replay_pixel_dataset(
            protocol,
            profile=str(result["profile"]),
            task=str(result["task"]),
            environment_seed=int(result["environment_seeds"]["training"]),
            arrays=corrupted,
        )
    except ValueError as error:
        if "rendered pixels differ" not in str(error):
            raise AssertionError("render-replay control failed for an unexpected reason") from error
    else:
        raise AssertionError("mutated rendered pixel was accepted by DMC replay")


def verify_smoke() -> None:
    root = smoke_output()
    protocol = load_protocol()
    profile = protocol["profiles"]["smoke"]
    if root.exists():
        result = verify_artifact(
            root, recompute_predictions=True, rederive_compute=True
        )
    else:
        result = run_benchmark(
            root,
            profile="smoke",
            task=profile["tasks"][0],
            seed=int(profile["seeds"][0]),
            track="equal_updates",
        )
        # The initial run already replays predictions; independently recompile
        # the stored cost evidence before certifying the smoke.
        result = verify_artifact(
            root, recompute_predictions=True, rederive_compute=True
        )
    if result["claim_eligible_for_primary_gate"] is not False:
        raise AssertionError("smoke was mislabeled as claim-bearing")
    if set(result["training"]) != set(ARM_ORDER):
        raise AssertionError("smoke did not train both objective arms")
    if set(result["metrics"]["frontier"]) != set(ARM_ORDER):
        raise AssertionError("smoke did not evaluate both objective arms")
    for arm in ARM_ORDER:
        if set(result["metrics"]["frontier"][arm]) != {"1", "2", "4"}:
            raise AssertionError("smoke omitted a registered NFE")
    _assert_tamper_rejected(root)
    _assert_render_replay_tamper_rejected(root)
    print(
        json.dumps(
            {
                "output": str(root),
                "task": result["task"],
                "seed": result["seed"],
                "claim_eligible_for_primary_gate": False,
                "primary_nfe_auc": {
                    arm: result["metrics"]["primary_nfe"][arm][
                        "normalized_visual_mse_auc"
                    ]
                    for arm in ARM_ORDER
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("PIXEL_BENCHMARK_SMOKE_VERIFIED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--contract", action="store_true")
    group.add_argument("--tests", action="store_true")
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--artifacts")
    group.add_argument("--aggregate")
    parser.add_argument("--inputs", nargs="+")
    parser.add_argument("--authentication-only", action="store_true")
    parser.add_argument("--rederive-compute", action="store_true")
    arguments = parser.parse_args()
    if arguments.contract:
        verify_contract()
    elif arguments.tests:
        verify_tests()
    elif arguments.smoke:
        verify_smoke()
    elif arguments.aggregate:
        summary = verify_aggregate(Path(arguments.aggregate), arguments.inputs)
        print(json.dumps(summary["tracks"], indent=2, sort_keys=True))
        print("PIXEL_BENCHMARK_AGGREGATE_VERIFIED")
    else:
        result = verify_artifact(
            Path(arguments.artifacts),
            recompute_predictions=not arguments.authentication_only,
            rederive_compute=arguments.rederive_compute,
        )
        print(json.dumps(result["metrics"]["primary_nfe"], indent=2, sort_keys=True))
        print("PIXEL_BENCHMARK_ARTIFACT_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
