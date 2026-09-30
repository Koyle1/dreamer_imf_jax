"""Immutable registration, Slurm launch and verification for the Dreamer ablation.

Never imports the older compact learner or any pretrained study. Each arm runs
in a separate process because the extension patches the upstream RSSM class.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[2]
PROTOCOL = SOURCE / "dreamer_imf_comparison/dreamer_ablation_protocol.json"
MODULE = "dreamer_imf_compare.dreamer_ablation_study"
PACKAGES = (
    "jax",
    "jaxlib",
    "numpy",
    "ninjax",
    "elements",
    "portal",
    "granular",
    "scope",
    "optax",
    "chex",
    "dm_control",
    "mujoco",
    "einops",
    "ruamel.yaml",
)


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def filehash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def publish(path, value):
    """Exclusive publication: no valid or invalid retained artifact is replaced."""
    with Path(path).open("x") as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")


def clean_commit(path):
    head = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"], text=True
    )
    if status.strip():
        raise ValueError(f"Source is not clean: {path}")
    return head


def runtime():
    # Bind transitive numeric/CUDA dependencies too, not just direct imports.
    return dict(
        sorted(
            (d.metadata["Name"].lower().replace("_", "-"), d.version)
            for d in importlib.metadata.distributions()
        )
    )


def cells(p, stage):
    seeds = p["seeds"][:1] if stage == "preflight" else p["seeds"]
    if stage not in ("preflight", "training"):
        raise ValueError(stage)
    return [
        dict(index=i, arm=a, seed=s)
        for i, (a, s) in enumerate((a, s) for a in p["arms"] for s in seeds)
    ]


def register(root, upstream):
    root, upstream = Path(root).resolve(), Path(upstream).resolve()
    p = read(PROTOCOL)
    if clean_commit(upstream) != p["upstream_commit"]:
        raise ValueError("Upstream pin differs")
    p["upstream_path"] = str(upstream)
    value = dict(
        source_commit=clean_commit(SOURCE),
        source_root=str(SOURCE),
        protocol_sha256=digest(p),
        runtime=runtime(),
        python=sys.version,
        upstream_commit=p["upstream_commit"],
        pretrained_dependencies=[],
    )
    root.mkdir(parents=True, exist_ok=False)
    (root / "submissions").mkdir()
    (root / "verified").mkdir()
    publish(root / "protocol.json", p)
    publish(root / "manifest.json", value)
    print("DREAMER_ABLATION_REGISTERED", digest(value), flush=True)


def manifest(root):
    root = Path(root)
    m, p = read(root / "manifest.json"), read(root / "protocol.json")
    expected = read(PROTOCOL)
    if {k: v for k, v in p.items() if k != "upstream_path"} != expected:
        raise ValueError("Protocol differs from frozen source")
    if (
        m["source_commit"] != clean_commit(SOURCE)
        or m["source_root"] != str(SOURCE)
        or m["protocol_sha256"] != digest(p)
        or m["runtime"] != runtime()
        or m["pretrained_dependencies"] != []
        or clean_commit(p["upstream_path"]) != p["upstream_commit"]
    ):
        raise ValueError("Source/runtime/provenance mismatch")
    return m, p


def cell_dir(root, stage, c):
    return (
        Path(root)
        / ("preflight" if stage == "preflight" else "cells")
        / c["arm"]
        / f"seed_{c['seed']}"
    )


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite result")
    if isinstance(value, dict):
        for x in value.values():
            finite(x)
    if isinstance(value, list):
        for x in value:
            finite(x)


def verify_cell(root, stage, c):
    m, p = manifest(root)
    directory = cell_dir(root, stage, c)
    r = read(directory / "complete.json")
    finite(r)
    preflight = stage == "preflight"
    budget = p["preflight"]["native_steps"] if preflight else p["native_steps"]
    marks = (
        p["preflight"]["eval_at_native_steps"]
        if preflight
        else p["eval_at_native_steps"]
    )
    episodes = 1 if preflight else p["eval_episodes"]
    from .dreamer_ablation_runner import _settings

    if r.get("settings") != _settings(p, preflight):
        raise ValueError("Effective settings differ")
    if (
        r["arm"] != c["arm"]
        or r["seed"] != c["seed"]
        or r["preflight"] != preflight
        or not r["completed"]
        or not r["from_scratch"]
        or not r["finite_parameters"]
        or r["upstream_commit"] != p["upstream_commit"]
        or r["native_steps"] != budget
        or r["agent_transitions"] * 2 != budget
        or r["replay_rows"] != r["agent_transitions"] + r["reset_rows"]
        or r["learner_updates"] < 2
        or [e["native_steps"] for e in r["evaluations"]] != marks
        or len(r["checkpoints"]) != len(marks)
    ):
        raise ValueError("Cell budget or completion evidence differs")
    for e in r["evaluations"]:
        if (
            len(e["returns"]) != episodes
            or len(e["episode_seeds"]) != episodes
            or e["episode_native_steps"] != [p["episode_native_steps"]] * episodes
            or e["mean_return"] != sum(e["returns"]) / episodes
        ):
            raise ValueError("Held-out evaluation evidence differs")
    for checkpoint in r["checkpoints"]:
        path = directory / checkpoint["path"]
        if path.parent != directory or filehash(path) != checkpoint["sha256"]:
            raise ValueError("Checkpoint hash mismatch")
    value = dict(
        cell=c,
        stage=stage,
        manifest_sha256=digest(m),
        result_sha256=filehash(directory / "complete.json"),
    )
    marker = Path(root) / "verified" / f"{stage}-{c['index']:03d}.json"
    if marker.exists():
        if read(marker) != value:
            raise ValueError("Retained marker differs")
    else:
        publish(marker, value)
    print("DREAMER_ABLATION_CELL_VERIFIED", stage, c["index"], flush=True)
    return r


def verify_stage(root, stage):
    _, p = manifest(root)
    results = [verify_cell(root, stage, c) for c in cells(p, stage)]
    print(f"DREAMER_ABLATION_{stage.upper()}_VERIFIED", flush=True)
    return results


def finalize(root):
    """Aggregate paired seed means; episodes are nested, not independent seeds."""
    m, p = manifest(root)
    scheduler_complete(root, "training", 9)
    results = verify_stage(root, "training")
    scores = {
        a: [
            next(r for r in results if r["arm"] == a and r["seed"] == w)["evaluations"][
                -1
            ]["mean_return"]
            for w in p["seeds"]
        ]
        for a in p["arms"]
    }

    # Fractional 25%-trimmed mean, including partial mass at each boundary.
    def iqm(values):
        ordered = sorted(values)
        lo, hi = len(ordered) * 0.25, len(ordered) * 0.75
        return sum(
            v * max(0, min(i + 1, hi) - max(i, lo)) for i, v in enumerate(ordered)
        ) / (hi - lo)

    contrasts = {}
    for a, b in (
        ("imf", "gaussian"),
        ("gaussian", "categorical"),
        ("imf", "categorical"),
    ):
        deltas = [x - y for x, y in zip(scores[a], scores[b])]
        contrasts[f"{a}_minus_{b}"] = dict(
            paired_seed_deltas=deltas,
            mean_delta=sum(deltas) / len(deltas),
            favorable_fraction=sum(x > 0 for x in deltas) / len(deltas),
        )
    report = dict(
        manifest_sha256=digest(m),
        statistical_unit="three paired independent training seeds; five nested evaluation episodes each",
        seeds=p["seeds"],
        final_seed_mean_returns=scores,
        seed_return_iqms={a: iqm(v) for a, v in scores.items()},
        contrasts=contrasts,
        cells=results,
        successful_cell_wall_seconds=sum(r["wall_seconds"] for r in results),
        limitations=[
            "Single task and three seeds; exploratory, not confirmation.",
            "Released upstream reimplementation, not an exact reproduction of the paper's reported score.",
            "Continuous arms share posterior/reference KL but have different transition families, losses and parameter counts.",
            "No pretrained weights; action selection and imagined actor-critic are upstream Dreamer throughout.",
        ],
    )
    finite(report)
    path = Path(root) / "report.json"
    if path.exists():
        if read(path) != report:
            raise ValueError("Existing report differs")
    else:
        publish(path, report)
    print("DREAMER_ABLATION_FINAL_VERIFIED", filehash(path), flush=True)
    return report


def run(root, stage, index):
    _, p = manifest(root)
    c = cells(p, stage)[index]
    # Reject even partial output. Recovery must be explicitly reviewed.
    if cell_dir(root, stage, c).exists():
        raise FileExistsError("Cell already exists; no automatic rerun or overwrite")
    from .dreamer_ablation_runner import run_cell

    run_cell(root, c["arm"], c["seed"], preflight=stage == "preflight")
    verify_cell(root, stage, c)


def scheduler_complete(root, stage, count):
    receipt = read(Path(root) / "submissions" / f"{stage}.json")
    accounting = subprocess.check_output(
        [
            "sacct",
            "-n",
            "-X",
            "-P",
            "-j",
            receipt["job_id"],
            "--format=JobID,State,ExitCode",
        ],
        text=True,
    )
    rows = [x.split("|") for x in accounting.splitlines() if x.strip()]
    if len(rows) != count or any(x[1:3] != ["COMPLETED", "0:0"] for x in rows):
        raise ValueError(f"{stage} scheduler tasks are not all successfully terminal")
    return rows


def submit(root, stage, partition):
    m, p = manifest(root)
    if partition not in ("gpu-l40s", "gpu-a30", "clara"):
        raise ValueError("Unreviewed partition")
    if stage == "training":
        verify_stage(root, "preflight")
        scheduler_complete(root, "preflight", 3)
    receipt_path = Path(root) / "submissions" / f"{stage}.json"
    intent_path = Path(root) / "submissions" / f"{stage}-intent.json"
    if receipt_path.exists() or intent_path.exists():
        raise FileExistsError(
            "Submission already attempted; inspect Slurm, never blindly duplicate"
        )
    for c in cells(p, stage):
        if cell_dir(root, stage, c).exists():
            raise FileExistsError("Existing cell evidence")
    count = len(cells(p, stage))
    argv = [
        sys.executable,
        "-m",
        MODULE,
        "run",
        "--root",
        str(Path(root).resolve()),
        "--stage",
        stage,
    ]
    command = shlex.join(argv) + ' --index "$SLURM_ARRAY_TASK_ID"'
    pythonpath = f"{SOURCE}/dreamer_imf_comparison:{SOURCE}/imf_dreamer_jax/src"
    script = (
        "#!/bin/bash\nset -euo pipefail\nmodule purge\nmodule load Python/3.12.3-GCCcore-13.3.0\n"
        f"export PYTHONPATH={shlex.quote(pythonpath)}\n"
        "unset JAX_PLATFORMS JAX_PLATFORM_NAME XLA_FLAGS\n"
        "export OMP_NUM_THREADS=8\nexport MUJOCO_GL=egl\n"
        "export XLA_PYTHON_CLIENT_PREALLOCATE=false\n"
        f"{command}\n"
    )
    script_path = Path(root) / "submissions" / f"{stage}.sh"
    with script_path.open("x") as file:
        file.write(script)
    publish(
        intent_path,
        dict(
            stage=stage,
            cells=cells(p, stage),
            manifest_sha256=digest(m),
            script_sha256=filehash(script_path),
            partition=partition,
        ),
    )
    job = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--account=dep_inin_dat",
            f"--partition={partition}",
            "--gres=gpu:1",
            "--cpus-per-task=8",
            "--mem=64G",
            "--time=" + ("02:00:00" if stage == "preflight" else "2-00:00:00"),
            f"--array=0-{count-1}%3",
            f"--job-name=dreamer-3arm-{stage}",
            f"--output={root}/submissions/{stage}-%A_%a.log",
            str(script_path),
        ],
        text=True,
    ).strip()
    if not job.split(";")[0].isdigit():
        raise ValueError(f"Unexpected scheduler response: {job}")
    publish(
        receipt_path,
        dict(
            job_id=job.split(";")[0], stage=stage, intent_sha256=filehash(intent_path)
        ),
    )
    print("DREAMER_ABLATION_SUBMITTED", stage, job, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("register", "run", "verify", "submit", "finalize")
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--upstream")
    parser.add_argument("--stage", choices=("preflight", "training"))
    parser.add_argument("--index", type=int)
    parser.add_argument("--partition", default="gpu-l40s")
    a = parser.parse_args()
    if a.action == "register":
        register(a.root, a.upstream)
    elif a.action == "run":
        run(a.root, a.stage, a.index)
    elif a.action == "verify":
        verify_stage(a.root, a.stage)
    elif a.action == "finalize":
        finalize(a.root)
    else:
        submit(a.root, a.stage, a.partition)


if __name__ == "__main__":
    main()
