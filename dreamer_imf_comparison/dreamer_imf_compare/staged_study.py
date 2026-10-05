"""Exclusive six-cell scratch experiment with matched staged Gaussian control."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from . import dreamer_ablation_study as evidence

evidence.PROTOCOL = evidence.SOURCE / "dreamer_imf_comparison/staged_protocol.json"
evidence.MODULE = "dreamer_imf_compare.staged_study"
_original_verify = evidence.verify_cell


def verify_cell(root, stage, cell):
    directory = evidence.cell_dir(root, stage, cell)
    result = evidence.read(directory / "complete.json")
    staged = result["staged"]
    if evidence.filehash(directory / "diagnostic_batch.npz") != staged["batch_sha256"]:
        raise ValueError("Fixed diagnostic batch digest mismatch")
    if (
        staged["protocol"] != "staged-scratch-v1"
        or staged["native"] != result["native_steps"]
        or staged["updates"] != result["learner_updates"]
        or not staged["diagnostics"]
        or not staged["freeze_checks"]
        or not all(c["passed"] for c in staged["freeze_checks"])
    ):
        raise ValueError("Missing staged execution evidence")
    for record in staged["diagnostics"]:
        path = directory / record["path"]
        if path.parent != directory or evidence.filehash(path) != record["sha256"]:
            raise ValueError("Diagnostic digest mismatch")
        diagnostic = evidence.read(path)
        evidence.finite(diagnostic)
        if not diagnostic["learner_unchanged"]:
            raise ValueError("Diagnostic modified learner")
    if stage == "preflight":
        modes = {r["values"]["imag_horizon"] for r in staged["controls"]}
        if modes != {5, 15} or result["learner_updates"] != 8:
            raise ValueError("Preflight did not exercise both reliance phases")
    else:
        if not any(r["values"]["transition_only"] for r in staged["controls"]):
            raise ValueError("No representation-frozen phase")
        if not any(not r["values"]["actor_enabled"] for r in staged["controls"]):
            raise ValueError("No scratch model warmup")
    return _original_verify(root, stage, cell)


evidence.verify_cell = verify_cell


def run(root, stage, index):
    _, protocol = evidence.manifest(root)
    cell = evidence.cells(protocol, stage)[index]
    if evidence.cell_dir(root, stage, cell).exists():
        raise FileExistsError(
            "Existing evidence; never overwrite or automatically retry"
        )
    from .staged_runner import run_cell

    run_cell(root, cell["arm"], cell["seed"], stage == "preflight")
    verify_cell(root, stage, cell)
    print("STAGED_CELL_VERIFIED", stage, index, flush=True)


def submit(root, stage, partition):
    root = Path(root).resolve()
    manifest, protocol = evidence.manifest(root)
    conditional = protocol.get("schema") == "imf-conditional-one-seed-v1"
    if partition not in ("gpu-l40s", "gpu-a30", "clara"):
        raise ValueError("Unreviewed partition")
    if stage == "training":
        evidence.scheduler_complete(
            root, "preflight", len(evidence.cells(protocol, "preflight"))
        )
        evidence.verify_stage(root, "preflight")
    intent = root / "submissions" / f"{stage}-intent.json"
    receipt = root / "submissions" / f"{stage}.json"
    if intent.exists() or receipt.exists():
        raise FileExistsError(
            "Already attempted; inspect scheduler before any recovery"
        )
    cells = evidence.cells(protocol, stage)
    if any(evidence.cell_dir(root, stage, c).exists() for c in cells):
        raise FileExistsError("Existing cell evidence")
    command = shlex.join(
        [
            sys.executable,
            "-m",
            evidence.MODULE,
            "run",
            "--root",
            str(root),
            "--stage",
            stage,
        ]
    )
    paths = f"{evidence.SOURCE}/dreamer_imf_comparison:{evidence.SOURCE}/imf_dreamer_jax/src"
    script = root / "submissions" / f"{stage}.sh"
    with script.open("x") as out:
        out.write(
            "#!/bin/bash\nset -euo pipefail\nmodule purge\nmodule load Python/3.12.3-GCCcore-13.3.0\n"
        )
        out.write(f"export PYTHONPATH={shlex.quote(paths)}\n")
        out.write(
            "unset JAX_PLATFORMS JAX_PLATFORM_NAME XLA_FLAGS\nexport OMP_NUM_THREADS=8\nexport MUJOCO_GL=egl\nexport XLA_PYTHON_CLIENT_PREALLOCATE=false\n"
        )
        if conditional:
            out.write("export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8\n")
            probe = "import faulthandler; faulthandler.enable(); faulthandler.dump_traceback_later(120); print('JAX_STARTUP_BEGIN', flush=True); import jax; d=jax.devices(); assert any(x.platform=='gpu' for x in d), d; print('JAX_GPU_READY', d, flush=True)"
            out.write(
                shlex.join(["timeout", "300s", sys.executable, "-u", "-c", probe])
                + "\n"
            )
        out.write(command + ' --index "$SLURM_ARRAY_TASK_ID"\n')
    evidence.publish(
        intent,
        dict(
            stage=stage,
            cells=cells,
            manifest_sha256=evidence.digest(manifest),
            script_sha256=evidence.filehash(script),
            partition=partition,
        ),
    )
    job = (
        subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--account=dep_inin_dat",
                f"--partition={partition}",
                "--gres=gpu:1",
                "--cpus-per-task=8",
                "--mem=64G",
                "--time="
                + (
                    "02:00:00"
                    if stage == "preflight"
                    else "12:00:00" if conditional else "08:00:00"
                ),
                f"--array=0-{len(cells)-1}%3",
                f"--job-name=imf-{'conditional' if conditional else 'staged'}-{stage}",
                f"--output={root}/submissions/{stage}-%A_%a.log",
                str(script),
            ],
            text=True,
        )
        .strip()
        .split(";")[0]
    )
    if not job.isdigit():
        raise ValueError("Ambiguous scheduler response; do not retry")
    evidence.publish(
        receipt, dict(job_id=job, stage=stage, intent_sha256=evidence.filehash(intent))
    )
    print("STAGED_SUBMITTED", stage, job, flush=True)


def finalize(root):
    manifest, protocol = evidence.manifest(root)
    evidence.scheduler_complete(root, "training", 6)
    results = evidence.verify_stage(root, "training")
    scores = {
        arm: [
            next(r for r in results if r["arm"] == arm and r["seed"] == seed)[
                "evaluations"
            ][-1]["mean_return"]
            for seed in protocol["seeds"]
        ]
        for arm in protocol["arms"]
    }
    deltas = [a - b for a, b in zip(scores["imf"], scores["gaussian"])]

    def iqm(values):
        values = sorted(values)
        lo, hi = len(values) * 0.25, len(values) * 0.75
        return sum(
            v * max(0, min(i + 1, hi) - max(i, lo)) for i, v in enumerate(values)
        ) / (hi - lo)

    report = dict(
        manifest_sha256=evidence.digest(manifest),
        seeds=protocol["seeds"],
        final_seed_mean_returns=scores,
        seed_return_iqms={a: iqm(v) for a, v in scores.items()},
        imf_minus_gaussian=dict(
            paired_seed_deltas=deltas,
            mean_delta=sum(deltas) / 3,
            favorable_fraction=sum(d > 0 for d in deltas) / 3,
        ),
        cells=results,
        successful_cell_wall_seconds=sum(r["wall_seconds"] for r in results),
        statistical_unit="three independent paired training seeds; episodes nested within seed",
        limitations=[
            "Exploratory one task, three seeds, bundled intervention; does not isolate individual changes.",
            "Model-only warmup consumes interaction budget and reduces actor-update count.",
            "Fixed initial replay diagnostic batch has limited coverage; quality gate is heuristic, not a guarantee.",
            "Short-horizon objective uses a 15-step computed rollout but bootstraps correctly at step five.",
            "Gaussian and iMF have different transition losses and parameter counts; same gradient-norm calibration rule.",
            "Artifact authentication is not independent deterministic retraining.",
        ],
    )
    path = Path(root) / "report.json"
    evidence.finite(report)
    if path.exists():
        if evidence.read(path) != report:
            raise ValueError("Retained report differs")
    else:
        evidence.publish(path, report)
    print("STAGED_FINAL_VERIFIED", evidence.filehash(path), flush=True)


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
    args = parser.parse_args()
    if args.action == "register":
        evidence.register(args.root, args.upstream)
    elif args.action == "run":
        run(args.root, args.stage, args.index)
    elif args.action == "verify":
        evidence.verify_stage(args.root, args.stage)
    elif args.action == "submit":
        submit(args.root, args.stage, args.partition)
    else:
        finalize(args.root)


if __name__ == "__main__":
    main()
