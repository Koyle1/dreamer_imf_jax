"""Fail-closed launch and evidence for exactly one scratch iMF schedule test."""

import argparse
from pathlib import Path
import pickle

import numpy as np

from . import staged_study as staged

evidence = staged.evidence


def configure():
    evidence.PROTOCOL = evidence.SOURCE / "dreamer_imf_comparison/joint_protocol.json"
    evidence.MODULE = "dreamer_imf_compare.joint_study"
    evidence.verify_cell = verify_cell


def verify_extra(directory, result, preflight):
    """Check extra evidence before publishing the inherited strict marker."""
    from .joint_diagnostics import assess

    state = result["staged"]
    if (
        state["protocol"] != "imf-joint-one-seed-v1"
        or state["native"] != result["native_steps"]
        or state["updates"] != result["learner_updates"]
        or not state["controls"]
        or not state["freeze_checks"]
        or not all(c["passed"] for c in state["freeze_checks"])
        or any(c["values"]["transition_only"] for c in state["controls"])
        or state["gate_interpretation"]
        != "legacy scheduling heuristic; not a control certificate"
    ):
        raise ValueError("Joint schedule evidence differs")
    if not any(not c["values"]["actor_enabled"] for c in state["controls"]):
        raise ValueError("Scratch actor warmup missing")
    if preflight and (
        result["learner_updates"] != 8
        or {c["values"]["imag_horizon"] for c in state["controls"]} != {5, 15}
    ):
        raise ValueError("Incomplete preflight phase coverage")
    if evidence.filehash(directory / "diagnostic_batch.npz") != state["batch_sha256"]:
        raise ValueError("Legacy diagnostic batch differs")
    for kind in ("diagnostics", "coverage_diagnostics", "retained_batches"):
        if not state[kind]:
            raise ValueError("Missing diagnostic evidence")
        for record in state[kind]:
            path = directory / record["path"]
            if path.parent != directory or evidence.filehash(path) != record["sha256"]:
                raise ValueError("Diagnostic artifact digest mismatch")
            if kind == "retained_batches":
                with np.load(path, allow_pickle=False) as batch:
                    if (
                        # The pinned stream prepends one context row to 64 learner steps.
                        batch["reward"].shape != (16, 65)
                        or int(np.count_nonzero(batch["reward"] > 0))
                        != record["positive_rewards"]
                        or batch["reward"].size != record["total_rewards"]
                        or not all(np.isfinite(batch[k]).all() for k in batch)
                    ):
                        raise ValueError("Retained batch coverage differs")
            else:
                value = evidence.read(path)
                evidence.finite(value)
                if not value["learner_unchanged"]:
                    raise ValueError("Diagnostic mutated learner")
                if kind == "coverage_diagnostics":
                    if value["batch_path"] not in {
                        r["path"] for r in state["retained_batches"]
                    }:
                        raise ValueError("Unbound coverage input")
                    clocks = value["optimizer_clocks"]
                    for group in ("representation", "transition"):
                        if (
                            clocks.get(f"opt/state/{group}/3/count")
                            != value["learner_updates"]
                        ):
                            raise ValueError("Representation/transition clocks differ")
                    for view in value["views"].values():
                        if view["assessment"] != assess(view["rows"]):
                            raise ValueError("Control assessment differs")
    if len(state["coverage_diagnostics"]) != len(state["retained_batches"]):
        raise ValueError("Coverage/batch count mismatch")
    if not preflight and {
        r["native_steps"] // 50000 for r in state["retained_batches"]
    } != set(range(1, 11)):
        raise ValueError("Missing later-policy samples")
    # Files are authored by this exact study, digest checked before unpickling.
    for record in result["checkpoints"]:
        path = directory / record["path"]
        if path.parent != directory or evidence.filehash(path) != record["sha256"]:
            raise ValueError("Checkpoint digest mismatch")
        with path.open("rb") as file:
            checkpoint = pickle.load(file)
        params = checkpoint["params"]
        rep = int(params["opt/state/representation/3/count"])
        trans = int(params["opt/state/transition/3/count"])
        if (
            rep != trans
            or rep <= 0
            or not all(np.isfinite(v).all() for v in params.values())
        ):
            raise ValueError("Checkpoint optimizer clocks or finiteness differ")


def verify_cell(root, stage, cell):
    if cell != dict(index=0, arm="imf", seed=431):
        raise ValueError("Only iMF seed 431 is authorized")
    directory = evidence.cell_dir(root, stage, cell)
    result = evidence.read(directory / "complete.json")
    verify_extra(directory, result, stage == "preflight")
    result = staged._original_verify(root, stage, cell)
    print("IMF_JOINT_CELL_VERIFIED", stage, flush=True)
    return result


def run(root, stage, index):
    _, protocol = evidence.manifest(root)
    if index != 0:
        raise ValueError("Exactly one cell is registered")
    cell = evidence.cells(protocol, stage)[index]
    if evidence.cell_dir(root, stage, cell).exists():
        raise FileExistsError("Existing artifacts must never be overwritten")
    from .staged_runner import run_cell

    run_cell(root, "imf", 431, stage == "preflight")
    verify_cell(root, stage, cell)


def finalize(root):
    manifest, _ = evidence.manifest(root)
    evidence.scheduler_complete(root, "training", 1)
    result = evidence.verify_stage(root, "training")[0]
    report = dict(
        manifest_sha256=evidence.digest(manifest),
        cell=result,
        final_mean_return=result["evaluations"][-1]["mean_return"],
        statistical_unit="one training seed; five evaluation episodes nested within seed",
        limitations=[
            "Single-seed exploratory schedule-only intervention, not confirmation.",
            "Legacy horizon and balancing heuristics retained to isolate the freeze schedule.",
            "Observable controls are descriptive; reward-selected windows are biased and correlated.",
            "Periodic learner minibatches are retained, not the full replay history.",
            "No velocity-rank, posterior-KL, or critic-gradient intervention was combined here.",
        ],
    )
    path = Path(root) / "report.json"
    if path.exists():
        if evidence.read(path) != report:
            raise ValueError("Retained report differs")
    else:
        evidence.publish(path, report)
    print("IMF_JOINT_FINAL_VERIFIED", evidence.filehash(path), flush=True)


def main():
    configure()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("register", "run", "verify", "submit", "finalize")
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--upstream")
    parser.add_argument("--stage", choices=("preflight", "training"))
    parser.add_argument("--index", type=int)
    parser.add_argument("--partition", default="gpu-a30")
    args = parser.parse_args()
    if args.action == "register":
        evidence.register(args.root, args.upstream)
    elif args.action == "run":
        run(args.root, args.stage, args.index)
    elif args.action == "verify":
        evidence.verify_stage(args.root, args.stage)
    elif args.action == "submit":
        staged.submit(args.root, args.stage, args.partition)
    else:
        finalize(args.root)


if __name__ == "__main__":
    main()
