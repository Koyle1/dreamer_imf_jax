"""Fresh single-seed bundled repair; historical results are never training input."""

import argparse
from pathlib import Path

from . import joint_study as joint
from .conditional_schedule import CONTROLS

evidence = joint.evidence


def configure():
    evidence.PROTOCOL = (
        evidence.SOURCE / "dreamer_imf_comparison/conditional_protocol.json"
    )
    evidence.MODULE = "dreamer_imf_compare.conditional_study"
    evidence.verify_cell = verify_cell


def verify_cell(root, stage, cell):
    from ruamel.yaml import YAML

    if cell != dict(index=0, arm="imf", seed=431):
        raise ValueError("Only iMF seed 431 is authorized")
    directory = evidence.cell_dir(root, stage, cell)
    result = evidence.read(directory / "complete.json")
    joint.verify_extra(directory, result, stage == "preflight", conditional=True)
    state = result["staged"]
    if any(row["values"] != CONTROLS for row in state["controls"]):
        raise ValueError("Fixed repair controls differ")
    if evidence.filehash(directory / "config.yaml") != state["config_sha256"]:
        raise ValueError("Configuration digest mismatch")
    cfg = YAML(typ="safe").load((directory / "config.yaml").read_text())["agent"]
    if cfg["repval_grad"] or not cfg["reward_grad"]:
        raise ValueError("Reward/replay-value gradient routing differs")
    if cfg["loss_scales"]["rep"] != 0.1 or cfg["loss_scales"]["dyn"] != 1.0:
        raise ValueError("Loss scales differ")
    if [r["native_steps"] for r in state["metrics_snapshots"]] != [
        r["native_steps"] for r in result["evaluations"]
    ]:
        raise ValueError("Missing checkpoint-time training metrics")
    for record in state["metrics_snapshots"]:
        path = directory / record["path"]
        if path.parent != directory or evidence.filehash(path) != record["sha256"]:
            raise ValueError("Metric snapshot hash mismatch")
        value = evidence.read(path)
        evidence.finite(value)
        if not any(k.endswith("loss/aux_dyn") for k in value):
            raise ValueError("Auxiliary loss not logged")
    result = joint.staged._original_verify(root, stage, cell)
    print("IMF_CONDITIONAL_CELL_VERIFIED", stage, flush=True)
    return result


def run(root, stage, index):
    _, protocol = evidence.manifest(root)
    if index != 0:
        raise ValueError("Exactly one cell is registered")
    cell = evidence.cells(protocol, stage)[0]
    if evidence.cell_dir(root, stage, cell).exists():
        raise FileExistsError("Existing evidence must never be overwritten")
    from .staged_runner import run_cell

    run_cell(root, "imf", 431, stage == "preflight")
    verify_cell(root, stage, cell)


def finalize(root):
    manifest, protocol = evidence.manifest(root)
    evidence.scheduler_complete(root, "training", 1)
    result = evidence.verify_stage(root, "training")[0]
    report = dict(
        manifest_sha256=evidence.digest(manifest),
        cell=result,
        final_mean_return=result["evaluations"][-1]["mean_return"],
        statistical_unit="one training seed; five nested evaluation episodes",
        limitations=protocol["limitations"],
    )
    evidence.finite(report)
    path = Path(root) / "report.json"
    if path.exists():
        if evidence.read(path) != report:
            raise ValueError("Retained report differs")
    else:
        evidence.publish(path, report)
    print("IMF_CONDITIONAL_FINAL_VERIFIED", evidence.filehash(path), flush=True)


def main():
    configure()
    p = argparse.ArgumentParser()
    p.add_argument(
        "action", choices=("register", "run", "verify", "submit", "finalize")
    )
    p.add_argument("--root", required=True)
    p.add_argument("--upstream")
    p.add_argument("--stage", choices=("preflight", "training"))
    p.add_argument("--index", type=int)
    p.add_argument("--partition", default="gpu-a30")
    args = p.parse_args()
    if args.action == "register":
        evidence.register(args.root, args.upstream)
    elif args.action == "run":
        run(args.root, args.stage, args.index)
    elif args.action == "verify":
        evidence.verify_stage(args.root, args.stage)
    elif args.action == "submit":
        joint.staged.submit(args.root, args.stage, args.partition)
    else:
        finalize(args.root)


if __name__ == "__main__":
    main()
