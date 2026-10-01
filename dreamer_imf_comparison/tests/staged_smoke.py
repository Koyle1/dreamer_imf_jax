"""CPU integration smoke: real environment, replay, outer Agent and checkpoints."""

import json
from pathlib import Path
import sys
import tempfile

from dreamer_imf_compare import dreamer_ablation_runner as base
from dreamer_imf_compare.staged_runner import run_cell

root = Path(tempfile.mkdtemp(prefix="staged-smoke-"))
protocol = json.loads((Path(__file__).parents[1] / "staged_protocol.json").read_text())
protocol["upstream_path"] = "/private/tmp/staged-dreamerv3-upstream"
protocol["preflight"].update(
    native_steps=128,
    eval_at_native_steps=[128],
    envs=1,
    batch_size=2,
    batch_length=8,
    jax_platform="cpu",
    episode_native_steps=20,
)
(root / "protocol.json").write_text(json.dumps(protocol))
original = base._config


def tiny(*args, **kwargs):
    cfg = original(*args, **kwargs)
    return cfg.update(
        {
            "agent.dyn.rssm.deter": 8,
            "agent.dyn.rssm.hidden": 8,
            "agent.dyn.rssm.stoch": 2,
            "agent.dyn.rssm.classes": 3,
            "agent.dyn.rssm.blocks": 2,
            "agent.enc.simple.units": 8,
            "agent.dec.simple.units": 8,
            "agent.enc.simple.layers": 1,
            "agent.dec.simple.layers": 1,
            "agent.rewhead.units": 8,
            "agent.conhead.units": 8,
            "agent.policy.units": 8,
            "agent.policy.layers": 1,
            "agent.value.units": 8,
            "agent.value.layers": 1,
            "replay_context": 0,
            "report_length": 8,
        }
    )


base._config = tiny
result = run_cell(root, sys.argv[1], 431, preflight=True)
assert result["native_steps"] == result["staged"]["native"] == 128
assert result["learner_updates"] == result["staged"]["updates"] == 8
assert result["staged"]["diagnostics"]
assert len(result["staged"]["freeze_checks"]) == 8
assert {r["values"]["imag_horizon"] for r in result["staged"]["controls"]} == {5, 15}
print("STAGED_CPU_INTEGRATION_VERIFIED", sys.argv[1], root)
