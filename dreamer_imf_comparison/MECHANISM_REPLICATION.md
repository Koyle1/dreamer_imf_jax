# Fixed three-task mechanism replication

Authorized 2026-09-24. The executable contract is
`mechanism_replication_protocol.json`, committed before any scientific training.
This is a narrow replication of two exploratory comparisons, not a full
FlowMPC reproduction or a new iMF-versus-shortcut benchmark.

## Scope

- Reacher Hard, Cartpole Swingup, Finger Spin.
- World seeds 431/433/439; nested actor seeds 541/547; five common evaluation
  environment seeds. Four arms, 72 cells, 360 episodes.
- A3 minus A1 tests independent-noise acceptance under persistent trust.
- K0 minus K1 tests a compound endpoint-training/recursive-versus-direct
  planning intervention. It does not isolate horizon consistency.
- No tuning, policy-tilt/pessimism/optimizer sweeps, shortcut training, pixels,
  negative-result retries, or historical outcome controls.

## Fresh prerequisites

Nine task/world cells each collect 50,000 random/smoothed-random transitions,
split whole episodes 40/10, and train a trajectory-iMF world model for 10,000
updates (batch 32, sequence 32). Architecture/objective settings are copied
from the authenticated prior iMF configuration; observation/action dimensions
are task-specific. A 3x512 ReLU state-action reward head trains for 10,000 updates
with the original world frozen. Reward testing samples test episodes only.
Two ReBRAC policies receive 1,000,000 updates each. Two endpoint models receive
5,000 updates each (batch 16), with all source model subtrees frozen.
Training-only anchors and 95th-percentile behavior distances are fixed before
evaluation. No training result is selected on return.

## Execution and evidence

Use only `python -m dreamer_imf_compare.mechanism_replication`.
Commands: `register`, `launch --stage preflight`, `verify --stage preflight`,
`launch --stage training`, `verify --stage training`, `launch --stage evaluation`,
`verify --stage evaluation`, `finalize`. Every command requires `--root PATH`.
The source checkout must have an exact clean tracked commit matching its
immutable manifest. Each stage is submitted once, held until the immutable
receipt is written, then released. Stage advancement rechecks Slurm completion
and every artifact digest. Existing or uncertain submissions fail closed.

Three GPU preflights exercise the full architecture and particle counts but
only two world/reward/endpoint updates, four ReBRAC updates and eight controller
steps for each arm. These are engineering checks, never scientific outcomes.
Every evaluation has a fresh job/index/arm cache, an external discarded primer,
separate creation/replay readers, in-process/post-exit cache digests and seals,
distinct process identities, and exact semantic plus bitwise trace equality.
The compiler-writer primer trajectory is not a historical/evaluation control.

All publication is exclusive; no retry overwrites an artifact. A failed cell
blocks its entire stage and final analysis. Inspect the error before proposing
a repair; source repairs require a new commit and fresh registered root.
No scheduler-pending/running unit may be resubmitted.

## Analysis and limitations

Episode means are nested within actor means within trained-model seeds. The
primary endpoint is the equal-task mean of the fractional IQM of paired
world-seed deltas. Bootstrap model-seed clusters separately within each fixed
task, with 20,000 resamples and 97.5% intervals for each of two contrasts.
Three clusters per task give sparse uncertainty support; this is not an
estimate over an unseen task population, and actors/episodes are not additional
trained-model replicates. Raw DMC returns are compared on their native 0–1000
scale; the >=800 fraction is not labeled native task success.

Retain all traces, proposal objectives, trust/acceptance/saturation diagnostics,
coverage and runtime evidence. Model objectives include terminal Q, so they
must not be subtracted from whole-episode returns and called a matched-horizon
model gap. The current four-arm runner does not identify that counterfactual
gap; it is explicitly unavailable rather than fabricated. A positive paired
effect supports only the registered intervention on these tasks; neither
causality of representation quality nor iMF superiority is established.

Previous terminal bookkeeping is separate: job 27657752 completed successfully
and created the canonical terminal attestation for original final job 27656864,
without changing its report or scientific artifacts.
