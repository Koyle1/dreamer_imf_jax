# Trajectory iMF versus shortcut forcing: matched-objective benchmark

## What this study can claim

The registered question is deliberately narrow:

> Under one shared recurrent world-model and actor–critic implementation, does trajectory iMF
> improve both free-running rollout fidelity and downstream actor return over a paper-derived
> Dreamer-4 Equation-(7)-style shortcut-forcing objective, under both matched updates/data exposure
> and matched compiler-reported training FLOPs?

This is an objective comparison. It is not a reproduction of the unreleased full Dreamer 4 system,
its transformer, tokenizer, scale, Minecraft data, or reported agent. A positive result may be
described only as outperforming the implemented Dreamer-4-Equation-(7)-style objective under the
frozen conditions.

## Comparison arms

- `shortcut_forcing`: x-space clean-target prediction, independent token signal levels and step
  sizes, the power-of-two schedule, stopped-gradient two-half-step bootstrap target, the published
  ramp weight, and 4-NFE primary generation. The bootstrap teacher is an EMA of the online model,
  composed intermediate states are bounded to `[-4, 4]`, and finest-level tokens never request a
  half-step below the trained support. The equivalent bootstrap regression is computed directly
  in x-space. Generation is deliberately unclipped so divergence remains observable. The paper
  does not disclose training `K_max`; the pilot therefore sweeps `4, 8, 16` instead of silently
  guessing it.

Every world-model checkpoint must contain finite parameters and metrics. Shortcut training fails
closed if its observed prior loss exceeds `1000`; such a cell cannot advance to rollout or actor
training.
- `trajectory_imf`: predicted-marginal-velocity iMF regression, independent per-token query
  intervals, separately sampled history-exposure times, shared query/history Gaussian noise, the
  clean/corrupted/teacher-corrupted-suffix context mixture, and 1-NFE primary generation.

Both arms use identical observations, actions, stochastic/deterministic state dimensions, encoder,
decoder, recurrent interface, reward and continuation heads, actor, critic, optimizers, batch
shapes, replay data, evaluation starts, and common random numbers. Objective-required parameters
and JVP/bootstrap compute are counted rather than assumed equal.

## Frozen study design

| Phase | Claim role | Tasks | Model seeds | Nested actor seeds | Candidate configs/arm | Cells |
|---|---|---:|---:|---:|---:|---:|
| Smoke | Engineering only | 1 | 1 | 1 | neutral config | 22 |
| Pilot HPO | Development only | 2 disjoint pilot tasks | 3 | 2 | 12 | 606 |
| Confirmatory | Only claim-bearing phase | 6 DM Control tasks | 10 | 3 | one frozen pilot winner | 1,746 |

The pilot evaluates the same exhaustive 12-candidate budget per arm on
`reacher_easy` and `pendulum_swingup`. For each candidate, rollout AUC is reduced by the exact
empirical IQM over six task-by-model-seed units. Episode returns are averaged within actor seed,
then across the two actor seeds nested within each task-by-model-seed unit, then reduced by IQM over
the same six units. The winning rank sum, FLOP tie-break, and config-hash tie-break are frozen before
confirmatory data are unsealed.

The confirmatory suite is `cartpole_swingup`, `ball_in_cup_catch`, `cheetah_run`, `walker_walk`,
`finger_spin`, and `hopper_hop`. Each arm is evaluated under:

1. equal optimizer updates, examples, and minibatch schedules; and
2. equal JAX-compiler-reported forward-and-backward training FLOPs, with integer update allocations
   frozen before outcomes.

World-model seeds are the independent units. Actor seeds and episodes are nested replicates, not
extra independent samples. Rollout curves use horizons `1, 2, 4, 8, 15, 30` and NFEs `1, 2, 4`.
The retained draws recompute standardized rollout error, reward and continuation errors, energy
score, central-90% coverage and width, absolute coverage-calibration error, and continuation Brier
score. The actor comparison freezes the final world-model checkpoint.

## Superiority rule

There are four claim-bearing intervals: two primary outcomes under each of two budget tracks.
Contrasts are oriented so positive favors trajectory iMF:

- shortcut rollout-error AUC minus trajectory rollout-error AUC; and
- trajectory real-environment normalized-return IQM minus shortcut return IQM.

The analysis uses paired, task-stratified bootstrap resampling of model seeds and the registered
multiplicity-adjusted interval confidence. “Outperforms” passes only if the lower endpoint of all
four intervals is strictly above zero. Smoke and pilot profiles are structurally unable to pass.
No secondary metric can rescue a failed primary gate.

## Integrity and cluster safeguards

The run is content-addressed by protocol, complete relevant-source, matrix, config, checkpoint, and
raw-artifact digests. The verifier reconstructs dataset windows and RNG streams, recomputes rollout
statistics and bootstrap indices, checks paired initializations and minibatch prefixes, validates
the exact retained file set, and rejects path traversal or output-root escapes.

Pilot and confirmatory freezes require a clean committed relevant source tree and a complete
`pip freeze`; staged, unstaged, or untracked benchmark source is rejected.
All compute cells must share the same Python, JAX, jaxlib, XLA, platform, device kind/count, and x64
state. Scheduler-assigned visible GPU ordinals are retained as provenance but do not falsely make
GPU 0 and GPU 1 different hardware.

All cluster stages share a fail-closed persistent JAX compilation cache. This makes compute
planning, independent verification, and training reuse the same backend executable rather than
rerunning hardware-sensitive XLA autotuning. Cache errors are fatal, and the cache configuration is
part of the frozen runtime contract. Optimized and unoptimized HLO, raw compiler cost analysis,
parameter counts, structural NFE, and derived update allocations remain retained and digest-bound.

The 1.034% active-world-model parameter gap in the current 256-wide configuration is below the
registered round 2% practical-equivalence trigger. The exact counts and FLOPs remain reported, but
the current source therefore does not require a width-control experiment. If a future source-bound
matrix exceeds 2%, claim interpretation fails closed until the registered joint width control is
provided. Registered deterministic and aleatoric-branching diagnostics are mandatory regardless;
missing or non-recomputable diagnostic evidence stops finalization rather than becoming a warning.

## Running it

Use a real Git clone on a homogeneous GPU worker image. From the repository root:

```sh
export PYTHONPATH=imf_dreamer_jax/src:dreamer_imf_comparison
PYTHON=/private/tmp/trajectory-imf-venv/bin/python

$PYTHON dreamer_imf_comparison/scripts/run_matched_objective_benchmark.py all \
  --profile smoke \
  --output dreamer_imf_comparison/results/matched_objective_smoke

$PYTHON dreamer_imf_comparison/scripts/run_matched_objective_benchmark.py all \
  --profile pilot \
  --output dreamer_imf_comparison/results/matched_objective_pilot

$PYTHON dreamer_imf_comparison/scripts/run_matched_objective_benchmark.py freeze \
  --profile confirmatory \
  --pilot-output dreamer_imf_comparison/results/matched_objective_pilot \
  --output dreamer_imf_comparison/results/matched_objective_confirmatory

# Run primary cells (locally or as dependency-ordered scheduler-array jobs).
$PYTHON dreamer_imf_comparison/scripts/run_matched_objective_benchmark.py run \
  --profile confirmatory \
  --output dreamer_imf_comparison/results/matched_objective_confirmatory

# Mandatory interpretation diagnostics; these never substitute for primary outcomes.
$PYTHON dreamer_imf_comparison/scripts/run_matched_objective_diagnostics.py \
  --output-root dreamer_imf_comparison/results/matched_objective_confirmatory

$PYTHON dreamer_imf_comparison/scripts/run_matched_objective_benchmark.py finalize \
  --profile confirmatory \
  --output dreamer_imf_comparison/results/matched_objective_confirmatory

$PYTHON dreamer_imf_comparison/scripts/run_matched_objective_benchmark.py verify \
  --profile confirmatory \
  --output dreamer_imf_comparison/results/matched_objective_confirmatory
```

For a scheduler array, freeze once, submit one `--cell-id` at a time in dependency order, run the
diagnostics, then run `finalize` and `verify`. `generate_matched_objective_matrix.py` also requires a verified
`--pilot-output` for confirmatory generation; a detached selection JSON is intentionally rejected.
Rerunning completed cells validates and skips them, while partial world/actor checkpoints resume.

The local iCloud-backed worktree currently times out on Git identity and occasionally dependency
capture. That is acceptable only for the nonclaim smoke profile. Do not use that fallback for pilot
or confirmatory results; freeze those from the cluster's ordinary Git clone and verify `git diff`
and `python -m pip freeze` before submitting the array.

## Current evidence status

The smoke profile tests orchestration, differentiation, compilation, checkpointing, stochastic
rollouts, actor updates, analysis, and artifact verification. It uses only two world-model and two
actor updates and therefore cannot estimate comparative performance. Its numerical ordering must
not be used for model selection, method claims, or publication conclusions.

The definitive result is not available until the disjoint pilot, required diagnostics, and all
1,746 primary-matrix confirmatory cells pass the independent verifier. A width control is additionally
required only if the source-bound active-parameter gap exceeds 2%.
