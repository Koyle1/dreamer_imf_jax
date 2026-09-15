# Trajectory-iMF actor-gap roadmap study

This is a frozen, exploratory diagnosis of one narrow question: trajectory-iMF
improves some representation and rollout metrics on DMC Reacher, but why does
that improvement not reliably become control return?  The study tests seven
candidate remedies while preserving the completed ReBRAC + ITPO study as an
immutable dependency.

It is not a confirmatory benchmark, a DreamerV4 comparison, or a full FlowMPC
reproduction.  The manifest sets `claim_eligible=false`.  No evaluation return
is used to choose a hyperparameter, alter an arm, or mutate the frozen matrix.

## Frozen design

- Task: `dmc_reacher_easy`.
- Top-level statistical units: world-model seeds `211`, `223`, and `227`.
- Nested replicates: actor seeds `311` and `313` within each world seed.
- Evaluation: the first two already-authenticated held-out environment seeds
  for each world/actor pair.  The dependency tuning seed is excluded.
- Controller horizon: 5.
- FlowMPC particles: 4,096.
- Action-sequence particles: 64.
- Model continuation: 5,000 updates, batch size 16, sequence length 32, learning
  rate `3e-4`, global gradient clip 10, and a fresh zero-moment Adam state.
- Every live controller has one pure, discarded compiled warm-up call.  It
  cannot touch the environment, belief, or persistent actor and is excluded
  from latency. Each diagnostic task additionally starts from a fresh
  job/index-scoped cache. A separate discard-only process runs the exact `A0`
  path for both actor seeds without applying the dependency-equality gate,
  then runs one complete diagnostic before result creation and replay become
  cache readers. The primer opens isolated simulator instances but publishes
  no scientific artifact and mutates no checkpoint.

The matrix contains 3 diagnostic cells, 21 model-training cells, and 90 fresh
evaluation cells.  The 90 evaluations are the Cartesian product of 15 fresh
arms, 3 world seeds, and 2 nested actor seeds.  Two reference arms are read
from authenticated dependency evidence and are never retrained.

## The seven interventions and exact arms

### 1. Frozen causal diagnostics

One diagnostic cell per world seed evaluates both nested actors without
selecting or changing a model.  It records:

- train-replay observation-action coverage for executed actions and for the
  horizon-5 diagnostic proposal, predicted-best, and reference trajectories;
- one-step Bellman residual decomposition;
- rollout errors at horizons 1, 3, and 5;
- proposal-noise versus independent-noise objective and gradient agreement,
  with proposal improvement, held-out improvement, and their generalization
  gap reported separately;
- exact simulator snapshot/restore one-step reward counterfactuals and causal
  horizon-5 planner-objective ranking. The latter compares world-model reward
  plus terminal-Q predictions against exact simulator reward plus the same
  frozen critic evaluated on the real terminal observation. Its three action
  sequences are a fixed local causal probe (reference and full-horizon ±0.1 in
  action dimension 0), not validation of the complete CEM population.

The diagnostics are descriptive.  They cannot select an arm or tune an
evaluation-time threshold.

The diagnostic code does not regenerate the persistent controller between
measurements.  During the exact `A0` rollout it retains the live posterior
beliefs and, for steps 0, 1, 5, and 15 of the first episode, the in-memory
adaptation-start actor, updated actor, current observation, and proposal-noise
bank.  Only after the complete rollout reproduces the authenticated dependency
trace exactly are independent-noise gradients and counterfactual measurements
computed from those contexts.  The actor trees remain ephemeral; the result
publishes only their SHA-256 digests.  This avoids treating a numerically close
second persistent rollout as the same causal controller state.

Every fresh evaluation arm also records bounded live-planner imagined
occupancy every 50 environment steps using the first 8 particles from the
actual shared planner-noise bank. Coverage is reported separately for the
proposal, retained/post-safeguard best, and frozen-reference plan. For CEM,
“proposal” is the actually evaluated first population; for gradient search it
is the retained gradient proposal; for FlowMPC it is the pre-safeguard adapted
actor. These measurements are excluded from controller latency.

### 2. Reference-policy action trust region

The trust region constrains behavior, not parameter distance.  A proposal is
backtracked along its parameter update until both conditions hold relative to
the immutable ReBRAC actor:

- summed anchor-state mean squared action drift is at most `0.01` over 256
  training-only anchor observations;
- current-state maximum absolute action drift is at most `0.10`.

The isolated contrast is:

- `A0_persistent_unconstrained_replay`: exact persistent, unconstrained replay
  bridge to the authenticated dependency;
- `A1_persistent_trust`: the same controller with only the action trust region.

`F1` and `F3` repeat the same trust mechanism for the uniformly continued and
policy-tilted priors.

### 3. Persistence and independent-noise acceptance

- `A2_reset_trust`: start every environment-step adaptation from the frozen
  reference actor instead of carrying the previous adapted actor.  Compare to
  `A1_persistent_trust` to isolate persistence.
- `A3_persistent_trust_heldout`: retain persistence and trust, but accept the
  proposed actor only when it does not reduce the objective on an independent
  held-out noise bank.  Rejection falls back to the adaptation-start actor.
  Compare to `A1_persistent_trust` to isolate acceptance.

Proposal noise and held-out noise are derived from separate deterministic key
namespaces.  “Held out” here means held out from the current update, not a new
environment test set.

### 4. Same-variable gradient versus CEM search

- `O1_recursive_action_sequence_gradient`;
- `O2_recursive_action_sequence_cem`.

Both optimize the same horizon-5 residual action sequence around the reference
policy, with every residual component clipped to `[-0.10, 0.10]`, and both use
the same recursive source trajectory-iMF world model.  Each decision consumes
exactly 10 scalar objective calls: the gradient arm uses nine
value-and-gradient evaluations plus one final candidate evaluation, while CEM
uses two iterations of four candidates plus its initial and final-mean
evaluations.  This is an
objective-call match, not a FLOP match: reverse-mode derivatives make the
gradient arm more expensive per call.

### 5. Policy-density-tilted trajectory-iMF

The prior-only continuations are:

- `F0_uniform_prior_unconstrained`;
- `F1_uniform_prior_trust`;
- `F2_policy_tilt_prior_unconstrained`;
- `F3_policy_tilt_prior_trust`.

Only the trajectory-iMF prior and a fresh optimizer train; the encoder,
posterior, recurrent dynamics, observation/reward/continuation heads, and
ReBRAC actor/critics remain frozen.  Because ReBRAC is deterministic and has no
Lebesgue action density, the policy score is explicitly an approximation: a
fixed-variance Gaussian kernel of standard deviation `0.20` around the actor's
action.  Training-transition weights are

```
w_i = exp(eta * log pi_approx(a_i | s_i))
      / mean_j exp(eta * log pi_approx(a_j | s_j)).
```

`eta` is selected globally from the frozen training-only grid subject to an
effective-sample-size fraction of at least `0.50` and a clipped-weight fraction
of at most `0.05`.  Evaluation returns never enter this selection.

The primary causal contrasts are `F2 - F0` without trust and `F3 - F1` with
trust.  `F0` and `F1` are required continuation controls; comparing `F2` only
to the old source model would confound density weighting with extra updates.

### 6. Ensemble-relative pessimism

- `P0_recursive_cem_ensemble_relative_mean`: CEM action sequences scored by
  the mean relative residual across the three frozen world models;
- `P1_recursive_cem_relative_pessimism`: the identical controller scored by
  mean minus one standard deviation.

`P1 - P0` isolates the pessimism term.  `P0 - O2` measures the effect of
ensemble averaging relative to the single-world CEM arm.  Mean-minus-standard-
deviation is a heuristic risk score, not a calibrated confidence bound.  The
three ensemble members are the same three top-level world seeds and do not
create additional independent statistical units. All ensemble members receive
the same posterior random draw at each observed state (common random numbers),
so the reported dispersion does not mix model ordering with member-specific
posterior noise.

### 7a. Planner-value-equivalent training

- `V1_cvaml_value_prior_unconstrained` versus
  `F0_uniform_prior_unconstrained`.

Only the prior trains.  With `K=4` stochastic planner backups `Y_k`, target
`T`, and sample variance `s_Y^2` using `ddof=1`, the proof-consistent squared
estimator is

```
(mean_k Y_k - T)^2 - s_Y^2 / K.
```

It is normalized by a training-only Bellman-target scale and added to the full
trajectory-iMF objective.  The correction can make a finite-batch loss
negative; clipping it would change the estimator.  This implementation is
labelled separately from the paper's main-text shorthand and released-code
finite-`K` convention.

### 7b. Direct action-chunk endpoints

- `K0_endpoint_h1_recursive_cem_filtered`: a horizon-1 endpoint-iMF control
  trained and used recursively;
- `K1_endpoint_anystep_direct_cem_filtered`: an any-step endpoint-iMF trained
  on horizons 1 through 5 and queried directly for action chunks.

These two fresh endpoint models have identical architecture, initialization,
batch/start schedule, optimizer, update count, maximum horizon-5 condition
shape, and full iMF compound objective.  Only the sampled training horizon and
direct-versus-recursive query differ.  Both use the same empirical
behavior-distance filter calibrated from training replay.  That filter is a
rejection diagnostic, not proof of support.

## Reference arms and registered contrasts

- `R0_zero_shot_dependency`: immutable zero-shot ReBRAC result.
- `R1_unconstrained_dependency`: immutable published-style ITPO adaptation
  result.
- `A0_persistent_unconstrained_replay`: fresh exact bridge that must reproduce
  the matching prefix of `R1` before any new controller conclusion is trusted.

The manifest freezes these comparisons:

| Question | Candidate minus baseline |
|---|---|
| Trust | `A1 - A0` |
| Reset versus persistence | `A2 - A1` |
| Held-out acceptance | `A3 - A1` |
| CEM versus gradient search | `O2 - O1` |
| Ensemble mean versus single model | `P0 - O2` |
| Relative pessimism | `P1 - P0` |
| Policy tilt, unconstrained | `F2 - F0` |
| Policy tilt, trusted | `F3 - F1` |
| Planner-value equivalence | `V1 - F0` |
| Any-step direct chunks | `K1 - K0` |

## Causal isolation rules

1. The dependency manifest, report, cell markers, traces, checkpoints, source
   commit, and reward artifacts are SHA-256 authenticated before loading.
2. Controller-only arms keep every world-model and ReBRAC input bitwise frozen.
3. Prior-training arms return only a changed prior plus a new optimizer; all
   other source subtrees are digest-checked as frozen.
4. Density-tilt and CVAML cells are actor-specific because their objective
   depends on the actor.  Uniform and endpoint cells are shared within a world
   seed.  This produces exactly 21 model cells rather than silently averaging
   incompatible actor objectives.
5. Calibration uses training replay only.  The policy-tilt `eta`, trust
   anchors, behavior thresholds, and Bellman scale are frozen before the first
   evaluation.
6. Every cell has deterministic keys, immutable outputs, and a digest-bound
   marker. Creation and strict replay are separate Python processes. Their
   Linux process identities must differ, while both must identify the same
   registered Slurm array index and exactly one visible JAX GPU. Diagnostic and
   evaluation tasks additionally start from fresh job/index-scoped compilation
   caches, run exact discard-only primer processes, fingerprint the populated
   caches, and then run creation and replay as two cache-reader processes. The
   diagnostic primer performs its two-actor-seed `A0` prepass before the full
   discarded diagnostic. Each reader rechecks the actual tree before
   publication. After that reader exits, a separate process rechecks the tree
   and publishes a digest-bound creation or verification cache seal. Missing
   creation seals cannot be retried as retained data, and markers without
   verification seals cannot be promoted.
7. The canonical submitter holds each array before release, persists an
   immutable receipt first, and only then releases the workers. Results bind
   their creation map and receipt; markers independently bind their replay map
   and receipt. Existing artifacts are never overwritten by a retry.

## Aggregation and interpretation

For each arm, episode returns are averaged within an actor seed, the two actor
means are averaged within their world-model seed, and only then are the three
world-seed values summarized.  Paired contrasts are computed at that same
world-seed level.  Actor seeds and episodes are nested measurements, not six or
twelve independent samples.

The report includes the three-world interquartile mean, every paired world-seed
contrast, favorable-seed fractions, saturation, objective improvement,
gradient and parameter deltas, trust backtracks, frozen-reference fallbacks,
held-out acceptance improvement, empirical executed/imagined coverage, causal
horizon-5 ranking diagnostics, and runtime. Controller latency excludes only
discarded compilation warm-ups, including the separate diagnostic and
evaluation primers; it
includes the first live step of every retained episode. Each primer publishes
only a runtime receipt—not scientific cell output—and the final report adds
the unique receipt times to total accounted compute. Failed attempts remain
separately visible in Slurm/provenance evidence and are not folded into the
successful-path total. This is scientific compute-path time rather than total
allocated GPU time: shell cache hashing, post-exit seal processes, queue time,
and scheduler overhead remain in Slurm accounting. With only
three top-level units on one easy task, these estimates are useful for deciding
which mechanism deserves a preregistered multi-task study; they do not support
a NeurIPS-level performance claim by themselves.

Imagined coverage is sampled every 50 live steps from the actual shared
planner-noise particles. It compares the actual proposal, selected/best, and
frozen-ReBRAC reference trajectories. For ensemble action-sequence arms this
bounded telemetry uses the focal world model; it must not be interpreted as
coverage under every ensemble member. Coverage measures the horizon's stage
reward-query occupancy; the terminal critic query is explicitly excluded. The
separate three-sequence horizon-5
counterfactual diagnostic is a local causal probe, not a validation of the
full CEM search distribution.

## Run and verification stages

The only valid order is:

```
preflight
  -> calibration
  -> diagnostic-primer[0:3] -> diagnostic-cell[0:3]
                              -> post-exit creation cache seal
                              -> fresh-process replay
                              -> post-exit verification cache seal
                              -> verify-diagnostics
  -> model-cell[0:21]      -> fresh-process replay -> verify-models
  -> evaluation-primer[0:90] -> evaluation-cell[0:90]
                              -> post-exit creation cache seal
                              -> fresh-process replay
                              -> post-exit verification cache seal
                              -> verify-evaluations
  -> finalize
```

Each array stage may run cells in parallel, but the whole-stage verifier must
finish successfully before the next stage starts.  The cluster entry point is
`cluster/actor_gap_roadmap/actor_gap_roadmap.sbatch`; it requires exact source,
dependency, output, commit, and stage environment variables.  The portable
entry points are:

```
python scripts/run_actor_gap_roadmap_study.py <stage> ...
python scripts/verify_actor_gap_roadmap_study.py <gate> --output-root <root>
```

There are 114 logical scientific cells and 120 Slurm tasks/jobs including
preflight, calibration, the three whole-stage verifiers, and finalization. A
cell's discard-only primer exits before creation, and creation exits before its
strict replay starts, even though all commands execute sequentially inside the
same Slurm array task. The primer cannot publish a result, trace, marker, or
checkpoint; its only output-root artifact is an authenticated runtime receipt.

Required terminal tokens are:

- `ACTOR_GAP_ROADMAP_PREFLIGHT_VERIFIED`;
- `ACTOR_GAP_ROADMAP_CALIBRATION_VERIFIED`;
- `ACTOR_GAP_ROADMAP_DIAGNOSTICS_VERIFIED`;
- `ACTOR_GAP_ROADMAP_MODELS_VERIFIED`;
- `ACTOR_GAP_ROADMAP_EVALUATIONS_VERIFIED`;
- `ACTOR_GAP_ROADMAP_STUDY_FINAL_VERIFIED`.

A failed or missing cell is retried only at its registered index under the same
source/output contract.  A source change invalidates the run: preserve the old
evidence, create a fresh exact commit and output root, repeat GPU preflight, and
rerun every stage.  Never weaken strict replay to make a result pass.

## Known limits

- One Reacher task, three world seeds, two nested actors, and two held-out
  episodes per actor are too small for a broad performance claim.
- The study reuses frozen data, reward models, and ReBRAC checkpoints; it does
  not jointly train policy-tilted dynamics as full FlowMPC does.
- Approximate deterministic-policy density, relative mean-minus-SD pessimism,
  and empirical behavior filtering have deliberately narrow interpretations.
- Equal objective-call counts do not mean equal compute.
- Decoded-observation endpoint models are a targeted mechanism test, not a
  general pixel-world-model benchmark.
