# Fixed planning-value pilot (not a controller experiment)

This protocol is frozen before the first scientific head fit. It answers only:
does a separately fitted finite-return value head improve the ranking of the
already recorded candidate endpoints? It does not deploy a controller, change
the feasibility mask, train a world model, or confirm a policy-return benefit.

## Inputs and split

Use the nine authenticated diagnostic traces from completed controller-repair
commit `7d44604d2fe55970c87e6d980bf208cceae71143`. The analysis module pins the
report, manifest, selection, nine strict markers, receipts, cache seals, and
trace digests. No checkpoint is deserialized or modified. Use actor seed 541,
the three tasks and world-model seeds 431, 433, 439 already in those traces.

Fit one head per task/world checkpoint (nine heads total). Calibration seed
76001 supplies fitting data; seed 76003 supplies held-out-for-fitting evaluation.
No final controller-evaluation episode is used. The validation split has already
been inspected in the old-critic diagnosis: this is exploratory, **not blind
validation or independent confirmation**. No architecture, update, seed, or
checkpoint selection follows validation results.

Each head receives the calibration frozen-policy baseline observations with
exact discounted Monte Carlo suffix returns and the unique recorded candidate
endpoints with their frozen-policy continuation returns. Only selected `latent`
reference mode, both endpoint families, and eight nonexecuted candidates enter.
The executed duplicate is excluded. Candidate plans are deduplicated by initial
snapshot and exact action-sequence bytes, including across families. Intermediate
fixed-plan states are never mislabelled as on-policy value observations.

The full baseline suffix group and the unique endpoint group receive equal
total fitting weight. Rows inside each group receive equal weight. This does
not make correlated suffixes or branches independent observations. There is
only **one root fitting episode per checkpoint**.

## Estimand and model

The head approximates `V_pi(s,h) = E[sum_{k=0}^{h-1} .99^k r_k]` for the frozen
observation-space actor, including recorded continuation factors. `h` is the
remaining native episode length. The input is the observation plus `h/1000`.
Train-only weighted observation and target normalizers are used. Model:
two width-128 ReLU hidden layers, scalar output; Adam learning rate 0.0003,
batch size 256, exactly 2000 updates, seed `88000 + diagnostic_index`.
No validation-based stopping or selection. Predictions are clipped to the
known reward-[0,1] finite-horizon interval; remaining-time zero is exactly zero.
Training uses the unclipped standardized scalar regression output.

All fits run on CPU. Old actor, regularized ReBRAC critics, reward model,
world model, endpoint models and immutable study outputs remain untouched.
The old critic retains its original offline-RL role. No arbitrary correction
to its regularization constant is attempted.

## Evaluation and decision rule

For every held-out candidate, score the real five-step reward plus `.99^5`
times the fitted head at the **real** terminal observation. Compare against
the old critic at that same real endpoint, original model score, a time-only
baseline (calibration baseline MC return at the same remaining time), and the
stored real continuation oracle. Keep original feasibility masks fixed.
Report endpoint value MAE/MSE, candidate-relative gain MAE, informative pairwise
agreement, and exact-first-index decision regret. Report empty feasible sets
as undefined decisions, not zero regret or successful safe fallback.

Average four snapshots within each task/world model, then equal-weight three
world models within a task. Report all paired world-model deltas. Candidate
pairs, suffixes, and snapshots are not independent seeds. No p-values or
confirmatory confidence intervals. Report reward support, unique endpoints,
training/validation error, physical clipping, and per-head CPU runtime.

A head is a promising *ranking diagnostic* only if both mean gain MAE and
decision regret improve over the real-endpoint critic, with at least two of
three world seeds improving each, separately for each task and endpoint family.
This descriptive criterion is not a statistical success claim. Any failure or
mixed result stops at this report: no automatic new data, tuning, controller
comparison, GPU job, or confirmation. A lower absolute value MSE alone is not
success, and unchanged infeasible sets remain a separate obstruction.

## Publication and replay

Fit only from a clean exact source commit on main. Publish to a fresh separate
directory, never inside the input artifact tree, with protocol/source/input
bindings, nine JSON heads, report, and file hashes. A second independent CPU
process reauthenticates the inputs, refits all nine heads, requires exact head
and scientific-report equality (runtime is explicitly excluded), validates all
file hashes, and only then creates the strict marker. Existing output files
are never overwritten. Failed evidence remains intact.
