# Planning value: target semantics and a bounded diagnostic

The existing ReBRAC critic is a regularized, smoothed, off-policy TD surrogate.
It is not trained to equal the finite raw-reward continuation used by the repair
diagnostic. A discrepancy in their absolute scales therefore does not establish
a critic bug or establish that a new value head would improve control. The
decision-relevant evidence is the variation of that discrepancy across candidate
plans from the same snapshot, with the candidate set and feasibility mask fixed.

This document audits the frozen implementation. It proposes a separate head only
conditional on the authenticated ranking analysis implicating the old terminal
surrogate. It neither changes old study artifacts nor authorizes a controller
comparison, threshold change, or closed-loop deployment. Numerical empirical
conclusions belong to the separately generated decision-ranking report.

## What the implemented critic learns

For a replay tuple `(s, a, r, s', a'_D, d)` and supplied standard-normal noise
`epsilon`, the actual target is

```text
a_tilde = clip(pi_target(s') + clip(sigma * epsilon, -c, c), -1, 1)
b       = sum_over_action_dimensions (a_tilde - a'_D)^2
y       = r + gamma * (1 - d) * [min_i Q_target,i(s', a_tilde) - beta_Q * b]
L_Q     = sum_i mean_batch (Q_i(s, a) - stop_gradient(y))^2.
```

The source is [flowmpc.py:340–368](../imf_dreamer_jax/src/imf_dreamer_jax/flowmpc.py#L340).
Defaults are `gamma=.99`, `beta_Q=1`, `sigma=.2`, and `c=.5`
([lines 34–52](../imf_dreamer_jax/src/imf_dreamer_jax/flowmpc.py#L34)).
The penalty is a **sum** of squared action differences, not the RMS constraint
used by the controller. It is evaluated after both noise and action clipping.
It penalizes the successor action, inside the discounted continuation; it does
not directly subtract a penalty from the current replay reward.

The offline replay was collected from a mixture of uniform and temporally smooth
random actions, not from the eventual frozen actor
([mechanism_replication.py:259–307](../dreamer_imf_comparison/dreamer_imf_compare/mechanism_replication.py#L259)).
The replay builder selects training episodes, supplies their actual next
behavior actions as `a'_D`, drops the final timeout transition, and sets
`d=1-continuation` on retained transitions
([flowmpc_actor_study.py:457–485](../dreamer_imf_comparison/dreamer_imf_compare/flowmpc_actor_study.py#L457)).
Thus even an ideal fixed point would reflect the distribution of behavior
successors against which the noisy actor is penalized. Under the idealized case
of a stationary actor, exact critics, a suitable Markov replay distribution,
and consistent bootstrapping, it resembles a discounted return with successor
behavior costs under a smoothed policy. Those assumptions are not a description
of the learned finite-data network, and that idealization is not raw `V^pi`.

Several additional distinctions matter:

- The bootstrap uses a noisy **target** actor; planning queries the final
  deterministic actor. Delayed Polyak updates are implemented at
  [flowmpc.py:406–423](../imf_dreamer_jax/src/imf_dreamer_jax/flowmpc.py#L406).
- Taking the smaller of two estimates can reduce overestimation, but does not
  prove a lower bound on true return or supply an uncertainty interval. The
  two critics can share approximation errors.
- Critic predictions on actor actions at imagined or counterfactual endpoints
  may lack replay support. Off-policy training does not by itself make a
  function approximation error small there.
- Actor training has a *separate* behavior loss, using `beta_pi * ||pi(s)-a_D||²`
  minus a batch-normalized minimum Q. The normalizer rescales the actor's Q term;
  it does not normalize Q training targets into an episode-return estimate
  ([flowmpc.py:322–337](../imf_dreamer_jax/src/imf_dreamer_jax/flowmpc.py#L322)).
- State input has no explicit remaining episode time. The critic replay does
  not impose a zero boundary at the native timeout; this differs from the
  diagnostic's finite episode objective. Off-policy TD bootstrapping can be
  appropriate for its own continuing objective while missing that boundary.

The test file executes the actual loss with real three-layer ReBRAC networks
having analytically controlled outputs. Its positive/negative controls isolate
noise scaling/clipping, summed penalty, the minimum critic, removal of the
penalty, and terminal masking. A separate test executes the real replay builder
to verify the behavior-successor alignment and omitted timeout transition.

## What planning and the real continuation measure

For each candidate sequence `u=(a_0,...,a_4)`, the endpoint scorer uses

```text
J_model(u) = mean_particles sum_{t=0}^{4} gamma^t r_hat(s_hat_t, a_t)
             + gamma^5 mean_particles min_i Q_i(s_hat_5, pi(s_hat_5)).
```

The minimum is taken **before** the particle average. The stage rewards are raw
predicted rewards; the terminal critic imports its training regularization and
bootstrap semantics into only the tail. There is no learned continuation/time
mask in this scorer
([repaired_controllers.py:119–168](../dreamer_imf_comparison/dreamer_imf_compare/repaired_controllers.py#L119)).

The real diagnostic restores a matched simulator snapshot, executes the same
fixed open-loop plan, then follows the frozen deterministic actor, stopping at
the native episode's `is_last`. Rewards before index five are stage rewards;
the continuation starts at reward index five. If `c_j` is the environment's
continuation multiplier and `T` the actual remaining native episode length,

```text
R_real(u) = sum_{t=0}^{min(5,T)-1} gamma^t [product_{j<t} c_j] r_t
C_real(u) = sum_{t=5}^{T-1} gamma^t [product_{j<t} c_j] r_t
J_real(u) = R_real(u) + C_real(u).
```

This is a realized Monte Carlo return, or an estimate of its conditional
expectation if the environment is stochastic. It is not the return of repeated
replanning. Snapshot restoration and reward indexing are explicit in
[repair_diagnostics.py:140–215](../dreamer_imf_comparison/dreamer_imf_compare/repair_diagnostics.py#L140).
The adapter carries `continuation` and `is_last` separately
([dmc.py:147–164](../dreamer_imf_comparison/dreamer_imf_compare/dmc.py#L147));
stopping on `is_last` matters even when the final environment discount is one.
Full diagnostics reject trajectories that do not reach that native end
([repair_diagnostics.py:895–902](../dreamer_imf_comparison/dreamer_imf_compare/repair_diagnostics.py#L895)).
The executable timeout test gives identical observations and a fixed actor two
different remaining horizons; their raw continuations differ despite nonzero
continuation multipliers throughout.

The existing trace records the old critic also at the **real** endpoint:

```text
C_old_real(u) = gamma^5 min_i Q_i(s_real_5, pi(s_real_5))
C_model(u) - C_real(u)
  = [C_model(u) - C_old_real(u)] + [C_old_real(u) - C_real(u)].
```

The first difference isolates the consequence of substituting generated
endpoints for real ones *under the old critic*. The second includes critic
approximation/support error, regularization/smoothing differences, and the
finite-time objective mismatch. It does not isolate which of these caused the
error. The fields are assigned at
[repair_diagnostics.py:750–829](../dreamer_imf_comparison/dreamer_imf_compare/repair_diagnostics.py#L750).
The trace assigns zero `C_old_real` if the episode ends before the full plan; if
it ends exactly at step five, the code still queries Q at the terminal endpoint
while the MC tail is empty. Do not describe this as a terminal-aware bootstrap.
For data with fractional intermediate continuations, the stored old-critic
endpoint term also omits their survival product; this must be included in the
interpretation rather than silently changing the authenticated trace.

## Absolute error, candidate differences, and regret

Fix one matched snapshot, one candidate set `A`, and the original feasibility
mask. Let `J` be its real score and `J_hat=J+e` a candidate scoring rule. Then

```text
J_hat(u) - J_hat(v) = J(u) - J(v) + e(u) - e(v).
```

Any common additive error cancels. A state-independent Q shift `b` contributes
the same `gamma^5 b` when every candidate uses the same horizon and terminal
weight. It changes reported absolute value errors without changing the argmax.
With early termination, differing survival weights, or different horizons that
common-shift condition needs to be checked rather than assumed. Multiplying
only the terminal Q by a scale is not innocuous: it changes the tradeoff between
stage reward and continuation.

If `u*` maximizes real score on nonempty `A` and `u_hat` maximizes estimated
score on exactly the same `A`, then

```text
0 <= J(u*) - J(u_hat)
   <= e(u_hat) - e(u*)
   <= max_A e - min_A e
   <= 2 max_A |e - b|, for any common b.
```

This follows by adding and subtracting the two estimated scores and using
`J_hat(u_hat)>=J_hat(u*)`. A pair with true gap `Delta>0` reverses only if the
error favors the worse action by at least `Delta` (strictly greater for strict
reversal). Thus a large uncentered MAE is neither necessary nor sufficient for
poor ranking. Candidate-dependent error can be harmful even with a modest
absolute error. Infeasible candidates or empty feasible sets are outside this
argmax claim; ties must use the recorded deterministic tie rule.

The tests call the actual endpoint scorer and `feasible_first_order` with
analytical model components, and compare with actual `rollout_fixed_plan`
returns. They demonstrate a harmless large offset and a smaller, directional
error that reverses a real preference. This anchors the algebra to the deployed
score composition and selection ordering; it is not evidence of performance on
the nine trained checkpoints.

## Why feasibility and fallback must be held fixed

The endpoint constraint is

```text
D(u) = max_{t=0,...,4} mean_particles sqrt(mean_action_dims
          (u_t - pi(s_hat_t))^2)
feasible = finite scores/actions and |u|<=1 and D(u)<=threshold.
```

Source: [repaired_controllers.py:119–168](../dreamer_imf_comparison/dreamer_imf_compare/repaired_controllers.py#L119).
The threshold is the 95th percentile of **single-transition replay** RMS action
residuals, evaluated on real training states
([mechanism_replication.py:472–475](../dreamer_imf_comparison/dreamer_imf_compare/mechanism_replication.py#L472)).
That calibration distribution and statistic differ from a maximum over time of
particle-averaged residuals at generated states. A latent reference can disagree
with the endpoint rollout, and even a same-endpoint zero-noise reference can
disagree with stochastic endpoint particles. Jensen/nonlinearity and the
timewise maximum mean that scoring the reference does not guarantee zero drift.

The CEM search sorts feasible candidates before infeasible ones, minimizing
violation among infeasible candidates
([robust_flowmpc.py:1091–1111](../imf_dreamer_jax/src/imf_dreamer_jax/robust_flowmpc.py#L1091)).
If the final proposal is infeasible, the planner executes the entire reference
and reports its original metrics, even if that reference is itself infeasible
([repaired_controllers.py:255–275](../dreamer_imf_comparison/dreamer_imf_compare/repaired_controllers.py#L255)).
Our executable counterexample checks this behavior with symmetric endpoint
particles: their mean state is zero, but their mean policy drift is positive.

A new value score does not repair a fixed empty feasible set. Analyze the
unrestricted candidate ranking and the fixed feasible subset separately, report
how often references are excluded and execution falls back, and do not assign a
zero regret to an empty set. Loosening a threshold changes the search problem
and its support risk; these findings provide no authorization or validation for
that change. Finite output from a new head is a required validity check, not a
new opportunity to redefine feasibility.

## Conditional pure-return head: minimum useful experiment

Proceed only if authenticated posthoc ranking shows material candidate regret
or sign errors persisting when both stage rewards and endpoints are replaced
by their real counterparts, with a nontrivial true candidate spread. That
replacement leaves `C_old_real` as the remaining surrogate. Such evidence
supports investigating its suitability; it does not prove that target mismatch
alone caused the ranking errors or that learning a new head will resolve them.
If only common offsets differ, stop after the analysis. If reward or generated
endpoint substitutions resolve the ranking, pursue that explanation instead.

If the trigger holds, use one new scalar observation-value head per existing
task × world checkpoint for frozen actor 541: nine heads total for the current
three tasks and three world seeds. Share the head within a checkpoint across
recursive/direct endpoint families. Its target is

```text
V_pi(o, tau) = E[sum_{k=0}^{tau-1} gamma^k product_{j<k} c_j r_k
                 | current observation o, remaining native steps tau,
                   frozen actor pi thereafter].
V_pi(o, 0) = 0.
```

Use `(standardized observation, tau/native_episode_limit)` as input. Observation
normalization must be fitted on the new training partition only. Remaining
time is necessary unless it is already encoded injectively in the observation
or an explicit audit shows it irrelevant over the queried support. With partial
observability this is a conditional predictor, not a proof that observation and
time are Markov. Do not invent a full-state input unavailable to the planner.

The immediate minimum experiment can reuse the authenticated diagnostic traces:
fit only calibration environment 76001 and evaluate only environment 76003,
with no new simulator or controller runs. Use the
[fixed pilot protocol](planning_value_pilot_protocol.md): two-layer
128-unit ReLU MLP, one initialization, Adam at `3e-4`, batch size 256, and exactly
2,000 updates (or fail early on nonfinite training). Fit MSE on scaled return
targets. Fit normalization and one positive target scale only on calibration
data; invert that scale for scoring. Fix all choices before training, with no
validation-based stopping, model selection, or subsequent sweep. The architecture
is already flexible relative to one calibration trajectory, so low training loss
alone says little about useful generalization.

Environment 76003 has already been inspected for the old critic's diagnosis.
It is held out from **fitting**, but is not a blind or independent confirmation
set for a posthoc head motivated by that diagnosis. Call this an exploratory
reused-trace pilot. The time-only training baseline reveals whether the head
learns useful action-sensitive state information; a constant training-mean
baseline is another useful scale control.
An intercept-only old-Q calibration may be reported as a scale diagnostic but
cannot improve within-snapshot ranking under common weights.

Data construction and evaluation must satisfy all of the following:

1. Enforce environment 76001 for all training observations, labels, normalizers,
   and scales, and 76003 for evaluation only. All snapshots, descendants,
   duplicate plans, and trajectory suffixes of an episode stay together. Do
   not split transitions or branches randomly. Do not import old controller
   evaluation labels. Keep the existing candidate-generation recipe, original
   residual limit, and authenticated candidate bank unchanged.
2. Use the saved frozen-actor baseline observations with exact MC suffix labels,
   plus saved real five-step endpoints of calibration reference, proposal, and
   directional plans from both existing endpoint families, restricted to the
   selected latent reference mode in the fixed pilot. This gives at least
   some support for endpoints beyond B0 occupancy. Deduplicate identical action
   sequences within a matched snapshot (including executed/reference duplicates),
   and duplicated baseline endpoints; check duplicate inputs for target conflicts.
   Equal total loss weight for the baseline-suffix pool and the unique
   candidate-endpoint pool prevents 1,000 highly correlated baseline labels from
   overwhelming the scarce candidate endpoints. Within each pool use fixed
   equal weights. Save the weights; this defines a deliberate training support
   mixture, not an empirical occupancy estimate.
3. The head's target at the endpoint is obtained by following the **frozen actor
   from that real endpoint to the native end**, with no bootstrap from the old
   critic or learned reward. Reconstruct from saved rewards/continuations and
   authenticate absolute episode step, native end, policy/checkpoint identity,
   and episode provenance against the original diagnostic receipt.
   The resulting endpoint-relative label equals the diagnostic's discounted
   tail divided by `gamma^5` only when the full plan survives with continuation
   product one. Ended episodes have zero tail; otherwise handle that product
   explicitly. In principle suffix states after the branch are valid extra labels
   because they too follow the actor, but these traces save full observations
   only for the baseline and the first five plan steps. Branch suffix rewards
   alone do not supply the missing state inputs. The first five fixed-plan
   states are not actor-policy MC labels merely because the actor resumes later.
4. Do not label behavior-replay MC returns as `V_pi`: they follow the behavior
   policy. Do not label the entire fixed-plan return as `V_pi` at its initial
   state: it conditions on that plan. A plan-conditioned value predictor would
   be a different estimand and design. Learned-model rollouts are also not real
   raw-return targets for this diagnostic.
5. Limit this pilot to the existing nine traces, actor 541, one fit per head,
   and the frozen 2,000-update budget. No new environment episodes, actor runs,
   candidate search, or controller study is part of this test. If a task has no
   positive-reward events or insufficient candidate variation, report that
   head's decision result as inconclusive rather than call zero MSE successful.
   Count positive rewards and nonzero continuation returns by partition/task;
   report conditional error on positive-return cases. Do not modify sampling
   after examining evaluation errors or oversample evaluation reward events.
6. Freeze all old actors, both old critics and their targets, world/endpoint
   models, reward head, replay, thresholds, reference modes, candidates, and
   evaluation traces. The optimizer owns only the new head and its state. Hash
   all accessed old artifacts before/after fitting; save the new head in a
   separate path and verify the optimizer tree cannot update an old parameter.
   For original checkpoints never opened by a trace-only pilot, bind their
   authenticated receipt identities rather than claim a fresh runtime digest.
   Evaluation data must not fit weights, normalizers, offsets, thresholds,
   hyperparameters, or a choice between endpoint families.

On the existing fitting-held-out matched snapshots, first evaluate
`R_real + gamma^5 V_new` at
**real** endpoints against the old-Q score at those same endpoints. This tests
the head's decision relevance with endpoint and reward error held out of the
comparison. Generated terminal particles were not stored in these traces, so
the trace-only pilot cannot test the head at generated endpoints. Means or an
old scalar Q do not recover them. Record that transfer as unmeasured; a separate
frozen-model check would be needed before claiming usefulness inside planning.
Recompute no candidate plan or feasibility mask for the head comparison.
Report centered pairwise error/sign accuracy (with explicit ties), selected-plan
real gain and regret, positive-event support, calibration, empty sets, and
duplicate plans. A lower uncentered MAE alone does not pass this diagnostic.

Use episode-level matched comparisons, then preserve the nested task/checkpoint
structure when aggregating. There is only **one training root episode and one
evaluation root episode per checkpoint**. Hundreds of suffix labels and branches
do not create hundreds of independent observations. The nine already-seen
checkpoints are not nine independent new model replications; fitting a head
separately for each is not held-out-model generalization. Do not use one
checkpoint's evaluation labels to select settings for another. Report checkpoint
and task results transparently, with no confidence claim based on resampling
individual suffixes, snapshots, branches, or candidate pairs as independent units.

Freeze the descriptive success rule before fitting: the pilot protocol requires
improvement in both candidate-relative gain MAE and decision regret over the
real-endpoint critic, with at least two of three world seeds improving each,
separately by task and endpoint family. Report positive-event support and the
time-only baseline so trivial time dependence is visible. If the fixed head fails, is nonfinite,
or lacks reward/support evidence, retain that negative or inconclusive outcome;
do not select a second head using environment 76003. This pilot cannot establish
that its motivation or architecture generalizes beyond the reused diagnostic.

Only a separately scoped confirmation could use fresh explicit environment
train/validation/test lists, all disjoint from original replay, old diagnostics,
old controller evaluations and one another, with whole root episodes and their
branches kept together. That future design would choose stopping/hyperparameters
on validation before unlocking test and would need a prospectively held-out
model seed for unseen-model claims. A successful reused-trace head does not
authorize that expansion or step 3's closed-loop controller study.

## Reproduction and review boundary

```sh
PYTHONPATH=dreamer_imf_comparison:imf_dreamer_jax/src \
  /private/tmp/imf-replication-venv/bin/python -m pytest -q \
  dreamer_imf_comparison/tests/test_planning_value_semantics.py
```

The tests run on CPU without study checkpoints or simulator launches. They prove
the listed implementation semantics and analytic decision examples, not empirical
critic quality or a successful planning head. The parent independently reviews
this derivation and integrates it with authenticated candidate ranking before
deciding whether the conditional experiment's trigger is met.
