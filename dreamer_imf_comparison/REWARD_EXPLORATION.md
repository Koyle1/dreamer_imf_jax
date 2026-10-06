# Reward integration and uncertainty-guided collection pilot

This diagnostic continues the repaired **200k-native-step iMF checkpoint, world
seed431**. It is not a new from-scratch benchmark or evidence from three world
model seeds. All historical runs and the earlier 96,710-step diagnostic budget
remain untouched.

## What is implemented

1. **Matched-data reward comparison.** Both the original reward-head architecture
   and a two-hidden-layer128-unit categorical `{0,1,2}` readout are freshly fitted
   on exactly the same retained training episodes, minibatches, and5,000-update
   budget. Three initializations each; validation chooses checkpoint/seed, never
   test. The original head retains its original preprocessing, objective, and
   expected-raw-reward decoding. This compares readout packages, not architecture
   in isolation. The previously inspected test split makes this exploratory.
2. **Online integration.** The original reward head and its gradients into the
   representation remain intact. A separate control readout learns from stopped
   real posterior features and real AR2 rewards; the task actor uses its imagined
   reward predictions. Normalization is train-only and fixed during continuation;
   feature drift is logged instead of silently changing coordinates.
3. **Exploration pilot.** Five independently initialized and episode-bootstrapped
   MLPs predict normalized next real observation from current belief and executed
   action. Their predicted-mean variance supplies a capped, train-calibrated
   intrinsic reward to a separate imagined-policy actor/critic. This is inspired
   by [Plan2Explore](https://proceedings.mlr.press/v119/sekar20a.html), not an exact
   reproduction: observable targets, mixed task/exploration collection, and a
   pretrained iMF backbone are explicit adaptations. Particle variance is not
   used as epistemic uncertainty. Shared model bias remains a limitation.

## Four arms

| Arm | Control reward | Real collection |
|---|---|---|
| A | Matched-refitted original architecture | Task actor |
| B | New categorical readout | Task actor |
| C | New categorical readout | Task actor +20% uniform-action blocks |
| D | New categorical readout | Task actor +20% disagreement-actor blocks |

All arms train the same auxiliary ensemble and exploration actor/critic, even
when those modules are unused for collection. This matches update/compute clocks;
it is not a minimal-runtime deployment comparison. The exploration actor/value
and target are cloned identically from the parent across arms, while their new
optimizer states and reward normalization start fresh. The exploration value
therefore initially reflects task training rather than intrinsic returns.

Three paired continuation seeds701/702/703. Each cell collects80,000 native
steps; five held-out task-policy episodes at40k and80k add10,000 native steps.
The planned total is1,080,000, with hard ceilings100,000 per cell and1,200,000
overall. These ceilings include failures and cannot be reset by retrying.
GPU preflight and offline fitting spend **zero simulator steps**.

The original replay buffer was not retained. Every cell restarts replay equally,
retains all new raw transitions, and preserves all parent model/optimizer state.
The learner uses batch16,length64,train ratio512,AR2,16 collection environments.
Exploration uses one20-decision block in every five blocks, with identical C/D
schedules. Clipped/replaced actions are also written into the recurrent carry;
they are not merely substituted at the environment boundary.

## Evidence and execution

`reward_exploration_protocol.json` freezes design and dependency SHA-256 hashes.
The submitter requires exact clean source, authenticated parent/dataset/runtime,
write-once submission intents/receipts, scheduler success and independently
verified stage markers. Ambiguous, failed, running, pending and completed attempts
are never automatically resubmitted. Native charges are persisted before calls.

The cluster serves two architecture-specific Python builds under the same module
path. Both exact interpreter binary hashes/build strings are explicitly pinned;
all installed Python and CUDA package bytes must remain identical. The initial
`6e320d4` deployment failed this portability check before fitting or simulator use
and remains preserved. No numerical or replay-equality threshold was relaxed.

Sequence: tests → matched fit + independent replay → full GPU update preflight
for A and D →12 continuation cells at concurrency4 → independent final report.
CPU handoffs submit a next GPU stage only after the previous allocation completes
0:0 and all required markers authenticate. A scientifically negative result does
not fail verification; corrupt or incomplete evidence does.

Full-size preflight executes two genuine learner updates per readout architecture
on the retained real16x65 batch, with explicitly reconstructed zero replay carry
because historical carry entries were not retained. These are integration probes,
not recovered historical gradients. Every required parameter group must update
finitely. Gradient-routing tests additionally prove that original reward gradients
reach representation parameters and auxiliary objectives do not.

Each continuation keeps two checkpoints, raw evaluations, all real collection
transitions, episode IDs, policy proposals and executed actions, metrics,
periodic learner batches, synchronized update latency and wall time. Rewards used
for reported return are always environment rewards, never intrinsic bonuses.

Prediction diagnostics reuse held-out observations/actions without new simulator
calls. At four anchors in each of five episodes, eight15-decision iMF rollouts
measure posterior versus imagined reward error and observation error at1/3/5/10/15.
They are recomputed probes conditioned on recorded feedback-policy actions, **not
causal open-loop intervention evidence**. The independent process replays these
predictions exactly from checkpoints and raw inputs. A pure discarded warmup is
performed before collecting probe outputs. Fewer than five positive-reward probe
episodes makes reward-sensitive conclusions inconclusive.

The independent NumPy verifier reconstructs raw return, coverage, indexing,
budgets, exploration fractions, checkpoint clocks, horizon errors and paired
seed deltas. Primary contrast D−C isolates directed versus matched random
collection; B−A tests the matched reward-readout package; C−B tests random
exploration. Shared parent, one task, small episode counts and continuation-seed
dependence must accompany every result.

## Launch from an immutable checkout

Use `reward_exploration_protocol.initialize_manifest()` once with the new output,
exact deployed source, pinned parent cell and dataset. Then:

```sh
python dreamer_imf_comparison/scripts/submit_reward_exploration.py match \
  --output /path/to/fresh-output --continue-chain
```

No other study is authorized by this submitter. Any code repair requires a new
exact source contract; previous evidence and charged budgets remain preserved.
