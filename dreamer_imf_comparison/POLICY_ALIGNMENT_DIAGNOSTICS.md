# Policy-alignment diagnostics

This diagnostic suite tests why lower held-out rollout error may fail to improve
real-environment actors. It is supplementary development evidence and does not
change the frozen matched-objective protocol or its claim eligibility.

## Measurements

1. **Actor-visited model error.** Replays authenticated deterministic actor
   evaluation trajectories from their registered DMC seeds. At fixed visited
   states, it compares shared-noise predictive draws with the next real
   observation, reward, and continuation. Observation errors use each dataset
   seed's train-only standard deviation before pooling.
2. **Imagined-real alignment.** Across all actor cells belonging to each
   arm's selected pilot candidate, it reports Pearson and Spearman correlation
   between the final imagined-return metric and mean real episode return.
3. **Counterfactual action ranking.** At restored simulator snapshots, it
   applies a coherent offset to each action dimension of the retained 30-step
   open-loop plan. It compares model and simulator rankings, top-action
   agreement, tie rates, return spread, and simulator regret. Shared predictive
   noise isolates action effects.
4. **Gradient fidelity.** It differentiates the model's shared-noise expected
   30-step return with respect to the same coherent plan offset and compares
   that vector with central finite differences from exactly restored DMC
   snapshots. It reports cosine similarity, component sign agreement, norm
   ratio, each side's gradient norm, and hallucinated or missed nonzero rates.

## Isolation and interpretation

Diagnostics run from a separate source worktree and write to a sibling output
directory, never inside the frozen pilot root. Before and after snapshots bind
the frozen source commit, clean status, protocol, matrix, authenticated stage
markers, and a fixed prefix of completed actor result hashes.

The counterfactual metrics are fixed-plan tests, not full closed-loop policy
interventions. All six selected actor checkpoints per task and arm are included;
for each checkpoint, the retained evaluation episode with greatest real return
is chosen deterministically to make sparse-reward sensitivity measurable.
Simulator ties and zero finite differences are reported explicitly rather than
being assigned an arbitrary accuracy or cosine score.

Run execution only after pilot finalization has produced `hpo_selection.json`,
so the sampled checkpoints belong to the deterministically selected candidate
for each arm. The report retains per-cell NPZ evidence and aggregates both pilot
tasks and both arms.
