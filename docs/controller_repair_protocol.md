# Frozen-checkpoint controller repair

This follow-up implements the five audit recommendations. It does not reuse old
evaluation outcomes as controls, retrain the world model, or claim that software
repairs alone resolve prediction-to-control failure. The JSON protocol is frozen
with the source commit before GPU execution.

1. Compile A3, rank feasibility without a large floating-point sentinel, bind
   executed configuration, and make artifact authentication fail closed.
2. At matched real states from frozen-ReBRAC occupancy, restore the simulator
   before each candidate/reference plan. Separate H=5 rewards, terminal critic,
   real frozen-policy continuation, feasibility, and the action actually used.
   Preserve rejected proposals and finite-change direction ladders.
3. Measure reference feasibility and run a fresh frozen-ReBRAC-only baseline.
4. Test the same-endpoint reference as a narrowly defined interface repair. Use
   the JSON's calibration and held-out validation thresholds, with one mode per
   endpoint family across all tasks. If evidence fails, retain the bug-fixed
   latent reference and explicitly report that the mechanism remains unresolved.
   Reward/critic/data retraining is not silently substituted: it needs a bounded
   protocol justified by the diagnostic evidence and a new source/output identity.
5. Evaluate the four repaired controller arms against the same frozen baseline
   on three reserved environment seeds, both nested actors, all three world-model
   seeds and all three tasks: 90 cells / 270 episodes, exact independent replay.

The study is exploratory. Actor and episode repetitions are nested. Primary
effects are equal-task means of within-task IQMs of paired world-seed differences,
with 98.75% per-contrast cluster-bootstrap intervals for four comparisons. A
positive interval would motivate independent confirmation, not establish a
universal solution. Feasibility improvement alone is not return improvement.

Every stage authenticates source, protocol, checkpoint hashes, exact traces,
process identity, runtime, cache seals and successful scheduler completion before
the next stage is submitted. Previous studies remain immutable. First execution
is discarded as a cache primer; creation and replay are separate cache readers.
No equality tolerance is relaxed. An uncertain submission, existing cell, failed
replay or changed cache blocks automatic progress rather than overwriting data.

Runtime comprises Slurm allocation time and per-role wall time separately from
post-warmup controller latency. Queue time is not a compute estimate.
