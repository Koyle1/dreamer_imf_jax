# Corrected matched actor training

This diagnostic reruns every actor cell in the authenticated 606-cell pilot.
It deliberately reuses the frozen dataset, compute-plan, world-model, and
rollout artifacts, but never reads the old actor outcomes. Each of the 288
actors and critics starts fresh, receives 500 paired behavior-cloning and
replay-critic preparation updates, and then receives 10,000 paired
percentile-EMA normalized REINFORCE updates with behavior KL beta zero.

The complete grid must be rerun because actor return is part of the frozen HPO
rank-sum rule. Rerunning only the previously selected configurations would
retain selection bias from the invalid historical PMPO actor.

The final result remains diagnostic: it reuses pilot/HPO tasks and world
models and cannot support a confirmatory superiority claim. Its purpose is to
answer whether the corrected actor changes candidate selection and whether
trajectory-iMF yields better actor returns than shortcut forcing under this
matched pilot setup.

All source dependencies, schedules, checkpoints, environment traces, and
strict replay markers are hash-bound. A source world-model mutation, old actor
result, missing cell, non-finite metric, or trace mismatch invalidates the run.
