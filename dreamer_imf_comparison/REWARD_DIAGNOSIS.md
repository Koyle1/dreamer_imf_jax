# Matched frozen-feature reward diagnosis

An isolated offline extension of the retained trajectory and learned-prior
diagnostics. It does not change agent training, reward heads, policy, checkpoint
formats, or any existing study. There are zero simulator interactions.

## Fixed comparison

Refit five input variants: current observation; current plus previous observation
and executed action; h; z; and (h,z). Each uses three initializations (0/1/2),
two 128-unit ReLU layers, a three-class softmax for summed rewards 0/1/2, Adam
at 3e-4, batch 256, 5,000 updates and validation every 250 updates. The exact
minibatch stream is shared across variants and initializations. Only weights
differ. Normalization is fit on training episodes only, with a .01 std floor.
Validation NLL selects each seed's checkpoint with earlier-update tie breaking.
No test-based selection of heads, seeds, or checkpoints is permitted. Input
widths differ, so parameter counts are reported rather than claiming equal size.

Authenticate the original retained study and the verified correction from
commit d26e93c, update 500. Use its frozen 200k parent. Preserve complete episode
splits and all paired action plans. Observation input excludes reward and reset
metadata. For the first branch transition, previous observation is the anchor;
later transitions use the preceding retained observation. Check actions, reward
indexing, posterior features, reset masks and raw-file hashes before fitting.

## Three diagnoses

1. Compare observation and history/action references with latent readouts and
   the original reward expectation, constant zero, and training-class frequencies.
2. Report raw MSE, NLL, multiclass Brier score, ten fixed calibration bins,
   false-positive reward predictions at P(R>0)>=.5, reward labels 0/1/2, and
   endpoint distance/radius bins <.8, [.8,1], (1,1.2], >1.2. Missing classes or
   strata retain zero counts and null estimates. Original scalar expectations
   have no invented categorical probabilities. Assets are hashed, and a static
   physics model (no environment, reset, or stepping) supplies the finger radius;
   the hard-task target radius comes from the installed task. Check compatibility
   with retained zero/two rewards. Reward one can lie on either side of the
   endpoint boundary: the native intermediate observation was not retained.
3. Apply the SAME selected h/z/hz readouts to real posterior features, Gaussian
   prior rollouts, corrected one-evaluation rollouts, and existing four-evaluation
   iMF rollouts. Use 32 particles, identical histories/action plans, and recorded
   deterministic seeds. Observation readouts are never fed decoded observations.
   Report horizons 1/3/5/10/15, cumulative rewards and paired plan effects/ranking.

Repeat full real-history stochastic inference eight times for ALL test episodes,
sharing anchor beliefs across branches. Measure individual-draw and averaged-draw
errors, prediction variance and cumulative-reward variance. This is full inference
variability, not an isolated current-step sampling effect. Draw seeds are derived
from retained inference seeds, draw index, and fixed salt 9127.

Bootstrap 2,000 times by episode. Report all seeds and average ERRORS across seeds,
not the error of an ensemble. The primary practical-improvement flag requires at
least 10% lower real-posterior raw MSE than the original head, a paired 95% interval
below zero, and at least five positive-reward test episodes. This is exploratory
and pointwise, not confirmation or a causal decomposition. Failure of a modest
readout does not prove missing information. Report calibration and cumulative
error even when the primary flag is favorable.

## Execution and acceptance

CLI: `python -m dreamer_imf_compare.reward_diagnosis` with stages `preflight`,
`fit`, `evaluate`, `verify`; all take `--output`. Initial stages additionally
take `--parent` and `--correction` (the verified correction's `fit` directory).

Preflight fits all five inputs for two updates with one seed and one episode per
split, then exercises all transfer paths and one posterior draw. Separate-process
verification must pass before the full suite starts. Retain all candidate weights,
validation probabilities/selections, input metadata, raw predictions/draws,
feature hashes, environment/source bindings and final report. Independent replay
recomputes every validation candidate, selected predictions and all reported
metrics exactly, checks frozen hashes, then prints `MATCHED_REWARD_DIAGNOSIS_VERIFIED`.

The canonical submitter creates an exclusive output reservation and immutable
intent before contacting Slurm. Existing outputs and ambiguous submissions cannot
be resubmitted automatically. One A30 GPU, 64 GB host RAM, four CPUs, two-hour cap.
The shell driver advances preflight -> verify -> fit -> evaluate -> verify with
fail-closed exits. Corrupted evidence is preserved, not overwritten or accepted.

This is a single frozen world-model seed and previously inspected test episodes.
Readout seeds, branches and particles are not independent environment/model
replications. Even a verified favorable result makes no online-return claim.
