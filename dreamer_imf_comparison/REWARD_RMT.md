# RMT informed reward generalization diagnosis

This isolated offline study distinguishes early overfitting, unstable feature
directions, posterior sampling, and reward timing. It does not enlarge the world
model, alter training or policy, collect observations, or repair production
weights. The existing 40/12/12 episode splits and retained action branches stay
fixed. Previously inspected test episodes remain exploratory.

## Input and execution boundary

The dependency is the authenticated full reward-capacity study at source
`f0b1be7f04e2fb1fcf2ecda58d68d6bed70dcad3`, whose report SHA-256 is
`890b613ffcf49c143b9b760c711e38634f7461625d6fcda0542d55b5d6c31b16`.
The harness follows its immutable bindings through the reward diagnosis to the
dataset, raw episode histories, original inference seeds, and frozen checkpoint.
It does not select a new world-model checkpoint.

Every stage checks exact source identity, train-only transforms, finite outputs,
and input hashes. Output directories and artifacts are exclusive-create. A
failure preserves partial evidence and stops; there is no automatic retry,
overwrite, quarantine, or reuse of failed outputs. Submission ambiguity records
an immutable `submission-uncertain.json` and requires scheduler inspection.

Local unit tests use synthetic fixtures and the CPU. They do not establish exact
replay of the retained checkpoint on the GPU. In particular, returning additional
posterior statistics can change JAX lowering. The mandatory GPU preflight compares
the original carry and sampled features against the read-only statistics path
exactly. Any difference, including one ULP, stops before fitting. The extraction
uses the canonical carry for every subsequent step and never resamples to obtain
a favorable match. Mean/std outputs are recomputed probes, not historically
logged statistics.

## Registered comparisons

| Component | Fixed protocol |
|---|---|
| Summed reward MLP | observation, h, hz, h plus posterior mean; two width-128 ReLU layers; seeds 0/1/2 |
| Optimization | Adam, learning rate 0.0003, batch 256, original minibatch seed 6211, 5000 updates |
| Explicit penalty | alpha/2 times sum of squared weight matrices; alpha 0, 1e-6, 1e-4, 1e-2; biases unpenalized |
| Candidates | update 0, every 10 through 250, then every 250 through 5000 |
| Selection | each seed's earliest minimum validation NLL; choose alpha by mean selected validation NLL across seeds; lower alpha wins ties |
| Ridge | all five original input groups; train-standardized, intercept unpenalized; mean squared training error plus alpha times squared coefficient norm |
| PCA and ridge grid | ranks 16/64/128 when available and full; alpha 1e-6 through 1e2 in powers of ten; raw validation MSE selection |
| Ridge ties and outputs | smaller rank, then larger alpha; both unconstrained and predefined [0,2]-clipped predictions; no categorical NLL/Brier |
| Endpoint classifiers | observation, h, hz; binary targets; alpha inherited from each input's summed-reward family, checkpoint selected using endpoint validation NLL |

There are 48 summed-reward MLP fits, nine endpoint fits and five ridge grids.
The normalization standard-deviation floor is 0.01. No statistics, projection,
checkpoint, penalty, or score threshold is fitted on test episodes.

Three fresh matched comparisons separate effects: dense early selection versus
the original 250-update schedule; regularization versus the dense unregularized
baseline; and their combined change versus coarse selection. The final checkpoint
is also retained. Errors are averaged over readout seeds; predictions are not
ensembled. Posterior-mean versus sampled-state comparisons are reported separately.

Endpoint occupancy is `distance <= radius`. The summed reward target remains the
sum of two native rewards. A reward of one permits either endpoint label, and
the missing intermediate native observation is never reconstructed. Endpoint
cumulative error describes the number of occupied endpoints, not native return.

## Spectral and gradient diagnostics

The five original feature groups receive raw and train-standardized covariance
spectra, effective ranks, spectral reward alignment and cross-episode alignment.
Mode-stratified whole-episode label permutations preserve within-episode
correlation. Episode bootstrap stability measures variance and reward correlation
in fixed training-fitted high- and low-variance PCA directions. These are not
confidence intervals obtained by refitting PCA spectra on every bootstrap.

Seed-zero, unregularized observation/h/hz MLPs are inspected at initialization,
250 and 5000 updates. The same 256 training examples are selected once with
episode counts differing by at most one; branch and horizon identities are
retained. The empirical kernel traces all three class-logit Jacobians. These
are recomputed readout diagnostics, not historical world-model gradients or an
infinite-width approximation.

GradPCA uses gradients of the maximum logit and centered class means. This is
an explicit max-logit variant, not the paper's default sum-of-logits scalar.
The reported projection ratio is higher for greater class-subspace alignment;
its negative is correlated with held-out squared prediction error. The subspace
has rank at most two with three classes, and actual rank and degeneracy are
retained. The comparison baseline is nearest training-example RMS distance in
train-standardized features. Both receive episode-bootstrap association intervals.
Neither score is called calibrated uncertainty, nor is every error called OOD.

Full-data N-by-N kernels and N-by-parameter Jacobians are avoided. Feature
covariances are D-by-D; readout gradients and nearest-neighbor distances are
streamed in bounded batches.

## Reporting and decision rule

Raw predictions, all candidates and transforms, diagnostic arrays, selections,
and reports are sealed. The separate verification process reconstructs posterior
statistics, replays every retained MLP candidate, recomputes every ridge candidate,
rechecks validation selections and diagnostics, and reproduces predictions and
metrics exactly. It checks final artifact inventory and frozen-parameter hashes
again. Verification does not claim to retrain all optimizers from scratch.

Reports include per-seed and seed-averaged raw reward MSE, categorical NLL and
Brier scores where applicable, calibration bins, cumulative error at horizons
1/3/5/10/15, and train-validation gaps. Errors are stratified by collection mode,
action plan, reward class and endpoint distance. Uncertainty resamples episodes,
not branches, transitions or particles. Intervals are pointwise without multiple
comparison correction. There is one frozen world-model seed.

The exploratory screen requires at least 10% held-out MSE reduction and a paired
95% interval entirely below zero, with at least five reward-positive test episodes.
To operationalize the plan's unspecified “material deterioration,” the protocol
fixes a 10% relative margin for NLL and cumulative MSE before execution: the upper
paired interval must not exceed 10% of the matched baseline metric. This rule
screens mechanisms; it does not prove equivalence, superiority or causal
attribution. Ridge lacks categorical NLL and is not silently granted this screen.

## Running after an approved deployment

Use a clean exact source commit on main, a bundle containing that commit, an
immutable detached checkout and a fresh output root outside source/dependencies.
The canonical submitter checks source, bundle and dependency bindings before
creating an immutable intent. Its job envelope is one A30, four CPUs, 64 GB,
and four hours; the allocation limit is not a completion-time prediction.

```sh
python dreamer_imf_comparison/scripts/submit_reward_rmt.py \
  --source /path/to/immutable-source \
  --output /path/to/fresh-reward-rmt-output \
  --dependency /path/to/reward-capacity-full \
  --expected-commit FULL_COMMIT \
  --bundle /path/to/source.bundle \
  --bundle-sha256 FULL_SHA256
```

The batch script invokes separate processes for `prepare`, `fit`, `evaluate`,
and `verify`, first for a small GPU preflight and only then for the full study.
It prints `REWARD_RMT_PREFLIGHT_VERIFIED` only after preflight verification, and
`REWARD_RMT_ALL_STAGES_VERIFIED` only after full verification. Require Slurm
COMPLETED with exit 0:0 in addition to the final token. Queueing, preparation,
fitting, verification and total allocation time are distinct costs.

Do not submit from a dirty development checkout or treat local fixtures as a
scientific result. Existing world-model training and previous studies remain
untouched.

## Local acceptance

```sh
JAX_PLATFORMS=cpu PYTHONPATH=dreamer_imf_comparison:imf_dreamer_jax/src \
  ../.venv-dreamerv3-ref/bin/python -m unittest discover \
  -s dreamer_imf_comparison/tests -p 'test_reward_rmt*.py'
```

Acceptance covers analytical ridge/intercept checks, high- and low-variance
reward signals, exact unregularized optimizer parity on a fixture, bias-free L2,
early selection, split isolation, correlated/shuffled-label controls, kernel
agreement with autodiff, GradPCA rank/degeneracy, posterior draw/carry consistency,
AR2 indexing, and a tiny end-to-end report with independent replay. Tampering,
unregistered artifacts and ambiguous submissions must fail closed.
