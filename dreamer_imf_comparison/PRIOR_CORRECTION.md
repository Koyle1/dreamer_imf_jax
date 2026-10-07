# Frozen learned-prior iMF correction

This implements the three agreed safeguards without changing the live agent,
checkpoint format, actor, reward head, representation, or existing iMF sampler.
It is an offline diagnostic, not an online-learning repair or a Dreamer ablation.

## 1. Identity initialization and bypass

At each transition the frozen recurrent core computes h from the preceding
belief and chosen action. The existing learned auxiliary Gaussian becomes the
source: b = mu(h) + sigma(h) epsilon. A separate two-hidden-layer, width-256 iMF
network predicts a displacement. Its joint u/v output projection starts at zero.
The deployed corrected state is the physical base plus the standardized flow
displacement. This avoids a normalization-roundtrip error at identity. An
explicit bypass omits the correction entirely. The original agent is unmodified.

Identity is an available predictor, not a theorem of non-degradation. The
validation candidate list includes update zero, with ties selecting the earlier
checkpoint. This does not guarantee test performance or online stability.

## 2. Real targets, frozen source

Only correction parameters enter AdamW. The source mean/std, conditioning and
real posterior targets are all stopped at the loss boundary. For fixed h:

    x_t = (1-t) z_posterior + t b
    target_velocity = b - z_posterior
    V = u + (t-r) stop_gradient(JVP(u; v, 0, 1))

Source noise and retained posterior target samples are independently coupled
conditional on h. Boundary supervision uses v(x_t,h,t,t). No endpoint-MSE,
teacher-imagined targets, reward penalty, or joint representation updates are
introduced. A learned Gaussian source changes the transport endpoints, not the
MeanFlow identity. Frozen conditional source gradients are intentionally zero.
This is not a tractable KL to the corrected implicit distribution.

The dataset is the authenticated parallel-trajectory study's retained real
branches; the parent is the same repaired 200k checkpoint. Whole-episode splits
are preserved. Only training targets determine normalization (std floor .01).
Future posterior observations never enter free-rollout inputs. The recurrent
component of teacher-forced targets depends only on prior belief and action.

One correction initialization seed (811), 2,000 updates, batch 256, AdamW
3e-4 / weight decay 1e-4, global clipping 1. Validation every 250 updates, using
8 fixed particles and all validation branches, selects normalized observation
MSE plus normalized 15-step cumulative reward MSE. Test uses 32 particles.
There are **zero additional simulator steps**. No automatic hyperparameter sweep.

## 3. Evaluate the final samples

Compare Gaussian bypass, corrected one-step, corrected four-step and the
original four-step iMF sampler on identical histories/actions. Real-posterior
head decoding exposes errors present before imagination. At horizons 1/3/5/10/15,
report decoded observation/reward/cumulative-return errors, CRPS, 90% interval
coverage, positive/zero reward coverage, paired plan effects and action ranking.
Uncertainty is clustered by episode, not particle or anchor. The shared metric
implementation marks insufficient positive-reward coverage as inconclusive.

Also score the final continuous latent distribution at the first teacher-forced
transition using an unbiased sample energy score, with train-only coordinate
scaling and episode-level results. This score uses transformed samples, not a
Gaussian-prior KL. It is not a density estimate or a measure of actor return.
Multi-step free rollout errors measure drift; neither low one-step error nor
identity availability proves control stability. These previously inspected test
episodes provide exploratory evidence only, not fresh confirmation.

## Execution and evidence

Run from a clean exact checkout with the existing GPU Dreamer/JAX environment:

    python -m dreamer_imf_compare.prior_correction_study run --parent <authenticated-parallel-study> --output <fresh-directory>
    python -m dreamer_imf_compare.prior_correction_study verify --output <same-directory>

The run refuses an existing output directory, authenticates the parent with its
original verifier, and records source commit, dataset hash, frozen parameter
digest, protocol, selected checkpoints, predictions and file hashes. A separate
process must exactly replay predictions and independently recompute metrics
before `PRIOR_CORRECTION_DIAGNOSTIC_VERIFIED` is printed. Creation alone is not
verification. Parent arrays are hashed before/after; no parent optimizer exists.

Tests cover identity, bypass, gradient routing, analytical translated Gaussian
transport, conditional synthetic learning, split-leak negative controls,
causal rollout prefixes, retained-input immutability, selection and replay.
Synthetic results establish implementation behavior, not Reacher improvements.

This change does not fix reward-head error on real features, establish an
alignment objective for joint online learning, or promise faster rollouts.
The new path adds prior computation to flow computation and must be measured
before any efficiency claim. A favorable diagnostic would justify a separately
specified online experiment, not automatic deployment into the active agent.
