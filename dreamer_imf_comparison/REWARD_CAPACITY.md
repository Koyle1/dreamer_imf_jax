# Frozen reward-readout capacity sweep

This is an offline readout experiment, not a change to the agent or a larger
world model. It consumes only the authenticated retained inputs of diagnostic
e1fc21d (report SHA-256 f1b7e9af472856c069a8e4ea7d2c29b2d6204fdf487d5c2933dd32a09bc4b689).
Existing artifacts remain immutable. No simulator, collection, or policy calls.

## Fixed protocol

Hidden widths 64, 128, 512; exactly two ReLU layers and three reward classes.
Five input variants: observation, observation/history/action, h, z, hz.
Three initialization seeds 0/1/2. All 45 cells are fresh fits, including the
128-width reference. Initialization follows the existing He-normal convention;
the 128-width initializer is exactly the previous initializer. Input dimensions
differ: comparisons are within input variant, with actual parameter counts.

Unchanged data and 40/12/12 episode splits, Adam 3e-4, batch 256, 5,000 updates,
minibatch seed 6211 reset identically for every cell, categorical cross entropy,
train-only normalization with .01 standard-deviation floor, no extra regularizer.
Every 250 updates retain weights and full training/validation probabilities.
Select each cell by validation NLL, resolving ties toward earlier updates.
Test predictions are produced only after fitting, at the validation-selected
checkpoint and at update 5,000. No selection between capacities using test data.

## Outputs and interpretation

Retain training/validation curves, gradient norms, parameter counts and runtimes.
Report raw per-step MSE, categorical NLL/Brier, calibration, labels 0/1/2,
15-step cumulative error and endpoint-distance strata. Report both selected and
final checkpoints on train/validation/test to expose overfitting or unfinished
optimization. Aggregate errors across initialization seeds, not predictions.
Use paired episode bootstrap (2,000 replicates) for width-versus-128 contrasts,
and latent-versus-observation gaps at the same width. Keep all per-seed results.

The exploratory practical-improvement screen is >=10% lower test MSE than width
128, a paired 95% delta interval below zero, and >=5 positive test episodes.
Failure to meet it is not equivalence. Training improvement without a supported
test improvement is compatible with overfitting or uncertainty, not proof of a
specific cause. A persistent observation/latent gap cannot prove information
absence, nor that increasing backbone capacity would fix it. Equal updates are
not equal FLOPs or guaranteed equal convergence. These are reused test episodes
from one frozen world-model seed; intervals are pointwise, not multiplicity
adjusted, and no online-return claim is allowed.

## Execution

One A30, four CPUs, 64 GB, two-hour cap. Preflight exercises all widths and inputs
with one seed, two updates and one episode per split. A separate verifier must
replay all candidate train/validation predictions and selected/final predictions
before full fitting begins. Full verification repeats those checks and metrics,
authenticates all input/output hashes and emits REWARD_CAPACITY_VERIFIED.
Submission is exclusive, with immutable intent and receipt; no automatic retry.
