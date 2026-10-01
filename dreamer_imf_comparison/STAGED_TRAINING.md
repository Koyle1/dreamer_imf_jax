# Staged scratch training

This is an exploratory bundled intervention, not a claim that iMF needs or
benefits from every intervention. The completed three-arm ablation is untouched.
Run three paired Gaussian/iMF training seeds (431,433,439) on Reacher Hard, each
from fresh weights and empty replay, with 500,000 native interactions, action
repeat two, and five nested evaluation episodes at each 100k checkpoint.

## Registered interventions

1. Every 50k interactions, evaluate a fixed initial replay batch (2 sequences,
   16 observations), with common noise. Record raw time-binned velocity errors,
   conditional posterior/prior distribution distances, one/four-step differences,
   recorded-action rollout errors, reward error, latent drift and shared-core
   loss-gradient norms. The diagnostics do not modify learner parameters or RNG.
2. First 50k interactions train the world model only. The initial stochastic
   policy collects data; actor/critic parameters, optimizer clocks/momenta,
   slow value targets and return normalizers do not learn. No pretrained weights,
   externally generated replay, or previous-run checkpoints initialize anything.
3. After warmup, each 50k cycle has a 10k representation-learning block followed
   by a 40k transition-only world-model block. Encoder, posterior, recurrent core,
   decoders, reward and continuation parameters and optimizer states are frozen
   during the latter. The actor still learns after warmup. This is shared by both
   arms. Freeze evidence is recorded at phase changes; behavioral tests additionally
   verify inactive momentum, weight decay, clocks and recurrent-core parameters.
4. iMF transport intervals are capped at 0.1,0.25,0.5,1.0 at native-step boundaries
   0,50k,100k,150k. Boundary velocity supervision is retained. Gaussian has no
   analogous transport-time objective.
5. Initially use four-step iMF sampling and a five-step bootstrapped actor
   objective. At or after 150k, require two distinct consecutive diagnostic
   checkpoints with candidate one-step normalized five-step latent error <=0.5
   and reward error <=1.0 before using one-step sampling and horizon15. A failing
   checkpoint returns to conservative settings. Gaussian uses the same horizon
   gate; its sampling distribution does not depend on sampler steps. Evaluation
   returns never enter this rule. Calibrate dynamics weight using the ratio of
   observation-plus-reward gradient norm to unscaled dynamics gradient norm on
   the shared recurrent core; clip the target to [0.1,10], smooth halfway toward
   it, and retain the previous value if either norm is effectively zero.

## Interpretation and limitations

The noise-time coordinate in the transport objective is not environment time.
A generator on a fixed image/video dataset ordinarily has a more stationary
target distribution than a jointly learned online latent world model. Freezing
latent coordinates for blocks addresses one possible mismatch; it cannot prove
that this was the cause of the earlier failure. Correct image generation also
does not guarantee accurate action-conditioned multi-step control predictions.

The diagnostic batch is deliberately fixed to make drift measurable, but is
small and drawn early: it does not certify later policy-state coverage. Its
reward normalization uses variance floored at one, which can be permissive for
sparse rewards. Thresholds are operational heuristics, not validated guarantees.
Longer imagined trajectories are still computed for a fixed compiled shape;
only the first five steps and their correct value bootstrap enter the short
actor objective. Thus a short horizon does not imply proportional runtime savings.
Model warmup reduces actor updates within the fixed interaction budget.

Compare the new paired arms first. Historical returns may be reported as context,
not as a controlled estimate of any individual change. Three training seeds are
the independent units, not fifteen evaluation episodes. Negative results must be
reported without promoting a favorable seed. A successful bundled experiment
would motivate, not substitute for, later individually authorized ablations.

## Execution

Use `python -m dreamer_imf_compare.staged_study register --root NEW_ROOT
--upstream PINNED_UPSTREAM`, then `submit --stage preflight`. Preflight contains
two independent full-geometry scratch cells, each exercising eight updates across
all phases, real environment/replay, diagnostics, evaluation and checkpointing.
Only after both scheduler tasks exit zero and authenticate may `submit --stage
training` create the six production cells. Each cell and submission is exclusive;
existing or partial evidence is never overwritten. Finalization requires six
successful Slurm tasks and digest-checked checkpoints/diagnostics. This is artifact
authentication, not a claim of independent deterministic retraining.
