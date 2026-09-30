# Dreamer dynamics ablation

This experiment answers two separate questions: what changes when Dreamer's
categorical stochastic representation is made continuous, and what changes
when the continuous Gaussian transition is replaced by iMF? It does not test
the earlier A1/ReBRAC controller.

The upstream implementation is `danijar/dreamerv3`, commit
`e3f02248693a79dc8b0ebd62c93683888ddaccfe`. Its source is unmodified. The extension
replaces only the RSSM class for the two continuous arms. The upstream encoder,
observation/reward/continuation heads, imagined actor-critic objective, return
normalization, optimizer, replay stream and policy action selection are used.

## Fixed protocol

- Reacher Hard, proprioceptive input, training entirely from scratch.
- Three paired independent training seeds: 431, 433, 439, in all three arms.
- 500,000 native simulator steps, action repeat 2, 16 environments.
- `size12m`, batch 16, sequence length 64, training ratio 512.
- Five separate held-out episodes at each of 100k, 200k, 300k, 400k and 500k
  native steps. The final checkpoint is the endpoint; no best-checkpoint selection.
- Reset observations enter replay but count as neither environment interactions
  nor training-ratio credit. Actual simulator transitions determine the budget.
- Evaluation uses the current parameters with the upstream stochastic eval
  policy. Its latent carry and random stream are separate, and collection state
  is restored afterwards. Evaluation steps do not consume training budget.

## Arms and interpretation

1. `categorical`: unchanged upstream RSSM with categorical dynamics and
   representation KL terms.
2. `gaussian`: diagonal-normal posterior and transition, same feature width and
   recurrent core. Dynamics uses detached posterior-sample negative log likelihood.
3. `imf`: the same continuous posterior/core, with a one-step iMF transition
   trained by compound velocity regression and boundary velocity supervision.

Both continuous arms use the same standard-normal posterior reference KL and
upstream free-nats threshold. This is not the categorical arm's learned-prior
representation KL. No tractable iMF density or KL is asserted. Consequently:

- Gaussian minus categorical measures a representation/objective bridge, not
  just distribution family.
- iMF minus Gaussian measures transition family **and its training objective**,
  conditional on the same continuous representation protocol.
- iMF minus categorical measures the total intervention.

Loss scales and parameter counts are not identical across transition families.
The `size12m` preset is held fixed, not a promise of exactly equal parameter
counts; measured model-only counts are saved. No hyperparameter tuning is done.

This is the released Dreamer reimplementation with explicit budget/size/ratio
overrides, not an exact reproduction of every paper setting. Its current default
proprioceptive preset differs from the historical paper configuration. The
paper's Reacher Hard score is contextual, not an interchangeable paired control;
the newly run categorical arm is the direct experimental control.

## Execution and evidence

`dreamer_ablation_study.py register` exclusively creates a fresh output root and
binds the exact clean source, upstream pin, protocol and complete installed
runtime package list. Preflight is three independent GPU cells, one per arm:
full production size, batch and sequence geometry, 4096 native steps and two
genuine learner updates, followed by held-out evaluation. Its checkpoints are
never loaded into production.

Only after all three preflight artifacts authenticate and all three scheduler
tasks complete with exit zero can `submit --stage training` launch nine fresh
cells. Submission intents prevent blind retry after an uncertain scheduler
response. Existing cell directories, including failed partial cells, are never
overwritten or automatically retried. Source changes require a new commit and
fresh output root.

Each cell records native/reset/replay/update counts, finite metrics and
parameters, JAX device identity, parameter counts, timed updates, held-out seed
returns and SHA-256 checkpoint bindings. Independent artifact verification
checks counts and hashes; it is **not** an independent semantic replay of a
500k-step stochastic training trajectory. Final statistical units are the three
paired training seeds, not the fifteen nested evaluation episodes. Three seeds
on one task provide exploratory evidence only.

Successful-cell wall time excludes queue waits and failed preflights; report
Slurm allocations and any failures separately when summarizing the result.
