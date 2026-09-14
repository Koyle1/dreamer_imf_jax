# Frozen reward-head MTP study

This study isolates the reward-model failure identified by the Reacher audit.
The trained trajectory-iMF encoder, posterior, recurrence, prior, decoder, and
continuation head are frozen exactly. Only the reward head and its Adam moments
can change.

The design is a two-by-two factorial on world-model seeds 211 and 223:

| Arm | Reward target | Counterfactual advantage |
|---|---|---|
| `one_step_mse` | scalar offset 0 MSE | no |
| `dreamer4_twohot_mtp` | symexp/two-hot offsets 0–8 | no |
| `counterfactual_advantage` | scalar offset 0 MSE | yes |
| `twohot_mtp_advantage` | symexp/two-hot offsets 0–8 | yes |

For initialization fairness, every arm retains the source reward MLP's hidden
trunk but resets its output projection with the same seed. Every update uses
the same replay minibatch schedule and exposes the head to posterior, mildly
corrupted, and mildly generated latent contexts. Multi-token targets are
masked whenever they would cross sequence padding, a terminal transition, or
an episode reset.

Evaluation uses the pre-existing held-out simulator probe banks and reports
action-ranking accuracy, state-wise Spearman correlation, simulator regret,
posterior/corrupted/generated-context reward MSE, mean calibration error, and
paired actor returns at imagination horizons 5 and 15. The pre-existing
shortcut-forcing results are an immutable external reference; this small
fine-tuning ablation is exploratory and is not a matched end-to-end superiority
claim.

## Cluster stages

`reward_head_mtp_study.sbatch` supports `preflight`, `reward`, `evaluation`,
`actor`, and `finalize`. Arrays contain 8, 8, and 16 cells respectively. The
verifier independently checks the frozen manifest, source hashes, identical
per-seed schedules, reward-only parameter and optimizer isolation, result
counts, and retained artifact hashes.
