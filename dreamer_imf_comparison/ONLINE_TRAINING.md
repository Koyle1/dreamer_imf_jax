# Bounded online fine-tuning benchmark

This is **offline-to-online fine-tuning**, not training from scratch and not a
full Dreamer/FlowMPC reproduction. The protocol is fixed in
`online_training_protocol.json`. Original dependency checkpoints and evidence
are read-only; historical returns are not used as contemporaneous baselines.

## Comparison

Reacher-hard, three existing world-model seeds (431, 433, 439), actor seed 541:

| Arm | Online learner | Controller |
| --- | --- | --- |
| F | Frozen weights | Fresh paired A1 inference-time adaptation |
| P | ReBRAC actor and critic | Same A1, fixed world and reward models |
| M | World, scalar reward head, ReBRAC actor and critic | Same A1 |

Each P/M cell collects 500,000 additional native environment steps. Parameters
are fixed within each 1,000-step episode. At the boundary, P performs 250
ReBRAC updates; M additionally performs 250 world and 250 reward updates.
Both sample half original training replay and half newly collected replay.
Evaluation episodes never enter replay. P/M share environment-seed schedules,
but their evolving policies produce different data; this is not a fixed-data
causal comparison. Compute is deliberately unequal and measured separately.

The collector adds clipped Gaussian noise (sigma 0.1) only during training and
conditions the next posterior on the **executed** action. It retains final
observations and native time-limit boundaries. The base actor becomes the new
trust reference at each episode reset; latent belief and persistent adapted
actor reset there. Original normalization and trust anchors remain fixed.
Trust-region diagnostics describe the clean controller action, before noise.

World/reward Adam states start fresh because the dependency omitted them;
ReBRAC parameters, targets, optimizer moments and clocks are retained. The
ReBRAC reciprocal Q scale has a finite 1e-8 floor. It does not otherwise change
the released objective. Sparse random exploration may still be insufficient.

## Evidence and restart contract

Every completed training epoch archives its complete learner, shifted episode,
controller trace, numeric checks, exact update clocks, parameter ancestry and
SHA-256 marker. Evaluation snapshots at 100k/200k/300k/400k/500k must equal the
exported archived learner. Completed epochs can be read on restart; an unsealed
partial epoch is never overwritten, and submission is not automatically retried.

Each evaluation has five fresh paired seeds. F is evaluated once per world,
as its parameters never change. There are six training cells and 33 evaluation
cells. Evaluation uses an external discarded cache primer, then separate
creation/replay reader processes. Their complete traces must agree exactly;
post-exit cache fingerprints and receipts must also agree. The GPU preflight
exercises real P/M collection and updates plus independent reader replay before
any production training can be submitted. Compiler-writer output is not accepted
as the evaluation baseline.

The final report uses the three world seeds as the top-level statistical unit,
reports all paired P-F/M-F/M-P final deltas and fixed-grid learning curves, and
does not select a best checkpoint. Five episodes are nested observations, not
15 independent trained models. It includes controller diagnostics, reward
coverage, exact update budgets, measured phase timings and Slurm accounting.

## 500k budget and published DreamerV3 reference

This budget-only successor starts afresh from the same authenticated offline
dependency, not the completed online study. The completed 100k run at commit
`ec24278c419d24670309b8947785d87095ec6752` remains immutable. Each training
cell now has 500 archived episodes and 125,000 policy updates; M additionally
has 125,000 world and 125,000 reward updates. Seeds, learning rates, update
ratios, controller, exploration and replay mixture are unchanged. The same six
training and 33 evaluation cells are used; no best checkpoint is selected.

[DreamerV3 arXiv v2](https://arxiv.org/pdf/2301.04104v2), Table 2,
specifies 500k environment steps and action repeat 2 for proprioceptive control.
Its environment-step convention counts before action repeat (the same table
lists Atari100K as 400k steps with repeat 4). Table 11 reports Reacher Hard
return **938**. We match the 500k additional native-step budget, but retain our
native repeat 1: 500k decisions versus Dreamer's 250k. We also retain offline
pretraining, our own learner/update ratio and controller. Thus this is a
published reference, not a matched DreamerV3 reproduction or equal-total-data
comparison. A separately trained Dreamer baseline is not part of this run.
Compare our fresh 100k and 500k checkpoints within this study; historical 100k
results are descriptive context, not evidence reused in the new run.

## Commands (exact clean deployed commit only)

Set `PYTHONPATH=dreamer_imf_comparison:imf_dreamer_jax/src` and use the CUDA
environment configured by the canonical submitter. `<root>` must be fresh.

```bash
python -m dreamer_imf_compare.online_training_study register --root <root> --dependency <authenticated-dependency>
python -m dreamer_imf_compare.online_training_study launch --root <root> --stage preflight
# Only after the job completes successfully:
python -m dreamer_imf_compare.online_training_study verify --root <root> --stage preflight
python -m dreamer_imf_compare.online_training_study launch --root <root> --stage training
# Only after all six training cells complete successfully:
python -m dreamer_imf_compare.online_training_study launch --root <root> --stage evaluation
# Only after all 33 evaluation cells complete and authenticate:
python -m dreamer_imf_compare.online_training_study finalize --root <root>
```

Each launch authenticates its previous complete stage against terminal Slurm
accounting. Intent/receipt/release records prevent duplicate submission.
Preflight requests one hour, training requests 24 hours per cell, and evaluation
requests two hours per cell; the short stages do not inherit a full-day allocation.
A failed stage is a stop condition requiring diagnosis, not a reason to weaken
validation. Completion requires `ONLINE_BENCHMARK_FINAL_VERIFIED` and independent
report/manifest validation. No benchmark improvement is assumed in advance.
