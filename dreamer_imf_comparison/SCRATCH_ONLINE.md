# From-scratch online Reacher Hard benchmark

This supersedes the cancelled warm-start 500k proposal. No weights, optimizer
moments, replay, reward normalization or anchors are loaded from older studies.
The separate CLI has no dependency/checkpoint input argument. Existing studies
and the old warm-start CLI remain unchanged.

Three independently initialized seeds (431,433,439), actor seed541, train the
current compact trajectory-iMF world model, scalar state-action reward head and
ReBRAC actor/critic with persistent trust-constrained A1 control. These are not
DreamerV3's architecture or actor-learning algorithm.

Each seed collects **500,000 native environment steps,250,000 decisions** with
action repeat2. Repeated rewards are summed. Each full episode has1000native
steps/500decisions. The first5000native steps are uniform random exploration,
included in the budget. They supply the fresh replay, reward-head normalization
and256 fixed trust anchors. No gradient updates occur before this collection;
the learner then catches up with1250updates. Every later episode adds250world,
250reward and250ReBRAC critic updates; the delayed actor updates every second
critic step. Totals125000world/reward/critic and62500actor updates perseed.
Only new training episodes enter replay. Normalizers and anchors stay frozen
after prefill. Old ReBRAC behavioral regularization remains, so successful
online exploration/learning is not assumed.

Five held-out paired evaluation episodes at100k,200k,300k,400k,500k give15cells.
Evaluation never enters replay or selects a checkpoint. Three training seeds
are statistical units; episodes are nested. Independent discarded primer,
creation and replay processes require exact trace equality and identical cache
fingerprints/post-exit seals. Training initialization is independently
reconstructed from the recorded random seed and counted prefill. Every epoch
binds its parent, parameters, exact optimizer clocks, episode and cumulative
native/decision counters. Partial artifacts are never overwritten.

Preflight uses a separate fresh random model and96native steps of discarded
prefill plus96native evaluation steps in each of three processes; it verifies
actual world/reward/policy updates and external replay before production.
These smoke-test interactions and held-out evaluation interactions are reported
separately from training and are never reused in production.

## Published comparison

[DreamerV3 arXiv2301.04104v2](https://arxiv.org/pdf/2301.04104v2), Table2,
specifies500k native environment steps with action repeat2 for proprioceptive
control; Table11 reports ReacherHard return**938**. Compare our final mean,
all three seed means and their differences from938. This matches task,
interaction budget, repeat and from-scratch status, not architecture, update
ratio, parallel environment count, compute or an independently rerun baseline.
No significance or general-superiority claim is justified from three seeds.

## Canonical execution

Use the exact clean deployed commit, fresh root and configured CUDA environment:

```bash
python -m dreamer_imf_compare.scratch_online_study register --root ROOT
python -m dreamer_imf_compare.scratch_online_study launch --root ROOT --stage preflight
# After successful terminal accounting and canonical verification:
python -m dreamer_imf_compare.scratch_online_study launch --root ROOT --stage training
# After all three training cells authenticate:
python -m dreamer_imf_compare.scratch_online_study launch --root ROOT --stage evaluation
# After all15evaluation cells authenticate:
python -m dreamer_imf_compare.scratch_online_study finalize --root ROOT
```

Each launch verifies its preceding stage and rejects duplicate/uncertain intent.
Completion requires `SCRATCH_ONLINE_FINAL_VERIFIED` plus independent report/hash
validation. Training has3bootstrap markers and1488epoch markers (episodes5..500),
with1500total recorded episodes including the15prefill episodes. GPU allocation
limits are preflight1h,training8h,evaluation2h; concurrency3. These are limits,
not forecasts. Any source fix requires a fresh exact commit/root/preflight.
