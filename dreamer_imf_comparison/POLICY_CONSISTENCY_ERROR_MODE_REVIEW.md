# Policy-consistency error mode: evidence review and minimal roadmap

## Bottom line

The observed failure is best explained as **decision-objective mismatch under
distribution shift**, not as an iMF numerical instability and not primarily as
an inability to reconstruct or roll out observations.

The causal auxiliary trained the world model to reproduce local, one-step
observation and reward differences around behavior-data actions. The evaluated
actor, however, is a likelihood-ratio (REINFORCE) actor trained for 15 imagined
steps. Its model actions are stop-gradiented. It therefore does not optimize the
local dynamics Jacobian that the auxiliary targets; it optimizes model-generated
advantages and can exploit their ranking and calibration errors over an
actor-induced state/action distribution.

The smallest defensible next step is therefore not an SDE and not a Pearson
loss. It is an actor-only test on the already frozen models:

1. implement Dreamer 4's actual sign-only PMPO update;
2. behavior-clone the policy on replay, freeze that policy as a prior, and use
   the published reverse-KL coefficient;
3. compare imagination horizons 15 and 5 with a real-data-grounded terminal
   value;
4. evaluate all models on one shared state/action probe bank.

Only if that fails should the one-step causal auxiliary be replaced by a
short-horizon, advantage-consistency objective.

## What the retained evidence says

All numbers below come from authenticated retained artifacts. The small causal
study is exploratory: two world-model seeds, one actor seed, and extra simulator
labels for causal iMF only.

| Reacher, same seeds 211/223 and actor seed 311 | Shortcut | Original trajectory-iMF | Causal trajectory-iMF |
|---|---:|---:|---:|
| Mean normalized actor return | 0.0997 | 0.0980 | 0.0018 |
| Mean rollout-error AUC (lower is better) | 0.12449 | 0.12070 | 0.12505 |
| Seed 211 actor return | 0.1994 | 0.1960 | 0.0036 |
| Seed 223 actor return | 0.0000 | 0.0000 | 0.0000 |

Thus causal iMF lost about 98.2% of the original iMF mean actor return while
making rollout-error AUC about 3.6% worse. This is a large intervention effect,
but its statistical support is weak because only seed 211 distinguishes the
actors.

The wider completed pilot (aggregating its multiple candidates/budget tracks) is
consistent with "no robust control advantage":

- Reacher over 72 actor cells per arm: trajectory-iMF 0.05158 versus shortcut
  0.04643 mean normalized return.
- Pendulum over 72 actor cells per arm: trajectory-iMF 0.002336 versus shortcut
  0.002467.
- Returns were highly zero-inflated, so these small mean differences do not
  support superiority.

### Diagnostics before the causal intervention

On Reacher actor-visited states, original iMF had better reward prediction and
better across-cell imagined/real return correlation, but worse decision-local
metrics:

| Metric | Shortcut | Original trajectory-iMF |
|---|---:|---:|
| Observation standardized MSE | 1.676 | 2.547 |
| Reward MSE | 0.631 | 0.366 |
| Action-ranking pairwise accuracy | 0.366 | 0.346 |
| Mean per-state Spearman rank correlation | -0.255 | -0.353 |
| Model/simulator action-gradient mean cosine | 0.167 | -0.164 |
| Hallucinated nonzero-gradient fraction | 0.750 | 0.875 |
| Across-cell imagined/real Pearson | 0.125 | 0.627 |

This already showed that scalar predictive metrics and aggregate imagined/real
correlation did not identify whether the model chose the right action locally.

### Diagnostics after the causal intervention

Causal iMF obtained pairwise action-ranking accuracy 0.786 and action-gradient
cosine 0.939 on its own actor-visited states. Those apparent gains are not
reliable cross-model comparisons:

- only 2 of 8 states had informative action rankings;
- only 1 of 8 states had a nonzero simulator action gradient;
- the model hallucinated a nonzero gradient in 7 of 8 states;
- the states were sampled from the new, failed actor rather than from the same
  state distribution used for the other models;
- imagined mean return was still 31.75 for seed 211 while real episode return
  was only 3.6 (0.0036 normalized).

The intervention therefore improved a mostly flat, actor-dependent local probe
without making the actor useful.

## Failure chain

### 1. The auxiliary optimizes the wrong mathematical object

The causal loss predicts one-step observation and reward differences. The actor
update is instead

\[
  \nabla_\phi J_{\hat M}(\pi_\phi)
  \approx
  \mathbb E_{\hat d^{\pi}}
  [\nabla_\phi\log\pi_\phi(a\mid s)\,\hat A^{\pi}_{\hat M}(s,a)].
\]

For the configured REINFORCE estimator, actions entering the world model are
stop-gradiented. Local agreement of
\(\partial \hat s_{t+1}/\partial a_t\) with the simulator is therefore neither
necessary nor sufficient. The relevant target is agreement of advantages (or at
least action ordering) over the states and actions sampled by the changing
policy.

### 2. One-step behavior-local accuracy need not compose for 15 steps

The auxiliary perturbs recorded actions by 0.1 and observes one simulator step.
The actor samples its own action sequence and optimizes a 15-step lambda return.
Small signed reward/dynamics biases can compound, and the learned actor can move
away from the behavior distribution where the probes were collected.

### 3. The critic and actor can form a self-consistent hallucination

The frozen protocol uses a scalar-MSE critic trained on imagined lambda returns.
Those targets inherit the learned model's reward and transition errors. The slow
critic EMA stabilizes this target but does not ground it in real returns. Low
critic loss can therefore coexist with large imagined/real return bias.

### 4. The current actor lacks the main Dreamer 4 safeguards

The frozen protocol uses REINFORCE, horizon 15, no behavior-prior KL, and a
single-bin scalar critic. Dreamer 4 instead behavior-clones a policy, freezes it
as a behavioral prior, uses reverse KL with coefficient 0.3, and uses a sign-only
PMPO objective so advantage magnitude errors cannot dominate the update.

The library contains a `pmpo` option, but it is not the published sign-only
objective: it exponentially weights advantage magnitudes. Thus it should be
corrected before it is called a Dreamer-4-style control.

### 5. The causal auxiliary has three additional optimization risks

- Gradients flow through the source posterior state into the representation;
  the auxiliary is not isolated to action-conditioned dynamics and reward
  prediction.
- Normalization is recomputed per minibatch and per observation feature. Nearly
  action-invariant dimensions receive a scale near the 1e-3 floor and can exert
  disproportionate, noisy gradients.
- A small scalar contribution to total loss does not imply a small gradient.
  Per-loss gradient norms and cosine conflicts were not recorded.

### 6. Aleatoric stochasticity is not epistemic uncertainty

Common random numbers correctly cancel sampling noise in paired predictions,
but one iMF model still has no signal that an actor-proposed trajectory is
outside its knowledge. Switching from an ODE to an SDE would primarily enrich
conditional stochasticity; it would not by itself reveal parameter/model
uncertainty or prevent policy exploitation.

## What the primary literature implies

| Literature result | Relevance here | Limitation/counterargument |
|---|---|---|
| [Objective Mismatch in MBRL](https://proceedings.mlr.press/v120/lambert20a.html), [VAML](https://proceedings.mlr.press/v54/farahmand17a.html), [value equivalence](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html), and [PAML](https://arxiv.org/abs/2003.00030) argue that prediction loss should reflect how planning uses the model. | Directly matches the gap between observation/reward deltas and REINFORCE advantages. | A value-aware loss can itself be uncalibrated; [CVAML](https://proceedings.mlr.press/v267/voelcker25a.html) proves this for common sampled stochastic variants. |
| [VaGraM](https://arxiv.org/abs/2204.01464) weights model error by value sensitivity. | Supports focusing capacity on decision-relevant dimensions. | Our actor is score-function REINFORCE, not a pathwise value-gradient actor; copying VaGraM literally would again target the wrong gradient. |
| [MBPO](https://proceedings.neurips.cc/paper/2019/hash/5faf461eff3099671ad63c6f3f094f7f-Abstract.html) controls model bias with short rollouts branched from real data. | Supports reducing horizon from 15 to 5 (or less) and bootstrapping with a grounded value. | A shorter horizon can introduce terminal-value bias and may reduce exploration. |
| [PETS](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html) and [MOPO](https://proceedings.neurips.cc/paper_files/paper/2020/hash/a322852ce0df73e204b7e67cbbef0d0a-Abstract.html) use ensemble uncertainty or uncertainty penalties. | Direct response to actor exploitation of unsupported trajectories. | Ensembles can be correlated or miscalibrated and increase compute; use only after cheaper support constraints are tested. |
| [TD-MPC2](https://arxiv.org/html/2310.16828v2) jointly trains latent consistency, reward, distributional value, policy prior, EMA Q ensembles, and short-horizon MPC. | Supports real-reward/value grounding, distributional critics, latent normalization, and policy-constrained planning. | Replacing the generative objective with TD-MPC2 would abandon rather than test trajectory-iMF's contribution. |
| [Diffusion Forcing](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2aee1c4159e48407d68fe16ae8e6e49e-Abstract-Conference.html), PlaNet's [latent overshooting](https://proceedings.mlr.press/v97/hafner19a.html), and [Self Forcing](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f4823f831af67a3ef15e41a85434422a-Abstract-Conference.html) address exposure and multistep consistency. | Supports training on generated contexts and sequence-level losses if shared-state rollout drift remains. | Our immediate causal failure occurs despite similar rollout AUC; these are second-line, not first-line, changes. |
| [Dreamer 4](https://arxiv.org/html/2509.24527v1) uses x-prediction, RMS-normalized losses, slight context corruption, multi-token reward/action prediction, frozen dynamics during imagination, sign-only PMPO, and a behavioral-prior KL. | Gives a coherent set of safeguards, several of which our actor test omitted. | Most are established controls, not trajectory-iMF novelty; they must be applied to both arms in a claim-bearing comparison. |
| Score-based [SDE modeling](https://openreview.net/pdf?id=PxTIG12RRHS) and [SDE Matching](https://proceedings.mlr.press/v267/bartosh25a.html) enrich stochastic dynamics. | Appropriate if conditional multimodality or aleatoric calibration is the measured bottleneck. | They do not automatically estimate epistemic uncertainty and add sampling/training complexity. |

## Recommended algorithm, in order

### Stage A: repair the policy-learning harness without retraining a world model

Apply the same actor to shortcut, original iMF, and causal iMF:

1. Train the existing behavior-cloning policy on posterior replay features.
2. Snapshot it as a frozen prior.
3. Replace the current exponential `pmpo` weighting with the exact sign-only,
   separately averaged positive/negative objective from Dreamer 4.
4. Set the reverse policy-to-prior KL to 0.3 for the first preregistered test.
5. Use a symlog/two-hot critic and its existing EMA target.
6. Compare horizons 15 and 5; do not tune further on the same two seeds.

Why this is first: it is actor-only, cheap, applies fairly to all world models,
and directly removes two exploitation channels—advantage magnitude and action
support—without modifying trajectory-iMF.

Falsifiable prediction: imagined/real return bias and action saturation should
fall. If real return stays near zero while imagined return stays high, the world
model/reward surface, rather than the policy optimizer, is the binding failure.

### Stage B: replace one-step causal consistency with advantage consistency

If Stage A fails, revert the current causal auxiliary and train a policy-aware
loss on short counterfactual suffixes. From a shared real start state, compare
candidate actions sampled from a mixture of behavior and current-policy actions.
For horizon \(h\in\{1,3,5\}\), define simulator/model returns
\(G_i^h,\hat G_i^h\), and state-centred advantages

\[
 A_i^h=G_i^h-\frac1m\sum_jG_j^h,\qquad
 \hat A_i^h=\hat G_i^h-\frac1m\sum_j\hat G_j^h.
\]

Use a calibrated magnitude term, an informative-pair ranking term, and an
explicit flat-target suppression term:

\[
\begin{aligned}
\mathcal L_{\rm PA}={}&
 \mathbb E\,\rho\!\left(
 \frac{\hat A_i^h-\operatorname{sg}(A_i^h)}
      {\operatorname{sg}(\operatorname{RMS}_{\rm EMA}(A^h))+\epsilon}
 \right)\\
&+\lambda_r\,\mathbb E_{|A_i-A_j|>\tau}
 \operatorname{softplus}\!\left(
 -\operatorname{sg}(\operatorname{sign}(A_i-A_j))
 \frac{\hat A_i-\hat A_j}{T}\right)\\
&+\lambda_0\,\mathbb E_{|A_i-A_j|\le\tau}
 \rho(\hat A_i-\hat A_j).
\end{aligned}
\]

Implementation constraints:

- stop gradients through the source posterior and all simulator targets;
- supervise reward/return and latent dynamics, not decoded observation pixels or
  every state coordinate;
- use a running RMS with a meaningful floor, not a noisy minibatch denominator;
- log each loss's gradient norm and cosine with the base iMF loss;
- cap or project the auxiliary gradient if it conflicts persistently;
- refresh current-policy candidate actions as the actor changes, because a
  policy-aware model becomes stale.

This is the correct place for "direction" information. Pearson correlation
alone is not: for any positive scale \(c\), correlation is unchanged by
\(\hat A=cA+b\), even though a large \(c\) can produce exploitable values.
Sparse/tied rewards also make correlation undefined or misleading. Pairwise
ranking plus a robust magnitude anchor retains direction without discarding
calibration.

### Stage C: add epistemic pessimism only if unsupported trajectories persist

Use three bootstrapped reward/value heads (or, at higher cost, three world
models). Penalize imagined return by ensemble disagreement or use a lower
quantile/minimum target. Keep iMF sampling variance separate from ensemble
variance. This is the appropriate response to epistemic uncertainty; an SDE is
not.

### Stage D: only then improve autoregressive generation

If shared-state rollout diagnostics reveal actual exposure drift:

- parameterize/train the iMF prediction in clean-endpoint (`x`) coordinates
  while retaining the mathematically equivalent iMF JVP;
- replace teacher-corrupted suffixes with some stop-gradient self-generated
  contexts;
- add small context corruption at rollout as a declared sensitivity;
- train on horizons longer than the context used at evaluation.

These changes target generated-context drift. They are not justified by the
current actor failure alone because causal and shortcut rollout AUC are already
nearly tied.

## Minimal experimental ladder

### Gate 0: shared probe bank (no training)

Evaluate all three frozen models on identical replay states, candidate actions,
random seeds, and suffixes. Report horizon-resolved (1, 3, 5, 15):

- action ranking, regret, and informative-state count;
- advantage magnitude bias and sign accuracy;
- imagined/real return bias;
- flat-target hallucination rate;
- only as a secondary diagnostic, model/simulator action-Jacobian agreement.

This removes the current actor-visited-state confound.

### Gate 1: cheap actor-only study

For each of the two retained Reacher world seeds and at least three actor seeds:

- current REINFORCE, horizon 15 (existing reference);
- exact sign-PMPO + behavior prior KL, horizon 15;
- exact sign-PMPO + behavior prior KL, horizon 5.

Run on original iMF first, then causal iMF, then shortcut for the matched control.
Pre-register that the one-step causal loss is discarded if original iMF with the
repaired actor is at least as good.

### Gate 2: one small world-model intervention

Only if Gate 1 fails, train the advantage-consistency iMF at one fixed low scale
and horizons {1, 3, 5}. Do not launch a weight grid. Compare against original
iMF on the shared probes and paired actor seeds.

### Gate 3: uncertainty or sequence exposure

Choose exactly one based on Gate 0 evidence:

- ensemble pessimism if errors concentrate on unsupported actions/states;
- self-generated-context/x-prediction if errors grow mainly with rollout
  horizon on supported trajectories.

### Suggested preregistered success/stop rules

- Do not accept a method whose rollout AUC degrades by more than 5% relative to
  original iMF.
- Require improvement across paired world and actor seeds, not just the mean of
  a zero-inflated sample.
- Require at least 20 informative shared-probe states before interpreting rank
  or gradient metrics.
- Require imagined/real return bias to fall by at least 50% on the failed seed.
- Stop adding world-model machinery if actor-only support constraints repair the
  transfer.
- Do not pursue an SDE unless a calibrated multimodality/aleatoric test fails
  after epistemic uncertainty is measured separately.

## Publication implication

The failure mode—prediction accuracy failing to imply control quality—is known.
The current one-step causal loss is therefore neither a solution nor, by itself,
a novel contribution. A publishable trajectory-iMF contribution would need to
show that its per-token-time JVP yields a principled **policy-aware trajectory
objective**, with reductions to ordinary iMF/conditional flow matching and a
bound or consistency statement connecting advantage error to the score-function
policy-gradient error. Empirically, that objective would need matched actor
controls, shared-state diagnostics, multiple tasks/seeds, calibration, rollout
fidelity, and NFE/compute reporting.

The negative causal result is valuable internally: it rules out the tempting
but incorrect claim that matching local observation motion is sufficient. It is
not yet a NeurIPS result on its own.
