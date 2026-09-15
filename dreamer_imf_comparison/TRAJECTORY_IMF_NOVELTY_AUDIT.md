# Trajectory iMF novelty audit

**Audit cutoff:** 2026-09-10
**Claim object:** the `imf_dreamer_jax` fixed-context trajectory-iMF objective, not the complete Dreamer 4 agent
**Machine-readable record:** `trajectory_imf_novelty_sources.json`

## Bounded verdict

The safest verdict is **bounded combination novelty only**.

No exact combination found in the primary sources audited through 2026-09-10 uses all of the
following together: improved MeanFlow's v-regression parameterization; independently sampled
per-target query intervals `(r_k, t_k)` that are decoupled from history exposure times `tau_j`;
the same per-token Gaussian draw to construct query and history views; an explicit mixture of
clean-history, independently corrupted-history, and clean-prefix/teacher-corrupted-suffix schedules; action
conditioning; and 1-NFE autoregressive world transitions. That absence supports only the wording
“apparently unreported narrow method combination.” It is **not a priority claim**, and it is not
evidence that no unpublished or differently named construction exists.

The broad ingredients are already known. Flow Matching supplies conditional vector-field
regression [S01]. MeanFlow supplies the interval-average identity and 1-NFE generation [S02]. iMF
supplies the v-space regression reformulation and predicted marginal-velocity tangent [S03].
Diffusion Forcing, Dreamer 4, Self-Flow, and StreamFlow supply independent or heterogeneous token
times and causal sequence corruption [S05, S06, S12, S13]. Most importantly, Causal-rCM already
implements a causal continuous-time CM/MeanFlow JVP whose clean context tangent is zero and whose
noisy target branch alone moves; it also describes a noisy-history extension [S18]. Therefore the
fixed-context causal JVP itself is established prior art, not the novelty.

MeLISA is even closer at the world-dynamics level: its Window-Consistency MeanFlow objective is in
the improved-MeanFlow lineage, it defines a stochastic autoregressive spatiotemporal transition
kernel, and it generates each forecast block with 1 NFE [S21]. This is not the identical iMF
objective used here: MeLISA's displayed WinC-MF JVP uses the pathwise `epsilon - x` tangent, whereas
iMF uses a predicted marginal-velocity tangent. MeLISA also shares one `(r,t)` pair across all
frames in a window rather than supplying the proposal's independent per-token query/history
schedules. It remains the closest 1-NFE autoregressive-dynamics precedent and rules out language
that implies there is no closely related MeanFlow/iMF-style predecessor.

There is likewise no open claim in merely connecting MeanFlow to trajectory flow matching or
shortcut consistency. AlphaFlow explicitly decomposes MeanFlow into trajectory-flow-matching and
trajectory-consistency terms and unifies trajectory flow matching, Shortcut Models, and MeanFlow in
one formulation [S19]. Here AlphaFlow's “trajectory” is a transport/noise-time trajectory, not an
environment-time rollout, but its objective relationship still preempts that broad theory claim.
SplitMeanFlow derives interval-splitting consistency and proves that the
MeanFlow differential identity is recovered in the infinitesimal-split limit [S20].

The current audit **does not establish empirical superiority** over Dreamer-4-style shortcut forcing.
Measurable normalized autoregressive conditional kernels do induce a normalized joint rollout law
by sequential composition. What is missing is a proof that the learned scheduled conditionals agree
with the data distribution and with the desired all-subsequence marginals, a tractable likelihood
interpretation, and a guarantee on compounding control error over environment time.
Aleatoric randomness does not provide epistemic uncertainty. A planner or learned actor may still
exploit model error.

## Method under audit

For target token `x_k`, Gaussian draw `epsilon_k`, query time `t_k`, and history exposure time
`tau_k`, the implementation forms two coupled views:

```text
query:   z_k(t_k)   = (1 - t_k) x_k + t_k epsilon_k
history: h_k(tau_k) = (1 - tau_k) x_k + tau_k epsilon_k
```

This **shared-noise two-view** construction uses the same `epsilon_k` in both views, while
`(r_k, t_k)` and `tau_k` are sampled separately.
At target `k`, a causal condition `c_k` is built from action/base features and the latest valid
strictly earlier history view, its exposure time, and a validity bit. For the iMF field
`u_theta(z_k, c_k, r_k, t_k)`, the implementation computes the partial material derivative

```text
D_k u_theta = partial_z u_theta · v_theta(z_k, c_k, t_k, t_k)
              + partial_t u_theta,
```

with zero tangent for `r_k` and with `c_k` captured as a fixed realized condition. Earlier tokens,
their exposure times, actions, masks, and other context therefore have zero tangent in this JVP.
The regression prediction is `u_theta + (t_k - r_k) stopgrad(D_k u_theta)`, together with the iMF
boundary/marginal-velocity loss. The target under this code's time orientation is
`epsilon_k - x_k`.

The schedule chooses one of three contexts per trajectory:

1. clean history (`tau_j = 0`);
2. independently timed corrupted history;
3. a clean prefix followed by an independently timed **teacher-corrupted suffix**, with supervised
   loss on that suffix (not a self-generated suffix).

This description matters because “trajectory iMF” could otherwise be mistaken for a joint flow
over an entire trajectory. The current library vectorizes conditional token objectives and uses a
strictly causal recurrent context summary. If its per-step conditional sampling kernels are
measurable and normalized, their autoregressive product defines a normalized joint rollout law.
The training objective does not, however, prove that conditionals learned under different exposure
schedules match the data conditionals or a consistent family of all-subsequence marginals, and it
does not provide a tractable likelihood or all-subsequence ELBO.

## Search boundary and evidence policy

The audit used primary papers, official proceedings pages, author project pages, and
author-maintained repositories. The source families were Flow Matching; MeanFlow and iMF; Shortcut
Models; Diffusion Forcing and rolling/history-conditioned diffusion; Dreamer 4; token-wise,
autoregressive, streaming, multi-time, and fully causal continuous generation; and flow/MeanFlow
world models and control. Core mathematical distinctions were checked in the method text, including
Dreamer 4 Equations (6)–(8) and Causal-rCM Equations (15)–(19), rather than inferred from titles.

The cutoff is prospective and explicit: sources publicly available by **2026-09-10** are in scope.
This is a literature audit, not a patent search. Search-index coverage, naming differences, private
work, and later revisions are unavoidable limitations. The audit should be rerun immediately before
submission.

## Claim-element audit

Status meanings: **established** means the claimed abstraction is already explicit in prior work;
**partial** means the ingredients exist but the narrower implementation combination was not found
verbatim; **exact combination not found** is a search-bounded observation, not priority evidence;
**unresolved** and **missing theory** are not contributions yet.

| ID | Claim element | Status | Closest evidence | Defensible residual |
|---|---|---|---|---|
| E01 | Conditional straight-path flow regression | Established | Flow Matching [S01] | None. |
| E02 | Interval-average velocity identity and direct 1-NFE generation | Established | MeanFlow [S02] | None. |
| E03 | iMF v-regression through an average-velocity predictor, predicted marginal-velocity tangent, and boundary evaluation | Established | iMF [S03] | Applying it to this sequence condition only. |
| E04 | Independent/heterogeneous token noise levels | Established | Diffusion Forcing [S05], Dreamer 4 [S06], Rolling Diffusion [S07], Self-Flow [S12] | None. |
| E05 | Independent iMF interval pairs `(r_k,t_k)` per target, decoupled from history `tau_j` | Partial | iMF intervals [S03] plus token-dependent times [S05, S12, S13] and causal CM/MeanFlow [S18] | A narrow schedule/objective combination needing an ablation or theorem. |
| E06 | Fixed-context causal JVP with zero context tangent | Established | Causal-rCM Eq. (17) explicitly uses zero clean-context tangent [S18]; conditional differentiation is already implicit in iMF [S03] | Correctness-critical semantics, not novelty. |
| E07 | Clean-history, corrupted-history, and teacher-corrupted-suffix supervision mixture | Partial | Flexible/noisy histories and rolling patterns [S05–S08, S13, S18] | Exact mixture, separately sampled history times, and shared-noise coupling as implementation choices. |
| E08 | Causal action-conditioned continuous-token world model and imagination | Established | Diffusion Forcing [S05], Dreamer 4 [S06], FlowWM [S15], Flow-JEPA [S16], Causal-rCM [S18] | None at this abstraction. |
| E09 | Step-size conditioning and stopped-gradient two-half-step bootstrap | Established | Shortcut Models [S04] and Dreamer 4 [S06] | Baseline, not proposed novelty. |
| E10 | One-step MeanFlow transition used in world-model/dynamical rollouts | Established | ITPO [S14] and MeLISA [S21] | Per-token scheduling and action-control details may differ; the broad category is occupied. |
| E11 | Action-conditioned trajectory-level flow world model | Established | Flow-JEPA [S16], with adjacent stochastic flow world modeling in FlowWM [S15] | Only objective and conditioning details remain. |
| E12 | All proposed details combined | Exact combination not found | Strong overlap spans iMF [S03], Diffusion Forcing [S05], Dreamer 4 [S06], multi-time flow [S12, S13], MeanFlow control [S14], Flow-JEPA [S16], Causal-rCM [S18], and MeLISA [S21] | At most an apparently unreported narrow combination; not a priority claim. |
| E13 | Better matched-budget rollout fidelity **and** actor learning than shortcut forcing | Unresolved | Neither iMF [S03] nor Dreamer 4 [S06] runs this comparison | Must pass the preregistered paired confirmatory rule; smoke data do not count. |
| E14 | Agreement of scheduled conditionals with data/all-subsequence marginals, tractable likelihood, and controlled compounding rollout error | Missing theory | Autoregressive kernels already induce a normalized rollout law; Diffusion Forcing additionally has an all-subsequence variational result [S05], and MeLISA adds finite-lag temporal-increment consistency [S21], but trajectory iMF has no analogous consistency proof, likelihood result, or control-error bound | Open theory target. |
| E15 | Epistemic uncertainty, shift calibration, and planner-exploitation protection | Missing theory | Stochastic rollout work motivates the problem [S05, S14, S15, S18] | Requires an explicit uncertainty and control mechanism. |
| E16 | MeanFlow generation of a continuous trajectory for control | Established | MP1 generates action trajectories in 1 NFE [S17] | A world-dynamics role is narrower, but “MeanFlow trajectory generation” is occupied. |
| E17 | Unifying trajectory flow matching, Shortcut Models, and MeanFlow | Established | AlphaFlow [S19] | The benchmark compares related objectives; it does not discover their relation. |
| E18 | MeanFlow differential identity as an infinitesimal interval-splitting limit | Established | SplitMeanFlow [S20] | Any theory contribution must go beyond this result and address causal conditional trajectories. |
| E19 | MeanFlow/iMF-style training of a stochastic 1-NFE autoregressive dynamical transition kernel over temporal windows | Established | MeLISA [S21] | MeLISA's displayed JVP uses the pathwise `epsilon - x` tangent rather than iMF's predicted marginal-velocity tangent; the exact objective, independent per-token query/history schedule, shared-noise coupling, and action-control details remain distinct. |

### The closest overlap: Causal-rCM

Causal-rCM [S18] materially narrows the claim. Causal-rCM Equation (17) belongs to a teacher-forcing
sCM/MeanFlow construction that packs clean context and noisy targets into a causal forward. The
equation assigns zero tangent to the
clean context and teacher-velocity/time tangents only to the noisy branch. Section 3.1.3 replaces
clean histories by noisy historical tokens while retaining loss on the current target. A paper must
therefore **not** claim that holding causal context fixed in a JVP is new.

The remaining distinction is narrower and should be tested rather than advertised as self-evident:
Causal-rCM is a teacher-distillation/self-forcing recipe for causal video diffusion/CM, whereas this
proposal is a from-scratch iMF v-regression objective. The current proposal samples separate
`(r_k,t_k)` for every supervised target, samples history exposure `tau_j` separately, couples the
query and history views through the same `epsilon_j`, and mixes three context patterns. The audited
Causal-rCM text does not clearly specify that exact per-token decoupling and coupling combination.
That is an absence in the checked text—not proof of priority.

## Closest prior work

### Objective lineage

- Flow Matching [S01] establishes conditional pathwise vector-field regression.
- MeanFlow [S02] replaces instantaneous transport with an interval-average velocity identity to
  support one-step sampling from scratch.
- Improved MeanFlow [S03] diagnoses the network-dependent target in original MeanFlow and expresses
  a v-loss through an average-velocity predictor using a predicted marginal-velocity tangent. The
  current loss is an application of this identity, not a new identity.
- Shortcut Models [S04] instead learn requested step sizes and bootstrap a larger step from two half
  steps. This provides the matched competing objective and a quality/NFE curve.
- AlphaFlow [S19] already unifies trajectory flow matching, Shortcut Models, and MeanFlow, and
  SplitMeanFlow [S20] already derives the differential MeanFlow identity as the infinitesimal limit
  of interval-splitting consistency. A conceptual relation between the two benchmark arms is not a
  contribution of this project.

### Sequence and causal-time lineage

- Diffusion Forcing [S05] explicitly trains a causal next-token model with independent per-token
  noise levels and proves a variational lower bound for all subsequences. It is stronger than the
  current proposal on probabilistic interpretation.
- Rolling Diffusion [S07] varies corruption along temporal position; History-Guided Video Diffusion
  [S08] supports variable history and long rollouts. Neither makes schedule-shaped context novel.
- FlowTime [S09] factorizes trajectory forecasting into autoregressive conditional flows. FELLE
  [S10] uses token-wise flow matching in a continuous autoregressive model. Transition Matching
  [S11] includes partially and fully causal continuous generation.
- Self-Flow [S12] uses heterogeneous token noise through dual-timestep scheduling. StreamFlow [S13]
  uses causal noising and predicts multiple time-indexed vector fields for token-wise streaming.
- Causal-rCM [S18] supplies the most direct prior art for a zero-context-tangent causal JVP and noisy
  context in continuous-time CM/MeanFlow training.

### World-model and control lineage

- Dreamer 4 [S06] combines Diffusion Forcing and Shortcut Models in an action-conditioned,
  block-causal world model. Its published Eq. (7) is an x-prediction shortcut target, it uses
  independent per-token signal levels and step sizes, a ramp weight, slightly corrupted prior
  context, and `K=4` inference per generated frame. This comparison is not a reproduction of Dreamer 4.
  The full architecture, data scale, tokenizer, and training system are not released or matched.
- ITPO [S14] already uses a one-step MeanFlow sampler in a differentiable world-model pipeline.
- FlowWM [S15] performs stochastic flow matching in feature-space world dynamics and evaluates
  horizon robustness.
- Flow-JEPA [S16] jointly generates an action-conditioned future latent trajectory with conditional
  flow matching.
- MP1 [S17] uses MeanFlow for 1-NFE action-trajectory generation. It is a policy, not a transition
  model, but it blocks broad “MeanFlow trajectory” language.
- MeLISA [S21] is the closest autoregressive-dynamics precedent. Its Window-Consistency MeanFlow
  is in the improved-MeanFlow lineage, operates jointly on a partially observed temporal window,
  and samples each forecast block in one evaluation. It is not identical to iMF: its displayed JVP
  tangent is the pathwise `epsilon - x`, rather than iMF's predicted marginal velocity. Its `(r,t)`
  is shared across frames, and it adds a Time Increment Consistency regularizer that this proposal
  does not have. This both narrows the novelty claim and supplies a serious long-horizon
  baseline/ablation idea.

### Dreamer-4 shortcut forcing versus trajectory iMF

| Dimension | Dreamer-4-style shortcut forcing | Current trajectory iMF | Consequence |
|---|---|---|---|
| Learned output | Clean representation (x-prediction) | iMF average velocity plus marginal-velocity output | Different heads/conditioning must be parameter- and FLOP-reported. |
| Large interval supervision | Stopped-gradient composition of two half steps | iMF differential identity and direct velocity regression | One is bootstrap/self-distillation; the other is a JVP objective. |
| Sequence times | Discrete per-token signal levels and requested step sizes | Continuous independent `(r_k,t_k)` plus separate history `tau_k` | Schedule support is not automatically compute-matched. |
| Context robustness | Slight past-input corruption at inference; Diffusion-Forcing lineage | Explicit clean/corrupted/teacher-corrupted-suffix supervision mixture | Whether this reduces generated-history shift is empirical. |
| Primary sampler | Published `K=4` per frame, with variable `K` supported | `K=1` per transition | Compare fidelity/return and compiler FLOPs, not NFE alone. |
| Probabilistic guarantee | No general no-compounding guarantee in Eq. (7) | Normalized autoregressive rollout law, but no data/all-subsequence agreement, tractable-likelihood, or no-compounding guarantee yet | One-step sample quality is insufficient for actor learning. |

## Likely reviewer objections

| ID | Objection | What would answer it |
|---|---|---|
| O01 | “This is iMF plus Diffusion Forcing, an obvious vectorization of known objectives.” | Show a nontrivial failure of the naive alternative and an ablation isolating the residual schedule/coupling choices. |
| O02 | “Fixed-context differentiation is ordinary conditional calculus.” | Agree and cite Causal-rCM [S18]; claim correctness, not novelty. |
| O03 | “Causal-rCM already implements zero-context-tangent MeanFlow/sCM JVPs and noisy history.” | Contrast exact objectives and schedules, then empirically compare or narrow the claim further. |
| O04 | “The three-way schedule mixture is augmentation, not a principled objective.” | Derive a target context distribution or show robust gains across frozen mixture weights. |
| O05 | “The recurrent context stores only the latest valid prior token, so ‘trajectory model’ overstates its memory.” | State the actual conditional state and test richer causal summaries or a block-causal transformer. |
| O06 | “Autoregressive conditionals define a joint rollout law, but scheduled training may not recover the data law or mutually agreeing all-subsequence marginals.” | Acknowledge normalization by autoregressive composition; prove schedule/data and subsequence-marginal consistency, or explicitly limit the claim to conditional sampling without a tractable-likelihood guarantee. |
| O07 | “Corrupted teacher histories do not match histories generated by the learned model.” | Measure train/generated context divergence and add a prespecified self-forcing control. |
| O08 | “Independent token intervals can create mutually inconsistent denoising tasks through shared context.” | Give a population consistency result or a counterexample-bounded compatibility test. |
| O09 | “One NFE per transition does not prevent small errors compounding over environment time.” | Report free-running horizon curves, calibrated stochastic scores, and a formal or empirical growth-rate analysis. |
| O10 | “Gaussian source noise is aleatoric; epistemic uncertainty remains absent.” | Add ensembles/posterior approximations and evaluate OOD uncertainty calibration. |
| O11 | “A planner can exploit tiny differentiable-model errors even when average rollout MSE improves.” | Report policy-exploitation gaps, action Jacobians, OOD action distance, and uncertainty-constrained controls. |
| O12 | “A 1-NFE versus 4-NFE result may only reflect unequal parameters, data, updates, or compiled compute.” | Enforce shared trunks/data and both equal-update and equal-compiler-FLOP tracks with parameter/FLOP disclosure. |
| O13 | “The shortcut baseline silently chooses an unpublished Dreamer 4 `K_max`.” | Expose `K_max`, label it as unspecified by the paper, and sweep it rather than guessing. |
| O14 | “Three local seeds on a diagnostic task are not evidence of general actor improvement.” | Use nested actor seeds, paired task-seed contrasts, frozen checkpoints, and the six-task confirmatory suite. |
| O15 | “Recent Flow-JEPA and ITPO already occupy flow/MeanFlow world-model territory.” | Avoid broad category claims and anchor the contribution in the exact residual combination and matched test. |
| O16 | “AlphaFlow and SplitMeanFlow already explain the MeanFlow/trajectory/shortcut relationship.” | Cite both [S19, S20] and avoid presenting that relation or limiting argument as new theory. |
| O17 | “MeLISA already supplies a closely related MeanFlow/iMF-style 1-NFE stochastic autoregressive dynamical transition.” | Agree that it is the closest precedent while distinguishing objectives: MeLISA's displayed WinC-MF JVP uses the pathwise `epsilon - x` tangent, not iMF's predicted marginal velocity. Restrict the residual to the exact objective plus per-token schedule/coupling/action setting, and compare its window/timestep and temporal-increment design where feasible. |

## Falsifiable stronger-theory targets

Each target below is stronger than the current claim and includes a result that would falsify it.

| ID | Candidate theorem or prediction | Required assumptions/evidence | Falsifier |
|---|---|---|---|
| T01 | **Conditional population consistency:** for almost every realized causal context, the fixed-context iMF population minimizer transports the prescribed conditional base law to the one-token data conditional in 1 NFE. | State regularity, support, boundary, function-class, and optimization assumptions; distinguish the learned marginal-velocity head from the oracle field. | A finite-dimensional counterexample in which zero population loss yields the wrong conditional endpoint law. |
| T02 | **Derivative-bias separation:** rebuilding corrupted history inside a joint JVP adds identifiable cross-token terms absent from the desired conditional derivative. | Derive the chain rule for the actual context constructor and quantify when cross terms vanish. | A proof that joint and fixed-context JVPs are identical under the claimed nontrivial conditions, or experiments showing no predicted bias where cross terms are nonzero. |
| T03 | **Schedule/data and subsequence consistency:** the normalized joint rollout law induced by measurable autoregressive kernels matches the data law at the target exposure schedule, and its marginals agree with the learned conditionals for every claimed subsequence schedule. | First state the ordinary autoregressive factorization; then supply a population-consistency proof or variational objective analogous in scope to Diffusion Forcing's subsequence result [S05]. A separate construction is needed for any tractable-likelihood claim. | A finite-dimensional population counterexample where scheduled losses are minimized but an induced subsequence marginal differs from the corresponding data marginal or scheduled conditional. |
| T04 | **Exposure-distribution bound:** generated-history risk is bounded by training-context risk plus a measurable divergence between the schedule kernel and generated-history distribution. | Define the divergence and show it is estimable from held-out rollouts. | Low schedule-kernel divergence coexists systematically with an unbounded or oppositely ordered generated-history loss gap. |
| T05 | **Rollout error growth:** under an explicit transition Lipschitz/contraction condition, horizon-`H` Wasserstein error is bounded by a geometric sum of one-step conditional approximation errors. | Estimate or upper-bound the relevant local Lipschitz constants and one-step distributional error. | Measured rollouts violate the bound with confidence after accounting for estimator error, or the assumptions fail on all benchmark tasks. |
| T06 | **Schedule-mixture benefit:** decoupled history exposure reduces a prespecified rollout-error growth coefficient relative to clean-only and query-tied history schedules. | Freeze mixture weights and comparators before evaluation; isolate the same-noise coupling separately. | The paired interval for the coefficient includes no improvement or favors either ablation on the confirmatory suite. |
| T07 | **One-step efficiency frontier:** at equal compiler FLOPs, trajectory iMF improves the joint fidelity/return frontier over Eq. (7) shortcut forcing, not merely NFE count. | Matched trunk/data/optimizer, disclosed active parameters, 1/2/4-NFE curves, and both primary confidence intervals. | Either primary paired interval includes zero, or shortcut forcing weakly dominates the Pareto curve. |
| T08 | **Planner-safe uncertainty:** an epistemic extension bounds model-exploitation regret as a function of calibrated uncertainty and action-support distance. | Add an explicit ensemble/posterior mechanism and assumptions linking uncertainty to transition/reward error. | Optimized policies achieve low predicted uncertainty yet retain a large real-versus-imagined return gap on held-out starts. |
| T09 | **Shared-noise coupling effect:** using the same `epsilon_k` for query and history views reduces gradient variance without changing the desired conditional population target. | Compare analytic expectations where possible and run a paired shared-versus-independent-noise variance study. | The population target changes, or gradient variance is not reduced under the prespecified estimand. |
| T10 | **Per-token times versus MeLISA-style shared window time:** independent `(r_k,t_k)` improves long-horizon conditional calibration without requiring a finite-lag auxiliary loss. | Add shared-window-time and Time-Increment-Consistency controls under matched compute, with prespecified horizon and energy-score metrics. | Shared window time matches or beats the per-token arm, or the gain vanishes when both receive the same finite-lag regularizer. |

Until one of T01–T05 is proved, the theoretical description should be “a conditional iMF
construction that induces an autoregressive rollout law,” not “a data-consistent all-subsequence
model with a tractable likelihood or control-error guarantee.” Until T07 passes, the empirical
description should be “we compare,” not “we improve.”

## Current novelty score and theorem target

As of the audit cutoff, a useful subjective assessment is **4/10 for the exact method combination**
and **2/10 for theory currently established**. The first number reflects an apparently unreported
combination of predicted-tangent iMF, decoupled token-wise query/history schedules, shared-noise
views, causal actions, and one-NFE autoregressive transitions. The second is low because the present
derivations establish the conditional JVP implementation and limiting reductions, but not a new
population-consistency or rollout-stability theorem. A non-vacuous loss-to-rollout result could
raise the theoretical contribution into the 5–6/10 range.

The strongest compact theorem target is the following. Let the learned one-step kernel be

\[
Q_{\theta,h}(\cdot\mid c)
=T_{\theta,h}(c,\cdot)_\#\gamma,
\qquad
T_{\theta,h}(c,\epsilon)=\epsilon-u_\theta(\epsilon,c,0,1),
\]

and let \(P_h(\cdot\mid c)\) be the true conditional transition. On an \(r=0\) characteristic,
write

\[
F_\theta(s)=u_\theta+s\left(\partial_su_\theta
+J_z u_\theta\,v_\theta\right).
\]

For oracle fields \(F^\star,u^\star,v^\star\), the endpoint error obeys the exact identity

\[
e(1)=\int_0^1\!\left[F_\theta(s)-F^\star(s)
-sJ_z u_\theta(s)(v_\theta(s)-v^\star(s))\right]ds,
\qquad e=u_\theta-u^\star.
\]

If the schedule gives the relevant slice mass \(\alpha_h>0\), the raw regression excess risk is
\(\mathcal E_h\), the velocity residual has weight \(\lambda_v>0\), and
\(\lVert J_z u_\theta\rVert_{\mathrm{op}}\leq G\), the candidate endpoint lemma is

\[
\mathbb E_{c\sim\mu_h}W_1^2(P_h,Q_{\theta,h})
\leq \frac{2}{\alpha_h}\max\!\left\{1,\frac{G^2}{\lambda_v}\right\}\mathcal E_h,
\]

up to explicit schedule-density and adaptive-weight equivalence constants. With generated-context
law \(\nu_h^\pi\ll\mu_h\), change of measure gives a factor
\(\sqrt{1+\chi^2(\nu_h^\pi\Vert\mu_h)}\). If the true closed-loop kernel is
\(\rho\)-Lipschitz in \(W_1\), this yields the familiar rollout recursion

\[
D_H^\pi\leq\sum_{h=0}^{H-1}\rho^{H-1-h}\delta_h^\pi.
\]

This is a **theorem target, not a proved result**. Three concrete gaps must be closed before using
it in a paper: the current sampler has an atom at \(r=t\), not at \(r=0\), so the displayed
\(\alpha_h\) can be zero; stopped adaptive weighting is not automatically equivalent to the raw
squared excess risk; and the Wasserstein coupling must compare learned and oracle endpoint maps
under the same base noise, not pair a generated sample with an independent data target. A clean
theory ablation would add a registered \(r=0\) schedule component and set adaptive power to zero.

## Safe publication wording

Recommended:

> We study an iMF objective with decoupled per-token query and history-exposure schedules for
> causal world-model transitions. Among the primary sources audited through 2026-09-10, we did not
> find the exact combination of independent token-wise iMF intervals, separately sampled history
> exposure times, shared-noise query/history views, a three-way causal schedule mixture, and 1-NFE
> autoregressive transitions. Fixed-context causal JVPs themselves are prior art in Causal-rCM.

Avoid “the first MeanFlow world model,” “the first trajectory flow world model,” “the first causal
MeanFlow JVP,” “provably stable rollouts,” and “outperforms Dreamer 4.” Even after a positive matched
benchmark, use “outperforms the implemented Dreamer-4-Equation-(7)-style objective under the frozen
conditions,” because this is not the unreleased full Dreamer 4 system.

## Primary-source manifest

Dates are first-publication/submission dates followed by the latest checked revision where one was
listed. Every evidence link below is a primary paper or official proceedings page.

| ID | Date | Primary source | Claim relevance |
|---|---|---|---|
| S01 | 2022-10-06; rev. 2023-02-08 | [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) | Conditional-path vector-field regression. |
| S02 | 2025-05-19 | [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447) | Average-velocity identity and 1-NFE generation. |
| S03 | 2025-12-01; rev. 2026-05-09 | [Improved Mean Flows](https://arxiv.org/abs/2512.02012) | iMF v-regression, predicted tangent, boundary velocity. |
| S04 | 2024-10-16; rev. 2025-06-23 | [One Step Diffusion via Shortcut Models](https://arxiv.org/abs/2410.12557) | Step-size conditioning and half-step bootstrap. |
| S05 | 2024-07-01; rev. 2024-12-10 | [Diffusion Forcing](https://arxiv.org/abs/2407.01392) | Independent per-token noise, causal sequences, subsequence ELBO. |
| S06 | 2025-09-29 | [Training Agents Inside of Scalable World Models (Dreamer 4)](https://arxiv.org/abs/2509.24527) | Action-conditioned shortcut forcing and imagination. |
| S07 | 2024-02-12; rev. 2024-09-09 | [Rolling Diffusion Models](https://arxiv.org/abs/2402.09470) | Position-dependent temporal corruption and rolling generation. |
| S08 | 2025-02-10; rev. 2025-07-24 | [History-Guided Video Diffusion](https://arxiv.org/abs/2502.06764) | Flexible history and long rollout via DFoT. |
| S09 | 2025-03-13; rev. 2026-02-08 | [Probabilistic Forecasting via Autoregressive Flow Matching](https://arxiv.org/abs/2503.10375) | Conditional-flow factorization of future trajectories. |
| S10 | 2025-02-16; rev. 2025-09-03 | [FELLE](https://arxiv.org/abs/2502.11128) | Token-wise flow matching in autoregressive continuous generation. |
| S11 | 2025-06-30 | [Transition Matching](https://arxiv.org/abs/2506.23589) | Partially and fully causal continuous generation. |
| S12 | 2026-03-06 | [Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis](https://arxiv.org/abs/2603.06507) | Heterogeneous token noise via dual-timestep scheduling. |
| S13 | NeurIPS 2025 | [StreamFlow: Streaming Audio Generation from Discrete Tokens via Streaming Flow Matching](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0713495297dab18fabca4795cb9ef8ef-Abstract-Conference.html) | Causal noising and simultaneous multi-time fields for token-wise streaming. |
| S14 | 2026-03-23; rev. 2026-05-20 | [Inference Time Policy Optimization for Offline RL with Differentiable World Models](https://arxiv.org/abs/2603.22430) | One-step MeanFlow sampler in differentiable world-model rollouts. |
| S15 | 2026-06-27; rev. 2026-07-15 | [Flow Matching in Feature Space for Stochastic World Modeling](https://arxiv.org/abs/2606.29059) | Stochastic feature-space flow world model. |
| S16 | 2026-08-29 | [Flow-JEPA](https://arxiv.org/abs/2608.29029) | Conditional flow of action-conditioned future latent trajectories. |
| S17 | 2026-03-14 | [MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation](https://ojs.aaai.org/index.php/AAAI/article/view/38919) | MeanFlow 1-NFE action-trajectory policy. |
| S18 | 2026-06-24 | [Causal-rCM](https://arxiv.org/abs/2606.25473) | Fixed-context causal JVP, continuous-time CM/MeanFlow, noisy histories, interactive world models. |
| S19 | 2025-10-23 | [AlphaFlow: Understanding and Improving MeanFlow Models](https://arxiv.org/abs/2510.20771) | Unifies trajectory flow matching, Shortcut Models, and MeanFlow. |
| S20 | 2025-07-22 | [SplitMeanFlow: Interval Splitting Consistency in Few-Step Generative Modeling](https://arxiv.org/abs/2507.16884) | Proves the MeanFlow differential identity is an infinitesimal interval-splitting limit. |
| S21 | 2026-05-07; rev. 2026-07-27 | [Autoregressive One-Step Generative Modeling for Dynamical System Forecasting (MeLISA)](https://arxiv.org/abs/2605.05540) | MeanFlow/iMF-style 1-NFE stochastic autoregressive dynamics; its displayed WinC-MF JVP uses a pathwise `epsilon - x` tangent, not iMF's predicted marginal velocity. |

Official code/project links are retained in the JSON manifest where available, including the
MeanFlow, iMF, Shortcut Models, Diffusion Forcing, FlowWM, and Flow-JEPA repositories. A missing code
URL is not evidence that code does not exist; it means no author-maintained repository was relied on
for this claim.

## Verification

Run from the repository root:

```sh
/private/tmp/trajectory-imf-venv/bin/python dreamer_imf_comparison/scripts/verify_trajectory_imf_novelty.py
```

The verifier checks the cutoff, primary-link structure, required topic coverage, cross-references,
bounded verdict, element table, objections, falsifiable targets, and presence of every source URL in
this audit. It is intentionally offline and deterministic; it validates the frozen audit artifact,
not the continued availability or semantic content of remote pages.
