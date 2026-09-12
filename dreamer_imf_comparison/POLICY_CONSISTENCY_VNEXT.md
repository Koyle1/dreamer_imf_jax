# Policy-consistency vNext

This extension addresses the observed failure mode in which a trajectory-iMF
world model can reconstruct and rank some local actions well while its learned
actor still exploits unsupported imagined rewards. It is an exploratory
intervention, not a result claim.

## 1. Frozen-world actor repair

The actor is first behavior-cloned on replay posterior features. A frozen copy
of that actor becomes the behavior prior. The critic is grounded on stopped
replay features and replay lambda returns, using the same 51-bin symlog/two-hot
head and slow EMA target later used in imagination. Actor optimization then
uses sign-only PMPO with reverse KL coefficient 0.3.

For positive and negative advantage sets \(D^+\) and \(D^-\), the policy part
of the minimized loss is

\[
  \mathcal L_{\rm PMPO}
  = \frac{1-\alpha}{|D^-|}\sum_{i\in D^-}\log\pi(a_i|s_i)
  - \frac{\alpha}{|D^+|}\sum_{i\in D^+}\log\pi(a_i|s_i)
  + 0.3\,\mathbb E\,D_{\rm KL}(\pi\|\pi_{\rm BC}),
\]

with \(\alpha=1/2\). Advantage magnitudes do not enter after classification
by sign. Empty sets contribute zero. The world-model tree and frozen prior
must have exactly zero parameter delta. Horizons 5 and 15 are separate cells.

## 2. Shared fixed probe bank

The probe bank is constructed before checkpoint evaluation and contains only
test-split replay state indices, the replay action plus coordinatewise
\(\pm0.1\) alternatives, shared replay-action suffixes, horizons 1/3/5/15,
simulator returns, and standard-normal model draws. The same bank digest is
required for shortcut forcing, original trajectory-iMF, and causal
trajectory-iMF. No learned policy selects a state or action in this evaluation.

Reported metrics include informative-pair accuracy, statewise rank behavior,
top-1 simulator regret, action-independent flatness, and horizon dependence.
The bank separates world-model action geometry from actor visitation.

## 3. Short-horizon advantage consistency

When actor repair does not convert model quality into real return, the
one-step observation-delta surrogate is replaced by a reward/return objective.
For fixed action suffixes and horizons \(h\in\{1,3,5\}\), define centered
advantages

\[
 A_h(a)=G_h(a)-\frac{1}{K}\sum_{j=1}^K G_h(a_j).
\]

The implemented loss is

\[
 \mathcal L_{\rm AC}=\lambda_m\,\rho((\hat A-A)/\sigma_A)
 +\lambda_r\,\operatorname{softplus}(-\operatorname{sign}(\Delta A)\Delta\hat A/\sigma_A)
 +\lambda_f\,\mathbf 1[\operatorname{range}(G)\le\epsilon]\rho(\hat A/\sigma_A).
\]

The first term matches robust normalized magnitudes, the second supervises
only informative action pairs, and the third suppresses hallucinated action
dependence where the simulator is flat. \(\sigma_A\) is a persistent running RMS,
not a fresh per-minibatch divisor. Simulator targets and posterior source
states are stopped; gradients train only the model rollout.

## 4. Epistemic pessimism

Independent bootstrap reward heads are fitted on stopped replay posterior
features. Between-head standard deviation is treated as epistemic uncertainty
and produces the lower-confidence imagined reward

\[
  \tilde r(s,a)=\hat r(s,a)-\beta\,\operatorname{Std}_m[\hat r_m(s,a)].
\]

No stochastic transition samples appear in that variance, so aleatoric noise
is not relabelled epistemic. This compact version covers reward-head
uncertainty; it does not claim full transition-posterior uncertainty.

## 5. Exposure-drift controls

Trajectory-iMF can add one auxiliary exposure expectation using a mixture of
slightly corrupted posterior histories and stopped self-generated histories.
It can also add direct endpoint prediction
\(\hat x=\epsilon-u_\theta(\epsilon,c,0,1)\). The controls are independently
ablatable through `PolicyConsistencyConfig`: `exposure_meanflow_scale`,
`endpoint_scale`, `generated_context_probability`,
`context_corruption_max`, and `training_context_length`. The context-length
field is enforced against the supplied batch so a claimed longer context must
actually be present.

## Gated execution and limitations

The first run changes only actor/value training on frozen checkpoints and
evaluates the shared probes. Advantage-consistency world-model training is
activated only if that actor repair fails. Epistemic pessimism is activated
when shared probes show unsupported value optimism. Exposure controls are
activated when fixed-state probes are healthy but error grows in closed-loop
rollouts. Each intervention receives a new output root and source commit.

The two-seed Reacher study is exploratory. It can diagnose a mechanism and
select a confirmatory design, but it cannot support a NeurIPS-level superiority claim.
Such a claim still requires the registered multi-task, multi-seed,
matched-compute evaluation with uncertainty intervals.
