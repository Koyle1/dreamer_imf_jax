# Continuous actor repair study

This study diagnoses the Reacher actor failure without changing the frozen
trajectory-iMF world model. Its primary actor is continuous REINFORCE rather
than sign-PMPO:

\[
L_\pi = -\mathbb E\left[w_t\,\mathrm{sg}\left(
\frac{R_t^\lambda - V(s_t)}{S}\right)\log\pi(a_t\mid s_t)\right]
-\eta H(\pi)+\beta\,\mathrm{KL}(\pi\Vert\pi_{\mathrm{BC}}),
\]

where

\[
S=\max\left(1,\operatorname{EMA}_{0.99}
\left[P_{95}(R^\lambda)-P_5(R^\lambda)\right]\right).
\]

The running scale is explicit optimizer-side state. It is deliberately not
added to `AgentState`, so checkpoints produced before this repair remain
loadable. The preregistered primary behavior constraint is `beta=0.1`;
`beta=0.0` and `beta=0.3` are sensitivity checks and cannot replace the
primary result after seeing outcomes.

## Causal ladder

The 59-cell Reacher pilot changes one source of approximation at a time:

1. `exact_bandit_h1`: exact reward, horizon one, no dynamics and no critic.
2. `exact_finite_no_bootstrap`: exact finite-horizon returns, no dynamics,
   critic, or bootstrap.
3. `exact_reward_learned_critic`: the same exact task with a learned value
   baseline, still without world-model dynamics.
4. `analytic_reward_learned_dynamics`: frozen learned dynamics and the exact
   analytic Reacher reward computed from decoded predicted observations.
5. `learned_reward_learned_dynamics`: the frozen learned reward and dynamics.

The first three rungs pass when deterministic action MAE to the known optimum
is at most `0.25`. The last two pass when real normalized return exceeds the
shared random-policy baseline by `0.005`. Only `beta=0.1` determines the first
failing rung. Actor seeds measure conditional optimization variance; this
pilot has only one world-model seed and therefore cannot support a population
claim.

Every actor cell records deterministic actions, pre-tanh mean, policy
standard deviation, saturation, squashed entropy, behavior KL, advantage
mean and positive fraction, return scale, actor and critic gradient norms,
imagined return, real return, and their per-step gap where both exist.

## Independent gradient diagnostic

The score-function gradient for the exact action reward is compared with an
independent reparameterized Monte Carlo gradient of the same expectation.
Cosine at least `0.9` and informative-coordinate sign agreement at least
`0.8` classify the estimator as competent. Failure is reported as an
estimator mismatch; it is never converted into a model result.

## Safe multi-action MPO control

The separate MPO control draws exactly four actions from a frozen reference
policy for each identical state, assigns positive softmax weights only to the
best two, regresses the current policy toward those samples, and constrains
the mean and scale parts of `KL(reference || current)` separately. The
reference policy refreshes every 100 updates. It never assigns negative
likelihood weights, so it does not reproduce the unstable sign-PMPO
objective.

## Cluster stages

`cluster/neurips/actor_repair_study.sbatch` provides append-only `preflight`,
`smoke`, `random`, `pure`, `world`, and `finalize` stages. A stage refuses a
dirty source tree or a source-commit mismatch. Finalization verifies every
registered cell, dependency digest, shared evaluation scenario, raw return,
and the single preregistered causal decision.
