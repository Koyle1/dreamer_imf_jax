# ITPO-style state-action reward intervention

This is an exploratory, iMF-only Reacher study. It asks whether the actor
failure is caused by our Dreamer-style reward interface rather than the
trajectory dynamics. The replacement follows the reward model in ITPO's
one-step MeanFlow experiment (FlowMPC).

The intervention replaces scalar reward prediction with

\[
\hat r_t = g_\xi\!\left(\operatorname{standardize}(\hat s_t), a_t\right),
\qquad
\mathcal L_r =
\frac{\sum_t m_t(\hat r_t-r_t)^2}{\sum_t m_t}.
\]

`g` has three width-512 hidden layers with ReLU activations and a scalar linear
output. State statistics are computed from training episodes only; actions are
not normalized. Replay training uses scalar MSE, learning rate `3e-4`, and
global gradient clipping at `1.0`. During imagination, the current latent is
decoded into environment observation space before evaluating `g`. Thus reward
has the published causal form `r(s_t, a_t)` and does not depend on the sampled
next state. Only `g` is optimized. The encoder, posterior, recurrent state
transition, trajectory-iMF prior, observation decoder, continuation head, and
original reward decoder remain bitwise frozen.

The paper trains for 200,000 updates with batch size 2,048 on D4RL datasets.
This selected diagnostic deliberately retains the existing matched pilot's
10,000-update budget and replay schedule; it tests the interface and
architecture rather than claiming a full reproduction of the paper.

The study uses the selected trajectory-iMF Reacher candidate from the
completed corrected-actor pilot, three world-model seeds, and the two nested
actor seeds. Actor initialization, replay minibatches, evaluation seeds,
500 preparation updates, and 10,000 percentile-EMA-normalized REINFORCE
updates are exactly paired with the immutable selected iMF and shortcut
references. Shortcut is not retrained.

Primary diagnostic: the world-seed IQM of real normalized actor return for
the state-action head versus the original iMF and shortcut references. Supporting
diagnostics are held-out one-step reward MSE, action saturation, actor-gradient
norm, and the imagined-versus-real per-step return gap.

This cannot support a publication claim by itself: it is one selected task,
uses pilot selection data, and adds a small reward MLP. A positive result
identifies the reward interface as a plausible bottleneck and justifies a
pre-registered multi-task confirmatory study.
