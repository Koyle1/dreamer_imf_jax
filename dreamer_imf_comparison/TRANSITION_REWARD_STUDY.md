# Direct transition-reward intervention

This is an exploratory, iMF-only Reacher study. It asks whether the actor
failure is caused by the state-only reward decoder rather than the trajectory
dynamics.

The intervention replaces the scalar reward prediction with

\[
\hat r_t = g_\xi\!\left(z_t, a_t, z_{t+1}-z_t\right),
\qquad
\mathcal L_r =
\frac{\sum_t m_t(\hat r_t-r_t)^2}{\sum_t m_t}.
\]

`g` is a three-hidden-layer MLP. Only its parameters are optimized. The
encoder, posterior, recurrent state transition, trajectory-iMF prior,
observation decoder, continuation head, and original reward decoder remain
bitwise frozen. When the direct head is present it predicts the complete
reward; it is not added as a residual.

The study uses the selected trajectory-iMF Reacher candidate from the
completed corrected-actor pilot, three world-model seeds, and the two nested
actor seeds. Actor initialization, replay minibatches, evaluation seeds,
500 preparation updates, and 10,000 percentile-EMA-normalized REINFORCE
updates are exactly paired with the immutable selected iMF and shortcut
references. Shortcut is not retrained.

Primary diagnostic: the world-seed IQM of real normalized actor return for
the direct head versus the original iMF and shortcut references. Supporting
diagnostics are held-out one-step reward MSE, action saturation, actor-gradient
norm, and the imagined-versus-real per-step return gap.

This cannot support a publication claim by itself: it is one selected task,
uses pilot selection data, and adds a small reward MLP. A positive result
identifies the reward interface as a plausible bottleneck and justifies a
pre-registered multi-task confirmatory study.
