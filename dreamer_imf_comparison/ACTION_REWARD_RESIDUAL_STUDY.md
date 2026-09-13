# Transition-Conditioned Reward Residual Study

This append-only Reacher study tests the smallest repair implied by the frozen-representation audit. It preserves the completed trajectory-iMF dynamics and symexp two-hot MTP reward head, attaches a zero-initialized scalar residual, and trains only that residual on centered counterfactual returns.

The reward used during imagination is

\[
\hat r_t=b(h_t)+u(\operatorname{sg}(h_{t-1}),a_t,\operatorname{sg}(h_t-h_{t-1})).
\]

The output projection of \(u\) is initialized to zero, so attaching it leaves every source reward exactly unchanged. Latent features and actions are stop-gradient inputs. Original parameters and their Adam moments must remain bitwise unchanged during residual training.

The residual uses the already-authenticated counterfactual schedule from the Dreamer 4-style MTP arm. Its loss is normalized centered-return pseudo-Huber regression over horizons 1, 3, and 5, plus a small output-projection penalty. Separate ranking and flat-region losses are disabled because they are redundant once centered returns are matched.

Evaluation uses the unchanged held-out probe bank at horizons 1, 3, 5, and 15. Actor cells reuse the exact MTP actor initialization, replay schedule, PMPO objective, behavioral-prior KL, critic preparation, horizons 5 and 15, and real-environment evaluation protocol. The final report records matched deltas against the unmodified MTP reward head and the frozen shortcut reference.

This is an exploratory two-seed causal test, not confirmatory evidence. A positive result would isolate missing action access in the reward model; a negative result would motivate thin representation unfreezing.
