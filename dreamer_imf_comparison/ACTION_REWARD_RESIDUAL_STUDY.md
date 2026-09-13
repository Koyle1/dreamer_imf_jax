# Transition-Conditioned Reward Residual Study

This append-only Reacher study tests the smallest repair implied by the frozen-representation audit. It preserves the completed trajectory-iMF dynamics and symexp two-hot MTP reward head, attaches a zero-initialized scalar residual, and trains only that residual on counterfactual returns.

The reward used during imagination is

\[
\hat r_t=b(h_t)+u(\operatorname{sg}(h_{t-1}),a_t,\operatorname{sg}(h_t-h_{t-1})).
\]

The output projection of \(u\) is initialized to zero, so attaching it leaves every source reward exactly unchanged. Latent features and actions are stop-gradient inputs. Original parameters and their Adam moments must remain bitwise unchanged during residual training.

The first version used candidate-centered pseudo-Huber regression. That objective has an exact translation nullspace: adding the same return error to every candidate leaves the loss unchanged. For squared error, the missing mean-error term and the centered term satisfy

\[
\frac1K\sum_k(e_k-\bar e)^2+\bar e^2=\frac1K\sum_k e_k^2.
\]

For the existing pseudo-Huber penalty, adding the exact difference between the uncentered and centered losses simplifies algebraically to the uncentered pseudo-Huber loss. The repaired implementation therefore uses that single expression directly over horizons 1, 3, and 5. There is no calibration coefficient, ranking term, flat-region term, or output-projection regularizer. The completed centered study's candidate-independent running-RMS normalization is retained so centering is the only objective change.

Evaluation uses the unchanged held-out probe bank at horizons 1, 3, 5, and 15. Actor cells reuse the exact MTP actor initialization, replay schedule, PMPO objective, behavioral-prior KL, critic preparation, horizons 5 and 15, and real-environment evaluation protocol. The final report records matched deltas against the completed centered residual, the unmodified MTP reward head, and the frozen shortcut reference.

This is an exploratory two-seed causal test, not confirmatory evidence. A positive result would isolate missing action access in the reward model; a negative result would motivate thin representation unfreezing.
