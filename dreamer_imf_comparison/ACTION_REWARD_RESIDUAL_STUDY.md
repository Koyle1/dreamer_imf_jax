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

For the existing pseudo-Huber penalty, adding the exact difference between the uncentered and centered losses simplifies algebraically to the uncentered pseudo-Huber loss. The second version therefore used that single expression directly over horizons 1, 3, and 5. There was no calibration coefficient, ranking term, flat-region term, or output-projection regularizer. The completed centered study's candidate-independent running-RMS normalization was retained so centering was the only objective change.

The third, dense-horizon version isolates the remaining temporal-identification gap. If $e_k$ denotes per-step reward error, the sparse objective observes only

\[
\sum_{k=0}^{H-1}\gamma^k e_k,\qquad H\in\{1,3,5\}.
\]

Those three projections cannot identify a 15-step error vector. The dense objective keeps the same uncentered pseudo-Huber expression but supervises every prefix $H=1,\ldots,15$. Its lower-triangular prefix-sum operator has a nonzero diagonal and is therefore full rank. No new loss term or fitted coefficient is introduced.

The old training bank contains labels only at horizons 1, 3, 5, and 15. The dense study therefore does not interpolate missing labels. It replays the exact same simulator states and candidate action suffixes, regenerates all 15 cumulative returns, and requires the regenerated legacy labels to agree within $10^{-6}$. The original bank and both earlier result trees remain immutable. The dense study uses a new schema and output root and records matched deltas against both earlier residual versions.

Evaluation uses the unchanged held-out probe bank at horizons 1, 3, 5, and 15. Actor cells reuse the exact MTP actor initialization, replay schedule, PMPO objective, behavioral-prior KL, critic preparation, horizons 5 and 15, and real-environment evaluation protocol. The final report records matched deltas against the dense study's sparse raw-residual parent, the completed centered residual, the unmodified MTP reward head, and the frozen shortcut reference.

This is an exploratory two-seed causal test, not confirmatory evidence. A positive result would isolate missing action access in the reward model; a negative result would motivate thin representation unfreezing.
