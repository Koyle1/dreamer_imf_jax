# Advantage-consistency repair study

This is an exploratory Reacher mechanism test activated after the frozen-world
actor repair failed to remove trajectory-iMF's action-ranking deficit. It is
not matched-compute superiority evidence.

The study reads the completed policy-consistency vNext result and never writes
into it. For each world-model seed, it constructs one immutable train-split
simulator probe bank before loading an iMF checkpoint. Thirty-two decision
states are selected from 1,024 replay locations by simulator return range at
horizon 5. Candidate trajectories are the replay suffix and coordinatewise
plus/minus 0.5 perturbations for five steps. At least 16 retained states must
be informative.

Each update uses two batches. The ordinary world-model loss receives a
canonical random replay batch. The advantage term receives a separate replay
history ending at a labeled decision state, so every row has exactly one
supervised candidate set. Simulator targets and source posterior states are
stopped. Common transition noise is shared across candidates. The first
variant adds only

\[
0.1\left(0.25\,L_{\mathrm{magnitude}} + L_{\mathrm{rank}}
          + 0.1\,L_{\mathrm{flat}}\right)
\]

for 1,000 fine-tuning updates. The first `advantage_reward` variant freezes
the encoder, decoder, deterministic transition, posterior, continuation head,
and iMF prior—including their Adam moments—and updates only the reward head.
This directly tests whether counterfactual reward geometry is the failure while
state dynamics remain adequate. `advantage` unfreezes the whole world model;
`advantage_endpoint` and `advantage_exposure` are separately named variants;
they are run only if their corresponding diagnostics activate them. Likewise,
epistemic reward pessimism is an actor-side scale ablation rather than part of
the base advantage repair.

After fine-tuning, each iMF checkpoint is evaluated on the exact held-out
probe bank used for the frozen shortcut comparison. A useful signal requires
improved pairwise accuracy, Spearman ordering, and simulator regret without
using the held-out bank for optimization. Any subsequent actor comparison uses
the same actor initialization, replay schedule, sign-only PMPO, frozen behavior
prior, reverse-KL scale 0.3, distributional critic, and replay grounding.

Two Reacher seeds can identify a mechanism and reject bad variants. They cannot
establish a NeurIPS-level general result.
