# Decisive actor-failure diagnostic

This frozen Reacher Easy study asks which interface first breaks between a
trajectory-iMF world model and PMPO actor learning. It does not tune the model
and it does not treat actor restarts or evaluation episodes as independent
world-model replicates.

The six paired actor arms are:

1. `bc_no_pmpo`: behavior-cloned policy before imagination optimization.
2. `base_mtp_pmpo`: the existing multi-token reward head and learned continuation.
3. `residual_pmpo`: the exact dense-quadratic residual checkpoint.
4. `analytic_reward_pmpo`: exact Reacher reward computed from the decoded imagined next observation.
5. `analytic_reward_unit_continuation_pmpo`: arm 4 with continuation fixed to one.
6. `synthetic_action_reward_pmpo`: known concave action reward with continuation fixed to one.

A seventh random-policy cell anchors environment difficulty. Every actor arm
uses world-model seeds 211 and 223, fresh actor seeds 311 and 313, imagination
horizons 5 and 15, and the same 50 environment reset seeds. Actor initialization,
replay batches, posterior randomness, imagination starts, and objective randomness
are paired across arms for a fixed world-model seed, actor seed, and horizon.

The analytic reward is an intervention on the learned reward head, not a true
simulator oracle: it still depends on the model transition and decoder. The
synthetic action reward is only an actor/critic competence control; its real
environment return is not evidence of task performance.

Interpretation is ordered and descriptive because there are only two independent
world-model seeds. Synthetic-control failure stops world-model attribution.
Otherwise the study tests residual rollback, analytic reward replacement, and
continuation removal in that order. An effect must exceed 0.005 normalized return
on average and have the same positive sign after averaging conditional repeats
within both world-model seeds.
