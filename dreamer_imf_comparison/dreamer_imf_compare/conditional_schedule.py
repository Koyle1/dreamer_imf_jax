"""Fixed, outcome-independent controls and full-history diagnostic selection."""

import numpy as np

CONTROLS = dict(
    actor_enabled=True,
    transition_only=False,
    max_gap=1.0,
    sample_steps=4,
    imag_horizon=15,
    dyn_scale=1.0,
)


class ConditionalSchedule:
    def controls(self, native):
        return dict(CONTROLS)

    def observe(self, native, metrics):
        # Diagnostics never gate learning or select a policy by evaluation outcomes.
        return None


def retained_views(batch):
    views = {"ordinary_prefix": {k: v[:2].copy() for k, v in batch.items()}}
    # Select only a row with an eligible endpoint after five transitions.
    valid = np.zeros_like(batch["is_first"], bool)
    for t in range(5, valid.shape[1]):
        valid[:, t] = ~batch["is_first"][:, t - 4 : t + 1].any(1)
    hits = np.argwhere(valid & (batch["reward"] > 0))
    if len(hits):
        row = int(hits[0, 0])
        views["positive_selected"] = {
            k: v[row : row + 1].copy() for k, v in batch.items()
        }
    return views
