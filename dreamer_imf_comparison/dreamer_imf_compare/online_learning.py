"""Episode-safe, fixed-shape mixed replay for warm-start online learning.

Replay entry t stores the action and reward arriving at observation t. Native
timeouts without a successor action are omitted from ReBRAC (as offline), while
their final observation remains available to world/reward learning.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from imf_dreamer_jax.agent import jit_train_world_model
from imf_dreamer_jax.config import DreamerConfig
from imf_dreamer_jax.flowmpc import ReBRACConfig, ReBRACDataset, jit_rebrac_update
from imf_dreamer_jax.optim import init_adam
from imf_dreamer_jax.transition_reward import (
    TransitionRewardState,
    TransitionRewardConfig,
    jit_train_transition_reward_step,
)
from imf_dreamer_jax.types import AgentParams, AgentState

FIELDS = ("observations", "actions", "rewards", "continuations", "is_first")


def initialize(model, actor_seed):
    """Retain parameters and full ReBRAC state; start missing Adam states fresh."""
    cfg = model["config"]
    rc = model["rebrac_config"]
    cfg = DreamerConfig(**cfg) if isinstance(cfg, dict) else cfg
    rc = ReBRACConfig(**rc) if isinstance(rc, dict) else rc
    world = model["world"]
    reward = model["reward_world"]["reward_transition"]
    return dict(
        config=cfg,
        rebrac_config=rc,
        world=world,
        world_optimizer=init_adam(world),
        reward=reward,
        reward_optimizer=init_adam(reward),
        policy=model["policies"][actor_seed],
    )


def export_model(state, model, actor_seed):
    """Export a new model mapping, preserving normalization, anchors and metadata."""
    return {
        **model,
        "world": state["world"],
        "reward_world": {**state["world"], "reward_transition": state["reward"]},
        "policies": {**model["policies"], actor_seed: state["policy"]},
    }


def _episodes(arrays, *, offline):
    if isinstance(arrays, (list, tuple)):
        if offline:
            raise ValueError("offline replay requires explicit train_episode_ids")
        if not arrays:
            raise ValueError("mixed replay requires nonempty online episodes")
        return [ep for item in arrays for ep in _episodes(item, offline=False)]
    if arrays.get("training") is False or arrays.get("split") in (
        "test",
        "eval",
        "evaluation",
    ):
        raise ValueError("evaluation replay cannot enter training")
    count = np.asarray(arrays["rewards"]).shape[0]
    ids = np.asarray(arrays["train_episode_ids"] if offline else np.arange(count))
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("train episode IDs must be an integer vector")
    if np.any(ids < 0) or np.any(ids >= count):
        raise ValueError("invalid training episode ID")
    if set(ids) & set(np.asarray(arrays.get("test_episode_ids", []))):
        raise ValueError("training and heldout episode IDs overlap")
    result = []
    for i in ids:
        ep = {k: np.asarray(arrays[k][i]) for k in FIELDS}
        length = len(ep["rewards"])
        if any(len(v) != length for v in ep.values()):
            raise ValueError("episode arrays have inconsistent lengths")
        if not np.all(
            np.isfinite(np.concatenate([v.reshape(-1) for v in ep.values()]))
        ):
            raise ValueError("nonfinite replay")
        if length < 2 or not ep["is_first"][0] or np.any(ep["is_first"][1:]):
            raise ValueError("replay must contain one reset per episode")
        if np.any(ep["actions"][0]) or ep["rewards"][0] != 0:
            raise ValueError("initial action/reward must be zero")
        if np.any((ep["continuations"] < 0) | (ep["continuations"] > 1)):
            raise ValueError("continuations must lie in [0,1]")
        ends = np.flatnonzero(ep["continuations"] == 0)
        if "is_last" in arrays:
            ends = np.union1d(ends, np.flatnonzero(arrays["is_last"][i]))
        if len(ends):
            ep = {k: v[: int(ends[0]) + 1] for k, v in ep.items()}
        if len(ep["rewards"]) < 2:
            raise ValueError("episode contains no transition")
        result.append(ep)
    if not result:
        raise ValueError("mixed replay requires nonempty offline and online episodes")
    return result


def _dataset(episodes):
    rows = [[] for _ in ReBRACDataset._fields]
    for ep in episodes:
        n = len(ep["rewards"])
        t = np.arange(1, n if ep["continuations"][-1] == 0 else n - 1)
        if not len(t):
            continue
        next_actions = ep["actions"][np.minimum(t + 1, n - 1)].copy()
        next_actions[t + 1 == n] = 0  # ignored by the true-terminal Bellman mask
        values = (
            ep["observations"][t - 1].reshape(len(t), -1),
            ep["actions"][t],
            ep["rewards"][t],
            ep["observations"][t].reshape(len(t), -1),
            next_actions,
            1.0 - ep["continuations"][t],
        )
        for bucket, value in zip(rows, values):
            bucket.append(value)
    if not rows[0]:
        raise ValueError("replay has no transitions with observed successor actions")
    return ReBRACDataset(*(np.concatenate(v).astype(np.float32) for v in rows))


def _sequence_batch(offline, online, rng, batch_size, sequence_length, burn_in):
    # One context slot is necessary even with configured burn-in zero: reward
    # entry t needs observation t-1, and no loss may use a fabricated predecessor.
    context = max(1, burn_in)
    width = context + sequence_length
    rows = []
    sources = (online,) if offline is None else (offline, online)
    for source in sources:
        for _ in range(batch_size // len(sources)):
            ep = source[int(rng.integers(len(source)))]
            # Production episodes provide exactly sequence_length loss tokens;
            # only short preflight/early-terminal episodes require masked tails.
            last_start = max(1, len(ep["rewards"]) - sequence_length)
            start = int(rng.integers(1, last_start + 1))
            offsets = np.arange(start - context, start + sequence_length)
            valid = (offsets >= 0) & (offsets < len(ep["rewards"]))
            ix = np.clip(offsets, 0, len(ep["rewards"]) - 1)
            row = {k: ep[k][ix].copy() for k in FIELDS}
            # Leading padding repeats the reset, keeping latent history zeroed.
            row["is_first"][:context] |= offsets[:context] <= 0
            row["loss_mask"] = (valid & (np.arange(width) >= context)).astype(
                np.float32
            )
            rows.append(row)
    return {
        k: jnp.asarray(np.stack([r[k] for r in rows])) for k in (*FIELDS, "loss_mask")
    }


def _policy_batch(offline, online, rng, size):
    pieces = []
    sources = (online,) if offline is None else (offline, online)
    for source in sources:
        ids = rng.integers(source.states.shape[0], size=size // len(sources))
        pieces.append([v[ids] for v in source])
    return ReBRACDataset(
        *(jnp.asarray(np.concatenate(columns)) for columns in zip(*pieces))
    )


def update(
    state,
    offline,
    online,
    *,
    full_model,
    seed,
    world_updates,
    policy_updates,
    batch_size=16,
    sequence_length=32,
):
    """Use online-only replay if offline=None, otherwise explicit 50/50 replay."""
    for name, value in (
        ("world_updates", world_updates),
        ("policy_updates", policy_updates),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if batch_size < 2 or batch_size % 2 or sequence_length <= 0:
        raise ValueError(
            "positive sequence length and positive even batch size required"
        )
    old = None if offline is None else _episodes(offline, offline=True)
    new = _episodes(online, offline=False)
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    result = dict(state)
    cfg, rc = state["config"], state["rebrac_config"]
    metrics = {
        "world_updates": 0,
        "reward_updates": 0,
        "policy_updates": 0,
        "offline_fraction": 0.0 if offline is None else 0.5,
    }
    if full_model:
        empty = init_adam({})
        world = AgentState(
            AgentParams(state["world"], {}, {}),
            state["world_optimizer"],
            empty,
            empty,
            {},
            None,
        )
        head = TransitionRewardState(state["reward"], state["reward_optimizer"])
        for i in range(world_updates):
            batch = _sequence_batch(
                old, new, rng, batch_size, sequence_length, cfg.burn_in
            )
            world, wm = jit_train_world_model(
                world, batch, jax.random.fold_in(key, 2 * i), cfg
            )
            head, rm = jit_train_transition_reward_step(
                head,
                world.params.world_model,
                batch,
                jax.random.fold_in(key, 2 * i + 1),
                cfg,
                TransitionRewardConfig(),
            )
            metrics.update(world_loss=float(wm.total), reward_loss=float(rm.loss))
        result.update(
            world=world.params.world_model,
            world_optimizer=world.model_optimizer,
            reward=head.params,
            reward_optimizer=head.optimizer,
        )
        metrics.update(world_updates=world_updates, reward_updates=world_updates)
    if policy_updates:
        a, b = None if old is None else _dataset(old), _dataset(new)
        for i in range(policy_updates):
            result["policy"], pm = jit_rebrac_update(
                result["policy"],
                _policy_batch(a, b, rng, rc.batch_size),
                jax.random.fold_in(key, 2 * world_updates + i),
                rc,
            )
            metrics.update({f"policy_{k}": float(v) for k, v in pm._asdict().items()})
        metrics["policy_updates"] = policy_updates
    if not all(np.isfinite(v) for v in metrics.values()) or not all(
        np.all(np.isfinite(np.asarray(x)))
        for name in ("world", "world_optimizer", "reward", "reward_optimizer", "policy")
        for x in jax.tree_util.tree_leaves(result[name])
    ):
        raise FloatingPointError("nonfinite online learner state or metrics")
    return result, metrics
