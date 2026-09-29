"""Artifact-free random initialization using only newly collected prefill data."""

from __future__ import annotations

import jax
import numpy as np

from imf_dreamer_jax.config import DreamerConfig
from imf_dreamer_jax.flowmpc import ReBRACConfig, init_rebrac_state
from imf_dreamer_jax.transition_reward import init_transition_reward_state
from imf_dreamer_jax.world_model import init_world_model

from .online_learning import _episodes


def _random_parameters(cfg, rc, seed, actor_seed, mean, std, reward_hidden_dim):
    """Canonical CPU initialization, exported without retained device placement.

    GPU and login-node verification must execute the same initializer backend.
    NumPy output lets subsequent JIT learning select its normal default device;
    this local context does not change the process-wide JAX platform setting.
    """
    with jax.default_device(jax.devices("cpu")[0]):
        world_key, reward_key, policy_key = jax.random.split(
            jax.random.PRNGKey(seed), 3
        )
        world = init_world_model(cfg, world_key)
        reward = init_transition_reward_state(
            cfg,
            reward_key,
            observation_mean=mean,
            observation_std=std,
            hidden_dim=reward_hidden_dim,
        ).params
        policy = init_rebrac_state(jax.random.fold_in(policy_key, actor_seed), rc)
        return jax.tree_util.tree_map(
            lambda value: np.asarray(value).copy(), (world, reward, policy)
        )


def initialize_model(protocol, seed, episodes=None):
    """Return random weights, zero clocks, and own-prefill normalization/anchors.

    This interface accepts arrays, never checkpoints or replay paths. With no
    episodes it supplies identity statistics and zero anchors solely to bootstrap
    random collection. Calling again with prefill retains the same random weight
    initialization, not any learned parameters from the bootstrap model.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be a uint32 integer")
    template = dict(protocol["world_model_template"])
    rows = None
    if episodes is not None:
        supplied = list(episodes) if isinstance(episodes, (list, tuple)) else [episodes]
        if not supplied:
            raise ValueError("prefill must be nonempty")
        for item in supplied:
            if not isinstance(item, dict) or item.get("training") is not True:
                raise ValueError("prefill requires explicitly training=True data")
            if "train_episode_ids" in item or "test_episode_ids" in item:
                raise ValueError("prefill cannot use inherited offline replay splits")
        clean = _episodes(supplied, offline=False)
        shape = tuple(clean[0]["observations"].shape[1:])
        action_dim = clean[0]["actions"].shape[-1]
        for ep in clean:
            if ep["observations"].shape[1:] != shape or ep["actions"].shape != (
                len(ep["rewards"]),
                action_dim,
            ):
                raise ValueError("inconsistent prefill observation/action dimensions")
            if np.any(np.abs(ep["actions"]) > 1):
                raise ValueError("prefill action outside normalized bounds")
        rows = np.concatenate(
            [ep["observations"].reshape(-1, int(np.prod(shape))) for ep in clean]
        )
        if (
            "observation_shape" in template
            and tuple(template["observation_shape"]) != shape
        ):
            raise ValueError("prefill observation shape differs from protocol")
        if "action_dim" in template and template["action_dim"] != action_dim:
            raise ValueError("prefill action dimension differs from protocol")
        template.update(observation_shape=shape, action_dim=action_dim)
    elif "observation_shape" not in template or "action_dim" not in template:
        raise ValueError("zero-data initialization requires explicit dimensions")
    for name in ("observation_shape", "overshooting_distances"):
        if name in template:
            template[name] = tuple(template[name])
    cfg = DreamerConfig(**template)
    state_dim = int(np.prod(cfg.observation_shape))
    settings = dict(protocol["rebrac_config"])
    for name, value in (("state_dim", state_dim), ("action_dim", cfg.action_dim)):
        if name in settings and settings[name] != value:
            raise ValueError("ReBRAC dimensions differ from world model")
        settings[name] = value
    rc = ReBRACConfig(**settings)
    anchor_count = protocol.get("anchor_count", 64)
    if (
        isinstance(anchor_count, bool)
        or not isinstance(anchor_count, int)
        or anchor_count <= 0
    ):
        raise ValueError("anchor_count must be a positive integer")
    if rows is None:
        mean, std = np.zeros(state_dim, np.float32), np.ones(state_dim, np.float32)
        anchors = np.zeros((anchor_count, state_dim), np.float32)
    else:
        mean = rows.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = np.maximum(rows.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
        ids = np.linspace(0, len(rows) - 1, anchor_count, dtype=np.int64)
        anchors = rows[ids].astype(np.float32).copy()
    actor_seed = protocol["actor_seed"]
    if (
        isinstance(actor_seed, bool)
        or not isinstance(actor_seed, int)
        or not 0 <= actor_seed < 2**32
    ):
        raise ValueError("actor_seed must be a uint32 integer")
    world, reward, policy = _random_parameters(
        cfg, rc, seed, actor_seed, mean, std, protocol.get("reward_hidden_dim", 512)
    )
    return dict(
        config=cfg,
        rebrac_config=rc,
        world=world,
        reward_world={**world, "reward_transition": reward},
        policies={actor_seed: policy},
        anchors=anchors,
        task=protocol["task"],
    )
