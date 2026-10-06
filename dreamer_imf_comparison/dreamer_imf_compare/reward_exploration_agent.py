"""Detached control reward and bootstrapped-disagreement continuation agent.

The parent world-model objective, optimizer, and parameter names are unchanged.
Auxiliary models consume the pre-update posterior already produced by that
objective. No auxiliary loss differentiates through the learned representation.
All arms train identical exploration auxiliaries; collection mode is external.
"""

import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax
from dreamerv3 import agent as upstream, rssm

from .conditional_dynamics import ConditionalAgent, ConditionalIMFRSSM
from .staged_dynamics import control, parameter_group

sg = jax.lax.stop_gradient
ENSEMBLE_SIZE = 5
AUX_UNITS = 128
AUX_LR = 3e-4
BONUS_CAP = 10.0
BOOTSTRAP_PROBABILITY = 0.5
BOOTSTRAP_SEED = 9191
IMAG_HORIZON = 15


class RewardCategorical:
    """Categorical labels 0/1/2; prediction is the raw-reward expectation."""

    def __init__(self, logits):
        self.logits = logits.astype(jnp.float32)

    def pred(self):
        return (jax.nn.softmax(self.logits) * jnp.arange(3)).sum(-1)

    def loss(self, target):
        # No bin clipping: an unsupported real label is an invalid experiment.
        labels = jax.nn.one_hot(target.astype(jnp.int32), 3)
        valid = (target == target.astype(jnp.int32)) & (target >= 0) & (target <= 2)
        return jnp.where(
            valid, -(labels * jax.nn.log_softmax(self.logits)).sum(-1), jnp.nan
        )


class CategoricalControlHead(nj.Module):
    def __call__(self, features, bdims):
        x = sg(features).astype(jnp.float32)
        x = x.reshape((*x.shape[:bdims], -1))
        mean = sg(self.value("input_mean", jnp.zeros, (x.shape[-1],), jnp.float32))
        std = sg(self.value("input_std", jnp.ones, (x.shape[-1],), jnp.float32))
        x = nn.cast((x - mean) / jnp.maximum(std, 1e-6))
        for i in range(2):
            x = jax.nn.relu(self.sub(f"hidden{i}", nn.Linear, AUX_UNITS)(x))
        return RewardCategorical(self.sub("logits", nn.Linear, 3)(x))


def build_control_head(kind, config):
    """Shared online/offline factory, with identical raw Ninjax export keys."""
    if kind == "existing":
        return embodied.jax.MLPHead(
            elements.Space(np.float32, ()), **config.rewhead, name="control_rew"
        )
    if kind == "categorical":
        return CategoricalControlHead(name="control_rew")
    raise ValueError(f"Unknown control reward head: {kind}")


def episode_bootstrap(episode_id, seed=0):
    """Independent Bernoulli membership, fixed for each entire episode/member.

    Stateless fold-in makes membership invariant to batch order and replay chunk
    boundaries, unlike drawing a fresh per-transition mask each learner call.
    """
    flat = jnp.asarray(episode_id, jnp.uint32).reshape(-1)
    seeds = jax.random.split(jax.random.PRNGKey(seed), ENSEMBLE_SIZE)
    member = lambda key: jax.vmap(
        lambda eid: jax.random.bernoulli(
            jax.random.fold_in(key, eid), BOOTSTRAP_PROBABILITY
        )
    )(flat)
    return jax.vmap(member)(seeds).reshape((ENSEMBLE_SIZE, *episode_id.shape))


def transition_mask(is_first, is_last, episode_id):
    """Validity of (posterior[t], executed action[t], observation[t+1])."""
    return (
        (~is_first[:, 1:])
        & (~is_last[:, :-1])
        & (episode_id[:, :-1] == episode_id[:, 1:])
    )


def predicted_mean_disagreement(means):
    """Variance over independently learned predicted means, not particles."""
    means = means.astype(jnp.float32)
    # Translation invariance avoids numerical pseudo-disagreement for identical
    # large predictions and leaves the population variance definition unchanged.
    centered = means - means[:1]
    return jnp.mean(jnp.var(centered, axis=0), axis=-1)


class DisagreementEnsemble(nj.Module):
    observation_dim: int = 6

    def __call__(self, features, actions):
        # Read every fixed state entry on every forward: Ninjax differentiation
        # requires all module entries to be accessed even though these are fixed.
        self.statistics()
        x = sg(features).astype(jnp.float32)
        a = sg(actions).astype(jnp.float32)
        mean = sg(self.value("input_mean", jnp.zeros, (x.shape[-1],), jnp.float32))
        std = sg(self.value("input_std", jnp.ones, (x.shape[-1],), jnp.float32))
        x = nn.cast(jnp.concatenate([(x - mean) / jnp.maximum(std, 1e-6), a], -1))
        predictions = []
        for member in range(ENSEMBLE_SIZE):
            hidden = x
            for layer in range(2):
                hidden = jax.nn.relu(
                    self.sub(f"member{member}_hidden{layer}", nn.Linear, AUX_UNITS)(
                        hidden
                    )
                )
            predictions.append(
                self.sub(f"member{member}_output", nn.Linear, self.observation_dim)(
                    hidden
                )
            )
        return jnp.stack(predictions).astype(jnp.float32)

    def normalize_target(self, target):
        mean, std, _ = self.statistics()
        return (sg(target).astype(jnp.float32) - mean) / jnp.maximum(std, 1e-6)

    def statistics(self):
        mean = sg(
            self.value("target_mean", jnp.zeros, (self.observation_dim,), jnp.float32)
        )
        std = sg(
            self.value("target_std", jnp.ones, (self.observation_dim,), jnp.float32)
        )
        scale = sg(self.value("bonus_scale", jnp.ones, (), jnp.float32))
        return mean, std, scale

    def bonus(self, features, actions):
        _, _, scale = self.statistics()
        disagreement = predicted_mean_disagreement(self(features, actions))
        return sg(jnp.clip(disagreement / jnp.maximum(scale, 1e-8), 0, BONUS_CAP))


def build_ensemble(config=None, observation_dim=6):
    return DisagreementEnsemble(
        observation_dim=observation_dim, name="explore_ensemble"
    )


def action_tensor(actions, act_space):
    """Stable action layout; discrete spaces use one-hot actual actions."""
    result = []
    for key in sorted(act_space):
        action, space = actions[key], act_space[key]
        if space.discrete:
            classes = np.asarray(space.classes).reshape(-1)
            if not (classes == classes[0]).all():
                raise ValueError("Heterogeneous discrete actions are unsupported")
            action = jax.nn.one_hot(action.astype(jnp.int32), int(classes[0]))
            event_dims = len(space.shape) + 1
        else:
            event_dims = len(space.shape)
        batch_shape = action.shape[:-event_dims] if event_dims else action.shape
        result.append(action.reshape((*batch_shape, -1)).astype(jnp.float32))
    return jnp.concatenate(result, -1)


class RewardExplorationAgent(ConditionalAgent):
    control_kind = "categorical"

    def __init__(self, obs_space, act_space, config):
        super().__init__(obs_space, act_space, config)
        self.observation_keys = tuple(
            sorted(
                key
                for key in obs_space
                if key not in ("reward", "is_first", "is_last", "is_terminal")
            )
        )
        if any(upstream.isimage(obs_space[k]) for k in self.observation_keys):
            raise ValueError("This pilot ensemble requires real vector observations")
        observation_dim = sum(
            int(np.prod(obs_space[k].shape)) for k in self.observation_keys
        )
        self.control_rew = build_control_head(self.control_kind, config)
        self.ensemble = build_ensemble(observation_dim=observation_dim)
        self.bootstrap_seed = nj.Variable(
            jnp.array, BOOTSTRAP_SEED, jnp.int32, name="explore_bootstrap_seed"
        )
        scalar = elements.Space(np.float32, ())
        outs = {
            k: config.policy_dist_disc if v.discrete else config.policy_dist_cont
            for k, v in act_space.items()
        }
        self.explore_pol = embodied.jax.MLPHead(
            act_space, outs, **config.policy, name="explore_pol"
        )
        self.explore_val = embodied.jax.MLPHead(
            scalar, **config.value, name="explore_val"
        )
        self.explore_slowval = embodied.jax.SlowModel(
            embodied.jax.MLPHead(scalar, **config.value, name="explore_slowval"),
            source=self.explore_val,
            **config.slowvalue,
        )
        self.explore_retnorm = embodied.jax.Normalize(
            **config.retnorm, name="explore_retnorm"
        )
        self.explore_valnorm = embodied.jax.Normalize(
            **config.valnorm, name="explore_valnorm"
        )
        self.explore_advnorm = embodied.jax.Normalize(
            **config.advnorm, name="explore_advnorm"
        )
        # Do not append anything to self.modules or replace self.opt: checkpoints
        # retain every original parameter/optimizer namespace verbatim.
        self.control_opt = embodied.jax.Optimizer(
            self.control_rew, optax.adam(AUX_LR), name="control_opt"
        )
        self.ensemble_opt = embodied.jax.Optimizer(
            self.ensemble, optax.adam(AUX_LR), name="ensemble_opt"
        )
        self.explore_opt = embodied.jax.Optimizer(
            [self.explore_pol, self.explore_val], optax.adam(AUX_LR), name="explore_opt"
        )

    @property
    def policy_keys(self):
        return "^(enc|dyn|dec|pol|explore_pol|rew|control_rew|explore_ensemble)/"

    @property
    def ext_space(self):
        return dict(super().ext_space, episode_id=elements.Space(np.int32))

    def imagined_reward(self, features, bdims):
        return sg(self.control_rew(sg(features), bdims).pred())

    def policy(self, carry, obs, mode="train"):
        enc_carry, dyn_carry, dec_carry, prevact = carry
        kw = dict(training=False, single=True)
        reset = obs["is_first"]
        enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
        dyn_carry, dyn_entry, feat = self.dyn.observe(
            dyn_carry, tokens, prevact, reset, **kw
        )
        dec_entry = {}
        if dec_carry:
            dec_carry, dec_entry, _ = self.dec(dec_carry, feat, reset, **kw)
        actor = self.explore_pol if mode == "explore" else self.pol
        features = self.feat2tensor(feat)
        distribution = actor(features, bdims=1)
        act = upstream.sample(distribution)
        out = {
            "finite": elements.tree.flatdict(
                jax.tree.map(
                    lambda x: jnp.isfinite(x).all(tuple(range(1, x.ndim))),
                    dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act),
                )
            )
        }
        if self.config.replay_context:
            out.update(
                elements.tree.flatdict(
                    dict(enc=enc_entry, dyn=dyn_entry, dec=dec_entry)
                )
            )
        out.update(
            {
                "log/control_reward": self.control_rew(sg(features), 1).pred(),
                "log/original_reward": self.rew(features, 1).pred(),
                "log/disagreement": self.ensemble.bonus(
                    features, action_tensor(act, self.act_space)
                ),
                "log/feature_rms": jnp.sqrt(
                    jnp.mean(features.astype(jnp.float32) ** 2, -1)
                ),
                "log/action_entropy": sum(
                    value.entropy() for value in distribution.values()
                ),
            }
        )
        mean = nj.context()["explore_ensemble/input_mean"]
        std = nj.context()["explore_ensemble/input_std"]
        out["log/feature_zscore_rms"] = jnp.sqrt(
            jnp.mean(((features - mean) / jnp.maximum(std, 1e-6)) ** 2, -1)
        )
        # The selected action, never the unused task actor sample, is recurrent.
        return (enc_carry, dyn_carry, dec_carry, act), act, out

    def _control_loss(self, features, reward):
        output = self.control_rew(sg(features), 2)
        loss = output.loss(sg(reward)).mean()
        return loss, {"control/mse": jnp.mean((output.pred() - reward) ** 2)}

    def _ensemble_loss(self, features, actions, target, valid, episode_id):
        prediction = self.ensemble(sg(features), sg(actions))
        normalized = self.ensemble.normalize_target(target)
        error = jnp.mean((prediction - normalized[None]) ** 2, -1)
        mask = episode_bootstrap(episode_id, self.bootstrap_seed.read()) & valid[None]
        count = mask.sum(tuple(range(1, mask.ndim)))
        member_loss = jnp.where(mask, error, 0).sum(
            tuple(range(1, mask.ndim))
        ) / jnp.maximum(count, 1)
        raw = predicted_mean_disagreement(prediction)
        return member_loss.mean(), {
            "ensemble/mse": jnp.where(valid[None], error, 0).sum()
            / jnp.maximum(valid.sum() * ENSEMBLE_SIZE, 1),
            "ensemble/valid": valid.sum(),
            "ensemble/membership": mask.sum(),
            "ensemble/min_member_count": count.min(),
            "ensemble/disagreement": jnp.where(valid, raw, 0).sum()
            / jnp.maximum(valid.sum(), 1),
            "ensemble/target_z_rms": jnp.sqrt(jnp.mean(normalized**2)),
        }

    def _explore_loss(self, starts, first, training=True):
        starts, first = sg((starts, first))
        policyfn = lambda feat: upstream.sample(
            self.explore_pol(sg(self.feat2tensor(feat)), 1)
        )
        _, imagined, actions = self.dyn.imagine(
            starts, policyfn, IMAG_HORIZON, training
        )
        feats = upstream.concat([first, sg(imagined)], 1)
        lastact = jax.tree.map(
            lambda x: x[:, None], policyfn(jax.tree.map(lambda x: x[:, -1], feats))
        )
        actions = sg(upstream.concat([actions, lastact], 1))
        inp = sg(self.feat2tensor(feats))
        # imag_loss uses reward[t+1] for action[t]. The uncertainty of a proposed
        # current action belongs to its successor reward, not to its predecessor.
        transition_bonus = self.ensemble.bonus(
            inp[:, :-1],
            action_tensor(jax.tree.map(lambda x: x[:, :-1], actions), self.act_space),
        )
        bonus = jnp.concatenate(
            [jnp.zeros_like(transition_bonus[:, :1]), transition_bonus], 1
        )
        losses, _, metrics = upstream.imag_loss(
            actions,
            bonus,
            sg(self.con(inp, 2).prob(1)),
            self.explore_pol(inp, 2),
            self.explore_val(inp, 2),
            self.explore_slowval(inp, 2),
            self.explore_retnorm,
            self.explore_valnorm,
            self.explore_advnorm,
            update=training,
            contdisc=self.config.contdisc,
            horizon=self.config.horizon,
            **self.config.imag_loss,
        )
        total = sum(loss.mean() * self.scales[name] for name, loss in losses.items())
        metrics.update({f"loss/{name}": loss.mean() for name, loss in losses.items()})
        metrics["bonus_cap_fraction"] = jnp.mean(transition_bonus >= BONUS_CAP)
        return total, upstream.prefix(sg(metrics), "explore")

    def train(self, carry, data):
        for variable in self.controls.values():
            variable.read()
        before = dict(nj.context())
        carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
        metrics, (carry, entries, outs, mets) = self.opt(
            self.loss, carry, obs, prevact, training=True, has_aux=True
        )
        metrics.update(mets)
        self.slowval.update()
        # Preserve the staged parent's exact inactive-state behavior.
        for key, old in before.items():
            if key.startswith(("slowval/", "retnorm/", "valnorm/", "advnorm/")):
                nj.context()[key] = jnp.where(
                    control("actor_enabled"), nj.context()[key], old
                )
            elif key.startswith(
                ("enc/", "dyn/", "dec/", "rew/", "con/", "pol/", "val/")
            ):
                group = parameter_group(key)
                if group != "transition":
                    active = (
                        control("actor_enabled")
                        if group == "actor"
                        else ~control("transition_only")
                    )
                    nj.context()[key] = jnp.where(active, nj.context()[key], old)

        features = sg(self.feat2tensor(outs["repfeat"]))
        B, T = obs["is_first"].shape
        # Replay context truncation always drops the same leading K transitions.
        episode_id = data["episode_id"][:, -T:]
        actions = action_tensor(
            {k: data[k][:, -T:] for k in self.act_space}, self.act_space
        )
        observations = jnp.concatenate(
            [
                obs[key].reshape((B, T, -1)).astype(jnp.float32)
                for key in self.observation_keys
            ],
            -1,
        )
        control_metrics, control_aux = self.control_opt(
            self._control_loss, features, obs["reward"], has_aux=True
        )
        ensemble_metrics, ensemble_aux = self.ensemble_opt(
            self._ensemble_loss,
            features[:, :-1],
            actions[:, :-1],
            observations[:, 1:],
            transition_mask(obs["is_first"], obs["is_last"], episode_id),
            episode_id[:, :-1],
            has_aux=True,
        )
        K = min(self.config.imag_last or T, T)
        starts = sg(self.dyn.starts(entries[1], carry[1], K))
        first = sg(
            jax.tree.map(
                lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), outs["repfeat"]
            )
        )
        explore_metrics, explore_aux = self.explore_opt(
            self._explore_loss, starts, first, has_aux=True
        )
        self.explore_slowval.update()
        for extra in (
            control_metrics,
            control_aux,
            ensemble_metrics,
            ensemble_aux,
            explore_metrics,
            explore_aux,
        ):
            metrics.update(extra)
        metrics["control/feature_rms"] = jnp.sqrt(jnp.mean(features**2))
        result = {}
        if self.config.replay_context:
            result["replay"] = elements.tree.flatdict(
                dict(stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2])
            )
        carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
        return carry, result, metrics


def install(arm):
    """Install in a fresh process before construction; preserve parent RSSM."""
    if arm not in ("A", "B", "C", "D"):
        raise ValueError("Reward/exploration pilot arm must be A, B, C or D")
    rssm.RSSM = ConditionalIMFRSSM
    kind = "existing" if arm == "A" else "categorical"
    return type(
        f"RewardExplorationAgent{arm}",
        (RewardExplorationAgent,),
        {"control_kind": kind},
    )
