"""Scratch online adapters with explicit, trace-safe training phase controls.

Inactive optimizer groups retain both parameters and their complete optimizer
state, including clocks. This is not gradient masking masquerading as freezing.
"""

import math

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax
from dreamerv3 import agent as upstream
from dreamerv3 import rssm
from imf_dreamer_jax.imf import (
    improved_meanflow_loss,
    sample_imf_steps,
    sample_time_pairs,
)

from .dreamer_ablation_dynamics import GaussianRSSM, IMFRSSM, _UPSTREAM_RSSM

DEFAULT_CONTROLS = dict(
    actor_enabled=False,
    transition_only=False,
    max_gap=0.1,
    sample_steps=4,
    imag_horizon=5,
    dyn_scale=1.0,
)
sg = jax.lax.stop_gradient


def validate_controls(controls):
    result = dict(DEFAULT_CONTROLS, **controls)
    if set(result) != set(DEFAULT_CONTROLS):
        raise ValueError("Unknown staged controls")
    for key in ("actor_enabled", "transition_only"):
        if not isinstance(result[key], (bool, np.bool_)):
            raise ValueError(key)
    if not math.isfinite(result["max_gap"]) or not 0 < result["max_gap"] <= 1:
        raise ValueError("max_gap must be in (0, 1]")
    if result["sample_steps"] not in (1, 4):
        raise ValueError("sample_steps must be 1 or 4")
    if result["imag_horizon"] not in (5, 15):
        raise ValueError("imag_horizon must be 5 or 15")
    if not math.isfinite(result["dyn_scale"]) or not 0 < result["dyn_scale"] <= 100:
        raise ValueError("dyn_scale must be positive and at most 100")
    return result


def control(name):
    return nj.context()[f"staged_{name}/value"]


def parameter_group(path):
    if path.startswith(("pol/", "val/")):
        return "actor"
    if path.startswith("dyn/prior") or path.startswith("dyn/imf"):
        return "transition"
    return "representation"


def grouped_optimizer(base, enabled):
    """Three independent upstream optimizer states with exact inactive retention."""
    groups = ("representation", "transition", "actor")

    def split(tree, group):
        return {k: v for k, v in tree.items() if parameter_group(k) == group}

    def init(params):
        return {g: base.init(split(params, g)) for g in groups}

    def update(grads, state, params):
        updates, states = {}, {}
        for g in groups:
            upd, new = base.update(split(grads, g), state[g], split(params, g))
            active = enabled(g)
            updates.update(jax.tree.map(lambda x: jnp.where(active, x, 0), upd))
            states[g] = jax.tree.map(
                lambda x, y: jnp.where(active, x, y), new, state[g]
            )
        return updates, states

    return optax.GradientTransformation(init, update)


class StagedGaussianRSSM(GaussianRSSM):
    pass


class StagedIMFRSSM(IMFRSSM):
    def _sample_prior(self, deter):
        shape = deter.shape[:-1] + (self.stoch, self.classes)
        condition = deter.astype(jnp.float32).reshape((-1, self.deter))
        params, key = self._flow_params(), nj.seed()
        value = jax.lax.cond(
            control("sample_steps") == 1,
            lambda _: sample_imf_steps(params, condition, key, steps=1),
            lambda _: sample_imf_steps(params, condition, key, steps=4),
            None,
        )
        return value.reshape(shape)

    def _dynamics_loss(self, deter, target):
        condition = deter.astype(jnp.float32).reshape((-1, self.deter))
        target = sg(target.astype(jnp.float32)).reshape((-1, self.stoch * self.classes))
        pairkey, losskey = jax.random.split(nj.seed())
        r, t = sample_time_pairs(pairkey, target.shape[0])
        # Preserve the exact r=t boundary mass while limiting transport length.
        r = t - jnp.minimum(t - r, control("max_gap"))
        loss = improved_meanflow_loss(
            self._flow_params(),
            target,
            condition,
            losskey,
            r=r,
            t=t,
            reduction="none",
            boundary_velocity_supervision=True,
        )
        return loss.reshape(deter.shape[:-1])


for _cls in (StagedGaussianRSSM, StagedIMFRSSM):
    _cls.__annotations__ = dict(_UPSTREAM_RSSM.__annotations__)
    _cls._defaults = dict(_UPSTREAM_RSSM._defaults)


class StagedAgent(upstream.Agent):
    def __init__(self, obs_space, act_space, config):
        super().__init__(obs_space, act_space, config)
        self.controls = {
            k: nj.Variable(jnp.array, v, name=f"staged_{k}")
            for k, v in DEFAULT_CONTROLS.items()
        }

    def _make_opt(self, **kwargs):
        def enabled(group):
            if group == "actor":
                return control("actor_enabled")
            if group == "representation":
                return ~control("transition_only")
            return jnp.array(True)

        return grouped_optimizer(super()._make_opt(**kwargs), enabled)

    def train(self, carry, data):
        for variable in self.controls.values():
            variable.read()
        before = dict(nj.context())
        result = super().train(carry, data)
        # Normalize and target-network state must not learn during actor warmup.
        frozen = ("slowval/", "retnorm/", "valnorm/", "advnorm/")
        for key, old in before.items():
            if key.startswith(frozen):
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
                    # Select the original bits, including signed zero; adding
                    # a zero optimizer update alone is not a bitwise guarantee.
                    nj.context()[key] = jnp.where(active, nj.context()[key], old)
        return result

    def loss(self, carry, obs, prevact, training):
        """Upstream objectives with true truncated (bootstrapped) 5/15 horizons."""
        for variable in self.controls.values():
            variable.read()
        enc_carry, dyn_carry, dec_carry = carry
        reset = obs["is_first"]
        B, T = reset.shape
        enc_carry, enc_entries, tokens = self.enc(enc_carry, obs, reset, training)
        dyn_carry, dyn_entries, losses, repfeat, metrics = self.dyn.loss(
            dyn_carry, tokens, prevact, reset, training
        )
        dec_carry, dec_entries, recons = self.dec(dec_carry, repfeat, reset, training)
        inp = upstream.sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
        losses["rew"] = self.rew(inp, 2).loss(obs["reward"])
        con = jnp.float32(~obs["is_terminal"])
        if self.config.contdisc:
            con *= 1 - 1 / self.config.horizon
        losses["con"] = self.con(self.feat2tensor(repfeat), 2).loss(con)
        for key, recon in recons.items():
            value = obs[key]
            target = (
                jnp.float32(value) / 255
                if upstream.isimage(self.obs_space[key])
                else value
            )
            losses[key] = recon.loss(sg(target))
        K = min(self.config.imag_last or T, T)
        starts = self.dyn.starts(dyn_entries, dyn_carry, K)
        policyfn = lambda feat: upstream.sample(self.pol(self.feat2tensor(feat), 1))
        _, imagined, actions = self.dyn.imagine(starts, policyfn, 15, training)
        first = jax.tree.map(
            lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat
        )
        feats = upstream.concat(
            [upstream.sg(first, skip=self.config.ac_grads), sg(imagined)], 1
        )
        lastact = jax.tree.map(
            lambda x: x[:, None], policyfn(jax.tree.map(lambda x: x[:, -1], feats))
        )
        actions = upstream.concat([actions, lastact], 1)

        def actor_losses(horizon):
            inp = self.feat2tensor(jax.tree.map(lambda x: x[:, : horizon + 1], feats))
            act = jax.tree.map(lambda x: x[:, : horizon + 1], actions)
            los, out, mets = upstream.imag_loss(
                act,
                self.rew(inp, 2).pred(),
                self.con(inp, 2).prob(1),
                self.pol(inp, 2),
                self.val(inp, 2),
                self.slowval(inp, 2),
                self.retnorm,
                self.valnorm,
                self.advnorm,
                update=training,
                contdisc=self.config.contdisc,
                horizon=self.config.horizon,
                **self.config.imag_loss,
            )
            return (
                {k: v.mean(1).reshape((B, K)) for k, v in los.items()},
                out["ret"][:, 0].reshape(B, K),
                # A conditional's transpose can propagate zero cotangents
                # through unused metric outputs. At zero variance, std has an
                # undefined derivative and 0 * NaN poisons reward/con gradients.
                # Detach inside each branch, before the conditional boundary.
                jax.tree.map(sg, mets),
            )

        los, boot, mets = nj.cond(
            control("imag_horizon") == 5,
            lambda: actor_losses(5),
            lambda: actor_losses(15),
        )
        losses.update(los)
        metrics.update(mets)
        if self.config.repval_loss:
            feat = upstream.sg(repfeat, skip=self.config.repval_grad)
            last, term, rew = [obs[k] for k in ("is_last", "is_terminal", "reward")]
            feat, last, term, rew = jax.tree.map(
                lambda x: x[:, -K:], (feat, last, term, rew)
            )
            inp = self.feat2tensor(feat)
            los, _, mets = upstream.repl_loss(
                last,
                term,
                rew,
                boot,
                self.val(inp, 2),
                self.slowval(inp, 2),
                self.valnorm,
                update=training,
                horizon=self.config.horizon,
                **self.config.repl_loss,
            )
            losses.update(los)
            metrics.update(upstream.prefix(mets, "reploss"))
        assert set(losses) == set(self.scales)
        terms = []
        for key, value in losses.items():
            scale = self.scales[key]
            if key == "dyn":
                scale *= control("dyn_scale")
            elif key in ("policy", "value", "repval"):
                scale *= control("actor_enabled")
            terms.append(value.mean() * scale)
        metrics.update({f"loss/{k}": v.mean() for k, v in losses.items()})
        metrics.update(
            {
                f"staged/{k}": v.read().astype(jnp.float32)
                for k, v in self.controls.items()
            }
        )
        return sum(terms), (
            (enc_carry, dyn_carry, dec_carry),
            (enc_entries, dyn_entries, dec_entries),
            dict(tokens=tokens, repfeat=repfeat, losses=losses),
            metrics,
        )


def install(arm, schedule=None):
    """Call before Agent construction in a fresh arm process; returns constructor."""
    if arm not in ("gaussian", "imf"):
        raise ValueError("Staged study supports Gaussian and iMF only")
    rssm.RSSM = {"gaussian": StagedGaussianRSSM, "imf": StagedIMFRSSM}[arm]
    return StagedAgent


def set_controls(agent, controls):
    """Explicit host->device updates preserve sharding and fixed state keys."""
    controls = validate_controls(controls)
    with agent.train_lock:
        for name, value in controls.items():
            key = f"staged_{name}/value"
            if key not in agent.params:
                raise KeyError(f"Uninitialized staged control: {key}")
            old = agent.params[key]
            agent.params[key] = jax.device_put(
                np.asarray(value, dtype=old.dtype), old.sharding
            )
    return controls
