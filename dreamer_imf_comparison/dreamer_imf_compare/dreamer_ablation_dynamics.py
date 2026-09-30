"""Continuous transition-family *and loss* ablations for pinned DreamerV3.

The categorical arm is untouched. Both continuous arms share the Gaussian
posterior, recurrent core, feature width, and KL(q || standard normal) representation
regularizer (summed over features, with upstream free nats). Gaussian dynamics
uses sample NLL; iMF uses the repository's improved MeanFlow compound regression
with boundary velocity supervision and detached posterior samples. No tractable
iMF density or KL is claimed. Loss magnitudes are therefore not interchangeable.
Actor, critic, reward, continuation, encoder and decoder remain upstream code.
"""

import math

import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
from dreamerv3 import rssm
from imf_dreamer_jax.imf import improved_meanflow_loss, sample_imf_one_step

_UPSTREAM_RSSM = rssm.RSSM
sg = jax.lax.stop_gradient


def normal_kl(mean, std):
    """Per-transition KL to the same fixed reference for both arms."""
    return 0.5 * jnp.sum(mean**2 + std**2 - 1 - 2 * jnp.log(std), (-2, -1))


class ContinuousRSSM(_UPSTREAM_RSSM):
    """Preserves upstream carry and stochastic-feature tensor contracts."""

    def _normal(self, name, x):
        x = self.sub(
            name,
            nn.Linear,
            2 * self.stoch * self.classes,
            **dict(self.kw, outscale=self.outscale),
        )(x)
        mean, raw = jnp.split(x.astype(jnp.float32), 2, -1)
        shape = mean.shape[:-1] + (self.stoch, self.classes)
        # Bounded scale follows the continuous Dreamer convention.
        return mean.reshape(shape), (0.1 + 0.9 * jax.nn.sigmoid(raw)).reshape(shape)

    def _observe(self, carry, tokens, action, reset, training):
        deter, stoch, action = nn.mask((carry["deter"], carry["stoch"], action), ~reset)
        action = nn.mask(nn.DictConcat(self.act_space, 1)(action), ~reset)
        deter = self._core(deter, stoch, action)
        tokens = tokens.reshape((*deter.shape[:-1], -1))
        x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
        for i in range(self.obslayers):
            x = self.sub(f"obs{i}", nn.Linear, self.hidden, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f"obs{i}norm", nn.Norm, self.norm)(x))
        mean, std = self._normal("obsnormal", x)
        stoch = nn.cast(mean + std * jax.random.normal(nj.seed(), mean.shape))
        carry = dict(deter=deter, stoch=stoch)
        return carry, (carry, dict(**carry, mean=mean, std=std))

    def imagine(self, carry, policy, length, training, single=False):
        if not single:
            return super().imagine(carry, policy, length, training, single=False)
        action = policy(sg(carry)) if callable(policy) else policy
        actemb = nn.DictConcat(self.act_space, 1)(action)
        deter = self._core(carry["deter"], carry["stoch"], actemb)
        stoch = nn.cast(self._sample_prior(deter))
        carry = nn.cast(dict(deter=deter, stoch=stoch))
        return carry, (carry, action)

    def loss(self, carry, tokens, acts, reset, training):
        carry, entries, feat = self.observe(carry, tokens, acts, reset, training)
        dyn = self._dynamics_loss(feat["deter"], sg(feat["stoch"]))
        rep = normal_kl(feat["mean"], feat["std"])
        if self.free_nats:
            rep = jnp.maximum(rep, self.free_nats)
        entropy = jnp.sum(
            jnp.log(feat["std"]) + 0.5 * math.log(2 * math.pi * math.e), (-2, -1)
        )
        metrics = dict(
            rep_ent=entropy.mean(),
            posterior_sample_rms=jnp.sqrt(
                jnp.mean(feat["stoch"].astype(jnp.float32) ** 2)
            ),
        )
        # Agent concatenates replay and imagination feature dictionaries.
        features = {key: feat[key] for key in ("deter", "stoch")}
        return carry, entries, dict(dyn=dyn, rep=rep), features, metrics


class GaussianRSSM(ContinuousRSSM):
    def _prior_normal(self, deter):
        x = nn.cast(deter)
        for i in range(self.imglayers):
            x = self.sub(f"prior{i}", nn.Linear, self.hidden, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f"prior{i}norm", nn.Norm, self.norm)(x))
        return self._normal("priornormal", x)

    def _sample_prior(self, deter):
        mean, std = self._prior_normal(deter)
        return mean + std * jax.random.normal(nj.seed(), mean.shape)

    def _dynamics_loss(self, deter, target):
        mean, std = self._prior_normal(deter)
        target = sg(target.astype(jnp.float32))
        return jnp.sum(
            0.5 * ((target - mean) / std) ** 2
            + jnp.log(std)
            + 0.5 * math.log(2 * math.pi),
            (-2, -1),
        )


class IMFRSSM(ContinuousRSSM):
    def _flow_params(self):
        # Retrieve every Ninjax parameter before entering the helper's pure JVP.
        # Register arrays separately so upstream optimizer/parameter counting works.
        widths = [self.stoch * self.classes + self.deter + 2]
        widths += [self.hidden] * self.imglayers + [2 * self.stoch * self.classes]
        layers = []
        for i, (nin, nout) in enumerate(zip(widths[:-1], widths[1:])):
            gain = 1.0 if i == len(widths) - 2 else math.sqrt(2.0)

            def initialize(shape, gain=gain, nin=nin):
                return jax.random.normal(nj.seed(), shape) * (gain / math.sqrt(nin))

            layers.append(
                dict(
                    weight=self.value(f"imf{i}_weight", initialize, (nin, nout)),
                    bias=self.value(f"imf{i}_bias", jnp.zeros, (nout,), jnp.float32),
                )
            )
        return dict(layers=tuple(layers))

    def _sample_prior(self, deter):
        shape = deter.shape[:-1] + (self.stoch, self.classes)
        condition = deter.astype(jnp.float32).reshape((-1, self.deter))
        params = self._flow_params()
        return sample_imf_one_step(params, condition, nj.seed()).reshape(shape)

    def _dynamics_loss(self, deter, target):
        condition = deter.astype(jnp.float32).reshape((-1, self.deter))
        target = sg(target.astype(jnp.float32)).reshape((-1, self.stoch * self.classes))
        params = self._flow_params()
        loss = improved_meanflow_loss(
            params,
            target,
            condition,
            nj.seed(),
            reduction="none",
            boundary_velocity_supervision=True,
        )
        return loss.reshape(deter.shape[:-1])


# Ninjax's metaclass does not inherit configuration annotations/defaults.
# Restore the exact upstream field contract on all concrete adapter classes.
for _cls in (ContinuousRSSM, GaussianRSSM, IMFRSSM):
    _cls.__annotations__ = dict(_UPSTREAM_RSSM.__annotations__)
    _cls._defaults = dict(_UPSTREAM_RSSM._defaults)


def install(arm):
    """Install before constructing Agent; categorical is strictly a no-op.

    Use one arm per process. Refuse a categorical request after patching because
    silently retaining a continuous RSSM would invalidate the baseline.
    """
    if arm not in ("categorical", "gaussian", "imf"):
        raise ValueError(f"Unknown dynamics arm: {arm!r}")
    if arm == "categorical":
        if rssm.RSSM is not _UPSTREAM_RSSM:
            raise RuntimeError("Categorical baseline requires a fresh process")
        return
    rssm.RSSM = {"gaussian": GaussianRSSM, "imf": IMFRSSM}[arm]
