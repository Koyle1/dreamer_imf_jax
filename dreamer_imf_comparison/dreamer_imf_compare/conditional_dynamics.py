"""Conditional Gaussian *auxiliary* alignment for implicit iMF dynamics.

The auxiliary density is not the flow density and never generates imagined
states. Balanced stop-gradients train it to predict q(z|h,o), and regularize q
toward a learned p_aux(z|h), rather than a fixed standard normal. This does not
claim a tractable KL(q||p_iMF) or an exact DreamerV3 dynamics-only substitution.
"""

import math

import jax
import jax.numpy as jnp
from dreamerv3 import rssm

from .dreamer_ablation_dynamics import GaussianRSSM, _UPSTREAM_RSSM
from .staged_dynamics import StagedAgent, StagedIMFRSSM

sg = jax.lax.stop_gradient


def conditional_kl(qmean, qstd, pmean, pstd):
    return jnp.sum(
        jnp.log(pstd / qstd) + (qstd**2 + (qmean - pmean) ** 2) / (2 * pstd**2) - 0.5,
        (-2, -1),
    )


def balanced_alignment(qmean, qstd, pmean, pstd, free_nats=1.0):
    dynamics = conditional_kl(sg(qmean), sg(qstd), pmean, pstd)
    representation = conditional_kl(qmean, qstd, sg(pmean), sg(pstd))
    return jnp.maximum(dynamics, free_nats), jnp.maximum(representation, free_nats)


class ConditionalIMFRSSM(StagedIMFRSSM):
    # Named prior parameters are classified as transition optimizer parameters.
    _prior_normal = GaussianRSSM._prior_normal

    def loss(self, carry, tokens, acts, reset, training):
        carry, entries, feat = self.observe(carry, tokens, acts, reset, training)
        prior_mean, prior_std = self._prior_normal(feat["deter"])
        aux, rep = balanced_alignment(
            feat["mean"], feat["std"], prior_mean, prior_std, self.free_nats
        )
        flow = self._dynamics_loss(feat["deter"], sg(feat["stoch"]))
        metrics = dict(
            rep_ent=jnp.sum(
                jnp.log(feat["std"]) + 0.5 * math.log(2 * math.pi * math.e),
                (-2, -1),
            ).mean(),
            posterior_sample_rms=jnp.sqrt(
                jnp.mean(feat["stoch"].astype(jnp.float32) ** 2)
            ),
            posterior_mean_sq=jnp.mean(feat["mean"] ** 2),
            posterior_variance=jnp.mean(feat["std"] ** 2),
            auxiliary_prior_mean_sq=jnp.mean(prior_mean**2),
            auxiliary_prior_variance=jnp.mean(prior_std**2),
            conditional_kl_uncapped=conditional_kl(
                feat["mean"], feat["std"], prior_mean, prior_std
            ).mean(),
        )
        features = {k: feat[k] for k in ("deter", "stoch")}
        return (
            carry,
            entries,
            dict(dyn=flow, aux_dyn=aux, rep=rep),
            features,
            sg(metrics),
        )


ConditionalIMFRSSM.__annotations__ = dict(_UPSTREAM_RSSM.__annotations__)
ConditionalIMFRSSM._defaults = dict(_UPSTREAM_RSSM._defaults)


class ConditionalAgent(StagedAgent):
    def __init__(self, obs_space, act_space, config):
        if config.repval_grad or not config.reward_grad:
            raise ValueError(
                "Repair requires detached replay value and active reward gradients"
            )
        super().__init__(obs_space, act_space, config)
        self.scales["aux_dyn"] = 1.0

    def loss(self, carry, obs, prevact, training):
        total, (carry, entries, outs, metrics) = super().loss(
            carry, obs, prevact, training
        )
        prediction = self.rew(self.feat2tensor(outs["repfeat"]), 2).pred()
        truth = obs["reward"]
        for name, mask in (("positive", truth > 0), ("zero", truth == 0)):
            count = mask.sum()
            average = lambda x: jnp.where(mask, x, 0).sum() / jnp.maximum(count, 1)
            metrics[f"reward_coverage/{name}_count"] = count
            metrics[f"reward_coverage/{name}_mse"] = average((prediction - truth) ** 2)
            metrics[f"reward_coverage/{name}_zero_mse"] = average(truth**2)
            metrics[f"reward_coverage/{name}_prediction"] = average(prediction)
        # Counts are mandatory: zero-count reductions are placeholders, not evidence.
        return total, (carry, entries, outs, sg(metrics))


def install(arm):
    if arm != "imf":
        raise ValueError("Only the repaired iMF arm is authorized")
    rssm.RSSM = ConditionalIMFRSSM
    return ConditionalAgent
