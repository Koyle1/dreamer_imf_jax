"""Isolated posterior-mean reward distillation for online continuation.

The world-model objective, optimizer, parameter names, inference sample, and
recurrent carry remain the parent's. Auxiliary heads learn only from stopped
pre-update features; imagination never has access to a posterior mean.
"""

import hashlib
import json
from pathlib import Path
import pickle

import elements
import embodied.jax
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax
from dreamerv3 import agent as upstream, rssm

from .conditional_dynamics import ConditionalAgent, ConditionalIMFRSSM
from .staged_dynamics import control, parameter_group

sg = jax.lax.stop_gradient
WIDTH = 128
ALPHA = 0.01
LEARNING_RATE = 3e-4
LAYERS = ("hidden0", "hidden1", "logits")


def posterior_mean_features(features, statistics, bdims):
    """RMT h,mu layout; do not cast mu through the backbone's bfloat16 path."""
    h = features["deter"].astype(jnp.float32)
    mean = statistics["mean"].astype(jnp.float32)
    return jnp.concatenate(
        [h.reshape((*h.shape[:bdims], -1)), mean.reshape((*mean.shape[:bdims], -1))],
        -1,
    )


class RewardCategorical:
    def __init__(self, logits):
        if logits.shape[-1] != 3:
            raise ValueError("Reward support must be {0,1,2}")
        self.logits = logits.astype(jnp.float32)

    def probabilities(self):
        return jax.nn.softmax(self.logits)

    def pred(self):
        return (self.probabilities() * jnp.arange(3, dtype=jnp.float32)).sum(-1)

    def loss(self, target):
        labels = target.astype(jnp.int32)
        valid = (target == labels) & (labels >= 0) & (labels <= 2)
        loss = optax.softmax_cross_entropy_with_integer_labels(self.logits, labels)
        return jnp.where(valid, loss, jnp.nan)


def mlp_logits(parameters, normalized):
    x = normalized.astype(jnp.float32)
    for i, layer in enumerate(parameters):
        x = x @ layer["w"].astype(jnp.float32) + layer["b"].astype(jnp.float32)
        if i < 2:
            x = jax.nn.relu(x)
    return x


def weight_penalty(parameters, alpha=ALPHA):
    return (alpha * 0.5) * sum(jnp.sum(layer["w"] ** 2) for layer in parameters)


def masked_mean(values, valid):
    return jnp.where(valid, values, 0).sum() / jnp.maximum(valid.sum(), 1)


def reward_mask(is_first):
    """Incoming reward is valid on terminal rows, but never on reset rows."""
    return ~is_first.astype(bool)


def distillation_targets(reward, teacher_probabilities, coefficient):
    if coefficient not in (0.0, 0.5):
        raise ValueError("Only preregistered distillation coefficients are allowed")
    hard = jax.nn.one_hot(sg(reward).astype(jnp.int32), 3)
    return sg((1 - coefficient) * hard + coefficient * teacher_probabilities)


def readout_data_loss(parameters, normalized, reward, valid, targets=None):
    logits = mlp_logits(parameters, sg(normalized))
    loss = (
        RewardCategorical(logits).loss(sg(reward))
        if targets is None
        else optax.softmax_cross_entropy(logits, sg(targets))
    )
    # Student soft labels must not silently make unsupported real rewards valid.
    labels_valid = (reward == reward.astype(jnp.int32)) & (reward >= 0) & (reward <= 2)
    return masked_mean(jnp.where(labels_valid, loss, jnp.nan), valid)


class Float32RewardHead(nj.Module):
    """Exact RMT MLP algebra, independent of the global mixed-precision dtype."""

    def normalize(self, features, bdims):
        x = sg(features).astype(jnp.float32)
        x = x.reshape((*x.shape[:bdims], -1))
        mean = sg(self.value("input_mean", jnp.zeros, (x.shape[-1],), jnp.float32))
        std = sg(self.value("input_std", jnp.ones, (x.shape[-1],), jnp.float32))
        # Imported standard deviations already incorporate the fixed .01 floor.
        # Do not refit or re-floor them using the online continuation's data.
        return (x - mean) / std

    def parameters(self, dimension):
        shapes = (dimension, WIDTH, WIDTH, 3)
        params = []
        for i, name in enumerate(LAYERS):

            def initializer(shape, scale=np.sqrt(2 / shapes[i])):
                return jax.random.normal(nj.seed(), shape, jnp.float32) * scale

            params.append(
                {
                    "w": self.value(
                        f"{name}_kernel", initializer, (shapes[i], shapes[i + 1])
                    ),
                    "b": self.value(
                        f"{name}_bias", jnp.zeros, (shapes[i + 1],), jnp.float32
                    ),
                }
            )
        return tuple(params)

    def __call__(self, features, bdims):
        normalized = self.normalize(features, bdims)
        return RewardCategorical(
            mlp_logits(self.parameters(normalized.shape[-1]), normalized)
        )

    def objective(self, features, reward, valid, targets=None):
        normalized = self.normalize(features, reward.ndim)
        params = self.parameters(normalized.shape[-1])
        data, data_grad = jax.value_and_grad(readout_data_loss)(
            params, normalized, reward, valid, targets
        )
        penalty, penalty_grad = jax.value_and_grad(weight_penalty)(params)
        data_norm, penalty_norm = optax.global_norm(data_grad), optax.global_norm(
            penalty_grad
        )
        dot = sum(
            jnp.vdot(a, b).real
            for a, b in zip(jax.tree.leaves(data_grad), jax.tree.leaves(penalty_grad))
        )
        distribution = RewardCategorical(mlp_logits(params, normalized))
        probability, prediction = distribution.probabilities(), distribution.pred()
        metrics = dict(
            data_loss=data,
            weight_penalty=penalty,
            data_gradient_norm=data_norm,
            regularization_gradient_norm=penalty_norm,
            gradient_cosine=dot / jnp.maximum(data_norm * penalty_norm, 1e-20),
            gradient_cosine_defined=(data_norm > 0) & (penalty_norm > 0),
            nll=masked_mean(distribution.loss(reward), valid),
            mse=masked_mean((prediction - reward) ** 2, valid),
            brier=masked_mean(
                jnp.sum(
                    (probability - jax.nn.one_hot(reward.astype(jnp.int32), 3)) ** 2, -1
                ),
                valid,
            ),
            feature_zscore_rms=jnp.sqrt(
                masked_mean(jnp.mean(normalized**2, -1), valid)
            ),
            feature_mean_shift_rms=jnp.sqrt(
                jnp.mean(
                    jnp.sum(
                        jnp.where(valid[..., None], normalized, 0),
                        axis=tuple(range(valid.ndim)),
                    )
                    ** 2
                )
            )
            / jnp.maximum(valid.sum(), 1),
            valid_count=valid.sum(),
        )
        for label, mask in (("positive", reward > 0), ("zero", reward == 0)):
            mask = valid & mask
            metrics[f"{label}_count"] = mask.sum()
            metrics[f"{label}_mse"] = masked_mean((prediction - reward) ** 2, mask)
        # No second-order gradient path through diagnostics.
        return data + penalty, sg(metrics)


class PosteriorRewardAgent(ConditionalAgent):
    arm = "A"

    def __init__(self, obs_space, act_space, config):
        super().__init__(obs_space, act_space, config)
        self.reward_teacher = Float32RewardHead(name="reward_teacher")
        self.reward_student = Float32RewardHead(name="reward_student")
        self.reward_aux_opt = embodied.jax.Optimizer(
            [self.reward_teacher, self.reward_student],
            optax.adam(LEARNING_RATE),
            name="reward_aux_opt",
        )
        # Crucially, no additions to self.modules or replacement of self.opt.

    @property
    def policy_keys(self):
        return "^(enc|dyn|dec|pol|rew|reward_teacher|reward_student)/"

    @property
    def ext_space(self):
        # Retained replay identity only; no episode label is a model input.
        return dict(super().ext_space, episode_id=elements.Space(np.int32))

    def _dynamics_observation(self, carry, tokens, actions, reset, training):
        return self.dyn.loss_with_statistics(carry, tokens, actions, reset, training)

    def imagined_reward(self, features, bdims):
        if self.arm == "A":
            return super().imagined_reward(features, bdims)
        return sg(self.reward_student(sg(features), bdims).pred())

    def _auxiliary_loss(
        self, student_features, teacher_features, reward, valid, targets
    ):
        teacher_loss, teacher_metrics = self.reward_teacher.objective(
            sg(teacher_features), sg(reward), valid
        )
        student_loss, student_metrics = self.reward_student.objective(
            sg(student_features), sg(reward), valid, sg(targets)
        )
        metrics = {
            **upstream.prefix(teacher_metrics, "reward_teacher"),
            **upstream.prefix(student_metrics, "reward_student"),
        }
        return teacher_loss + student_loss, metrics

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
        student_features = sg(self.feat2tensor(outs["repfeat"]).astype(jnp.float32))
        teacher_features = sg(
            posterior_mean_features(outs["repfeat"], outs["posterior"], 2)
        )
        # World optimization above does not touch auxiliary modules. Both the
        # teacher and these features therefore precede this learner's aux update.
        teacher = self.reward_teacher(teacher_features, 2)
        student = self.reward_student(student_features, 2)
        valid, reward = reward_mask(obs["is_first"]), sg(obs["reward"])
        targets = distillation_targets(
            reward, teacher.probabilities(), 0.5 if self.arm == "C" else 0.0
        )
        metrics["reward_aux/teacher_student_mse"] = masked_mean(
            (teacher.pred() - student.pred()) ** 2, valid
        )
        metrics["reward_aux/teacher_student_kl"] = masked_mean(
            jnp.sum(
                teacher.probabilities()
                * (
                    jax.nn.log_softmax(teacher.logits)
                    - jax.nn.log_softmax(student.logits)
                ),
                -1,
            ),
            valid,
        )
        aux_before = {
            k: v
            for k, v in nj.context().items()
            if k.startswith(("reward_teacher/", "reward_student/"))
            and k.endswith(("_kernel", "_bias"))
        }
        aux_metrics, extra = self.reward_aux_opt(
            self._auxiliary_loss,
            student_features,
            teacher_features,
            reward,
            valid,
            targets,
            has_aux=True,
        )
        metrics.update(aux_metrics)
        metrics.update(extra)
        for prefix in ("reward_teacher", "reward_student"):
            old = {k: v for k, v in aux_before.items() if k.startswith(prefix + "/")}
            difference = {k: nj.context()[k] - v for k, v in old.items()}
            metrics[f"{prefix}/update_parameter_ratio"] = optax.global_norm(
                difference
            ) / jnp.maximum(optax.global_norm(old), 1e-20)
        result = {}
        if self.config.replay_context:
            result["replay"] = elements.tree.flatdict(
                dict(stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2])
            )
        carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
        return carry, result, sg(metrics)

    def policy(self, carry, obs, mode="train"):
        enc_carry, dyn_carry, dec_carry, previous = carry
        reset = obs["is_first"]
        kwargs = dict(training=False, single=True)
        enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kwargs)
        dyn_carry, dyn_entry, features = self.dyn.observe(
            dyn_carry, tokens, previous, reset, **kwargs
        )
        # Posterior statistics never enter carry, actor inputs, or decoder inputs.
        repfeat = {k: features[k] for k in ("deter", "stoch")}
        dec_entry = {}
        if dec_carry:
            dec_carry, dec_entry, _ = self.dec(dec_carry, repfeat, reset, **kwargs)
        flat = self.feat2tensor(repfeat)
        distribution = self.pol(flat, bdims=1)
        action = (
            upstream.predict(distribution)
            if mode == "eval"
            else upstream.sample(distribution)
        )
        student = self.reward_student(flat, 1)
        teacher_features = posterior_mean_features(repfeat, features, 1)
        teacher = self.reward_teacher(teacher_features, 1)
        original = self.rew(flat, 1)
        outputs = {
            "finite": elements.tree.flatdict(
                jax.tree.map(
                    lambda x: jnp.isfinite(x).all(tuple(range(1, x.ndim))),
                    dict(
                        obs=obs, carry=carry, tokens=tokens, feat=features, act=action
                    ),
                )
            )
        }
        if self.config.replay_context:
            outputs.update(
                elements.tree.flatdict(
                    dict(enc=enc_entry, dyn=dyn_entry, dec=dec_entry)
                )
            )
        outputs.update(
            {
                "log/original_reward": original.pred(),
                "log/original_loss": original.loss(obs["reward"]),
                "log/student_reward": student.pred(),
                "log/teacher_reward": teacher.pred(),
                "log/control_reward": (
                    original.pred() if self.arm == "A" else student.pred()
                ),
                "log/student_probabilities": student.probabilities(),
                "log/teacher_probabilities": teacher.probabilities(),
                "log/action_entropy": sum(
                    value.entropy() for value in distribution.values()
                ),
                "log/action_saturation": jnp.mean(
                    jnp.concatenate(
                        [
                            (
                                jnp.abs(value).reshape((value.shape[0], -1)) >= 0.99
                            ).astype(jnp.float32)
                            for key, value in action.items()
                            if not self.act_space[key].discrete
                        ],
                        -1,
                    ),
                    -1,
                ),
                "log/feature_rms": jnp.sqrt(
                    jnp.mean(flat.astype(jnp.float32) ** 2, -1)
                ),
                "log/student_feature_zscore_rms": jnp.sqrt(
                    jnp.mean(self.reward_student.normalize(flat, 1) ** 2, -1)
                ),
                "log/teacher_feature_zscore_rms": jnp.sqrt(
                    jnp.mean(
                        self.reward_teacher.normalize(teacher_features, 1) ** 2, -1
                    )
                ),
            }
        )
        return (enc_carry, dyn_carry, dec_carry, action), action, outputs


def install(arm):
    if arm not in ("A", "B", "C"):
        raise ValueError("Posterior-reward arm must be A, B, or C")
    rssm.RSSM = ConditionalIMFRSSM
    return type(f"PosteriorRewardAgent{arm}", (PosteriorRewardAgent,), {"arm": arm})


def import_rmt_heads(rmt_full, readout_seed, initialized_params):
    """Map authenticated selected RMT arrays, never optimizer state, to new keys.

    The run manifest binds the complete dependency. This mapper independently
    binds report/evaluation seals and checks selected record identities and shape.
    Checkpoints are trusted dependency pickle files, not arbitrary user uploads.
    """
    if type(readout_seed) is not int or readout_seed not in (0, 1, 2):
        raise ValueError("Readout seed must be 0, 1, or 2")
    root = Path(rmt_full)
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    verified = read(root / "verified.json")
    if verified.get("exact_replay") is not True or any(
        verified.get(f"{name}_sha256") != sha(root / f"{name}.json")
        for name in ("report", "evaluated")
    ):
        raise ValueError("RMT final verification binding differs")
    seal = read(root / "evaluated.json")
    if seal.get("stage") != "evaluated" or not seal.get("artifacts"):
        raise ValueError("RMT evaluated artifact registry is invalid")

    def checked(path):
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Selected RMT artifact escaped dependency")
        name = str(path.relative_to(root))
        if seal["artifacts"].get(name) != sha(path):
            raise ValueError("Selected RMT artifact binding differs: " + name)
        return path

    selected = read(checked(root / "selections.json"))
    exported = {}
    for group, prefix in (("hmean", "reward_teacher"), ("hz", "reward_student")):
        choice = selected[group]
        alpha = choice["alpha"]
        record = choice["selections"][str(readout_seed)]
        update = record["update"]
        directory = root / "heads" / f"reward-{group}-a{float(alpha):g}-s{readout_seed}"
        identity = read(checked(directory / "selection.json"))
        if any(
            identity.get(k) != v
            for k, v in dict(
                seed=readout_seed,
                alpha=alpha,
                width=WIDTH,
                depth=2,
                classes=3,
                selected=record,
            ).items()
        ):
            raise ValueError("Selected RMT readout identity differs")
        with checked(directory / f"checkpoint-{update:05d}.pkl").open("rb") as stream:
            layers = pickle.load(stream)
        if len(layers) != 3 or any(set(layer) != {"w", "b"} for layer in layers):
            raise ValueError("Unexpected RMT MLP parameter structure")
        with np.load(
            checked(directory / "normalization.npz"), allow_pickle=False
        ) as norm:
            for field in ("mean", "std"):
                exported[f"{prefix}/input_{field}"] = np.asarray(
                    norm[field], np.float32
                )
        if np.any(exported[f"{prefix}/input_std"] <= 0):
            raise ValueError("Invalid fixed RMT standard deviation")
        for name, layer in zip(LAYERS, layers):
            exported[f"{prefix}/{name}_kernel"] = np.asarray(layer["w"], np.float32)
            exported[f"{prefix}/{name}_bias"] = np.asarray(layer["b"], np.float32)
    expected = {
        k
        for k in initialized_params
        if k.startswith(("reward_teacher/", "reward_student/"))
    }
    if set(exported) != expected:
        raise ValueError(
            "RMT export versus initialized auxiliary key inventory differs"
        )
    for key, value in exported.items():
        if value.shape != initialized_params[key].shape or not np.isfinite(value).all():
            raise ValueError("RMT export shape or finiteness differs: " + key)
    return exported
