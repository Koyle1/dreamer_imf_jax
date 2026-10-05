"""Read-only checkpoint inference and exactly restorable Reacher adapter."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np


def parameter_digest(params):
    h = hashlib.sha256()
    for name, value in sorted(params.items()):
        array = np.asarray(value)
        h.update(name.encode())
        h.update(str((array.shape, array.dtype)).encode())
        h.update(array.tobytes())
    return h.hexdigest()


class FrozenModel:
    """Raw upstream modules with immutable parameter arrays, never agent.train."""

    obs_keys = ("position", "to_target", "velocity")

    def __init__(self, cell, arm):
        import jax
        import jax.numpy as jnp
        import ninjax as nj
        import elements
        import embodied.jax.nets as nn
        from ruamel.yaml import YAML
        from dreamerv3 import agent, rssm
        from .dreamer_ablation_dynamics import _UPSTREAM_RSSM
        from .conditional_dynamics import ConditionalAgent, ConditionalIMFRSSM

        if arm not in ("imf", "categorical"):
            raise ValueError("unsupported frozen model")
        self.arm, self.cell = arm, Path(cell)
        with (self.cell / "checkpoint_200000.pkl").open("rb") as stream:
            saved = pickle.load(stream)
        self.host_params = saved["params"]
        self.before = parameter_digest(self.host_params)
        cfg = YAML(typ="safe").load((self.cell / "config.yaml").read_text())
        cfg["agent"]["replay_context"] = cfg["replay_context"]
        self.cfg = cfg
        if (
            jnp.dtype(nn.COMPUTE_DTYPE) != jnp.dtype(jnp.bfloat16)
            or cfg["jax"]["compute_dtype"] != "bfloat16"
        ):
            raise ValueError(
                "frozen inference must use deployed bfloat16 upstream compute"
            )
        self.compute_dtype = str(jnp.dtype(nn.COMPUTE_DTYPE))
        dc = cfg["agent"]["dyn"]["rssm"]
        self.deter, self.stoch, self.classes = dc["deter"], dc["stoch"], dc["classes"]
        self.feature_dim = self.deter + self.stoch * self.classes
        self.action_dim = 2
        self.obs_space = {k: elements.Space(np.float32, (2,)) for k in self.obs_keys}
        self.obs_space.update(reward=elements.Space(np.float32, ()))
        self.obs_space.update(
            {
                k: elements.Space(bool, ())
                for k in ("is_first", "is_last", "is_terminal")
            }
        )
        self.act_space = {"action": elements.Space(np.float32, (2,), -1, 1)}
        old_rssm = rssm.RSSM
        try:
            rssm.RSSM = ConditionalIMFRSSM if arm == "imf" else _UPSTREAM_RSSM
            cls = ConditionalAgent if arm == "imf" else agent.Agent
            self.model = object.__new__(cls)
            self.model.__init__(
                self.obs_space, self.act_space, elements.Config(cfg["agent"])
            )
        finally:
            rssm.RSSM = old_rssm
        # Optimizer state is not needed for frozen inference. Retain parameter
        # arrays for the model/normalizers only; no optimizer can be called here.
        self.params = {
            k: jax.device_put(v)
            for k, v in self.host_params.items()
            if not k.startswith("opt/")
        }
        self.live_keys = frozenset(self.params)
        self.parameter_counts = {}
        for prefix in ("enc", "dyn", "dec", "rew", "con", "pol", "val", "slowval"):
            self.parameter_counts[prefix] = sum(
                v.size
                for k, v in self.host_params.items()
                if k.startswith(prefix + "/")
            )
        self.parameter_count = sum(self.parameter_counts.values())
        self.resident_array_scalars = sum(v.size for v in self.params.values())
        self.resident_state_scalars = self.resident_array_scalars - self.parameter_count
        self.active_dynamics_parameters = sum(
            v.size
            for k, v in self.host_params.items()
            if k.startswith("dyn/")
            and not k.startswith("dyn/obs")
            and not (arm == "imf" and k.startswith("dyn/prior"))
        )
        if arm == "imf":
            # Only the average-velocity half of the joint final projection
            # affects sampled states; the auxiliary half is resident, not active.
            for suffix in ("weight", "bias"):
                key = f"dyn/imf{dc['imglayers']}_{suffix}"
                self.active_dynamics_parameters -= self.host_params[key].size // 2
        m = self.model

        def observe_body(carry, obs, previous):
            enc, dyn = carry
            reset = obs["is_first"]
            enc, _, tokens = m.enc(enc, obs, reset, False, single=True)
            dyn, _, feat = m.dyn.observe(
                dyn, tokens, {"action": previous}, reset, False, single=True
            )
            return (enc, dyn), m.feat2tensor(feat).astype(jnp.float32)

        self._observe = jax.jit(
            lambda p, c, o, a, key: nj.pure(observe_body)(p, c, o, a, seed=key)[1]
        )

        def action_body(feature):
            return (
                m.pol(nn.cast(feature), 1)["action"]
                .sample(nj.seed())
                .astype(jnp.float32)
            )

        self._action = jax.jit(
            lambda p, f, key: nj.pure(action_body)(p, f, seed=key)[1]
        )

        def decode_body(features):
            f = self.unpack(features)
            decoded = m.dec(
                m.dec.initial(features.shape[0]),
                f,
                jnp.zeros(features.shape[0], bool),
                False,
                single=True,
            )[2]
            obs = jnp.concatenate(
                [
                    decoded[k].pred().reshape(features.shape[0], -1)
                    for k in self.obs_keys
                ],
                -1,
            )
            return obs.astype(jnp.float32), m.rew(m.feat2tensor(f), 1).pred().astype(
                jnp.float32
            )

        self._decode = jax.jit(lambda p, f: nj.pure(decode_body)(p, f, seed=17)[1])
        self._rollouts = {}
        # Discard first compiled executions without changing any live belief.
        dummy = {k: np.zeros(s.shape, s.dtype) for k, s in self.obs_space.items()}
        dummy["is_first"] = np.asarray(True)
        for _ in range(2):
            _, f = self.observe(self.initial(), dummy, np.zeros(2, np.float32), 719)
            self.action(f, 720)
        if self.frozen_digest() != self.before:
            raise RuntimeError("frozen model changed during setup")

    def initial(self):
        import jax

        return jax.tree.map(
            np.asarray, (self.model.enc.initial(1), self.model.dyn.initial(1))
        )

    def unpack(self, features):
        import embodied.jax.nets as nn

        return nn.cast(
            dict(
                deter=features[..., : self.deter],
                stoch=features[..., self.deter :].reshape(
                    (*features.shape[:-1], self.stoch, self.classes)
                ),
            )
        )

    def observe(self, carry, obs, prev_action, seed):
        import jax

        data = {
            k: np.asarray(obs[k], dtype=s.dtype).reshape((1, *s.shape))
            for k, s in self.obs_space.items()
        }
        result, feature = self._observe(
            self.params,
            jax.tree.map(jax.device_put, carry),
            jax.tree.map(jax.device_put, data),
            jax.device_put(np.asarray(prev_action, np.float32)[None]),
            jax.random.PRNGKey(int(seed)),
        )
        return (
            jax.tree.map(np.asarray, jax.device_get(result)),
            np.asarray(jax.device_get(feature))[0],
        )

    def action(self, feature, seed):
        import jax

        return np.asarray(
            jax.device_get(
                self._action(
                    self.params,
                    jax.device_put(np.asarray(feature)[None]),
                    jax.random.PRNGKey(int(seed)),
                )
            )
        )[0]

    def decode(self, features):
        import jax.numpy as jnp

        shape = features.shape[:-1]
        obs, reward = self._decode(
            self.params,
            jnp.asarray(features, jnp.float32).reshape(-1, self.feature_dim),
        )
        return obs.reshape((*shape, -1)), reward.reshape(shape)

    def rollout(self, starts, actions, seed, particles=32, flow_steps=4):
        import jax
        import jax.numpy as jnp
        import ninjax as nj
        import embodied.jax.nets as nn
        from imf_dreamer_jax.imf import sample_imf_steps

        cache = (particles, flow_steps)
        if cache not in self._rollouts:

            def pure_step(f, a, key):
                h = self.model.dyn._core(f["deter"], f["stoch"], nn.cast(a))
                if self.arm == "imf":
                    z = sample_imf_steps(
                        self.model.dyn._flow_params(),
                        h.astype(jnp.float32),
                        key,
                        steps=flow_steps,
                    )
                    z = nn.cast(z.reshape(h.shape[0], self.stoch, self.classes))
                else:
                    z = nn.cast(
                        self.model.dyn._dist(self.model.dyn._prior(h)).sample(seed=key)
                    )
                f = dict(deter=h, stoch=z)
                return f, self.model.feat2tensor(f).astype(jnp.float32)

            def run(p, starts, acts, key):
                b, h = acts.shape[:2]
                initial = self.unpack(jnp.repeat(starts, particles, axis=0))
                planned = jnp.repeat(acts, particles, axis=0).swapaxes(0, 1)
                keys = jax.random.split(key, h)

                def step(carry, pair):
                    a, k = pair
                    return nj.pure(pure_step)(p, carry, a, k, seed=k)[1]

                _, values = jax.lax.scan(step, initial, (planned, keys))
                return values.swapaxes(0, 1).reshape(b, particles, h, self.feature_dim)

            self._rollouts[cache] = jax.jit(run)
        return self._rollouts[cache](
            self.params, starts, actions, jax.random.PRNGKey(seed)
        )

    def frozen_digest(self):
        import jax

        # Check actual live arrays, not only a preserved host copy.
        if frozenset(self.params) != self.live_keys:
            raise ValueError("frozen parameter keys changed")
        live = {k: np.asarray(v) for k, v in jax.device_get(self.params).items()}
        restored = dict(self.host_params, **live)
        return parameter_digest(restored)


class ReacherAdapter:
    """AR2 wrapper with monotonic native counter and process-local snapshots."""

    action_repeat = 2

    def __init__(self, seed):
        from dm_control import suite

        self.dm = suite.load(
            "reacher", "hard", task_kwargs={"random": int(seed), "time_limit": 20.0}
        )
        spec = self.dm.action_spec()
        self.action_low = (
            np.broadcast_to(spec.minimum, spec.shape).astype(np.float32).copy()
        )
        self.action_high = (
            np.broadcast_to(spec.maximum, spec.shape).astype(np.float32).copy()
        )
        self.native_steps = 0
        self._last = None
        self._generation = 0

    @staticmethod
    def observation(timestep, reward=None):
        return {
            **{
                k: np.asarray(v, np.float32).copy()
                for k, v in timestep.observation.items()
            },
            "reward": np.asarray(
                float(timestep.reward or 0) if reward is None else reward, np.float32
            ),
            "is_first": np.asarray(timestep.first()),
            "is_last": np.asarray(timestep.last()),
            "is_terminal": np.asarray(timestep.last() and timestep.discount == 0),
        }

    def reset(self):
        self._last = self.dm.reset()
        self._generation += 1
        return self.observation(self._last)

    def step(self, action):
        if self._last is None or self._last.last():
            raise RuntimeError("explicit reset required")
        reward, count = 0.0, 0
        for _ in range(2):
            before_time = float(self.dm.physics.data.time)
            try:
                self._last = self.dm.step(np.asarray(action, np.float32))
            except BaseException:
                # A reward/observation callback can throw after integration.
                # Count the initiated native interval conservatively; its
                # durable reservation remains charged even on failure.
                if float(self.dm.physics.data.time) > before_time:
                    self.native_steps += 1
                raise
            self.native_steps += 1
            count += 1
            reward += float(self._last.reward or 0)
            if self._last.last():
                break
        return SimpleNamespace(
            observation=self.observation(self._last, reward),
            reward=reward,
            is_last=self._last.last(),
            native_steps=count,
        )

    def snapshot(self):
        p = self.dm.physics
        extras = {
            name: np.asarray(getattr(p.data, name)).copy()
            for name in (
                "qacc_warmstart",
                "ctrl",
                "qfrc_applied",
                "xfrc_applied",
                "mocap_pos",
                "mocap_quat",
                "userdata",
            )
        }
        return dict(
            identity=id(self),
            generation=self._generation,
            state=p.get_state().copy(),
            time=float(p.data.time),
            extras=extras,
            step_count=self.dm._step_count,
            reset_next_step=self.dm._reset_next_step,
            rng=copy.deepcopy(self.dm.task.random.get_state()),
            last=copy.deepcopy(self._last),
        )

    def restore(self, snapshot):
        if (
            snapshot["identity"] != id(self)
            or snapshot["generation"] != self._generation
        ):
            raise ValueError(
                "snapshot belongs to another simulator or reset generation"
            )
        p = self.dm.physics
        with p.reset_context():
            p.set_state(snapshot["state"])
            p.data.time = snapshot["time"]
        for name, value in snapshot["extras"].items():
            getattr(p.data, name)[:] = value
        self.dm._step_count = snapshot["step_count"]
        self.dm._reset_next_step = snapshot["reset_next_step"]
        self.dm.task.random.set_state(copy.deepcopy(snapshot["rng"]))
        self._last = copy.deepcopy(snapshot["last"])

    def close(self):
        self.dm.close()
