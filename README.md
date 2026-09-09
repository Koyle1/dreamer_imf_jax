# IMF Dreamer JAX

`imf-dreamer-jax` is a standalone, functional JAX implementation of the
compact world-model approach in this repository. It provides an
action-conditioned recurrent state-space model (RSSM) whose stochastic prior
is either a diagonal Gaussian or conditional **Improved MeanFlow (iMF)**.

This is a research library for controlled prior ablations. It is not an exact
reimplementation of published DreamerV3. In particular, it keeps the compact
continuous latent, reconstruction/reward/continuation losses, stopped
posterior-sample iMF matching, optional multistep overshooting, imagined
actor–critic, ensemble diagnostics, and safeguarded CEM planning of the local
prototype.

## Install

CPU development install:

```bash
python -m pip install -e /path/to/imf_dreamer_jax
```

On a CUDA 12 cluster, install the JAX wheel appropriate to the cluster driver
before installing this package, or use:

```bash
python -m pip install -e '/path/to/imf_dreamer_jax[cuda12]'
```

The library depends only on JAX and NumPy. It does not require PyTorch, Flax,
or Optax.

## Quickstart

JAX uses explicit immutable state and PRNG keys. Configurations are frozen and
can be static arguments to `jax.jit`.

```python
import jax
import jax.numpy as jnp

from imf_dreamer_jax import (
    DreamerConfig,
    create_agent,
    initial_state,
    jit_act,
    jit_train_world_model,
)

config = DreamerConfig(
    observation_shape=(8,),
    action_dim=2,
    prior="imf",
    overshooting_horizon=3,  # 1 for one-step iMF; >1 adds overshooting
)
key = jax.random.key(0)
init_key, train_key, act_key = jax.random.split(key, 3)
agent = create_agent(config, init_key)

batch = {
    "observations": jnp.zeros((4, 8, 8), dtype=jnp.float32),
    # action[t] immediately precedes observation[t]
    "actions": jnp.zeros((4, 8, 2), dtype=jnp.float32),
    "rewards": jnp.zeros((4, 8), dtype=jnp.float32),
    "continuations": jnp.ones((4, 8), dtype=jnp.float32),
}
agent, losses = jit_train_world_model(agent, batch, train_key, config)

belief = initial_state(config, batch_size=1)
action, belief = jit_act(
    agent.params,
    jnp.zeros((1, 8), dtype=jnp.float32),
    jnp.zeros((1, 2), dtype=jnp.float32),
    belief,
    act_key,
    config,
    deterministic=True,
)
assert action.shape == (1, 2)
```

All randomness is controlled by keys supplied by the caller. Reusing a key
reproduces the same stochastic computation; split or fold keys for new draws.

## Choosing the ablation

| Configuration | Meaning |
|---|---|
| `prior="gaussian"` | Compact Gaussian RSSM baseline with analytic KL |
| `prior="imf", overshooting_horizon=1` | One-step conditional iMF RSSM |
| `prior="imf", overshooting_horizon>1` | iMF plus sampled multistep overshooting |

`DreamerConfig.method_name` produces the corresponding stable method label.

## Functional training API

- `create_agent()` initializes model, actor, critic, and three Adam states.
- `jit_train_world_model()` performs a finite, clipped world-model update.
- `observe_sequence()` filters a full replay sequence into recurrent beliefs.
- `jit_train_actor_critic()` trains through an imagined latent rollout.
- `jit_act()` updates the posterior belief and produces a bounded action.
- `SequenceReplayBuffer` stores data on the host and transfers sampled batches
  to a selected JAX device.

The sequence convention matches the original prototype: `actions[:, t]` is
the action immediately preceding `observations[:, t]`. Begin an episode with a
fresh zero RSSM state and a zero previous action.

## Improved MeanFlow API

`sample_imf_one_step()` evaluates `u(noise, condition, 0, 1)` exactly once and
returns `noise - u`. `improved_meanflow_loss()` implements the compound JVP
target

```text
v = u(z, condition, t, t)
(u, dudt) = jvp(u, (z, r, t), (v, 0, 1))
V = u + (t - r) stop_gradient(dudt)
```

and regresses `V` to `noise - target`. The RSSM uses a stopped reparameterized
posterior sample as the iMF target; reconstruction, reward, continuation, and
standard-normal representation regularization train the posterior.

## Uncertainty and planning

Use `stack_ensemble()` and `sample_prior_ensemble()` to obtain samples shaped
`[member, noise, batch, stochastic]`. `predictive_moments()` applies the
empirical law of total variance, separating within-member aleatoric variance
from between-member epistemic variance. `energy_score()` uses off-diagonal
independent draw pairs.

`jit_cem_plan()` provides a bounded CEM planner. All candidates within an
evaluation share the same aleatoric noise paths, so duplicate candidates have
identical scores. Its score supports ensemble-risk penalization,
horizon-indexed epistemic truncation, a caller-supplied horizon-indexed
Gaussian behavior prior, and continuation-aware critic terminal fallback.
Ensemble disagreement remains a warning signal, not a safety certificate.

## Checkpoints

```python
from imf_dreamer_jax import load_checkpoint, save_checkpoint

save_checkpoint("run.ckpt", agent, config, metadata={"step": 1})
agent, config, metadata = load_checkpoint("run.ckpt")
```

Writes are atomic. Checkpoints use Python pickle to retain arbitrary PyTree
structure, so only load files from trusted sources.

## Device and precision notes

- Device placement follows ordinary JAX rules; use `jax.default_device`,
  `jax.device_put`, or the replay buffer's `device=` argument.
- The public `jit_*` functions compile for the device on which their arrays
  reside.
- The default parameter and replay dtype is `float32`. Mixed precision is left
  to an outer training policy rather than being hidden inside the model.
- Multi-GPU data parallelism can wrap the pure update functions with `pmap` or
  shard them with `jax.sharding`; no process-global RNG is used.
