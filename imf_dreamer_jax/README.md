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

Version 0.3 added the actor-learning safeguards used by the repaired local
study: finite-support reward heads with fractional binary cross-entropy,
zero-initialized value outputs, linear actor/critic learning-rate warmup,
survival-weighted critic training, and a helper that turns every post-burn-in
posterior state into an imagination start. These are configured through
`DreamerConfig`; unbounded scalar reward regression remains available for
environments without known reward support.

Version 0.4 adds opt-in model-fidelity controls for iMF: posterior/base-noise
coupling, exact one-step endpoint supervision, boundary-velocity supervision,
and explicit scaling of the gradient through the recurrent condition. Their
defaults reproduce the 0.3 objective, so existing configurations and
checkpoints remain valid.

Version 0.5 repairs shortcut-forcing self-distillation at its source. Shortcut
training now uses a post-update EMA bootstrap teacher, bounds only the stopped
intermediate bootstrap trajectory, never asks the teacher for a step below the
trained support, and evaluates the algebraically equivalent bootstrap loss
directly in x-space. Sampling is unclipped by default so an unstable model
cannot be made to look healthy at evaluation time. This state-layout change
increments the checkpoint format to version 2.

The package also exposes an experimental **trajectory iMF** objective. Each
trajectory token has its own query times and history-corruption time, and the
RSSM is trained with one mixture of clean-context, corrupted-context, and
future-suffix examples. This mode is opt-in and leaves all legacy defaults
unchanged.

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
    burn_in=8,
    overshooting_distances=(5, 15),
    overshooting_scale=0.1,
    # Fidelity-repaired iMF training; inference remains one NFE.
    prior_scale=1.0,
    imf_noise_coupling="posterior",
    imf_endpoint_scale=1.0,
    imf_condition_gradient_scale=1.0,
    imf_boundary_velocity_supervision=True,
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
| `prior="imf", overshooting_distances=(1,)` | One-step conditional iMF RSSM |
| `prior="imf", overshooting_distances=(5, 15)` | One-NFE iMF prior plus exact-distance consistency training |
| `prior="imf", imf_trajectory_enabled=True` | Multi-time trajectory iMF with one mixed dynamics objective |

`DreamerConfig.method_name` produces the corresponding stable method label.

## Functional training API

- `create_agent()` initializes model, actor, critic, and three Adam states.
- `jit_train_world_model()` performs a finite, clipped world-model update.
- `observe_sequence()` filters a full replay sequence into recurrent beliefs.
- `jit_train_actor_critic()` trains a tanh-squashed actor and slow-target
  critic from imagined latent rollouts. The default stopped-action REINFORCE
  estimator avoids following arbitrary action derivatives of learned dynamics;
  `actor_gradient="dynamics"` remains an explicit ablation.
- `jit_act()` updates the posterior belief and produces a bounded action.
- `SequenceReplayBuffer` stores data on the host and transfers sampled batches
  to a selected JAX device.

The sequence convention matches the original prototype: `actions[:, t]` is
the action immediately preceding `observations[:, t]`. Begin an episode with a
fresh zero RSSM state and a zero previous action. Replay batches can carry
`is_first` and `loss_mask`; `SequenceReplayBuffer.sample(..., burn_in=8)`
returns both, resets recurrent state at true boundaries, and excludes burn-in
steps from optimized losses.

## Improved MeanFlow API

`sample_imf_one_step()` evaluates `u(noise, condition, 0, 1)` exactly once and
returns `noise - u`. `improved_meanflow_loss()` implements the compound JVP
target

```text
v = v_head(z, condition, t, t)
(u, dudt) = jvp(u, (z, r, t), (v, 0, 1))
V = u + (t - r) stop_gradient(dudt)
```

The shared network emits `(u, v)`. Training separately regresses `V` and `v`
to `noise - target` with stopped adaptive iMF normalization, logit-normal time
pairs, and configurable boundary mass. Sampling still uses only `u`, so it
remains exactly one network evaluation. The RSSM uses a stopped
reparameterized posterior sample as the iMF target; reconstruction, reward,
continuation, and standard-normal representation regularization train the
posterior.

For fidelity-repaired training, set `imf_noise_coupling="posterior"` so the
transport endpoint uses the exact posterior reparameterization noise, set a
positive `imf_endpoint_scale` to train the inference query `(r, t) = (0, 1)`
directly, and enable `imf_boundary_velocity_supervision` so the auxiliary loss
supervises the same marginal velocity used by the JVP. The endpoint term is a
raw mean-squared sample error; only the original `u` and `v` terms use adaptive
normalization. `imf_condition_gradient_scale` changes only the backward signal
through the recurrent condition and leaves its forward value unchanged.

## Trajectory Improved MeanFlow

For clean token `X_k`, independent Gaussian endpoint `E_k`, and token time
`s`, trajectory iMF uses the path

```text
Z_k(s) = (1 - s) X_k + s E_k,          W_k = E_k - X_k.
```

The current-token query uses `(R_k, Q_k)`. Its causal context `C_k` contains
only actions and corrupted tokens strictly before `k`, whose separate
exposure times are `tau_<k`. Query and history views evaluate the same
per-token path, hence reuse exactly the same `E_k`; the integrated world model
requires `imf_noise_coupling="independent"`.

The crucial derivative holds `C_k` and every `tau_j` fixed:

```text
v*_k = u*_k + (Q_k - R_k) [partial_Q u*_k + J_Z u*_k v*_k]
```

In code, the JVP primals are `(Z_k, R_k, Q_k)` and its tangent is
`(v_k, 0, 1)`. Differentiating the recurrent context inside this JVP is a
different diagonal trajectory derivative; it adds cross-token terms through
earlier `Z_j(tau_j)` and `tau_j`. `trajectory_partial_jvp()` implements the
fixed-context derivative, while `naive_joint_context_jvp()` exists only as a
failing diagnostic control.

The unchanged masked loss is applied under one schedule mixture:

```text
V_k = u_k + (Q_k - R_k) stop_gradient(JVP_k)
L = sum_k m_k omega_k [||V_k-W_k||^2 + lambda_v ||vbar_k-W_k||^2]
    / sum_k m_k omega_k
```

- Clean-context samples set earlier `tau` values to zero.
- Corrupted-context samples independently corrupt earlier tokens.
- Future-suffix samples expose a clean prefix, corrupt the future, and mask
  the prefix out of the same loss.

With one token this is ordinary conditional iMF. Setting `R_k=Q_k` removes
the material-derivative correction and gives conditional flow matching
(literal unweighted quadratic CFM additionally sets `adaptive_power=0`). At
`R=0,Q=1`, generation is `X_hat=E-u(E,C,0,1)`, exactly one network evaluation
per transition and therefore `H` evaluations for an `H`-step recurrent
rollout.

The proposed trajectory dynamics mode adds no separate endpoint, shortcut,
context-dynamics, or overshooting penalty. Exact interval flows compose, and
exact covered one-step conditionals determine future marginals, so those
dynamics targets are redundant at a realizable population optimum. This is
not equality of the finite objectives: finite-capacity models can still
benefit from such regularizers. Reconstruction, representation, reward, and
continuation losses remain part of the world model.

Standalone use:

```python
from imf_dreamer_jax import (
    corrupt_trajectory,
    sample_trajectory_schedule,
    trajectory_imf_loss,
)

schedule = sample_trajectory_schedule(key_schedule, batch_size, steps)
noise = jax.random.normal(key_noise, targets.shape)
history = corrupt_trajectory(targets, noise, schedule.history_t)
# Build conditions causally from history and actions, then reuse the same noise.
loss = trajectory_imf_loss(
    imf_params,
    targets,
    conditions,
    key_loss,
    noise=noise,
    r=schedule.r,
    t=schedule.t,
    token_mask=schedule.loss_mask,
)
```

For the recurrent RSSM integration, configure:

```python
config = DreamerConfig(
    observation_shape=(8,),
    action_dim=2,
    prior="imf",
    imf_trajectory_enabled=True,
    imf_noise_coupling="independent",
)
```

The derivation, assumptions, and exact-versus-finite qualifications are in
`../dreamer_imf_comparison/TRAJECTORY_IMF_THEORY.md`; the novelty and
falsification audit is in
`../dreamer_imf_comparison/TRAJECTORY_IMF_NOVELTY_AUDIT.md`.
Corrupted-context exposure is not guaranteed to
match the learned model's rollout distribution, and Gaussian path noise is
aleatoric rather than epistemic uncertainty. Long-rollout stability and actor
benefit therefore remain empirical questions. The low-level API cannot verify
caller-supplied noise independence or query/history reuse; the recurrent
integration enforces them. Its scan handles episode resets, while callers with
variable-length padding must supply a suitable validity/reset policy. A fully
masked loss evaluates to zero but does not itself skip the optimizer update,
and continuous time sampling reaches the one-step endpoint only through
coverage and continuity rather than exact endpoint mass.

## Control diagnostics

`ActorCriticMetrics` reports actor/critic gradient norms, squashed entropy,
action saturation, pre-tanh magnitude, return scale, and slow-critic drift.
Sample quality alone is not treated as proof of control quality: compare
multi-step errors, policy returns against fixed controls, imagined-versus-real
return calibration, and real-versus-model action Jacobians in downstream
studies.

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

## License

Apache-2.0. See `LICENSE`.
