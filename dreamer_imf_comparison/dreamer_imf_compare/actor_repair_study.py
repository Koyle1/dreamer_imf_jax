"""Causal competence ladder for the repaired continuous Dreamer actor.

The study separates exact-policy optimization from learned critic, dynamics,
and reward components.  The primary beta is fixed before execution; the other
two behavior-KL values are sensitivity checks, not post-hoc rescue tuning.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import read_json, write_json_atomic
from . import actor_failure_diagnostic as previous
from . import matched_objective_benchmark as benchmark
from . import reward_head_mtp_study as mtp
from .policy_consistency_vnext import _tree_delta


SCHEMA = "trajectory-imf-continuous-actor-repair-v1"
RESULT_SCHEMA = "trajectory-imf-continuous-actor-repair-cell-v1"
REPORT_SCHEMA = "trajectory-imf-continuous-actor-repair-report-v1"
TASK = "dmc_reacher_easy"
WORLD_MODEL_SEEDS = (211,)
ACTOR_SEEDS = (311, 313)
BETAS = (0.0, 0.1, 0.3)
PRIMARY_BETA = 0.1
FINITE_HORIZONS = (5, 15)
EXACT_TARGET = 0.5
EXACT_PASS_MAE = 0.25
REAL_RETURN_MARGIN = 0.005
PURE_STAGES = (
    "exact_bandit_h1",
    "exact_finite_no_bootstrap",
    "exact_reward_learned_critic",
)
WORLD_STAGES = (
    "analytic_reward_learned_dynamics",
    "learned_reward_learned_dynamics",
)
SPECIAL_STAGES = ("safe_mpo_bandit", "gradient_oracle")


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _cell(
    cells: list[dict[str, Any]],
    *,
    stage: str,
    actor_seed: int | None,
    horizon: int | None,
    beta: float | None,
    world_model_seed: int | None = None,
) -> None:
    identity = {
        "index": len(cells),
        "stage": stage,
        "world_model_seed": world_model_seed,
        "actor_seed": actor_seed,
        "horizon": horizon,
        "behavior_kl_scale": beta,
    }
    slug = (
        f"{stage}/wm-{world_model_seed}-actor-{actor_seed}-"
        f"h{horizon}-beta-{beta}"
    ).replace(".", "p")
    identity["result_path"] = f"cells/{slug}/result.json"
    identity["raw_path"] = (
        f"cells/{slug}/action_traces.npz"
        if stage in WORLD_STAGES or stage == "random_policy"
        else None
    )
    digest_identity = {
        key: identity[key]
        for key in (
            "index",
            "stage",
            "world_model_seed",
            "actor_seed",
            "horizon",
            "behavior_kl_scale",
        )
    }
    identity["cell_id"] = benchmark.object_sha256(digest_identity)[:24]
    cells.append(identity)


def build_cell_matrix() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    _cell(
        cells,
        stage="random_policy",
        actor_seed=None,
        horizon=None,
        beta=None,
    )
    for actor_seed in ACTOR_SEEDS:
        for beta in BETAS:
            _cell(
                cells,
                stage="exact_bandit_h1",
                actor_seed=actor_seed,
                horizon=1,
                beta=beta,
            )
            for stage in (
                "exact_finite_no_bootstrap",
                "exact_reward_learned_critic",
            ):
                for horizon in FINITE_HORIZONS:
                    _cell(
                        cells,
                        stage=stage,
                        actor_seed=actor_seed,
                        horizon=horizon,
                        beta=beta,
                    )
            for stage in WORLD_STAGES:
                for world_model_seed in WORLD_MODEL_SEEDS:
                    for horizon in FINITE_HORIZONS:
                        _cell(
                            cells,
                            stage=stage,
                            world_model_seed=world_model_seed,
                            actor_seed=actor_seed,
                            horizon=horizon,
                            beta=beta,
                        )
        for stage in SPECIAL_STAGES:
            _cell(
                cells,
                stage=stage,
                actor_seed=actor_seed,
                horizon=1,
                beta=None,
            )
    return cells


def validate_cell_matrix(cells: Sequence[Mapping[str, Any]]) -> None:
    expected = build_cell_matrix()
    if json.loads(json.dumps(list(cells), sort_keys=True)) != expected:
        raise ValueError("actor-repair matrix differs from the frozen design")
    ids = [str(cell["cell_id"]) for cell in cells]
    if len(ids) != len(set(ids)):
        raise ValueError("actor-repair cell ids are not unique")


def shared_evaluation_seeds(episodes: int) -> list[int]:
    if episodes <= 0:
        raise ValueError("evaluation episodes must be positive")
    return [
        benchmark.derive_seed("actor-repair-shared-evaluation", TASK, episode)
        for episode in range(episodes)
    ]


def _source_row(manifest: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    rows = [
        row
        for row in manifest["source_artifacts"]
        if int(row["world_model_seed"]) == int(seed)
    ]
    if len(rows) != 1:
        raise ValueError("source world-model seed is absent or duplicated")
    return rows[0]


def build_manifest(
    mtp_root: str | Path,
    *,
    actor_updates: int = 1500,
    preparation_updates: int = 500,
    evaluation_episodes: int = 25,
    exact_batch_size: int = 256,
) -> dict[str, Any]:
    if min(actor_updates, preparation_updates, evaluation_episodes, exact_batch_size) <= 0:
        raise ValueError("study counts must be positive")
    root = Path(mtp_root).resolve(strict=True)
    dependency_manifest = read_json(root / "manifest.json")
    dependency_report = read_json(root / "report.json")
    if (
        dependency_report.get("status") != "complete"
        or dependency_report.get("manifest_sha256")
        != dependency_manifest.get("manifest_sha256")
    ):
        raise ValueError("MTP dependency is incomplete")
    sources = []
    for seed in WORLD_MODEL_SEEDS:
        source = mtp._source_row(dependency_manifest, seed)
        dataset = Path(source["dataset"])
        directory = root / "reward" / f"seed-{seed}-dreamer4_twohot_mtp"
        result = read_json(directory / "result.json")
        checkpoint = directory / "checkpoint.pkl"
        if not dataset.is_file() or not checkpoint.is_file():
            raise ValueError("required actor-repair dependency is missing")
        if benchmark.file_sha256(checkpoint) != result["checkpoint_sha256"]:
            raise ValueError("source checkpoint digest mismatch")
        arrays = benchmark.load_npz(dataset)
        if benchmark.array_sha256(arrays) != source["dataset_sha256"]:
            raise ValueError("source dataset payload digest mismatch")
        sources.append(
            {
                "world_model_seed": seed,
                "dataset": str(dataset.resolve()),
                "dataset_sha256": source["dataset_sha256"],
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": result["checkpoint_sha256"],
            }
        )
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "frozen_before_execution",
        "source_commit": _git_commit(),
        "task": TASK,
        "world_model_seeds": list(WORLD_MODEL_SEEDS),
        "actor_seeds": list(ACTOR_SEEDS),
        "behavior_kl_scales": list(BETAS),
        "primary_behavior_kl_scale": PRIMARY_BETA,
        "finite_horizons": list(FINITE_HORIZONS),
        "actor_updates": actor_updates,
        "preparation_updates": preparation_updates,
        "evaluation_episodes": evaluation_episodes,
        "exact_batch_size": exact_batch_size,
        "maximum_environment_steps": 1000,
        "shared_evaluation_seeds": shared_evaluation_seeds(evaluation_episodes),
        "source_artifacts": sources,
        "dependency": {
            "root": str(root),
            "manifest_sha256": dependency_manifest["manifest_sha256"],
            "report_file_sha256": benchmark.file_sha256(root / "report.json"),
        },
        "decision_thresholds": {
            "exact_action_mean_absolute_error": EXACT_PASS_MAE,
            "normalized_return_over_random": REAL_RETURN_MARGIN,
            "gradient_cosine": 0.9,
            "gradient_sign_agreement": 0.8,
        },
        "causal_order": list(PURE_STAGES + WORLD_STAGES),
        "primary_decision_beta": PRIMARY_BETA,
        "sensitivity_only_betas": [0.0, 0.3],
        "independent_unit": "world_model_seed",
        "actor_seed_role": "conditional_optimization_variance_only",
        "cells": build_cell_matrix(),
    }
    body["manifest_sha256"] = benchmark.object_sha256(body)
    return body


def write_manifest(
    mtp_root: str | Path,
    output_root: str | Path,
    **settings: Any,
) -> dict[str, Any]:
    manifest = build_manifest(mtp_root, **settings)
    path = Path(output_root) / "manifest.json"
    if path.is_file():
        if read_json(path) != manifest:
            raise ValueError("existing manifest differs from frozen actor-repair design")
    else:
        write_json_atomic(path, manifest)
    return manifest


def _cell_by_index(manifest: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    rows = [cell for cell in manifest["cells"] if int(cell["index"]) == int(index)]
    if len(rows) != 1:
        raise ValueError("actor-repair cell index is absent or duplicated")
    return rows[0]


def _exact_config(horizon: int, beta: float) -> Any:
    from imf_dreamer_jax import DreamerConfig

    return DreamerConfig(
        observation_shape=(1,),
        action_dim=2,
        deterministic_dim=8,
        stochastic_dim=4,
        embedding_dim=8,
        hidden_dim=32,
        prior="gaussian",
        imagination_horizon=horizon,
        actor_gradient="reinforce",
        behavior_kl_scale=beta,
        actor_entropy_scale=1e-3,
        actor_action_l2_scale=0.0,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        critic_output_init_scale=0.0,
        return_scale_ema_decay=0.99,
    )


def _exact_returns(rewards: Any, discount: float) -> Any:
    import jax.numpy as jnp

    accumulator = jnp.zeros_like(rewards[:, -1])
    values = []
    for index in range(rewards.shape[1] - 1, -1, -1):
        accumulator = rewards[:, index] + discount * accumulator
        values.append(accumulator)
    return jnp.stack(tuple(reversed(values)), axis=1)


def _run_exact_policy(
    *,
    actor_seed: int,
    horizon: int,
    beta: float,
    actor_updates: int,
    batch_size: int,
    learned_critic: bool,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import (
        actor_distribution,
        diagonal_normal_kl,
        init_return_scale_state,
        normalized_reinforce_objective,
        tanh_normal_log_prob,
        update_return_scale,
    )
    from imf_dreamer_jax.agent import init_actor, init_critic
    from imf_dreamer_jax.nn import clip_by_global_norm, mlp
    from imf_dreamer_jax.optim import adam_update, init_adam

    config = _exact_config(horizon, beta)
    actor_key, critic_key, loop_key = jax.random.split(
        benchmark.derive_jax_key("actor-repair-exact-init", actor_seed, horizon), 3
    )
    actor_params = init_actor(config, actor_key)
    critic_params = init_critic(config, critic_key)
    actor_optimizer = init_adam(actor_params)
    critic_optimizer = init_adam(critic_params)
    behavior_prior = jax.tree_util.tree_map(
        lambda value: jax.lax.stop_gradient(jnp.array(value, copy=True)), actor_params
    )
    scale_state = init_return_scale_state()
    features = jnp.zeros((batch_size, horizon, config.feature_dim), jnp.float32)

    def step(
        actor_parameters: Any,
        actor_opt: Any,
        critic_parameters: Any,
        critic_opt: Any,
        running_scale: Any,
        key: Any,
    ) -> tuple[Any, ...]:
        def actor_objective(parameters: Any) -> tuple[Any, tuple[Any, ...]]:
            distribution = actor_distribution(parameters, features, config)
            noise = jax.random.normal(key, distribution.mean.shape)
            pre_tanh = distribution.mean + distribution.std * noise
            stopped_pre_tanh = jax.lax.stop_gradient(pre_tanh)
            actions = jnp.tanh(stopped_pre_tanh)
            rewards = 1.0 - jnp.mean(jnp.square(actions - EXACT_TARGET), axis=-1)
            returns = _exact_returns(rewards, config.discount)
            next_scale, scale = update_return_scale(
                running_scale, returns, decay=config.return_scale_ema_decay
            )
            if learned_critic:
                baseline = jax.lax.stop_gradient(
                    mlp(critic_parameters, features)[..., 0]
                )
            else:
                baseline = (
                    jnp.sum(returns, axis=0, keepdims=True) - returns
                ) / float(batch_size - 1)
            advantages = jax.lax.stop_gradient(returns - baseline)
            log_probs = tanh_normal_log_prob(
                distribution.mean, distribution.std, stopped_pre_tanh
            )
            weights = jnp.power(
                jnp.asarray(config.discount, returns.dtype),
                jnp.arange(horizon, dtype=returns.dtype),
            )[None, :]
            policy = normalized_reinforce_objective(
                log_probs, advantages, jnp.broadcast_to(weights, returns.shape), scale
            )
            current = actor_distribution(parameters, features, config)
            prior = actor_distribution(behavior_prior, features, config)
            behavior_kl = jnp.mean(diagonal_normal_kl(current, prior))
            entropy = jnp.mean(-tanh_normal_log_prob(
                distribution.mean, distribution.std, pre_tanh
            ))
            loss = -policy - config.actor_entropy_scale * entropy + beta * behavior_kl
            return loss, (
                next_scale,
                returns,
                advantages,
                actions,
                distribution.mean,
                distribution.std,
                entropy,
                behavior_kl,
            )

        (loss, auxiliary), gradients = jax.value_and_grad(
            actor_objective, has_aux=True
        )(actor_parameters)
        gradients, gradient_norm = clip_by_global_norm(gradients, config.grad_clip)
        next_actor, next_actor_opt = adam_update(
            actor_parameters,
            gradients,
            actor_opt,
            learning_rate=config.actor_learning_rate,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2,
            epsilon=config.adam_epsilon,
        )
        (
            next_scale,
            returns,
            advantages,
            actions,
            means,
            stds,
            entropy,
            behavior_kl,
        ) = auxiliary
        critic_loss = jnp.asarray(0.0, jnp.float32)
        critic_gradient_norm = jnp.asarray(0.0, jnp.float32)
        next_critic, next_critic_opt = critic_parameters, critic_opt
        if learned_critic:
            targets = jax.lax.stop_gradient(returns)

            def critic_objective(parameters: Any) -> Any:
                predictions = mlp(parameters, features)[..., 0]
                return jnp.mean(jnp.square(predictions - targets))

            critic_loss, critic_gradients = jax.value_and_grad(critic_objective)(
                critic_parameters
            )
            critic_gradients, critic_gradient_norm = clip_by_global_norm(
                critic_gradients, config.grad_clip
            )
            next_critic, next_critic_opt = adam_update(
                critic_parameters,
                critic_gradients,
                critic_opt,
                learning_rate=config.critic_learning_rate,
                beta1=config.adam_beta1,
                beta2=config.adam_beta2,
                epsilon=config.adam_epsilon,
            )
        telemetry = (
            loss,
            critic_loss,
            gradient_norm,
            critic_gradient_norm,
            next_scale.percentile_range,
            jnp.mean(returns[:, 0]),
            jnp.mean(advantages),
            jnp.mean(advantages > 0.0),
            jnp.mean(jnp.abs(actions) >= 0.95),
            jnp.mean(means),
            jnp.mean(stds),
            entropy,
            behavior_kl,
        )
        return (
            next_actor,
            next_actor_opt,
            next_critic,
            next_critic_opt,
            next_scale,
            telemetry,
        )

    step = jax.jit(step)
    latest = None
    for update in range(actor_updates):
        (
            actor_params,
            actor_optimizer,
            critic_params,
            critic_optimizer,
            scale_state,
            latest,
        ) = step(
            actor_params,
            actor_optimizer,
            critic_params,
            critic_optimizer,
            scale_state,
            jax.random.fold_in(loop_key, update),
        )
    evaluation_features = jnp.zeros((1, config.feature_dim), jnp.float32)
    final_distribution = actor_distribution(actor_params, evaluation_features, config)
    deterministic_action = jnp.tanh(final_distribution.mean)[0]
    deterministic_reward = 1.0 - jnp.mean(
        jnp.square(deterministic_action - EXACT_TARGET)
    )
    deterministic_return = deterministic_reward * sum(
        config.discount**index for index in range(horizon)
    )
    values = [float(np.asarray(value)) for value in latest]
    names = (
        "actor_loss",
        "critic_loss",
        "actor_grad_norm",
        "critic_grad_norm",
        "return_percentile_range",
        "mean_imagined_return",
        "advantage_mean",
        "advantage_positive_fraction",
        "action_saturation",
        "pre_tanh_mean",
        "policy_std_mean",
        "squashed_entropy",
        "behavior_kl",
    )
    telemetry = dict(zip(names, values, strict=True))
    telemetry.update(
        {
            "deterministic_action_mean": float(np.mean(deterministic_action)),
            "deterministic_action": np.asarray(deterministic_action).tolist(),
            "action_target_mean_absolute_error": float(
                np.mean(np.abs(np.asarray(deterministic_action) - EXACT_TARGET))
            ),
            "exact_deterministic_return": float(deterministic_return),
            "imagined_real_return_gap": float(
                telemetry["mean_imagined_return"] - deterministic_return
            ),
        }
    )
    return telemetry


def _run_safe_mpo(
    *, actor_seed: int, actor_updates: int, batch_size: int
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import (
        SafeMPOConfig,
        actor_distribution,
        init_safe_mpo_state,
        safe_multi_action_mpo_objective,
        sample_safe_mpo_candidates,
        tanh_normal_log_prob,
        update_safe_mpo_reference,
    )
    from imf_dreamer_jax.agent import init_actor
    from imf_dreamer_jax.nn import clip_by_global_norm
    from imf_dreamer_jax.optim import adam_update, init_adam

    config = _exact_config(1, 0.0)
    mpo = SafeMPOConfig()
    actor_params = init_actor(
        config, benchmark.derive_jax_key("actor-repair-safe-mpo-init", actor_seed)
    )
    optimizer = init_adam(actor_params)
    mpo_state = init_safe_mpo_state(actor_params)
    features = jnp.zeros((batch_size, config.feature_dim), jnp.float32)
    loop_key = benchmark.derive_jax_key("actor-repair-safe-mpo-loop", actor_seed)

    def step(parameters: Any, opt: Any, reference_state: Any, key: Any) -> tuple[Any, ...]:
        reference_distribution = actor_distribution(
            reference_state.reference_actor, features, config
        )
        candidates = sample_safe_mpo_candidates(reference_distribution, key, mpo)
        scores = 1.0 - jnp.mean(
            jnp.square(candidates.actions - EXACT_TARGET), axis=-1
        )

        def objective(current_parameters: Any) -> tuple[Any, Any]:
            current_distribution = actor_distribution(
                current_parameters, features, config
            )
            candidate_mean = jnp.broadcast_to(
                current_distribution.mean[:, None, :], candidates.pre_tanh.shape
            )
            candidate_std = jnp.broadcast_to(
                current_distribution.std[:, None, :], candidates.pre_tanh.shape
            )
            log_probs = tanh_normal_log_prob(
                candidate_mean,
                candidate_std,
                candidates.pre_tanh,
            )
            result = safe_multi_action_mpo_objective(
                log_probs,
                scores,
                reference_distribution,
                current_distribution,
                mpo,
            )
            return result.loss, result

        (_, objective_result), gradients = jax.value_and_grad(
            objective, has_aux=True
        )(parameters)
        gradients, gradient_norm = clip_by_global_norm(gradients, config.grad_clip)
        next_parameters, next_optimizer = adam_update(
            parameters,
            gradients,
            opt,
            learning_rate=config.actor_learning_rate,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2,
            epsilon=config.adam_epsilon,
        )
        next_reference = update_safe_mpo_reference(
            reference_state,
            next_parameters,
            refresh_interval=mpo.reference_refresh_interval,
        )
        return next_parameters, next_optimizer, next_reference, objective_result, gradient_norm

    step = jax.jit(step)
    latest = None
    gradient_norm = None
    for update in range(actor_updates):
        actor_params, optimizer, mpo_state, latest, gradient_norm = step(
            actor_params,
            optimizer,
            mpo_state,
            jax.random.fold_in(loop_key, update),
        )
    distribution = actor_distribution(
        actor_params, jnp.zeros((1, config.feature_dim)), config
    )
    action = jnp.tanh(distribution.mean)[0]
    return {
        "deterministic_action": np.asarray(action).tolist(),
        "deterministic_action_mean": float(jnp.mean(action)),
        "action_target_mean_absolute_error": float(
            jnp.mean(jnp.abs(action - EXACT_TARGET))
        ),
        "actor_grad_norm": float(gradient_norm),
        "mean_kl": float(latest.mean_kl),
        "std_kl": float(latest.std_kl),
        "elite_fraction": float(latest.elite_fraction),
        "effective_sample_size": float(latest.effective_sample_size),
        "reference_updates_since_refresh": int(mpo_state.updates_since_refresh),
    }


def _flatten_gradient(tree: Any) -> np.ndarray:
    import jax

    return np.concatenate(
        [np.asarray(value).ravel() for value in jax.tree_util.tree_leaves(tree)]
    )


def _run_gradient_oracle(*, actor_seed: int, samples: int = 65536) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import actor_distribution, tanh_normal_log_prob
    from imf_dreamer_jax.agent import init_actor

    config = _exact_config(1, 0.0)
    actor_params = init_actor(
        config, benchmark.derive_jax_key("actor-repair-gradient-init", actor_seed)
    )
    features = jax.random.normal(
        benchmark.derive_jax_key("actor-repair-gradient-features", actor_seed),
        (samples, config.feature_dim),
    )
    score_noise = jax.random.normal(
        benchmark.derive_jax_key("actor-repair-gradient-score-noise", actor_seed),
        (samples, config.action_dim),
    )
    oracle_noise = jax.random.normal(
        benchmark.derive_jax_key("actor-repair-gradient-oracle-noise", actor_seed),
        (samples, config.action_dim),
    )

    def score_objective(parameters: Any) -> Any:
        distribution = actor_distribution(parameters, features, config)
        pre_tanh = jax.lax.stop_gradient(
            distribution.mean + distribution.std * score_noise
        )
        reward = jax.lax.stop_gradient(
            1.0 - jnp.mean(jnp.square(jnp.tanh(pre_tanh) - EXACT_TARGET), axis=-1)
        )
        centered_reward = reward - jnp.mean(reward)
        log_prob = tanh_normal_log_prob(
            distribution.mean, distribution.std, pre_tanh
        )
        return jnp.mean(log_prob * centered_reward)

    def pathwise_oracle(parameters: Any) -> Any:
        distribution = actor_distribution(parameters, features, config)
        actions = jnp.tanh(distribution.mean + distribution.std * oracle_noise)
        return jnp.mean(
            1.0 - jnp.mean(jnp.square(actions - EXACT_TARGET), axis=-1)
        )

    score_gradient = jax.grad(score_objective)(actor_params)
    oracle_gradient = jax.grad(pathwise_oracle)(actor_params)
    left = _flatten_gradient(score_gradient)
    right = _flatten_gradient(oracle_gradient)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    cosine = float(np.dot(left, right) / max(left_norm * right_norm, 1e-30))
    informative = (np.abs(left) + np.abs(right)) > 1e-7
    sign_agreement = float(np.mean(np.sign(left[informative]) == np.sign(right[informative])))
    return {
        "score_gradient_norm": left_norm,
        "oracle_gradient_norm": right_norm,
        "gradient_cosine": cosine,
        "gradient_sign_agreement": sign_agreement,
        "informative_coordinates": int(np.sum(informative)),
        "monte_carlo_samples_per_estimator": samples,
        "oracle": "independent_reparameterized_exact_reward_monte_carlo",
        "classification": (
            "policy_gradient_estimator_verified"
            if cosine >= 0.9 and sign_agreement >= 0.8
            else "policy_gradient_estimator_mismatch"
        ),
    }


def _finite_metrics(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_metrics(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_metrics(item) for item in value)
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return bool(np.isfinite(value))
    return True


def run_pure_cell(
    mtp_root: str | Path,
    output_root: str | Path,
    index: int,
    **settings: Any,
) -> dict[str, Any]:
    manifest = write_manifest(mtp_root, output_root, **settings)
    cell = _cell_by_index(manifest, index)
    if cell["stage"] not in PURE_STAGES + SPECIAL_STAGES:
        raise ValueError("selected cell is not a pure actor cell")
    result_path = Path(output_root) / str(cell["result_path"])
    if result_path.is_file():
        return read_json(result_path)
    started = time.perf_counter()
    stage = str(cell["stage"])
    if stage in PURE_STAGES:
        metrics = _run_exact_policy(
            actor_seed=int(cell["actor_seed"]),
            horizon=int(cell["horizon"]),
            beta=float(cell["behavior_kl_scale"]),
            actor_updates=int(manifest["actor_updates"]),
            batch_size=int(manifest["exact_batch_size"]),
            learned_critic=stage == "exact_reward_learned_critic",
        )
    elif stage == "safe_mpo_bandit":
        metrics = _run_safe_mpo(
            actor_seed=int(cell["actor_seed"]),
            actor_updates=int(manifest["actor_updates"]),
            batch_size=int(manifest["exact_batch_size"]),
        )
    else:
        metrics = _run_gradient_oracle(actor_seed=int(cell["actor_seed"]))
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": cell["index"],
        "stage": stage,
        "world_model_seed": cell["world_model_seed"],
        "actor_seed": cell["actor_seed"],
        "horizon": cell["horizon"],
        "behavior_kl_scale": cell["behavior_kl_scale"],
        "actor_updates": 0 if stage == "gradient_oracle" else manifest["actor_updates"],
        "preparation_updates": 0,
        "metrics": metrics,
        "world_model_parameter_delta": None,
        "raw_action_traces_sha256": None,
        "wall_seconds": time.perf_counter() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "runtime": benchmark.runtime_fingerprint(),
    }
    if not _finite_metrics(result):
        raise FloatingPointError("pure actor result contains non-finite values")
    write_json_atomic(result_path, result)
    return result


def _world_signal(stage: str) -> Any | None:
    if stage == "analytic_reward_learned_dynamics":
        return previous.analytic_reacher_reward
    if stage == "learned_reward_learned_dynamics":
        return None
    raise ValueError(f"unknown world actor stage {stage!r}")


def run_world_cell(
    mtp_root: str | Path,
    output_root: str | Path,
    index: int,
    **settings: Any,
) -> dict[str, Any]:
    import jax
    from imf_dreamer_jax import (
        AgentParams,
        AgentState,
        create_agent,
        diverse_imagination_starts,
        init_return_scale_state,
        jit_observe_sequence,
        jit_train_actor_critic_dreamer3,
        jit_train_behavior_cloning,
        jit_train_replay_critic,
        load_checkpoint,
        snapshot_behavior_prior,
    )

    manifest = write_manifest(mtp_root, output_root, **settings)
    cell = _cell_by_index(manifest, index)
    if cell["stage"] not in WORLD_STAGES:
        raise ValueError("selected cell is not a learned-world actor cell")
    result_path = Path(output_root) / str(cell["result_path"])
    raw_path = Path(output_root) / str(cell["raw_path"])
    if result_path.is_file():
        return read_json(result_path)
    seed = int(cell["world_model_seed"])
    actor_seed = int(cell["actor_seed"])
    horizon = int(cell["horizon"])
    beta = float(cell["behavior_kl_scale"])
    source = _source_row(manifest, seed)
    checkpoint = Path(source["checkpoint"])
    if benchmark.file_sha256(checkpoint) != source["checkpoint_sha256"]:
        raise ValueError("actor source checkpoint digest mismatch")
    world_state, stored_config, _ = load_checkpoint(checkpoint)
    config = replace(
        stored_config,
        actor_gradient="reinforce",
        behavior_kl_scale=beta,
        critic_bins=51,
        critic_output_init_scale=0.0,
        imagination_horizon=horizon,
        return_scale_ema_decay=0.99,
    )
    fresh = create_agent(
        config,
        benchmark.derive_jax_key(
            "actor-repair-world-init", seed, actor_seed, horizon, beta
        ),
    )
    frozen_world = world_state.params.world_model
    state = AgentState(
        AgentParams(frozen_world, fresh.params.actor, fresh.params.critic),
        world_state.model_optimizer,
        fresh.actor_optimizer,
        fresh.critic_optimizer,
        fresh.slow_critic,
        world_state.world_model_teacher,
    )
    arrays = benchmark.load_npz(source["dataset"])
    batch_size, sequence_length = 32, 32
    actor_updates = int(manifest["actor_updates"])
    preparation_updates = int(manifest["preparation_updates"])
    schedule = benchmark._batch_schedule(
        arrays,
        task=TASK,
        world_model_seed=benchmark.derive_seed(
            "actor-repair-shared-batches", seed, actor_seed, horizon
        ),
        updates=preparation_updates + actor_updates,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    posterior_key = benchmark.derive_jax_key(
        "actor-repair-posterior", seed, actor_seed, horizon
    )
    latest_bc = math.nan
    latest_replay_critic = math.nan
    started = time.perf_counter()
    for update in range(preparation_updates):
        replay = benchmark._materialize_batch(
            arrays,
            schedule,
            update,
            sequence_length=sequence_length,
            burn_in=config.burn_in,
        )
        sequence = jit_observe_sequence(
            state.params.world_model,
            replay["observations"],
            replay["actions"],
            jax.random.fold_in(posterior_key, update),
            config,
            is_first=replay["is_first"],
        )
        features = sequence.states.feature[:, config.burn_in :]
        actions = replay["actions"][:, config.burn_in :]
        state, bc_loss = jit_train_behavior_cloning(
            state,
            features.reshape((-1, config.feature_dim)),
            actions.reshape((-1, config.action_dim)),
            config,
        )
        state, critic_loss = jit_train_replay_critic(
            state,
            sequence.states.feature,
            replay["rewards"],
            replay["continuations"],
            config,
            loss_mask=replay["loss_mask"],
        )
        latest_bc = float(bc_loss)
        latest_replay_critic = float(critic_loss)
    behavior_prior = snapshot_behavior_prior(state.params.actor)
    return_scale_state = init_return_scale_state()
    start_key = benchmark.derive_jax_key(
        "actor-repair-start", seed, actor_seed, horizon
    )
    objective_key = benchmark.derive_jax_key(
        "actor-repair-objective", seed, actor_seed, horizon
    )
    latest = None
    for update in range(actor_updates):
        schedule_index = preparation_updates + update
        replay = benchmark._materialize_batch(
            arrays,
            schedule,
            schedule_index,
            sequence_length=sequence_length,
            burn_in=config.burn_in,
        )
        sequence = jit_observe_sequence(
            state.params.world_model,
            replay["observations"],
            replay["actions"],
            jax.random.fold_in(posterior_key, schedule_index),
            config,
            is_first=replay["is_first"],
        )
        starts = diverse_imagination_starts(
            sequence.states,
            config.burn_in,
            jax.random.fold_in(start_key, update),
        )
        state, return_scale_state, metrics = jit_train_actor_critic_dreamer3(
            state,
            return_scale_state,
            starts,
            jax.random.fold_in(objective_key, update),
            config,
            behavior_prior=behavior_prior if beta > 0.0 else None,
            reward_fn=_world_signal(str(cell["stage"])),
            continuation_fn=None,
        )
        latest = benchmark._metrics_dict(metrics)
    world_delta = _tree_delta(frozen_world, state.params.world_model)
    if world_delta != 0.0:
        raise RuntimeError("actor training mutated the frozen world model")
    returns, traces = previous._evaluate_policy_shared(
        state,
        config,
        evaluation_seeds=manifest["shared_evaluation_seeds"],
        world_model_seed=seed,
        actor_seed=actor_seed,
        horizon=horizon,
        maximum_steps=int(manifest["maximum_environment_steps"]),
    )
    benchmark._write_npz_atomic(raw_path, traces)
    trace_metrics = previous._trace_metrics(returns, traces)
    trace_metrics["mean_imagined_horizon_return"] = float(
        latest["mean_imagined_return"]
    )
    trace_metrics["mean_real_episode_return"] = float(np.mean(returns))
    trace_metrics["imagined_real_per_step_return_gap"] = float(
        latest["mean_imagined_return"] / horizon - np.mean(returns) / 1000.0
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": cell["index"],
        "stage": cell["stage"],
        "world_model_seed": seed,
        "actor_seed": actor_seed,
        "horizon": horizon,
        "behavior_kl_scale": beta,
        "actor_updates": actor_updates,
        "preparation_updates": preparation_updates,
        "metrics": {**trace_metrics, "actor_telemetry": latest},
        "world_model_parameter_delta": world_delta,
        "raw_action_traces_sha256": benchmark.file_sha256(raw_path),
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "final_behavior_cloning_loss": latest_bc,
        "final_replay_critic_loss": latest_replay_critic,
        "wall_seconds": time.perf_counter() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "runtime": benchmark.runtime_fingerprint(),
    }
    if not _finite_metrics(result):
        raise FloatingPointError("world actor result contains non-finite values")
    write_json_atomic(result_path, result)
    return result


def run_random_cell(
    mtp_root: str | Path, output_root: str | Path, **settings: Any
) -> dict[str, Any]:
    manifest = write_manifest(mtp_root, output_root, **settings)
    cell = manifest["cells"][0]
    result_path = Path(output_root) / str(cell["result_path"])
    raw_path = Path(output_root) / str(cell["raw_path"])
    if result_path.is_file():
        return read_json(result_path)
    started = time.perf_counter()
    returns, traces = previous._evaluate_random_shared(
        evaluation_seeds=manifest["shared_evaluation_seeds"],
        maximum_steps=int(manifest["maximum_environment_steps"]),
    )
    benchmark._write_npz_atomic(raw_path, traces)
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": 0,
        "stage": "random_policy",
        "world_model_seed": None,
        "actor_seed": None,
        "horizon": None,
        "behavior_kl_scale": None,
        "actor_updates": 0,
        "preparation_updates": 0,
        "metrics": previous._trace_metrics(returns, traces),
        "world_model_parameter_delta": None,
        "raw_action_traces_sha256": benchmark.file_sha256(raw_path),
        "wall_seconds": time.perf_counter() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "runtime": benchmark.runtime_fingerprint(),
    }
    write_json_atomic(result_path, result)
    return result


def validate_cell_result(
    result: Mapping[str, Any],
    cell: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    common = {
        "schema_version",
        "status",
        "source_commit",
        "manifest_sha256",
        "cell_id",
        "cell_index",
        "stage",
        "world_model_seed",
        "actor_seed",
        "horizon",
        "behavior_kl_scale",
        "actor_updates",
        "preparation_updates",
        "metrics",
        "world_model_parameter_delta",
        "raw_action_traces_sha256",
        "wall_seconds",
        "slurm_job_id",
        "runtime",
    }
    world_extra = {
        "source_checkpoint_sha256",
        "final_behavior_cloning_loss",
        "final_replay_critic_loss",
    }
    expected_keys = common | (world_extra if cell["stage"] in WORLD_STAGES else set())
    if set(result) != expected_keys:
        raise ValueError("actor-repair result schema is not exact")
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("status") != "complete"
        or result.get("source_commit") != manifest["source_commit"]
        or result.get("manifest_sha256") != manifest["manifest_sha256"]
    ):
        raise ValueError("actor-repair result identity is invalid")
    mapping = {
        "cell_id": "cell_id",
        "index": "cell_index",
        "stage": "stage",
        "world_model_seed": "world_model_seed",
        "actor_seed": "actor_seed",
        "horizon": "horizon",
        "behavior_kl_scale": "behavior_kl_scale",
    }
    for cell_key, result_key in mapping.items():
        if cell[cell_key] != result[result_key]:
            raise ValueError(f"actor-repair result differs at {cell_key}")
    if not _finite_metrics(result):
        raise ValueError("actor-repair result contains non-finite metrics")
    if cell["stage"] in WORLD_STAGES or cell["stage"] == "random_policy":
        if (
            cell["stage"] in WORLD_STAGES
            and result["world_model_parameter_delta"] != 0.0
        ):
            raise ValueError("world model changed during actor training")
        raw_path = Path(manifest["output_root"]) / str(cell["raw_path"])
        if not raw_path.is_file():
            raise ValueError("raw action trace is missing")
        if benchmark.file_sha256(raw_path) != result["raw_action_traces_sha256"]:
            raise ValueError("raw action trace digest mismatch")
        raw = benchmark.load_npz(raw_path)
        if not np.array_equal(
            raw["evaluation_seeds"],
            np.asarray(manifest["shared_evaluation_seeds"], dtype=np.uint32),
        ):
            raise ValueError("world actor did not use shared evaluation scenarios")
        recomputed = [
            float(np.sum(raw["rewards"][index, : int(length)]))
            for index, length in enumerate(raw["lengths"])
        ]
        if not np.array_equal(
            np.asarray(recomputed), np.asarray(result["metrics"]["episode_returns"])
        ):
            raise ValueError("environment returns do not recompute from raw traces")


def _stage_rows(
    rows: Sequence[Mapping[str, Any]], stage: str, beta: float | None = None
) -> list[Mapping[str, Any]]:
    selected = [row for row in rows if row["stage"] == stage]
    if beta is not None:
        selected = [row for row in selected if row["behavior_kl_scale"] == beta]
    return selected


def causal_decision(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    random_rows = _stage_rows(rows, "random_policy")
    if len(random_rows) != 1:
        raise ValueError("random baseline is absent or duplicated")
    random_return = float(random_rows[0]["metrics"]["normalized_return_mean"])
    rung_results: list[dict[str, Any]] = []
    first_failure = None
    for stage in manifest["causal_order"]:
        selected = _stage_rows(rows, stage, float(manifest["primary_decision_beta"]))
        if not selected:
            raise ValueError(f"causal rung {stage} is missing")
        if stage in PURE_STAGES:
            metric = float(np.mean([
                row["metrics"]["action_target_mean_absolute_error"] for row in selected
            ]))
            passed = metric <= float(
                manifest["decision_thresholds"]["exact_action_mean_absolute_error"]
            )
            criterion = "action_target_mean_absolute_error"
        else:
            metric = float(np.mean([
                row["metrics"]["normalized_return_mean"] for row in selected
            ]))
            passed = metric >= random_return + float(
                manifest["decision_thresholds"]["normalized_return_over_random"]
            )
            criterion = "normalized_return_mean"
        rung_results.append(
            {"stage": stage, "metric": metric, "criterion": criterion, "passed": passed}
        )
        if not passed and first_failure is None:
            first_failure = stage
    gradient_rows = _stage_rows(rows, "gradient_oracle")
    gradient_verified = bool(all(
        row["metrics"]["classification"] == "policy_gradient_estimator_verified"
        for row in gradient_rows
    ))
    return {
        "primary_behavior_kl_scale": manifest["primary_decision_beta"],
        "sensitivity_only_betas": manifest["sensitivity_only_betas"],
        "rungs": rung_results,
        "first_failing_rung": first_failure,
        "gradient_estimator_verified": gradient_verified,
        "classification": (
            "all_preregistered_actor_rungs_passed"
            if first_failure is None and gradient_verified
            else (
                "policy_gradient_estimator_mismatch"
                if not gradient_verified
                else f"first_failure_at_{first_failure}"
            )
        ),
        "independent_unit": manifest["independent_unit"],
        "actor_seed_role": manifest["actor_seed_role"],
        "claim_status": "diagnostic_only_single_world_model_seed",
    }


def finalize(
    mtp_root: str | Path, output_root: str | Path, **settings: Any
) -> dict[str, Any]:
    manifest = write_manifest(mtp_root, output_root, **settings)
    manifest = {**manifest, "output_root": str(Path(output_root).resolve())}
    rows = []
    for cell in manifest["cells"]:
        path = Path(output_root) / str(cell["result_path"])
        if not path.is_file():
            raise ValueError(f"missing actor-repair cell {cell['index']}")
        result = read_json(path)
        validate_cell_result(result, cell, manifest)
        rows.append(result)
    aggregates: dict[str, Any] = {}
    for stage in PURE_STAGES + WORLD_STAGES + SPECIAL_STAGES:
        aggregates[stage] = {}
        betas: Sequence[float | None] = BETAS if stage not in SPECIAL_STAGES else (None,)
        for beta in betas:
            selected = _stage_rows(rows, stage, beta)
            if stage in SPECIAL_STAGES:
                selected = _stage_rows(rows, stage)
            if not selected:
                continue
            numeric: dict[str, Any] = {"cells": len(selected)}
            if "action_target_mean_absolute_error" in selected[0]["metrics"]:
                numeric["mean_action_target_mae"] = float(np.mean([
                    row["metrics"]["action_target_mean_absolute_error"]
                    for row in selected
                ]))
            if "normalized_return_mean" in selected[0]["metrics"]:
                numeric["mean_normalized_return"] = float(np.mean([
                    row["metrics"]["normalized_return_mean"] for row in selected
                ]))
            if "gradient_cosine" in selected[0]["metrics"]:
                numeric["mean_gradient_cosine"] = float(np.mean([
                    row["metrics"]["gradient_cosine"] for row in selected
                ]))
            aggregates[stage][str(beta)] = numeric
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "completed_cells": len(rows),
        "registered_cells": len(manifest["cells"]),
        "aggregates": aggregates,
        "decision": causal_decision(rows, manifest),
        "total_cell_wall_seconds": float(sum(float(row["wall_seconds"]) for row in rows)),
        "runtime": benchmark.runtime_fingerprint(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    report["report_sha256"] = benchmark.object_sha256(report)
    write_json_atomic(Path(output_root) / "report.json", report)
    return report


def validate_complete_evidence(output_root: str | Path) -> dict[str, Any]:
    """Independently authenticate a completed actor-repair study directory."""

    root = Path(output_root).resolve(strict=True)
    manifest = read_json(root / "manifest.json")
    recorded_manifest_digest = manifest.get("manifest_sha256")
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("manifest_sha256", None)
    if (
        manifest.get("schema_version") != SCHEMA
        or recorded_manifest_digest != benchmark.object_sha256(unsigned_manifest)
    ):
        raise ValueError("actor-repair manifest identity is invalid")
    validate_cell_matrix(manifest["cells"])
    if manifest.get("source_commit") != _git_commit():
        raise ValueError("actor-repair evidence does not match checked-out source")
    dependency_root = Path(manifest["dependency"]["root"])
    if benchmark.file_sha256(dependency_root / "report.json") != manifest["dependency"][
        "report_file_sha256"
    ]:
        raise ValueError("actor-repair dependency report changed")
    for source in manifest["source_artifacts"]:
        checkpoint = Path(source["checkpoint"])
        dataset = Path(source["dataset"])
        if benchmark.file_sha256(checkpoint) != source["checkpoint_sha256"]:
            raise ValueError("actor-repair source checkpoint changed")
        if benchmark.array_sha256(benchmark.load_npz(dataset)) != source["dataset_sha256"]:
            raise ValueError("actor-repair source dataset changed")
    validation_manifest = {**manifest, "output_root": str(root)}
    rows = []
    for cell in manifest["cells"]:
        result_path = root / str(cell["result_path"])
        if not result_path.is_file():
            raise ValueError(f"actor-repair result {cell['index']} is missing")
        result = read_json(result_path)
        validate_cell_result(result, cell, validation_manifest)
        rows.append(result)
    report = read_json(root / "report.json")
    recorded_report_digest = report.get("report_sha256")
    unsigned_report = dict(report)
    unsigned_report.pop("report_sha256", None)
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("source_commit") != manifest["source_commit"]
        or report.get("manifest_sha256") != manifest["manifest_sha256"]
        or recorded_report_digest != benchmark.object_sha256(unsigned_report)
        or report.get("completed_cells") != len(rows)
        or report.get("registered_cells") != len(manifest["cells"])
        or report.get("decision") != causal_decision(rows, validation_manifest)
    ):
        raise ValueError("actor-repair final report is invalid")
    return report


def run_smoke(output_root: str | Path) -> dict[str, Any]:
    """Run a dependency-free engineering smoke of rungs 1--3 and diagnostics."""

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    for beta in BETAS:
        rows.append(
            {
                "stage": "exact_bandit_h1",
                "behavior_kl_scale": beta,
                "metrics": _run_exact_policy(
                    actor_seed=311,
                    horizon=1,
                    beta=beta,
                    actor_updates=500,
                    batch_size=256,
                    learned_critic=False,
                ),
            }
        )
    for stage, learned_critic in (
        ("exact_finite_no_bootstrap", False),
        ("exact_reward_learned_critic", True),
    ):
        rows.append(
            {
                "stage": stage,
                "behavior_kl_scale": PRIMARY_BETA,
                "metrics": _run_exact_policy(
                    actor_seed=311,
                    horizon=5,
                    beta=PRIMARY_BETA,
                    actor_updates=500,
                    batch_size=256,
                    learned_critic=learned_critic,
                ),
            }
        )
    safe = _run_safe_mpo(actor_seed=311, actor_updates=500, batch_size=256)
    gradient = _run_gradient_oracle(actor_seed=311, samples=32768)
    primary = [
        row
        for row in rows
        if row["behavior_kl_scale"] == PRIMARY_BETA
    ]
    exact_passed = all(
        row["metrics"]["action_target_mean_absolute_error"] <= EXACT_PASS_MAE
        for row in primary
    )
    result = {
        "schema_version": SCHEMA,
        "status": "complete",
        "source_commit": _git_commit(),
        "rows": rows,
        "safe_mpo": safe,
        "gradient_oracle": gradient,
        "exact_primary_rungs_passed": exact_passed,
        "first_failing_exact_rung": next(
            (
                row["stage"]
                for row in primary
                if row["metrics"]["action_target_mean_absolute_error"] > EXACT_PASS_MAE
            ),
            None,
        ),
        "wall_seconds": time.perf_counter() - started,
        "claim_status": "engineering_smoke_not_scientific_evidence",
    }
    write_json_atomic(output / "smoke_report.json", result)
    return result


__all__ = [
    "ACTOR_SEEDS",
    "BETAS",
    "PRIMARY_BETA",
    "PURE_STAGES",
    "SPECIAL_STAGES",
    "WORLD_MODEL_SEEDS",
    "WORLD_STAGES",
    "build_cell_matrix",
    "build_manifest",
    "causal_decision",
    "finalize",
    "run_pure_cell",
    "run_random_cell",
    "run_smoke",
    "run_world_cell",
    "shared_evaluation_seeds",
    "validate_cell_matrix",
    "validate_cell_result",
    "validate_complete_evidence",
    "write_manifest",
]
