"""Bounded finite-policy values, never regularized/off-policy critic targets."""

from dataclasses import replace
import json
import subprocess
import sys

import jax
import numpy as np
import pytest

from dreamer_imf_compare.planning_value_head import (
    HeadSettings,
    finite_return_cap,
    fit_head,
    head_bytes,
    mc_suffix_returns,
    predict_head,
)

SMALL = HeadSettings(hidden_width=16, updates=600, batch_size=64, learning_rate=0.003)


def test_production_settings_are_fixed():
    settings = HeadSettings()
    assert (settings.hidden_width, settings.updates, settings.batch_size) == (
        128,
        2000,
        256,
    )
    assert settings.learning_rate == 3e-4
    assert settings.gamma == 0.99
    assert settings.maximum_steps == 1000


@pytest.fixture(scope="module")
def learned():
    rng = np.random.default_rng(91)
    obs = rng.uniform(-1, 1, (256, 2)).astype(np.float32)
    remaining = rng.integers(200, 1001, 256)
    target = 4 + 0.7 * obs[:, 0] - 0.4 * obs[:, 1] + remaining / 1000
    head = fit_head(obs, remaining, target, seed=27, settings=SMALL)
    return head, obs, remaining, target


def test_known_signal_is_learned_on_unseen_points(learned):
    head, *_ = learned
    rng = np.random.default_rng(300)
    obs = rng.uniform(-0.9, 0.9, (100, 2)).astype(np.float32)
    remaining = rng.integers(200, 1001, 100)
    target = 4 + 0.7 * obs[:, 0] - 0.4 * obs[:, 1] + remaining / 1000
    predicted = predict_head(head, obs, remaining)
    assert np.mean((np.full_like(target, target.mean()) - target) ** 2) > 0.1
    assert np.mean((predicted - target) ** 2) < 0.01
    assert head["training"]["final_normalized_mse"] < 0.01
    assert head["training"]["initial_normalized_mse"] > 0.1
    assert head["training"]["updates"] == 600
    assert head["training"]["platform"] == "cpu"


def test_normalizers_use_only_training_data_and_queries_do_not_mutate(learned):
    head, obs, remaining, targets = learned
    stats = head["normalizers"]
    np.testing.assert_allclose(
        stats["observation_mean"], obs.astype(np.float64).mean(0), atol=1e-8
    )
    np.testing.assert_allclose(
        stats["observation_std"], obs.astype(np.float64).std(0), rtol=1e-7
    )
    assert stats["return_mean"] == pytest.approx(targets.mean(), rel=1e-7)
    assert stats["return_std"] == pytest.approx(targets.std(), rel=1e-7)
    before = head_bytes(head)
    predict_head(head, obs * 1000, remaining)
    assert head_bytes(head) == before
    restored = json.loads(before)
    np.testing.assert_array_equal(
        predict_head(head, obs, remaining), predict_head(restored, obs, remaining)
    )


def test_deterministic_cpu_fit_has_identical_serialized_bytes(learned):
    head, obs, remaining, targets = learned
    previous_default = jax.config.jax_default_device
    repeated = fit_head(obs, remaining, targets, seed=27, settings=SMALL)
    assert head_bytes(repeated) == head_bytes(head)
    assert jax.config.jax_default_device == previous_default


def test_weighted_sampling_and_normalization_are_deterministic():
    obs = np.array([[0], [1], [20]], np.float32)
    targets = np.array([1.0, 3.0, 40.0])
    weights = np.array([1.0, 3.0, 0.0])
    settings = replace(SMALL, updates=30)
    a = fit_head(
        obs, [1000] * 3, targets, seed=2, settings=settings, sample_weights=weights
    )
    b = fit_head(
        obs, [1000] * 3, targets, seed=2, settings=settings, sample_weights=weights
    )
    assert head_bytes(a) == head_bytes(b)
    assert a["normalizers"]["observation_mean"] == [0.75]
    assert a["normalizers"]["return_mean"] == 2.5
    assert len(a["training"]["normalized_weight_sha256"]) == 64
    assert len(a["training"]["sampling_probability_sha256"]) == 64
    assert a["training"]["sampling"] == "fixed_row_probabilities_with_replacement"


def test_prediction_bound_and_zero_remaining(learned):
    head = json.loads(head_bytes(learned[0]))
    head["normalizers"]["return_mean"] = 1e9
    remaining = np.array([0, 1, 2, 1000])
    predicted = predict_head(head, np.zeros((4, 2)), remaining)
    cap = finite_return_cap(remaining)
    assert predicted[0] == 0 and predicted[1] == 1
    assert not np.signbit(predicted[0])
    assert np.all(predicted >= 0) and np.all(predicted.astype(np.float64) <= cap)
    np.testing.assert_allclose(predicted, cap, rtol=1e-7)
    head["normalizers"]["return_mean"] = -1e9
    np.testing.assert_array_equal(
        predict_head(head, np.zeros((4, 2)), remaining), np.zeros(4)
    )


def test_suffix_discount_terminal_and_timeout_semantics():
    np.testing.assert_allclose(
        mc_suffix_returns([1, 0.5, 1], [1, 1, 1], gamma=0.5), [1.5, 1, 1, 0]
    )
    np.testing.assert_allclose(
        mc_suffix_returns([1, 0.5, 1], [0, 0.5, 1], gamma=0.5), [1, 0.75, 1, 0]
    )
    np.testing.assert_array_equal(mc_suffix_returns([], []), [0])
    np.testing.assert_array_equal(mc_suffix_returns([1, 1], [1, 1], gamma=0), [1, 1, 0])
    values = mc_suffix_returns(np.ones(1000), np.ones(1000))
    np.testing.assert_allclose(
        values, finite_return_cap(np.arange(1000, -1, -1)), rtol=2e-15
    )
    assert values[-1] == 0  # Native timeout continuation=1 does not bootstrap.


@pytest.mark.parametrize(
    "rewards,continuations",
    [
        ([np.nan], [1]),
        ([1.001], [1]),
        ([-0.001], [1]),
        ([1], [np.inf]),
        ([1], [-0.1]),
        ([1], [1.1]),
        ([1, 0], [1]),
        ([[1]], [[1]]),
        ([1] * 1001, [1] * 1001),
    ],
)
def test_bad_episode_data_rejected(rewards, continuations):
    with pytest.raises(ValueError):
        mc_suffix_returns(rewards, continuations)


@pytest.mark.parametrize(
    "obs,remaining,targets",
    [
        ([[np.nan]], [2], [1]),
        ([[np.inf]], [2], [1]),
        ([[1e100]], [2], [1]),
        ([[0]], [2.5], [1]),
        ([[0]], [-1], [1]),
        ([[0]], [1001], [1]),
        ([[0]], [0], [1e-20]),
        ([[0]], [1], [1.000001]),
        ([[0]], [2], [-1]),
        ([[0]], [2], [np.nan]),
        ([[0]], [2], [np.inf]),
        ([[0]], [2, 3], [1]),
        ([[0]], [2], [[1]]),
        (np.empty((0, 1)), [], []),
        (np.empty((2, 0)), [2, 2], [1, 1]),
    ],
)
def test_invalid_fit_data_rejected_before_training(obs, remaining, targets):
    with pytest.raises(ValueError):
        fit_head(obs, remaining, targets, seed=1, settings=replace(SMALL, updates=1))


@pytest.mark.parametrize(
    "weights", [[-1, 2], [0, 0], [np.nan, 1], [np.inf, 1], [1], [[1], [1]]]
)
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):
        fit_head([[0], [1]], [2, 2], [1, 1], seed=1, sample_weights=weights)


@pytest.mark.parametrize(
    "settings",
    [
        replace(SMALL, updates=0),
        replace(SMALL, batch_size=True),
        replace(SMALL, hidden_width=-1),
        replace(SMALL, gamma=1),
        replace(SMALL, gamma=np.nan),
        replace(SMALL, maximum_steps=100),
        replace(SMALL, learning_rate=0),
        replace(SMALL, normalization_floor=np.inf),
    ],
)
def test_invalid_settings_rejected(settings):
    with pytest.raises(ValueError):
        fit_head([[0]], [2], [1], seed=1, settings=settings)


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5, True])
def test_invalid_seed_rejected(seed):
    with pytest.raises(ValueError):
        fit_head([[0]], [2], [1], seed=seed)


def test_constant_training_and_zero_time_targets_are_valid():
    head = fit_head(
        np.ones((5, 1)),
        np.zeros(5),
        np.zeros(5),
        seed=1,
        settings=replace(SMALL, updates=2),
    )
    assert head["normalizers"]["observation_std"][0] > 0
    assert head["normalizers"]["return_std"] > 0
    np.testing.assert_array_equal(
        predict_head(head, np.ones((5, 1)), np.zeros(5)), np.zeros(5)
    )


def test_cap_accepts_exact_maximum_reward_mc_with_float64_roundoff():
    values = mc_suffix_returns(np.ones(1000), np.ones(1000))
    head = fit_head(
        [[0], [1], [2]],
        [1000, 999, 998],
        values[:3],
        seed=1,
        settings=replace(SMALL, updates=1),
    )
    assert head["training"]["examples"] == 3


def test_fresh_process_reproduces_exact_head_bytes():
    script = """
import hashlib
import numpy as np
from dreamer_imf_compare.planning_value_head import HeadSettings, fit_head, head_bytes
head = fit_head(np.arange(10, dtype=np.float32)[:, None], [100] * 10,
                np.arange(10, dtype=np.float64), seed=3,
                settings=HeadSettings(hidden_width=8, updates=10, batch_size=8))
print(hashlib.sha256(head_bytes(head)).hexdigest())
"""
    first = subprocess.check_output([sys.executable, "-c", script], text=True)
    second = subprocess.check_output([sys.executable, "-c", script], text=True)
    assert len(first.strip()) == 64
    assert first == second


def test_query_and_corrupt_head_are_rejected(learned):
    head = learned[0]
    with pytest.raises(ValueError):
        predict_head(head, [[np.nan, 0]], [1])
    with pytest.raises(ValueError):
        predict_head(head, [[0]], [1])
    invalid = json.loads(head_bytes(head))
    invalid["layers"][0]["weight"][0][0] = float("nan")
    with pytest.raises(ValueError):
        predict_head(invalid, [[0, 0]], [1])
    with pytest.raises(ValueError):
        head_bytes(invalid)
