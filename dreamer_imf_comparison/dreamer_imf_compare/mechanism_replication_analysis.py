"""Dependency-free paired analysis; this does not authenticate experiment artifacts."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path
from statistics import mean


def iqm(values):
    values = sorted(float(x) for x in values)
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("IQM requires nonempty finite values")
    lo, hi = len(values) * 0.25, len(values) * 0.75
    return sum(
        x * max(0, min(i + 1, hi) - max(i, lo)) for i, x in enumerate(values)
    ) / (hi - lo)


def quantile(values, p):
    values = sorted(values)
    point = (len(values) - 1) * p
    lower = int(point)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (point - lower) * (values[upper] - values[lower])


def analyze(records, protocol):
    tasks = protocol["tasks"]
    worlds = protocol["world_model_seeds"]
    actors = protocol["nested_actor_seeds"]
    arms = protocol["arms"]
    expected = set(itertools.product(tasks, worlds, actors, arms))
    rows = {}
    for row in records:
        key = (row["task"], row["world_model_seed"], row["actor_seed"], row["arm"])
        if key not in expected or key in rows:
            raise ValueError("Unexpected or duplicate cell")
        values = row["episode_returns"]
        if (
            row.get("evaluation_environment_seeds")
            != protocol["evaluation_environment_seeds"]
        ):
            raise ValueError("Evaluation seed sequence differs")
        if len(values) != protocol["evaluation_episodes_per_cell"]:
            raise ValueError("Incomplete episode set")
        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            for x in values
        ):
            raise ValueError("Nonfinite or invalid return")
        rows[key] = mean(values)
    if set(rows) != expected:
        raise ValueError("Incomplete paired matrix")
    units = {
        (t, w, arm): mean(rows[t, w, a, arm] for a in actors)
        for t, w, arm in itertools.product(tasks, worlds, arms)
    }
    result = {
        "scope": "numerical_analysis_only_requires_separate_artifact_authentication",
        "tasks": tasks,
        "cells": len(rows),
        "episodes": len(rows) * protocol["evaluation_episodes_per_cell"],
        "arm_task_iqm": {
            arm: {t: iqm(units[t, w, arm] for w in worlds) for t in tasks}
            for arm in arms
        },
        "contrasts": {},
    }
    rng = random.Random(protocol["uncertainty"]["seed"])
    distributions = {c["id"]: [] for c in protocol["contrasts"]}
    deltas = {
        c["id"]: {
            t: [
                units[t, w, c["candidate"]] - units[t, w, c["baseline"]] for w in worlds
            ]
            for t in tasks
        }
        for c in protocol["contrasts"]
    }
    for _ in range(protocol["uncertainty"]["resamples"]):
        indices = {t: [rng.randrange(len(worlds)) for _ in worlds] for t in tasks}
        for c in protocol["contrasts"]:
            name = c["id"]
            distributions[name].append(
                mean(iqm(deltas[name][t][i] for i in indices[t]) for t in tasks)
            )
    alpha = (1 - protocol["uncertainty"]["interval_level_per_contrast"]) / 2
    for c in protocol["contrasts"]:
        name = c["id"]
        task_effects = {t: iqm(deltas[name][t]) for t in tasks}
        interval = [
            quantile(distributions[name], alpha),
            quantile(distributions[name], 1 - alpha),
        ]
        result["contrasts"][name] = {
            "candidate": c["candidate"],
            "baseline": c["baseline"],
            "paired_world_seed_deltas": {
                t: dict(zip(map(str, worlds), deltas[name][t])) for t in tasks
            },
            "task_paired_delta_iqm": task_effects,
            "primary_effect": mean(task_effects.values()),
            "bootstrap_interval": interval,
            "interval_level": protocol["uncertainty"]["interval_level_per_contrast"],
            "positive_task_fraction": mean(x > 0 for x in task_effects.values()),
            "positive_world_seed_fraction": mean(
                x > 0 for xs in deltas[name].values() for x in xs
            ),
            "positive_interval": interval[0] > 0,
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument(
        "--protocol", type=Path, default=Path(__file__).with_name("protocol.json")
    )
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                json.loads(args.records.read_text()),
                json.loads(args.protocol.read_text()),
            ),
            indent=2,
            allow_nan=False,
        )
    )
