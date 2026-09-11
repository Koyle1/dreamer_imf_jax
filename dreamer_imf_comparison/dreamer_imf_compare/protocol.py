"""Protocol loading, validation, hashing, and cell enumeration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REQUIRED_COMMON_FIELDS = (
    "suite",
    "tasks",
    "train_seeds",
    "observation",
    "action_repeat",
    "episode_termination",
    "action_space",
    "score",
    "evaluation_policy",
    "evaluation_schedule",
    "environment_seed_formula",
    "evaluation_seed_offset",
    "step_unit",
    "reset_transitions_counted",
    "learning_starts_policy",
)
REQUIRED_ARMS = (
    "dreamerv3_reference",
    "gaussian_rssm",
    "imf_rssm_one_step",
)
REQUIRED_PROFILES = ("local_canary", "full_gpu")


def load_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    validate_protocol(protocol)
    return protocol


def protocol_digest(protocol: Mapping[str, Any]) -> str:
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _positive_int(mapping: Mapping[str, Any], name: str) -> None:
    value = mapping.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported protocol schema")
    reference = protocol.get("reference", {})
    commit = reference.get("commit", "")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("reference commit must be a full lowercase Git SHA")
    common = protocol.get("common", {})
    missing = [name for name in REQUIRED_COMMON_FIELDS if name not in common]
    if missing:
        raise ValueError(f"missing common fields: {missing}")
    if common["suite"] != "dm_control" or common["observation"] != "proprio":
        raise ValueError("this comparison is defined for proprioceptive DMC")
    if common["action_repeat"] != 1:
        raise ValueError("action repeat must be one for the reference DMC protocol")
    if common["reset_transitions_counted"] is not True:
        raise ValueError("Dreamer-compatible counters must include reset transitions")
    tasks = common["tasks"]
    seeds = common["train_seeds"]
    if not tasks or len(tasks) != len(set(tasks)) or any(not task.startswith("dmc_") for task in tasks):
        raise ValueError("common tasks must be unique DMC task names")
    if not seeds or len(seeds) != len(set(seeds)) or any(
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds
    ):
        raise ValueError("common seeds must be unique nonnegative integers")
    arms = protocol.get("arms", {})
    if tuple(arms) != REQUIRED_ARMS:
        raise ValueError(f"arms must be ordered exactly as {REQUIRED_ARMS}")
    if arms["gaussian_rssm"].get("runner") != arms["imf_rssm_one_step"].get("runner"):
        raise ValueError("controlled Gaussian and iMF arms must use the same runner")
    for profile_name in REQUIRED_PROFILES:
        profile = protocol.get("profiles", {}).get(profile_name)
        if not isinstance(profile, Mapping):
            raise ValueError(f"missing profile {profile_name}")
        for name in (
            "training_environment_steps",
            "batch_size",
            "sequence_length",
            "prefill_steps",
            "evaluation_episodes",
        ):
            _positive_int(profile, name)
        if not isinstance(profile.get("train_ratio"), (int, float)) or profile["train_ratio"] <= 0:
            raise ValueError("train_ratio must be positive")
        if not set(profile["tasks"]).issubset(tasks):
            raise ValueError(f"{profile_name} tasks must be a common-task subset")
        if not set(profile["train_seeds"]).issubset(seeds):
            raise ValueError(f"{profile_name} seeds must be a common-seed subset")
        model = profile.get("compact_model", {})
        for name in (
            "deterministic_dim",
            "stochastic_dim",
            "embedding_dim",
            "hidden_dim",
            "imagination_horizon",
        ):
            _positive_int(model, name)
    full = protocol["profiles"]["full_gpu"]
    if full["tasks"] != tasks or full["train_seeds"] != seeds:
        raise ValueError("full_gpu must cover every declared common task and seed")
    interpretation = protocol.get("interpretation", {})
    if not all(name in interpretation for name in ("primary", "controlled", "forbidden_claim")):
        raise ValueError("comparison interpretation boundary is incomplete")


def profile_config(protocol: Mapping[str, Any], profile: str) -> dict[str, Any]:
    validate_protocol(protocol)
    if profile not in protocol["profiles"]:
        raise ValueError(f"unknown profile {profile!r}")
    value = dict(protocol["profiles"][profile])
    value["common"] = dict(protocol["common"])
    value["arms"] = dict(protocol["arms"])
    value["reference"] = dict(protocol["reference"])
    value["protocol_digest"] = protocol_digest(protocol)
    value["profile_name"] = profile
    return value


def expected_cells(protocol: Mapping[str, Any], profile: str) -> list[dict[str, Any]]:
    config = profile_config(protocol, profile)
    return [
        {"profile": profile, "arm": arm, "task": task, "seed": seed}
        for task in config["tasks"]
        for seed in config["train_seeds"]
        for arm in protocol["arms"]
    ]


def cell_id(arm: str, task: str, seed: int) -> str:
    if "/" in arm or "/" in task:
        raise ValueError("cell components cannot contain slashes")
    return f"{task}__seed{seed}__{arm}"
