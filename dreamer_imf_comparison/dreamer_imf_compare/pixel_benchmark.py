"""Authenticated secondary pixel benchmark for shortcut forcing and trajectory iMF.

This module deliberately keeps the visual study outside the confirmatory gate in
``matched_objective_protocol.json``.  It uses real DeepMind Control renders, a
deterministic uint8 preprocessing path, the same compact JAX world-model code for
both objectives, and checkpoint-recomputed open-loop predictions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, fields
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import stat
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
PROTOCOL_PATH = PROJECT_ROOT / "pixel_benchmark_protocol.json"
BASE_PROTOCOL_PATH = PROJECT_ROOT / "matched_objective_protocol.json"
ARM_ORDER = ("shortcut_forcing", "trajectory_imf")
ARTIFACT_FILES = (
    "protocol.json",
    "dataset.npz",
    "batch_schedule.npz",
    "objective_keys.npz",
    "checkpoint_shortcut_forcing.pkl",
    "checkpoint_trajectory_imf.pkl",
    "predictions.npz",
    "result.json",
)
SEALED_ARTIFACT_FILES = (*ARTIFACT_FILES, "manifest.json", "seal.json")
RESULT_KEYS = {
    "schema_version",
    "status",
    "study_role",
    "claim_eligible_for_primary_gate",
    "primary_gate_substitution",
    "profile",
    "task",
    "seed",
    "track",
    "renderer_backend",
    "action_dim",
    "action_repeat",
    "environment_seeds",
    "protocol_sha256",
    "base_protocol_sha256",
    "runtime",
    "shared_initial_trunk_sha256",
    "allocations",
    "training",
    "compute",
    "evaluation_identity",
    "metrics",
    "artifact_digests",
}
RUNTIME_KEYS = {
    "python",
    "platform",
    "jax",
    "jaxlib",
    "backend",
    "jax_enable_x64",
    "device_count",
    "visible_device_count",
    "device_platforms",
    "device_kinds",
    "mujoco_gl",
    "jax_enable_x64_environment",
    "jax_platform_name",
    "cuda_visible_devices",
    "numpy",
    "dm_control",
    "mujoco",
}
LOSS_KEYS = {
    "total",
    "reconstruction",
    "reward",
    "continuation",
    "representation",
    "prior",
    "overshooting",
    "overshooting_distance_5",
    "overshooting_distance_15",
    "imf_loss_u",
    "imf_loss_v",
    "imf_shortcut",
    "imf_endpoint",
    "causal_consistency",
    "causal_observation",
    "causal_reward",
}
METRIC_KEYS = {
    "horizons",
    "normalized_visual_mse",
    "predictive_mean_psnr_db",
    "predictive_mean_global_ssim",
    "energy_score",
    "temporal_difference_mse",
    "normalized_visual_mse_auc",
}
COMPILER_EVIDENCE_KEYS = {
    "flops",
    "cost_analysis",
    "cost_analysis_sha256",
    "stablehlo_sha256",
}
DATASET_ARRAY_KEYS = {
    "observations",
    "actions",
    "rewards",
    "continuations",
    "is_first",
    "is_last",
    "native_steps",
    "episode_id",
    "episode_step",
}
LIBRARY_SOURCE_BASENAMES = (
    "__init__.py",
    "agent.py",
    "checkpoint.py",
    "config.py",
    "fidelity.py",
    "imf.py",
    "nn.py",
    "optim.py",
    "planner.py",
    "replay.py",
    "shortcut.py",
    "trajectory.py",
    "types.py",
    "uncertainty.py",
    "world_model.py",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], label: str
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ValueError(
            f"{label} schema is not exact: "
            f"missing={sorted(set(expected) - actual)}, "
            f"extra={sorted(actual - set(expected))}"
        )


def _require_sha256(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _reject_symlink_chain(path: str | Path, *, label: str) -> Path:
    """Return an absolute lexical path after rejecting every symlink component.

    Checking only ``Path.is_symlink()`` on the final component is insufficient:
    an otherwise regular cell can be redirected through a symlinked track or
    task directory.  Walk the lexical path with ``lstat`` before resolving it so
    both existing and broken links fail closed.  ``..`` is rejected because its
    filesystem interpretation after a symlink is not a lexical containment
    operation.
    """

    lexical = Path(path).expanduser()
    if any(part == ".." for part in lexical.parts):
        raise ValueError(f"{label} must not contain parent traversal")
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor = cursor / part
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label} must not contain a symlink component: {cursor}")
    return lexical


def _artifact_root(root: str | Path) -> Path:
    """Resolve a real artifact directory with no symlink in its lexical chain."""

    lexical = _reject_symlink_chain(root, label="pixel artifact root")
    if not lexical.is_dir():
        raise FileNotFoundError(f"pixel artifact directory does not exist: {lexical}")
    return lexical.resolve(strict=True)


def _validate_exact_artifact_entries(root: Path) -> None:
    entries = {entry.name: entry for entry in root.iterdir()}
    expected = set(SEALED_ARTIFACT_FILES)
    if set(entries) != expected:
        raise ValueError(
            "sealed pixel artifact entry set is not exact: "
            f"missing={sorted(expected - set(entries))}, "
            f"extra={sorted(set(entries) - expected)}"
        )
    for name, entry in entries.items():
        mode = entry.lstat().st_mode
        if entry.is_symlink() or not stat.S_ISREG(mode):
            raise ValueError(
                f"sealed pixel artifact entry must be a regular non-symlink file: {name}"
            )


def _json_load(path: str | Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"{path} contains non-finite JSON constant {value}")

    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _atomic_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=destination.name + ".",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _atomic_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=destination.name + ".", suffix=".npz"
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError(f"NPZ archive contains duplicate member names: {path}")
        return {name: np.asarray(archive[name]) for name in archive.files}


def derive_numpy_seed(domain: str, *parts: object) -> int:
    payload = canonical_bytes(["pixel-benchmark", domain, *parts])
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def derive_jax_key(domain: str, *parts: object):
    import jax.numpy as jnp

    payload = canonical_bytes(["pixel-benchmark", domain, *parts])
    digest = hashlib.sha256(payload).digest()
    return jnp.asarray(
        [
            int.from_bytes(digest[:4], "big", signed=False),
            int.from_bytes(digest[4:8], "big", signed=False),
        ],
        dtype=jnp.uint32,
    )


def derive_environment_seed(domain: str, task: str, seed: int) -> int:
    if domain not in ("environment-train", "environment-heldout"):
        raise ValueError("environment seed domain is not registered")
    return derive_numpy_seed(domain, task, int(seed)) % (2**32 - 1)


def load_protocol(path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = _json_load(path)
    validate_protocol(protocol)
    return protocol


def _base_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    path = PROJECT_ROOT / str(protocol["base_model_contract"]["path"])
    if path.resolve() != BASE_PROTOCOL_PATH.resolve():
        raise ValueError("base model contract must resolve to matched_objective_protocol.json")
    expected = str(protocol["base_model_contract"]["sha256"])
    if file_sha256(path) != expected:
        raise ValueError("matched objective protocol digest differs from the pixel freeze")
    return _json_load(path)


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "study_role",
        "base_model_contract",
        "environment",
        "rendering",
        "observation",
        "evaluation",
        "analysis",
        "compute",
        "runtime_contract",
        "profiles",
        "randomness",
        "provenance",
    }
    if set(protocol) != required:
        raise ValueError("pixel protocol top-level keys are incomplete or contain extras")
    if protocol["schema_version"] != "trajectory-imf-pixel-secondary-v1":
        raise ValueError("unexpected pixel protocol schema")
    role = protocol["study_role"]
    if role.get("claim_eligible_for_primary_gate") is not False:
        raise ValueError("pixel track must be ineligible for the primary gate")
    non_substitution = str(role.get("non_substitution", "")).lower()
    if "substitute" not in non_substitution or "primary" not in non_substitution:
        raise ValueError("pixel track must state its non-substitution rule")
    base = _base_protocol(protocol)
    pointer = protocol["base_model_contract"]["common_config_pointer"]
    if pointer != "canonical_executable_config.dreamer_config_non_task_shape_common":
        raise ValueError("common DreamerConfig pointer is not frozen")
    if tuple(protocol["base_model_contract"]["arm_order"]) != ARM_ORDER:
        raise ValueError("arm order must be canonical")

    rendering = protocol["rendering"]
    observation = protocol["observation"]
    if (
        rendering.get("raw_height") != 64
        or rendering.get("raw_width") != 64
        or rendering.get("channels") != 3
        or rendering.get("dtype") != "uint8"
        or rendering.get("hard_backend") != "egl"
        or rendering.get("local_smoke_backend") != "glfw"
    ):
        raise ValueError("renderer geometry/backend contract changed")
    if (
        observation.get("model_height") != 32
        or observation.get("model_width") != 32
        or observation.get("frame_stack") != 3
        or observation.get("model_shape") != [32, 32, 9]
        or observation.get("jax_preprocessing") != "astype(float32)/255.0"
    ):
        raise ValueError("pixel observation construction changed")
    environment = protocol["environment"]
    if (
        environment.get("action_repeat") != 2
        or environment.get("native_termination") is not True
        or environment.get("time_limit_override") is not None
        or environment.get("reset_action")
        != "all-zero normalized action aligned with the reset observation"
    ):
        raise ValueError("reset/action-repeat semantics changed")
    tasks = environment.get("tasks", {})
    if not tasks or any(row.get("camera_id") != 0 for row in tasks.values()):
        raise ValueError("task camera IDs must be explicit")

    evaluation = protocol["evaluation"]
    if evaluation.get("nfe_frontier") != [1, 2, 4]:
        raise ValueError("NFE frontier must be exactly 1/2/4")
    if evaluation.get("primary_nfe") != {
        "shortcut_forcing": 4,
        "trajectory_imf": 1,
    }:
        raise ValueError("primary NFE mapping changed")
    if not evaluation.get("paired_windows_actions_context_keys_and_base_noise"):
        raise ValueError("visual evaluation must be paired")
    expected_metrics = {
        "normalized_visual_mse",
        "predictive_mean_psnr_db",
        "predictive_mean_global_ssim",
        "energy_score",
        "temporal_difference_mse",
        "auc",
    }
    if set(evaluation.get("metrics", {})) != expected_metrics:
        raise ValueError("visual metric set changed")
    analysis = protocol["analysis"]
    if (
        analysis.get("bootstrap_draws") != 10000
        or analysis.get("confidence_level") != 0.95
        or analysis.get("bootstrap_seed") != 704729
        or "cannot enter" not in analysis.get("multiplicity_role", "")
    ):
        raise ValueError("secondary visual analysis rule changed")

    profiles = protocol["profiles"]
    if set(profiles) != {"smoke", "hard"}:
        raise ValueError("pixel profiles must be exactly smoke and hard")
    if profiles["smoke"].get("claim_eligible") is not False:
        raise ValueError("smoke must be claim-ineligible")
    if profiles["hard"].get("claim_eligible") is not False:
        raise ValueError("hard visual track remains secondary")
    if profiles["smoke"].get("tasks") != ["dmc_pendulum_swingup"]:
        raise ValueError("smoke task changed")
    if profiles["hard"].get("tasks") != [
        "dmc_walker_walk",
        "dmc_cheetah_run",
        "dmc_finger_spin",
    ]:
        raise ValueError("hard visual task suite changed")
    for profile_name, profile in profiles.items():
        if not set(profile["tasks"]).issubset(tasks):
            raise ValueError(f"{profile_name} contains an undeclared DMC task")
        if sorted(profile["horizons"]) != profile["horizons"]:
            raise ValueError("horizons must be sorted")
        if profile["context_length"] <= 0 or profile["windows"] <= 0:
            raise ValueError("context length and windows must be positive")
        if profile["sequence_length"] <= profile["burn_in"]:
            raise ValueError("burn-in must leave a loss-bearing suffix")
        if profile["heldout_observations"] < (
            profile["context_length"] + max(profile["horizons"])
        ):
            raise ValueError("held-out replay cannot contain even one evaluation window")
        if profile_name == "hard" and profile["horizons"] != evaluation["horizons"]:
            raise ValueError("hard profile must use every registered horizon")
    if protocol["compute"].get("tracks") != [
        "equal_updates",
        "equal_compiler_flops",
    ]:
        raise ValueError("compute tracks changed")
    runtime_contract = protocol["runtime_contract"]
    _require_exact_keys(
        runtime_contract, {"jax_enable_x64", "smoke", "hard"}, "runtime contract"
    )
    if runtime_contract["jax_enable_x64"] is not False:
        raise ValueError("pixel runtime must keep JAX x64 disabled")
    _require_exact_keys(
        runtime_contract["smoke"],
        {
            "claim_eligible",
            "allowed_backends",
            "version_binding",
            "device_binding",
            "visible_device_count",
        },
        "smoke runtime contract",
    )
    if (
        runtime_contract["smoke"]["claim_eligible"] is not False
        or runtime_contract["smoke"]["allowed_backends"] != ["cpu", "gpu"]
        or runtime_contract["smoke"]["version_binding"] != "informational_only"
        or runtime_contract["smoke"]["device_binding"] != "informational_only"
        or runtime_contract["smoke"]["visible_device_count"] is not None
    ):
        raise ValueError("smoke runtime portability or claim isolation changed")
    hard_runtime = runtime_contract["hard"]
    _require_exact_keys(
        hard_runtime,
        {
            "claim_eligible",
            "backend",
            "visible_device_count",
            "device_platforms",
            "device_kind_substring",
            "versions",
            "environment",
            "cuda_visible_devices",
        },
        "hard runtime contract",
    )
    expected_versions = {
        "python": "3.12.3",
        "jax": "0.8.1",
        "jaxlib": "0.8.1",
        "numpy": "2.5.3",
        "dm_control": "1.0.46",
        "mujoco": "3.13.0",
    }
    expected_environment = {
        "JAX_ENABLE_X64": "0",
        "JAX_PLATFORM_NAME": "gpu",
        "MUJOCO_GL": "egl",
    }
    if (
        hard_runtime["claim_eligible"] is not False
        or hard_runtime["backend"] != "gpu"
        or hard_runtime["visible_device_count"] != 1
        or hard_runtime["device_platforms"] != ["gpu"]
        or hard_runtime["device_kind_substring"] != "L40S"
        or hard_runtime["versions"] != expected_versions
        or hard_runtime["environment"] != expected_environment
        or "one nonempty selector" not in hard_runtime["cuda_visible_devices"]
    ):
        raise ValueError("hard GPU runtime freeze changed")
    randomness = protocol["randomness"]
    if randomness.get("environment_seed_domains") != {
        "training": "environment-train",
        "heldout": "environment-heldout",
    } or "modulo (2^32-1)" not in randomness.get("environment_seed_formula", ""):
        raise ValueError("environment seed derivation changed")

    # This also detects a missing DreamerConfig field in the inherited freeze.
    for profile_name, profile in profiles.items():
        task = profile["tasks"][0]
        del task
        for arm in ARM_ORDER:
            resolved_config(protocol, profile_name, arm, action_dim=1)
    common = base["canonical_executable_config"][
        "dreamer_config_non_task_shape_common"
    ]
    if common.get("overshooting_scale") != 0.0:
        raise ValueError("visual objective comparison must not add overshooting")


def resolved_config(
    protocol: Mapping[str, Any],
    profile: str,
    arm: str,
    *,
    action_dim: int,
    nfe: int | None = None,
):
    from imf_dreamer_jax import DreamerConfig

    if profile not in protocol["profiles"]:
        raise ValueError(f"unknown pixel profile {profile!r}")
    if arm not in ARM_ORDER:
        raise ValueError(f"unknown objective arm {arm!r}")
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    base = _base_protocol(protocol)
    common = dict(
        base["canonical_executable_config"][
            "dreamer_config_non_task_shape_common"
        ]
    )
    # Preserve the frozen parent protocol identity while explicitly disabling
    # the later causal-consistency extension for every legacy pixel arm.
    common.update(
        imf_causal_consistency_scale=0.0,
        imf_causal_reward_scale=1.0,
        imf_causal_huber_delta=1.0,
        imf_causal_normalization_epsilon=1e-3,
        reward_prediction_horizon=0,
        reward_bins=1,
        reward_symlog_min=-20.0,
        reward_symlog_max=20.0,
        return_scale_ema_decay=0.99,
    )
    common["overshooting_distances"] = tuple(common["overshooting_distances"])
    common.update(protocol["profiles"][profile]["model_overrides"])
    common["burn_in"] = int(protocol["profiles"][profile]["burn_in"])
    common.update(
        protocol["base_model_contract"][
            "shortcut_overrides" if arm == "shortcut_forcing" else "trajectory_overrides"
        ]
    )
    if nfe is not None:
        if int(nfe) not in protocol["evaluation"]["nfe_frontier"]:
            raise ValueError("NFE is outside the frozen frontier")
        common["shortcut_sampling_steps" if arm == "shortcut_forcing" else "imf_sampling_steps"] = int(nfe)
    common["observation_shape"] = tuple(protocol["observation"]["model_shape"])
    common["action_dim"] = int(action_dim)
    expected = {field.name for field in fields(DreamerConfig)}
    if set(common) != expected:
        raise ValueError(
            "resolved pixel DreamerConfig is incomplete: "
            f"missing={sorted(expected - set(common))}, extra={sorted(set(common) - expected)}"
        )
    return DreamerConfig(**common)


def area_downsample_uint8(frame: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Deterministic integer area averaging with round-half-up."""

    value = np.asarray(frame)
    if value.ndim < 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
        raise ValueError("RGB frame must have uint8 shape [..., height, width, 3]")
    target_h, target_w = (int(output_shape[0]), int(output_shape[1]))
    height, width = value.shape[-3:-1]
    if target_h <= 0 or target_w <= 0 or height % target_h or width % target_w:
        raise ValueError("area downsampling requires positive integer scale factors")
    factor_h, factor_w = height // target_h, width // target_w
    reshaped = value.reshape(
        (*value.shape[:-3], target_h, factor_h, target_w, factor_w, 3)
    ).astype(np.uint32)
    total = reshaped.sum(axis=(-4, -2), dtype=np.uint32)
    divisor = factor_h * factor_w
    return ((total + divisor // 2) // divisor).astype(np.uint8)


def preprocess_pixels_jax(observation: Any):
    import jax.numpy as jnp

    value = jnp.asarray(observation)
    if value.dtype != jnp.uint8:
        raise ValueError("stored pixel observations must be uint8")
    if value.shape[-3:] != (32, 32, 9):
        raise ValueError("pixel observation must end in frozen shape (32, 32, 9)")
    return value.astype(jnp.float32) / jnp.asarray(255.0, jnp.float32)


@dataclass(frozen=True)
class PixelStep:
    observation: np.ndarray
    reward: float
    continuation: float
    is_last: bool
    native_steps: int


def _make_environment(task: str, seed: int):
    from dreamer_imf_compare.dmc import _load_standard_domain, split_task

    domain, task_name = split_task(task)
    return _load_standard_domain(domain, task_name, np.random.RandomState(int(seed)))


class PixelDMCAdapter:
    """Reset-safe real RGB DMC adapter with explicit repeated-action semantics."""

    def __init__(
        self,
        task: str,
        *,
        seed: int,
        action_repeat: int,
        raw_height: int,
        raw_width: int,
        model_height: int,
        model_width: int,
        frame_stack: int,
        camera_id: int,
        renderer_backend: str,
        environment: Any | None = None,
    ) -> None:
        if action_repeat <= 0 or frame_stack <= 0:
            raise ValueError("action_repeat and frame_stack must be positive")
        if environment is None:
            configured = os.environ.get("MUJOCO_GL")
            if configured is None:
                os.environ["MUJOCO_GL"] = renderer_backend
            elif configured != renderer_backend:
                raise RuntimeError(
                    f"MUJOCO_GL={configured!r} conflicts with frozen backend {renderer_backend!r}"
                )
            environment = _make_environment(task, int(seed))
        self.task = task
        self.seed = int(seed)
        self.action_repeat = int(action_repeat)
        self.raw_height = int(raw_height)
        self.raw_width = int(raw_width)
        self.model_height = int(model_height)
        self.model_width = int(model_width)
        self.frame_stack = int(frame_stack)
        self.camera_id = int(camera_id)
        self.renderer_backend = renderer_backend
        self._environment = environment
        self._action_spec = environment.action_spec()
        self.action_shape = tuple(int(value) for value in self._action_spec.shape)
        if len(self.action_shape) != 1 or not self.action_shape[0]:
            raise ValueError("pixel comparison requires a flat continuous action space")
        self._frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self._needs_reset = True

    @property
    def action_dim(self) -> int:
        return self.action_shape[0]

    def _render(self) -> np.ndarray:
        frame = np.asarray(
            self._environment.physics.render(
                height=self.raw_height,
                width=self.raw_width,
                camera_id=self.camera_id,
                depth=False,
                segmentation=False,
            )
        )
        if frame.shape != (self.raw_height, self.raw_width, 3) or frame.dtype != np.uint8:
            raise RuntimeError("DMC renderer violated the frozen RGB uint8 contract")
        return area_downsample_uint8(frame, (self.model_height, self.model_width))

    def _stacked(self) -> np.ndarray:
        if len(self._frames) != self.frame_stack:
            raise RuntimeError("frame stack is not initialized")
        return np.concatenate(tuple(self._frames), axis=-1).astype(np.uint8, copy=False)

    def reset(self) -> np.ndarray:
        self._environment.reset()
        frame = self._render()
        self._frames.clear()
        self._frames.extend(frame.copy() for _ in range(self.frame_stack))
        self._needs_reset = False
        return self._stacked().copy()

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        normalized = np.asarray(action, dtype=np.float32)
        if normalized.shape != self.action_shape:
            raise ValueError(f"action must have shape {self.action_shape}")
        normalized = np.clip(normalized, -1.0, 1.0)
        minimum = np.asarray(self._action_spec.minimum, dtype=np.float32)
        maximum = np.asarray(self._action_spec.maximum, dtype=np.float32)
        return minimum + (normalized + 1.0) * 0.5 * (maximum - minimum)

    def step(self, action: np.ndarray) -> PixelStep:
        if self._needs_reset:
            raise RuntimeError("reset is required before stepping the pixel environment")
        native_action = self._denormalize_action(action)
        reward = 0.0
        continuation = 1.0
        last = False
        native_steps = 0
        for _ in range(self.action_repeat):
            time_step = self._environment.step(native_action)
            native_steps += 1
            reward += float(time_step.reward or 0.0)
            continuation *= float(
                1.0 if time_step.discount is None else time_step.discount
            )
            last = bool(time_step.last())
            if last:
                break
        frame = self._render()
        self._frames.append(frame)
        self._needs_reset = last
        return PixelStep(
            observation=self._stacked().copy(),
            reward=reward,
            continuation=continuation,
            is_last=last,
            native_steps=native_steps,
        )

    def close(self) -> None:
        self._environment.close()


def _adapter_kwargs(
    protocol: Mapping[str, Any], task: str, seed: int, profile: str
) -> dict[str, Any]:
    if task not in protocol["profiles"][profile]["tasks"]:
        raise ValueError(f"task {task!r} is not registered for profile {profile!r}")
    rendering = protocol["rendering"]
    observation = protocol["observation"]
    return {
        "task": task,
        "seed": int(seed),
        "action_repeat": int(protocol["environment"]["action_repeat"]),
        "raw_height": int(rendering["raw_height"]),
        "raw_width": int(rendering["raw_width"]),
        "model_height": int(observation["model_height"]),
        "model_width": int(observation["model_width"]),
        "frame_stack": int(observation["frame_stack"]),
        "camera_id": int(protocol["environment"]["tasks"][task]["camera_id"]),
        "renderer_backend": str(
            rendering["local_smoke_backend"]
            if profile == "smoke"
            else rendering["hard_backend"]
        ),
    }


def collect_pixel_dataset(
    adapter: PixelDMCAdapter,
    observations: int,
    *,
    exploration_seed: int,
    ar_coefficient: float,
    innovation_scale: float,
) -> dict[str, np.ndarray]:
    if observations <= 1:
        raise ValueError("dataset needs at least two observations")
    rng = np.random.default_rng(int(exploration_seed))
    frames: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    continuations: list[float] = []
    firsts: list[bool] = []
    lasts: list[bool] = []
    native_steps: list[int] = []
    episode_ids: list[int] = []
    episode_steps: list[int] = []
    episode = -1
    step_in_episode = 0
    needs_reset = True
    previous_action = np.zeros((adapter.action_dim,), dtype=np.float32)
    while len(frames) < observations:
        if needs_reset:
            episode += 1
            step_in_episode = 0
            previous_action = np.zeros_like(previous_action)
            frames.append(adapter.reset())
            actions.append(np.zeros_like(previous_action))
            rewards.append(0.0)
            continuations.append(1.0)
            firsts.append(True)
            lasts.append(False)
            native_steps.append(0)
            episode_ids.append(episode)
            episode_steps.append(0)
            needs_reset = False
            continue
        innovation = rng.normal(0.0, innovation_scale, size=previous_action.shape)
        action = np.clip(
            ar_coefficient * previous_action + innovation, -1.0, 1.0
        ).astype(np.float32)
        transition = adapter.step(action)
        step_in_episode += 1
        frames.append(transition.observation)
        actions.append(action)
        rewards.append(transition.reward)
        continuations.append(transition.continuation)
        firsts.append(False)
        lasts.append(transition.is_last)
        native_steps.append(transition.native_steps)
        episode_ids.append(episode)
        episode_steps.append(step_in_episode)
        previous_action = action
        needs_reset = transition.is_last
    result = {
        "observations": np.stack(frames).astype(np.uint8),
        "actions": np.stack(actions).astype(np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "continuations": np.asarray(continuations, dtype=np.float32),
        "is_first": np.asarray(firsts, dtype=np.bool_),
        "is_last": np.asarray(lasts, dtype=np.bool_),
        "native_steps": np.asarray(native_steps, dtype=np.int16),
        "episode_id": np.asarray(episode_ids, dtype=np.int32),
        "episode_step": np.asarray(episode_steps, dtype=np.int32),
    }
    validate_dataset(result, adapter.action_dim, observations, adapter.action_repeat)
    return result


def validate_dataset(
    arrays: Mapping[str, np.ndarray],
    action_dim: int,
    observations: int,
    action_repeat: int,
) -> None:
    required = DATASET_ARRAY_KEYS
    if set(arrays) != required:
        raise ValueError("pixel dataset arrays are incomplete or contain extras")
    if arrays["observations"].shape != (observations, 32, 32, 9):
        raise ValueError("pixel dataset has the wrong observation shape")
    if arrays["observations"].dtype != np.uint8:
        raise ValueError("pixel dataset observations must remain uint8")
    if arrays["actions"].shape != (observations, action_dim):
        raise ValueError("pixel dataset actions have the wrong shape")
    expected_dtypes = {
        "actions": np.dtype(np.float32),
        "rewards": np.dtype(np.float32),
        "continuations": np.dtype(np.float32),
        "is_first": np.dtype(np.bool_),
        "is_last": np.dtype(np.bool_),
        "native_steps": np.dtype(np.int16),
        "episode_id": np.dtype(np.int32),
        "episode_step": np.dtype(np.int32),
    }
    for name, expected_dtype in expected_dtypes.items():
        if np.asarray(arrays[name]).dtype != expected_dtype:
            raise ValueError(f"pixel dataset {name} has the wrong dtype")
    for name in ("rewards", "continuations", "is_first", "is_last", "native_steps", "episode_id", "episode_step"):
        if arrays[name].shape != (observations,):
            raise ValueError(f"pixel dataset {name} has the wrong shape")
    if not bool(arrays["is_first"][0]):
        raise ValueError("pixel dataset must begin at reset")
    first = arrays["is_first"].astype(bool)
    if not np.all(arrays["actions"][first] == 0.0):
        raise ValueError("reset observations must carry a zero action")
    if not np.all(arrays["rewards"][first] == 0.0):
        raise ValueError("reset observations must carry zero reward")
    if not np.all(arrays["continuations"][first] == 1.0) or np.any(
        arrays["is_last"][first]
    ):
        raise ValueError("reset observations must be continuing and nonterminal")
    if not np.all(arrays["native_steps"][first] == 0):
        raise ValueError("reset observations must execute zero native steps")
    nonfirst = ~first
    if np.any(arrays["native_steps"][nonfirst] < 1) or np.any(
        arrays["native_steps"][nonfirst] > action_repeat
    ):
        raise ValueError("non-reset action repeat count is invalid")
    if not np.isfinite(arrays["actions"]).all() or np.max(np.abs(arrays["actions"])) > 1.0:
        raise ValueError("normalized actions are invalid")
    if not np.isfinite(arrays["rewards"]).all() or not np.isfinite(
        arrays["continuations"]
    ).all():
        raise ValueError("reward/continuation is non-finite")
    if np.any(arrays["continuations"] < 0.0) or np.any(
        arrays["continuations"] > 1.0
    ):
        raise ValueError("continuations must lie in [0,1]")
    for index in np.flatnonzero(first):
        if arrays["episode_step"][index] != 0:
            raise ValueError("reset episode step must be zero")
        if index and not arrays["is_last"][index - 1]:
            raise ValueError("an internal reset must follow a terminal observation")
        stack = arrays["observations"][index]
        if not (
            np.array_equal(stack[..., :3], stack[..., 3:6])
            and np.array_equal(stack[..., 3:6], stack[..., 6:9])
        ):
            raise ValueError("reset frame stack must repeat one rendered frame")
    for index in range(1, observations):
        if bool(arrays["is_last"][index - 1]) != bool(first[index]):
            raise ValueError("terminal/reset adjacency is not exact")
        expected_episode = int(arrays["episode_id"][index - 1]) + int(first[index])
        expected_step = (
            0 if first[index] else int(arrays["episode_step"][index - 1]) + 1
        )
        if (
            int(arrays["episode_id"][index]) != expected_episode
            or int(arrays["episode_step"][index]) != expected_step
        ):
            raise ValueError("episode identifiers or steps are not canonical")
        if not first[index] and not np.array_equal(
            arrays["observations"][index, ..., :6],
            arrays["observations"][index - 1, ..., 3:9],
        ):
            raise ValueError("within-episode frame stack does not shift exactly")


def validate_exploration_actions(
    arrays: Mapping[str, np.ndarray],
    *,
    exploration_seed: int,
    ar_coefficient: float,
    innovation_scale: float,
) -> None:
    """Recreate the registered AR(1) policy instead of trusting retained actions."""

    actions = np.asarray(arrays["actions"])
    first = np.asarray(arrays["is_first"], np.bool_)
    if actions.ndim != 2 or len(actions) != len(first):
        raise ValueError("retained exploration arrays have incompatible shapes")
    rng = np.random.default_rng(int(exploration_seed))
    previous = np.zeros((actions.shape[1],), np.float32)
    for index in range(len(actions)):
        if first[index]:
            previous = np.zeros_like(previous)
            expected = previous
        else:
            innovation = rng.normal(
                0.0, float(innovation_scale), size=previous.shape
            )
            expected = np.clip(
                float(ar_coefficient) * previous + innovation, -1.0, 1.0
            ).astype(np.float32)
            previous = expected
        if not np.array_equal(actions[index], expected):
            raise ValueError(
                f"retained action at index {index} is not the frozen AR(1) replay"
            )


def replay_pixel_dataset(
    protocol: Mapping[str, Any],
    *,
    profile: str,
    task: str,
    environment_seed: int,
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Replay retained actions in DMC and byte-compare every rendered observation."""

    adapter = PixelDMCAdapter(
        **_adapter_kwargs(protocol, task, int(environment_seed), profile)
    )
    episode = -1
    episode_step = 0
    try:
        if adapter.action_dim != np.asarray(arrays["actions"]).shape[1]:
            raise ValueError("DMC replay action dimension differs from the retained dataset")
        for index in range(len(arrays["observations"])):
            if bool(arrays["is_first"][index]):
                episode += 1
                episode_step = 0
                observation = adapter.reset()
                reward = np.float32(0.0)
                continuation = np.float32(1.0)
                is_last = False
                native_steps = 0
            else:
                transition = adapter.step(np.asarray(arrays["actions"][index], np.float32))
                episode_step += 1
                observation = transition.observation
                reward = np.float32(transition.reward)
                continuation = np.float32(transition.continuation)
                is_last = bool(transition.is_last)
                native_steps = int(transition.native_steps)
            if not np.array_equal(observation, arrays["observations"][index]):
                raise ValueError(
                    f"DMC replay rendered pixels differ at retained index {index}"
                )
            if (
                np.float32(arrays["rewards"][index]).tobytes() != reward.tobytes()
                or np.float32(arrays["continuations"][index]).tobytes()
                != continuation.tobytes()
                or bool(arrays["is_last"][index]) != is_last
                or int(arrays["native_steps"][index]) != native_steps
                or int(arrays["episode_id"][index]) != episode
                or int(arrays["episode_step"][index]) != episode_step
            ):
                raise ValueError(
                    f"DMC replay transition metadata differs at retained index {index}"
                )
    finally:
        adapter.close()


def dataset_archive_arrays(
    train: Mapping[str, np.ndarray], heldout: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {
        f"{split}_{name}": np.asarray(value)
        for split, arrays in (("train", train), ("heldout", heldout))
        for name, value in sorted(arrays.items())
    }


def split_dataset_archive(arrays: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    expected = {
        f"{split}_{name}"
        for split in ("train", "heldout")
        for name in DATASET_ARRAY_KEYS
    }
    if set(arrays) != expected:
        raise ValueError("dataset archive array set is not exact")
    train = {name[6:]: value for name, value in arrays.items() if name.startswith("train_")}
    heldout = {
        name[8:]: value for name, value in arrays.items() if name.startswith("heldout_")
    }
    return train, heldout


def make_batch_schedule(
    observations: int,
    *,
    updates: int,
    batch_size: int,
    sequence_length: int,
    seed: int,
) -> np.ndarray:
    if sequence_length > observations:
        raise ValueError("sequence length exceeds the replay")
    if updates <= 0 or batch_size <= 0:
        raise ValueError("updates and batch size must be positive")
    rng = np.random.default_rng(int(seed))
    return rng.integers(
        0,
        observations - sequence_length + 1,
        size=(updates, batch_size),
        dtype=np.int64,
    )


def batch_from_starts(
    dataset: Mapping[str, np.ndarray], starts: np.ndarray, sequence_length: int, burn_in: int
) -> dict[str, Any]:
    import jax.numpy as jnp

    offsets = np.arange(sequence_length, dtype=np.int64)[None, :]
    indices = np.asarray(starts, np.int64)[:, None] + offsets
    observations = np.asarray(dataset["observations"])[indices]
    actions = np.asarray(dataset["actions"])[indices].copy()
    is_first = np.asarray(dataset["is_first"])[indices].copy()
    # Every sampled chunk is a new recurrent context. Its first stored action
    # came from outside that context and is deliberately erased.
    actions[:, 0] = 0.0
    is_first[:, 0] = True
    loss_mask = np.ones(indices.shape, np.float32)
    loss_mask[:, :burn_in] = 0.0
    return {
        "observations": preprocess_pixels_jax(observations),
        "actions": jnp.asarray(actions, jnp.float32),
        "rewards": jnp.asarray(np.asarray(dataset["rewards"])[indices], jnp.float32),
        "continuations": jnp.asarray(
            np.asarray(dataset["continuations"])[indices], jnp.float32
        ),
        "is_first": jnp.asarray(is_first, jnp.bool_),
        "loss_mask": jnp.asarray(loss_mask, jnp.float32),
    }


def _tree_sha256(tree: Any) -> str:
    import jax

    digest = hashlib.sha256()
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        digest.update(str(path).encode("utf-8"))
        value = np.asarray(leaf)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(canonical_bytes(list(value.shape)))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _validate_checkpoint_state(state: Any, *, updates: int, arm: str) -> None:
    """Validate the complete trained world-model and Adam state numerically.

    Prediction replay alone cannot see NaNs in heads that are not used by the
    visual rollout.  Likewise, coercing an optimizer step through ``int`` would
    accept fractional floating-point counters.  Check the exact scalar integer
    counter, the optimizer/tree correspondence, and every parameter and moment
    leaf before a checkpoint is saved or trusted.
    """

    import jax

    if arm not in ARM_ORDER:
        raise ValueError("checkpoint validation received an unknown objective arm")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
        raise ValueError("checkpoint update allocation must be a positive integer")
    try:
        world_model = state.params.world_model
        optimizer = state.model_optimizer
    except AttributeError as error:
        raise ValueError(f"{arm} checkpoint state has the wrong structure") from error

    step = np.asarray(optimizer.step)
    if (
        step.shape != ()
        or not np.issubdtype(step.dtype, np.integer)
        or np.issubdtype(step.dtype, np.bool_)
        or step.item() != updates
    ):
        raise ValueError(
            f"{arm} checkpoint optimizer step must be the exact scalar integer allocation"
        )

    parameter_structure = jax.tree_util.tree_structure(world_model)
    first_structure = jax.tree_util.tree_structure(optimizer.first_moment)
    second_structure = jax.tree_util.tree_structure(optimizer.second_moment)
    if first_structure != parameter_structure or second_structure != parameter_structure:
        raise ValueError(f"{arm} checkpoint optimizer trees do not match world-model parameters")

    parameter_leaves = jax.tree_util.tree_leaves(world_model)
    first_leaves = jax.tree_util.tree_leaves(optimizer.first_moment)
    second_leaves = jax.tree_util.tree_leaves(optimizer.second_moment)
    if not parameter_leaves:
        raise ValueError(f"{arm} checkpoint world model has no parameter leaves")
    for index, (parameter, first, second) in enumerate(
        zip(parameter_leaves, first_leaves, second_leaves, strict=True)
    ):
        parameter_array = np.asarray(parameter)
        first_array = np.asarray(first)
        second_array = np.asarray(second)
        if (
            not np.issubdtype(parameter_array.dtype, np.floating)
            or not np.issubdtype(first_array.dtype, np.floating)
            or not np.issubdtype(second_array.dtype, np.floating)
            or first_array.shape != parameter_array.shape
            or second_array.shape != parameter_array.shape
        ):
            raise ValueError(
                f"{arm} checkpoint parameter/optimizer leaf {index} has an invalid dtype or shape"
            )
        if (
            not np.isfinite(parameter_array).all()
            or not np.isfinite(first_array).all()
            or not np.isfinite(second_array).all()
        ):
            raise ValueError(
                f"{arm} checkpoint parameter/optimizer leaf {index} is non-finite"
            )


def _compiler_evidence(compiled: Any, lowered: Any, name: str) -> dict[str, Any]:
    analysis = compiled.cost_analysis()
    if analysis is None:
        raise RuntimeError(f"compiler cost analysis unavailable for {name}")
    rows = analysis if isinstance(analysis, (list, tuple)) else [analysis]
    normalized = [
        {str(key): float(value) for key, value in sorted(row.items())}
        for row in rows
    ]
    if not all(
        math.isfinite(value) for row in normalized for value in row.values()
    ):
        raise FloatingPointError(f"non-finite compiler cost evidence for {name}")
    flops = float(sum(row.get("flops", 0.0) for row in normalized))
    if not math.isfinite(flops) or flops <= 0.0:
        raise RuntimeError(f"compiler reported no positive FLOPs for {name}")
    hlo = str(lowered.compiler_ir(dialect="stablehlo"))
    return {
        "flops": flops,
        "cost_analysis": normalized,
        "cost_analysis_sha256": object_sha256(normalized),
        "stablehlo_sha256": hashlib.sha256(hlo.encode("utf-8")).hexdigest(),
    }


def _parameter_counts(params: Any, config: Any) -> dict[str, int]:
    from imf_dreamer_jax import world_model_parameter_counts

    counts = world_model_parameter_counts(params, config)
    return {
        name: int(getattr(counts, name))
        for name in ("total", "active", "prior_total", "prior_active")
    }


def _dummy_batch(config: Any, profile_cfg: Mapping[str, Any]) -> dict[str, Any]:
    zeros = np.zeros(
        (
            int(profile_cfg["batch_size"]),
            int(profile_cfg["sequence_length"]),
            *config.observation_shape,
        ),
        dtype=np.uint8,
    )
    dataset = {
        "observations": zeros.reshape((-1, *config.observation_shape)),
        "actions": np.zeros(
            (zeros.shape[0] * zeros.shape[1], config.action_dim), np.float32
        ),
        "rewards": np.zeros((zeros.shape[0] * zeros.shape[1],), np.float32),
        "continuations": np.ones((zeros.shape[0] * zeros.shape[1],), np.float32),
        "is_first": np.zeros((zeros.shape[0] * zeros.shape[1],), np.bool_),
    }
    starts = np.arange(0, zeros.shape[0] * zeros.shape[1], zeros.shape[1])
    return batch_from_starts(
        dataset, starts, zeros.shape[1], int(profile_cfg["burn_in"])
    )


def compile_training(
    state: Any, config: Any, profile_cfg: Mapping[str, Any], key: Any
) -> tuple[Any, dict[str, Any]]:
    import jax
    from imf_dreamer_jax import train_world_model

    batch = _dummy_batch(config, profile_cfg)
    function = jax.jit(lambda current, value, rng: train_world_model(current, value, rng, config))
    lowered = function.lower(state, batch, key)
    compiled = lowered.compile()
    return compiled, _compiler_evidence(compiled, lowered, "world_model_train")


def _inference_transition(params: Any, state: Any, action: Any, noise: Any, config: Any):
    from imf_dreamer_jax import (
        RSSMState,
        decode,
        sample_prior_with_nfe,
        transition_deterministic,
    )

    deterministic = transition_deterministic(params, state, action, config)
    sample = sample_prior_with_nfe(
        params, deterministic, None, config, noise=noise
    )
    next_state = RSSMState(deterministic, sample.stochastic)
    return next_state, decode(params, next_state.feature, config), sample.nfe


def compile_inference(params: Any, config: Any) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import initial_state

    state = initial_state(config, 1)
    action = jnp.zeros((1, config.action_dim), jnp.float32)
    noise = jnp.zeros((1, config.stochastic_dim), jnp.float32)
    function = jax.jit(
        lambda model, current, act, eps: _inference_transition(
            model, current, act, eps, config
        )
    )
    lowered = function.lower(params, state, action, noise)
    compiled = lowered.compile()
    evidence = _compiler_evidence(compiled, lowered, "world_model_inference")
    _, _, nfe = compiled(params, state, action, noise)
    evidence["field_evaluations"] = int(np.asarray(nfe))
    return evidence


def compile_prior_field(params: Any, config: Any) -> dict[str, Any]:
    """Compile exactly one objective-specific prior-network evaluation.

    XLA cost analysis often reports a ``fori_loop`` body once rather than
    multiplying by its static trip count.  Keeping this separate evidence lets
    the result report an explicit structural NFE adjustment without altering
    the executable sampler used for predictions.
    """

    import jax
    import jax.numpy as jnp

    sample = jnp.zeros((1, config.stochastic_dim), jnp.float32)
    condition = jnp.zeros((1, config.deterministic_dim), jnp.float32)
    first_time = jnp.ones((1, 1), jnp.float32)
    second_time = jnp.zeros((1, 1), jnp.float32)
    if config.prior == "shortcut":
        from imf_dreamer_jax.nn import mlp

        function = jax.jit(
            lambda prior, state, context, tau, step: mlp(
                prior,
                jnp.concatenate((state, context, tau, step), axis=-1),
                activation=jax.nn.silu,
            )
        )
        arguments = (params["prior"], sample, condition, second_time, first_time)
    else:
        from imf_dreamer_jax import imf_outputs

        function = jax.jit(
            lambda prior, state, context, r, t: imf_outputs(
                prior, state, context, r, t
            )
        )
        arguments = (params["prior"], sample, condition, second_time, first_time)
    lowered = function.lower(*arguments)
    compiled = lowered.compile()
    return _compiler_evidence(compiled, lowered, "prior_field_evaluation")


def runtime_fingerprint() -> dict[str, Any]:
    import jax
    import jaxlib

    devices = jax.devices()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "device_count": len(devices),
        "visible_device_count": len(devices),
        "device_platforms": sorted({str(device.platform) for device in devices}),
        "device_kinds": sorted({str(getattr(device, "device_kind", "unknown")) for device in devices}),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "jax_enable_x64_environment": os.environ.get("JAX_ENABLE_X64"),
        "jax_platform_name": os.environ.get("JAX_PLATFORM_NAME"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "numpy": np.__version__,
        "dm_control": importlib.metadata.version("dm-control"),
        "mujoco": importlib.metadata.version("mujoco"),
    }


def _one_cuda_selector(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value == value.strip()
        and "," not in value
        and value not in {"-1", "NoDevFiles"}
    )


def _validate_runtime_fingerprint(
    runtime: Mapping[str, Any], protocol: Mapping[str, Any], profile: str
) -> None:
    _require_exact_keys(runtime, RUNTIME_KEYS, "pixel runtime fingerprint")
    if runtime["device_count"] != runtime["visible_device_count"]:
        raise ValueError("pixel runtime device-count aliases disagree")
    if (
        type(runtime["visible_device_count"]) is not int
        or runtime["visible_device_count"] <= 0
        or not isinstance(runtime["device_platforms"], list)
        or not runtime["device_platforms"]
        or runtime["device_platforms"] != sorted(set(runtime["device_platforms"]))
        or not isinstance(runtime["device_kinds"], list)
        or not runtime["device_kinds"]
        or runtime["device_kinds"] != sorted(set(runtime["device_kinds"]))
    ):
        raise ValueError("pixel runtime device inventory is malformed")
    if runtime["jax_enable_x64"] is not protocol["runtime_contract"]["jax_enable_x64"]:
        raise ValueError("pixel runtime violated the frozen JAX precision mode")
    for key in ("python", "platform", "jax", "jaxlib", "backend", "numpy", "dm_control", "mujoco"):
        if not isinstance(runtime[key], str) or not runtime[key]:
            raise ValueError(f"pixel runtime field {key!r} is malformed")
    if profile == "smoke":
        if runtime["backend"] not in protocol["runtime_contract"]["smoke"][
            "allowed_backends"
        ]:
            raise ValueError("smoke used an unregistered JAX backend")
        return
    if profile != "hard":
        raise ValueError("runtime validation received an unknown profile")
    hard = protocol["runtime_contract"]["hard"]
    versions = {
        key: runtime[key]
        for key in ("python", "jax", "jaxlib", "numpy", "dm_control", "mujoco")
    }
    environment = {
        "JAX_ENABLE_X64": runtime["jax_enable_x64_environment"],
        "JAX_PLATFORM_NAME": runtime["jax_platform_name"],
        "MUJOCO_GL": runtime["mujoco_gl"],
    }
    if (
        runtime["backend"] != hard["backend"]
        or runtime["visible_device_count"] != hard["visible_device_count"]
        or runtime["device_platforms"] != hard["device_platforms"]
        or versions != hard["versions"]
        or environment != hard["environment"]
        or not _one_cuda_selector(runtime["cuda_visible_devices"])
        or len(runtime["device_kinds"]) != 1
        or hard["device_kind_substring"] not in runtime["device_kinds"][0]
    ):
        raise ValueError("hard pixel artifact runtime is outside the registered L40S lock")


def _loss_dict(loss: Any) -> dict[str, float]:
    names = getattr(loss, "_fields", None)
    if names is None:
        names = tuple(loss.__dataclass_fields__)
    return {
        name: float(np.asarray(getattr(loss, name)))
        for name in names
    }


def train_arm(
    *,
    arm: str,
    state: Any,
    config: Any,
    executable: Any,
    train_dataset: Mapping[str, np.ndarray],
    schedule: np.ndarray,
    keys: np.ndarray,
    profile_cfg: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    import jax
    import jax.numpy as jnp

    started = time.perf_counter()
    last_loss: dict[str, float] | None = None
    for update, starts in enumerate(schedule):
        batch = batch_from_starts(
            train_dataset,
            starts,
            int(profile_cfg["sequence_length"]),
            int(profile_cfg["burn_in"]),
        )
        state, loss = executable(state, batch, jnp.asarray(keys[update], jnp.uint32))
        jax.block_until_ready(state)
        last_loss = _loss_dict(loss)
        if not all(math.isfinite(value) for value in last_loss.values()):
            raise FloatingPointError(f"non-finite {arm} world-model loss")
    elapsed = time.perf_counter() - started
    if last_loss is None:
        raise AssertionError("training allocation unexpectedly had zero updates")
    _validate_checkpoint_state(state, updates=int(len(schedule)), arm=arm)
    return state, {
        "updates": int(len(schedule)),
        "wall_seconds": float(elapsed),
        "final_loss": last_loss,
        "world_model_parameter_sha256": _tree_sha256(state.params.world_model),
    }


def eligible_window_starts(
    heldout: Mapping[str, np.ndarray], context_length: int, max_horizon: int
) -> np.ndarray:
    episodes = np.asarray(heldout["episode_id"])
    width = int(context_length + max_horizon)
    starts = [
        start
        for start in range(len(episodes) - width + 1)
        if np.all(episodes[start : start + width] == episodes[start])
    ]
    if not starts:
        raise ValueError("held-out replay has no complete within-episode evaluation window")
    return np.asarray(starts, dtype=np.int64)


def select_windows(
    eligible: np.ndarray, windows: int, *, seed: int
) -> np.ndarray:
    if windows > len(eligible):
        raise ValueError(
            f"requested {windows} windows but only {len(eligible)} complete windows exist"
        )
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(eligible, size=windows, replace=False)).astype(np.int64)


def _rollout(params: Any, start: Any, actions: Any, noise: Any, config: Any):
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import RSSMState, decode, sample_prior_with_nfe, transition_deterministic

    draws = noise.shape[0]
    batch = actions.shape[0]
    state = RSSMState(
        jnp.broadcast_to(
            start.deterministic[None], (draws, *start.deterministic.shape)
        ),
        jnp.broadcast_to(start.stochastic[None], (draws, *start.stochastic.shape)),
    )

    def step(current: Any, inputs: tuple[Any, Any]):
        action, epsilon = inputs
        action = jnp.broadcast_to(action[None], (draws, batch, config.action_dim))
        deterministic = transition_deterministic(params, current, action, config)
        flat_deterministic = deterministic.reshape((-1, config.deterministic_dim))
        flat_noise = epsilon.reshape((-1, config.stochastic_dim))
        sample = sample_prior_with_nfe(
            params, flat_deterministic, None, config, noise=flat_noise
        )
        stochastic = sample.stochastic.reshape(
            (draws, batch, config.stochastic_dim)
        )
        next_state = RSSMState(deterministic, stochastic)
        decoded = decode(params, next_state.feature, config)
        return next_state, decoded

    _, decoded = jax.lax.scan(
        step,
        state,
        (jnp.swapaxes(actions, 0, 1), jnp.transpose(noise, (2, 0, 1, 3))),
    )
    # time, draws, batch, H, W, C -> draws, batch, time, H, W, C
    return jnp.transpose(decoded, (1, 2, 0, 3, 4, 5))


def global_ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    x = np.asarray(prediction, np.float64)
    y = np.asarray(target, np.float64)
    if x.shape != y.shape or x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError("SSIM inputs must be equal-shape RGB images")
    mean_x = np.mean(x, axis=(0, 1))
    mean_y = np.mean(y, axis=(0, 1))
    centered_x = x - mean_x
    centered_y = y - mean_y
    variance_x = np.mean(centered_x * centered_x, axis=(0, 1))
    variance_y = np.mean(centered_y * centered_y, axis=(0, 1))
    covariance = np.mean(centered_x * centered_y, axis=(0, 1))
    numerator = (2.0 * mean_x * mean_y + 0.0001) * (
        2.0 * covariance + 0.0009
    )
    denominator = (mean_x * mean_x + mean_y * mean_y + 0.0001) * (
        variance_x + variance_y + 0.0009
    )
    return float(np.mean(numerator / denominator))


def prediction_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    channel_variance: np.ndarray,
    horizons: Sequence[int],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Recompute every aggregate from retained [window,draw,horizon,H,W,RGB]."""

    prediction = np.asarray(predictions, np.float32)
    target = np.asarray(targets, np.float32)
    if prediction.ndim != 6 or target.shape != (
        prediction.shape[0],
        prediction.shape[2],
        prediction.shape[3],
        prediction.shape[4],
        prediction.shape[5],
    ):
        raise ValueError("retained prediction/target tensor shapes are inconsistent")
    if prediction.shape[-1] != 3 or len(horizons) != prediction.shape[2]:
        raise ValueError("prediction tensor does not match RGB horizons")
    variance = np.maximum(np.asarray(channel_variance, np.float64), 0.0004)
    if variance.shape != (3,):
        raise ValueError("training channel variance must have shape [3]")
    errors = prediction.astype(np.float64) - target[:, None].astype(np.float64)
    normalized_mse = np.mean(
        np.square(errors) / variance.reshape((1, 1, 1, 1, 1, 3)),
        axis=(1, 3, 4, 5),
    )
    mean_prediction = np.mean(prediction.astype(np.float64), axis=1)
    clipped = np.clip(mean_prediction, 0.0, 1.0)
    clipped_target = np.clip(target.astype(np.float64), 0.0, 1.0)
    mean_mse = np.mean(np.square(clipped - clipped_target), axis=(2, 3, 4))
    psnr = 10.0 * np.log10(1.0 / np.maximum(mean_mse, 1e-12))
    ssim = np.empty_like(mean_mse)
    for window in range(prediction.shape[0]):
        for horizon_index in range(prediction.shape[2]):
            ssim[window, horizon_index] = global_ssim(
                clipped[window, horizon_index], clipped_target[window, horizon_index]
            )
    flat_prediction = prediction.reshape(
        prediction.shape[0], prediction.shape[1], prediction.shape[2], -1
    ).astype(np.float64)
    flat_target = target.reshape(target.shape[0], target.shape[1], -1).astype(np.float64)
    dimension = float(flat_target.shape[-1])
    distance_to_target = np.sqrt(
        np.mean(np.square(flat_prediction - flat_target[:, None]), axis=-1)
    )
    energy = np.mean(distance_to_target, axis=1)
    if prediction.shape[1] >= 2:
        pair_terms = []
        for left in range(prediction.shape[1]):
            for right in range(prediction.shape[1]):
                if left != right:
                    pair_terms.append(
                        np.sqrt(
                            np.sum(
                                np.square(
                                    flat_prediction[:, left] - flat_prediction[:, right]
                                ),
                                axis=-1,
                            )
                            / dimension
                        )
                    )
        energy -= 0.5 * np.mean(np.stack(pair_terms, axis=0), axis=0)
    temporal = np.zeros_like(mean_mse)
    temporal[:, 0] = 0.0
    if prediction.shape[2] > 1:
        predicted_changes = mean_prediction[:, 1:] - mean_prediction[:, :-1]
        target_changes = target[:, 1:].astype(np.float64) - target[:, :-1].astype(np.float64)
        temporal[:, 1:] = np.mean(
            np.square(predicted_changes - target_changes), axis=(2, 3, 4)
        )
    horizon_values = np.asarray(horizons, np.float64)
    horizon_normalized_mse = np.mean(normalized_mse, axis=0)
    auc = float(
        np.trapezoid(horizon_normalized_mse, horizon_values)
        / (horizon_values[-1] - horizon_values[0])
        if len(horizons) > 1
        else horizon_normalized_mse[0]
    )
    raw = {
        "normalized_visual_mse": normalized_mse.astype(np.float32),
        "predictive_mean_psnr_db": psnr.astype(np.float32),
        "predictive_mean_global_ssim": ssim.astype(np.float32),
        "energy_score": energy.astype(np.float32),
        "temporal_difference_mse": temporal.astype(np.float32),
    }
    summary = {
        "horizons": [int(value) for value in horizons],
        "normalized_visual_mse": [float(value) for value in horizon_normalized_mse],
        "predictive_mean_psnr_db": [float(value) for value in np.mean(psnr, axis=0)],
        "predictive_mean_global_ssim": [float(value) for value in np.mean(ssim, axis=0)],
        "energy_score": [float(value) for value in np.mean(energy, axis=0)],
        "temporal_difference_mse": [float(value) for value in np.mean(temporal, axis=0)],
        "normalized_visual_mse_auc": auc,
    }
    return summary, raw


def evaluate_models(
    *,
    protocol: Mapping[str, Any],
    profile: str,
    task: str,
    seed: int,
    heldout: Mapping[str, np.ndarray],
    states: Mapping[str, Any],
    action_dim: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import observe_sequence

    profile_cfg = protocol["profiles"][profile]
    horizons = [int(value) for value in profile_cfg["horizons"]]
    context_length = int(profile_cfg["context_length"])
    max_horizon = max(horizons)
    windows = int(profile_cfg["windows"])
    draws = int(profile_cfg["predictive_draws"])
    eligible = eligible_window_starts(heldout, context_length, max_horizon)
    starts = select_windows(
        eligible,
        windows,
        seed=derive_numpy_seed("evaluation-window", task, seed, profile),
    )
    context_observations = np.stack(
        [heldout["observations"][start : start + context_length] for start in starts]
    ).astype(np.uint8)
    context_actions = np.stack(
        [heldout["actions"][start : start + context_length] for start in starts]
    ).astype(np.float32)
    context_first = np.stack(
        [heldout["is_first"][start : start + context_length] for start in starts]
    ).astype(np.bool_)
    context_actions[:, 0] = 0.0
    context_first[:, 0] = True
    future_actions = np.stack(
        [
            heldout["actions"][
                start + context_length : start + context_length + max_horizon
            ]
            for start in starts
        ]
    ).astype(np.float32)
    target_stacks = np.stack(
        [
            heldout["observations"][
                start + context_length : start + context_length + max_horizon
            ]
            for start in starts
        ]
    ).astype(np.uint8)
    target_indices = np.asarray([value - 1 for value in horizons], np.int64)
    targets = (
        target_stacks[:, target_indices, :, :, -3:].astype(np.float32) / 255.0
    )
    posterior_keys = np.stack(
        [
            np.asarray(
                derive_jax_key("posterior-context", task, seed, int(start)),
                dtype=np.uint32,
            )
            for start in starts
        ]
    )
    stochastic_dim = int(
        resolved_config(protocol, profile, ARM_ORDER[0], action_dim=action_dim).stochastic_dim
    )
    noise_rng = np.random.default_rng(
        derive_numpy_seed("evaluation-noise", task, seed, profile)
    )
    base_noise = noise_rng.normal(
        size=(windows, draws, max_horizon, stochastic_dim)
    ).astype(np.float32)
    all_predictions = np.empty(
        (
            len(ARM_ORDER),
            len(protocol["evaluation"]["nfe_frontier"]),
            windows,
            draws,
            len(horizons),
            32,
            32,
            3,
        ),
        np.float32,
    )
    for arm_index, arm in enumerate(ARM_ORDER):
        for nfe_index, nfe in enumerate(protocol["evaluation"]["nfe_frontier"]):
            config = resolved_config(
                protocol, profile, arm, action_dim=action_dim, nfe=int(nfe)
            )
            rollout = jax.jit(
                lambda params, start_state, actions, noise: _rollout(
                    params, start_state, actions, noise, config
                )
            )
            for window in range(windows):
                observations = preprocess_pixels_jax(
                    context_observations[window : window + 1]
                )
                actions = jnp.asarray(context_actions[window : window + 1], jnp.float32)
                first = jnp.asarray(context_first[window : window + 1], jnp.bool_)
                sequence = observe_sequence(
                    states[arm].params.world_model,
                    observations,
                    actions,
                    jnp.asarray(posterior_keys[window], jnp.uint32),
                    config,
                    is_first=first,
                )
                start_state = type(sequence.states)(
                    sequence.states.deterministic[:, -1],
                    sequence.states.stochastic[:, -1],
                )
                decoded = rollout(
                    states[arm].params.world_model,
                    start_state,
                    jnp.asarray(future_actions[window : window + 1], jnp.float32),
                    jnp.asarray(base_noise[window, :, None], jnp.float32),
                )
                jax.block_until_ready(decoded)
                value = np.asarray(decoded)[:, 0, target_indices, :, :, -3:]
                all_predictions[arm_index, nfe_index, window] = value.astype(
                    np.float32
                )
    retained = {
        "arm_names": np.asarray(ARM_ORDER, dtype="<U32"),
        "nfe_values": np.asarray(protocol["evaluation"]["nfe_frontier"], np.int16),
        "horizons": np.asarray(horizons, np.int16),
        "window_starts": starts,
        "context_observations": context_observations,
        "context_actions": context_actions,
        "context_is_first": context_first,
        "future_actions": future_actions,
        "posterior_keys": posterior_keys,
        "base_noise": base_noise,
        "targets": targets.astype(np.float32),
        "predictions": all_predictions,
    }
    return retained, {
        "eligible_window_count": int(len(eligible)),
        "selected_window_count": windows,
        "paired_windows": True,
        "paired_actions": True,
        "paired_posterior_keys": True,
        "paired_base_noise": True,
    }


def summarize_predictions(
    retained: Mapping[str, np.ndarray], channel_variance: np.ndarray
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arms = [str(value) for value in retained["arm_names"].tolist()]
    nfes = [int(value) for value in retained["nfe_values"].tolist()]
    horizons = [int(value) for value in retained["horizons"].tolist()]
    predictions = np.asarray(retained["predictions"], np.float32)
    targets = np.asarray(retained["targets"], np.float32)
    if arms != list(ARM_ORDER) or nfes != [1, 2, 4]:
        raise ValueError("retained arm/NFE axes are not canonical")
    summary: dict[str, Any] = {}
    raw: dict[str, np.ndarray] = {}
    for arm_index, arm in enumerate(arms):
        summary[arm] = {}
        for nfe_index, nfe in enumerate(nfes):
            row, metrics = prediction_metrics(
                predictions[arm_index, nfe_index], targets, channel_variance, horizons
            )
            summary[arm][str(nfe)] = row
            for name, value in metrics.items():
                raw[f"metric__{arm}__nfe_{nfe}__{name}"] = value
    primary = {
        arm: summary[arm][str(PRIMARY)]
        for arm, PRIMARY in (
            ("shortcut_forcing", 4),
            ("trajectory_imf", 1),
        )
    }
    return {"frontier": summary, "primary_nfe": primary}, raw


def _shared_trunk(params: Any) -> dict[str, Any]:
    return {
        name: params.world_model[name]
        for name in (
            "encoder",
            "decoder",
            "recurrence",
            "posterior",
            "reward",
            "continuation",
        )
    }


def _training_keys(task: str, seed: int, updates: int) -> np.ndarray:
    import jax

    base = derive_jax_key("world-objective", task, seed)
    return np.stack(
        [np.asarray(jax.random.fold_in(base, update), np.uint32) for update in range(updates)]
    )


def _source_paths() -> list[Path]:
    library_root = WORKSPACE_ROOT / "imf_dreamer_jax" / "src" / "imf_dreamer_jax"
    paths = [
        PROTOCOL_PATH,
        BASE_PROTOCOL_PATH,
        PROJECT_ROOT / "dreamer_imf_compare" / "pixel_benchmark.py",
        PROJECT_ROOT / "dreamer_imf_compare" / "dmc.py",
        PROJECT_ROOT / "scripts" / "run_pixel_benchmark.py",
        PROJECT_ROOT / "scripts" / "verify_pixel_benchmark.py",
        PROJECT_ROOT / "tests" / "__init__.py",
        PROJECT_ROOT / "tests" / "test_pixel_benchmark.py",
        PROJECT_ROOT / "PIXEL_BENCHMARK.md",
    ]
    paths.extend(library_root / name for name in LIBRARY_SOURCE_BASENAMES)
    checked: list[Path] = []
    for path in paths:
        lexical = _reject_symlink_chain(path, label="pixel benchmark source")
        try:
            lexical.relative_to(WORKSPACE_ROOT)
        except ValueError as error:
            raise ValueError("pixel benchmark source escapes the workspace") from error
        if not lexical.is_file():
            raise FileNotFoundError(
                f"pixel benchmark executable source is incomplete: {lexical}"
            )
        checked.append(lexical)
    return checked


def source_manifest() -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(WORKSPACE_ROOT)),
            "bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
        for path in _source_paths()
    ]


def _artifact_manifest(root: Path) -> dict[str, Any]:
    files = [
        {
            "path": name,
            "bytes": int((root / name).stat().st_size),
            "sha256": file_sha256(root / name),
        }
        for name in ARTIFACT_FILES
    ]
    body = {
        "schema_version": "trajectory-imf-pixel-artifact-manifest-v1",
        "artifact_files": files,
        "source_files": source_manifest(),
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "base_protocol_sha256": file_sha256(BASE_PROTOCOL_PATH),
    }
    return {**body, "body_sha256": object_sha256(body)}


def _validate_source_manifest(records: Sequence[Mapping[str, Any]]) -> None:
    expected = source_manifest()
    if list(records) != expected:
        raise ValueError("pixel artifact executable source identity changed")


def _same_json_number(left: Any, right: Any, *, atol: float = 1e-6) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _same_json_number(left[key], right[key], atol=atol) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_json_number(a, b, atol=atol) for a, b in zip(left, right)
        )
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=atol)
    return left == right


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a JSON number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return number


def _validate_compiler_evidence(
    value: Mapping[str, Any], *, label: str, inference: bool = False
) -> None:
    expected = set(COMPILER_EVIDENCE_KEYS)
    if inference:
        expected |= {"field_evaluations", "nfe_adjusted_flops"}
    _require_exact_keys(value, expected, label)
    _finite_number(value["flops"], f"{label}.flops", positive=True)
    if inference:
        if type(value["field_evaluations"]) is not int or value["field_evaluations"] <= 0:
            raise ValueError(f"{label}.field_evaluations must be a positive integer")
        _finite_number(
            value["nfe_adjusted_flops"],
            f"{label}.nfe_adjusted_flops",
            positive=True,
        )
    analyses = value["cost_analysis"]
    if not isinstance(analyses, list) or not analyses:
        raise ValueError(f"{label}.cost_analysis must be a nonempty list")
    for row in analyses:
        if not isinstance(row, Mapping) or not row:
            raise ValueError(f"{label}.cost_analysis rows must be nonempty objects")
        for key, item in row.items():
            if not isinstance(key, str):
                raise ValueError(f"{label}.cost_analysis keys must be strings")
            _finite_number(item, f"{label}.cost_analysis[{key!r}]")
    if value["cost_analysis_sha256"] != object_sha256(analyses):
        raise ValueError(f"{label}.cost_analysis digest is invalid")
    _require_sha256(value["cost_analysis_sha256"], f"{label}.cost_analysis_sha256")
    _require_sha256(value["stablehlo_sha256"], f"{label}.stablehlo_sha256")


def _validate_metric_row(
    value: Mapping[str, Any], horizons: Sequence[int], label: str
) -> None:
    _require_exact_keys(value, METRIC_KEYS, label)
    if value["horizons"] != list(horizons):
        raise ValueError(f"{label}.horizons differ from the registered profile")
    for key in METRIC_KEYS - {"horizons", "normalized_visual_mse_auc"}:
        sequence = value[key]
        if not isinstance(sequence, list) or len(sequence) != len(horizons):
            raise ValueError(f"{label}.{key} has the wrong length")
        for index, item in enumerate(sequence):
            _finite_number(item, f"{label}.{key}[{index}]")
    _finite_number(value["normalized_visual_mse_auc"], f"{label}.normalized_visual_mse_auc")


def _validate_result_schema(
    result: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    """Validate every controlled result field before consuming any of its values."""

    _require_exact_keys(result, RESULT_KEYS, "pixel result")
    if (
        result["schema_version"] != "trajectory-imf-pixel-result-v1"
        or result["status"] != "complete"
        or result["study_role"] != "secondary_hard_visual_evaluation"
        or result["claim_eligible_for_primary_gate"] is not False
        or result["primary_gate_substitution"] is not False
    ):
        raise ValueError("pixel result status or claim boundary is invalid")
    profile = result["profile"]
    if not isinstance(profile, str) or profile not in protocol["profiles"]:
        raise ValueError("pixel result profile is invalid")
    profile_cfg = protocol["profiles"][profile]
    if (
        not isinstance(result["task"], str)
        or type(result["seed"]) is not int
        or not isinstance(result["track"], str)
        or type(result["action_dim"]) is not int
        or result["action_dim"] <= 0
        or type(result["action_repeat"]) is not int
        or result["action_repeat"] <= 0
        or not isinstance(result["renderer_backend"], str)
    ):
        raise ValueError("pixel result scalar identity fields are malformed")
    for key in ("protocol_sha256", "base_protocol_sha256", "shared_initial_trunk_sha256"):
        _require_sha256(result[key], f"pixel result {key}")
    _require_exact_keys(result["environment_seeds"], {"training", "heldout"}, "environment seeds")
    for split in ("training", "heldout"):
        value = result["environment_seeds"][split]
        if type(value) is not int or not (0 <= value < 2**32 - 1):
            raise ValueError("pixel environment seed is outside uint32 derivation range")
    _require_exact_keys(result["allocations"], set(ARM_ORDER), "pixel allocations")
    for arm in ARM_ORDER:
        if type(result["allocations"][arm]) is not int or result["allocations"][arm] <= 0:
            raise ValueError("pixel update allocation must be a positive integer")
    _validate_runtime_fingerprint(result["runtime"], protocol, profile)
    _require_exact_keys(result["training"], set(ARM_ORDER), "pixel training arms")
    _require_exact_keys(result["compute"], set(ARM_ORDER), "pixel compute arms")
    for arm in ARM_ORDER:
        training = result["training"][arm]
        _require_exact_keys(
            training,
            {
                "updates",
                "wall_seconds",
                "final_loss",
                "world_model_parameter_sha256",
                "compiler_flops_per_update",
                "realized_compiler_flops",
            },
            f"pixel training {arm}",
        )
        if type(training["updates"]) is not int or training["updates"] <= 0:
            raise ValueError("pixel training updates must be a positive integer")
        _finite_number(training["wall_seconds"], f"pixel training {arm}.wall_seconds", positive=True)
        _finite_number(
            training["compiler_flops_per_update"],
            f"pixel training {arm}.compiler_flops_per_update",
            positive=True,
        )
        _finite_number(
            training["realized_compiler_flops"],
            f"pixel training {arm}.realized_compiler_flops",
            positive=True,
        )
        _require_sha256(
            training["world_model_parameter_sha256"],
            f"pixel training {arm}.world_model_parameter_sha256",
        )
        _require_exact_keys(training["final_loss"], LOSS_KEYS, f"pixel training {arm}.final_loss")
        for name, value in training["final_loss"].items():
            _finite_number(value, f"pixel training {arm}.final_loss.{name}")

        compute = result["compute"][arm]
        _require_exact_keys(
            compute,
            {
                "resolved_config",
                "resolved_config_sha256",
                "parameters",
                "training",
                "prior_field",
                "inference",
                "allocated_updates",
                "realized_compiler_train_flops",
            },
            f"pixel compute {arm}",
        )
        if not isinstance(compute["resolved_config"], Mapping):
            raise ValueError("pixel resolved config must be an object")
        _require_sha256(compute["resolved_config_sha256"], "pixel resolved config digest")
        _require_exact_keys(
            compute["parameters"],
            {"total", "active", "prior_total", "prior_active"},
            f"pixel parameters {arm}",
        )
        for name, value in compute["parameters"].items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"pixel parameter count {arm}.{name} must be positive")
        _validate_compiler_evidence(compute["training"], label=f"pixel compute {arm}.training")
        _validate_compiler_evidence(compute["prior_field"], label=f"pixel compute {arm}.prior_field")
        _require_exact_keys(compute["inference"], {"1", "2", "4"}, f"pixel inference {arm}")
        for nfe in (1, 2, 4):
            _validate_compiler_evidence(
                compute["inference"][str(nfe)],
                label=f"pixel compute {arm}.inference.{nfe}",
                inference=True,
            )
        if type(compute["allocated_updates"]) is not int or compute["allocated_updates"] <= 0:
            raise ValueError("pixel compute allocated updates must be positive")
        _finite_number(
            compute["realized_compiler_train_flops"],
            f"pixel compute {arm}.realized_compiler_train_flops",
            positive=True,
        )

    _require_exact_keys(
        result["evaluation_identity"],
        {
            "eligible_window_count",
            "selected_window_count",
            "paired_windows",
            "paired_actions",
            "paired_posterior_keys",
            "paired_base_noise",
        },
        "pixel evaluation identity",
    )
    identity = result["evaluation_identity"]
    if (
        type(identity["eligible_window_count"]) is not int
        or identity["eligible_window_count"] <= 0
        or type(identity["selected_window_count"]) is not int
        or identity["selected_window_count"] <= 0
        or any(
            identity[key] is not True
            for key in (
                "paired_windows",
                "paired_actions",
                "paired_posterior_keys",
                "paired_base_noise",
            )
        )
    ):
        raise ValueError("pixel evaluation identity is malformed or unpaired")
    _require_exact_keys(result["metrics"], {"frontier", "primary_nfe"}, "pixel metrics")
    _require_exact_keys(result["metrics"]["frontier"], set(ARM_ORDER), "pixel metric arms")
    _require_exact_keys(result["metrics"]["primary_nfe"], set(ARM_ORDER), "pixel primary metrics")
    horizons = [int(value) for value in profile_cfg["horizons"]]
    for arm in ARM_ORDER:
        _require_exact_keys(
            result["metrics"]["frontier"][arm], {"1", "2", "4"}, f"pixel metric frontier {arm}"
        )
        for nfe in (1, 2, 4):
            _validate_metric_row(
                result["metrics"]["frontier"][arm][str(nfe)],
                horizons,
                f"pixel metrics {arm}.{nfe}",
            )
        _validate_metric_row(
            result["metrics"]["primary_nfe"][arm],
            horizons,
            f"pixel primary metrics {arm}",
        )
        primary_nfe = int(protocol["evaluation"]["primary_nfe"][arm])
        if result["metrics"]["primary_nfe"][arm] != result["metrics"]["frontier"][arm][str(primary_nfe)]:
            raise ValueError("pixel primary metric row is not its registered frontier row")
    expected_digest_names = set(ARTIFACT_FILES) - {"result.json"}
    _require_exact_keys(result["artifact_digests"], expected_digest_names, "pixel artifact digests")
    for name, digest in result["artifact_digests"].items():
        _require_sha256(digest, f"pixel artifact digest {name}")


def run_benchmark(
    output: str | Path,
    *,
    profile: str,
    task: str,
    seed: int,
    track: str,
) -> dict[str, Any]:
    """Run both paired objective arms into an immutable authenticated directory."""

    from imf_dreamer_jax import create_agent, save_checkpoint

    protocol = load_protocol()
    if profile not in protocol["profiles"]:
        raise ValueError(f"unknown pixel profile {profile!r}")
    profile_cfg = protocol["profiles"][profile]
    if task not in profile_cfg["tasks"] or int(seed) not in profile_cfg["seeds"]:
        raise ValueError("task/seed is outside the frozen profile")
    if track not in profile_cfg["tracks"]:
        raise ValueError("compute track is outside the frozen profile")
    try:
        _validate_runtime_fingerprint(runtime_fingerprint(), protocol, profile)
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    lexical_destination = _reject_symlink_chain(
        output, label="pixel artifact output root"
    )
    destination = lexical_destination.resolve()
    if destination.exists():
        existing = verify_artifact(
            destination, recompute_predictions=True, rederive_compute=False
        )
        requested = (profile, task, int(seed), track)
        observed = (
            existing["profile"],
            existing["task"],
            int(existing["seed"]),
            existing["track"],
        )
        if observed != requested:
            raise ValueError(
                "existing sealed pixel artifact has a different profile/task/seed/track identity"
            )
        return existing
        destination.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink_chain(destination.parent, label="pixel artifact output parent")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        _atomic_json(staging / "protocol.json", protocol)
        exploration = protocol["environment"]["exploration_policy"]
        environment_seeds = {
            "training": derive_environment_seed("environment-train", task, int(seed)),
            "heldout": derive_environment_seed("environment-heldout", task, int(seed)),
        }
        if environment_seeds["training"] == environment_seeds["heldout"]:
            raise RuntimeError("domain-separated training and held-out DMC seeds collided")
        train_adapter = PixelDMCAdapter(
            **_adapter_kwargs(protocol, task, environment_seeds["training"], profile)
        )
        try:
            train_dataset = collect_pixel_dataset(
                train_adapter,
                int(profile_cfg["training_observations"]),
                exploration_seed=derive_numpy_seed("exploration-train", task, seed),
                ar_coefficient=float(exploration["coefficient"]),
                innovation_scale=float(exploration["innovation_scale"]),
            )
            action_dim = train_adapter.action_dim
        finally:
            train_adapter.close()
        heldout_adapter = PixelDMCAdapter(
            **_adapter_kwargs(protocol, task, environment_seeds["heldout"], profile)
        )
        try:
            heldout_dataset = collect_pixel_dataset(
                heldout_adapter,
                int(profile_cfg["heldout_observations"]),
                exploration_seed=derive_numpy_seed("exploration-heldout", task, seed),
                ar_coefficient=float(exploration["coefficient"]),
                innovation_scale=float(exploration["innovation_scale"]),
            )
            if heldout_adapter.action_dim != action_dim:
                raise RuntimeError("training and held-out action dimensions differ")
        finally:
            heldout_adapter.close()
        _atomic_npz(
            staging / "dataset.npz",
            dataset_archive_arrays(train_dataset, heldout_dataset),
        )

        initialization_key = derive_jax_key("world-init", task, seed)
        initial_states: dict[str, Any] = {}
        configs: dict[str, Any] = {}
        executables: dict[str, Any] = {}
        compute: dict[str, Any] = {}
        for arm in ARM_ORDER:
            config = resolved_config(protocol, profile, arm, action_dim=action_dim)
            state = create_agent(config, initialization_key)
            executable, compiler = compile_training(
                state,
                config,
                profile_cfg,
                derive_jax_key("compile", task, seed, arm),
            )
            configs[arm] = config
            initial_states[arm] = state
            executables[arm] = executable
            compute[arm] = {
                "resolved_config": asdict(config),
                "resolved_config_sha256": object_sha256(asdict(config)),
                "parameters": _parameter_counts(state.params.world_model, config),
                "training": compiler,
                "prior_field": compile_prior_field(state.params.world_model, config),
                "inference": {
                    str(nfe): compile_inference(
                        state.params.world_model,
                        resolved_config(
                            protocol,
                            profile,
                            arm,
                            action_dim=action_dim,
                            nfe=int(nfe),
                        ),
                    )
                    for nfe in protocol["evaluation"]["nfe_frontier"]
                },
            }
            one_nfe = compute[arm]["inference"]["1"]["flops"]
            field_flops = compute[arm]["prior_field"]["flops"]
            for nfe in protocol["evaluation"]["nfe_frontier"]:
                compute[arm]["inference"][str(nfe)]["nfe_adjusted_flops"] = float(
                    one_nfe + (int(nfe) - 1) * field_flops
                )
        shared_hashes = {
            arm: _tree_sha256(_shared_trunk(initial_states[arm].params))
            for arm in ARM_ORDER
        }
        if len(set(shared_hashes.values())) != 1:
            raise RuntimeError("paired objective arms do not share identical trunk initialization")

        reference_updates = int(profile_cfg["reference_updates"])
        if track == "equal_updates":
            allocations = {arm: reference_updates for arm in ARM_ORDER}
        else:
            budget = reference_updates * compute["shortcut_forcing"]["training"]["flops"]
            allocations = {
                "shortcut_forcing": reference_updates,
                "trajectory_imf": max(
                    1,
                    int(
                        math.floor(
                            budget / compute["trajectory_imf"]["training"]["flops"]
                        )
                    ),
                ),
            }
        max_updates = max(allocations.values())
        schedule = make_batch_schedule(
            len(train_dataset["observations"]),
            updates=max_updates,
            batch_size=int(profile_cfg["batch_size"]),
            sequence_length=int(profile_cfg["sequence_length"]),
            seed=derive_numpy_seed("batch-schedule", task, seed),
        )
        keys = _training_keys(task, seed, max_updates)
        _atomic_npz(staging / "batch_schedule.npz", {"starts": schedule})
        _atomic_npz(staging / "objective_keys.npz", {"world_model_loss_keys": keys})
        trained_states: dict[str, Any] = {}
        training_results: dict[str, Any] = {}
        for arm in ARM_ORDER:
            updates = allocations[arm]
            state, training = train_arm(
                arm=arm,
                state=initial_states[arm],
                config=configs[arm],
                executable=executables[arm],
                train_dataset=train_dataset,
                schedule=schedule[:updates],
                keys=keys[:updates],
                profile_cfg=profile_cfg,
            )
            trained_states[arm] = state
            training["compiler_flops_per_update"] = compute[arm]["training"]["flops"]
            training["realized_compiler_flops"] = (
                updates * compute[arm]["training"]["flops"]
            )
            training_results[arm] = training
            save_checkpoint(
                staging / f"checkpoint_{arm}.pkl",
                state,
                configs[arm],
                metadata={
                    "schema_version": "trajectory-imf-pixel-checkpoint-v1",
                    "profile": profile,
                    "task": task,
                    "seed": int(seed),
                    "track": track,
                    "arm": arm,
                    "updates": updates,
                    "protocol_sha256": file_sha256(PROTOCOL_PATH),
                    "dataset_sha256": file_sha256(staging / "dataset.npz"),
                    "batch_schedule_sha256": file_sha256(staging / "batch_schedule.npz"),
                    "objective_keys_sha256": file_sha256(staging / "objective_keys.npz"),
                },
            )

        retained, evaluation_identity = evaluate_models(
            protocol=protocol,
            profile=profile,
            task=task,
            seed=int(seed),
            heldout=heldout_dataset,
            states=trained_states,
            action_dim=action_dim,
        )
        train_newest = (
            train_dataset["observations"][:, :, :, -3:].astype(np.float64) / 255.0
        )
        channel_variance = np.var(train_newest, axis=(0, 1, 2), dtype=np.float64)
        metrics, raw_metrics = summarize_predictions(retained, channel_variance)
        retained = {
            **retained,
            "training_channel_variance": channel_variance.astype(np.float64),
            **raw_metrics,
        }
        _atomic_npz(staging / "predictions.npz", retained)
        for arm in ARM_ORDER:
            compute[arm]["allocated_updates"] = allocations[arm]
            compute[arm]["realized_compiler_train_flops"] = training_results[arm][
                "realized_compiler_flops"
            ]
        result = {
            "schema_version": "trajectory-imf-pixel-result-v1",
            "status": "complete",
            "study_role": "secondary_hard_visual_evaluation",
            "claim_eligible_for_primary_gate": False,
            "primary_gate_substitution": False,
            "profile": profile,
            "task": task,
            "seed": int(seed),
            "track": track,
            "renderer_backend": os.environ.get("MUJOCO_GL"),
            "action_dim": action_dim,
            "action_repeat": int(protocol["environment"]["action_repeat"]),
            "environment_seeds": environment_seeds,
            "protocol_sha256": file_sha256(PROTOCOL_PATH),
            "base_protocol_sha256": file_sha256(BASE_PROTOCOL_PATH),
            "runtime": runtime_fingerprint(),
            "shared_initial_trunk_sha256": shared_hashes[ARM_ORDER[0]],
            "allocations": allocations,
            "training": training_results,
            "compute": compute,
            "evaluation_identity": evaluation_identity,
            "metrics": metrics,
            "artifact_digests": {
                name: file_sha256(staging / name)
                for name in ARTIFACT_FILES
                if name != "result.json"
            },
        }
        _atomic_json(staging / "result.json", result)
        manifest = _artifact_manifest(staging)
        _atomic_json(staging / "manifest.json", manifest)
        seal = {
            "schema_version": "trajectory-imf-pixel-seal-v1",
            "manifest_sha256": file_sha256(staging / "manifest.json"),
            "manifest_body_sha256": manifest["body_sha256"],
        }
        _atomic_json(staging / "seal.json", seal)
        _reject_symlink_chain(destination.parent, label="pixel artifact output parent")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                "pixel artifact destination appeared while its staged result was running"
            )
        os.replace(staging, destination)
        return verify_artifact(
            destination, recompute_predictions=True, rederive_compute=False
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _verify_manifest(root: Path) -> dict[str, Any]:
    _validate_exact_artifact_entries(root)
    manifest = _json_load(root / "manifest.json")
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "artifact_files",
            "source_files",
            "protocol_sha256",
            "base_protocol_sha256",
            "body_sha256",
        },
        "pixel artifact manifest",
    )
    if manifest["schema_version"] != "trajectory-imf-pixel-artifact-manifest-v1":
        raise ValueError("pixel artifact manifest version is invalid")
    body = {key: value for key, value in manifest.items() if key != "body_sha256"}
    _require_sha256(manifest["body_sha256"], "pixel artifact manifest body digest")
    if manifest["body_sha256"] != object_sha256(body):
        raise ValueError("pixel artifact manifest body digest mismatch")
    if not isinstance(manifest["artifact_files"], list):
        raise ValueError("pixel artifact manifest file inventory must be a list")
    for row in manifest["artifact_files"]:
        _require_exact_keys(row, {"path", "bytes", "sha256"}, "pixel artifact file record")
        if type(row["bytes"]) is not int or row["bytes"] < 0:
            raise ValueError("pixel artifact byte count must be a nonnegative integer")
        _require_sha256(row["sha256"], "pixel artifact file digest")
        path = root / str(row["path"])
        if path.stat().st_size != row["bytes"] or file_sha256(path) != row["sha256"]:
            raise ValueError(f"pixel artifact digest mismatch for {row['path']}")
    if [row["path"] for row in manifest["artifact_files"]] != list(ARTIFACT_FILES):
        raise ValueError("pixel artifact manifest paths are not canonical")
    if not isinstance(manifest["source_files"], list) or not manifest["source_files"]:
        raise ValueError("pixel source manifest must be a nonempty list")
    for row in manifest["source_files"]:
        _require_exact_keys(row, {"path", "bytes", "sha256"}, "pixel source record")
        if type(row["bytes"]) is not int or row["bytes"] < 0:
            raise ValueError("pixel source byte count must be a nonnegative integer")
        _require_sha256(row["sha256"], "pixel source digest")
    _validate_source_manifest(manifest["source_files"])
    _require_sha256(manifest["protocol_sha256"], "pixel manifest protocol digest")
    _require_sha256(manifest["base_protocol_sha256"], "pixel manifest base protocol digest")
    if manifest["protocol_sha256"] != file_sha256(PROTOCOL_PATH):
        raise ValueError("pixel protocol source digest mismatch")
    if manifest["base_protocol_sha256"] != file_sha256(BASE_PROTOCOL_PATH):
        raise ValueError("base matched protocol source digest mismatch")
    seal = _json_load(root / "seal.json")
    _require_exact_keys(
        seal,
        {"schema_version", "manifest_sha256", "manifest_body_sha256"},
        "pixel artifact seal",
    )
    if seal != {
        "schema_version": "trajectory-imf-pixel-seal-v1",
        "manifest_sha256": file_sha256(root / "manifest.json"),
        "manifest_body_sha256": manifest["body_sha256"],
    }:
        raise ValueError("pixel artifact seal does not bind the manifest")
    return manifest


def _replay_predictions(
    root: Path,
    protocol: Mapping[str, Any],
    result: Mapping[str, Any],
    retained: Mapping[str, np.ndarray],
) -> None:
    from imf_dreamer_jax import load_checkpoint

    states: dict[str, Any] = {}
    for arm in ARM_ORDER:
        state, config, metadata = load_checkpoint(root / f"checkpoint_{arm}.pkl")
        expected_config = resolved_config(
            protocol,
            str(result["profile"]),
            arm,
            action_dim=int(result["action_dim"]),
        )
        if asdict(config) != asdict(expected_config):
            raise ValueError("pixel checkpoint resolved config mismatch")
        expected_metadata = {
            "schema_version": "trajectory-imf-pixel-checkpoint-v1",
            "profile": result["profile"],
            "task": result["task"],
            "seed": result["seed"],
            "track": result["track"],
            "arm": arm,
            "updates": result["allocations"][arm],
            "protocol_sha256": result["protocol_sha256"],
            "dataset_sha256": result["artifact_digests"]["dataset.npz"],
            "batch_schedule_sha256": result["artifact_digests"]["batch_schedule.npz"],
            "objective_keys_sha256": result["artifact_digests"]["objective_keys.npz"],
        }
        if metadata != expected_metadata:
            raise ValueError("pixel checkpoint metadata mismatch")
        if _tree_sha256(state.params.world_model) != result["training"][arm][
            "world_model_parameter_sha256"
        ]:
            raise ValueError("pixel checkpoint parameter digest mismatch")
        _validate_checkpoint_state(
            state,
            updates=int(result["allocations"][arm]),
            arm=arm,
        )
        if _parameter_counts(state.params.world_model, config) != result["compute"][arm][
            "parameters"
        ]:
            raise ValueError("pixel checkpoint parameter counts differ from result accounting")
        states[arm] = state
    reconstructed, identity = evaluate_models_from_retained(
        protocol=protocol,
        profile=str(result["profile"]),
        action_dim=int(result["action_dim"]),
        states=states,
        retained=retained,
    )
    if identity != result["evaluation_identity"]:
        raise ValueError("retained evaluation identity is inconsistent")
    expected = np.asarray(retained["predictions"], np.float32)
    if not np.allclose(reconstructed, expected, rtol=2e-6, atol=2e-6):
        maximum = float(np.max(np.abs(reconstructed - expected)))
        raise ValueError(f"checkpoint-recomputed pixel predictions differ (max={maximum})")


def evaluate_models_from_retained(
    *,
    protocol: Mapping[str, Any],
    profile: str,
    action_dim: int,
    states: Mapping[str, Any],
    retained: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import observe_sequence

    context_observations = np.asarray(retained["context_observations"], np.uint8)
    context_actions = np.asarray(retained["context_actions"], np.float32)
    context_first = np.asarray(retained["context_is_first"], np.bool_)
    future_actions = np.asarray(retained["future_actions"], np.float32)
    posterior_keys = np.asarray(retained["posterior_keys"], np.uint32)
    base_noise = np.asarray(retained["base_noise"], np.float32)
    horizons = [int(value) for value in retained["horizons"].tolist()]
    nfes = [int(value) for value in retained["nfe_values"].tolist()]
    arms = [str(value) for value in retained["arm_names"].tolist()]
    if arms != list(ARM_ORDER) or nfes != [1, 2, 4]:
        raise ValueError("retained evaluation axes are not canonical")
    indices = np.asarray(horizons, np.int64) - 1
    output = np.empty_like(np.asarray(retained["predictions"], np.float32))
    for arm_index, arm in enumerate(arms):
        for nfe_index, nfe in enumerate(nfes):
            config = resolved_config(
                protocol, profile, arm, action_dim=action_dim, nfe=nfe
            )
            rollout = jax.jit(
                lambda params, start_state, actions, noise: _rollout(
                    params, start_state, actions, noise, config
                )
            )
            for window in range(len(context_observations)):
                sequence = observe_sequence(
                    states[arm].params.world_model,
                    preprocess_pixels_jax(context_observations[window : window + 1]),
                    jnp.asarray(context_actions[window : window + 1]),
                    jnp.asarray(posterior_keys[window]),
                    config,
                    is_first=jnp.asarray(context_first[window : window + 1]),
                )
                start_state = type(sequence.states)(
                    sequence.states.deterministic[:, -1],
                    sequence.states.stochastic[:, -1],
                )
                decoded = rollout(
                    states[arm].params.world_model,
                    start_state,
                    jnp.asarray(future_actions[window : window + 1]),
                    jnp.asarray(base_noise[window, :, None]),
                )
                jax.block_until_ready(decoded)
                output[arm_index, nfe_index, window] = np.asarray(decoded)[
                    :, 0, indices, :, :, -3:
                ]
    return output, {
        "eligible_window_count": int(result_value(retained, "eligible_window_count", default=-1)),
        "selected_window_count": int(len(context_observations)),
        "paired_windows": True,
        "paired_actions": True,
        "paired_posterior_keys": True,
        "paired_base_noise": True,
    }


def result_value(retained: Mapping[str, np.ndarray], name: str, *, default: int) -> int:
    key = f"identity__{name}"
    if key not in retained:
        return default
    return int(np.asarray(retained[key]).reshape(()))


def _validate_retained_archive_schema(
    retained: Mapping[str, np.ndarray],
    protocol: Mapping[str, Any],
    profile: str,
    action_dim: int,
) -> None:
    profile_cfg = protocol["profiles"][profile]
    windows = int(profile_cfg["windows"])
    draws = int(profile_cfg["predictive_draws"])
    horizons = [int(value) for value in profile_cfg["horizons"]]
    max_horizon = max(horizons)
    context = int(profile_cfg["context_length"])
    stochastic_dim = int(
        resolved_config(protocol, profile, ARM_ORDER[0], action_dim=action_dim).stochastic_dim
    )
    core = {
        "arm_names": ((len(ARM_ORDER),), np.dtype("<U32")),
        "nfe_values": ((3,), np.dtype(np.int16)),
        "horizons": ((len(horizons),), np.dtype(np.int16)),
        "window_starts": ((windows,), np.dtype(np.int64)),
        "context_observations": (
            (windows, context, 32, 32, 9),
            np.dtype(np.uint8),
        ),
        "context_actions": ((windows, context, action_dim), np.dtype(np.float32)),
        "context_is_first": ((windows, context), np.dtype(np.bool_)),
        "future_actions": (
            (windows, max_horizon, action_dim),
            np.dtype(np.float32),
        ),
        "posterior_keys": ((windows, 2), np.dtype(np.uint32)),
        "base_noise": (
            (windows, draws, max_horizon, stochastic_dim),
            np.dtype(np.float32),
        ),
        "targets": ((windows, len(horizons), 32, 32, 3), np.dtype(np.float32)),
        "predictions": (
            (len(ARM_ORDER), 3, windows, draws, len(horizons), 32, 32, 3),
            np.dtype(np.float32),
        ),
        "training_channel_variance": ((3,), np.dtype(np.float64)),
    }
    raw_names = {
        f"metric__{arm}__nfe_{nfe}__{metric}"
        for arm in ARM_ORDER
        for nfe in (1, 2, 4)
        for metric in (
            "normalized_visual_mse",
            "predictive_mean_psnr_db",
            "predictive_mean_global_ssim",
            "energy_score",
            "temporal_difference_mse",
        )
    }
    if set(retained) != set(core) | raw_names:
        raise ValueError("retained pixel prediction archive array set is not exact")
    for name, (shape, dtype) in core.items():
        value = np.asarray(retained[name])
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(
                f"retained pixel array {name} has shape/dtype {value.shape}/{value.dtype}, "
                f"expected {shape}/{dtype}"
            )
    for name in raw_names:
        value = np.asarray(retained[name])
        if value.shape != (windows, len(horizons)) or value.dtype != np.float32:
            raise ValueError(f"retained raw metric array has invalid schema: {name}")


def _rederive_compute(
    protocol: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    from imf_dreamer_jax import create_agent

    initialization_key = derive_jax_key(
        "world-init", result["task"], result["seed"]
    )
    profile_cfg = protocol["profiles"][result["profile"]]
    for arm in ARM_ORDER:
        config = resolved_config(
            protocol,
            result["profile"],
            arm,
            action_dim=int(result["action_dim"]),
        )
        state = create_agent(config, initialization_key)
        _, train = compile_training(
            state,
            config,
            profile_cfg,
            derive_jax_key("compile", result["task"], result["seed"], arm),
        )
        if not _same_json_number(train, result["compute"][arm]["training"]):
            raise ValueError("pixel training compiler evidence does not rederive")
        prior_field = compile_prior_field(state.params.world_model, config)
        if not _same_json_number(
            prior_field, result["compute"][arm]["prior_field"]
        ):
            raise ValueError("pixel prior-field compiler evidence does not rederive")
        one_nfe_flops = float(result["compute"][arm]["inference"]["1"]["flops"])
        for nfe in protocol["evaluation"]["nfe_frontier"]:
            inference = compile_inference(
                state.params.world_model,
                resolved_config(
                    protocol,
                    result["profile"],
                    arm,
                    action_dim=int(result["action_dim"]),
                    nfe=int(nfe),
                ),
            )
            inference["nfe_adjusted_flops"] = float(
                one_nfe_flops + (int(nfe) - 1) * prior_field["flops"]
            )
            if not _same_json_number(
                inference, result["compute"][arm]["inference"][str(nfe)]
            ):
                raise ValueError("pixel inference compiler evidence does not rederive")


def verify_artifact(
    root: str | Path,
    *,
    recompute_predictions: bool,
    rederive_compute: bool,
) -> dict[str, Any]:
    directory = _artifact_root(root)
    protocol = load_protocol()
    _verify_manifest(directory)
    stored_protocol = _json_load(directory / "protocol.json")
    if stored_protocol != protocol:
        raise ValueError("stored pixel protocol differs from the frozen source")
    result = _json_load(directory / "result.json")
    _validate_result_schema(result, protocol)
    profile = str(result["profile"])
    profile_cfg = protocol["profiles"][profile]
    if (
        result["task"] not in profile_cfg["tasks"]
        or result["seed"] not in profile_cfg["seeds"]
        or result["track"] not in profile_cfg["tracks"]
        or result["action_repeat"] != protocol["environment"]["action_repeat"]
    ):
        raise ValueError("pixel result identity is outside the frozen profile")
    if result["protocol_sha256"] != file_sha256(PROTOCOL_PATH) or result[
        "base_protocol_sha256"
    ] != file_sha256(BASE_PROTOCOL_PATH):
        raise ValueError("pixel result protocol digests are invalid")
    expected_backend = (
        protocol["rendering"]["local_smoke_backend"]
        if profile == "smoke"
        else protocol["rendering"]["hard_backend"]
    )
    if (
        result.get("renderer_backend") != expected_backend
        or result.get("runtime", {}).get("mujoco_gl") != expected_backend
    ):
        raise ValueError("pixel result used a renderer backend outside its frozen profile")
    expected_environment_seeds = {
        "training": derive_environment_seed(
            "environment-train", result["task"], result["seed"]
        ),
        "heldout": derive_environment_seed(
            "environment-heldout", result["task"], result["seed"]
        ),
    }
    if result.get("environment_seeds") != expected_environment_seeds:
        raise ValueError("pixel result DMC seeds are not the frozen derivation")
    expected_digest_names = set(ARTIFACT_FILES) - {"result.json"}
    if set(result.get("artifact_digests", {})) != expected_digest_names:
        raise ValueError("pixel result nested artifact digest set is not exact")
    for name, digest in result["artifact_digests"].items():
        if name == "result.json" or name not in ARTIFACT_FILES:
            raise ValueError("pixel result contains an invalid artifact digest key")
        if file_sha256(directory / name) != digest:
            raise ValueError(f"pixel result nested digest mismatch for {name}")

    archive = _load_npz(directory / "dataset.npz")
    train, heldout = split_dataset_archive(archive)
    validate_dataset(
        train,
        int(result["action_dim"]),
        int(profile_cfg["training_observations"]),
        int(protocol["environment"]["action_repeat"]),
    )
    validate_dataset(
        heldout,
        int(result["action_dim"]),
        int(profile_cfg["heldout_observations"]),
        int(protocol["environment"]["action_repeat"]),
    )
    exploration = protocol["environment"]["exploration_policy"]
    validate_exploration_actions(
        train,
        exploration_seed=derive_numpy_seed(
            "exploration-train", result["task"], result["seed"]
        ),
        ar_coefficient=float(exploration["coefficient"]),
        innovation_scale=float(exploration["innovation_scale"]),
    )
    validate_exploration_actions(
        heldout,
        exploration_seed=derive_numpy_seed(
            "exploration-heldout", result["task"], result["seed"]
        ),
        ar_coefficient=float(exploration["coefficient"]),
        innovation_scale=float(exploration["innovation_scale"]),
    )
    schedules = _load_npz(directory / "batch_schedule.npz")
    key_archive = _load_npz(directory / "objective_keys.npz")
    if set(schedules) != {"starts"} or set(key_archive) != {"world_model_loss_keys"}:
        raise ValueError("pixel schedule/key archives have extra or missing arrays")
    expected_schedule = make_batch_schedule(
        len(train["observations"]),
        updates=max(int(value) for value in result["allocations"].values()),
        batch_size=int(profile_cfg["batch_size"]),
        sequence_length=int(profile_cfg["sequence_length"]),
        seed=derive_numpy_seed("batch-schedule", result["task"], result["seed"]),
    )
    expected_keys = _training_keys(
        result["task"], result["seed"], len(expected_schedule)
    )
    if (
        schedules["starts"].dtype != np.int64
        or key_archive["world_model_loss_keys"].dtype != np.uint32
        or schedules["starts"].shape != expected_schedule.shape
        or key_archive["world_model_loss_keys"].shape != expected_keys.shape
        or not np.array_equal(schedules["starts"], expected_schedule)
        or not np.array_equal(key_archive["world_model_loss_keys"], expected_keys)
    ):
        raise ValueError("pixel batch/objective RNG schedules are not canonical")
    reference_updates = int(profile_cfg["reference_updates"])
    if result["track"] == "equal_updates":
        expected_allocations = {arm: reference_updates for arm in ARM_ORDER}
    else:
        shortcut_flops = float(result["compute"]["shortcut_forcing"]["training"]["flops"])
        trajectory_flops = float(result["compute"]["trajectory_imf"]["training"]["flops"])
        expected_allocations = {
            "shortcut_forcing": reference_updates,
            "trajectory_imf": max(
                1, int(math.floor(reference_updates * shortcut_flops / trajectory_flops))
            ),
        }
    if result.get("allocations") != expected_allocations:
        raise ValueError("pixel update allocations do not follow the frozen compute rule")
    for arm in ARM_ORDER:
        config = resolved_config(
            protocol, profile, arm, action_dim=int(result["action_dim"])
        )
        compute = result["compute"][arm]
        parameters = compute["parameters"]
        if (
            object_sha256(compute["resolved_config"])
            != object_sha256(asdict(config))
            or compute["resolved_config_sha256"] != object_sha256(asdict(config))
            or compute["allocated_updates"] != expected_allocations[arm]
            or result["training"][arm]["updates"] != expected_allocations[arm]
            or not math.isclose(
                float(result["training"][arm]["compiler_flops_per_update"]),
                float(compute["training"]["flops"]),
                rel_tol=1e-12,
                abs_tol=0.0,
            )
            or parameters["active"] > parameters["total"]
            or parameters["prior_active"] > parameters["prior_total"]
            or parameters["prior_total"] > parameters["total"]
            or not math.isclose(
                float(compute["realized_compiler_train_flops"]),
                float(compute["training"]["flops"]) * expected_allocations[arm],
                rel_tol=1e-12,
                abs_tol=0.0,
            )
            or not math.isclose(
                float(result["training"][arm]["realized_compiler_flops"]),
                float(compute["realized_compiler_train_flops"]),
                rel_tol=1e-12,
                abs_tol=0.0,
            )
        ):
            raise ValueError("pixel training/config compute accounting is inconsistent")
        if set(compute["inference"]) != {"1", "2", "4"}:
            raise ValueError("pixel inference compute frontier is incomplete")
        for nfe in (1, 2, 4):
            inference = compute["inference"][str(nfe)]
            expected_adjusted = float(
                compute["inference"]["1"]["flops"]
                + (nfe - 1) * compute["prior_field"]["flops"]
            )
            if inference["field_evaluations"] != nfe or not math.isclose(
                inference["nfe_adjusted_flops"],
                expected_adjusted,
                rel_tol=1e-12,
                abs_tol=0.0,
            ):
                raise ValueError("pixel prior field-evaluation accounting is invalid")
    retained = _load_npz(directory / "predictions.npz")
    _validate_retained_archive_schema(
        retained, protocol, profile, int(result["action_dim"])
    )
    channel_variance = np.var(
        train["observations"][:, :, :, -3:].astype(np.float64) / 255.0,
        axis=(0, 1, 2),
        dtype=np.float64,
    )
    if not np.allclose(
        retained["training_channel_variance"], channel_variance, rtol=0.0, atol=0.0
    ):
        raise ValueError("retained training pixel normalizer is invalid")
    metrics, raw_metrics = summarize_predictions(retained, channel_variance)
    if not np.isfinite(retained["predictions"]).all() or not np.isfinite(
        retained["targets"]
    ).all():
        raise ValueError("retained pixel predictions or targets are non-finite")
    if not _same_json_number(metrics, result["metrics"], atol=2e-6):
        raise ValueError("pixel aggregate metrics do not recompute from raw predictions")
    for name, value in raw_metrics.items():
        if name not in retained or not np.allclose(
            retained[name], value, rtol=2e-6, atol=2e-6
        ):
            raise ValueError(f"retained raw visual metric does not recompute: {name}")
    selected = len(retained["window_starts"])
    if selected != int(profile_cfg["windows"]):
        raise ValueError("retained visual window count differs from the protocol")
    expected_eligible = eligible_window_starts(
        heldout,
        int(profile_cfg["context_length"]),
        max(int(value) for value in profile_cfg["horizons"]),
    )
    expected_starts = select_windows(
        expected_eligible,
        int(profile_cfg["windows"]),
        seed=derive_numpy_seed(
            "evaluation-window", result["task"], result["seed"], profile
        ),
    )
    if not np.array_equal(retained["window_starts"], expected_starts):
        raise ValueError("retained visual windows are not the frozen selection")
    context_length = int(profile_cfg["context_length"])
    max_horizon = max(int(value) for value in profile_cfg["horizons"])
    target_indices = np.asarray(profile_cfg["horizons"], np.int64) - 1
    expected_context_observations = np.stack(
        [heldout["observations"][start : start + context_length] for start in expected_starts]
    ).astype(np.uint8)
    expected_context_actions = np.stack(
        [heldout["actions"][start : start + context_length] for start in expected_starts]
    ).astype(np.float32)
    expected_context_first = np.stack(
        [heldout["is_first"][start : start + context_length] for start in expected_starts]
    ).astype(np.bool_)
    expected_context_actions[:, 0] = 0.0
    expected_context_first[:, 0] = True
    expected_future_actions = np.stack(
        [
            heldout["actions"][start + context_length : start + context_length + max_horizon]
            for start in expected_starts
        ]
    ).astype(np.float32)
    expected_target_stacks = np.stack(
        [
            heldout["observations"][
                start + context_length : start + context_length + max_horizon
            ]
            for start in expected_starts
        ]
    )
    expected_targets = (
        expected_target_stacks[:, target_indices, :, :, -3:].astype(np.float32) / 255.0
    )
    expected_posterior_keys = np.stack(
        [
            np.asarray(
                derive_jax_key(
                    "posterior-context", result["task"], result["seed"], int(start)
                ),
                np.uint32,
            )
            for start in expected_starts
        ]
    )
    noise_rng = np.random.default_rng(
        derive_numpy_seed(
            "evaluation-noise", result["task"], result["seed"], profile
        )
    )
    stochastic_dim = resolved_config(
        protocol, profile, ARM_ORDER[0], action_dim=int(result["action_dim"])
    ).stochastic_dim
    expected_noise = noise_rng.normal(
        size=(
            int(profile_cfg["windows"]),
            int(profile_cfg["predictive_draws"]),
            max_horizon,
            stochastic_dim,
        )
    ).astype(np.float32)
    expected_inputs = {
        "context_observations": expected_context_observations,
        "context_actions": expected_context_actions,
        "context_is_first": expected_context_first,
        "future_actions": expected_future_actions,
        "posterior_keys": expected_posterior_keys,
        "base_noise": expected_noise,
        "targets": expected_targets,
        "arm_names": np.asarray(ARM_ORDER, dtype="<U32"),
        "nfe_values": np.asarray([1, 2, 4], np.int16),
        "horizons": np.asarray(profile_cfg["horizons"], np.int16),
    }
    for name, expected in expected_inputs.items():
        if not np.array_equal(retained[name], expected):
            raise ValueError(f"retained visual evaluation input is not canonical: {name}")
    expected_identity = {
        "eligible_window_count": int(len(expected_eligible)),
        "selected_window_count": int(profile_cfg["windows"]),
        "paired_windows": True,
        "paired_actions": True,
        "paired_posterior_keys": True,
        "paired_base_noise": True,
    }
    if result.get("evaluation_identity") != expected_identity:
        raise ValueError("pixel evaluation identity does not recompute from held-out data")
    # Bind the available-window count into replay without trusting result.json.
    retained = dict(retained)
    retained["identity__eligible_window_count"] = np.asarray(
        len(expected_eligible), np.int64
    )
    if recompute_predictions:
        if profile == "hard":
            current_runtime = runtime_fingerprint()
            _validate_runtime_fingerprint(current_runtime, protocol, profile)
            if _runtime_identity(current_runtime) != _runtime_identity(result["runtime"]):
                raise ValueError(
                    "hard checkpoint/environment replay requires the original homogeneous GPU runtime"
                )
        replay_pixel_dataset(
            protocol,
            profile=profile,
            task=str(result["task"]),
            environment_seed=int(result["environment_seeds"]["training"]),
            arrays=train,
        )
        replay_pixel_dataset(
            protocol,
            profile=profile,
            task=str(result["task"]),
            environment_seed=int(result["environment_seeds"]["heldout"]),
            arrays=heldout,
        )
        _replay_predictions(directory, protocol, result, retained)
    if rederive_compute:
        current = runtime_fingerprint()
        stored = result["runtime"]
        _validate_runtime_fingerprint(current, protocol, profile)
        if _runtime_identity(current) != _runtime_identity(stored):
            raise ValueError("compiler rederivation requires the original JAX device runtime")
        _rederive_compute(protocol, result)
    return result


def interquartile_mean(values: Sequence[float]) -> float:
    """Exact empirical 25%-trimmed mean, including fractional boundary mass."""

    sorted_values = np.sort(np.asarray(values, np.float64).reshape(-1))
    if len(sorted_values) == 0 or not np.isfinite(sorted_values).all():
        raise ValueError("IQM requires at least one finite value")
    lower = 0.25 * len(sorted_values)
    upper = 0.75 * len(sorted_values)
    total = 0.0
    mass = 0.0
    for index, value in enumerate(sorted_values):
        weight = max(0.0, min(index + 1.0, upper) - max(float(index), lower))
        total += weight * float(value)
        mass += weight
    if mass <= 0.0:
        raise AssertionError("IQM trimming unexpectedly retained no mass")
    return total / mass


def _runtime_identity(runtime: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: runtime[key]
        for key in (
            "python",
            "platform",
            "jax",
            "jaxlib",
            "numpy",
            "dm_control",
            "mujoco",
            "backend",
            "jax_enable_x64",
            "device_count",
            "visible_device_count",
            "device_platforms",
            "device_kinds",
            "mujoco_gl",
            "jax_enable_x64_environment",
            "jax_platform_name",
        )
    }


def _requested_tracks(
    protocol: Mapping[str, Any], tracks: Sequence[str] | None
) -> list[str]:
    registered = list(protocol["profiles"]["hard"]["tracks"])
    if tracks is None:
        return registered
    if isinstance(tracks, (str, bytes)) or not tracks:
        raise ValueError("requested pixel tracks must be a nonempty sequence")
    requested = [str(track) for track in tracks]
    if len(requested) != len(set(requested)) or not set(requested).issubset(registered):
        raise ValueError("requested pixel tracks are duplicated or unregistered")
    return [track for track in registered if track in requested]


def _expected_hard_identities(
    protocol: Mapping[str, Any], tracks: Sequence[str]
) -> set[tuple[str, str, int]]:
    hard = protocol["profiles"]["hard"]
    return {
        (track, task, int(seed))
        for track in tracks
        for task in hard["tasks"]
        for seed in hard["seeds"]
    }


def _track_summaries_from_auc(
    protocol: Mapping[str, Any],
    tracks: Sequence[str],
    auc: Mapping[tuple[str, str, int, str, int], float],
) -> dict[str, Any]:
    hard = protocol["profiles"]["hard"]
    tasks = list(hard["tasks"])
    seeds = [int(value) for value in hard["seeds"]]
    summaries: dict[str, Any] = {}
    for track in tracks:
        primary_delta = np.empty((len(tasks), len(seeds)), np.float64)
        for task_index, task in enumerate(tasks):
            for seed_index, seed in enumerate(seeds):
                primary_delta[task_index, seed_index] = (
                    auc[(track, task, seed, "shortcut_forcing", 4)]
                    - auc[(track, task, seed, "trajectory_imf", 1)]
                )
        rng = np.random.default_rng(
            derive_numpy_seed(
                "aggregate-bootstrap",
                int(protocol["analysis"]["bootstrap_seed"]),
                track,
            )
        )
        bootstrap = np.empty((int(protocol["analysis"]["bootstrap_draws"]),), np.float64)
        for draw in range(len(bootstrap)):
            sampled_tasks = rng.integers(0, len(tasks), size=len(tasks))
            sampled_values: list[float] = []
            for task_index in sampled_tasks:
                sampled_seeds = rng.integers(0, len(seeds), size=len(seeds))
                sampled_values.extend(primary_delta[task_index, sampled_seeds].tolist())
            bootstrap[draw] = interquartile_mean(sampled_values)
        alpha = 1.0 - float(protocol["analysis"]["confidence_level"])
        summaries[track] = {
            "primary_visual_delta_shortcut_minus_trajectory": interquartile_mean(
                primary_delta.ravel()
            ),
            "positive_favors_trajectory_imf": True,
            "descriptive_hierarchical_bootstrap_interval": [
                float(np.quantile(bootstrap, alpha / 2.0)),
                float(np.quantile(bootstrap, 1.0 - alpha / 2.0)),
            ],
            "task_by_seed_primary_delta": primary_delta.tolist(),
            "task_order": tasks,
            "seed_order": seeds,
            "frontier_iqm": {
                arm: {
                    str(nfe): interquartile_mean(
                        [
                            auc[(track, task, seed, arm, int(nfe))]
                            for task in tasks
                            for seed in seeds
                        ]
                    )
                    for nfe in protocol["evaluation"]["nfe_frontier"]
                }
                for arm in ARM_ORDER
            },
        }
    return summaries


def _input_record(root: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    identity = {
        "track": str(result["track"]),
        "task": str(result["task"]),
        "seed": int(result["seed"]),
    }
    digests = {
        "manifest_sha256": file_sha256(root / "manifest.json"),
        "seal_sha256": file_sha256(root / "seal.json"),
        "result_sha256": file_sha256(root / "result.json"),
        "predictions_sha256": file_sha256(root / "predictions.npz"),
    }
    commitment = {**identity, **digests}
    return {
        **identity,
        "path": str(root),
        **digests,
        "artifact_identity_sha256": object_sha256(commitment),
    }


def aggregate_artifacts(
    roots: Sequence[str | Path],
    output: str | Path | None = None,
    *,
    tracks: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Aggregate exactly the requested complete hard-track matrices."""

    protocol = load_protocol()
    requested = _requested_tracks(protocol, tracks)
    expected_identities = _expected_hard_identities(protocol, requested)
    if len(roots) != len(expected_identities):
        raise ValueError(
            "pixel aggregate input count is not the exact requested matrix: "
            f"expected={len(expected_identities)}, observed={len(roots)}"
        )
    rows: dict[tuple[str, str, int], tuple[Path, dict[str, Any]]] = {}
    seen_roots: set[Path] = set()
    runtime_identity: dict[str, Any] | None = None
    for root_value in roots:
        root = _artifact_root(root_value)
        if root in seen_roots:
            raise ValueError("duplicate pixel artifact root")
        seen_roots.add(root)
        result = verify_artifact(root, recompute_predictions=False, rederive_compute=False)
        if result["profile"] != "hard":
            raise ValueError("pixel aggregation accepts only hard-profile artifacts")
        identity = (str(result["track"]), str(result["task"]), int(result["seed"]))
        if identity in rows:
            raise ValueError("duplicate task/seed/track pixel artifact")
        if identity not in expected_identities:
            raise ValueError(f"unexpected pixel aggregate identity: {identity}")
        current_runtime = _runtime_identity(result["runtime"])
        if runtime_identity is None:
            runtime_identity = current_runtime
        elif current_runtime != runtime_identity:
            raise ValueError("hard pixel artifacts do not use one homogeneous GPU runtime")
        rows[identity] = (root, result)
    observed = set(rows)
    if observed != expected_identities:
        raise ValueError(
            "hard pixel aggregate identities are incomplete: "
            f"missing={sorted(expected_identities - observed)}, "
            f"extra={sorted(observed - expected_identities)}"
        )
    if runtime_identity is None or runtime_identity["backend"] != "gpu":
        raise ValueError("hard pixel aggregation requires authenticated GPU artifacts")
    auc = {
        (track, task, seed, arm, int(nfe)): float(
            rows[(track, task, seed)][1]["metrics"]["frontier"][arm][str(nfe)][
                "normalized_visual_mse_auc"
            ]
        )
        for track, task, seed in expected_identities
        for arm in ARM_ORDER
        for nfe in protocol["evaluation"]["nfe_frontier"]
    }
    hard = protocol["profiles"]["hard"]
    ordered_identities = [
        (track, task, int(seed))
        for track in requested
        for task in hard["tasks"]
        for seed in hard["seeds"]
    ]
    body = {
        "schema_version": "trajectory-imf-pixel-aggregate-v1",
        "status": "complete",
        "study_role": "descriptive_secondary_hard_visual_evaluation",
        "claim_eligible_for_primary_gate": False,
        "primary_gate_substitution": False,
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "requested_tracks": requested,
        "task_order": list(hard["tasks"]),
        "seed_order": [int(seed) for seed in hard["seeds"]],
        "expected_input_count": len(expected_identities),
        "runtime_identity": runtime_identity,
        "tracks": _track_summaries_from_auc(protocol, requested, auc),
        "inputs": [_input_record(*rows[identity]) for identity in ordered_identities],
    }
    summary = {**body, "analysis_sha256": object_sha256(body)}
    if output is not None:
        destination = _reject_symlink_chain(
            output, label="pixel aggregate output"
        )
        if destination.exists():
            if not stat.S_ISREG(destination.lstat().st_mode):
                raise ValueError("pixel aggregate output must be a regular file")
            if _json_load(destination) != summary:
                raise ValueError("refusing to overwrite a different pixel aggregate")
        else:
            _atomic_json(destination, summary)
        return verify_aggregate(destination, roots)
    return summary


def _independent_iqm(values: Sequence[float]) -> float:
    """Second implementation used only by aggregate verification."""

    ordered = np.sort(np.asarray(values, np.float64).reshape(-1))
    if len(ordered) == 0 or not np.isfinite(ordered).all():
        raise ValueError("independent IQM received invalid values")
    left = len(ordered) / 4.0
    right = 3.0 * len(ordered) / 4.0
    weights = np.asarray(
        [max(0.0, min(index + 1.0, right) - max(float(index), left)) for index in range(len(ordered))],
        np.float64,
    )
    return float(np.dot(ordered, weights) / np.sum(weights))


def _independent_auc_by_cell(
    root: Path, result: Mapping[str, Any]
) -> dict[tuple[str, int], float]:
    retained = _load_npz(root / "predictions.npz")
    arms = [str(value) for value in retained["arm_names"].tolist()]
    nfes = [int(value) for value in retained["nfe_values"].tolist()]
    horizons = np.asarray(retained["horizons"], np.float64)
    target = np.asarray(retained["targets"], np.float64)
    predictions = np.asarray(retained["predictions"], np.float64)
    variance = np.maximum(
        np.asarray(retained["training_channel_variance"], np.float64), 0.0004
    )
    output: dict[tuple[str, int], float] = {}
    for arm_index, arm in enumerate(arms):
        for nfe_index, nfe in enumerate(nfes):
            error = predictions[arm_index, nfe_index] - target[:, None]
            horizon_mse = np.mean(
                np.square(error) / variance.reshape((1, 1, 1, 1, 1, 3)),
                axis=(0, 1, 3, 4, 5),
            )
            output[(arm, nfe)] = float(
                np.trapezoid(horizon_mse, horizons) / (horizons[-1] - horizons[0])
                if len(horizons) > 1
                else horizon_mse[0]
            )
            stored = float(
                result["metrics"]["frontier"][arm][str(nfe)][
                    "normalized_visual_mse_auc"
                ]
            )
            if not math.isclose(output[(arm, nfe)], stored, rel_tol=2e-6, abs_tol=2e-6):
                raise ValueError("independent raw-prediction AUC differs from the cell result")
    return output


def _independent_track_summaries(
    protocol: Mapping[str, Any],
    tracks: Sequence[str],
    auc: Mapping[tuple[str, str, int, str, int], float],
) -> dict[str, Any]:
    """Independent task/seed matrix, bootstrap, and frontier rederivation."""

    tasks = [str(task) for task in protocol["profiles"]["hard"]["tasks"]]
    seeds = [int(seed) for seed in protocol["profiles"]["hard"]["seeds"]]
    nfes = [int(nfe) for nfe in protocol["evaluation"]["nfe_frontier"]]
    output: dict[str, Any] = {}
    for track in tracks:
        delta_rows: list[list[float]] = []
        for task in tasks:
            row: list[float] = []
            for seed in seeds:
                shortcut = auc[(track, task, seed, "shortcut_forcing", 4)]
                trajectory = auc[(track, task, seed, "trajectory_imf", 1)]
                row.append(float(shortcut - trajectory))
            delta_rows.append(row)
        delta = np.asarray(delta_rows, np.float64)
        rng = np.random.default_rng(
            derive_numpy_seed(
                "aggregate-bootstrap",
                int(protocol["analysis"]["bootstrap_seed"]),
                track,
            )
        )
        replicated: list[float] = []
        for _ in range(int(protocol["analysis"]["bootstrap_draws"])):
            task_draw = rng.integers(0, len(tasks), size=len(tasks))
            blocks: list[np.ndarray] = []
            for task_index in task_draw:
                seed_draw = rng.integers(0, len(seeds), size=len(seeds))
                blocks.append(delta[int(task_index), seed_draw])
            replicated.append(_independent_iqm(np.concatenate(blocks)))
        alpha = 1.0 - float(protocol["analysis"]["confidence_level"])
        bootstrap = np.asarray(replicated, np.float64)
        frontier: dict[str, dict[str, float]] = {}
        for arm in ARM_ORDER:
            frontier[arm] = {}
            for nfe in nfes:
                cell_values: list[float] = []
                for task in tasks:
                    for seed in seeds:
                        cell_values.append(auc[(track, task, seed, arm, nfe)])
                frontier[arm][str(nfe)] = _independent_iqm(cell_values)
        output[track] = {
            "primary_visual_delta_shortcut_minus_trajectory": _independent_iqm(
                delta.reshape(-1)
            ),
            "positive_favors_trajectory_imf": True,
            "descriptive_hierarchical_bootstrap_interval": [
                float(np.quantile(bootstrap, alpha / 2.0)),
                float(np.quantile(bootstrap, 1.0 - alpha / 2.0)),
            ],
            "task_by_seed_primary_delta": delta.tolist(),
            "task_order": tasks,
            "seed_order": seeds,
            "frontier_iqm": frontier,
        }
    return output


def _validate_aggregate_schema(summary: Mapping[str, Any]) -> None:
    _require_exact_keys(
        summary,
        {
            "schema_version",
            "status",
            "study_role",
            "claim_eligible_for_primary_gate",
            "primary_gate_substitution",
            "protocol_sha256",
            "requested_tracks",
            "task_order",
            "seed_order",
            "expected_input_count",
            "runtime_identity",
            "tracks",
            "inputs",
            "analysis_sha256",
        },
        "pixel aggregate",
    )
    if (
        summary["schema_version"] != "trajectory-imf-pixel-aggregate-v1"
        or summary["status"] != "complete"
        or summary["study_role"] != "descriptive_secondary_hard_visual_evaluation"
        or summary["claim_eligible_for_primary_gate"] is not False
        or summary["primary_gate_substitution"] is not False
    ):
        raise ValueError("pixel aggregate status or claim boundary is invalid")
    _require_sha256(summary["protocol_sha256"], "pixel aggregate protocol digest")
    _require_sha256(summary["analysis_sha256"], "pixel aggregate analysis digest")
    for record in summary["inputs"] if isinstance(summary["inputs"], list) else ():
        _require_exact_keys(
            record,
            {
                "track",
                "task",
                "seed",
                "path",
                "manifest_sha256",
                "seal_sha256",
                "result_sha256",
                "predictions_sha256",
                "artifact_identity_sha256",
            },
            "pixel aggregate input",
        )
        if (
            not isinstance(record["track"], str)
            or not isinstance(record["task"], str)
            or type(record["seed"]) is not int
            or not isinstance(record["path"], str)
            or not Path(record["path"]).is_absolute()
        ):
            raise ValueError("pixel aggregate input identity is malformed")
        for key in (
            "manifest_sha256",
            "seal_sha256",
            "result_sha256",
            "predictions_sha256",
            "artifact_identity_sha256",
        ):
            _require_sha256(record[key], f"pixel aggregate input {key}")


def verify_aggregate(
    path: str | Path, roots: Sequence[str | Path] | None = None
) -> dict[str, Any]:
    """Independently authenticate aggregate inputs and rederive every reported estimate."""

    protocol = load_protocol()
    source = _reject_symlink_chain(path, label="pixel aggregate")
    if not source.is_file() or not stat.S_ISREG(source.lstat().st_mode):
        raise ValueError("pixel aggregate must be a regular non-symlink file")
    summary = _json_load(source)
    _validate_aggregate_schema(summary)
    body = {key: value for key, value in summary.items() if key != "analysis_sha256"}
    if summary["analysis_sha256"] != object_sha256(body):
        raise ValueError("pixel aggregate analysis digest is invalid")
    if summary["protocol_sha256"] != file_sha256(PROTOCOL_PATH):
        raise ValueError("pixel aggregate protocol digest differs from source")
    requested = _requested_tracks(protocol, summary["requested_tracks"])
    hard = protocol["profiles"]["hard"]
    expected = _expected_hard_identities(protocol, requested)
    if (
        summary["requested_tracks"] != requested
        or summary["task_order"] != list(hard["tasks"])
        or summary["seed_order"] != [int(seed) for seed in hard["seeds"]]
        or summary["expected_input_count"] != len(expected)
        or not isinstance(summary["inputs"], list)
        or len(summary["inputs"]) != len(expected)
    ):
        raise ValueError("pixel aggregate registered matrix declaration is invalid")
    supplied = list(roots) if roots is not None else [record["path"] for record in summary["inputs"]]
    if len(supplied) != len(expected):
        raise ValueError("pixel aggregate verifier received the wrong number of roots")
    root_by_identity: dict[tuple[str, str, int], tuple[Path, dict[str, Any]]] = {}
    runtime_identity: dict[str, Any] | None = None
    auc: dict[tuple[str, str, int, str, int], float] = {}
    for root_value in supplied:
        root = _artifact_root(root_value)
        result = verify_artifact(root, recompute_predictions=False, rederive_compute=False)
        identity = (str(result["track"]), str(result["task"]), int(result["seed"]))
        if identity in root_by_identity or identity not in expected:
            raise ValueError("pixel aggregate verifier found a duplicate or unexpected identity")
        root_by_identity[identity] = (root, result)
        current_runtime = _runtime_identity(result["runtime"])
        if runtime_identity is None:
            runtime_identity = current_runtime
        elif current_runtime != runtime_identity:
            raise ValueError("pixel aggregate verifier found heterogeneous runtimes")
        for (arm, nfe), value in _independent_auc_by_cell(root, result).items():
            auc[(*identity, arm, nfe)] = value
    if set(root_by_identity) != expected:
        raise ValueError("pixel aggregate verifier did not receive the exact requested matrix")
    records = {
        (str(record["track"]), str(record["task"]), int(record["seed"])): record
        for record in summary["inputs"]
    }
    ordered_expected = [
        (track, task, int(seed))
        for track in requested
        for task in hard["tasks"]
        for seed in hard["seeds"]
    ]
    embedded_order = [
        (str(record["track"]), str(record["task"]), int(record["seed"]))
        for record in summary["inputs"]
    ]
    if (
        len(records) != len(summary["inputs"])
        or set(records) != expected
        or embedded_order != ordered_expected
    ):
        raise ValueError("pixel aggregate embedded input identities are not exact and unique")
    for identity in expected:
        root, result = root_by_identity[identity]
        expected_record = _input_record(root, result)
        if roots is not None:
            # The digest/identity commitment is relocatable; the original absolute
            # path remains provenance rather than part of artifact identity.
            expected_record["path"] = records[identity]["path"]
        if records[identity] != expected_record:
            raise ValueError(f"pixel aggregate input commitment differs for {identity}")
    if summary["runtime_identity"] != runtime_identity:
        raise ValueError("pixel aggregate runtime identity does not rederive")
    independently_derived = _independent_track_summaries(protocol, requested, auc)
    if not _same_json_number(summary["tracks"], independently_derived, atol=2e-6):
        raise ValueError("pixel aggregate estimates do not independently rederive")
    return summary


__all__ = [
    "ARM_ORDER",
    "PixelDMCAdapter",
    "PixelStep",
    "area_downsample_uint8",
    "aggregate_artifacts",
    "batch_from_starts",
    "collect_pixel_dataset",
    "eligible_window_starts",
    "global_ssim",
    "interquartile_mean",
    "load_protocol",
    "make_batch_schedule",
    "prediction_metrics",
    "preprocess_pixels_jax",
    "replay_pixel_dataset",
    "resolved_config",
    "run_benchmark",
    "select_windows",
    "summarize_predictions",
    "validate_dataset",
    "validate_exploration_actions",
    "validate_protocol",
    "verify_aggregate",
    "verify_artifact",
]
