"""Capped, immutable, inference-only branch collection for parallel trajectory iMF.

The environment factory takes an integer seed and returns an adapter exposing
``action_low/high``, ``action_repeat == 2``, and a monotonic ``native_steps``
counter. ``reset()`` returns an observation mapping; ``step(action)`` returns
an object with observation, reward (summed over both native steps), is_last,
and native_steps. Snapshot/restore must include physics, wrapper and RNG state
but must *not* rewind the lifetime native_steps audit counter. Neither reset
nor snapshot/restore may advance physics. The collector fails closed otherwise.

All simulator calls are preceded by durable pessimistic reservations. Failed
or interrupted reservations are never refunded. Use one BudgetLedger across
preflight and collection attempts; reopening it replays and verifies the log.
No automatic retry or resume overwrites prior evidence. Each call creates a
new immutable attempt, including on failure. Successful raw NPZ histories and
the per-transition JSONL journals support independent posterior filtering.
Anchor pickles retain trusted, process-local forensic simulator/belief snapshots;
they are not a portable restore format and must never be loaded from untrusted
sources. Independent reconstruction uses retained episode seeds/raw histories.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import threading
import uuid

import numpy as np

EPISODES = 64
DECISIONS = 500
ACTION_REPEAT = 2
HORIZON = 15
ANCHORS = (100, 200, 300, 400)
NATIVE_CAP = 100_000
PREFLIGHT_CAP = 4_800
BASE_SEED = 431_700
MODES = ("frozen_policy", "policy_20pct_uniform_replacement")
PLAN_NAMES = ("zero", "held", "uniform", "suffix")


class BudgetExceeded(RuntimeError):
    """A durable reservation would exceed the authorized simulation cap."""


class CollectionIntegrityError(RuntimeError):
    """The adapter, model, artifact, or ledger violates its frozen contract."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if int(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


class BudgetLedger:
    """Exclusive, fsync-backed hash-chained ledger; reservations never decrease.

    ``charged`` is the conservative bound, ``actual`` the accounted successful
    and failed calls. Pending reservations remain charged after a restart.
    Existing ledgers must be reopened with the same limit. No lock stealing.
    This detects malformed/partial logs, not deliberate deletion by an owner.
    """

    def __init__(self, path, *, limit=NATIVE_CAP):
        self.path = Path(path)
        self.limit = _integer(limit, "limit", minimum=1)
        if self.limit > NATIVE_CAP:
            raise ValueError("limit cannot exceed the 100000 native-step cap")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        self._fd = os.open(self.path, flags, 0o600)
        self._closed = False
        self._poisoned = False
        self.charged = 0
        self.actual = 0
        self.preflight_charged = 0
        self._records = 0
        self._digest = "0" * 64
        self._pending = {}
        self._mutex = threading.RLock()
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with os.fdopen(os.dup(self._fd), "r", encoding="utf-8") as stream:
                stream.seek(0)
                for line in stream:
                    if not line.endswith("\n"):
                        raise CollectionIntegrityError("partial budget ledger record")
                    event = json.loads(line)
                    digest = event.pop("hash")
                    if (
                        event.get("seq") != self._records
                        or event.get("previous") != self._digest
                    ):
                        raise CollectionIntegrityError("broken budget ledger chain")
                    if hashlib.sha256(_json(event).encode()).hexdigest() != digest:
                        raise CollectionIntegrityError("budget ledger hash mismatch")
                    self._consume(event)
                    self._digest = digest
                    self._records += 1
            if not self._records:
                self._append(
                    {
                        "kind": "header",
                        "version": 1,
                        "limit": self.limit,
                        "preflight_limit": PREFLIGHT_CAP,
                    }
                )
            _fsync_directory(self.path.parent)
        except BaseException:
            self.close()
            raise

    def _consume(self, event):
        kind = event.get("kind")
        if self._records == 0:
            if (
                kind != "header"
                or event.get("version") != 1
                or event.get("limit") != self.limit
                or event.get("preflight_limit") != PREFLIGHT_CAP
            ):
                raise CollectionIntegrityError("budget ledger header/limit mismatch")
            return
        if kind == "reserve":
            count = _integer(event["count"], "reservation", minimum=1)
            category = event["category"]
            if (
                category not in ("main", "preflight")
                or event["id"] in self._pending
                or event["id"] != f"r{event['seq']:08d}"
            ):
                raise CollectionIntegrityError("invalid budget reservation")
            self.charged += count
            self.preflight_charged += count if category == "preflight" else 0
            if self.charged > self.limit or self.preflight_charged > PREFLIGHT_CAP:
                raise CollectionIntegrityError("over-budget ledger")
            self._pending[event["id"]] = count
        elif kind == "finish":
            count = self._pending.pop(event["id"], None)
            actual = _integer(event["actual"], "actual")
            if count is None or actual > count:
                raise CollectionIntegrityError("unreserved native steps")
            if event["outcome"] not in ("ok", "failed"):
                raise CollectionIntegrityError("invalid budget outcome")
            self.actual += actual
        else:
            raise CollectionIntegrityError("unexpected budget ledger event")

    def _append(self, fields):
        if self._closed or self._poisoned:
            raise CollectionIntegrityError("budget ledger closed or poisoned")
        event = dict(fields, seq=self._records, previous=self._digest)
        digest = hashlib.sha256(_json(event).encode()).hexdigest()
        payload = (_json(dict(event, hash=digest)) + "\n").encode()
        try:
            if os.write(self._fd, payload) != len(payload):
                raise OSError("partial budget ledger append")
            os.fsync(self._fd)
            self._consume(event)
            self._records += 1
            self._digest = digest
        except BaseException:
            self._poisoned = True
            raise

    @property
    def remaining(self):
        return self.limit - self.charged

    @property
    def pending(self):
        return dict(self._pending)

    def reserve(self, count, reason, *, category="main"):
        with self._mutex:
            count = _integer(count, "count", minimum=1)
            if (
                category not in ("main", "preflight")
                or not isinstance(reason, str)
                or not reason
            ):
                raise ValueError(
                    "reservation needs a valid category and nonempty reason"
                )
            if self.charged + count > self.limit:
                raise BudgetExceeded("native-step reservation exceeds total cap")
            if (
                category == "preflight"
                and self.preflight_charged + count > PREFLIGHT_CAP
            ):
                raise BudgetExceeded("preflight reservation exceeds 4800-step cap")
            identity = f"r{self._records:08d}"
            self._append(
                dict(
                    kind="reserve",
                    id=identity,
                    count=count,
                    category=category,
                    reason=reason,
                )
            )
            return identity

    def finish(self, identity, actual, *, outcome="ok"):
        with self._mutex:
            actual = _integer(actual, "actual")
            if (
                identity not in self._pending
                or actual > self._pending[identity]
                or outcome not in ("ok", "failed")
            ):
                raise CollectionIntegrityError(
                    "invalid or overrun reservation completion"
                )
            self._append(
                dict(kind="finish", id=identity, actual=actual, outcome=outcome)
            )

    def close(self):
        if not self._closed:
            os.close(self._fd)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_artifact(path, writer):
    """Publish with link-not-replace: an existing destination is never touched."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path, value):
    _atomic_artifact(path, lambda stream: stream.write((_json(value) + "\n").encode()))


def _write_npz(path, values):
    for name, value in values.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise CollectionIntegrityError(f"object array forbidden: {name}")
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise CollectionIntegrityError(f"nonfinite array: {name}")
    _atomic_artifact(path, lambda stream: np.savez_compressed(stream, **values))


class _Journal:
    def __init__(self, path):
        self.stream = Path(path).open("x", encoding="utf-8")

    def append(self, event):
        self.stream.write(_json(event) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


def episode_design():
    """Whole-episode 40/12/12 split; every split has equal mode counts."""
    return [
        dict(
            episode=i,
            seed=BASE_SEED + i,
            mode=i % 2,
            split=0 if i < 40 else 1 if i < 52 else 2,
        )
        for i in range(EPISODES)
    ]


def duplicate_design():
    """Sixteen preregistered, mode/anchor/plan-balanced replay checks."""
    return [(4 * i + (i % 2), ANCHORS[i % 4], (i // 4) % 4) for i in range(16)]


def _seed(*parts):
    return int(
        np.random.SeedSequence([BASE_SEED, *map(int, parts)]).generate_state(1)[0]
    )


def action_plans(low, high, held_action, *, episode, anchor):
    """Exogenous plans; suffix plan is bitwise equal to uniform for steps 1..5."""
    low, high = np.asarray(low, np.float32), np.asarray(high, np.float32)
    held = np.asarray(held_action, np.float32)
    if (
        low.ndim != 1
        or high.shape != low.shape
        or not low.size
        or not np.isfinite(low).all()
        or not np.isfinite(high).all()
        or not np.all(low < high)
        or np.any(low > 0)
        or np.any(high < 0)
        or held.shape != low.shape
        or not np.isfinite(held).all()
        or np.any(held < low)
        or np.any(held > high)
    ):
        raise ValueError("invalid action bounds or held action")
    rng = np.random.default_rng(_seed(10, episode, anchor))
    uniform = rng.uniform(low, high, (HORIZON, len(low))).astype(np.float32)
    suffix = uniform.copy()
    suffix[5:] = rng.uniform(low, high, (HORIZON - 5, len(low))).astype(np.float32)
    return np.stack(
        (np.zeros_like(uniform), np.broadcast_to(held, uniform.shape), uniform, suffix)
    ).astype(np.float32)


def _observation(obs, keys):
    if not isinstance(obs, Mapping) or any(key not in obs for key in keys):
        raise CollectionIntegrityError(
            "observation must contain every model.obs_keys entry"
        )
    raw = {}
    for key, value in obs.items():
        array = np.asarray(value)
        if (
            not isinstance(key, str)
            or array.dtype.kind not in "biuf"
            or not np.isfinite(array).all()
        ):
            raise CollectionIntegrityError(
                "raw observations must be finite numeric mappings"
            )
        raw[key] = array.copy()
    flat = np.concatenate([np.asarray(raw[k], np.float32).reshape(-1) for k in keys])
    if not flat.size or not np.isfinite(flat).all():
        raise CollectionIntegrityError("invalid flattened observation")
    return raw, flat


def _raw_json(raw):
    return {
        k: dict(dtype=v.dtype.str, shape=list(v.shape), values=v.tolist())
        for k, v in raw.items()
    }


def _feature(value):
    value = np.asarray(value, np.float32)
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise CollectionIntegrityError(
            "posterior feature must be a finite nonempty vector"
        )
    return value.copy()


def _observe(model, carry, raw, action, seed):
    # An adapter may normalize its inputs in place. Never let that alter the
    # original replay evidence or the action that actually reached physics.
    carry, feature = model.observe(carry, copy.deepcopy(raw), action.copy(), seed)
    return carry, _feature(feature)


def _counter(env):
    return _integer(env.native_steps, "environment native_steps")


def _no_steps(env, function, *args):
    before = _counter(env)
    result = function(*args)
    if _counter(env) != before:
        raise CollectionIntegrityError(
            "reset/snapshot/restore advanced unreserved physics"
        )
    return result


def _step(env, action, budget, category, journal, context, observe_seed, keys):
    before = _counter(env)
    reservation = budget.reserve(ACTION_REPEAT, context, category=category)
    try:
        step = env.step(action.copy())
        actual = _counter(env) - before
        if (
            actual != ACTION_REPEAT
            or _integer(step.native_steps, "step native_steps") != actual
        ):
            raise CollectionIntegrityError(
                "step counter does not match two native steps"
            )
        raw, flat = _observation(step.observation, keys)
        reward = float(step.reward)
        if not np.isfinite(reward):
            raise CollectionIntegrityError("nonfinite reward")
        journal.append(
            dict(
                kind="step",
                context=context,
                reservation=reservation,
                action=action.tolist(),
                observation=_raw_json(raw),
                reward=reward,
                is_last=bool(step.is_last),
                native_steps=actual,
                observe_seed=observe_seed,
            )
        )
        budget.finish(reservation, actual)
        return raw, flat, reward, bool(step.is_last)
    except BaseException as error:
        actual = _counter(env) - before
        if reservation in budget.pending and 0 <= actual <= ACTION_REPEAT:
            budget.finish(reservation, actual, outcome="failed")
        journal.append(
            dict(
                kind="failure",
                context=context,
                reservation=reservation,
                actual_native_steps=actual,
                error=type(error).__name__,
            )
        )
        raise


def _history_arrays(
    raw, observations, actions, rewards, features, seeds, lasts, previous_action
):
    names = list(raw[0])
    if any(list(item) != names for item in raw):
        raise CollectionIntegrityError("raw observation key schema changed")
    values = dict(
        observations=np.asarray(observations, np.float32),
        actions=np.asarray(actions, np.float32),
        rewards=np.asarray(rewards, np.float64),
        features=np.asarray(features, np.float32),
        observe_seeds=np.asarray(seeds, np.uint32),
        is_last=np.asarray(lasts, np.bool_),
        previous_action=np.asarray(previous_action, np.float32),
        raw_keys=np.asarray(names, np.str_),
    )
    for i, name in enumerate(names):
        values[f"raw_{i:03d}"] = np.stack([item[name] for item in raw])
    return values


def load_raw_history(path):
    """Read without pickle; returns all original observation fields at every t."""
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    names = values["raw_keys"].tolist()
    count = len(values["observations"])
    observations = [
        {name: values[f"raw_{i:03d}"][t].copy() for i, name in enumerate(names)}
        for t in range(count)
    ]
    return values, observations


def reconstruct_features(model, episode_path, *, anchor=None, branch_path=None):
    """Filter full raw prefix using another frozen teacher's own initial carry.

    An anchor indexes the posterior *after* that many executed base actions.
    Returned branch trajectory excludes the anchor and has H future features.
    """
    base, raw = load_raw_history(episode_path)
    stop = len(raw) - 1 if anchor is None else _integer(anchor, "anchor")
    if stop >= len(raw):
        raise ValueError("anchor outside base episode")
    carry = model.initial()
    carry, start = _observe(
        model, carry, raw[0], base["previous_action"], int(base["observe_seeds"][0])
    )
    values = [start]
    for t in range(stop):
        carry, value = _observe(
            model,
            carry,
            raw[t + 1],
            base["actions"][t],
            int(base["observe_seeds"][t + 1]),
        )
        values.append(value)
    if branch_path is None:
        return carry, np.stack(values)
    branch, branch_raw = load_raw_history(branch_path)
    if any(
        k not in branch_raw[0] or not np.array_equal(v, branch_raw[0][k])
        for k, v in raw[stop].items()
    ):
        raise CollectionIntegrityError(
            "branch initial observation differs from base anchor"
        )
    if not np.array_equal(
        branch["previous_action"],
        base["actions"][stop - 1] if stop else base["previous_action"],
    ):
        raise CollectionIntegrityError(
            "branch previous action differs from base anchor"
        )
    future = []
    for t, action in enumerate(branch["actions"]):
        carry, value = _observe(
            model, carry, branch_raw[t + 1], action, int(branch["observe_seeds"][t + 1])
        )
        future.append(value)
    return values[-1], np.stack(future)


@dataclass(frozen=True)
class CollectionResult:
    attempt: Path
    dataset: Path
    manifest: Path
    charged_steps: int
    actual_steps: int


def collect(root, model, env_factory, *, budget, preflight=False):
    """Collect all 1024 branches or a 1510-step single-episode preflight.

    Main collection: 64*500*2 + 64*4*4*15*2 + 16*15*2 = 95200.
    Even episodes use the frozen policy; odd episodes independently replace
    each policy action with a uniform draw with probability .2. Held branches
    repeat a new policy draw at the anchor, not the last executed base action.
    Preflight uses episode 0, all its branches, and duplicate (0,100,0).
    Output paths are returned, never assumed to overwrite a previous attempt.
    ``budget`` must be a shared open BudgetLedger, not a fresh integer cap.
    """
    if not isinstance(budget, BudgetLedger):
        raise TypeError("budget must be the shared BudgetLedger for all attempts")
    if not isinstance(preflight, bool):
        raise TypeError("preflight must be bool")
    keys = tuple(model.obs_keys)
    if (
        not keys
        or len(set(keys)) != len(keys)
        or any(not isinstance(k, str) for k in keys)
    ):
        raise ValueError("model.obs_keys must be ordered unique strings")
    digest = str(model.frozen_digest())
    design = episode_design()[:1] if preflight else episode_design()
    duplicates = {(0, 100, 0)} if preflight else set(duplicate_design())
    category = "preflight" if preflight else "main"
    required = (
        len(design) * (DECISIONS + len(ANCHORS) * 4 * HORIZON) * ACTION_REPEAT
        + len(duplicates) * HORIZON * ACTION_REPEAT
    )
    if budget.remaining < required or (
        preflight and budget.preflight_charged + required > PREFLIGHT_CAP
    ):
        raise BudgetExceeded(
            f"insufficient remaining budget for complete {category}: {required}"
        )
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    attempt = root / f"{category}-{uuid.uuid4().hex}"
    attempt.mkdir()
    initial_charged, initial_actual = budget.charged, budget.actual
    _write_json(
        attempt / "started.json",
        dict(
            version=1,
            category=category,
            model_digest=digest,
            obs_keys=list(keys),
            budget_path=str(budget.path.resolve()),
            charged_before=initial_charged,
            design=design,
            duplicates=sorted(duplicates),
            required_native_steps=required,
            modes=MODES,
            plans=PLAN_NAMES,
            anchors=ANCHORS,
            horizon=HORIZON,
            decisions=DECISIONS,
            action_repeat=ACTION_REPEAT,
            uniform_replacement_probability=0.2,
            held_action_source="new frozen-policy draw at anchor posterior",
            base_seed=BASE_SEED,
        ),
    )
    rows, duplicate_rows, duplicate_checks, snapshot_artifacts = [], [], [], []
    try:
        for entry in design:
            eid, seed = entry["episode"], entry["seed"]
            env = env_factory(seed)
            journal = None
            try:
                low, high = np.asarray(env.action_low, np.float32), np.asarray(
                    env.action_high, np.float32
                )
                if _integer(env.action_repeat, "action_repeat") != ACTION_REPEAT:
                    raise CollectionIntegrityError(
                        "collection requires action repeat two"
                    )
                zero = np.zeros_like(low)
                action_plans(low, high, zero, episode=eid, anchor=0)  # validates bounds
                journal = _Journal(attempt / f"episode-{eid:03d}.jsonl")
                raw0, flat0 = _observation(_no_steps(env, env.reset), keys)
                obs_seed = _seed(20, eid, 0)
                journal.append(
                    dict(
                        kind="reset",
                        episode=eid,
                        seed=seed,
                        observation=_raw_json(raw0),
                        previous_action=zero.tolist(),
                        observe_seed=obs_seed,
                    )
                )
                carry, feature = _observe(model, model.initial(), raw0, zero, obs_seed)
                raw, flats, acts, rewards, features, seeds, lasts, action_seeds = (
                    [raw0],
                    [flat0],
                    [],
                    [],
                    [feature],
                    [obs_seed],
                    [],
                    [],
                )
                policy_actions, replaced, replacement_seeds = [], [], []
                for decision in range(1, DECISIONS + 1):
                    action_seed = _seed(30, eid, decision)
                    policy_action = np.asarray(
                        model.action(feature.copy(), action_seed), np.float32
                    )
                    if (
                        policy_action.shape != low.shape
                        or not np.isfinite(policy_action).all()
                        or np.any(policy_action < low)
                        or np.any(policy_action > high)
                    ):
                        raise CollectionIntegrityError(
                            "policy action outside declared bounds"
                        )
                    replacement_seed = _seed(35, eid, decision)
                    replacement_rng = np.random.default_rng(replacement_seed)
                    replace = entry["mode"] == 1 and replacement_rng.random() < 0.2
                    action = (
                        replacement_rng.uniform(low, high).astype(np.float32)
                        if replace
                        else policy_action.copy()
                    )
                    if (
                        action.shape != low.shape
                        or not np.isfinite(action).all()
                        or np.any(action < low)
                        or np.any(action > high)
                    ):
                        raise CollectionIntegrityError(
                            "base action outside declared bounds"
                        )
                    obs_seed = _seed(20, eid, decision)
                    new_raw, flat, reward, last = _step(
                        env,
                        action,
                        budget,
                        category,
                        journal,
                        f"episode/{eid}/decision/{decision}",
                        obs_seed,
                        keys,
                    )
                    if last and decision != DECISIONS:
                        raise CollectionIntegrityError(
                            "base episode terminated before 500 decisions"
                        )
                    carry, feature = _observe(model, carry, new_raw, action, obs_seed)
                    raw.append(new_raw)
                    flats.append(flat)
                    acts.append(action.copy())
                    rewards.append(reward)
                    features.append(feature)
                    seeds.append(obs_seed)
                    lasts.append(last)
                    action_seeds.append(action_seed)
                    policy_actions.append(policy_action.copy())
                    replaced.append(replace)
                    replacement_seeds.append(replacement_seed)
                    if decision not in ANCHORS:
                        continue
                    snapshot = _no_steps(env, env.snapshot)
                    anchor_carry = copy.deepcopy(carry)
                    held_action = np.asarray(
                        model.action(feature.copy(), _seed(31, eid, decision)),
                        np.float32,
                    )
                    plans = action_plans(
                        low, high, held_action, episode=eid, anchor=decision
                    )
                    # Serialize before running any branch so the forensic
                    # anchor cannot be contaminated by later mutable carries.
                    # Simulator identity/reset-generation checks remain intact
                    # inside its opaque snapshot; this is not a restore API.
                    forensic = dict(
                        version=1,
                        kind="process-local-forensic-anchor",
                        portable_restore=False,
                        trusted_pickle_only=True,
                        reconstruction="recreate episode from seed and filter retained raw histories",
                        episode=eid,
                        episode_seed=seed,
                        anchor=decision,
                        split=entry["split"],
                        mode=entry["mode"],
                        model_digest=digest,
                        obs_keys=keys,
                        simulator_snapshot=copy.deepcopy(snapshot),
                        belief=copy.deepcopy(anchor_carry),
                        start=feature.copy(),
                        raw_observation=copy.deepcopy(new_raw),
                        observation=flat.copy(),
                        reward=reward,
                        previous_action=action.copy(),
                        action_plans=plans.copy(),
                        plan_names=PLAN_NAMES,
                        observe_seed=obs_seed,
                        policy_action_seed=action_seed,
                        held_action_seed=_seed(31, eid, decision),
                        branch_observe_seeds=np.asarray(
                            [_seed(40, eid, decision, t) for t in range(HORIZON)],
                            np.uint32,
                        ),
                        native_steps_at_capture=_counter(env),
                    )
                    snapshot_name = f"anchor-{eid:03d}-{decision:03d}.pkl"
                    _atomic_artifact(
                        attempt / snapshot_name,
                        lambda stream: pickle.dump(
                            forensic, stream, protocol=pickle.HIGHEST_PROTOCOL
                        ),
                    )
                    snapshot_artifacts.append(snapshot_name)
                    for plan_id, planned_actions in enumerate(plans):

                        def run_branch(is_duplicate=False):
                            _no_steps(env, env.restore, snapshot)
                            branch_carry = copy.deepcopy(anchor_carry)
                            suffix = "-duplicate" if is_duplicate else ""
                            stem = f"branch-{eid:03d}-{decision:03d}-{plan_id}{suffix}"
                            branch_journal = _Journal(attempt / f"{stem}.jsonl")
                            branch_raw, branch_flats = [new_raw], [flat]
                            (
                                branch_features,
                                branch_rewards,
                                branch_lasts,
                                branch_seeds,
                            ) = ([feature], [], [], [obs_seed])
                            try:
                                branch_journal.append(
                                    dict(
                                        kind="anchor",
                                        episode=eid,
                                        anchor=decision,
                                        plan=plan_id,
                                        duplicate=is_duplicate,
                                        observation=_raw_json(new_raw),
                                        previous_action=action.tolist(),
                                        observe_seed=obs_seed,
                                    )
                                )
                                for t, planned in enumerate(planned_actions):
                                    # Same filter randomness for paired plans and exact duplicate replay.
                                    branch_seed = _seed(40, eid, decision, t)
                                    braw, bflat, breward, blast = _step(
                                        env,
                                        planned,
                                        budget,
                                        category,
                                        branch_journal,
                                        f"{stem}/decision/{t + 1}",
                                        branch_seed,
                                        keys,
                                    )
                                    if blast:
                                        raise CollectionIntegrityError(
                                            "branch terminated before fixed horizon"
                                        )
                                    branch_carry, bfeature = _observe(
                                        model, branch_carry, braw, planned, branch_seed
                                    )
                                    branch_raw.append(braw)
                                    branch_flats.append(bflat)
                                    branch_features.append(bfeature)
                                    branch_rewards.append(breward)
                                    branch_lasts.append(blast)
                                    branch_seeds.append(branch_seed)
                                history = _history_arrays(
                                    branch_raw,
                                    branch_flats,
                                    planned_actions,
                                    branch_rewards,
                                    branch_features,
                                    branch_seeds,
                                    branch_lasts,
                                    action,
                                )
                                _write_npz(attempt / f"{stem}.npz", history)
                                row = dict(
                                    episode=eid,
                                    split=entry["split"],
                                    mode=entry["mode"],
                                    anchor=decision,
                                    plan=plan_id,
                                    start=feature.copy(),
                                    actions=planned_actions.copy(),
                                    targets=np.stack(branch_features[1:]),
                                    observations=np.stack(branch_flats[1:]),
                                    rewards=np.asarray(branch_rewards, np.float64),
                                    initial_observation=flat.copy(),
                                )
                                return row, history
                            finally:
                                branch_journal.close()

                        row, row_history = run_branch()
                        rows.append(row)
                        if (eid, decision, plan_id) in duplicates:
                            duplicate, duplicate_history = run_branch(True)
                            duplicate_rows.append(duplicate)
                            equal = all(
                                np.array_equal(row[name], duplicate[name])
                                for name in (
                                    "actions",
                                    "targets",
                                    "observations",
                                    "rewards",
                                    "start",
                                )
                            )
                            equal = equal and all(
                                np.array_equal(value, duplicate_history[name])
                                for name, value in row_history.items()
                            )
                            duplicate_checks.append(
                                dict(
                                    episode=eid,
                                    anchor=decision,
                                    plan=plan_id,
                                    exact=equal,
                                )
                            )
                            if not equal:
                                raise CollectionIntegrityError(
                                    "duplicate replay differed after snapshot restoration"
                                )
                    _no_steps(env, env.restore, snapshot)
                    carry = anchor_carry
                history = _history_arrays(
                    raw, flats, acts, rewards, features, seeds, lasts, zero
                )
                history.update(
                    action_seeds=np.asarray(action_seeds, np.uint32),
                    episode=np.int32(eid),
                    policy_actions=np.asarray(policy_actions, np.float32),
                    uniform_replaced=np.asarray(replaced, np.bool_),
                    replacement_seeds=np.asarray(replacement_seeds, np.uint32),
                    split=np.int8(entry["split"]),
                    mode=np.int8(entry["mode"]),
                    seed=np.uint32(seed),
                )
                _write_npz(attempt / f"episode-{eid:03d}.npz", history)
            finally:
                try:
                    if journal is not None:
                        journal.close()
                finally:
                    env.close()
        if str(model.frozen_digest()) != digest:
            raise CollectionIntegrityError(
                "frozen teacher parameters changed during inference"
            )

        def arrays(items):
            return {
                name: np.asarray(
                    [item[name] for item in items],
                    dtype=(
                        np.int32
                        if name in ("episode", "anchor")
                        else (
                            np.int8
                            if name in ("split", "plan", "mode")
                            else np.float64 if name == "rewards" else np.float32
                        )
                    ),
                )
                for name in items[0]
            }

        dataset, duplicate_path = attempt / "dataset.npz", attempt / "duplicates.npz"
        _write_npz(dataset, arrays(rows))
        _write_npz(duplicate_path, arrays(duplicate_rows))
        charged, actual = (
            budget.charged - initial_charged,
            budget.actual - initial_actual,
        )
        if charged != required or actual != required:
            raise CollectionIntegrityError(
                "collection native-step total differs from fixed design"
            )
        if len(snapshot_artifacts) != len(design) * len(ANCHORS):
            raise CollectionIntegrityError("incomplete anchor snapshot coverage")
        files = sorted(p for p in attempt.iterdir() if p.is_file())
        hashes = {}
        for path in files:
            with path.open("rb") as stream:
                hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest = attempt / "manifest.json"
        _write_json(
            manifest,
            dict(
                version=1,
                complete=True,
                category=category,
                rows=len(rows),
                duplicate_rows=len(duplicate_rows),
                duplicate_checks=duplicate_checks,
                episode_counts=[sum(e["split"] == s for e in design) for s in range(3)],
                charged_steps=charged,
                actual_steps=actual,
                total_budget_charged=budget.charged,
                total_budget_actual=budget.actual,
                unsettled_reservations=budget.pending,
                total_budget_actual_is_lower_bound=bool(budget.pending),
                model_digest=digest,
                obs_keys=list(keys),
                snapshot_count=len(snapshot_artifacts),
                snapshot_artifacts=snapshot_artifacts,
                snapshot_format="trusted process-local forensic pickle; not portable restore",
                dataset=dataset.name,
                duplicates=duplicate_path.name,
                sha256=hashes,
            ),
        )
        return CollectionResult(attempt, dataset, manifest, charged, actual)
    except BaseException as error:
        _write_json(
            attempt / "failure.json",
            dict(
                complete=False,
                error=type(error).__name__,
                detail=str(error),
                charged_steps=budget.charged - initial_charged,
                actual_steps=budget.actual - initial_actual,
                pending=budget.pending,
            ),
        )
        raise
