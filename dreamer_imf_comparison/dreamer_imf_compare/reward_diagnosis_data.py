"""Named retained observations and stochastic refiltering, with no env.step."""

import copy
import hashlib
import inspect
from pathlib import Path
import numpy as np
from .parallel_collection import load_raw_history
from .reward_readout_study import equal, sha

GROUPS = ("observation", "history_action", "h", "z", "hz")


def group_features(
    group, features, observation=None, previous=None, actions=None, deter=2048
):
    if group == "h":
        return features[..., :deter]
    if group == "z":
        return features[..., deter:]
    if group == "hz":
        return features
    if group == "observation":
        return observation
    if group == "history_action":
        return np.concatenate((observation, previous, actions), -1)
    raise ValueError("unknown input group")


def flatten(raw):
    return np.concatenate(
        [
            np.asarray(raw[k], np.float32).reshape(-1)
            for k in ("position", "to_target", "velocity")
        ]
    )


def check_branch(data, i, branch, raw):
    if len(raw) != 16 or len(branch["actions"]) != 15:
        raise ValueError("invalid branch length")
    if any(bool(o["is_first"]) or bool(o["is_last"]) for o in raw):
        raise ValueError("reset/terminal inside retained branch")
    obs = np.stack([flatten(o) for o in raw])
    equal(obs[1:], data["observations"][i], "current observation indexing")
    equal(branch["actions"], data["actions"][i], "action indexing")
    equal(branch["rewards"], data["rewards"][i], "reward indexing")
    equal(
        np.asarray([o["reward"] for o in raw[1:]]),
        data["rewards"][i],
        "raw reward indexing",
    )
    equal(branch["features"][0], data["start"][i], "anchor belief")
    equal(branch["features"][1:], data["targets"][i], "posterior indexing")
    return obs[:-1], np.asarray([np.linalg.norm(o["to_target"]) for o in raw[1:]])


def retained_inputs(data, location):
    previous, distance, bindings = [], [], {}
    for i, (episode, anchor, plan) in enumerate(
        zip(data["episode"], data["anchor"], data["plan"])
    ):
        path = Path(location) / f"branch-{episode:03d}-{anchor:03d}-{plan}.npz"
        branch, raw = load_raw_history(path)
        prev, dist = check_branch(data, i, branch, raw)
        previous.append(prev)
        distance.append(dist)
        bindings[str(path)] = sha(path)
    for e in np.unique(data["episode"]):
        path = Path(location) / f"episode-{e:03d}.npz"
        bindings[str(path)] = sha(path)
    return np.stack(previous), np.stack(distance), bindings


def geometry():
    """Construct only the static physics model, never an environment or reset."""
    from dm_control.suite import reacher
    from dm_control.utils import rewards

    xml, assets = reacher.get_model_and_assets()
    physics = reacher.Physics.from_xml_string(xml, assets)
    radius = float(reacher._SMALL_TARGET + physics.named.model.geom_size["finger", 0])
    return dict(
        radius=radius,
        target_radius=float(reacher._SMALL_TARGET),
        finger_radius=float(physics.named.model.geom_size["finger", 0]),
        source_sha256=sha(inspect.getfile(reacher)),
        reward_source_sha256=sha(inspect.getfile(rewards)),
        xml_sha256=hashlib.sha256(
            xml.encode() if isinstance(xml, str) else xml
        ).hexdigest(),
        assets={k: hashlib.sha256(v).hexdigest() for k, v in sorted(assets.items())},
        limitation="Installed environment assets; intermediate native observation was not retained",
    )


def validate_endpoint(distance, reward, radius):
    inside = distance <= radius
    if np.any((reward == 2) & ~inside) or np.any((reward == 0) & inside):
        raise ValueError("geometry incompatible with retained AR2 reward labels")


def posterior_draws(teacher, data, location, indices, draws):
    """Yield (draw, global row indices, features), with shared anchor beliefs."""
    for draw in range(draws):

        def seed(v):
            return int(
                np.random.SeedSequence([int(v), draw, 9127]).generate_state(1)[0]
            )

        for e in np.unique(data["episode"][indices]):
            rows = indices[data["episode"][indices] == e]
            history, obs = load_raw_history(Path(location) / f"episode-{e:03d}.npz")
            carry, _ = teacher.observe(
                teacher.initial(),
                copy.deepcopy(obs[0]),
                history["previous_action"].copy(),
                seed(history["observe_seeds"][0]),
            )
            anchors = set(map(int, data["anchor"][rows]))
            for t in range(1, max(anchors) + 1):
                carry, _ = teacher.observe(
                    carry,
                    copy.deepcopy(obs[t]),
                    history["actions"][t - 1].copy(),
                    seed(history["observe_seeds"][t]),
                )
                if t not in anchors:
                    continue
                selected = rows[data["anchor"][rows] == t]
                features = []
                for i in selected:
                    path = (
                        Path(location) / f'branch-{e:03d}-{t:03d}-{data["plan"][i]}.npz'
                    )
                    branch, raw = load_raw_history(path)
                    state, future = copy.deepcopy(carry), []
                    for j in range(15):
                        state, f = teacher.observe(
                            state,
                            copy.deepcopy(raw[j + 1]),
                            branch["actions"][j].copy(),
                            seed(branch["observe_seeds"][j + 1]),
                        )
                        future.append(f)
                    features.append(future)
                yield draw, selected, np.asarray(features, np.float32)
