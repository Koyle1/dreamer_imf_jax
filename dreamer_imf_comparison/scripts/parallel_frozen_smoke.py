"""CPU read-only exact-checkpoint inference; zero simulator interactions."""

import jax
from embodied.jax import internal

internal.setup(
    platform="cpu",
    compute_dtype="bfloat16",
    transfer_guard=False,
    compilation_cache=False,
)
import numpy as np
from dreamer_imf_compare.parallel_frozen import FrozenModel

base = "/work2/ci72buri-dreamer_imf_neurips/"
imf = FrozenModel(
    base
    + "imf-conditional-study/89dc0d74ec488c59f15e4c5599eca53823e2dfdf/cells/imf/seed_431",
    "imf",
)
cat = FrozenModel(
    base
    + "dreamer-ablation-study/6b3d39484e702bc8b3debea00d0c1b224971f112/cells/categorical/seed_431",
    "categorical",
)
for model in (imf, cat, imf):
    obs = {k: np.zeros(s.shape, s.dtype) for k, s in model.obs_space.items()}
    obs["is_first"] = np.asarray(True)
    carry, feature = model.observe(model.initial(), obs, np.zeros(2, np.float32), 431)
    action = model.action(feature, 91)
    assert action.shape == (2,) and np.isfinite(action).all()
    assert np.all(action >= -1) and np.all(action <= 1)
    for nfe in ((1, 4) if model.arm == "imf" else (1,)):
        f = model.rollout(
            feature[None], np.repeat(action[None, None], 15, axis=1), 917, 2, nfe
        )
        decoded = model.decode(f)
        for x in jax.tree.leaves(decoded):
            assert np.isfinite(np.asarray(x)).all()
    assert model.frozen_digest() == model.before
    print(
        "FROZEN_MODEL_OK",
        model.arm,
        model.feature_dim,
        model.parameter_count,
        model.parameter_counts,
        model.active_dynamics_parameters,
        flush=True,
    )
print("PARALLEL_FROZEN_CHECKPOINT_SMOKE_VERIFIED", flush=True)
