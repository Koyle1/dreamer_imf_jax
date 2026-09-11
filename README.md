# Trajectory iMF world models in JAX

This repository contains two deliberately separated pieces:

- `imf_dreamer_jax/` — an importable, functional JAX research library for
  Gaussian, ordinary Improved MeanFlow (iMF), shortcut-forcing, and trajectory
  iMF latent dynamics.
- `dreamer_imf_comparison/` — the frozen matched-compute benchmark, theorem and
  reduction checks, strong controls, real-render pixel evaluation, and Slurm
  workflow used to test trajectory iMF against the declared
  Equation-(7)-style shortcut-forcing baseline.

The code is research software. It is not an implementation or reproduction of
an unreleased Dreamer 4 system, and an engineering smoke test is not evidence
of empirical superiority.

## Install the library

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ./imf_dreamer_jax
```

The comparison harness uses the library directly from the monorepo layout. For
the exact GPU environment, use
`dreamer_imf_comparison/requirements-neurips-cuda12-lock.txt` and the bootstrap
script under `dreamer_imf_comparison/cluster/neurips/`.

## Verify

With the project dependencies installed:

```bash
PYTHONPATH=imf_dreamer_jax/src:dreamer_imf_comparison \
  python -m unittest discover -s imf_dreamer_jax/tests -p 'test_*.py'

PYTHONPATH=imf_dreamer_jax/src:dreamer_imf_comparison \
  python dreamer_imf_comparison/scripts/verify_neurips_core.py --regressions
```

The matched-objective, controls, pixel, theory, and cluster protocols document
their evidence boundaries next to the corresponding runners. Confirmatory
claims are permitted only when the authenticated four-interval gate and the
prespecified practical-effect checks both pass.

## Minimal import

```python
import jax
from imf_dreamer_jax import DreamerConfig, create_agent

config = DreamerConfig(
    observation_shape=(8,),
    action_dim=2,
    prior="imf",
    imf_trajectory_enabled=True,
)
agent = create_agent(config, jax.random.key(0))
```

See `imf_dreamer_jax/README.md` for the public API and
`dreamer_imf_comparison/MATCHED_OBJECTIVE_BENCHMARK.md` for the scientific
comparison contract.

## License

The repository and both contained packages are licensed under Apache-2.0.
