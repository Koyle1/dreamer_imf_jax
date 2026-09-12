# Trajectory iMF comparison harness

The policy-consistency interventions and their gated Reacher protocol are
documented in [POLICY_CONSISTENCY_VNEXT.md](POLICY_CONSISTENCY_VNEXT.md).

This directory contains the registered comparison between a trajectory-wise
Improved MeanFlow dynamics objective and the paper-derived
Equation-(7)-style shortcut-forcing objective. It also includes Gaussian RSSM,
ordinary-iMF, temporal-consistency, component-ablation, and real-render pixel
controls.

The harness is designed to fail closed:

- profiles, tasks, seeds, budgets, inference NFEs, outcome estimands, and
  practical-effect thresholds are frozen in JSON protocols;
- task × world-model seed is the statistical unit, with actor seeds and
  episodes nested inside it;
- raw schedules, checkpoints, predictions, action traces, compiler evidence,
  runtime identity, source files, and final reports are content-bound;
- pilot selection cannot read confirmatory outcomes;
- controls and pixel tracks are secondary and cannot rescue a failed primary
  superiority gate.

An engineering smoke only validates execution. It is never scientific
evidence. This repository does not claim to reproduce the unreleased official
Dreamer 4 implementation; the shortcut objective is the declared comparator.

Before a claim run, `run_shortcut_stability_study.py` can execute the 2^3
EMA-teacher × bounded-intermediate × support-safe-composition diagnostic on
the six archived pilot stress seeds. Its 48 cells are explicitly engineering
evidence: they gate unstable implementations but cannot support superiority.

The harness is monorepo-only research tooling, not a self-contained PyPI
distribution. Its `pyproject.toml` supports editable imports from this checkout;
the registered scripts, protocols, documents, dependency lock, and cluster
workflow must remain beside the package as committed repository files.

## Local verification

From the repository root, with dependencies installed:

```bash
export PYTHONPATH=imf_dreamer_jax/src:dreamer_imf_comparison

python dreamer_imf_comparison/scripts/verify_trajectory_imf_theory.py
python dreamer_imf_comparison/scripts/verify_neurips_core.py --regressions
python dreamer_imf_comparison/scripts/verify_neurips_controls.py --contract
python dreamer_imf_comparison/scripts/verify_pixel_benchmark.py --contract
```

Run the small nonclaiming matched-objective smoke:

```bash
python dreamer_imf_comparison/scripts/run_matched_objective_benchmark.py all \
  --profile smoke \
  --output /tmp/trajectory-imf-matched-smoke \
  --workspace .
```

The exact cluster environment and stage-by-stage launch procedure are in
`cluster/neurips/README.md`. Submission commands are dry runs unless the
explicit execution flag is supplied.

## Protocols and boundaries

- `MATCHED_OBJECTIVE_BENCHMARK.md` — primary matched-compute design
- `TRAJECTORY_IMF_THEORY.md` — assumptions, theorem, reductions, and limits
- `NEURIPS_CONTROLS.md` — strong baselines and factorial ablations
- `PIXEL_BENCHMARK.md` — real-render long-horizon visual evaluation
- `NEURIPS_READINESS_REPORT.md` — result-bound report; explicitly nonfinal
  until authenticated cluster outputs exist
