# Matched strong controls and trajectory-iMF ablations

This package is the registered secondary-control layer for the trajectory-iMF study. It is deliberately separate from the two-arm primary analysis: neither these runs nor their unadjusted descriptive intervals can change, replace, rescue, or fail the four-interval superiority gate in `matched_objective_protocol.json`.

## Exact comparison family

The protocol expands to 33 arms:

- the paper-derived Dreamer-4-Equation-7-style shortcut-forcing reimplementation at four field evaluations;
- ordinary conditional iMF on one-step posterior transitions, with no trajectory schedule or corrupted-history training;
- a native diagonal-Gaussian RSSM prior;
- a one-NFE ordinary-iMF control with stopped consistency between one full transport step and two half steps;
- an exploratory theorem-aligned trajectory arm that adds a prespecified weight-0.25 raw endpoint-certificate slice with `r = 0`, independent per-token `s ~ Uniform[0,1]`, and unadapted squared average-field and boundary-velocity regressions;
- 28 trajectory-iMF cells from the complete `2 × 2 × (2³ − 1)` factorial.

The factorial crosses query/history noise coupling (`shared` or `independent`), query/history time relation (`separate` or tied on positions that the selected component corrupts), and every nonempty subset of clean-context, corrupted-context, and future-suffix training. Enabled context components receive equal probability. Exact clean prefixes remain clean in the tied-time cells, preserving the meaning of the component factors.

The arm scope was frozen before any control outcome: smoke and development execute the complete 33-arm registry, while confirmatory controls execute only six prespecified arms (`shortcut_forcing`, `ordinary_imf`, `gaussian_rssm`, `temporal_increment_imf`, `trajectory_endpoint_certificate`, and `trajectory_imf`). The 27 nonproposed factorial cells are development-only mechanism ablations. This keeps the complete factorial available without turning a 19,806-cell confirmatory sweep into post-hoc inferential fishing. Exact matrix sizes are 100 cells for smoke, 938 for development, and 3,606 for confirmatory.

`temporal_increment_imf` is only a simple MeLISA-inspired temporal-increment consistency control. It is not a reproduction of MeLISA and must never be named as one.

## What is held fixed

The runner imports rather than reimplements the parent harness's:

- DM Control collection and whole-episode train/test split;
- task and world-model seeds;
- contiguous minibatch schedule and burn-in mask;
- namespaced objective PRNG keys;
- full Dreamer configuration template, dimensions, optimizer and precision;
- free-running evaluation windows, base noise and rollout statistics;
- actor/critic initialization, imagination budget, environment seeds and normalized returns;
- compiler cost analysis, runtime fingerprinting, checkpoint format and replay checks.

Every trajectory factorial cell has the same parameter tree, initialization, sampler, NFE and actor code. Only its five registered schedule inputs change. The proposed cell is `shared` noise, `separate` times, and all three context components. A reduction test requires that this cell's loss and gradients agree with the canonical trajectory-iMF implementation. The endpoint certificate is a separate statically compiled executable, so its extra regression is neither evaluated nor charged to any zero-weight factorial arm.

The Gaussian and ordinary-iMF priors necessarily have different prior/recurrent interfaces. Their total and active parameter counts, training FLOPs, actor FLOPs and inference FLOPs are therefore reported rather than hidden. Both equal-update and nearest-integer equal-compiler-FLOP allocations are frozen. The engineering smoke executes only equal updates; development and confirmatory jobs use the tracks specified in the protocol.

## Selection and inference isolation

Only three strong-control families are tuned: ordinary iMF, Gaussian RSSM and temporal-increment iMF. Their finite candidate grids and deterministic joint rollout/actor rank rule are frozen in `neurips_controls_protocol.json`. Selection uses only `reacher_easy` and `pendulum_swingup`, only the equal-update track, and resolves both within-metric and final-score ties lexicographically. Factorial settings are never tuned.

Confirmatory controls require both:

1. the parent benchmark's completed, verified pilot `hpo_selection.json`; and
2. this controls benchmark's completed, verified development `controls_selection.json`.

When a directory is supplied to the CLI, every retained artifact in that pilot or development root is independently revalidated before its selection is read. The controls selection also binds the development analysis, matrix, and exact source digest; a selection made from different code is rejected. A direct JSON-file input still receives strict schema, digest, frozen-grid, and source-identity validation, but directory inputs are preferred because they additionally prove the result-to-selection derivation.

All control intervals are unadjusted 95% task-stratified paired percentile intervals. They are labeled descriptive, contain no superiority boolean, and do not call the parent's primary decision routine. Actor seeds are averaged within each task-by-world-model-seed unit; they are not treated as independent replicates.

## Commands

From the workspace root with the project environment active:

```bash
export PYTHONPATH=imf_dreamer_jax/src:dreamer_imf_comparison
PY=/private/tmp/trajectory-imf-venv/bin/python
```

Contract and unit verification:

```bash
$PY dreamer_imf_comparison/scripts/verify_neurips_controls.py --contract
$PY dreamer_imf_comparison/scripts/verify_neurips_controls.py --tests
```

Engineering smoke across all 33 arms:

```bash
$PY dreamer_imf_comparison/scripts/run_neurips_controls.py all \
  --profile smoke \
  --output /private/tmp/trajectory-imf-neurips-controls-smoke
```

Development selection:

```bash
$PY dreamer_imf_comparison/scripts/run_neurips_controls.py all \
  --profile development \
  --output /work2/$USER/trajectory-imf/neurips-controls-development
```

Confirmatory descriptive controls, after both development studies verify:

```bash
$PY dreamer_imf_comparison/scripts/run_neurips_controls.py all \
  --profile confirmatory \
  --parent-selection /work2/$USER/trajectory-imf/matched-objective-pilot \
  --controls-selection /work2/$USER/trajectory-imf/neurips-controls-development \
  --output /work2/$USER/trajectory-imf/neurips-controls-confirmatory
```

For a scheduler array, freeze once, run each canonical dataset cell once, then dispatch exact controls cells in dependency order (compute, world, then rollout/actor). The selectors resolve against the immutable matrices and refuse zero or multiple matches:

```bash
$PY dreamer_imf_comparison/scripts/run_neurips_controls.py freeze \
  --profile smoke --output /work2/$USER/trajectory-imf/controls-smoke

$PY dreamer_imf_comparison/scripts/run_neurips_controls.py run \
  --profile smoke --output /work2/$USER/trajectory-imf/controls-smoke \
  --dataset-cell-id DATASET_CELL_ID

$PY dreamer_imf_comparison/scripts/run_neurips_controls.py run \
  --profile smoke --output /work2/$USER/trajectory-imf/controls-smoke \
  --cell-id CONTROLS_CELL_ID
```

Dataset jobs may equivalently use the paired `--dataset-task TASK --dataset-world-model-seed SEED` selector. Cell ids come from `parent_dataset_matrix.json` and `matrix.json`. Repeating an exact cell invocation performs the normal artifact/checkpoint validation and resume behavior; it does not launch any sibling cell.

Interrupted unfrozen runs are resumable by repeating `run` or `all`; completed cells are independently validated before they are skipped. Once `FINALIZED.json` exists, mutation commands refuse the root and only full verification is allowed.

```bash
$PY dreamer_imf_comparison/scripts/run_neurips_controls.py verify \
  --output /work2/$USER/trajectory-imf/neurips-controls-confirmatory
```

## Retained evidence

The frozen root retains both protocols, a controls-specific source manifest, Git identity, dependency lock, environment fingerprint, exact matrix, canonical parent dataset artifacts, resolved configs, content-addressed optimized and unoptimized compiler IR, raw compiler cost analyses, update allocations, minibatch and objective-key schedules, world and actor checkpoints, predictive draws, actions, rewards, continuations, terminal flags, unit-level metrics, descriptive bootstrap output, a generated report, and an exact-file artifact manifest.

Verification rederives schedules and evaluation windows, recomputes all rollout statistics and actor returns, reruns every rollout from its bound world checkpoint, and replays every actor checkpoint in its seeded DM Control environment. Added files, missing files, traversal paths, symlinks, altered bytes, stale reports and re-signed but inconsistent analyses fail closed.

Development and confirmatory freezing require a clean committed relevant source tree and one visible device, so a development selection can be source-identical to the later confirmation. Confirmatory verification additionally requires complete dependency capture, JAX x64 disabled, and homogeneous compiler/training/evaluation runtime identity.

## Interpretation limits

- The local smoke is an engineering test with two world and two actor updates. It cannot establish model quality or publication readiness.
- The Equation-7-style arm is a disclosed paper-derived implementation, not the unreleased full Dreamer 4 system.
- The temporal-increment control tests whether a small consistency regularizer explains gains; it does not test the full MeLISA method.
- The 28-cell factorial is secondary, exploratory, and development-only (apart from the all-arm engineering smoke). Its unadjusted intervals are useful for mechanism diagnosis, not a familywise-controlled discovery claim.
- The endpoint-certificate arm is also exploratory and cannot alter the primary two-arm comparison; it exists to test the sufficient-risk condition used by the theorem package.
- State-based DM Control results do not establish broad video-world-model superiority; the separate hard visual benchmark addresses that scope.
