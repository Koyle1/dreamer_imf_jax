# Secondary hard visual benchmark

## Scope and claim boundary

This benchmark asks a deliberately narrower question than the primary study: when the same compact
recurrent world model must predict real RGB observations, how do paper-derived Dreamer-4
Equation-(7)-style shortcut forcing and trajectory iMF compare across long open-loop horizons and
the 1/2/4-NFE frontier?

It is **secondary evidence**. It cannot replace, rescue, modify, or satisfy any of the four frozen
primary intervals in `matched_objective_protocol.json`. The smoke profile is engineering evidence
only. The hard profile is also not a full Dreamer 4 reproduction: both arms use the project's shared
compact JAX recurrent model, and only their generative objectives differ.

The machine-readable contract is `pixel_benchmark_protocol.json`. It SHA-256-locks the inherited
matched-objective model contract, and every completed result stores the exact protocol, executable
source hashes, runtime identity, datasets, schedules, objective keys, checkpoints, predictions,
metrics, compiler evidence, a complete artifact manifest, and a manifest seal.

Sealed cell roots are literal directories with no symlink anywhere in their lexical path. Their
direct entries are the exact registered regular-file set: symlinks, special files, and even otherwise
harmless extra directories fail verification. The controlled `result.json` schema is exact at every owned nested level;
compiler-generated cost-analysis maps remain digest-bound opaque numeric maps.

## Frozen environment and observations

The hard suite contains `walker_walk`, `cheetah_run`, and `finger_spin` from DeepMind Control. The
one-run local smoke uses `pendulum_swingup`. Each task uses camera 0 and native termination, without
a time-limit override. An input action is clipped to `[-1,1]`, mapped to the native action spec, and
repeated twice. Rewards are summed and discounts multiplied over the native steps actually executed;
the loop stops immediately on `LAST`.

The environment is rendered after reset and after the last native step of each repeated action:

- raw render: 64 x 64 RGB `uint8`;
- hard renderer: EGL; local macOS smoke renderer: GLFW;
- deterministic 2 x 2 integer area mean, with round-half-up, to 32 x 32;
- three frames stacked oldest-to-newest on the channel axis, giving `(32,32,9)`;
- a reset repeats its first rendered frame three times and is aligned with an all-zero action;
- JAX receives `float32(uint8) / 255`, while the authenticated replay remains `uint8`.

Train and held-out environments have disjoint deterministic seeds. Both use the same frozen clipped
Gaussian AR(1) exploration policy. Training chunks erase the unobserved action at their artificial
left boundary, mark that token `is_first`, and mask the two/eight burn-in tokens specified by the
smoke/hard profile. Held-out windows are sampled without replacement only when their complete context
and maximum-horizon suffix remain in one native episode.

Worker verification creates fresh DMC environments from the registered train and held-out seeds,
replays every retained normalized action, and byte-compares every rendered stacked observation. It
also rechecks rewards, discounts, termination, action-repeat counts, and episode indices. Separately,
it regenerates the frozen AR(1) action stream from its domain-separated seed. Internally consistent
hashes alone therefore cannot authenticate invented actions or pixels.

## Objectives and compute matching

Both arms inherit every common DreamerConfig field from the digest-locked primary protocol. They use
identical initialization seeds, data, batch-start prefixes, per-update objective keys, evaluation
windows, actions, posterior keys, and base Gaussian noise. Same-shaped encoder, decoder, recurrence,
posterior, reward, and continuation tensors must be bit-identical before training.

The hard profile registers two tracks:

1. `equal_updates`: both arms receive 20,000 world-model updates over 100,000 rendered observations.
2. `equal_compiler_flops`: shortcut receives 20,000 updates; trajectory iMF receives the floor of
   the shortcut training-FLOP budget divided by its own XLA-reported full-update cost.

The runner compiles the full forward/backward Adam update, a complete generated transition at 1, 2,
and 4 NFEs, and one isolated objective-specific prior-field evaluation. It retains raw XLA cost
analyses and StableHLO digests, parameter counts, allocations, realized compiler FLOPs, exact prior
field evaluations, and synchronized training time. Because XLA can report a loop body once rather
than multiply by its trip count, the runner reports both raw transition FLOPs and an explicit
NFE-adjusted quantity: one-NFE transition FLOPs plus `(NFE-1)` separately compiled field evaluations.
Compiler evidence can only be rederived on the same JAX/compiler/device class; login-node
verification may authenticate its hashes without pretending CPU compilation reproduces GPU costs.

The hard runtime is frozen to Python 3.12.3, JAX/JAXlib 0.8.1, NumPy 2.5.3, dm-control 1.0.46,
MuJoCo 3.13.0, `JAX_ENABLE_X64=0`, `JAX_PLATFORM_NAME=gpu`, EGL, exactly one non-disabled
`CUDA_VISIBLE_DEVICES` selector, exactly one JAX GPU, and an L40S device-kind string. Aggregation
requires the complete runtime identity to be homogeneous. The engineering smoke remains portable:
it permits CPU or GPU JAX, records rather than freezes local versions, and stays claim-ineligible.

## Visual evaluation

The hard evaluation uses 32 paired held-out windows, eight predictive draws, eight context frames,
and horizons 1, 2, 4, 8, 15, 30, and 60 repeated-action transitions. Every arm is evaluated at all
three NFEs. The declared primary visual comparison is shortcut at four NFEs versus trajectory iMF at
one NFE, but it remains secondary to the main study.

Metrics are computed on the newest RGB frame:

- channel-variance-normalized per-draw visual MSE and its normalized horizon AUC;
- PSNR and global RGB SSIM of the predictive mean;
- a finite-ensemble energy score;
- temporal-difference MSE between successive registered horizons.

Targets, contexts, future actions, window indices, posterior keys, base noises, every selected-horizon
prediction for every arm/NFE, and per-window metrics are retained. Verification loads each trusted
local checkpoint, requires an exact scalar-integer optimizer step plus finite shape-matched world-model
parameters and Adam moments, and regenerates predictions from the retained inputs. Aggregate metrics are then
recomputed from the regenerated raw tensors. A byte mutation in a sealed output is fatal. A distinct
aggregate verifier reads raw cell predictions, rederives each AUC independently, and uses a second
implementation of fractional IQM trimming and the registered hierarchical bootstrap.

## Commands

From the workspace root, with the project environment activated or the paths below available:

```bash
/private/tmp/trajectory-imf-venv/bin/python \
  dreamer_imf_comparison/scripts/verify_pixel_benchmark.py --contract

/private/tmp/trajectory-imf-venv/bin/python \
  dreamer_imf_comparison/scripts/verify_pixel_benchmark.py --tests

/private/tmp/trajectory-imf-venv/bin/python \
  dreamer_imf_comparison/scripts/verify_pixel_benchmark.py --smoke
```

The smoke output root is source-addressed under `/private/tmp`; rerunning the command verifies the
sealed artifact instead of overwriting it. On sandboxed Darwin hosts, GLFW needs access to the host
graphics service. Run the same ordinary-Python command with that access; `mjpython` is deliberately
not used because its worker-thread execution is incompatible with GLFW's `NSWindow` creation here.

One hard GPU cell (both arms) is run as follows. Use the frozen cluster runtime wrapper so
`JAX_ENABLE_X64=0`, `JAX_PLATFORM_NAME=gpu`, `MUJOCO_GL=egl`, and exactly one non-disabled CUDA
selector are present before the first JAX or MuJoCo import:

```bash
MUJOCO_GL=egl python dreamer_imf_comparison/scripts/run_pixel_benchmark.py run \
  --profile hard \
  --task dmc_walker_walk \
  --seed 19 \
  --track equal_updates \
  --output /work2/$USER/dreamer_imf_pixel/equal_updates/dmc_walker_walk/seed_19

MUJOCO_GL=egl python dreamer_imf_comparison/scripts/run_pixel_benchmark.py verify \
  --profile hard \
  --output /work2/$USER/dreamer_imf_pixel/equal_updates/dmc_walker_walk/seed_19 \
  --rederive-compute
```

Run the Cartesian product of the three hard tasks, five seeds, and two compute tracks. Each cell runs
both arms, so there is no arm-specific dataset or scheduler drift. A login node can do a hash-only
aggregation check with `--authentication-only`; checkpoint prediction replay and compiler rederivation
must occur on a homogeneous worker GPU before that check.

After every cell in one or both tracks is present, aggregate only complete 3-by-5 matrices:

```bash
python dreamer_imf_comparison/scripts/run_pixel_benchmark.py aggregate \
  --inputs /work2/$USER/dreamer_imf_pixel/equal_updates/*/seed_* \
           /work2/$USER/dreamer_imf_pixel/equal_compiler_flops/*/seed_* \
  --output /work2/$USER/dreamer_imf_pixel/secondary_visual_summary.json

python dreamer_imf_comparison/scripts/run_pixel_benchmark.py verify-aggregate \
  --inputs /work2/$USER/dreamer_imf_pixel/equal_updates/*/seed_* \
           /work2/$USER/dreamer_imf_pixel/equal_compiler_flops/*/seed_* \
  --output /work2/$USER/dreamer_imf_pixel/secondary_visual_summary.json
```

Omitting `--tracks` requests both registered tracks and therefore requires exactly 30 unique cell
identities. To intentionally summarize one track, pass `--tracks equal_updates`; that requires its
complete 15-cell task-by-seed matrix. Partial inferred matrices, duplicate roots or identities,
unrequested tracks, and missing inputs fail. Each input record commits its track/task/seed identity
and manifest, seal, result, and prediction digests; `verify-aggregate` checks those commitments
against the supplied roots.

The registered contrast is shortcut 4-NFE AUC minus trajectory-iMF 1-NFE AUC, so positive values
favor trajectory iMF. The aggregate is an exact 25%-trimmed IQM over task-by-seed units with a
10,000-draw paired hierarchical task/seed bootstrap interval. It is descriptive, has no authority to
enter the primary multiplicity family, and fails if a cell is missing, duplicated, non-GPU, or from a
different compiler/device identity.

## Honest limitations

- The visual encoder and decoder are compact MLPs over flattened pixels, not the convolutional or
  transformer visual stack of a state-of-the-art video model. This isolates the objective but limits
  absolute quality.
- Three tasks and five world-model seeds support secondary robustness evidence, not a new primary
  significance claim.
- Global SSIM is an explicitly defined diagnostic, not windowed multiscale SSIM or LPIPS.
- Random/AR(1) replay does not establish closed-loop visual control performance.
- A successful local smoke proves rendering, training, rollout, authentication, and replay mechanics;
  its one update, one window, and one seed are scientifically uninterpretable.
- Pickle checkpoints are loaded only after their file hashes are authenticated and must never be
  accepted from an untrusted source.
