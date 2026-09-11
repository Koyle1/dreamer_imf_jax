# NeurIPS cluster workflow

This directory is a submission-ready workflow for the Leipzig `login01`
cluster. No cluster job or scientific result is created by installing or
testing these files locally.

## Frozen execution contract

- Workspace: `/work2/ci72buri-dreamer_imf_neurips`
- Slurm account/partition: `dep_inin_dat` / `gpu-l40s`
- Every job: one L40S, 8 CPU cores, 64 GiB, at most 48 hours
- Arrays: at most four concurrent tasks (`%4`); main and controls compute-plan
  cells are serialized (`%1`) so a compiler cache writer exits before its exact
  independent cache-reading verification and no concurrent autotuning can race
- Python: `Python/3.12.3-GCCcore-13.3.0`
- Runtime: JAX/JAXlib 0.8.1, NumPy 2.5.3, dm-control 1.0.46,
  MuJoCo 3.13.0, CUDA 12 wheels from the hash-locked requirements file
- Environment: `JAX_ENABLE_X64=0`, `JAX_PLATFORM_NAME=gpu`, `MUJOCO_GL=egl`, plus a shared
  fail-closed JAX compilation cache at
  `/work2/ci72buri-dreamer_imf_neurips/jax-compilation-cache`. Every compilation is cache-eligible;
  cache read/write errors stop the job.

The preflight fails unless JAX sees exactly one L40S and a real 16x16 DMC
render succeeds through EGL. It binds the clean Git commit, dependency-lock
digest, scheduler account, runtime, and storage evidence. Storage is checked in
two deliberately separate ways: `shutil.disk_usage` is labelled only as
filesystem capacity, while `ws_find` plus the exact `dreamer_imf_neurips` block
from `ws_list` authenticate the allocator path/lifetime and `lfs quota -u
ci72buri /work2` authenticates user block and inode limits. The workspace must
have at least seven days remaining, at least 100 GiB of both filesystem
capacity and user quota, a hard block limit of at least 5,368,709,120 KiB, and
positive remaining inodes. Whitespace may vary, but duplicated, missing,
unbounded, malformed, or contradictory fields fail closed.

## Stage model

Each array task resolves one array index through an immutable map derived from
the already-frozen matrix and invokes the canonical runner with exactly one
`--cell-id`.

```text
preflight + pilot freeze
  -> dataset array -> whole-stage verifier
  -> compute array -> whole-stage verifier
  -> world array   -> whole-stage verifier
  -> rollout array -> whole-stage verifier
  -> actor array   -> whole-stage verifier
  -> pilot finalize + verify + frozen selection
  -> confirmatory freeze
  -> the same five verified stages (1,746 cells)
  -> registered diagnostics
  -> confirmatory finalize + checkpoint-backed verify

verified pilot
  -> development-controls freeze
  -> five mapped arrays, each followed by a whole-stage verifier
  -> development-controls finalize + frozen controls selection
  -> (verified matched confirmatory + authenticated development selection)
  -> confirmatory-controls freeze and the same five-stage chain

verified pilot
  -> hard-pixel freeze -> exact 30-unit array -> aggregate/finalize
```

The next stage checks the previous whole-stage marker and every retained result
digest. Scheduler dependencies use `afterok` only. Per-index `aftercorr`
dependencies are forbidden because the matrix dependencies are not generally
index aligned.

## Launch sequence

First create one clean checkout at the registered location. Authentication for
the clone is intentionally left to the user's existing Git configuration; no
credential is stored here.

```bash
ssh login01
module load Python/3.12.3-GCCcore-13.3.0
mkdir -p /work2/ci72buri-dreamer_imf_neurips/logs
git clone https://github.com/Koyle1/dreamer_imf_jax /work2/ci72buri-dreamer_imf_neurips/source
git -C /work2/ci72buri-dreamer_imf_neurips/source status --porcelain=v1 --untracked-files=all
```

Every submit command is a dry run unless `--execute` is explicitly present.

```bash
CLUSTER=/work2/ci72buri-dreamer_imf_neurips/source/dreamer_imf_comparison/cluster/neurips
python "$CLUSTER/submit.py" preflight
python "$CLUSTER/submit.py" preflight --execute
```

After the preflight job succeeds, verify it and submit one pilot stage at a
time. Wait for the returned stage-verifier job to succeed before issuing the
next command; the code also enforces this condition.

```bash
PYTHON=/work2/ci72buri-dreamer_imf_neurips/venv-cuda12/bin/python
"$PYTHON" "$CLUSTER/verify_cluster_evidence.py" --preflight
"$PYTHON" "$CLUSTER/submit.py" stage --profile pilot --stage dataset --execute
"$PYTHON" "$CLUSTER/submit.py" stage --profile pilot --stage compute_plan --execute
"$PYTHON" "$CLUSTER/submit.py" stage --profile pilot --stage world_model --execute
"$PYTHON" "$CLUSTER/submit.py" stage --profile pilot --stage rollout --execute
"$PYTHON" "$CLUSTER/submit.py" stage --profile pilot --stage actor --execute
"$PYTHON" "$CLUSTER/submit.py" finalize --profile pilot --execute
"$PYTHON" "$CLUSTER/verify_cluster_evidence.py" --pilot
```

Only a verified pilot can freeze the confirmatory matrix. The freeze consumes
the immutable pilot selection and refuses pre-existing confirmatory outcomes.

```bash
"$PYTHON" "$CLUSTER/submit.py" freeze-confirmatory --execute
"$PYTHON" "$CLUSTER/submit.py" stage --profile confirmatory --stage dataset --execute
"$PYTHON" "$CLUSTER/submit.py" stage --profile confirmatory --stage compute_plan --execute
"$PYTHON" "$CLUSTER/submit.py" stage --profile confirmatory --stage world_model --execute
"$PYTHON" "$CLUSTER/submit.py" stage --profile confirmatory --stage rollout --execute
"$PYTHON" "$CLUSTER/submit.py" stage --profile confirmatory --stage actor --execute
"$PYTHON" "$CLUSTER/submit.py" finalize --profile confirmatory --execute
"$PYTHON" "$CLUSTER/verify_cluster_evidence.py" --confirmatory
```

The confirmatory finalize command schedules registered diagnostics first and
uses whole-job `afterok` for finalization. The verifier then replays retained
checkpoints and recomputes the analysis; completion does not imply that the
superiority gate passed.

## Selective retry

Calling a stage submitter again schedules only indices whose canonical result
is missing. For a result that exists but fails strong validation, create an
immutable GPU-side retry audit and then pass its exact map back to the
submitter:

```bash
"$PYTHON" "$CLUSTER/submit.py" audit-retry --profile pilot --stage world_model --execute
# after the audit finishes, use the emitted retry-<jobid>.json path
"$PYTHON" "$CLUSTER/submit.py" stage --profile pilot --stage world_model \
  --retry-map /work2/ci72buri-dreamer_imf_neurips/results/matched_objective_pilot/cluster_retry_maps/world_model/retry-<jobid>.json \
  --execute
```

The submitter validates the retry-map digest, matrix identity, stage-map
identity, indices, and cell IDs. It cannot expand the audited subset.
Supplementary retry maps additionally bind a byte-level snapshot of every
selected artifact and receipt. That snapshot is checked both at submission and
again inside the array worker before anything is moved. Invalid existing
supplementary artifacts are moved into a per-audit `cluster_state/.../quarantine`
directory, never deleted, so canonical runners can recreate their immutable
paths. A valid artifact with a missing/invalid GPU receipt is retained and
reverified in the retry worker.

## Controls and visual track

`supplementary_plan.json` registers the implemented controls and 30-unit hard
visual workflows. Their order is deliberately after matched-objective evidence:

1. Development controls consume only the verified pilot and freeze their own
   selection.
2. Confirmatory controls require both matched-objective profiles and the
   development-control selection.
3. The hard visual matrix is three tasks by five seeds by two compute tracks.

All use the same L40S runtime and `%4` cap (except serialized main and controls
compute planning).
Cluster maps, receipts, retry maps, stage markers, and profile markers live
under `cluster_state/supplementary`, outside canonical scientific result roots.
Every controls array cell runs its canonical strong validator on GPU and writes
a ledger-authenticated, file-inventory-bound receipt. A whole-stage verifier
accepts only the complete ordered receipt set before enabling the next stage.
Development and confirmatory controls then run the canonical finalizer and
verifier and seal an authenticated profile marker.

Run controls one stage at a time, waiting for each returned whole-stage verifier
to succeed:

```bash
"$PYTHON" "$CLUSTER/submit.py" freeze-controls --controls-profile development --execute
# Wait for each command's verifier job before issuing the next line.
"$PYTHON" "$CLUSTER/submit.py" controls-stage --controls-profile development --controls-stage dataset --execute
"$PYTHON" "$CLUSTER/submit.py" controls-stage --controls-profile development --controls-stage compute --execute
"$PYTHON" "$CLUSTER/submit.py" controls-stage --controls-profile development --controls-stage world --execute
"$PYTHON" "$CLUSTER/submit.py" controls-stage --controls-profile development --controls-stage rollout --execute
"$PYTHON" "$CLUSTER/submit.py" controls-stage --controls-profile development --controls-stage actor --execute
"$PYTHON" "$CLUSTER/submit.py" finalize-controls --controls-profile development --execute

# This freeze fails until the matched confirmatory profile and authenticated
# development-controls selection both exist.
"$PYTHON" "$CLUSTER/submit.py" freeze-controls --controls-profile confirmatory --execute
"$PYTHON" "$CLUSTER/submit.py" controls-stage --controls-profile confirmatory --controls-stage dataset --execute
"$PYTHON" "$CLUSTER/submit.py" controls-stage --controls-profile confirmatory --controls-stage compute --execute
"$PYTHON" "$CLUSTER/submit.py" controls-stage --controls-profile confirmatory --controls-stage world --execute
"$PYTHON" "$CLUSTER/submit.py" controls-stage --controls-profile confirmatory --controls-stage rollout --execute
"$PYTHON" "$CLUSTER/submit.py" controls-stage --controls-profile confirmatory --controls-stage actor --execute
"$PYTHON" "$CLUSTER/submit.py" finalize-controls --controls-profile confirmatory --execute
"$PYTHON" "$CLUSTER/verify_cluster_evidence.py" --controls
```

A controls retry audit and retry submission are explicit:

```bash
"$PYTHON" "$CLUSTER/submit.py" audit-controls-retry \
  --controls-profile confirmatory --controls-stage actor --execute
"$PYTHON" "$CLUSTER/submit.py" controls-stage \
  --controls-profile confirmatory --controls-stage actor \
  --retry-map /work2/ci72buri-dreamer_imf_neurips/cluster_state/supplementary/controls/confirmatory/retry_maps/actor/retry-<audit-jobid>.json \
  --execute
```

The pixel worker runs checkpoint prediction replay, both retained DMC dataset
replays, and compiler-compute rederivation on the original homogeneous L40S
runtime before writing its receipt. Finalization requires the exact unique
Cartesian product (two tracks, three tasks, five seeds), independently rederives
metrics from retained predictions, and authenticates the aggregate and profile
marker:

```bash
"$PYTHON" "$CLUSTER/submit.py" freeze-pixel --execute
"$PYTHON" "$CLUSTER/submit.py" pixel --execute
"$PYTHON" "$CLUSTER/verify_cluster_evidence.py" --pixel

# If the array/finalizer reports invalid units:
"$PYTHON" "$CLUSTER/submit.py" audit-pixel-retry --execute
"$PYTHON" "$CLUSTER/submit.py" pixel \
  --retry-map /work2/ci72buri-dreamer_imf_neurips/cluster_state/supplementary/pixel/hard/retry_maps/artifact/retry-<audit-jobid>.json \
  --execute
```

Controls and pixel evidence are secondary; neither alters or substitutes for
the frozen four-interval primary gate. Local contract checks prove structure
and negative controls but explicitly do not claim that cluster results exist:

```bash
"$PYTHON" "$CLUSTER/verify_local_contract.py" --controls
"$PYTHON" "$CLUSTER/verify_local_contract.py" --pixel
"$PYTHON" "$CLUSTER/verify_local_contract.py" --negative-controls
```

## Provenance

`cluster_state/provenance_job_ledger.jsonl` is an append-only, file-locked hash
chain. Initialization, preflight, freezes, submissions, cell outcomes, retry
audits, stage verification, and profile verification append records containing
the previous event digest. Run:

```bash
"$PYTHON" "$CLUSTER/workflow.py" verify-ledger
```

to verify the full chain.
