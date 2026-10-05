# Frozen-representation parallel trajectory diagnostic

This diagnostic adds a separately trained joint trajectory-iMF field; it does not
modify `agent.train`, `agent.policy`, old checkpoints, or existing study roots.
`parallel_protocol.json` is the frozen machine-readable experimental contract.

## Prediction and fitting

The input belief is the repaired agent's flattened `(h,z)` pair, 2,560 features.
`parallel_runner.Predictor.predict_trajectory(start_belief, action_sequence,
noise, flow_steps)` returns all 15 future feature pairs, suitable for the frozen
decoder and reward head. Shapes are `[B,2560]`, `[B,15,2]`, and `[B,15,2560]`.
Flow evaluations 1/2/4 share trained weights. Flow time is not physical time.

The four-block width-128 transformer uses full-trajectory JVPs. The average and
auxiliary velocity readouts are elementwise affine `(1+alpha)*x+beta`: a plain
width-128 additive projection cannot remove Gaussian noise outside its output
subspace in 2,560 dimensions. Analytical and beyond-width Gaussian transport
tests cover this otherwise unavoidable noise floor. This readout detail is
explicitly part of the protocol, not an empirical hyperparameter selection.

All parent parameters are frozen. Only real posterior branch trajectories are
targets; future observations never enter the predictor's conditioning inputs.
Feature statistics and metric scales use training episodes only. Three head
initializations are not three independent world-model seeds. Validation uses
fixed 32-particle draws every 500 updates; the first minimum of normalized
observation plus cumulative-reward MSE selects the checkpoint.

## Dataset and budget

Main collection has 64 episodes, 4 anchors, and 4 exogenous action plans per
anchor. Entire episodes are assigned 40/12/12, balanced by collection mode.
The 16 exact duplicate restores bring the main cost to 95,200 native steps.
Full GPU preflight uses 1,510 more, giving 96,710 for one successful path.
The remaining budget is not automatically spent. An append-only, locked ledger
reserves steps before every simulator call. Failed and interrupted reservations
remain charged; registration of a repaired source cannot reset the ledger.

Raw episode and branch observations, actions, inference seeds, targets, and
simulator/belief snapshots are retained. Snapshot pickles are trusted,
process-local forensic records, not a portable unchecked restoration API.
The monotonic accounting counter is never restored. Physics, task RNG, episode
clock/reset state, wrapper state, and integration warm-start fields are restored.

## Execution

Deploy a clean exact commit and bundle into a fresh immutable checkout. Use the
existing pinned Dreamer/JAX runtime and authenticate both fixed 200k seed431
dependencies before registration. The original study directories are read-only.

1. Run `scripts/test_parallel.sh SOURCE` on the deployment toolchain. It includes
   numerical, collection, metric, protocol, prior-agent, and real-checkpoint tests.
2. Register with `python -m dreamer_imf_compare.parallel_study register --root ROOT
   --budget-path SHARED_LEDGER --bundle-path EXACT_BUNDLE` using the immutable checkout on `PYTHONPATH`.
3. Invoke `scripts/submit_parallel_study.py preflight --root ROOT --continue-chain`.
4. CPU-only after-any handoffs inspect exact Slurm IDs, exit codes, source hashes,
   and strict stage markers before submitting the next scientific stage.
5. A failed stage stops the chain. Existing intents/receipts are never blindly
   resubmitted; uncertain submissions require read-only reconciliation.

Stages are GPU preflight → collection → three head fits → evaluation/profiling →
independent verification. Final `completion.json` authenticates terminal Slurm
completion in addition to the numerical `report.json`. Existing artifacts are
exclusive, hash-bound, and never overwritten. Compilation caches are disabled
for cross-process replay; any exact prediction replay failure blocks acceptance.

## Reading results

Compare decoded observables, not latent coordinates across agents. Report all
NFE variants, sequential iMF 1/4, categorical Dreamer, posterior-head error,
persistence and zero reward. Errors and distributions are evaluated at
horizons 1/3/5/10/15, with episode-clustered uncertainty and reward coverage.
Fewer than five positive-reward test episodes makes reward conclusions
inconclusive. Prefix and direct/composed checks compare distributions, not
identically seeded paths. Composition is the declared 5+10 zero-padding
diagnostic, not a general variable-horizon semigroup assertion.

Profiles use 32 samples, batch sizes 1/64, synchronized warm latency median/p95,
cold first-call compilation-plus-execution time, and allocator memory peaks.
Cold latency is an upper bound on compilation cost, not a pure compiler timer.
All models share one GPU allocation; process-local allocator peaks include
compilation and warmup. Resident counts include the complete frozen agent and
new head; active counts exclude the unused auxiliary velocity head and inactive
parent components. This is not a size-matched or from-scratch comparison.

The exploratory screen requires at least 2× lower median decoded prediction
latency and observation/cumulative-reward MSE within 10% of sequential four-step
iMF in every head seed, with no detected material action or temporal defect.
Failure or inconclusive evidence is a valid outcome. Passing does not establish
statistical equivalence, superiority, or faster closed-loop actor training.
