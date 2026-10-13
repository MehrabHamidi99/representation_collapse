# Prefix acquisition discovery v3 — prospective local protocol

The [complete v2 grid](local-competence-grid-2026-10-13.md) failed before any
periodic distractor stage. This prospective discovery experiment asks whether
immediate-WRITE pretraining helps learn the sixteen-WRITE R0 task, compared
with fresh initialization, at equal subsequent R0 training exposure. It changes
the acquisition schedule in a new protocol; it does not extend, rescue, replace
or reinterpret the retained v2 fits.

The [configuration](../../configs/experiments/prefix-acquisition-discovery-v3.json)
and [implementation](../../src/representation_collapse/prefix_acquisition.py)
are prepared for independent review. No production v3 training was executed
while implementing this protocol. Only explicitly labelled infrastructure
fixtures were exercised. Configuration preparation flags are not execution
authorization: an external, hash-bound coordinator freeze is required by the
production entrypoint.

## Conditions, exposure and endpoints

There are eight sequential CPU fits: GRU/LSTM × categorical/continuous vector
memory × two conditions. Fresh models use initialization seed 0. Transfer models
load only the four pinned v2 LR-0.001 current step-500 parameter checkpoints,
after successful immediate-task validation; their hashes and original source
revision are in the configuration. Both conditions construct a new AdamW
optimizer with empty state. Pretraining optimizer moments, step counters and
learning-rate schedules are never inherited.

Each condition receives exactly 5,000 new R0 optimizer updates, with evaluations
at 500, 1,000, …, 5,000. A transfer model additionally has its known 500 prior
immediate-task updates; equality refers to R0 exposure, not total historical
training. The endpoint is the current parameter state at update 5,000. All
scheduled checkpoint/evaluation results are retained. First passing validation
step is descriptive only: a passing intermediate checkpoint does not stop a
fit or replace its endpoint. No checkpoint or learning rate is selected.

The unchanged architecture is hidden 128/encoder 64, native one-layer GRU or
LSTM and the existing categorical/continuous output head. Training batch is
32 paired arm instances, containing 16 independent synthetic histories. R0
has sixteen ordered WRITE updates, required addresses 0–7, nuisance addresses
8–15, and a terminal native QUERY, with no distractor body. Its two arm inputs
are identical by construction. Loss remains categorical cross-entropy or raw
continuous MSE. AdamW uses LR 0.001, weight decay 0.01 and clipping 1. Each
fit uses one CPU thread, FP32 parameters, deterministic algorithms, TF32 off
and `highest` matmul precision; actual runtime flags and source/lock hashes are
saved. There is no device, precision, batch, loss or data-layout override in the
production CLI.

The total bound is eight fits, 40,000 new optimizer updates and 1,200 seconds
from preparation inside the experiment engine through execution. Deadline
checks occur during stream prehashing, before each optimizer update, before
each native evaluation chunk and after each evaluation. A deadline preserves
the actual partial history/checkpoint and marks execution incomplete/censored,
with endpoint competence unknown and unstarted fits listed. No failure is
fabricated, no fit is replaced and no budget is extended. A single currently
executing CPU operation or atomic write completes before the next cooperative
deadline check. An identity or nonfinite error stops the experiment and retains
its explicit execution failure. No successful gate triggers early stopping.
If execution stops between scheduled evaluations, cleanup saves a separately
labelled censored current-weight checkpoint at the realized optimizer step and
binds its hash in fit status and the final manifest. It is never selected or
treated as an endpoint evaluation. Optimizer state is not saved for resumption;
the protocol grants no resume, replacement or extension authority.
An optimizer call that returns is counted before checking its resulting weights.
If those weights are nonfinite, its realized update/history row and diagnostic
weight artifact are retained at the correct step, with explicit nonfinite and
unusable-for-evaluation labels. An optimizer call that itself raises is not
fabricated as a completed update.

## Matching, fresh validation and provenance

Training stream seed is 2026101301. Before any fitting, the engine generates
every declared R0 batch once, computes canonical input/target/query identities,
and persists every batch hash, rolling prefix hash and final stream hash. It
retains hashes rather than gigabytes of training arrays. Each fit then
regenerates that deterministic stream and checks its batch and rolling identities
before each optimizer update, plus the complete stream identity at the endpoint.
Conditions within each model/task pair therefore receive the exact same data
in the same order. The task-specific stream also matches across architectures;
that sharing does not create independent training-seed replicates.

Fresh R0 validation uses sampling seed 20001, distinct from the v2 validation
seed 10001 and the training stream. It contains 128 independent episodes per
required coordinate per arm: 1,024 paired units, 2,048 arm instances. Arrays,
array identities and file hashes are persisted before the first optimizer
update and checked again during evaluation. Both conditions use the identical
task-specific validation manifest. Every coordinate is retained. Categorical
competence requires accuracy at least 0.95; continuous competence requires
MSE/(1/3) at most 0.05, using the fixed Uniform[-1,1] population variance and
the unchanged coordinate scorer's declared four-ULP threshold allowance.

A direct input-target oracle verifies WRITE types, ordered one-hot addresses,
task payloads, terminal QUERY and its addressed required target, independently
of renderer and competence-evaluation helpers. Behavior comes from the
uninterrupted native terminal-QUERY logits. Raw validation arrays include
complete post-WRITE, pre-QUERY and post-QUERY state: GRU hidden coordinates or
LSTM hidden then cell. The native capture routine records input/weight/source
identities and forked/uninterrupted numerical checks; its adapter tolerances
are not a memory-resolution test or Floquet certification. Independent endpoint
checkpoint/label/state replay remains a coordinator acceptance step.

Initial weights, the transfer checkpoint identity, empty optimizer-state count,
complete update history, per-step training-stream hashes, all scheduled
checkpoints/raw validation, actual precision and execution status are preserved
in unique create-only output paths. File manifests permit independent reduction.
The whole run refuses to overwrite an existing output directory. A malformed
warm checkpoint is rejected before even a fresh condition begins fitting.

## Required freeze and command

The coordinator must first obtain independent source/chronology/unit review,
commit the exact implementation/configuration, and prepare an ignored freeze
receipt. Production checks the current source commit, a clean Git checkout,
the committed configuration bytes and executing module bytes, and the receipt's
exact source/config/review identities before creating outputs or training.
The review binding must refer to the final implementation/configuration hashes;
an earlier review of changed code cannot authorize execution.

The receipt has this shape; hashes and paths below are placeholders, not a
completed freeze or permission record:

```json
{
  "kind": "prefix_acquisition_v3_freeze",
  "protocol_id": "prefix-acquisition-discovery-v3",
  "local_execution_authorized": true,
  "scientific_acceptance": false,
  "source_commit": "FULL_COMMITTED_SOURCE_SHA",
  "config_relative_to_project": "configs/experiments/prefix-acquisition-discovery-v3.json",
  "config_sha256": "EXACT_CONFIG_SHA256",
  "independent_review": {
    "accepted": true,
    "receipt_relative_to_project": "runs/review/ACTUAL_REVIEW.json",
    "receipt_sha256": "ACTUAL_REVIEW_RECEIPT_SHA256",
    "implementation_sha256": "FINAL_PREFIX_ACQUISITION_MODULE_SHA256",
    "config_sha256": "EXACT_CONFIG_SHA256"
  }
}
```

After those gates, the prepared command is:

```bash
PYTHONPATH=src .venv/bin/python -m representation_collapse.prefix_acquisition \
  --config configs/experiments/prefix-acquisition-discovery-v3.json \
  --freeze-receipt runs/ops/prefix-acquisition-v3-freeze.json \
  --output runs/prefix-acquisition-v3-UNIQUE_ATTEMPT
```

The production entrypoint has no implicit fixture fallback. The separate Python
`run_fixture(..., budget=FixtureBudget(...))` API reduces only update, fit and
walltime budgets and marks every artifact `fixture_only` and infrastructure
evidence. Architecture, batch, task, units and native chronology stay fixed.
At least one update, fit or walltime bound must actually decrease; the complete
production envelope is rejected by the fixture API.
Fixture warm weights may be synthetic stand-ins and cannot be used as empirical
pretraining evidence. Focused controls test source/review gates, altered stream
and checkpoint rejection, explicit fixture bounds, native LSTM h+c capture,
reset AdamW, fixed endpoints despite mocked early success and censored timeouts.

## Interpretation limits

This is one initialization/discovery seed. The matched conditions, learning
rates from preceding discovery, arms, coordinates, optimizer steps and validation
episodes are not independent trained replications. Report paired descriptive
endpoint values and every failed/incomplete outcome; no population-competence
guarantee or confirmation inference is enabled here. Successful R0 acquisition
would resolve a prerequisite for future periodicity experiments, not establish
their effect. Failure at the larger fixed endpoint would still not establish
architectural impossibility.

No periodic distractor body, held-out test, covariance/tangent estimator,
intervention/sham, causal-periodicity inference, cross-modal replication,
equivalence test, cluster submission or model-service call is part of this
protocol. Theoretical, causal, numerical and confirmation gates remain separate.
The coordinator owns the actual freeze, invocation and scientific acceptance.
