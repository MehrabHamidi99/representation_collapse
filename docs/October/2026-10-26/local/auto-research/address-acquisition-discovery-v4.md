# Address acquisition discovery v4 — prospective preparation

The [v3 discovery](../../../2026-10-13/local/auto-research/prefix-acquisition-v3-results-2026-10-13.md) completed all
eight fixed 5,000-update fits without recovering every required coordinate.
This separate experiment asks whether the same recurrent architecture and
sixteen-WRITE payload stream can acquire a *predictable singleton target* at
address 0, 7 or 15. The original task randomly queried one of eight required
addresses. Changing to a fixed queried address jointly changes required load
and selection uncertainty; success cannot be called pure capacity or retention-
lag causality. Address identity changes with WRITE position as well.

The [configuration](../../../../../configs/experiments/address-acquisition-discovery-v4.json),
[new implementation](../../../../../src/representation_collapse/address_acquisition.py)
and focused controls are prospective preparation. No v4 production training
has run. Only explicitly tagged reduced-budget infrastructure fixtures were
used during implementation. V2/v3 source, eligibility guards, results and source
snapshots are preserved.

## Declared fits and fixed endpoints

There are twelve fresh seed-0 fits: GRU/LSTM × categorical/continuous × terminal
QUERY address 0/7/15. Each begins with the exact initial parameters recorded
for its v3 fresh cell, checked both by weight identity and tensor equality.
Initial checkpoint, fixed v3 endpoint, metric/provenance and original reduction
hashes are pinned in the configuration. Unchanged model/data source is pinned
to `999d88d8679c0f353c6bb7fb4f5ab8433a7e730c` and checked again at execution.
Each fit constructs a new empty-state AdamW optimizer; no warm parameters or
optimizer history enter the singleton fits.

Architecture and numerical controls remain hidden 128/encoder 64/batch 32,
LR 0.001, AdamW weight decay 0.01, clipping 1, one CPU thread and native FP32
with deterministic algorithms, TF32 off and highest matmul precision. There
are 5,000 new optimizer updates per fit, evaluation every 500 updates, a
60,000-update/twelve-fit ceiling and a 1,200-second overall engine deadline.
Preparation, baseline reevaluation and fitting share that walltime ceiling.
All scheduled results and the current 5,000-update endpoint are retained.
First passing step is descriptive; no early-success stop, historical best,
successful-checkpoint selection, rescue or replacement is allowed.

Sequences have exactly sixteen ordered WRITEs and one terminal QUERY, with no
body. Querying addresses 0/7/15 leaves respectively 15/8/0 later WRITE updates
after the target-bearing WRITE, followed by the common QUERY update. Address 15,
formerly nuisance in the original task, becomes the sole task-required value
in that diagnostic. A singleton target is known from its training distribution;
the model could solve it without interpreting a varying QUERY selector.

## Exact training match and fresh validation

Every condition derives from the original v3 training stream, seed 2026101301.
The existing base constructor consumes every original RNG draw, including the
random QUERY draw. V4 copies its batch and changes only the terminal selector,
selector metadata and the supervised target for the chosen address. All sixteen
WRITE payloads, event types, addresses and ordering remain byte-identical.
No extra randomness is consumed to choose a fixed address. Each regenerated
base batch and rolling prefix must match the hash-bound v3 base manifest.

Before any optimizer step, the engine streams and hashes all declared derived
batches, storing per-batch, rolling-prefix and complete stream identities.
It then regenerates them per fit and checks identities before each update.
Only hashes are retained for the entire training stream, avoiding resident
copies of all twelve streams. Models/tasks use their declared task-specific
base streams; within a task the address conditions share precisely the same
WRITE histories and RNG advancement. Identical paired R0 arms do not create
independent histories: batch 32 contains sixteen paired episode units.

New validation uses seed 20003 and 128 independent full-prefix histories per
address/arm. The same task-specific 128 histories are used at all three fixed
addresses; only their selectors and targets change. Arrays and hashes are
persisted before optimizer step 0, then checked again during scoring. The two
arms are identical R0 copies and are reported as paired units, not independent
replications. Each singleton gate requires accuracy ≥0.95 or MSE/(1/3) ≤0.05,
with a fixed four-ULP float64 threshold allowance. The continuous normalization
is the Uniform[-1,1] population variance, not an observed sample variance.
These singleton gates are separate from main v2/v3 all-coordinate eligibility.

## Pinned baseline reevaluation and paired ledger

The v3 fresh step-5,000 checkpoint for every model/task is a predeclared random-
QUERY-0–7 reference. Its parameter identity is immutable; it is not retrained,
selected from a trajectory or optimized on v4 inputs. Before singleton fitting,
it is prospectively reevaluated on the exact new v4 seed-20003 validation
histories and terminal selectors 0 and 7. These addresses lie within its
original trained QUERY support. Baseline and singleton endpoint arrays therefore
have identical complete histories, target values and episode identities for
an episode-paired native-error contrast.

The ledger reports singleton-trained minus pinned-random-query-trained error
at the current endpoint. Positive means worse; categorical units are error
probability and continuous units MSE/(1/3). All 128 within-seed paired differences
are retained. No confidence interval, p-value, population guarantee or causal
capacity/lag attribution is enabled. Conditioning evaluation on a supported
QUERY does not make the original training objective a singleton objective.

Address 15 lies outside the original model's trained QUERY-selection support.
Its primary baseline evaluation/contrast is omitted and explicitly labelled
excluded. V3's historical seed-20001 scores remain provenance; they are not
pooled with the new draws or used for an unpaired primary contrast. Baseline
reevaluation adds no optimizer updates or fits and performs no held-out
confirmation evaluation.

## New validated sixteen-address adapter

The existing v2/v3 native capture deliberately rejects QUERY addresses ≥8.
It remains unchanged. V4 provides its own oracle and native adapter for the
full 0–15 domain. They validate sixteen ordered one-hot WRITE addresses,
WRITE/QUERY event types, valid categorical or bounded real payloads, matching
terminal address, and zero QUERY payload. A direct addressed-WRITE oracle
checks each supervised target. No label is inserted into the QUERY or native
state update.

Uninterrupted native terminal-QUERY outputs define behavior. Separate native
core/head and forked continuation checks compare outputs and complete states
at absolute 1e-6 plus relative 1e-5 tolerance. Raw arrays preserve post-WRITE,
pre-QUERY and post-QUERY state: 128 hidden coordinates for GRU; 128 hidden then
128 cell coordinates for LSTM. With no body, post-WRITE and pre-QUERY states
coincide. Packing, input/weight/source identities and phase update indices are
recorded. Numerical agreement is an adapter check, not Floquet certification,
retained-information resolution, geometry or causal memory evidence.

Deadline checks occur during prehashing, before each optimizer update and each
evaluation chunk. A censored run retains the actual completed update history
and a diagnostic current-weight checkpoint at its realized step; endpoint
competence remains unknown and remaining fits remain unstarted. A returned
optimizer update is counted before checking its resulting weights. Nonfinite
results preserve the correct realized step and explicitly unusable diagnostic
weights. No resume, replacement or extension authority is inferred. A current
CPU operation or atomic write completes before the next cooperative deadline
check. File manifests bind all preserved artifacts in unique create-only paths.

## Source freeze and execution gate

The coordinator owns independent review, the committed source/config freeze
and actual execution. Production requires a clean committed checkout and an
ignored receipt with this shape; values below are placeholders:

```json
{
  "kind": "address_acquisition_v4_freeze",
  "protocol_id": "address-acquisition-discovery-v4",
  "local_execution_authorized": true,
  "scientific_acceptance": false,
  "source_commit": "FULL_COMMITTED_SOURCE_SHA",
  "config_relative_to_project": "configs/experiments/address-acquisition-discovery-v4.json",
  "config_sha256": "EXACT_CONFIG_SHA256",
  "independent_review": {
    "accepted": true,
    "receipt_relative_to_project": "runs/review/ACTUAL_V4_REVIEW.json",
    "receipt_sha256": "ACTUAL_REVIEW_SHA256",
    "implementation_sha256": "FINAL_ADDRESS_ACQUISITION_MODULE_SHA256",
    "config_sha256": "EXACT_CONFIG_SHA256"
  }
}
```

The gate checks source/config committed bytes, current implementation/review
hashes and all pinned v3 references/base-stream identities. A source archive
used for execution must include its own Git identity and copies of the exact
ignored references and review receipt at their configuration-relative paths.
Prepared command, only after those gates:

```bash
PYTHONPATH=src .venv/bin/python -m representation_collapse.address_acquisition \
  --config configs/experiments/address-acquisition-discovery-v4.json \
  --freeze-receipt runs/ops/address-acquisition-v4-freeze.json \
  --output runs/address-acquisition-v4-UNIQUE_ATTEMPT
```

Production exposes no overrides or implicit fixture fallback. The separate
`FixtureBudget` API must reduce at least one update/fit/wall bound and marks all
artifacts as infrastructure-only evidence. Focused tests cover exact payload/RNG
derivation, pinned v3 stream prefixes, QUERY-15 native LSTM h+c with original
guard rejection preserved, no QUERY leakage, target/address errors, missing or
dirty freezes, strict fixture budgets, supported paired-baseline contrasts,
fixed endpoints despite mocked early success, and realized-step censor/nonfinite
artifact preservation. No new empirical result follows from those fixtures.

## Claim boundary and subsequent controls

One initialization/discovery seed is used. Coordinates, addresses, task types,
models, paired arms, optimizer steps and validation episodes are not independent
trained-seed replications. Interpret singleton acquisition and paired conditional
error descriptively, retaining every failed or incomplete fit. This protocol
does not establish pure capacity, lag causality, architectural impossibility,
cross-modal replication, periodicity-induced harm, equivalence, state covariance
collapse or a certified Floquet cycle.

Address-identity/position counterbalancing or a matched nuisance-WRITE replacement
could be separately designed after this gate. Neither is silently added here.
The original v2/v3 outcomes stay intact; this acquisition study supplies no
cluster/provider entitlement or expanded compute budget.
