# Native chronology calibration v1 — implementation preparation

2026-10-13. This implements the independently reviewed
[numerical proposal](native-chronology-calibration-proposal-2026-10-13.md)
at proposal SHA-256 `9438ec2626b30b3f85aca2f784ed5d9d16bc7af18ccc6ad6216181d36cd83f32`
and proposal-review receipt
`ec445dea80061e72446881ac838d30f3621aed441952f76a68cf15a321575c3f`.
**Implementation preparation only: no full matrix, known trained checkpoint
replay, acquisition fit, competence score, source repair or canonical-only
scientific contract has been executed or admitted.**

Source is [chronology_calibration.py](../../src/representation_collapse/chronology_calibration.py),
configuration is [native-chronology-calibration-v1.json](../../configs/experiments/native-chronology-calibration-v1.json),
and focused checks are [test_chronology_calibration.py](../../tests/test_chronology_calibration.py).
The inherited native model, renderer and capture code remain unchanged. The
failed v5 source `f766df5e792c8a6dedaa3ea17f2c92fd630f067d` and its censored
4,000-update evidence are immutable; this module cannot resume or relabel them.

## Diagnostic contract and observable graph

The engine accepts no target, competence or optimization arguments. Synthetic
WRITE payloads are inputs; it never extracts their addressed supervised values.
The eventual known-weight stage loads only pinned checkpoint parameters and
the pinned NPZ `inputs`/`pair_ids` entries **after execution freeze**; it never
accesses its `targets` array. Task label arrays and scores are absent from new
outputs. Every run has zero optimizer updates, provider requests and cluster
operations. Scientific acceptance and canonical-only admission remain false.

P0 is one ordinary `model(inputs)` call. Non-mutating encoder/core hooks capture
its exact encoded tensor, native hidden sequence and returned complete terminal
state, with no additional forward. P1 reuses the exact P0 encoding and splits
the core into prefix and QUERY. P2 separately encodes and runs the prefix and
QUERY. P3 calls the unchanged public `read_query` method from P1's copied
complete origin, capturing its fresh QUERY encoding and native terminal state;
it does not recompute the prefix. P4 reuses P0 encoding in seventeen native
one-row updates and records complete state at every update. Core calls per chunk
are therefore exactly **1/2/2/1/17**. Repetitions recompute independently.

Directed comparisons preserve the reviewed reference operand: terminal outputs
P1–P4 → P0; P2/P4 complete prefix → P1; P3 prefix → P1 bitwise; P1 prefix
hidden → P0 hidden at update 16; P2 prefix encoding and P2/P3 QUERY encoding →
P0 cached encoding. Cached P1/P4 encoder reuse receives an identity check.
Each smaller chunk points to the same path's B=64 result; FP32 points to the
same path/chunk's FP64 result using the FP32 bound. Repeat 2 points to repeat 1
with a bitwise requirement. All reference arrays are hash-checked when reread.

Explicit model/task observable schemas require **all LSTM terminal h/c** and
all separately exposed prefix h/c. P0 intermediate LSTM cell is unavailable:
it is marked with a reason, never filled or inferred. P0 hidden sequence and
terminal cell are observed normally. A missing terminal cell, invented prefix
cell or incompatible complete-state shape is a failure. GRU requires h only;
P4 packing is hidden then cell for LSTM. The copy-null has its own complete-state
schema, including both LSTM components. Native field shape/dtype/stride/
contiguity is recorded before contiguous numerical-artifact export.

The original bounds remain unchanged:
FP32 `1e-6 + 1e-5*abs(reference)`; FP64
`1e-12 + 1e-10*abs(reference)`. Same-dtype edges use their own bound; cross-dtype
arithmetic/reference values are evaluated in FP64 with the FP32 bound.
The known old-source path disagreement is expected to remain a strict
interoperability failure. It is retained like every other residual failure.
**Completing a diagnostic matrix does not mean strict interoperability passed.**
The engine reports these two flags separately and grants no fitting authority.

## Exact registry, clock and failure handling

The production registry contains **372 aggregate conditions** and exactly
**120,000 attempted native-core invocations**:

- Primary: 4 model/task cells × 2 initialization seeds × 2 input streams ×
  2 dtypes × 2 repetitions × 79 chunks × 23 calls = **116,288**.
- Known GRU/categorical θ4,000 discovery regression: 2,048 rows/B64 ×
  2 dtypes × 2 repetitions × 23 = **2,944**.
- N1 zero head, four cells/two dtypes/two repeats/all five paths = **368**.
- N2 all-zero parameters/inputs, four cells/two dtypes/two repeats/P0–P1–P4
  = **320**. This native algebraic null deliberately bypasses semantic WRITE
  validation; it never enters a task scorer.
- N3 unchanged parameters and value-preserving complete-state copies, four
  cells/two dtypes/two repeats/one prefix plus four QUERY calls = **80**.

Primary architecture is 128/64; input seeds 30013/30017 and initialization
seeds 0/17; 64 histories with QUERY `row_id mod 8`. Chunk execution order is
64/32/16/8/1; dtype order FP64 then FP32 ensures frozen directed references
exist. Seeds, tensors and row ordering remain matched. N1/N3 use the first
32 seed-30013 input histories duplicated into 64 rows, fixed seed-0 parameters;
N1 zeroes head weight/bias. N2 zeroes every parameter and uses zero initial
state/all-zero `[64,17,38]` input. N3 compares original, cloned, twice-cloned
and contiguous-cloned h/c, with input identity and immutability checks.

A forward pre-hook counts attempted invocations **before** the native core
executes, rejecting an exhausted budget; the post-hook counts completed calls.
Calls returning nonfinite values remain counted. There are no warmups, hidden
state-recovery forwards, retries or replacement conditions. A single 600-second
clock begins in the production entrypoint before configuration and external
source/review-freeze checks, then covers engine preparation, input loading,
all stages, reductions and artifact finalization. Output creation and trained
loading still wait for successful freeze. Fixtures start their clock at their
own engine entry. One CPU thread, Torch 2.6.0, deterministic algorithms, TF32 off
and highest matmul precision are required/recorded; no GPU device is used.

The complete planned condition/120,000-call ledger is written before running.
Budget or identity failure retains actual per-stage counts, the interrupted
condition and every unstarted condition; it does not shrink the planned total.
Nonfinite required observables stop with an explicit numerical error. Finite
residual failures are retained and remaining prescribed diagnostics can finish
within the same ceiling. Timeout/call exhaustion is censored/incomplete.
Finalization beyond the deadline also clears matrix completion and acceptance.

Outputs are create-only run directories with atomic JSON/NPZ, flushed and
fsynced before replacement. They include configuration, environment/native RNN
source hash/build flags, source/lock hashes, input/row identities, case intentions,
weight/input before-and-after identity, exact FP32-origin/promoted parameter
identity, complete native arrays, per-edge residuals and an artifact manifest.
Partial conditions have intentions and counts, never fabricated final arrays.

Residual reports distinguish bitwise changes from true threshold violations,
give violation entry indices, signed/absolute extrema and quantiles, relative
undefined-zero counts, and directed threshold ratios. Saved candidate/reference
arrays support full per-coordinate reconstruction. Ordered ULP keys collapse
±0 globally; safe integer subtraction handles zero crossings and large finite
distances. Bitwise signed-zero identity remains a separate rule. Arithmetic
overflow in subtraction is a numerical failure; unrepresentable relative/ratio
summaries are null with explicit counts. Empty comparisons and nonfinite operands
are rejected. Unobserved subnormal-flush mode remains unknown.

## Source freeze before full execution

The CLI has only config/freeze/output arguments, with no fixture, source,
precision, dimension, call or walltime override:

```sh
PYTHONPATH=src .venv/bin/python -m representation_collapse.chronology_calibration \
  --config configs/experiments/native-chronology-calibration-v1.json \
  --freeze-receipt runs/ops/native-chronology-calibration-v1-freeze.json \
  --output runs/native-chronology-calibration-v1-UNIQUE_ATTEMPT
```

This is a prepared command, **not** a completed receipt or execution permission.
Before output creation or trained-reference loading, the entrypoint verifies
the exact configuration, external numerical-only authorization, independent
final code/config review receipt, clean current full Git revision and committed
executing/source bytes. Native model/data/capture hashes are fixed to the
unchanged inherited source. The full engine then checks the known checkpoint
and input-file SHA-256 before loading and verifies current step 4,000/input
geometry. A proposal review alone cannot authorize this implementation.

The external freeze must declare `kind=native_chronology_calibration_v1_freeze`,
this protocol ID, `calibration_execution_authorized=true`, and false scientific,
training and canonical-only flags, plus full source SHA, config path/hash and
independent review path/hash. Both review references and review contents must
bind `implementation_sha256` and `config_sha256`; the independent receipt must
state `implementation_preparation_accepted=true` and `scientific_acceptance=false`.
No such production receipt has been created by this implementation agent.

## Restricted fixtures and acceptance limits

`run_fixture(output, FixtureSpec(...))` is the separate preparation API. It
permits one cell, at most two synthetic rows, hidden ≤5/encoder ≤4, at most
500 attempted native calls and 120 seconds. It has no trained checkpoint/input
argument and never enters the known-weight loader. Full 128/64/64 geometry
and expanded budgets are rejected. Outputs are tagged fixture-only; no fitted
parameters, task labels, competence eligibility or empirical claims exist.

The eight-test focused campaign exercises a 284-call tiny GRU categorical matrix including
all nulls, a 92-call LSTM continuous matrix, and one genuine call-ceiling stop:
**377 native forwards** independently counted at the GRU/LSTM methods, ≤500.
Additional checks use only exact registry arithmetic, frozen admission rejection,
directed residual fixtures, ULP adjacency/zero crossing, missing/invented cell
schemas, production-clock propagation and deadline-finalization bookkeeping.
No trained checkpoint was loaded,
no target array accessed and no optimizer update made. Campaign proof and raw
tiny arrays stay in ignored `runs/native-chronology-calibration-v1-fixtures/`.
Independent engineering review uses its own separately bounded fixture campaign.

Remaining admission: independent final code/config acceptance, immutable clean
source publication and external numerical execution freeze. Completing a future
matrix only supports its recorded finite numerical audit. Strict interoperability
additionally requires every required frozen comparison/null to pass; unchanged
θ4,000 already has known discovery failure. A canonical-only execution contract,
source repair, new tolerance, gradient/tangent calibration, new acquisition fit
or periodicity experiment requires its own prospective review and authority.
This forward-only engine neither certifies trained-weight gradient parity nor
accepts representation collapse, competence, causal memory or Floquet claims.
