# Selected-checkpoint evaluation diagnostic v2

Status: frozen before held-out access, 2026-10-19. Independent numerical-flow
and contract review passed; an explicit admitted launch after independent fit
terminal/reporting/scheduler verification is required before test access. This document
does not report a result or authorize scheduler resources. Scope is the CPU
FP32 seed-0 capped fresh reconstruction of Pdet(2,32), not the original paper's
five-seed experiment or proof of representation collapse.

## Inputs and authority

The standalone worker is
`python -m representation_collapse.tgrab_figure10_evaluate_job --help`.
Every path and SHA is explicit: reconstruction config; converted manifest;
environment lock, wheel manifest, environment receipt; node2a receipt;
independent conversion receipt; pinned native source root/cell; fit run root;
external fit-summary SHA; finalized ledger head SHA; fit run/attempt/admission
identities; fresh output root/run identity; STOP path and deadline. The
submitter must establish these pins outside the worker. Hash consistency cannot
authenticate the submitter. The source closure and scheduler allocation remain
the outer launcher’s responsibility.

The worker verifies pinned configuration/environment/conversion/source inputs,
then enters mandatory offline `ExperimentReporter` before importing numerical
modules, constructing native models, or opening graph splits. One CPU thread
is required. The environment variable thread limits must all equal one.
Reporter failure stops work. Reporting destination is the existing
`representation-collapse` project; offline history is retained and does not
establish online delivery.

The finalized ledger is opened with the externally pinned context/head and its
entire history is verified. Selected candidate metadata and six payload files
are checked by the existing object-free v1 codec. The pinned fit summary must
match the finalized ledger head and candidate identities. All initialization
references resolve as bounded, single-link, non-symlink primitive JSON files
under the fit root, with exact byte count, role and SHA. Fit reporting identity
and module-mode records must resolve by their candidate-pinned SHA. The
initial-reporting-receipt digest is an historical identity embedded in the fit
reporting record; the mutable finalized W&B receipt is not silently substituted
for that initial receipt.

Native models are freshly seeded/constructed from the pinned public wrappers
and cell. The initial and post-construction RNG snapshots, initial component
semantic hashes, and complete native state-dictionary metadata versions must
match the selected candidate's fit records. These are verification comparisons;
saved RNG state is never restored. No optimizer is created. Source/runtime
receipt deviations stop execution.

## One selected load and node2b state gate

The evaluation output directory is create-only and permanently spends this
attempt, including crashes. Before copying any selected weight, an fsynced
`charged` event records one selected-load attempt and zero test-read attempts.
Strict loading prechecks both components' native catalog, order, CPU FP32
shape, and storage separation before copying either. It loads both recurrent
and link-head weights, rechecks every resulting payload byte and verifies all
native modules entered evaluation mode. Optimizer, RNG and saved terminal H/C
are not restored.

Under evaluation/no-grad mode, a fresh positive-zero full H/C state consumes
train 0..2559 then validation 2560..2815 once, at fixed selected weights. The
complete replay state at s[2816] must match every saved hidden and cell byte,
all 100 canonical node IDs, and next-target timestamp 2816 exactly, including
signed zeros. No tolerance or pooled-state substitute is allowed. Only after
the fsynced replay-verification event can the test capability be invoked.
This diagnostic verifies selected-load/state replay; it does not create a new
selected-gradient/loss parity experiment or change the previously frozen
node2a bootstrap checks.

## Held-out chronology and outputs

An fsynced `test_read` event charges the one permitted payload-read attempt
before calling the sealed `graph_data` test opener. Failure consumes the
attempt. No reopen, retry, resumed state or alternate candidate is allowed.
For each target t=2816..3071, score all 9,900 ordered off-diagonal node pairs
from s[t] at threshold >=0.5, compute existing GR confusion/F1 metrics using
the actual predecessor graph, then consume target G[t]. Both first and last
timestamps and exactly 256 targets are enforced. An extra or missing target
fails the run. The final successor request may load only the already admitted
test split; the sealed loader validates its complete file.

Primary statistic is unweighted temporal mean of 256 target F1 values.
Transition/nontransition means use 8/248 targets; expected phase transitions at
0 and 32 modulo 64 must equal actual adjacent-graph changes. Per-target rows,
raw F1, reductions, checkpoint SHA and final-head SHA are retained in
`evaluation-result.json` and the evaluation diagnostic event. All 256 rows are
also logged offline. Scientific acceptance is always false at this execution
layer. Subsequent independent reduction and interpretation are separate work.

## Explicit departure from v1 receipt serialization

The selected candidate's six-file v1 manifest/provenance remain unchanged.
This contract introduces supplemental event kind
`figure10_evaluation_diagnostic_event_v1`, not schema v1 `test_attempt` receipts.
Events bind run/attempt/admission/final-head identities, sequence, and the
previous event SHA. Required successful order is charged -> loaded ->
replay_verified -> test_read -> complete. A failure after any prefix records
failed and terminates. Completion reporting must succeed before complete is
written. The outer job manifest separately records reporting failure after
numerical completion; it never upgrades that failure to accepted evidence.

The CLI preserves selected manifest/provenance and execution module identities
in `evaluation-inputs.json`, all durable event files, offline reporting history
and final manifest. The changed diagnostic receipt encoding received explicit independent
review; it must not be described as previously accepted v1
test-attempt receipt compliance. Directory identity and create-only output
prevent retry within a run; preventing a second separately named evaluation
is additionally prevented cooperatively by the fit-root create-only
`selected-evaluation-claim.json`. This file is fsynced before fresh native model
construction and binds fit/head/admission/selected checkpoint plus evaluation
run identity. A different output directory cannot bypass it. Any later failure
remains spent. The outer coordinator must still admit only one submission;
neither file ownership nor hashes provide hostile same-user security.
