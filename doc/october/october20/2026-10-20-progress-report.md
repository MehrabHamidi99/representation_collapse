# Research progress — 2026-10-20

## 15:01 UTC — long fit completed overnight; access and monitor reconciled

Direct noninteractive Mila SSH succeeds. Scheduler accounting independently
reports job **11322268 COMPLETED 0:0, elapsed 09:12:16**. The run finished
its registered **200 complete epochs and 511,800 optimizer updates**, stopping
at `complete_epoch_cap`. Validation-only selection chose **epoch 160**, mean
F1 **0.8918793674179636**. These are training/validation results, not a new
test curve. The fit's terminal record says the test payload was never opened
and selected-load attempts remain zero.

The existing local monitor PID97799 had recorded27 consecutive SSH/probe
errors and was backing off30minutes. Therefore uninterrupted overnight
monitoring is not claimed. Its same read-only probe now succeeds and confirms
the completed job, source and selected tuple. The coordinator saved that
observation in ignored `runs/ops/figure10-long-20261020/verified-live-completion.json`
and requested the monitor-only STOP after successful completion was verified.
No Slurm job was canceled, restarted or resubmitted. Existing local Qwen
heartbeat is current:42 completed schema-valid reviews, zero current provider
errors and waiting for evidence. Model participation remains proposal-only;
no new scientific claim follows from worker liveness.

## 15:11 UTC — independent streaming fit audit accepted

Collected1,620 metadata/reporting files,1,362,855,068 bytes, with remote/local
SHA checks and safe regular-file extraction. No binary model weights, state
arrays or graph payloads were transferred. The immutable source-long checkout
and candidate payloads stay on Mila scratch. Collection is under ignored
`runs/ops/figure10-long-20261020/collection-001/returned`.

An independently peer-reviewed verifier streams optimizer intent/outcome
hashchains and epoch seals, raw update/validation events, complete candidate
metadata, the strict earliest-tie selection chain, finalization, authenticated
Git source and runtime identities, actual supervisor cleanup and W&B CRC/
protobuf history. Twelve synthetic checks include fragmented/torn protobuf
records and real primitive-v2 selection roundtrips. Actual audit entered its
own mandatory offline reporter before numerical verification.

Accepted result:200 complete epochs,511,800 confirmed updates, epoch160
selected, validation F1=0.8918793674179636. All563,200 actual offline W&B
history rows and1,689,613 protobuf records reconcile, including the result
metadata artifact and successful exit. W&B stream SHA is
`cebae169ffe2327bfe1625e67127b21472bc8b4dc15d0ed14a2765f1e9a17472`.
Cloud delivery remains unverified. This is a full compact execution audit,
not independent native inference or a reanalysis of binary state/weight arrays.

The independently accepted v2 receipt is
`runs/ops/figure10-long-20261020/collection-001/fit-acceptance.json`.
The accepted selected manifest SHA is
`4abeba2b64261468c8844ee302570d6c5d48cec873a51f9eafd1e6010f75db61`;
authoritative finalized selection-pointer SHA is
`b40bde9b67af8cc5b1c32e8f043886fd75eb6649196588e7fc0b70d3b644662c`.
Actual fit supervisor receipt SHA is
`efbef6041c944ad4af26415e6be7f057f0017e0d0412df44c6024a213946cd60`.
These are distinct identities; none is relabelled as a v1 selection head.

## Implementation in progress

Native agents have disjoint tasks: a true data-only primitive-v2 semantic
codec; the v3 native selected evaluator and protocol; and independent fit/
code review. The coordinator owns integration, cluv admission and scientific
acceptance. The codec checks exact inventory, catalog/order/shape, finite
FP32 component/complete-state bytes, all external execution identities and
validation/provenance. Nine synthetic checks pass. An integration check fixed
its source-closure JSON hash convention to match the actual writer exactly;
no saved evidence was rewritten. A new outer supervisor has11 passing tests
for CPU/resources, explicit v2 receipt/pointer/contract pins, four accepted
artifact bindings and missing-gate rejection.

Next: complete independent evaluator review, commit/stage immutable source,
and submit one separately admitted CPU evaluation through cluv. It must
strictly load both model components and exactly match complete H/C state from
fresh-zero prefix replay before the single held-out opener. The same frozen
within-motif windows and complete-state diagnostics remain in force. No new
held-out test has yet been opened. Full Figure10 multi-model/five-seed
coverage and the causal matched-order memory question remain separate goals.

## 15:16 UTC — primitive-v2 native evaluator reviewed and frozen

The new data-only v2 decoder, CPU supervisor and native v3 evaluator have
independent review. Focused coordinator checks pass33tests:9codec,11supervisor,
4native-flow/selection tests and9version-aware plotting checks. The first local
invocation omitted `PYTHONPATH=src`; the two import-failed test modules were
rerun with the correct repository path and all13 passed. This was a test-harness
path error, not a model or numerical failure. The evaluator author separately
checked34tests including unchanged v1 load/replay helpers.

Frozen protocol: `doc/auto/figure10-selected-evaluation-v3.md`, actual protocol
`tgrab_figure10_primitive_v2_selected_evaluation_v3`. Original v2 selection,
contract, pointer and payload identities are retained. No v1 admission/head
alias or manifest is manufactured. The new launcher pins the accepted long-fit
receipt, four completed fit artifacts, actual supervisor and unchanged native
runtime/data. The native selected load/replay/single-test attempt is not yet
launched. The intended allocation is Mila/main-cpu, accountmila/QOSnormal,
cn-f004, oneCPU/2GiB/zeroGPU/10minutes, work540seconds, shell570seconds,
one submission/no retry/requeue/race. Root will stage a clean immutable source
and create-only admission before the cluv submission.

## 15:20 UTC — pre-launch event-schema regression fixed

Independent compact-verifier construction exposed a lifecycle receipt collision:
`metrics_complete` spread the numerical result's `kind` over the event schema's
`kind`. This was found before any native v3 launch, selected-load claim or test
opener. The event now nests the result, and schema/identity fields cannot be
overwritten. A regression checks the actual successful event sequence. The
numerical/model/data path is unchanged. All35focused tests pass; an independent
review reran five worker tests and the six compact-verifier checks and accepted
the corrected source. Commit f2f2523 was never deployed as an evaluation source;
a new immutable revision will bind the actual launch.

The next-model source audit also recovered five official T-GCN blobs at the
pinned public T-GRAB revision and verified their Git blob IDs. Its documented
complete state is hidden-only, with12,800 coordinates; it needs its own backend,
catalog and native parity checks rather than GC-LSTM's H/C codec. No T-GCN
training or test was performed. See the readiness report dated today.

## 15:29–15:32 UTC — long-fit native evaluation accepted

Reviewed evaluation source **052b660fe776862764880261dffaa6d323b0cfaf** was
staged in the clean main scratch checkout; the completed long-fit source-long
snapshot stayed1891bdd. All59 closure files matched. The large accepted fit
receipt initially exceeded the SSH mux command transport: no files were created
and no job ran. After a read-only reconciliation, the staging helper streamed
metadata through stdin; both create-only pins then verified.

Cluv request11329984 was PENDING/ReqNodeNotAvail on exact nodecn-f004, with no
start estimate. It was canceled using a pending-state-only filter; sacct confirms
CANCELLED, zero elapsed time, Start=None, and the fit-bound evaluation claim did
not exist. It is a retained unexecuted submission, not a failed numerical run.
A fresh admission restricted the replacement to the same Rome CPU family,
with no exact node. Two scheduler submissions are recorded, with only one
worker execution. Numerical source, FP32/thread/model/data controls and exact
H/C replay gate were unchanged. No retry after model/test execution occurred.

Replacement job **11330003** ran on **cn-f003**, accountmila/QOSnormal/main-cpu,
oneCPU/2GiB/zeroGPU/10minutes maximum. It completed **0:0 in56seconds**.
Actual run `evaluation-v3-20261020T152554Z-052b660f`, admission SHA
`75d9f15fd8cbb429d06a30eae911cc4faf3bcdd5158b06c4ebf91712e74cd593`.
The original v2 selected components loaded strictly, and fresh-zero replay
matched every selected terminal H/C byte before the single held-out opener.
All256 targets were scored once, with the passive complete-state capture.

Independent execution acceptance verified the six-event chain, charged unique
fit claim, actual source/selection/contract bindings, all confusion-derived
scores, retained-state descriptor and remote SHA, finished reporter and all258
actual offline W&B history rows. No independent native replay or binary-state
reaudit is claimed. New evaluation result SHA is
`5dda0b134ac5961a6c145e60eb825cd5e4d91d6187a9037b64c7a9ba3cc657f3`;
trajectory SHA is
`efa3a9faf5655230dc8fa3f97b5fd8c63d6fb6f8f1323b7cb37f50ff282c9a6d`.
The25MiB binary stays on scratch and was not copied locally or sent to Qwen.

Test time-mean F1 is **0.8918999850286112**. Frozen early phases1–8 versus
late24–31, excluding transition phase0: balanced **0.9592968674867655 →
0.8187707665977763**, difference **−0.1405261008889892** (−14.0526pp).
Motif A difference−0.1665281480287064, motif B−0.11452405374927199.
Transition/nontransition means are0.6531746997538121/0.899600800682637.
This stronger behavioral reconstruction remains one model seed/graph and
below the analytic Persistence mean0.96875. No causal erasure or full Figure10
claim is accepted. The shared phase/window analysis was frozen before this test.

The version-aware plotter produced the new curve under its own offline reporter;
PNG visual QA passed. Original v2 pointer/contract identity is retained in its
version2 manifest without a v1 head alias. Complete-state analysis is next,
using the same frozen numerical worker and explicitly accepted v3 provenance.
