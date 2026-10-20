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

### 15:38 UTC — complete-state analysis completed

Cluv job **11330070** completed `0:0` in **40 seconds** on `cn-f002`,
source `3a1a297bc84fb96c66ddd34df7aa4ff62e46716d`, one CPU/2 GiB/zero
GPUs. This reads the accepted retained H/C trajectory only: no retraining,
weights/dataset access or new test opener. A premature compact collection
returned an empty archive before terminal status; that failure was retained,
the scheduler reconciled, and a new terminal collection initiated. No job
resubmission occurred. Independent numerical acceptance is pending.

Independent behavioral reduction confirms the 14.0526 pp frozen-window F1
loss. Confusion counts/F1 agree exactly across four cycles at **61/64 phases**;
phases 61–63 differ slightly (maximum phase F1 range 0.00274851). This is
approximately recurring within-motif deterioration, not exact periodicity
or a demonstrated cumulative decline. The interpretation receipt is
`b5b44374ae69b442737d6a27d5da0d9682f16baf5349e7cbc243a17e39c98b9b`.

### 15:41 UTC — long-fit complete-state diagnostics independently accepted

Job11330070 compact audit accepts21 spectra and192 lag64 comparisons, source
closure, scheduler success, actual22 W&B history rows/artifact/exit. The binary
state array remains on scratch and was not independently reanalyzed locally.
Complete H+C PR changes A **1.132672→1.107023**, B **1.128029→1.098119**;
within-motif covariance trace ratios are **0.112735/0.106876**. Hidden-only PR
slightly rises for A and slightly falls for B. Pooled H+C PR drops
**1.830506→1.027825** while trace increases **2.343351×**: between-motif
mean separation affects that pooled estimand. Maximum lag64 relative state
residual **0.01167929** does not certify periodic states/Jacobians.
The hash-bound epoch160 state figure rendered successfully and its PDF was
visually inspected. No causal retention-loss or Floquet claim is accepted.

The Qwen continuous worker heartbeat at15:39:45UTC is current, with42 verified
reviews, no provider error/backoff, waiting for evidence. A new authorized
project-summary task was published at15:40:34UTC for proposal-only critique of
these accepted results; no datasets, weights or credentials were supplied.
No external OpenAI service is assumed; native agents handle implementation
and independent review. Public report and manuscript integration are active.

### 15:42 UTC — proposal review and manuscript integration

Qwen call43 completed with a schema-valid proposal; heartbeat reports43reviews,
zero provider errors, waiting for new evidence. Its caution about mixed geometry
is reasonable, but two statements are rejected: an old October16 engineering-only
status does not describe today's trained-model execution, and its assertion of
missing selected-checkpoint replay contradicts accepted exact native H/C replay.
A generator-phase oracle cannot identify an ordering effect without a matched
ordering intervention. The proposed phase decomposition is exploratory, not
scientific acceptance. Native independent review remains authoritative.

The manuscript now contains the preserved pilot plus the accepted long-fit
behavior/state figures and distinct interpretations. LaTeX builds to23pages
(eightmaintext; graphappendix19–22). All four changed graph pages were rendered
and visually checked; no clipping or final build warnings. Public compact
evidence/report and final scientific review are being reconciled before commit.

### 15:44 UTC — independent scientific review and final publication checks

The independent scientific review accepts descriptive execution with scientific
limitations: actual fit/selection,256confusion rows, fixedwindow reductions,
21spectra/192closure reductions, source/input bindings and plotted bytes pass.
No native inference or statebinary reanalysis is claimed. The reviewer confirms
that the fit stopped at200epochs rather than convergence and that Persistence
still exceeds its overallF1. The report distinguishes PR decrease from
thresholdnumericalrank, which rises within both motifs.

All30published evidence files were hashverified byteexact. No credentials or
tracked symlinks were found. Seven stateplot metadata/display tests pass;
previous codec/evaluator/supervisor checks remain recorded above. Large weights,
arrays,journals and W&Bstreams stay in ignored localruns/cluster scratch.
Next: preregister additional model/seedcoverage, implement TGCNsourceparity, and
complete matchedordering causalretention controls before a collapseclaim.

### 19:23 UTC — next Figure 10 model: T-GCN engineering gate

Resumed the remaining model coverage using the recovered official T-GCN
wrapper/dependency sources. Bounded disjoint native workers own the hidden-only
backend, synthetic calibration, and independent protocol/code review. The
coordinator owns source integration and a single cluv CPU calibration job.
No new empirical fit or held-out evaluation has run in this continuation.

The dedicated backend preserves `add_self_loops=False`, `cached=False`,
`improved=False`, complete hidden state [1,N,H], canonical nodes, strict dual
component weights and source-derived parameter expectations. GC-LSTM formats
and accepted results remain untouched. Calibration will test a tiny dynamic
weighted graph and production tensor inventory before any T-GCN fit.

Actual local cluv is0.1.5. Mila project-filtered scheduler inspection finds no
matching project jobs. Remote `rg` is absent; the inspection was repeated using
available Python rather than treating the failed filter as an empty queue.
Prepared resource envelope: one Mila cluv submission, accountmila/QOSnormal,
main-cpu/Rome,1CPU/2GiB/zeroGPU/10minutes,540swork/570sphysicaltimeout,
no retry/requeue/race, synthetic inputs only. This follows the owner's explicit
Figure10 retraining/reproduction authorization without repeated compute approval.
Exact committed source, admission, output and command will be pinned before launch.
Six pure supervisor admission checks pass; native numerical gates remain pending.

### 19:30 UTC — T-GCN prelaunch review corrections

Independent review and coordinator inspection caught a calibration gap before
native execution: a first update from H=0 cannot test reset-gate gradients.
The frozen synthetic branch now warms A at t0 without gradients, detaches its
nonzero H, consumes B at t1 with gradients, and scores matching targetD at t2.
Both branches use the same explicit0.1 head-parameter fixture to avoid dead
ReLUs, paired dropout RNG and one Adam update. A nonzero reset-gate gradient
is required. This is deliberately synthetic engineering evidence, not a fit.

Backend safeguards now reject parameter/payload aliases before mutation, compare
FP32 readback bytes including signedzero, and enforce normalization/source-to-target
flow as well as no cache/selfloops. The independent supervisor stops by560s,
worker guard by540s and shell by570s. Fourteen backend, seven calibration and
six supervisor tests pass, without native models. Repeated seed0 constructor
byte/RNG checks are being added before source freeze.

The independently downloaded lockedPyG2.6.1 wheel SHA8faeb353f9655f7dbec44c5e0b44c721773bdfb279994da96b9b8b12fd30f427
contains GCNConv SHA036b464769ad1d98501c9fc18ff3132b507890b2c5cb41b3e6f9e1f22d4f57f6,
matching the installed source read on Mila (10,415bytes). No native code was
executed to verify this pin. Three public wrapper/base/head files were staged
create-only/hashverified on scratch. Qwen heartbeat remains current with43
valid reviews, zero provider errors and waiting for evidence.

### 19:35 UTC — actual native T-GCN calibration accepted

Single cluv job **11332092** completed `0:0` in **38s** on `cn-f002`, source
`e0e6c20a72a98134aa1f2a2450f9eeb011718ab2`. The independent Reporter-wrapped
compact audit accepts **248 checks**, **161 byte-exact tensor comparisons**
(maximum absolute error0), four nonzero reset-gate gradient tensors, native
production inventory137472+49537, source closure, actual scheduler/cgroup/argv,
process-group cleanup and actual offline protobuf history/artifact/exit.
The worker reported one chronological W&B history row. Native operations were
not independently repeated by the compact verifier; cloud delivery is unknown.

Result SHA23b68f8c78e0f4b1fd00959c812463318cb687d8b9b95404c709c6ade5001619;
verifier SHA2eb0c905f8767188a1009e4d3d304293f6bf6199635cfed0e2482e010eb60d3d.
This accepts synthetic implementation evidence, not a trained T-GCN benchmark,
competence, causal collapse or Floquet result. Public report and ten exact
compact evidence files are under `doc/auto/tgcn-native-calibration-results-2026-10-20.*`
and `doc/auto/evidence/tgcn-native-calibration-2026-10-20/`.

Next isolated production prefix:256 real chronological training updates on the
same accepted primary conversion, one CPU/2GiB/zeroGPU/30minutes through cluv,
no validation/test/selection/checkpoint. Its own protocol/admission/source are
being reviewed; the calibration itself does not admit that training run.

### 19:44 UTC — production T-GCN prefix source frozen

Independent prelaunch review passes18 prefix/outer tests. The new T-GCN journal
has parent-directory fsync, fd/path/size/prior-byte checks, checked writes,
terminal callback failures and a512-record canonical/hashchain readback.
Model/data boundaries are explicit: production100×128 random initialization,
modelseed0/negative1, exactly256witnessed Adam steps, train payload only,
zero complete epochs, no validation/test/checkpoint/selection/inference admission.
The shell/supervisor/work clocks are1770/1760/1740seconds under Slurm1800.
One new cluv CPU submission is prepared under this separate envelope.

Qwen call44 completed schema-validly, heartbeat current/no provider errors.
Its production-path caution is reasonable; its suggested selected-TGCN-checkpoint
replay is rejected because calibration did not create such a checkpoint.
Its suggested1e-6 parity threshold does not change the frozen exact-byte gates.
Native OpenAI implementation/review workers remain the fallback and scientific
acceptance owners; no unavailable external OpenAI API was assumed or called.

### 19:51 UTC — real T-GCN training prefix accepted

Job **11332237** completed`0:0` in **30s** on`cn-f003`, source
`371aba996af02826df5fee03abb8a5223b9849b5`. Independent compact verification
accepts256 real optimizer updates/lossrows,512 durable paired journal records,
256 actual offline W&B histories, source/data/nativeinventory/scheduler bindings
and reporting artifact/exit. Mean balanced BCE-sum training loss is
**1.3214061078615487**. This is zero complete epochs and no validation/test,
selection, checkpoint or competence/collapse result. TransientH is training
carry produced before the final weight update, not frozen-weight replay.

Verification001 passed its core audit and wrote a provisional result, then
its final logging metric`raw_loss_rows_verified` was rejected by the existing
privacy guard. That failed reporting receipt was retained, and canonical
acceptance was NOT promoted. A metric-only repair (`training/loss_rows_verified`)
plus unique offline verification runID passed7pure tests/peerreview.
Verification002 reused the identical collected artifacts, completed SDK finish,
and only then promoted acceptance. **No cluster/model/training retry occurred.**
Final verifier SHA`dd077db2f4c854953010b3c0340ba5a77a160d96d7a0d1b08208ed7954b6d60d`.
An earlier sourcefreeze check caught whitespace in one test; it was corrected
in371aba99 before source staging/submission. Numerical/protocol bytes unchanged.

Public prefix report/evidence and independent interpretation are being integrated.
The next bounded implementation node is a dedicated hidden-only T-GCN
checkpoint/catalog+fit adapter with validation-only selection and exact fresh
train-plus-validation replay; the full-fit allocation is not yet submitted.

Public calibration/prefix evidence inventories now independently reconcile all
10 and19 listed file hashes, respectively. The final independent engineering
review is public; no source, graph, weight, recurrent/RNG payload or W&Bstream
was copied into the prefix evidence. The full-fit readiness audit names the
reusable opaque-state epoch core and required dedicated T-GCN fit adapter,
hidden-only codec/selection, exact2816-graph replay and subsequent controlled
held-out pass. First256loss/init identity is now a concrete parity reference.
All calibration/prefixcode, protocols and verified reports are committed;
unrelated owner changes remain untouched.

### New owner-directed subgoal — repetition length and memorization

The owner requested longer A/B dwell experiments and a representation-collapse
and memorization research goal. The existing tool-level broad goal is unfinished
and paused, so create_goal refused replacement; no completion/status change was
invented. The immediate subgoal is recorded in
`doc/auto/repetition-memory-goal-2026-10-20.md`.
Mila SSH and a successful queue read show no current representation-collapse
jobs. A new generated-forcing diagnostic is being prepared with fixed accepted
GC-LSTM epoch160 weights, train-only A/B motifs, dwell8..256, both starts and
2048 steps each. Native implementation/protocol/review have separate owners.
No new numerical result or launch is claimed at this preparation stage.

### 20:11 UTC — repetition sweep launched and running

Reviewed source3541ae02441357664a9328fcfebd515a472d9668 was pushed and staged
in its own scratch worktree. Native cluv0.1.5 submitted exactly one job11332452;
actual scheduler reports RUNNING oncn-f003. OneCPU/4GiB/zeroGPU/twohours,
worker7080s/shell7140s, no retry/requeue/race. Offline reporter and first
condition rows exist. An initial local cluv preview failed because its isolated
controller lacked the local jobscript; copied reviewed script before preview
and submission. No failed cluster job or numerical retry resulted.

The worker/protocol/source review and14focused tests passed. Exact immutable
preview, launch, create-only submission intent and returnedjob are under ignored
`runs/ops/repetition-dwell-20261020/dwell-20261020T200909Z-3541ae02/`.
The run keeps selected state clouds on scratch. Independent compact reduction
and plots are being reviewed. A postfreeze suggested prose clarification of
last1024 supplemental windows was reverted; executed protocol remains exact.
No sweep results have been accepted at this point.

### Repetition sweep completion — independently accepted descriptive evidence

Cluv11332452 completed0:0 in11min26s oncn-f003, MaxRSS614352K. All12conditions
and24576rows were collected with matching remote hashes. Source3541ae0/result
92242b8ad720a3e42b512eceb64f75e4798d0a79a1969d180e932787c517a60f.
Independent row reduction verifies chronology, confusion/F1, fixed windows,
controls and compact spectral arithmetic. A separate actualprotobuf audit
verifies384offlinehistories, artifact, terminalexit and source/scheduleridentity;
its acceptance is published only after its own reporter finishes successfully.
Online delivery unknown; binary state clouds not independently reanalyzed.

Local reducer001 failed before numericalwork because sandbox denied W&BIPC.
Its receipt remains; distinct authorizedreducer002 completed. No clusterretry.
FourPNG/fourPDFplots rendered and visually checked. Exactmarker was validated
throughV8 because noNodeCLI wasavailable. Compactpublicmetrics/CSV/plots,
verified source/reporting/schedulerprovenance and hashes are under
`doc/auto/figures/repetition-dwell-2026-10-20/`.

Scientific acceptance: meanF1 falls .8894→.7665→.6975→.6565 overL32..256,
whilepersistenceimproves. Primarydrop is nonmonotone at256/startup-sensitive.
H+C PR rises early-to-late in everyL8..128condition; lowertrace is not equivalent
tolowerdimension. No memorycapacity, harmfulcausalcollapse orFloquet claim.
Report `doc/auto/repetition-dwell-results-2026-10-20.md` preserves allconditions,
limitations and independentcritique. One-bitmemorycandidate documented butnotrun.
Currentfixedcheckpoint diagnosticcompleted; broaderresearchgoalunfinished.
