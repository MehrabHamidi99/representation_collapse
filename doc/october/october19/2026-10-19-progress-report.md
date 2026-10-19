# Progress report — 2026-10-19

## Figure 10 reproduction: durable optimizer-attempt journal

Added `Figure10AttemptJournal` and focused tests. The journal takes exclusive
ownership of a fresh attempts directory, fsyncs each optimizer intent before
the fit callback returns, pairs the observed result to the intent hash, and
refuses reopening or resuming an incomplete attempt. A callback failure leaves
the attempt charged and ambiguous. A failed optimizer call is terminal for the
fit journal, consistent with the fit core re-raising rather than continuing.

Independent review found that the first draft accepted inconsistent cumulative
optimizer counters and allowed later attempts after a raised optimizer call.
Both issues were corrected. The journal now checks and records the completed
epoch and confirmed optimizer-step counters, including a raised call whose
optimizer step is explicitly confirmed complete; it rejects any later intent
after a failed callback. Independent re-review accepted these changes.

Added `run_journaled_figure10_fit`, which joins the native fit adapter to the
attempt journal and selection sink. It checks exact component types and
matching run/admission context before fitting, wires all callbacks, runs once,
and closes the journal on validation or fit failure. Independent review
prompted exact-type checks and cleanup of a rejected subclass journal through
the base close implementation. Its focused suite passes **6 tests**. The final
Figure 10-focused suite passes **107 tests**, with **3 platform-specific
skips**; the journal and fit lifecycle subset passes **23/23**. `git diff --check`
passes.

These are synthetic engineering checks: no graph model was constructed, no
dataset or held-out test target was opened, and no training or cluster job was
run. The output directory and lock are an accounting boundary, not a
tamper-proof trust root or a recovery mechanism.

## Autoresearch and execution status

Autonomous local research advanced the reproduction pipeline today through
implementation, adversarial tests, and independent review. The last recorded
successful Qwen liveness check is **2026-10-16 13:54 UTC**: structured status
records 33 identity/schema-verified replies, a heartbeat age of 51.8 seconds at
the check, and no call in flight. I could not verify its current process or
make a new Qwen call today.
The current Mila attempt still requires interactive email OTP, so current
cluster queues, cluv connectivity, and the historical shared Mila allocation
are unverified. Do not interpret the historical status as a current worker or
active allocation. A native independent reviewer checked the implementation;
no Qwen/Cairn request, cluster submission, or compute mutation occurred in
this session.

## Remaining release gates

The callback wrapper is not a complete training/checkpoint/reporting runner.
The frozen readiness check still blocks launch because the local numerical
dependencies do not match the pinned environment, the installed GC-LSTM
source and source/data/
environment/admission receipts are not verified, and the native checkpoint
and reporting runner is not implemented. Next work is to complete that runner,
verify external provenance and admission, then complete selected-checkpoint
hidden/cell replay parity (node2b) before any
test split access. The original Figure 10 curves/checkpoints also remain
unresolved in the audited scopes. No Figure 10 curve or representation-collapse
finding is accepted.

## 20:21 UTC — live cluster progress check; Mila access restored

Read-only SSH now succeeds to `login-3.server.mila.quebec`. Mila's current
`squeue --me` contains no representation-collapse jobs. Its running
`radius-coordinator-20261019` (11319082) and pending `si-observer` are other
projects' entries and do not establish this project's research progress.
Shared metadata identifies the current radius coordinator and Tamia socket;
no allocation entry, route use, source synchronization, submission, or remote
mutation was performed.

Fresh `sacct` records reconfirm primary-conversion job11273685 completed with
exit `0:0` in 126 seconds, runtime capture job11280346 completed in 12 seconds,
the first node2a attempt11280406 failed in one second, and corrected synthetic
native parity job11280629 completed with exit `0:0` in 14 seconds. These are
the previously documented engineering results; no new training result appeared.
The remote source checkout remains `40942e5`, behind pushed local commit
`d4fee98`, so the new journal and orchestration helper are not deployed there.

The remote legacy worker at `qwen-continuous-20261013T093936Z` has explicit
terminal status: stopped by its STOP file at 2026-10-14 08:52:40 UTC after nine
provider attempts. A login-host process probe found no `qwen_worker.py` PID;
that probe alone does not inspect processes on every compute node. This
legacy worker is distinct from the newer Qwen evidence recorded locally on
October16 (33 verified replies), whose current liveness was not established.
No new Qwen request or currently active project autoresearch worker is
confirmed. Alliance queues and current native cluv version were not verified
in this check. The preceding OTP blocker is superseded by this successful
Mila authentication check.

## 20:35 UTC — native training-prefix diagnostic prepared and reviewed

Adopted and repaired the previously untracked canary module, supervisor, job
wrapper and tests; added focused supervisor tests. Independent review accepted
the resulting source for the separately frozen train-only diagnostic in
`doc/auto/figure10-prefix-diagnostic-2026-10-19.md`. Repairs include durable
pre-update intents, explicit optimizer-step witnesses, conservative failure
reconciliation, compatible scheduler receipt keys, authoritative Slurm/cgroup
admission and a job-start-relative deadline. The wrapper accepts explicit
cluv program arguments and retains process-group cleanup and source-only
bootstrap loading. Combined canary/supervisor/journal/orchestration tests pass
52 checks (4 platform-specific skips); shell syntax passes.

Fresh Mila inspection found no project job, a clean remote checkout, the
verified conversion manifest SHA `a6ff50183de570ff17cfbe92b3f1976bac9d2d85557ecc11379e24b2c737d1ce`,
node2a receipt SHA `550fa0aa5810819c8f154bbc5edb2ea8b4cd7345dcc8681394295279c7109a96`,
and the pinned native CPU environment. Native cluv 0.1.5 and uv 0.9.18 are
installed in the local controller environment. The planned single Mila
submission requests account mila/QOS normal/main-cpu, 1 CPU, 2 GiB, 600 seconds
and zero GPUs, with a 540-second job-start-relative work deadline and
570-second outer guard. It attempts 256 training targets with no validation,
test, checkpoint or selection. This scope is covered by the owner's existing
Figure 10 compute authorization. Source commit, exact admission and actual
job/outcome will be recorded below after staging and submission; no numerical
result is accepted at this preparation stage.

A final pre-submission integrated receipt test found and corrected an omitted
`run_id` in the actual emitted supervisor receipt and a nanosecond-to-seconds
rounding discrepancy in supervisor/child deadline comparison. These fixes
were made before any submission and preserve the same numerical protocol.

## First prefix submission — job11320580 failed before numerical startup

Native cluv 0.1.5 submitted source `7dadf7e` with admission SHA
`e27e91751e6dee8368f71ccdc79591d1d6791d39213a7120a944da8aff896c1a`.
The exact sbatch receipt identifies Mila job11320580. Cluv's polling API returned
no winning job because it failed rapidly; fresh accounting and the retained
native submission log reconcile the outcome: FAILED, exit `2:0`, one second.
Slurm copied the shell job into its spool directory, so `dirname($0)` resolved
the Python supervisor under `/var/spool/slurmd/job11320580/` rather than project
scratch. Python failed to open that file; no supervisor, model or optimizer
update ran. The admission/run ID and submission/reconciliation records are
preserved under ignored
`runs/ops/figure10-prefix-20261019/prefix-20261019T204002Z-7dadf7e1/`.
The correction uses the fixed admitted scratch source path and a regression
that runs a wrapper copied to an unrelated spool directory. Any subsequent
attempt uses a new source commit, admission and run ID under the existing
compute authorization; this failed run is never reused.

## 20:59 UTC — real optimizer progress verified; autoresearch participation refreshed

The corrected unique run `prefix-20261019T204605Z-c72323cd` was submitted via
native cluv 0.1.5 as **Mila job11320665**, from source
`c72323cd36a3aafd4d7f6cc22e4a1d81bf36bc3e`, admission SHA
`262bfcffee21f2e7b2a2c4caedfedcec6408d769c8b3a7a2d8852f5f3c0d72b4`.
It ran on cn-f002 and completed `0:0` in **156 seconds**, with one CPU, 2 GiB
and zero GPUs. It performed **256 actual optimizer updates**. The update
loop took 46.886 seconds; worker total was 148.597 seconds. No validation,
test, checkpoint selection or complete epoch occurred.

An independent verifier accepted all 256 durable intent/outcome pairs, exact
target/predecessor chronology, confirmed optimizer-step counts, and all raw
losses; recomputed mean loss is **1.2817290551029146**. The actual offline W&B
transaction has 784 CRC-checked fragments and 781 decoded protobuf records;
all **256 history rows** match losses/steps, and terminal summary/exit records
complete correctly. The owned process group was empty. Largest sampled cgroup
memory was 482,934,784 bytes; true peak is unavailable. Offline reporting is
accepted; remote dashboard delivery is unverified. The bounded metadata
collection comprises 520 files/486,497 bytes plus the separately hashed
1,324-byte result artifact, with no raw dataset/weights transfer. Verification
JSON SHA: `a78c4c87756b302af07e45f9697897f3491f83802a23cd39a40ce01fd26c8948`.
See [the verified results](../../auto/figure10-prefix-results-2026-10-19.md)
for paths, exact identities, limitations and the next implementation task.

Fresh local process and heartbeat checks also recovered the newer continuous
Qwen worker: PID84719, start October14, in the legacy-audit isolated Python
environment, output `runs/qwen/local-continuous-20261014T1650Y/`. Its heartbeat
is current and it has not been restarted or duplicated. The old remote cohort
is a separate stopped worker. A 1,483-byte public verified-result summary was
added to its existing approved evidence root. Call34 completed at20:56:26 UTC
with matching requested/returned `Qwen/Qwen3.8-Flash-Next-FP8`, valid schema,
5431 reported tokens and zero actions. The context budget omitted the new
result despite triggering on the change; its stale GRU-codec proposal was
rejected. An explicit immutable 1,245-character coordinator task then
foregrounded the actual Figure 10 next step without changing excerpt scope.

Call35 completed at20:58:31 UTC, bound to
`figure10-fit-only-next-20261019`, matching the required model, valid schema,
5721 reported tokens, and zero actions. It identified the missing factories,
seed receipt, reporting fanout and finalization reason mapping. Its assertion
that the prefix result alone releases a full fit is incorrect: production
runner/admission integration must still pass review. The narrow checklist is
advisory; no scientific result or submission authority follows. The worker
is alive locally with35 attempts and no in-flight call at the check; it is
not a cluster-hosted service. Its existing STOP file is the control for future
provider requests. No API credentials or endpoint configuration were copied.

The next implementation is a fit-only runner joining the already implemented
adapter, selected-candidate sink/ledger, journal, data opener and reporter.
Native selected loading and the s[2816] H/C observer are later node2b work;
they must precede held-out test access, and do not require another redundant
pre-fit synthetic training gate. No Figure 10 curve, competence or collapse
claim is accepted by this diagnostic.

## 21:12 UTC — complete-epoch Figure 10 fit launched

User asked to reproduce; the authorized next actual training job was submitted
through cluv0.1.5 rather than stopping at the prefix diagnostic. Native full-fit
runner, selected loader/replay evaluator and cadenced durable attempt journal
passed independent review. Coordinator ran31focused unittest cases. A mistaken
pytest invocation found no pytest in the local controller environment; running
the repository's unittest cases directly passed. No environment was modified.
Source committed/pushed as `6d89c657c41cf38ff3dd074a1e886b4bd2b26c4c`.

Mila remote clean checkout was advanced to that exact source after confirming
no representation-collapse job was active. The independent conversion receipt
was rehashed at its actual path, matching
`9e279518ab997e8a7596a3a23f67fd3d9ea0362d761d814149fb5eec8e88e2c8`.
A fresh create-only admission bound complete package source bytes, protocol,
config, existing environment/node2a/conversion receipts and arguments. Admission
SHA `e31d22396f77c86715cff7a39416df092b7fe61e1d8e0a317ebdcc7cac19d0c7`.
The isolated controller preview resolved exactly one full-fit wrapper and a
one-CPU/2GiB/zeroGPU, main-cpu, normal QOS/account mila, one-hour Slurm request.
The pre-submission intent and cluv result are retained under ignored
`runs/ops/figure10-full-20261019/full-20261019T211030Z-6d89c657/`.

Cluv returned **11321105**. Scheduler independently showed RUNNING on cn-f004
at21:11:21UTC. Remote output is scratch
`representation-collapse/runs/figure10-full/full-20261019T211030Z-6d89c657/`;
supervisor receipt and journal are adjacent hidden control files. Its sibling
`.STOP` control (run ID followed by `.STOP`) stops this job. No requeue, retry,
race or unrelated allocation is used. Source remains immutable while it runs.

This is seed0 corrected capped GC-LSTM reconstruction under existing v1, not
the paper's completed multi-model/five-seed Figure10. Optimizer/epoch outcomes
are pending, and held-out data remains sealed. A separate native evaluation
CLI and explicit receipt contract are being prepared locally; complete H/C
replay from zero is mandatory before test. A separate plotter will consume
only independently accepted evaluation artifacts. No empirical collapse claim.
The existing local continuous Qwen heartbeat remains current at21:10:31UTC,
35reviews, idle awaiting evidence, zero current requests; it is not restarted.


## 21:18 UTC — first complete validation epoch and native checkpoint

Job 11321105 is RUNNING. The first complete epoch stored 2,559 real updates,
training mean loss 0.631418777460137, and 256-target validation mean F1
0.07924821123094206. `validation-001.json` and a committed, nonfinal selection
head exist; the candidate is epoch 1. Epoch 2 has started. This validation
result is neither held-out evidence nor competence or collapse acceptance.
The fit-run admission identity is the actual supervisor receipt SHA
`5085efc15dc87c7ea981a924a6c5443d3cd9b216e62f3f71d4f93990d4fedb32`,
which differs intentionally from prequeue admission SHA `e31d2239…`.

Selected-evaluation CLI, independently reviewed diagnostic-v2 contract, source
supervisor and hash-bound plotter are committed/pushed as `fa4ba04`. Twenty-four
combined evaluation/supervisor/plot tests pass. These source changes remain
local; the active remote checkout stays `6d89c657` until fit exit. The evaluator
now embeds and validates its live supervisor receipt identity. No evaluator was
submitted and no held-out split was opened. An explicit proposal-only Qwen
task asks for scientific F1 interpretation controls using verified fit facts.

The coverage audit confirmed eleven distinct methods and two panels. Five
seeds is a paper fact, but historical seed values 0–4 are unverified. Paper
stopping has patience 100 without an epoch cap; retained public wrapper maximum
100,000 differs from the corrected v1 pilot cap of 200 and one hour. Future
longer runs require a versioned contract, not relabelled v1 results. The source
also explicitly describes F1 decreasing between change points; that within-motif
effect is distinct from cumulative decline across periods. Deterministic
persistence controls are being prepared from known generator metadata before
test; they do not alter the fit.

## 21:29 UTC — analytic controls verified; second validation improves

Epoch 2 completed with training mean loss 0.3152378972688193 and validation
mean F1 0.1687550041233902, selecting epoch 2 over epoch 1. All 2,559 updates for epoch 3 have also completed (7,677 total across the
three epochs); its validation is pending at this observation.
Job 11321105 remains RUNNING. No held-out model data has been opened.

Before any model test access, committed `0139fbd` froze analytic controls.
The actual metadata-only run then produced persistence F1 0.96875 overall,
0 at eight transitions and 1 at the other 248 targets; known-phase oracle
F1 is 1 and always-negative F1 is 0. Its graph metadata contains 100 and
120 ordered positive edges with zero overlap. Independent reduction verified
all 768 control rows and 771 actual offline W&B rows, CRCs, protobuf sequence,
compact artifact and completed reporting receipt. The public results report
and JSON distinguish these analytic controls from learned-model evidence.
No data split or weight file was accessed. Online delivery remains unverified.

Source commit `81715b3` froze within-motif descriptive windows before any
actual per-phase model curve: phases 1–8 versus 24–31, excluding transition
phase 0, for A/B separately and balanced pooled values. The new three-panel
plotter passed seven synthetic tests and visual QA. These correlated cycles
supply no inferential uncertainty or independent model replication.

Rechecked published arXiv PDF text and visually rendered page 17 from the
retained local copy `literature/papers/dizaji2025tgrab.pdf` (SHA
`16e117520457a0541efa729cc4cbc06e851833191b72f4f896b011c0e265ff25`).
Published Figure 10 has the same Pdet(2,32) behavioral target and explicitly
describes a decline between change points. This supports phase-aligned
inspection, not a representation-collapse conclusion. Web PDF screenshot
failed with cache miss; local PyMuPDF rendering provided the visual check.
The local draft/public PDF are not claimed byte-equivalent.

Qwen call 36 completed with verified model/schema identity, proposal only and
zero actions. Its general distinction between F1 and memory loss is useful;
its unregistered untrained-model overlay and extra acceptance gates were not
adopted. A verified analytic-control summary was added to the existing approved
evidence feed. Native agents are preparing single-pass complete H/C capture
and a separate frozen descriptive trajectory analysis; neither changes model
updates, held-out selection or the one-test-read restriction.


## 22:02 UTC — six completed epochs; fresh 48-hour fit admitted and held

The pilot job11321105 is still running on cn-f004. Six complete epochs
are retained; epoch6 training mean loss is0.13442240720458296 and validation
mean F1 is0.34604334968252604. Epoch7 is in progress. No model test payload
has been opened. All first256 pilot losses exactly match the independently
accepted native prefix, with matching chronology; the local parity record is
under the ignored full-fit operations directory. Selection remains provisional.

Committed/pushed source1891bdd adds the separately versioned long-fit v2,
chunked durable journal, CPU admission supervisor, compact evaluation W&B
artifact and the v1 normal-completion tuple-return correction. The active v1
checkout stays6d89c657: the one-hour deadline path does not exercise that
normal-completion bug. Combined focused validation passed89tests. The
new long fit preserves numerical parameters, CPU FP32, chronology, seeds and
validation-only selection. It has a fresh run/contract, 200-epoch cap, patience100
and 48-hour operational maximum; it is not an extension or relabeling of v1.
Its first256 losses must exactly match the already accepted reference before
update257. See the frozen long-fit protocol.

Cluv0.1.5 submitted long job11322268 from immutable source1891bdd in a separate
scratch source-long checkout. Resources: accountmila, normal QOS/main-cpu,
cn-f004, oneCPU/2GiB/zeroGPU, 48hours, no retry/requeue/race. Scheduler
initially reported PENDING/Resources. The coordinator deliberately applied
` scontrol hold 11322268 ` so the next free CPU slot can finish the pilot's
short selected-evaluation and descriptive state analysis before the 48-hour
fit occupies it. This is currently JobHeldUser, not running or failed. Release
the same job after these short steps; do not submit a duplicate. Unique run:
long-20261019T215117Z-1891bdd5, admission SHA
3169ac25bc98beb153a331f74823617c4b7f6a63add940fc8d51ae3194e5673d.

A separate read-only local long-job monitor was independently reviewed:
9tests pass, BatchMode-only bounded SSH, five-minute polling with backoff,
owner-scoped atomic journals/status and monitor-only STOP. Held PENDING jobs
are not called failures; there is no release/cancel/submit code or direct model
call. Optional already-authorized Qwen tasks occur only at prefix parity, ten
complete-candidate-epoch milestones or terminal evidence, with bounded project
excerpts and no scientific acceptance. Actual monitor launch is still pending.
