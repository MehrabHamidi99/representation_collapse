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
