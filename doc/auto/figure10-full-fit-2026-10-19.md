# GC-LSTM Figure 10 corrected reconstruction: complete fit admission

Frozen before this fit sees validation: the first actual complete-epoch run
uses the existing `tgrab-reconstruction-v1.json`, seed 0, negative seed 1,
100 nodes, hidden width 128, K=2, CPU FP32 and one CPU thread. The pinned
Pdet(2,32) conversion separates training [0,2560), validation [2560,2816),
and held-out test [2816,3072). Historical checkpoints are unavailable; this
is a prospective corrected reconstruction, not historical bitwise replay.

The unchanged v1 pilot has at most 200 complete epochs, loss patience 100
and a one-hour entire-pilot cap. This fit allocation reserves 3480 seconds
for work and 60 seconds for graceful finalization; its shell stops by 3570
seconds within Slurm's 3600 seconds. Deadline-truncated training must be
labelled capped. No claim of converged paper-equivalent training follows.

One cluv submission: Mila, account mila, QOS normal, main-cpu, one node,
one task, one CPU, 2 GiB, zero GPUs, one hour, no requeue or automatic retry.
Use a unique run ID under scratch `representation-collapse/runs/figure10-full/`.
The exact source commit, complete source closure, interpreter, conversion,
environment, node2a receipts, arguments and scheduler identity are bound in
the create-only admission and live supervisor receipt. A sibling `<run-id>.STOP`
terminates this job; immediate supervisor cleanup may preempt graceful STOP
finalization. Other projects' allocations and sockets are outside this job.

Durable optimizer intent/outcome records precede reporting. The journal
checks ownership and the pending record every callback, audits all prior
bytes every 256 callbacks, and audits before publishing candidates and
finalization. This changes storage overhead, preserving optimizer chronology,
batch semantics, precision, and training/validation policy.

Only complete epochs with all 256 validation rows and exact complete H/C
terminal state can become selected candidates. Interrupted partial epochs
remain evidence but are never selected. Offline W&B records accompany raw
durable observations; dashboard delivery requires separate verification.

This job has no held-out opener. Subsequent evaluation requires independent
fit-artifact acceptance, an externally pinned final ledger head, a separately
frozen evaluation contract, strict dual-component load, and exact train plus
validation terminal-state replay before test access. Figure 10's behavioral
F1 curve alone does not establish representation-dimension collapse.

Engineering review: eight full-fit, seven evaluator, thirteen attempt-journal
and three supervisor tests pass in the coordinator's 31-test focused suite.
Independent review also accepted the fit/supervisor/journal lifecycle.
