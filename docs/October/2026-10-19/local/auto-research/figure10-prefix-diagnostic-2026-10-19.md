# Figure 10 training-prefix diagnostic — 2026-10-19

Protocol: `tgrab_figure10_256_prefix_runtime_diagnostic_v1`.
Frozen before this diagnostic's first native execution. The owner's Figure 10
compute authorization applies; this document records a concrete execution
scope and does not change the reconstruction candidate or its held-out gates.

## Question and scope

Can the pinned native N=100, H=128, K=2 GC-LSTM and link head execute 256 real
CPU optimizer updates on the verified Pdet(2,32) training prefix, with finite
loss, correct chronology, durable attempt accounting and complete reporting?
This is an engineering diagnostic. Its outcome cannot establish competence,
full-fit cost/readiness, Figure 10 reproduction or representation collapse.
The conditional validation replay, warmup/windows and phase profiling in the
separate throughput-readiness protocol remain unexecuted by this diagnostic.

## Fixed numerical work

- Read only the converted training split [0,2560). Attempt targets t=1..256,
  starting from zero hidden/cell state; detach incoming hidden/cell state,
  consume G[t-1], then score/backpropagate/update against G[t]. No shuffling,
  validation, test, selection or
  checkpoint publication. There are zero complete epochs.
- Use the exact existing `configs/graph/tgrab-reconstruction-v1.json` bytes,
  primary conversion manifest, upstream wrapper and installed cell pins.
  Preserve seed 0, negative seed 1, CPU FP32, one numerical thread, H/C node
  ordering, loss/head dropout, Adam ordering and all registered hyperparameters.
- Require the independently accepted node2a synthetic native checkpoint,
  forward/loss/gradient and hidden/cell round-trip receipt before any update.
- Fsync an intent before every backend update. Persist returned-call and
  optimizer-step status separately; a missing outcome stays charged/unknown.
  Record returned updates before checking the diagnostic loss. Stop on any
  nonfinite, identity, persistence, reporting or numerical failure. Never
  retry an optimizer attempt or resume a partial diagnostic.

## Execution envelope

One Mila job, account `mila`, QOS `normal`, partition `main-cpu`, one node,
one task, one CPU, 2 GiB, zero GPUs, Slurm walltime 600 seconds. No array,
race, requeue or automatic resubmission. Work has a 540-second deadline derived
from job-wrapper start, an import-inclusive 570-second outer timeout and an
owned process-group cleanup reserve. Queue waiting consumes no work deadline.
STOP applies to the unique attempt and treats any directory entry as a request.

The job-specific admission binds the committed source closure, interpreter,
environment receipt, node2a receipt, data/config identities, exact argv,
resources, run/output/STOP paths and relative timing. Verify fresh scheduler
state, actual cluv 0.1.5 behavior and no existing run before submission. Actual
Slurm allocation and effective cgroup limits must match at execution. All
source, environment, inputs and outputs stay under project scratch. Never
change other projects' allocations or shared infrastructure.

## Reporting and acceptance

Enter explicit offline W&B reporting before train payload/model access. Keep
live loss records, resource samples, terminal state, source/input identity,
supervisor receipts and fsynced intent/outcome records. Preserve the offline
transaction for authenticated sync; remote dashboard delivery remains a
separate verification step. No credentials, raw data or weights are uploaded.

Engineering acceptance requires exit 0, 256 resolved attempts, 256 confirmed
single optimizer steps, 256 finite nonnegative loss values, exact t=1..256
chronology, empty owned process group and a complete offline reporting receipt.
Independently recompute counts and mean loss from retained records and verify
remote/local artifact hashes. A failed/short run reports actual resolved and
unresolved counts. Aggregate update throughput describes this prefix only.
No scientific claim follows from a passing diagnostic.

After a diagnosed execution failure, preserve its run ID/artifacts before
preparing any new unique attempt under the owner's existing authorization.
Before a full reproduction fit, complete production runner/provenance
integration and review. Selected-checkpoint node2b replay parity remains
mandatory before opening any held-out test bytes.
