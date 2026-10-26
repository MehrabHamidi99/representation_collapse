---
name: gpu-efficiency
description: Diagnose GPU waste in Slurm research jobs, prepare measured resource requests, and validate throughput improvements without changing the scientific experiment. Use for utilization alerts, GPU job submission or scaling, stalled training, and performance reviews on Mila, Tamia or other clusters.
---

# GPU efficiency

Optimize useful, scientifically valid work per allocated GPU-hour. A running
scheduler state, high memory usage or a successful exit does not establish
efficient training.

Read [GUIDELINES.md](GUIDELINES.md) before making allocation or performance
decisions. Use [commands](references/commands.md) for bounded inspection and
the [evidence template](references/efficiency-report-template.md) for the result.
Installation and project routing are in [ADOPTION.md](ADOPTION.md).

## Workflow

1. **Attribute the work.** Identify the project owner, cluster, job/step/array
   IDs, actual allocation, source/config/data identity, current phase and
   existing authority. An account-wide warning does not attribute waste to
   every project. Retain its reporting window and metric definition if known.
2. **Establish progress.** Inspect actual step/token/sample counters, recent
   history and checkpoint timestamps. Reconcile Slurm accounting. Separate
   queue time, staging, initialization, optimization, evaluation and shutdown.
3. **Measure the cause.** Use a bounded representative window on the allocated
   hardware. Record each assigned GPU, metric name/provider, missing samples,
   useful throughput, time waiting for data, memory, and resource cost. Inspect
   all ranks/nodes when diagnosing distributed imbalance. Missing telemetry is
   unknown, and a terminal `nvidia-smi` snapshot is not a training average.
4. **Choose the smallest valid intervention.** Diagnose preparation, I/O,
   rank placement, CPU contention, synchronization and tracking stalls before
   adding GPUs. Preserve effective batch, loss semantics, data order, optimizer,
   precision, schedules, checkpoint identity and evaluation unless the owning
   plan explicitly admits a change. Equal effective batch alone does not prove
   equivalence for batch-dependent losses such as SIGReg.
5. **Validate before scaling.** Reuse recent comparable evidence or run a
   finite real-data pilot within the project's authority and budget. Compare
   before/after useful throughput and GPU-hours on matched work; check numerical
   behavior and strict resume where affected. A toy pass alone does not prove
   real-data performance. Gate new long jobs/sweeps on this evidence or an
   explicitly documented, justified exception by the owning operator.
6. **Leave a compact handoff.** Record evidence, diagnosed cause, tested change,
   resource request, observed gain, uncertainty and next action. Use deterministic
   polling for routine observations; spend model calls only on meaningful
   diagnosis or decisions. Never start busy-work to raise a utilization metric.

## Authority

This skill supplies an operating method, not permission to cancel jobs, alter
an immutable campaign, increase budgets, access another owner's allocations,
or publish code/telemetry. Continue already-authorized work without redundant
approval. Follow the project's admitted stop/resume process for real failures;
do not cancel solely from a low percentage or retrofit a new admission rule
onto running jobs. Read-only model workers remain read-only. Release an owned
idle interactive allocation when authorized and its work is complete.

Scientific settings and scheduler constraints belong to the adopting project.
Do not import this package's originating project's job IDs, GPU limits,
datasets, four-GPU topology, W&B permissions or cluster credentials.

## Current documentation/reporting policy (2026-10-26)

Follow root `AGENTS.md` and `docs/October/2026-10-26/cross-cluster/guidelines/documentation-layout.md`.
That policy supersedes older notebook paths and W&B scope in this skill. Use dated
cluster/subproject/type docs and Git synchronization. W&B covers monitored scientific
result jobs; preprocessing, progress-only tasks and infrastructure checks are excluded.
