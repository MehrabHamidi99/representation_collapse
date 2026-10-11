# Autonomous research and agent handoff protocol

This project studies dimension collapse in recurrent models, beginning with
the graph temporal preliminary material. The user requested a senior research
coordinator with subagents. The current setup supports bounded work in active
sessions; unattended daemons, scheduled model calls, and remote job submissions
are not enabled. The user's willingness to approve Compute Canada work is a
pending approval, not a standing resource envelope.

## Coordinator responsibilities

Start with `configs/research-plan.json`, the tracker, and the latest dated
notebook. Choose ready tasks with scientific dependencies and acceptance
checks. Keep an explicit distinction between reported source claims, calibrated
synthetic fixtures, reproduced real-model measurements, and proposed studies.
The goal is evidence that can reject the hypothesis as well as support it.

Keep one owner per output or run. Parallelize independent source audits, asset
inventory, and implementation/review with disjoint paths; sequence dependent
calibration, reproduction, and analysis. Use the session's available slots.
Review artifacts before releasing dependencies. Follow `model-routing` using
the live tool surface; inherit the coordinator model unless an override is
explicitly requested by the user or applicable instructions.

## Every delegated assignment

Give the worker this compact contract, populated with actual values:

```text
Task/node and protocol revision:
Question and deliverable:
Read-only evidence paths and relevant source locations:
Owned output paths; paths managed by other workers:
Permitted local commands and side effects:
Inputs and dependencies already verified:
Acceptance checks and scientific limitations to preserve:
Time/iteration bound; stop on missing inputs or unresolved assumptions:
Return: files changed, commands/check results, evidence paths/hashes,
        unresolved questions, negative results, proposed next node.
```

Prefer a single artifact or a bounded repair per assignment. Do not send entire
logs when exact paths and a failure excerpt suffice. Workers may inspect their
scope and create reversible local artifacts; no worker independently authorizes
GPU submission, cluster racing, paid external services, or hidden continuation.
A task requiring outside inputs returns a documented blocker and useful partial
work. The coordinator persists the handoff, preserves failed attempts, and
replans smaller work when a task times out.

## Durable notebook

Use dated records under `doc/`, following the repository's existing notebook
convention. Each entry captures the question and protocol revision, execution
environment and commands, task owners, artifact paths/checksums, operational
status, scientific decision, limitations, failed/negative findings, and the
next actionable node. Update `doc/TRACKER.md` and the plan with links to those
records. Never use an agent transcript as the only evidence for a result.

Preserve code/data/checkpoint identities and enough arrays to recalculate
metrics. Use the artifact and measurement contract in
`skills/experiment-loop/references/workflow.md`. Claims based solely on a figure
or manuscript remain attributed observations until raw outputs and code are
found and independently checked.

## Compute and monitoring

Prepare a concrete Cluv submission or racing plan with pinned code/environment,
data locations, accounts, requested CPUs/GPUs/memory/walltime, allowed clusters,
maximum concurrent pending/running jobs, total allocation ceiling, retry policy,
artifact retrieval, and loser cleanup. Submission waits for approval of that
plan and its resource limits. If no applicable limits are recorded, readiness
checks fail closed for remote allocation. No inherited Tamia/H100 entitlement,
GPU-hour allowance, or paid-model wake cap comes from the template.

Observation should be deterministic: inspect actual scheduler state and output
manifests; wake a reasoning worker only for new evidence or an actionable task.
Do not implement an unattended dispatcher or scheduler in this milestone.
After an ambiguous remote submission, reconcile jobs by unique experiment
identity before retrying. A stopped race must confirm cancellation of losing
jobs and account for every allocation. A completed process is separately
verified for artifact integrity and reviewed for scientific interpretation.
