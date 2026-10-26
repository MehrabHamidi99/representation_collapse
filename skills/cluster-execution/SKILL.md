---
name: cluster-execution
description: Prepare, validate, submit and reconcile owned Slurm or Cluv research attempts on Mila, Tamia and configured Alliance clusters within a recorded resource envelope.
---

# Cluster execution

Read the owning project's instructions and applicable authorization. Use the
[Mila](../../docs/October/2026-10-26/mila/guidelines/mila.md), [Tamia](../../docs/October/2026-10-26/tamia/guidelines/tamia.md) or
[Cluv](../../docs/October/2026-10-26/cross-cluster/guidelines/cluv.md) guide. Historical connectivity and source-project
permissions do not establish current readiness or a new allocation budget.

1. Attribute the campaign, owner and attempt. Inspect owned pending/running jobs,
   active controllers, uncertain dispatches and reservations before any new launch.
2. Verify current access, source/config/data/lock identity, scratch paths, offline
   environment, scheduler/account eligibility and the scientific gate. Measure a
   representative pilot before long GPU work; read the GPU-efficiency skill.
3. Prepare the exact command, target, CPU/RAM/GPU resources, rounded walltime,
   submission count, dependencies, unique outputs/logs, cost and stop rule. Include
   all race configurations and requeues in the applicable envelope. Use existing
   authority; when absent, present this concrete request for the missing approval.
4. Reserve before dispatch in the owning executor's durable journal. Append every
   actual job ID to the dated Slurm ledger immediately, with source and attempt
   identity. A timeout creates an unknown dispatch, not an automatic retry.
5. Reconcile scheduler state and actual artifacts. Preserve array/step identities,
   failures, cancellations and costs without double counting. Verify scientific
   outputs independently of Slurm success; release dependents only after review.

Do not modify shared SSH settings, copy credentials, submit another project's
config, cancel unrelated jobs or mutate source used by queued/running attempts.
Persistent coordination uses a valid owned CPU allocation and explicit handoff;
one campaign has one authoritative controller.

Follow [documentation, reporting scope and Git sync](../../docs/October/2026-10-26/cross-cluster/guidelines/documentation-layout.md)
for dated records and W&B eligibility.
