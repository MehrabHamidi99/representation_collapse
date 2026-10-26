---
name: auto-research
description: Advance a research project through falsifiable hypotheses, calibrated experiments, evidence review and a durable handoff within its existing compute and model limits.
---

# Bounded auto-research

Read the project's instructions, `project.json`, current tracker, protocol and last
handoff. Use the project's existing structure if adopting into older research.
Read [workflow](references/workflow.md) for execution and journal commands and
[architecture](../../docs/October/2026-10-26/cross-cluster/guidelines/architecture.md) when designing a campaign.

Choose the smallest ready task that resolves an important uncertainty. State the
question, plausible competing explanation, discriminating evidence, acceptance
criterion and stop rule before running it. Literature claims, toy calibration,
real-data measurements and held-out findings are different evidence classes.

Advance through audit → calibration → baseline reproduction → measured pilot →
admitted experiment → analysis → scientific review → manuscript. Skip stages only
when existing attributable evidence already satisfies their purpose. Accept valid
negative findings; do not tune the protocol to manufacture a positive result.

For jobs requiring monitoring and producing scientific benchmarks/results, report
interpretable results to the relevant W&B project/subproject, including baselines,
comparisons, evaluations and valid partial or negative results. Exclude preprocessing,
progress-only tasks and infrastructure checks from W&B.
Follow [results reporting](../../docs/October/2026-10-26/cross-cluster/guidelines/results-reporting.md), verify delivery,
and preserve pending uploads when access is unavailable. Use
[Git synchronization](../../docs/October/2026-10-26/cross-cluster/guidelines/source-sync.md) for local/cluster updates
while retaining each attempt's pinned commit.

Use deterministic tools for observation, hashes and bookkeeping. Invoke reasoning
for unresolved scientific/engineering decisions, using capabilities actually
available. Delegate only when the user or project authorizes it, with bounded
scope, exclusive paths, evidence and acceptance checks. Record actual model usage;
unknown usage is not zero. Provider errors do not authorize a paid fallback.

Execute authorized work to completion without repeated permission questions. A new
cluster allocation, publication destination or expanded budget requires applicable
authority; prepare the exact reviewable command and envelope first when missing.
The default zero external budget is a bootstrap state, not a reason to stop local
analysis or preparation. Never import another project's authorization.

Before submission read [cluster execution](../cluster-execution/SKILL.md) and,
for GPUs, [efficiency](../gpu-efficiency/SKILL.md). Reconcile owned live jobs,
reservations and ambiguous receipts. Use immutable attempt inputs and the existing
owning controller; never create a second one to get around a stuck state.

Stop an affected action on exhausted limits, uncertain dispatch, changed input
identity, failed scientific gates or expired access/lease. Keep independent
authorized work moving. Record evidence, failed attempts, decisions and the exact
next action in the dated notebook and update the tracker. A live process, Slurm
success or well-formed JSON does not establish scientific acceptance.

Follow [documentation, reporting scope and Git sync](../../docs/October/2026-10-26/cross-cluster/guidelines/documentation-layout.md)
for dated records and W&B eligibility.
