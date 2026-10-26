---
name: compute-autoresearch
description: Run collaborative research coordination on an allocated Mila CPU node, reuse its authenticated Tamia route, and operate bounded Qwen/OpenAI research loops with separate project ownership.
---

# Compute-hosted autoresearch

Read [the shared workflow](references/workflow.md) before using the allocated
host or changing a research coordinator. Verify its live Slurm job, heartbeat,
expiry and SETUP.json; historical job IDs are not permanent connection settings.

Use the existing allocation's `srun --jobid ... --overlap` namespace. Login
nodes are gateways only. Share the Slurm-owned Tamia SSH master without changing
shared authentication or copying credentials. Load `httpproxy` on Tamia.

Register project/task/path ownership before edits. Each project retains its
own source pins, scientific gates, compute limits and model-call journal.
Keep SQLite private to one machine; publish consistent backups and small
status records. Never run duplicate controllers over the same campaign.

Keep deterministic monitoring active. Use Qwen for compact substantive events;
use bounded Sol 6.1/Astra reviews for difficult engineering/scientific decisions.
Idle polling and provider failures do not wake paid models. Advance admitted
work through its owning executor; model text cannot alter the execution plan.

Before expiry or handoff, preserve journals, uncertain submissions and pending
decisions. Check shared setup ownership before changing infrastructure.

## Current documentation/reporting policy (2026-10-26)

Follow root `AGENTS.md` and `docs/October/2026-10-26/cross-cluster/guidelines/documentation-layout.md`.
That policy supersedes older notebook paths and W&B scope in this skill. Use dated
cluster/subproject/type docs and Git synchronization. W&B covers monitored scientific
result jobs; preprocessing, progress-only tasks and infrastructure checks are excluded.
