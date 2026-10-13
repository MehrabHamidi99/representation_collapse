# Shared autoresearch compute node — October 13, 2026

User priority: run persistent research coordination inside Mila's allocated
compute job. Login nodes are SSH gateways and allocation launch points only.

## Connect now

Allocation **11228391**, node **cn-f003.server.mila.quebec**, partition
**unkillable-cpu**, account **mila**, **2 CPUs / 4 GiB / zero GPUs**.
Started **2026-10-13 08:55:56 UTC**; expires **2026-10-14 08:55:56 UTC**
(October 14, **12:55:56 Yerevan**). No automatic replacement is configured.

From the local computer:

```bash
ssh -t mila 'srun --jobid=11228391 --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 --pty bash -l'
```

This enters the existing allocation. Do not create another allocation or use
plain SSH to the hostname as an equivalent: Mila isolates `/tmp`, and the
shared sockets/journals are inside the job's namespace. `--overlap` shares the
existing two CPUs; it does not grant each agent a separate CPU budget.

Inside the allocation, connect to the authenticated Tamia master:

```bash
ssh -S /tmp/radius-control/tamia-11228391 -O check tamia
ssh -S /tmp/radius-control/tamia-11228391 tamia
module load httpproxy
export PATH="$HOME/.local/bin:$PATH"
cluv --help
```

The Tamia SSH master is owned by a tmux server started by the Slurm job, not
by an agent's SSH session. Use it; do not replace or terminate it. Tamia's
current destination was verified as `tamia3.tamia.ecpia.ca`. A successful
master check does not prove every other Cluv destination is authenticated.

## Read shared status and ownership

Shared runtime:
`/network/scratch/m/mehrab.hamidi/radius-autoresearch/control/20261013/`

- `lease.json`: allocation/node and private runtime location.
- `heartbeat.json`: Slurm supervisor and observer liveness.
- `SETUP.json`: current readiness, remaining setup and infrastructure owner.
- `observer-config.json`: migrated observer configuration; do not edit.
- `observer-backup.sqlite`: atomic recovery snapshot; never open as a live journal.
- `agent-claims/`: each agent's ownership/heartbeat record.

The radius observer is active in this allocation; its 44,423 historical
events were preserved at migration. Its journal is private to this node.
Tamia authentication from the Slurm-owned master is verified. The operator
is still checking Cluv destinations and moving model triage/paid decisions
off the Mac. Consult `SETUP.json` before claiming those parts are active.

## Collaborative workflow for the three agents

1. Read your project's own instructions, latest plan, journals, live jobs and
   source identities. This shared host does not transfer research ownership,
   GPU budgets, credentials or authority between projects.
2. Before modifying a project, create a uniquely named JSON claim under
   `agent-claims/` with agent/session name, project root, task, writable paths,
   existing job IDs, start time, heartbeat, finite deadline and status.
   Inspect other claims first. Update your own claim at least every five
   minutes; record a handoff before overlapping another agent's files/jobs.
   A stale claim requires reconciliation, not automatic takeover.
3. Use a separate Git branch/worktree for edits and separate private runtime
   under `/tmp/radius-control/<project>-<agent>/`. Keep credentials in their
   existing protected home locations. Publish small status/evidence files
   and consistent database backups to your project's durable storage.
4. Run substantial preparation, training, evaluation and tests in separately
   admitted Slurm/Cluv allocations. This shared two-CPU host is for compact
   coordination, SSH, scheduler checks and model API clients. Do not load
   model weights here or run concurrent heavy builds.
5. Keep a deterministic five-minute research loop: observe actual progress,
   reconcile accounting/artifacts, advance already-admitted jobs through
   their owning controller, update status, then wait. Preserve pending jobs
   and ambiguous submission receipts; never submit a duplicate on a timeout.
6. Use cluster-hosted Qwen for compact event triage and bounded mechanical
   work. Idle polling spends no model tokens. Wake `gpt-6.1-sol` for difficult
   engineering and `gpt-6-astra` for consequential scientific reasoning.
   Model proposals do not execute commands or change scientific admission.
   Record calls, usage, failures and finite worker limits; provider errors
   must not trigger an automatic paid fallback.
7. Radius's existing model limits remain 24 Qwen calls and two paid wakes per
   rolling 24 hours, one paid worker, 600-second worker timeout. Other projects
   retain their own explicitly authorized budgets; they must not borrow or
   reset Radius's journal. Coordinate before overlapping expensive reviewers.
8. Keep making progress on the current plan while jobs queue: inspect finished
   artifacts, maintain manifests/docs, prepare admitted next steps and resolve
   execution defects. Stop an affected action on failed scientific gates or
   exhausted limits; keep the observer and unrelated admitted work active.

## Work in progress and approval boundaries

Infrastructure owner: the Radius root agent in this conversation. Other
agents may use the verified compute/Tamia route now for their own authorized
work. Leave shared supervisor, sockets, observer/model journals and model
configuration to that owner until `SETUP.json` records completion/handoff.

Remaining setup at initial publication: verify all configured Cluv routes;
migrate Qwen/OpenAI dispatch into the allocation using existing cluster
credentials and preserved budgets; publish final workflow/status receipts;
resume the bounded Nexus recovery deployment. These are technical tasks,
not approval requests. Do not duplicate their deployment while waiting.

No user approval is currently pending. If Tamia Duo expires, retain queued
work and request normal MFA; never disable host-key verification. New methods,
data mixtures, scientific arms or increased experiment limits need the owning
project's admission. An agent must record the exact pending decision and safe
independent work, rather than declaring all research blocked.

Before allocation expiry, checkpoint journals and publish a handoff. The
partition name does not guarantee unlimited uptime. Reconcile this job before
requesting an authorized finite replacement; update all job/socket paths.

## Local files and reuse

Local workspace: `/Users/mehrabhamidi/Desktop/phd/paper/radius_increase_project`.
Use `python3 scripts/ssh_cluster.py mila 'COMMAND'` or
`python3 scripts/ssh_cluster.py tamia 'COMMAND'` there. The helper enters job
11228391 and refuses an expired route; it does not silently revert to login
node execution. Source configuration is `autoresearch/cluster-route.json`.

Portable skill: `skills/compute-autoresearch/SKILL.md` with its adjacent
`references/workflow.md`. Copy that whole skill directory to another project's
`skills/` or its agent's skill directory. Keep project budgets and scientific
instructions local to that project, and reread the live allocation receipt.
