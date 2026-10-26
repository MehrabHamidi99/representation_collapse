---
name: project-ops
description: Prepare and operate this Floquet research project's locked environments, scratch storage, cluv submissions and compute approval envelopes on Mila, Tamia and Alliance clusters.
---

# Project operations

Read root `AGENTS.md` and [compute plan](../../docs/October/2026-10-26/local/auto-research/compute-plan.md) before
cluster work. This project studies representation dimension collapse in
recurrent models; other projects' resource preferences do not apply.

## Preparation and approval

Local setup and read-only probes are authorized. Use existing SSH aliases,
BatchMode and bounded connection timeouts; never print or copy credentials.
Do not treat an available cluster account as an approved allocation. Before
the first cluster mutation or submission, prepare the exact source SHA,
command, candidates, resource envelope, output path and acceptance criterion.
The owner's offer to approve Compute Canada work does not grant an unstated
budget. The initial envelope is now recorded and its four GPU submissions are
consumed; follow existing authorization within its limits without asking again.
Prepare a new concrete envelope for subsequent training allocations.

The restored-access continuation on 2026-10-12 staged clean source `ec1c489`
on all nine configured Alliance targets and Mila using native cluv. Tamia CPU
job 612745 consumed its separate one-submission array-audit envelope and
completed `0:0`; no retry or new GPU training follows. The explicitly approved
Qwen continuation is separately bounded to 24 hours/24 attempts. Current
connections are ordinary native cluv masters on Tamia3, not the historical
Tamia2 project-private socket namespace. Recheck actual access and source.

Use the project's pinned `cluster-uv==0.1.5` via `uv run --group ops cluv`.
The same group pins uv 0.9.18, matching the remote CLIs and preventing cluv
from downgrading another project's shared remote uv installation.
Native config is `[tool.cluv]` in `pyproject.toml`; the example TOML mirrors
that schema. Do not invent YAML fields or run `cluv init` over the project.
Run `uv run --group ops python code/ops/preflight.py` for local validation;
add `--cluster tamia --run-id ID` for an exact single CPU smoke preview.
`cluv submit --dry-run` is not supported in this release.

## Hosts and storage

The initial allocation is now authorized within the ignored approval record and
frozen initial-phase protocol. Tamia CPU job 536757 and Narval calibration job
5234514 completed; a prior Mila CPU attempt failed before calibration. Four
seed-0 GPU pilots on Narval/Rorqual consume the initial GPU submission cap.
Do not infer an additional budget from successful execution.

Tamia was the proposed CPU-smoke target, reached through the existing Mila hop.
Read-only probes on 2026-10-11 verified `/scratch/m/mehrab`, account
`aip-ioannism`, QOS `normal`, and `cpubase_bycore_b1`. Mila direct SSH works;
scratch is `/network/scratch/m/mehrab.hamidi`. Tamia direct laptop SSH timed
out; do not repeatedly trigger interactive authentication. See the compute
plan and cluster-access report for current Alliance candidates and verified
entitlements; access is not an additional compute budget.

Keep code replicas, environments, cache, Python installs, data, runs and logs
under `$SCRATCH/representation-collapse`. Cluv's remote source directory is
`$SCRATCH/representation-collapse/source` so its automatically built `.venv`
is also on scratch. `prepare_environment.sh` fills the locked dependency
cache, installs Python under scratch without HOME shims, and verifies a fresh
offline environment on a login node before allocation. Never fetch packages or large data
in a compute worker. Keep durable copies of accepted evidence; scratch is
not a backup. No dataset/checkpoint downloads into the Mac checkout.

## Job contracts

Use `calibration_job.sh` for the synthetic NumPy smoke. Its full source SHA,
run ID and cluster name are explicit script arguments. It snapshots the
source commit, uses locked offline uv, caps BLAS threads at one, records the
lock checksum and SLURM ID, and writes calibration artifacts alongside the
cluv log directory. Explicit argument pinning avoids cluv 0.1.5's reliance on
new shell variables being exported by the login environment.

A staged working tree is not evidence that a remote job ran those changes.
Require a committed, remotely reachable source. Review dirty remote trees
before updating. Do not autocommit, force-push, erase directories, replace
unexpected symlinks, or disturb another project's queue entries. Do not
assume the inherited graph training code is distributed-ready.

## Racing and budgets

`cluv submit first` includes all active enabled configured clusters plus the
current cluster. There is no explicit candidate subset flag. Every entry in
a per-cluster `sbatch_args` list submits a job; GPU `--vram` expansion can add
more. Cost and approve the pessimistic whole race, inspect active connections
and disabled state, record job IDs, and cancel only jobs created by this
project's race. Racing is excluded from the initial smoke.

The original smoke requests one CPU, 2 GiB, ten minutes, zero GPUs. Its maximum
request is 1/6 CPU-hour; the bootstrap CPU cap is four submissions, of which
three were used including the failed Mila attempt. This is not a recurring allowance.
Future GPU count, walltime, model/data scope and concurrency follow measured
pilot needs and the approved research plan. Never import the template's
four-H100 allocation policy or other projects' remaining budgets.


## GPU efficiency

Before GPU admission, scaling or utilization diagnosis, read the complete
[gpu-efficiency skill](../gpu-efficiency/SKILL.md) and
[guideline](../gpu-efficiency/GUIDELINES.md). Apply measured useful throughput,
phase timing and per-device telemetry within this project's resource envelope.
Keep frozen batch/loss, precision and recurrence measurements fixed; a proposed
scientific change follows its separate protocol. Missing measurements are
unknown, and a low percentage alone does not authorize cancellation.

## Current documentation/reporting policy (2026-10-26)

Follow root `AGENTS.md` and `docs/October/2026-10-26/cross-cluster/guidelines/documentation-layout.md`.
That policy supersedes older notebook paths and W&B scope in this skill. Use dated
cluster/subproject/type docs and Git synchronization. W&B covers monitored scientific
result jobs; preprocessing, progress-only tasks and infrastructure checks are excluded.
