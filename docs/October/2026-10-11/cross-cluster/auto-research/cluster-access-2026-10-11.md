# Cluster access discovery — 2026-10-11

This read-only pass supports the user's new instruction to sync accessible
clusters and launch initial experiments. It created no remote checkout,
environment, allocation, or job, and changed no SSH settings. The coordinator
owns staging and submission. Detailed scheduler responses and route errors are
in the ignored `runs/ops/cluster-discovery-2026-10-11.json` artifact.

## Reachable routes

**Mila:** laptop `ssh mila` reaches `login-4.server.mila.quebec`. Scratch is
`/network/scratch/m/mehrab.hamidi`, account `mila`, user QOS `normal`.
`main-cpu`, `long-cpu` and `unkillable-cpu` are UP; `main-cpu` allows five days
and is suitable for a ten-minute CPU calibration. Its partition QOS limits
aggregate per-user CPU/memory; other projects already have CPU and GPU jobs,
so visible capacity is not a guarantee of an immediate start. Preemption is
possible. No project-name matches (`rc-`, `collapse`, `floquet`, `periodic`)
were observed. This is a name-based check, not proof about arbitrary job
working directories.

**Tamia:** the existing Mila-side `ssh tamia` route works noninteractively;
scratch is `/scratch/m/mehrab`, account `aip-ioannism`, user QOS `normal`.
`cpubase_bycore_b1` is UP, allows three hours, and supports the prepared
single-core CPU smoke. Longer `bycore_b2` and `bycore_b3` partitions allow
twelve hours and one day. The observed GPU partitions are
`gpubase_bynode_b1/b2/b3` and `gpubase_interac`, with H100/H200 nodes.
Do not infer that an individual GPU can be allocated from the by-node profiles;
the minimum/allocation policy was not validated. No GPU is needed for the
NumPy smoke or the first compact CPU pilot.

Both reachable users have `~/.local/bin/uv` version 0.9.18. Neither has an
existing `$SCRATCH/representation-collapse/source` checkout at discovery time.
Filesystem free-space output is in the local artifact; free bytes do not
establish user quota. No cache, dataset or Python installation was created.

## Other routes checked once

The laptop has aliases for Narval, Rorqual, Fir, Nibi, Trillium,
Trillium-GPU, Killarney and Vulcan. Mila's personal SSH config defines only
Tamia; bare names for the other clusters fail resolution. Explicit
`mehrab@NAME.alliancecan.ca` from Mila reaches host-key verification but
cannot authenticate under the current trusted-host configuration.

Direct laptop probes returned these blockers:

- Narval, Rorqual and Nibi: keyboard-interactive/MFA authentication required.
  Narval also advertises a system-access agreement step. The banner alone
  does not prove that this user's agreement is missing.
- Fir: keyboard-interactive authentication required.
- Trillium and Trillium-GPU: publickey/hostbased authentication denied.
- Killarney: publickey authentication denied.
- Vulcan: changed remote host key. Do not bypass strict checking or remove the
  existing key without independently verifying the operator's current identity.

No interactive MFA attempts or repeated authentication prompts were made.
Scheduler accounts, scratch paths and entitlements remain unknown on these
routes. Their failure to authenticate is not evidence of absent scientific
allocation. These initial laptop/Mila results are superseded for connection
planning by the Tamia-side discovery below.

## Tamia-side connection discovery

Tamia's personal SSH configuration defines `narval`, `rorqual`, `fir`,
`nibi`, `trillium`, `trillium-gpu`, `killarney` and `vulcan` with user
`mehrab` and hostname `%h.alliancecan.ca`. It enables `ControlMaster auto`,
`ControlPath ~/.ssh/cm-%C` and eight-hour persistence. Effective configuration
also enables hostbased and GSSAPI authentication and uses the system
`sss_ssh_knownhostsproxy`. No existing control socket was observed before the
new interactive login pass.

One bounded, sequential `BatchMode=yes`, `StrictHostKeyChecking=yes` probe
from Tamia reached authentication on **all eight aliases**. Every host key
passed Tamia's trusted-host checks, including Vulcan. Therefore Vulcan can
be approached through Tamia without bypassing or replacing the conflicting
laptop key. All eight probes stopped at keyboard-interactive authentication;
Narval, Rorqual, Nibi and both Trillium aliases also advertised hostbased
authentication. These outcomes require an interactive access test and do not
establish absent access. No interactive retries were made by discovery.

The user subsequently authorized interactive cluv access verification and
will approve Duo. The coordinator owns that single sequential login pass;
discovery stopped new authentication probes to avoid competing prompts.
Standalone `~/.local/bin/cluv` on Tamia reports **0.1.2**. For staging and
submission use the project's locked `uv run --group ops cluv` version 0.1.5.
Remote account/QOS/resource profiles must be read after successful login before
adding those targets to an experiment submission set. The ignored JSON artifact
contains exact alias settings and the eight Tamia-side outcomes.

After the user reported completing all eight Duo approvals on **tamia1**, a
read-only cluv 0.1.5 API pass on our **tamia2** controller returned no active
`Remote` for any of the eight explicit aliases. Thus the verified reusable
count from tamia2 is **0/8**, despite the user's successful tamia1 login report.
Shared-home socket paths do not make an SSH master usable on another login
node: its process and UNIX socket are local to the node that created it.
The strict BatchMode route `tamia2 -> tamia1.tamia.ecpia.ca` stopped at
keyboard-interactive authentication. The coordinator is establishing a single
interactive connection to tamia1; scheduler/account/scratch claims for the
other clusters remain pending verification on that controller. Discovery
triggered no new Duo prompt, staging, allocation or job.

## Exact controller workflow

Cluv 0.1.5 is installed in the locked `ops` group. Native config is
`[tool.cluv]` in `pyproject.toml`; arbitrary cluster names are accepted by the
schema, but a target must have a configured cluster entry. Root config now
lists Mila, Tamia and all eight Alliance aliases for access discovery; listing
a target does not establish authenticated access or verified scheduler resources.
Explicit `sync` targets work; `submit first` has no
candidate-subset flag and races every active configured connection. Use
single-target submission for initial work. `submit --dry-run` is unsupported.

These are coordinator commands, not actions executed during discovery:

```bash
# On the laptop, after committing/pushing the exact experiment source:
uv run --group ops python code/ops/preflight.py
uv run --group ops cluv sync mila --no-sync-datasets
```

Before cluv's first environment sync, arrange the project's exact Python and
locked package caches under each cluster's project scratch via
`code/ops/prepare_environment.sh`. The existing compute plan details the
preparation order; a pre-existing `.venv` alone does not prove offline readiness.
Cluv sync itself installs/updates uv, clones/updates code, builds environments
and fetches results, so the coordinator must treat it as staging.

On Mila, after its clean tracking checkout and locked controller environment
are prepared:

```bash
cd "$SCRATCH/representation-collapse/source"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$SCRATCH/representation-collapse/cache/uv"
export UV_PYTHON_INSTALL_DIR="$SCRATCH/representation-collapse/cache/python"
uv sync --locked --group ops
uv run --group ops cluv login tamia
uv run --group ops cluv sync tamia --no-sync-datasets
```

Pre-stage Tamia's scratch Python/cache on its login node before the final sync.
Mila is already the current cluster for this controller; `cluv sync mila`
from Mila intentionally skips self-sync. Do not forward private keys or copy
credentials to extend access. The discovered source SHA was
`7ca2be4c7827d0644b0216cb86bec1dbb855f5a4`; it is a discovery snapshot,
not the final experiment revision. Record the newly committed experiment SHA
in each launched command and wrapper provenance.

## Small-job configurations

Both CPU allocations use one node, one task, one CPU, 2 GiB, ten minutes,
no GPU and no race. Their requested CPU time is at most 1/6 CPU-hour each;
cluster billing weights and queue time are separate. Add `--no-requeue` if the
coordinator wants a preempted Mila smoke to terminate rather than restart.

Native per-cluster settings, validated against the installed schema in memory:

```toml
[tool.cluv.clusters.mila.sbatch_args]
account = "mila"
qos = "normal"
partition = "main-cpu"

[tool.cluv.clusters.tamia.sbatch_args]
account = "aip-ioannism"
qos = "normal"
partition = "cpubase_bycore_b1"
```

The coordinator can supply Mila's settings as CLI flags without changing the
configuration. From the prepared Mila controller, substitute the exact full
experiment commit for `FULL_SHA`:

```bash
uv run --group ops cluv submit mila code/ops/calibration_job.sh \
  --account=mila --qos=normal --partition=main-cpu \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:10:00 \
  --job-name=rc-calibration-mila --no-requeue --no-sync-datasets -- \
  --source-commit FULL_SHA --run-id cpu-smoke-mila-001 --cluster mila

uv run --group ops cluv submit tamia code/ops/calibration_job.sh \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:10:00 \
  --job-name=rc-calibration-tamia --no-sync-datasets -- \
  --source-commit FULL_SHA --run-id cpu-smoke-tamia-001 --cluster tamia
```

Outputs are under
`$SCRATCH/representation-collapse/logs/cluv/CLUSTER_JOBID/calibration/`.
The calibration wrapper snapshots the commit, checks locked offline readiness,
records source/lock/SLURM provenance and caps BLAS threads. Submission commands
must refer to an existing pushed commit containing that wrapper and its lock.

For the first compact trained-model CPU pilot, a candidate envelope is one
task, two CPUs, 8 GiB and thirty minutes on the same CPU partitions. This is a
configuration candidate, not a runtime prediction: validate the new pilot
locally and measure its runtime/memory before broadening the sweep. Keep
model/seed/trial bounds and its output manifest explicit. GPU training should
follow measured need; a single-GPU Mila profile is visible, but Tamia by-node
GPU policy and every other cluster's entitlement remain unverified.

Discovery references: [cluv commands](https://mila-iqia.github.io/cluv/),
[official racing guide](https://mila-iqia.github.io/cluv/guides/submit/racing-clusters/).
Operational conclusions above use the live installed cluv source and scheduler
responses, not unverified template defaults.

## Check in the owner's existing tamia1 shell

The owner subsequently supplied a transcript of `cluv login` returning after
Duo selection for all eight Alliance remote aliases, with no errors for those
aliases. Mila explicitly failed public-key authentication from that controller.
This is reported authentication evidence; it is not a fresh scheduler or
connection probe from tamia1. Coordinator's two attempts to enter tamia1 using
Duo did not complete before the SSH session closed. No additional Alliance
Duo requests are necessary to inspect the owner's existing masters.

Run the passive verifier from the same **tamia1** shell that performed login:

```bash
cd "$SCRATCH/representation-collapse/source"
uv run --locked --group ops python code/ops/verify_cluv_access.py
```

It uses pinned cluv 0.1.5, checks Tamia plus the eight remote aliases, and
returns JSON with controller hostname, active master status, remote identity,
scratch visibility and read-only account/QOS/partition responses. It never
requests authentication, edits SSH settings or submits a job. A missing master
is reported directly; a scheduler response still needs resource-policy review.
For a quick human-readable view, run `uv run --locked --group ops cluv status
clusters` in that same directory. Login-node identity matters even with a
shared home directory. Mila can continue to be operated from the laptop.
