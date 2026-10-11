# Compute plan: Floquet calibration to multimodal recurrent models

Latest execution update, 2026-10-11: the initial phase is **engineering complete**.
Narval CPU calibration **5234514** passes 10 + 36 checks; GPU jobs **5234860**,
**5235508**, **22980985**, **22980998** all complete `0:0`, with full training
and six independently verified artifact directories. All four native competence
gates fail, leaving harmful collapse inconclusive. See the
[final report](initial-phase-report.md) and [recovery protocol](competence-recovery-plan.md).
Four GPU submissions are consumed. Narval uses full A100; Rorqual's same pending
IDs were reduced to H100 MIG `3g.40gb` and fifteen-minute limits, without changing
source or training steps. Actual allocation is 555 device-seconds, counting MIG
as one device. Original requests and amendments are recorded separately.
The text below retains the historical preparation state; it does not supersede
the current ignored approval record or authorize additional jobs.

Current access update, 2026-10-11 12:31 UTC: all nine configured Alliance
targets authenticated and passed identity checks through native cluv. Eight
remote masters were reusable from the held Tamia2 controller; Mila separately
verified from the Mac. See [cluster access](cluster-access.md) for current
accounts, storage, adapter commands and the resolved Killarney/Vulcan login-shell
requirements. The preparation text below describes the original proposed allocation;
subsequent owner authorization and executed jobs are recorded in the tracker
and dated notebook.

Historical preparation, before authorization: no job was submitted and no remote checkout, environment,
SSH configuration, dataset, or queue entry was changed during preparation.
The owner's statement that Compute Canada work can be approved establishes
an approval route; it does not specify a compute budget. This project's first
allocation and later pilots therefore need a concrete envelope before execution.

## Verified tooling and configuration

Use `cluster-uv==0.1.5` from the locked `ops` dependency group:

```bash
uv sync --locked --group ops
uv run --group ops cluv --help
uv run --group ops python code/ops/preflight.py
```

The `ops` group also pins `uv==0.9.18`, matching both verified remote CLIs.
Invoke cluv through this group so its `uv` subprocess resolves to that version;
cluv sync otherwise updates or downgrades the shared remote user's uv to match
the controller's version. The standalone laptop uv was 0.8.13 during setup.

Cluv reads `[tool.cluv]` in root `pyproject.toml`. The native example is
[`configs/cluv.example.toml`](../../configs/cluv.example.toml); it is a review
mirror for Mila/Tamia plus optional Alliance candidates, not another
automatically loaded config. Root config activates only Mila/Tamia. Do not run `cluv init` over
the prepared project. Cluv 0.1.5 has no `submit --dry-run`; the dry-run flag in
its CLI belongs to `clean`. Our local preflight validates the installed schema
and builds the remote command without SSH or `sbatch`.

Source verification used the published 0.1.5 wheel and upstream commit
`d96a2dcdf0e4078b85ea38ce08ad402b19d9b783`. The
[official racing guide](https://mila-iqia.github.io/cluv/guides/submit/racing-clusters/)
and [configuration guide](https://mila-iqia.github.io/cluv/guides/submit/config/)
match the release. See [upstream submission implementation](https://github.com/mila-iqia/cluv/blob/d96a2dcdf0e4078b85ea38ce08ad402b19d9b783/cluv/cli/submit.py)
and [sync implementation](https://github.com/mila-iqia/cluv/blob/d96a2dcdf0e4078b85ea38ce08ad402b19d9b783/cluv/cli/sync.py)
for the behavior described below.

## Access and cluster order

- **Mac:** coordination, documentation, small NumPy calibration and results review.
- **Tamia:** first proposed CPU smoke. A read-only SSH probe through Mila
  reached `tamia2.tamia.ecpia.ca`; `$SCRATCH` and `~/scratch` resolve to
  `/scratch/m/mehrab`. Association `aip-ioannism` has QOS `normal`.
  `cpubase_bycore_b1` is UP and allows up to three hours; the proposed smoke
  uses ten minutes. GPU partitions advertise H100 nodes with four GPUs and
  H200 nodes with eight; none is requested for calibration.
- **Mila:** accessible directly with existing laptop SSH alias; scratch is
  `/network/scratch/m/mehrab.hamidi`. Several GPU partitions were visible.
  A CPU partition/account for this project has not been validated.
- **Other Alliance candidates:** `narval`, `rorqual`, `fir`, `nibi`, `trillium`,
  `trillium-gpu`, `killarney`, `vulcan` appear in the current cluv template and
  local SSH aliases. Project access, accounts, resources and storage remain
  unverified. The old aliases `beluga`, `cedar`, `graham`, `niagara` are not
  assumed to be current deployment targets. Trillium's two login names share
  a filesystem in cluv's implementation, but represent distinct race entries.

Direct laptop access to Tamia timed out. The existing Mila-to-Tamia hop
worked without copying keys or passwords. A read-only laptop `ssh -J mila
tamia` probe reached authentication but failed in BatchMode with
`Permission denied (keyboard-interactive)`. A possible local SSH addition,
after checking conflicts with existing stanzas, is `ProxyJump mila` under
`Host tamia`; it has not been applied, and a fresh laptop cluv login may need
the owner's interactive authentication. Alternatively run the
cluv controller on Mila using its already working Tamia alias. Do not forward
private keys or edit shared cluster SSH settings as a workaround.

Recheck with read-only commands before staging or scheduling:

```bash
bash code/ops/preflight_remote.sh mila
bash code/ops/preflight_remote.sh tamia mila
```

Existing Tamia jobs belonging to other projects were visible. They are not
this project's concurrency allowance and must not be cancelled or altered.

## Concrete first allocation to approve

Exactly **one Tamia CPU job**, no arrays, no requeue, no race, no GPU:
`account=aip-ioannism`, `qos=normal`, `partition=cpubase_bycore_b1`, one node,
one task, one CPU, 2 GiB requested memory, walltime `00:10:00`. Maximum request
is **1/6 CPU-hour and zero GPU-hours**. Purpose: run the existing synthetic
Floquet/rank calibration with a locked environment, immutable source snapshot,
SLURM provenance, JSON and NumPy array artifacts. This validates instrumentation;
it is not empirical evidence for trained-model dimension collapse.

Once a commit exists, prepare its exact reviewable command locally:

```bash
uv run --group ops python code/ops/preflight.py --cluster tamia --run-id cpu-smoke-001
```

The preview includes the full source SHA and executable cluv command. Get
approval for this resource envelope and bounded scratch staging on Mila/Tamia before
executing either sync or submit. Root `AGENTS.md` defines the continuing
project resource policy; this file grants no allocation authorization.

## Staging and execution after approval

Cluv sync and submit can push local commits, install/update remote uv,
clone/update the remote checkout, build `.venv`, create symlinks and fetch
results. They are not read-only preflight commands. Remote GitHub access and
a committed, pushed source are prerequisites. Laptop GitHub SSH authentication
failed; HTTPS could list the empty repository, and root setup may publish via
a separate HTTPS push URL while retaining the requested SSH origin URL.
Mila and Tamia could both list the empty origin over SSH (exit status zero,
BatchMode, bounded timeout). Both have `~/.local/bin/uv` version 0.9.18;
export that directory in PATH before approved staging. Never use `--autocommit` to hide a dirty
research tree or pretend unstaged changes belong to the recorded SHA.

Native `project_dir=$SCRATCH/representation-collapse/source` keeps cluv's
checkout and default `.venv` on scratch. Cache, Python downloads, datasets,
logs, outputs and per-job environments also belong under that scratch root.
Scratch can be purged: preserve the source in Git and back up accepted result
manifests and small artifacts to durable project storage.

During approved initial staging, clone the pushed source into that configured
scratch directory and run `bash code/ops/prepare_environment.sh` on the login
node. It explicitly installs the project Python into scratch without adding
HOME executable shims, fills the locked NumPy cache, then tests a fresh offline
environment before an allocation. A working pre-existing `.venv` alone does
not establish offline readiness. Then establish the working cluv connection and sync
**only Tamia** with `uv run --group ops cluv sync tamia --no-sync-datasets`.
The script wrapper explicitly exports its own cache/offline variables and
accepts the source SHA and cluster name as positional arguments; this avoids
depending on new variables being exported by cluv 0.1.5's submission shell.
Use the exact `cluv_command_after_approval` printed by preflight to submit.

For the verified noninteractive route, use Mila as the cluv controller. Stage
the source and locked `ops` environment in Mila scratch, and stage source plus
`prepare_environment.sh` on Tamia through Mila before cluv sync. Keep the
controller checkout on a clean tracking branch; only compute snapshots are
detached. Export scratch `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` during
controller environment preparation. No datasets or checkpoints are needed.
Pre-staging Tamia's scratch interpreter/environment avoids cluv's first sync
creating a default Python installation outside project scratch. Do not repeat
clone into an existing directory; inspect it and verify the source first.

The job snapshots the approved SHA under `$SLURM_TMPDIR`, falling back to
project scratch. Artifact output is
`$SCRATCH/representation-collapse/logs/cluv/tamia_<job_id>/calibration/`,
alongside SLURM output and provenance, so an explicit
`uv run --group ops cluv sync tamia --no-sync-datasets` retrieves the artifacts.
Do not delete datasets or remote outputs automatically after fetching them.

## Racing and expansion gates

`cluv submit first` races **every currently active, enabled configured SSH
connection**, plus the current cluster when applicable. It accepts no explicit
cluster subset. Logging into only two clusters does not prove that older
connections are inactive. A list of per-cluster `sbatch_args` contributes one
submission per entry even when using `cluv submit tamia`. `--vram` can expand
the submissions further. The first non-pending job wins and competitors are
cancelled; simultaneous starts and client interruption can still consume
resources. Do not assume only the winner incurs cost.

The smoke preflight rejects `first`, multiple configurations and non-smoke
resource flags. Later races require an exact inspected candidate list,
per-candidate resources, a pessimistic aggregate budget, and project-scoped
cancellation records. Never cancel by a broad user-wide queue selector.

After the CPU smoke passes, reproduce one inherited temporal graph result
with a scientifically defined slice, seed list and checkpoint provenance.
Measure CPU/GPU memory and runtime in a small approved pilot before choosing
training resources. The Skill Inference project's four-H100 preference and
budgets are not inherited here. Add sequence, vision, audio and control
recurrences only through the experiment-loop evidence and budget gates;
select accelerators from measured needs, not a blanket multimodal allocation.
