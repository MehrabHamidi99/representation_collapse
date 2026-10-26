# Cluv submission and staging guide

Cluv synchronizes uv projects and submits jobs across configured SSH clusters.
The existing Representation Collapse operations environment pins `cluster-uv==0.1.5`
and `uv==0.9.18`. Those pins are a tested compatibility example, not a latest-version
claim. The live Tamia user-level executable on 2026-10-13 reported **0.1.2**;
the per-project 0.1.5 environment is a separate installation. Inspect
`cluv --version`, `cluv submit --help` and the actual project config
before using any command. [Upstream overview](https://mila-iqia.github.io/cluv/).

## Project setup

Keep `[tool.cluv]` in the project's `pyproject.toml`. The generated
`configs/cluv.example.toml` is a merge reference; Cluv does not automatically load
that separate file. Use a dedicated operations dependency group or isolated tool
environment when scientific code has incompatible dependencies. Lock the tested
versions; avoid updating a shared remote uv installation as a side effect.

Use project-specific remote source/results directories under verified scratch.
Git remains the primary local/cluster synchronization mechanism; follow
[source synchronization](source-sync.md) for branches and immutable run inputs.
Cluv synchronization involves committed Git source; untracked local edits are not
proof of remote deployment. Record the actual remote commit and environment/data
identities before dispatch. Preserve dirty checkouts and independent forks.
[Configuration guide](https://mila-iqia.github.io/cluv/guides/submit/config/).

## Access, staging, submission

```bash
cluv --version
cluv submit --help
cluv sh -- hostname                   # bounded observation; can fan out
cluv sh -- bash -lc 'command -v sbatch' # compound commands need an explicit shell
# After target/source review and within the project's staging authority:
cluv sync tamia
# Only within a recorded compute envelope, from the owning checkout:
cluv submit tamia scripts/cluster_job.sh --time=00:10:00 -- "$PYTHON" -V
```

The job script receives the command following `--`; the included wrapper uses
`srun "$@"`. Prepare/export its required source and run identities in the actual
remote execution environment. Do not assume a newly set local environment variable
will reach a remote batch job; explicit script arguments or tested `[tool.cluv.env]`
entries avoid that ambiguity. [Job script guide](https://mila-iqia.github.io/cluv/guides/submit/job-scripts/).

`cluv init`, `sync`, `run`, `submit` and `clean` have side effects. Never use them
as read-only probes. Version 0.1.5 project evidence says `cluv submit --dry-run`
is unsupported. Inspect installed help; use a rendered native `sbatch --test-only`
request for scheduler-only validation. Do not pass an invented flag to a submitter.

## Racing is multiple submissions

`cluv submit first` considers active enabled connections plus the current cluster.
A list of `sbatch_args` creates a submission per configuration, multiplying the
race. Budget every candidate; near-simultaneous starts can consume resources on
more than one cluster. Record all IDs, configurations, cancellation outcomes and
uncertain receipts. The template does not enable racing by default.
[Race semantics](https://mila-iqia.github.io/cluv/guides/submit/racing-clusters/).

A connected target still needs a verified account, partition, writable scratch,
staged source/data, locked environment and project budget. Local records include
Narval, Rorqual, Fir, Nibi, Trillium, Trillium-GPU, Killarney and Vulcan. These are
observed candidates, not a portable enabled list. Some share filesystems; others
need a login shell for scheduler PATH. Check each independently with a deadline.

## After dispatch

Append the exact submission to the dated ledger immediately. Check scheduler state
on the actual target, including array tasks/steps. Cluv's cached status or exit code
alone cannot establish execution. Preserve unknown dispatches, avoid duplicate
retries, and verify output provenance before interpreting metrics. Never cancel
another campaign's jobs as part of race cleanup.
