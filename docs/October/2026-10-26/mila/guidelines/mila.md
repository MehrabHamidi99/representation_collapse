# Mila execution guide

Use the project's actual route, user and resource envelope. The inspected account
uses SSH alias `mila` and scratch `/network/scratch/m/mehrab.hamidi`; these are
observations, not portable defaults. Read-only scheduler checks succeeded from an
existing CPU allocation on 2026-10-13; the dated receipt is in the template's
`docs/evidence/`. Recheck before scheduling work.

## Inspect and prepare

```bash
hostname
printf '%s\n' "$SCRATCH"
sinfo -h -o '%P|%l|%G'
squeue -u "$USER" -o '%.18i %.16j %.10T %.10M %.8l %.30R'
sacctmgr -nP show assoc where user="$USER" format=Cluster,Account,Partition,QOS
```

Check the project's owned jobs separately from account-wide visibility. Verify
the source commit, lockfile, dataset and available scratch space. Stage dependencies
and data before a GPU allocation; test CUDA only inside an allocation. Use
`UV_CACHE_DIR`, `UV_PROJECT_ENVIRONMENT`, `UV_PYTHON_INSTALL_DIR` and `HF_HOME`
under the project's scratch root. Pin module/tool versions in the run manifest.

Mila documents CPU partitions `unkillable-cpu`, `main-cpu`, `long-cpu`; GPU
partitions include `unkillable`, `main`, `long`, `short-unkillable`. H100 access is
restricted to `short-unkillable`; its documented walltime limit is three hours.
Other partitions have different limits and preemption rules. Use live scheduler
validation for the complete request, rather than inferring eligibility from a GPU
visible in `sinfo`. [Official partitions](https://mila-docs.readthedocs.io/en/latest/technical_reference/clusters/mila/partitions/).

## Submit an admitted attempt

Prepare exact source/config/protocol hashes, owner, outputs, logs, walltime and
request first. The generated `scripts/cluster_job.sh` runs one explicit command in
an already staged clean source snapshot. It requires `RUN_DIR`, `SOURCE_DIR` and
`SOURCE_COMMIT`; pass an existing locked interpreter as its command.

```bash
# Example CPU smoke, after the project envelope admits it. Set verified paths first.
mkdir -p "$RUN_DIR"
sbatch --test-only --partition=unkillable-cpu --nodes=1 --ntasks=1 \
  --cpus-per-task=1 --mem=2G --time=00:10:00 scripts/cluster_job.sh "$PYTHON" -V
# The same request without --test-only creates a real job:
sbatch --parsable --partition=unkillable-cpu --nodes=1 --ntasks=1 \
  --cpus-per-task=1 --mem=2G --time=00:10:00 \
  --comment="$PROJECT_ID:$ATTEMPT_ID" --output="$RUN_DIR/%x-%j.out" \
  --error="$RUN_DIR/%x-%j.err" --export=ALL scripts/cluster_job.sh "$PYTHON" -V
```

`--test-only` asks Slurm to validate and estimate scheduling without creating a
job. A successful check does not establish data, environment or scientific
readiness. [sbatch documentation](https://slurm.schedmd.com/sbatch.html).

Immediately record the returned ID, command, dependencies, timestamps and full
reservation in the same-day submission ledger. On timeout, reconcile `squeue`
and `sacct` using the unique attempt comment; retain an unknown outcome rather
than resubmitting. Terminal Slurm success still needs artifact and scientific review.

## Shared CPU coordination

Read a current lease containing job, host, namespace, expiry and infrastructure
owner. Enter an existing authorized allocation through `srun --jobid=... --overlap`
when its owner permits shared use. Private `/tmp` sockets cannot be reused merely
because a shared filesystem exposes their path. Keep one campaign controller,
private journals, bounded work and durable handoffs before expiry. Never embed a
historical job ID in this template or request a replacement implicitly.
