# Bounded inspection commands

Adapt IDs, paths, accounts and device assignments to the owning project.
These examples inspect evidence; they do not submit, cancel or resize jobs.

## Scheduler and ownership — on the cluster login host

```bash
JOB_ID=123456
scontrol show job "$JOB_ID" -o
sacct -X -j "$JOB_ID" -nP \
  --format=JobIDRaw,JobName,State,ExitCode,Start,End,ElapsedRaw,AllocTRES
sacct -j "$JOB_ID" -nP \
  --format=JobID,State,MaxRSS,AveCPU,TRESUsageInAve,TRESUsageInMax
```

The first accounting query provides allocation rows; the second includes
steps for diagnosis. Field availability and semantics depend on site setup.
Do not add both sets as independent resource costs. For completed jobs, read
the immutable launch/config record when `scontrol` no longer has the job.
See [sacct](https://slurm.schedmd.com/sacct.html).

## GPU sample — inside the owned allocation

Use the project's established way to enter an existing allocation or add the
sampler to a future launcher. Login-node GPU readings do not describe the job.
Do not assume CUDA ordinal 0 identifies the same physical GPU as NVML index 0.
Resolve the UUID of each assigned device and confirm its job/rank ownership.
Do not sample an entire shared node and attribute all devices to this job.

For one verified assigned GPU, the following Linux command creates a bounded
three-minute sample. Supply `GPU_UUID` and a new output file under the run's
scratch artifact directory first:

```bash
: "${SLURM_JOB_ID:?Run inside the owned Slurm allocation}"
: "${GPU_UUID:?Set the verified assigned GPU UUID}"
: "${GPU_SAMPLE_CSV:?Set a new run-specific output file}"
test ! -e "$GPU_SAMPLE_CSV" || exit 1
sample_rc=0
timeout --signal=TERM --kill-after=5s 180s \
  nvidia-smi --id="$GPU_UUID" \
  --query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv --loop=5 > "$GPU_SAMPLE_CSV" || sample_rc=$?
case "$sample_rc" in
  0|124) ;; # 124 is the expected finite sampling deadline
  *) exit "$sample_rc" ;;
esac
```

Retain the CSV header, units, missing readings and actual coverage. Repeat for
each assigned GPU/node with distinct files, and align times with training
history. This records GPU activity and memory/power, not SM occupancy or FLOPs.
A missing `timeout`, unsupported device query or denied profiler counter is
an instrumentation gap; use a supported bounded method instead of reporting 0.

Where Milalib is already available, Mila documents this optional occupancy
probe; run it within the same allocation and bound its duration as above:

```bash
uvx milalib monitor -i 5 -m sm_occupancy
```

Check the installed CLI's help and record its version first. Provision new
tools separately; do not trigger an unbounded package install inside a costly
GPU job. Profiler traces and TensorBoard belong in compute allocations.
[Mila monitoring methods](https://docs.mila.quebec/userguides/compute_utilization_guidelines/profiling/).

## Existing cluster routes for this account

On the Mac, `ssh mila` reaches Mila; Tamia is normally reached through
`ssh mila 'ssh -o BatchMode=yes tamia COMMAND'`. Credentials and MFA remain in
the user's existing authentication setup. The package contains none.

On Tamia, this process-local setup makes existing Cluv/uvx tools and the HTTP
proxy available without editing shared shell configuration:

```bash
module load httpproxy
export PATH="$HOME/.local/bin:$PATH"
```

Use Cluv from the owning configured project. A read-only connectivity check is
`cluv sh hostname`; it takes command arguments, not a quoted shell program.
Use current Slurm evidence for job status. Another cluster may require a
different proxy/module setup. Do not copy Tamia environments or these account
routes into a different user's project without adaptation.

## Trackers

Use only the project's authorized destination and metric payload. The originating
Nexus permission covers aggregate preparation counts; it does not authorize
adding machine, profiler, GPU or source metadata to that run. Local measurement
receipts remain useful without an external upload. Check both tracker freshness
and local counters before calling a job stalled.
