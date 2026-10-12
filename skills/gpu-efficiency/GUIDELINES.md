# GPU efficiency guideline for research agents

Version 1.0, reviewed 2026-10-12. Apply to owned GPU work from admission through
shutdown. These are reusable operating conventions; cluster policy and the
owning experiment's scientific contract remain authoritative.

## Read measurements correctly

Mila recommends checking its utilization dashboard and profiling before
scaling. Its guide labels reported SM occupancy below 5% critical, around 15%
poor, around 30% good and at least 50% strong. Treat these as investigation
heuristics for that metric, not universal admission thresholds. The guide also
offers Milalib monitoring and warns that W&B CPU/RAM measurements can describe
the whole node rather than the job's allocation. [Mila guide](https://docs.mila.quebec/userguides/compute_utilization_guidelines/profiling/).

NVIDIA's `utilization.gpu` measures the fraction of a sampling interval with
kernel activity. It is not the fraction of peak FLOPs achieved; 100% can still
hide inefficient execution. `utilization.memory` is memory activity, whereas
`memory.used / memory.total` is capacity occupancy. Some metrics are unsupported
on particular devices, including MIG cases. Preserve N/A rather than replacing
it with zero. [NVIDIA reference](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

Keep the exact provider, field, units, sampling interval and phase. Do not
equate Milalib's `sm_occupancy`, profiler occupancy, SM activity, GPU busy time,
VRAM capacity and model FLOP utilization. A low value identifies a question;
throughput, workload constraints and a profile determine the remedy.

## Before submitting or scaling

- Establish the owner, immutable inputs, intended scientific comparison,
  available resource budget and stop rule. Inventory existing jobs first.
- Use CPU allocations for downloads, archive preparation, tokenization,
  integrity checks and other CPU-only work. Prepare a durable verified cache
  before requesting GPUs where the workflow allows it. Node-local storage is
  allocation-specific: a CPU job's temporary directory is not automatically
  available to a later GPU job. Budget and measure required local staging.
- Select the smallest valid resource request using a representative real-data
  pilot or recent comparable measurements. Inspect current partition limits;
  some partitions allocate whole GPU nodes. Record allocated GPUs separately
  from active workers, along with CPU, RAM and walltime.
- Predeclare phase budgets: expected staging/compile time, first useful update,
  acceptable progress gap, diagnostic window, checkpoint time and shutdown
  reserve. Derive them from observed workload timings. For a new lightweight
  pilot, a 3–5 minute steady-state sample after warmup is a reasonable starting
  point, subject to a finite overall limit; this is our convention, not a
  Mila requirement or an automatic cancellation rule.
- Record how telemetry starts, stops and survives failures. Keep its overhead
  bounded. Install optional tooling before the expensive allocation when
  possible; do not silently change a running scientific environment.

## During the allocation

Check the first useful update, then a representative steady-state interval.
Recheck when entering evaluation, changing phases or encountering a progress
gap. Periodic monitoring should record counters and raise attributable events
without repeatedly invoking a model.

Measure examples or new tokens per second and optimizer updates per second
alongside per-GPU activity, power, memory, data-wait time and step-time spread.
For SAM or another replay method, distinguish new data from repeated forward/
backward passes. Count validation and checkpointing as their own useful phases.
Compare both steady-state speed and allocation-wide productivity, including
staging and shutdown; a fast kernel cannot repay an arbitrarily long idle setup.

If progress stops beyond the phase's declared allowance, inspect recent logs,
data workers, rank status, communication, storage and tracking. Save evidence
and use the project's existing checkpoint/stop/escalation procedure. Preserve
uncertain submission receipts and pending jobs. Never submit a duplicate to
resolve an observation failure.

## Diagnose before changing resources

**Long preparation or startup.** Locate the time in archive extraction, data
transfer, environment creation, compilation, authentication or tracking setup.
Move reusable CPU work into a bounded prerequisite and cache immutable inputs.
Keep unavoidable node staging visible in the cost. Fail prerequisites before
GPU admission rather than holding a GPU while waiting for access or downloads.

**Data supply or CPU pressure.** Time batch acquisition separately from GPU
compute. Measure storage and worker behavior before increasing worker counts.
PyTorch offers asynchronous data loading with workers and pinned-memory host
transfers; tune these against the allocated CPU/RAM and actual dataset.
Additional workers can increase shared-memory and I/O pressure. Persistent
workers also require checking augmentation and epoch-seeding behavior in this
project. [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html),
[data loading](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html).

**Idle devices or uneven ranks.** Reconcile scheduler allocation, launcher
world size, device mapping and per-rank progress. Inspect whether one rank
performs serial work while the others wait. Rank skew, synchronization or I/O
is not remedied simply by requesting more GPUs. A whole-node minimum is a
constraint to document, not proof that using one worker is efficient.

**Small or latency-limited workload.** Compare an eligible smaller device or
allocation and approved independent-job packing. Respect cluster isolation and
accounting. Do not enable MPS, reconfigure MIG or borrow another job's GPUs as
an implicit optimization. A larger microbatch is a proposal requiring the
scientific-equivalence checks below.

**Busy GPU with weak useful throughput.** Profile a short bounded section for
kernel launch overhead, host synchronizations, communication, redundant work
and memory traffic. Use the profile to select a change. Consider framework
features such as compilation or mixed precision only under the experiment's
numerical and reproducibility rules; neither is automatically equivalent.

**Checkpoint, evaluation or tracker stalls.** Measure their actual contribution
to allocation time. Preserve required evaluation and full-state checkpoint
semantics. Use bounded reporting/shutdown behavior and a durable local artifact
when the project supports it. Do not disable scientific measurements just to
raise the utilization percentage.

## Preserve the experiment

For conventional data-parallel training, effective batch is per-device
microbatch × accumulation steps × world size. Preserving that product is
necessary for many comparisons but not sufficient. BatchNorm, contrastive
negatives, SIGReg statistics, dropout replay, gradient reduction, clipping,
schedulers and data order may change with the decomposition.

Treat changed precision, loss computation, kernel implementation, accumulation,
world size and checkpoint schema as potentially material. Use the owning
repository's numerical tests and a matched real-data check. Retain baseline
measurements and failed attempts. Never edit the source snapshot of a running
job or bypass its resume identity to apply an optimization.

For an execution-only improvement, demonstrate unchanged intended work and
acceptable numerical behavior, then compare throughput and actual allocated
cost. For a scientific change, follow the project's separate admission process.
No universal batch-size, GPU-count, worker-count or occupancy target replaces it.

## Accounting and completion

For a constant allocation, GPU-hours = allocated GPU count × allocated elapsed
seconds / 3600. For resizing/requeues, integrate the actual allocation intervals
without double counting. Count failed allocations and setup time. Exclude queue
wait from allocated time; keep it as a separate scheduling metric. Avoid adding
job-level totals to their own steps or array parents to their tasks. Track GPU
models and partition billing conventions separately.

Slurm GPU accounting requires site support. Its step TRES summaries may contain
per-rank high-water statistics rather than the time average you need; preserve
their exact semantics. Missing GPU fields do not prove zero use.
[Slurm GPU accounting](https://slurm.schedmd.com/gres.html).

An RGU-week, an A100-week and a hardware GPU-hour are not interchangeable.
Retain the dashboard's definitions, device weights and reporting window. Do not
reuse a ratio from one email as a universal conversion factor or infer wasted
GPU-hours by multiplying an occupancy percentage by allocation time.

At completion, retain the compact [evidence record](references/efficiency-report-template.md),
source identity, phase timings, numerical validation and recommended next
request. A scheduler success is distinct from a valid scientific result.
Release owned idle resources according to existing authority. Stop only
monitoring processes owned by this run; never use a broad process kill.

If the bottleneck remains unclear, prepare a small reproducible case and
redacted evidence for Mila's IDT channels or office hours. Sending code,
credentials, traces or messages outside the project requires the user's
authorization for that destination; drafting the support package does not.
