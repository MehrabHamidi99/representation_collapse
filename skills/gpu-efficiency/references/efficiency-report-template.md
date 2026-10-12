# Efficiency evidence record

Copy into the owning project's dated operations directory. Use unknown/null
for unavailable information. Attach small receipts and paths to large traces;
do not copy datasets, checkpoints, credentials or full environments.

- **Scope:** owner, project, cluster, job/step/array IDs, observation time/window.
- **Identity:** source commit/hash, config/data identity, environment, GPU model.
- **Authority:** existing campaign/controller, finite reservation, applicable
  stop rule; whether this is read-only diagnosis or an admitted change.
- **Allocation:** allocated GPUs versus workers, CPUs, RAM, partition, walltime;
  actual elapsed allocation, queue time separately, actual GPU-hours.
- **Phase:** preparation, warmup, optimization, evaluation, checkpoint, shutdown;
  phase budgets and measured time in each, including required node staging.
- **Progress:** starting/ending global update, new tokens/examples, last valid
  checkpoint, current phase, any evidence that progress stopped.
- **Measurements:** provider/version and exact metric names/units; per-device
  assignment, sample interval/window/coverage, missing readings; GPU activity,
  optional occupancy, power, used/total VRAM, data-wait and step-time statistics.
- **Productivity:** steady-state and allocation-wide useful throughput;
  replay/extra method work separately; samples/tokens per allocated GPU-hour.
- **Diagnosis:** observed evidence, likely bottleneck, alternative explanation
  and confidence; whether an account-wide notice has been attributed to jobs.
- **Intervention:** proposed/implemented change and how it preserves the
  experiment, including batch-dependent loss behavior and resume identity.
- **Validation:** matched before/after window and work, throughput/cost change,
  numerical/toy/resume checks as applicable, profiling overhead, remaining gaps.
- **Disposition:** continue, diagnose, right-size next admitted allocation,
  hold new scaling pending evidence, or execute the existing stop procedure;
  exact next action and owner. Preserve historical failures and dispatch receipts.
- **External reporting:** authorized destination/payload, or local-only evidence.

Keep account-level dashboard totals separate from this job's measured cost.
For a run with a fixed GPU count, state the explicit GPU-hour calculation.
