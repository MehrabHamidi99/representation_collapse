# GPU efficiency adoption and current evidence

Checked 2026-10-12. This project adopts the complete
[GPU-efficiency skill](../../../../../skills/gpu-efficiency/SKILL.md), its
[guideline](../../../../../skills/gpu-efficiency/GUIDELINES.md), diagnostic commands,
reporting template and adoption instructions. `AGENTS.md` routes GPU admission,
scaling and utilization diagnosis through that package. The `.agents/skills/`
discovery entry is a regular file, preserving Overleaf compatibility.

The six canonical package files match the shared installation at
`/Users/mehrabhamidi/.agents/skills/gpu-efficiency/` and the Mila installation at
`/home/mila/m/mehrab.hamidi/.agents/skills/gpu-efficiency/` byte for byte.
The cluster agent explicitly re-read the Mila skill, guideline and references.
Project ownership, frozen science and allocation envelopes govern every action;
permissions and workloads from the package's originating project do not transfer.

## Attributed workload and measurements

The passive native cluv 0.1.5 pass from Tamia1 at 17:01–17:02 UTC verified all
nine configured Alliance targets and found no matching active project jobs.
Mila was checked separately through its existing Mac-origin connection at
17:11 UTC; its matching project queue was empty. Other projects' jobs were
neither changed nor attributed to this study.

There is no current owned GPU allocation from which to measure activity,
occupancy, power, VRAM or steady-state update throughput. Those values are
**unknown**, not zero. Login-node GPU readings would not measure this workload.
The local environment is macOS/arm64 with Torch 2.6.0; CUDA and MPS are unavailable
to this session. Local engineering checks run on CPU and do not validate CUDA
performance or permit substituting a different backend into a frozen experiment.

Historical scheduler accounting records 555 allocated device-seconds for the
four initial fits and 608 for the four periodicity analyses: respectively
0.15417 and 0.16889 allocated device-hours. These totals use job allocation rows,
without adding their own steps or queue wait. A MIG slice is retained as an
allocated device, not converted to a full-GPU equivalent. Host MaxRSS does not
measure GPU memory, and allocation duration alone does not establish efficiency.

## Consequences for the next experiment

The four-fit Fir/Nibi discovery canary remains unapproved and unexecuted. Both
targets expose account `def-ioannism_gpu`, QOS `normal`, UP partition
`gpubase_bygpu_b1` and typed H100/full or H100-MIG devices. The availability
tuple `gpu:h100:1`, four CPUs, 16 GiB and 30 minutes is not a measured sizing
recommendation. Only the older `ec1c489` project source is staged there; the
new trainer/wrapper, offline environments/caches and canary preflight receipts
are missing.

Before GPU admission, prepare and verify immutable code and environments on
CPU, then record exact source/config/account/device bindings and the approved
finite allocation. The canary must measure first useful update, training versus
validation/checkpoint time, examples and optimizer updates per second,
allocated-device cost and available per-device telemetry. Sampling ends with
the useful experiment; it must not prolong an early competence failure merely
to fill a profiling window. Keep unsupported MIG readings unavailable.

Use the resulting matched-work receipt before a longer discovery grid or
scaling. Preserve batch 32, task/loss, optimizer, data order, pure FP32/TF32
settings, validation gates and complete recurrent-state measurements. Batch,
precision, kernels or GPU count cannot be changed solely to raise occupancy.
The profiling workflow follows
[Mila's GPU-waste guide](https://docs.mila.quebec/userguides/compute_utilization_guidelines/profiling/);
the installed project package adds ownership and scientific-control checks.

The current disposition is to continue the approved Qwen proposal worker and
local scientific preparation, while staging/readiness and exact allocation
approval remain prerequisites for new GPU work. No jobs were submitted,
cancelled, resized or retried, and no telemetry was uploaded externally.

Evidence is retained in ignored `runs/ops/overnight-compute-readiness-20261012.json`,
its GPU-efficiency amendment, and `gpu-efficiency-local-adoption-20261012.json`.
The current [tracker](../../../2026-10-26/local/report/TRACKER.md) and dated notebook record their hashes.
