# Current research state

Updated 2026-10-11. See the [setup notebook](october/october11/2026-10-11-progress-report.md)
and [dependency plan](../configs/research-plan.json).

- **Repository and operating skills — implemented.** Git main initialized,
  requested origin configured, locked Python environment, four canonical skills
  with `.agents/skills/` discovery links. Publication status is in the notebook.
- **Source audit — reviewed.** Both graph snapshots and Floquet draft preserved;
  [inventory](auto/source-inventory.md) separates attributed claims from verified
  evidence. Upstream T-GRAB code/data identified, not yet pinned or reconstructed.
- **CPU measurement infrastructure — verified locally.** Known-rank covariance,
  noncommuting Jacobian order, contracting periodic response, and nonnormal
  transient-gain controls pass. Latest synthetic artifacts:
  `runs/calibration/setup-2026-10-11-v4/`. These are not trained-model findings.
- **Graph measurement contract — protocol drafted; adapter pending.** Complete
  node/hidden/cell state, prediction chronology, JVP/finite-difference validation,
  cycle closure, and independent sampling required before empirical acceptance.
- **Legacy graph assets — blocked on recovery.** Missing checkpoint identity,
  raw trajectories and analysis source. Pin upstream training code and dataset
  revision; resolve AB/AABB mismatch. [Reproduction protocol](auto/graph-reproduction.md).
- **Compute preparation — implemented; allocation approval pending.** cluv 0.1.5
  native schema verified; Mila and Tamia reached read-only (Tamia through Mila).
  One Tamia CPU smoke is proposed: 1 CPU, 2 GB, 10 minutes, zero GPUs, no retries.
  Authentication/staging prerequisites are in [compute plan](auto/compute-plan.md).
- **Other Alliance clusters — prospective.** Example profiles supplied; access,
  allocations and current partitions require verification before enabling them.
- **Graph replication and modality pilots — not run.** Graph evidence review
  precedes matched vision/audio/text/time-series pilots. Valid negative outcomes
  advance the plan. [Research program](auto/research-program.md).
- **Automatic research — session workflow ready.** Subagent ownership, artifact
  contracts and scientific review gates implemented in skills/docs. Persistent
  controller and scheduled model calls are not installed.

Next bounded actions: approve the CPU infrastructure smoke after its preflight;
pin/audit T-GRAB source and data; recover legacy checkpoints; implement and
validate the complete-state graph adapter before requesting a GPU pilot.
