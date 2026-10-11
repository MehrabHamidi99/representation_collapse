# Current research state

Updated 2026-10-11. See the [setup notebook](october/october11/2026-10-11-progress-report.md)
and [dependency plan](../configs/research-plan.json).

- **Repository and operating skills — implemented.** Git main initialized,
  requested origin configured, locked Python environment, four canonical skills
  with `.agents/skills/` discovery links. Publication status is in the notebook.
- **Expanded research path — documented; initial implementation ready.** The
  [periodicity research plan](auto/periodicity-research-plan.md) defines theory,
  matched categorical/continuous memory tasks, broad architecture coverage,
  temporal and depth recurrence, statistical confirmation, and compute stages.
  The frozen vector pilot is in `configs/experiments/vector-pilot-v1.json`; broader DAG scheduling remains pending.
- **Novelty audit and paper library — completed, scoped to 2026-10-11.**
  [Critical report](literature/novelty-review.md), 90 distinct bibliography
  entries and 90 public PDF copies; 26 records have substantive full-text
  passage reviews. Broad Floquet/collapse/memory claims are substantially
  anticipated. Five candidate open questions and falsification gates prioritize
  controlled useful-history loss; affine/commuting nulls and separate readout
  effects are required before the broad screen. No new empirical result follows
  from this audit. [Library and search limitations](literature/README.md).
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
- **Compute execution — authorized; initial CPU smoke completed on Tamia.**
  Owner explicitly requested cluv synchronization and initial launches. Tamia
  job **536757** completed with exit 0; Mila job **11190579** failed before
  calibration because uv selected the inherited controller environment. The
  corrected wrappers use `uv python find --system`. Cluv 0.1.5 synchronized
  both scratch checkouts; actual submission and resource bounds are recorded
  in ignored operations journals. No trained-model result is accepted.
- **Analytic mechanism controls — implemented and tested locally.** Seven
  families include affine/commuting nulls, noncommuting positive construction,
  amplitude versus normalized dimension, finite categorical basins, neutral
  continuous memory, and fresh-noise geometry without retained history.
  Local calibration passes 36 checks. Full source suite passes 29 tests.
- **Vector pilots — implemented; GPU launch pending access verification.**
  Complete-state GRU/LSTM categorical/continuous memory pilots record paired
  errors, individual horizon effects, DID, absolute variance, covariance
  dimension, and held-out ridge probes. Initial launches use seed 0 only;
  fixed learning rate and jointly trained encoder are engineering limitations.
- **Other Alliance clusters — Duo completed by owner on tamia1; reuse pending.**
  Eight aliases are configured. Coordinator on tamia2 verified zero reusable
  remote masters; node-local sockets explain the mismatch. Two fresh tamia1
  controller login attempts closed before Duo approval. Access and scheduler entitlements require actual
  node-local probes before staging or submission.
- **Graph replication — not run.** Graph evidence review and asset recovery
  remain required. Independent synthetic work may proceed alongside recovery;
  valid negative outcomes advance the plan.
- **Automatic research — actual Qwen worker running on Mila.**
  Mila PID **2812601** produced a verified actual Qwen proposal at 08:15:24 UTC.
  Worker uses bounded evidence,
  schema-checked proposals and exact model identity; 24 hours/22 remaining attempts
  after two recorded research attempts,
  at most 2,048 output tokens per call, no paid fallback. It cannot submit jobs
  or accept scientific claims. Mock tests pass model identity, cluster-scoped
  monitoring, heartbeat, caps, unknown usage and three-error stopping. The first
  proposal requires revision: its 10 CPU-hour request exceeds the envelope and
  its significance-only stopping rule is insufficient. No proposal was submitted.

Next bounded actions: finish tamia1 node-local cluv access verification; inspect
accounts and one-GPU partitions; stage locked Torch caches on login nodes;
launch bounded seed-0 pilots; retrieve and verify manifests and review Qwen drafts.
Recover graph assets in parallel. No broad sweep follows automatically.
