# Current research state

Updated 2026-10-12. See the [latest notebook entry](october/october12/2026-10-12-progress-report.md),
the [setup notebook](october/october11/2026-10-11-progress-report.md)
and [dependency plan](../configs/research-plan.json) (revision v2).

- **Manuscript revision — venue-neutral diagnostic paper, 2026-10-12.**
  [When Does Repetition Erase Memory?](../paper/README.md) now has a seven-page
  main text and eight pages of references/appendices (15 total), complete
  proofs, four figures, and all 32 pilot cloud measurements. The scope is
  diagnostic limitations and controlled ordering, with no learned-collapse,
  empirical equivalence, or cross-modal claim. Reproducible plots use the
  verified seed-0 summaries and explicitly constructed analytic examples;
  a new probe-floor counterexample matches the actual ridge implementation
  at 101 amplitudes. The full source suite passes 48 tests, and all 36
  mechanism checks pass. The PDF builds without warnings or unresolved
  references and was visually reviewed. The paper dependency group is locked.
  [Readiness review](auto/manuscript-review-2026-10-12.md) records remaining
  novelty and evidence requirements; editorial completion does not establish
  scientific submission readiness. No new training or worker restart.
- **New-work review and live-status check — 2026-10-12.** Reviewed document
  commit `eb12ea1` and stopped-pass record `dc5e020`. Corrected numerical and
  statistical overstatements in the current design/theory/paper: FP32 tolerance
  does not imply bitwise equality; commuting cue propagation does not imply
  equal raw error; the readout-offset MSE gap needs its cross term; finite MSE
  competence alone does not bound local singular values; rank ceilings include
  state dimension; sign-flip exactness needs exchangeability; the adjusted
  ten-seed interval has half-width 1.1215 seed SD, not 0.7. The follow-up repair completes
  `diagnostics.py` as tested local utilities: R0/single-WRITE constructors,
  strict observation validation, streaming oracle and corruption controls.
  All 48 tests pass, including 14 new regressions; independent review finds
  no blockers. Source, tests and usage guide are verified and included in the
  publication update requested by the owner; no cluster deployment. Full Gate B training integration remains incomplete. Frozen v1
  model code, existing tests, run configurations and inherited snapshots are
  unchanged. [Diagnostic usage](auto/diagnostics-implementation.md). **Follow-up SSH succeeds with approved network access:** Qwen is verified
  stopped at its 22-call cap (October 11 23:28 UTC), with 22 schema-valid
  reviews and zero consecutive errors. Mila has no project-name matching queued
  jobs; Alliance scheduler status was not refreshed. No worker restart or new job.
  [Review and remaining gaps](auto/new-work-review-2026-10-12.md) and
  [dated status](auto/research-status-2026-10-12.json).
- **GitHub publication and Overleaf fix — 2026-10-12.** The owner requested
  committing and pushing all current changes. Four tracked skill-directory
  symlinks were replaced by regular discovery entries that require reading
  the canonical instructions. `paper/main.tex` is the current manuscript
  entrypoint. Qwen's 22-call cap was a coordinator launch choice: the original
  24-call bound minus two earlier attempts, not a provider restriction.
  Commit `4023674` was pushed to GitHub main, including the previous
  document commits `eb12ea1` and `dc5e020`. A fresh remote ref check confirms
  the publication; the tree has zero symlinks. Evidence is appended to the
  latest notebook.
- **Submission framing consolidated — documents only, no new empirical result
  (2026-10-12).** The [research charter](auto/research-charter.md) is now the
  canonical statement of the question, hypotheses, estimands, claims,
  non-claims, antecedent deltas and publication rule; the
  [theory propositions](auto/theory-propositions.md) consolidate P1–P12 with
  proofs, fixtures, limits and predictions and tabulate fourteen corrections to
  the inherited draft; [experimental design v2](auto/experimental-design-v2.md)
  fixes the Gate A–D order (the 351-checkpoint screen is deferred coverage),
  the competence-first ladder, the preregistered `δ16`/`DID` family, the
  measurement battery, an evidence ledger and the compute estimate (no
  envelope requested); the [implementation audit](auto/implementation-audit.md)
  maps code and tests to the contract; the original `paper/` skeleton was
  subsequently replaced by the diagnostic manuscript described above.
  Working framing: useful-history
  loss under controlled periodic forcing in fixed-weight recurrent models.
  Audit findings that qualify earlier evidence: the ridge-probe feature floor
  (`1e-8`) is active at the observed ~1e-9 per-feature state amplitudes, so
  the v1 R2-versus-R16 probe contrast carries a horizon-dependent effective
  regularization until re-reduced with the saved `probe_feature_scale`; the
  FP32 closure tolerance (`1e-7`) is scale-dependent and does not certify
  bitwise equality; report coordinate spacings and sparse versus dense changes,
  with fixed-weight FP64 replay as a separate assay. TF32/cuDNN flags are
  unrecorded in v1 provenance. Ten
  document-versus-code mismatches (evaluation pairs 256 vs 1,024, driver pairs
  2 vs 8, aggregate 0.25 gates vs coordinate-wise 0.95/0.05, R0 unsupported,
  single learning rate and seed) are listed in design v2 §1.5. Local checks:
  34/34 tests, 10 + 36 calibration checks with matching manifest hashes;
  `uv sync --locked` alone omits torch (use `--all-groups`), and `uv run` was
  broken on this Mac by hidden `.pth` files (PYTHONPATH fallback documented).
- **Periodicity diagnostics — all four jobs completed, verified scheduler at 16:19 UTC.** Frozen
  analysis source `458bc9cfce474b8e11e433e33bf132faf5032387` runs existing seed-0
  checkpoints: Narval GRU categorical **5254858**, continuous **5254871**;
  Rorqual LSTM categorical **22984956**, continuous **22984963**. All four
  finish `COMPLETED/0:0` and write completed artifact statuses. The last ends
  at 16:14:24 UTC. Worker summaries report all numerical checks passing,
  64 dense tangent windows and 1,536 step-Jacobian evaluations across the batch.
  Independent server checks verify all 28 manifest-listed artifact hashes and
  unchanged original checkpoints. Independent raw-array reduction remains
  pending. Each requests one GPU, four CPUs,
  16 GiB and fifteen minutes; four submissions exhaust this separate diagnostic
  batch's one requested GPU-hour cap, with at most two simultaneous jobs.
  No retraining. [Frozen protocol](auto/periodicity-analysis-v1.md) covers
  periodic versus matched varying inputs, full-state cycle closure, paired cue
  separation, finite-time tangent spectra and FP32/FP64 replay. All 34 local
  tests and independent prelaunch reviews pass; completed workers do not
  replace independent numerical verification. Every checkpoint previously failed competence;
  this batch cannot establish harmful collapse or certified Floquet dynamics.
  [Launch evidence](auto/periodicity-analysis-launch.json) and
  [current completion evidence](auto/periodicity-analysis-status.json).
- **Initial engineering phase — completed and independently verified.** Narval
  calibration job **5234514** passes 10 + 36 checks. All four full 1,500-step,
  seed-0 GPU pilots finish `COMPLETED/0:0`: GRU categorical **5234860**, GRU
  continuous **5235508**, LSTM categorical **22980985**, LSTM continuous
  **22980998**. Six artifact directories pass raw-output/source/hash checks and
  separate native scheduler reconciliation. All four models fail short-horizon
  native memory competence; categorical accuracy is near 6.25% chance and
  continuous error is near the exact zero-predictor baseline. **Harmful collapse
  remains inconclusive.** R2 conditional probes retain cue counterevidence;
  float32 resolution and native readout require diagnosis. See the
  [final report](auto/initial-phase-report.md), [scientific critique](auto/initial-phase-scientific-review.md),
  [committed results](auto/initial-phase-results.json), and
  [competence-recovery protocol](auto/competence-recovery-plan.md).
  Execution source is `9004a0c114a8b7c5335dca3d5247449fd10b5ffa`; Narval uses
  A100 and Rorqual uses amended H100 MIG `3g.40gb`/15-minute allocations.
  All four GPU submissions are consumed; no retry or new training is authorized
  by this result. Verified full copies are on Tamia/Mila scratch, with private
  project-volume archives on the execution clusters. No Mac checkpoints.
- **Next experiment — competence diagnosis first.** Preserve v1 failures.
  Separate immediate WRITE/query, full-prefix acquisition, short retention and
  native QUERY readout before long-horizon geometry or Floquet interpretation.
  A new frozen training protocol and allocation are required. Graph reproduction
  remains blocked on assets and complete-state adapter validation.
- **Repository and operating skills — implemented.** Git main initialized,
  requested origin configured, locked Python environment, four canonical skills
  with regular `.agents/skills/` discovery entries. Publication status is in the notebook.
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
- **Compute bootstrap — completed, with failed attempt retained.**
  Owner explicitly requested cluv synchronization and initial launches. Tamia
  job **536757** completed with exit 0; Mila job **11190579** failed before
  calibration because uv selected the inherited controller environment. The
  corrected wrappers use `uv python find --system`. Cluv 0.1.5 synchronized
  both scratch checkouts; actual submission and resource bounds are recorded
  in ignored operations journals. Trained outputs now pass engineering acceptance;
  no harmful-collapse scientific claim is accepted.
- **Analytic mechanism controls — implemented and tested locally.** Seven
  families include affine/commuting nulls, noncommuting positive construction,
  amplitude versus normalized dimension, finite categorical basins, neutral
  continuous memory, and fresh-noise geometry without retained history.
  Local calibration passes 36 checks. Full source suite passes 34 tests
  (29 at the time the families were added; 34 after the periodicity module).
- **Vector pilots — executed; all four competence gates failed.**
  Complete-state GRU/LSTM categorical/continuous memory pilots record paired
  errors, individual horizon effects, DID, absolute variance, covariance
  dimension, and held-out ridge probes. Initial launches use seed 0 only;
  fixed learning rate and jointly trained encoder are engineering limitations.
- **Alliance access — all nine configured targets verified at 12:31 UTC.**
  Fresh native cluv logins from a held Tamia2 controller established eight
  reusable remote masters. Separate passive checks verified all nine identities;
  seven expose scratch and scheduler associations directly. Follow-up login-shell
  probes verified scratch and scheduler associations on Killarney and Vulcan too;
  remote verification now initializes that shell environment.
  Mila separately verified from the Mac. Keep the project controller open and
  use its socket adapter; reuse after controller closure remains unproven.
  [Routes, accounts and evidence](auto/cluster-access.md).
- **Graph replication — not run.** Graph evidence review and asset recovery
  remain required. Independent synthetic work may proceed alongside recovery;
  valid negative outcomes advance the plan.
- **Automatic research — verified stopped at its bounded call cap.**
  Fresh Mila observation at 2026-10-12 01:50 UTC confirms PID 2812601 is absent,
  heartbeat and terminal status both say stopped, and exit code is 0. The
  worker finished October 11 at 23:28:39 UTC with 22 schema-valid reviews,
  zero consecutive provider errors and 125,730 reported tokens. Its final
  proposal recommends independent raw-array reduction, but its proposed
  tolerances and inference still need review. Schema validity is not scientific
  acceptance. The proposal-only worker cannot submit jobs; coordinator cluv
  polling is separate. Its 22-attempt allowance is consumed; no restart was
  performed. Mila source remains `8fce9b0`, predating the new documents.
  Approved SSH bypasses the earlier local sandbox restriction. Remote cluv is
  0.1.5; the old Tamia2 hostname failed DNS resolution from Mila, so a fresh
  Alliance connection/scheduler check remains outstanding.

Next bounded actions: independently reduce the completed periodicity raw arrays
(closure in ULP units against FP64; probe floor-hit counts from the saved
`probe_feature_scale`); complete the remaining Gate B prerequisites from the
implementation-audit backlog (stage-aware training validator, coordinate-wise
gates, three-phase state capture, intervention/sham and statistics modules) before any new compute request;
freeze the v2 statistics (SEI, family, interval method) in a hashed record;
review bounded Qwen drafts against verified failures; recover and pin graph
assets. Restore the Tamia controller route and reverify its cluv connections;
Mila SSH works but the old Tamia2 hostname currently fails resolution. No broad
sweep or additional GPU training follows automatically.
