# Subgoal: reproduce T-GRAB Figure 10

Set on 2026-10-16. Status: **active; Figure 10 retraining has not started**.
Native engineering prerequisites have passed, but no trained reconstruction
exists. This is the immediate graph-baseline subgoal of the broader
representation-collapse project. A successful behavioral reproduction will
not by itself establish representation collapse.

## Target and scientific boundary

The target is Figure 10 in the active KDD 2026 T-GRAB appendix, not the nearby
AB (`Pdet(2,1)`) or AABB (`Pdet(2,2)`) Floquet analyses. Figure 10 plots
per-test-timestep F1 for the two panels, DTDG (left) and CTDG (right), on the
deterministic 100-node ER condition `Pdet(2,32)`. The sequence is A repeated 32
updates followed by B repeated 32 updates (period 64). The source appendix says
performance drops at the start of repetition phases and recovers after models
consume the new snapshot; it interprets that pattern as persistence-like
behavior and imperfect counting.

The immediate model of interest is GC-LSTM in the DTDG panel. The literal
Figure 10 target includes both panels and every plotted series. Track completion
as two explicit milestones: (1) reproduce the full DTDG panel, including
GC-LSTM, and (2) reproduce the CTDG panel if the original run assets and
implementation gates permit it. Do not call milestone 1 a reproduction of the
entire figure. Recover the original legend and plotting inputs before fixing the
series roster, seed aggregation, phase alignment, and uncertainty rendering;
`avg_f1` in a filename alone does not establish averaging over seeds.

In the current inherited paper snapshot, the DTDG panel PDF has SHA-256
`d85a32023c271fbbb99ac90beb04d5bb70ca1b2f102a881d1fb76344cd42ea3c`, the CTDG
panel PDF has SHA-256
`de253bf7ced2b51fb0cea6b822719d80532e86103494a3f1f6077915daace3d1`, and the
shared legend PDF has SHA-256
`cdf54869d3237ee371627334afc50ba8ccfed997fca821a9e4c6c409b64d83e3`. These
pin the repository assets to compare against; they do not identify raw plotting
arrays or prove how the curves were aggregated.

An independent local asset audit confirms the visible DTDG x-axis ticks are
0, 32, 64, 96, 128, 160, 192 and 224, but extraction did not identify the
number of plotted observations or the aggregation rule. The shared legend
names CTAN, DyGFormer, EdgeBank, TGAT, TGN, GCLSTM, T-GCN, EGCN, GCN, GAT and
Persistence; the roster actually drawn in each panel still needs source-level
confirmation. The active appendix TeX SHA-256 is
`b75cea32e00a26b1e28fefdc845e3464f3c6111fac31c2a0d60e87a9349bfcce` and the
KDD snapshot `main.tex` SHA-256 is
`f9d50cfaa3061dad6fbf9732e5f4206d8220651f80548f8bf5bb45c772850327`.

Keep three evidence labels separate:

1. **Historical replay:** original checkpoint(s), data, run identity, evaluation
   chronology, and plotting inputs are linked and replayed. This is the only
   route to claiming the original trained Figure 10 curves were recovered.
2. **Source-faithful reconstruction:** train with the pinned T-GRAB code and
   disclosed source behavior, including any test-dependent selection/stopping
   that the historical pipeline used. Report this as a reconstruction with
   methodological limitations, never as recovery of original weights.
3. **Corrected prospective reconstruction:** use the project's frozen
   validation-only selection and lifecycle safeguards. This is scientifically
   preferable for new inference, but its curve is not an exact historical
   reproduction. Keep its results separate from milestones 1 and 2.

F1 is a behavior measurement. It cannot establish covariance-dimension loss,
tangent contraction, phase-orbit collapse, or causal erasure of task-required
history. After behavioral reproduction, any collapse follow-up must capture the
complete graph recurrent state (all nodes, hidden and cell components, fixed
ordering), test trajectory/Jacobian periodicity before using Floquet language,
and include a causal memory probe. Report covariance geometry, finite-time
tangent propagation, task performance, recoverability, and causal memory as
separate estimands.

## Known starting evidence

- Public T-GRAB source is pinned at
  `31293102b065feef7817ff86211427d7594edfdb`; the inspected GC-LSTM dependency
  is PyG Temporal `0.54.0`, commit
  `46abd39abec0cd8de23c0e489a617c320c3fc5e8`. These source identities do not
  identify the environment that produced the published curves.
- The exact public primary `Pdet(2,32)` CSV is staged on Mila at the path and
  SHA-256 recorded in `configs/graph/primary-data-conversion-v1.json`. The
  approved conversion/independent-verification job `11273685` completed on
  2026-10-16: 337,920 rows, 3,072 snapshots, all three split comparisons and
  manifest checks passed. This is data-engineering evidence, not model
  execution or Figure 10 reproduction.
- The bounded Mila TSA archive search found legacy Floquet arrays and analysis
  source, but no original GC-LSTM recurrent-plus-link-head checkpoint in the
  enumerated scope. It did not establish global absence. The arrays are not
  linked to a Figure 10 checkpoint or producer.
- Native v2 calibration job `11261627` completed on Mila (CPU 1, 2 GiB,
  74 seconds, 0 GPUs). The worker executed 50 pinned-cell forwards and the
  frozen derivative/permutation controls. Its independent audit reconciled 60
  arrays, 27 comparisons and 1,337 scalar checks with zero mismatches. This is
  an untrained engineering assay only; N=100 model training and benchmark
  competence remain unverified.
- The candidate architecture (128 hidden units, K=2) gives 232,833 parameters
  for the pinned GC-LSTM plus link head, consistent with the paper's rounded
  233k parameter table. The paper states five seeds and training-loss patience
  of 100 without a fixed epoch cap. The present candidate has one seed and a
  bounded 200-epoch/3600-second cap; a successful fit would be a corrected
  prospective reconstruction, not the paper's five-seed aggregate. Original
  run commands, seed list, checkpoint and raw plotting arrays remain missing.
- The owner explicitly authorized the compute needed for Figure 10 retraining
  on 2026-10-16 and waived separate compute approval and limits for this
  subgoal. This supersedes the earlier consumed-conversion-job restriction for
  this subgoal only. The trainer/checkpoint/reporting path remains incomplete
  and no trained run has started. This authorization does not waive source
  review, native training parity or GPU-throughput measurement.
- A local 256-update, train-only CPU prefix-canary draft has nine passing
  fake-backend contract tests, but independent engineering review rejected
  launch. It lacks an outer import-inclusive supervisor and physical resource
  attestation, external source/command/admission binding, unambiguous
  post-optimizer-step accounting, and the frozen canary's conditional timing
  schedule. It is not competence training and cannot size the full fit as
  currently implemented. Review:
  [`figure10-canary-engineering-review-20261016.md`](../../runs/review/figure10-canary-engineering-review-20261016.md).
- The complete-epoch train/validation lifecycle core is implemented separately
  in `src/representation_collapse/tgrab_figure10_fit.py`. Independent review
  found no lifecycle defect and its seven spy-backend tests pass. This core
  enforces 2,559 updates per epoch, fresh train-state replay before validation,
  score-before-consume validation and strict earliest-tie selection. It does not
  yet bind the production backend/data, persist/reload a selected checkpoint or
  authorize a final held-out test pass; this is not a complete fit runner.

The evidence sources are the [graph reproduction audit](tgrab-reproduction-audit-2026-10-14.md),
[runner readiness report](tgrab-reconstruction-runner-readiness-2026-10-15.md),
[primary conversion review](tgrab-primary-data-conversion-independent-review-2026-10-16.md),
[primary conversion verifier review](tgrab-primary-data-verifier-independent-review-2026-10-16.md),
and [graph baseline/history release map](graph-baseline-history-release-map-v1.md).
The [local figure-asset audit](../../runs/review/figure10-assets-20261016/evidence-report.md)
and [independent subgoal review](../../runs/review/figure10-plan-review-20261016/receipt.md)
confirm the target pin and execution gates; the audit found no local raw
per-seed curves or original T-GRAB GC-LSTM/link-head checkpoint.

## Work sequence and acceptance

1. **Freeze the figure target.** Extract and hash the Figure 10 source PDF(s),
   all-panel legend, active TeX include, paper/source revision, exact plotted
   timestep range and phase/change-point locations. Recover raw per-seed,
   per-timestep F1 arrays or the source plotting script and identify whether
   curves average across seeds, test periods, or both. Record what is missing;
   do not digitize the PDF and present it as raw observations.
2. **Reconcile exact data and model lineage.** Recheck primary CSV and converted
   artifact hashes against the closed receipts. Search only authorized
   owner-controlled archives for recurrent and link-head weights, experiment
   configuration, seed, selected epoch, environment and prediction chronology.
   Preserve a bounded search manifest. If exact weights remain unavailable,
   record the run as reconstruction before any training.
3. **Pass pre-fit native bootstrap parity (node2a).** Before any fit or
   validation-only selection, independently review and execute the deterministic
   synthetic initialized-state checkpoint fixture: observe the native
   state-dictionary catalog, round-trip both model components and complete H/C,
   and check forward, loss, gradient, dropout/mode and transaction crash/reopen
   behavior. It must use no training updates, primary data, validation or test
   payload. Preserve hashes and failures. This is an execution prerequisite,
   not model evidence.
4. **Release the runnable path.** Complete and independently review the pinned
   cell import, data loader, recurrent-plus-head checkpoint lifecycle, complete
   epoch train/validation chronology, per-step metric capture, offline
   reporting, timeout/process ownership and terminal cleanup. A package-import-
   only failure cannot be treated as a model result. The 256-update prefix
   diagnostic is a separately frozen runtime probe and is not fit sizing,
   competence, or a Figure 10 result.
5. **Prepare the authorized compute run.** Inspect live scheduler/cluv and
   existing project jobs; read the current compute plan and GPU-efficiency skill
   for GPU work. Freeze cluster/account/partition, CPUs/RAM/device, walltime,
   exact source/config/data hashes, output paths, STOP path and failure/timeout
   rule. The owner has waived separate compute approval and limits for this
   subgoal; this does not waive implementation or scientific release gates.
6. **Run the smallest valid reproduction campaign.** Historical replay takes
   precedence if exact checkpoints are found. Otherwise, first complete the
   frozen competence-first GC-LSTM reconstruction for the DTDG panel; do not
   tune seeds, threshold, split, stopping or phase alignment after seeing test
   curves. Add only the other DTDG/CTDG runs needed for the exact figure after
   each model's source, environment, data, stopping and resource gates are
   independently pinned. A bounded/censored or incompetent run is inconclusive
   for reproduction, not a negative collapse finding.
7. **Reduce and compare blind to desired appearance.** Store each target's
   confusion counts and F1 before averaging. Recreate paper ordering, per-step
   phases and change-point marks from raw arrays; retain seed-level traces,
   mean/uncertainty computation and the plotting script. Compare numeric curves
   and plot geometry to the source figure under predeclared tolerances, with a
   source-aligned overlay and residual report. Do not select the best-looking
   run.
8. **Separate state analysis.** Only after the behavioral target is accepted,
   replay the accepted checkpoint(s) while recording node-resolved H/C state and
   chronology. Test phase-wise state closure and Jacobian closure independently.
   A low-rank phase-conditioned trajectory or an F1 dip alone is not a collapse
   result; causal-history loss needs a task-required cue, paired control and
   intervention with shams.

## Stop conditions and deliverables

Stop before model construction on any unresolved source/data identity, missing
approval, environment mismatch, checkpoint failure, reporting failure, existing
ambiguous job, STOP signal, or invalid resource forecast. Stop at the approved
attempt/walltime bounds and preserve incomplete outputs; no automatic retries,
resume, seed expansion, or test-driven tuning.

Deliver a machine-readable run manifest; source/data/environment/checkpoint
hashes; raw per-timestep arrays and per-seed summaries; scheduler and process
receipts; a reproducible figure script; source-aligned comparison plot and
residuals; an independent review receipt; and a short report stating whether
each milestone was historically replayed, reconstructed, or remains open.
Only after those pass may a separate report assess representation geometry or
causal memory. A positive behavioral reproduction is a prerequisite for
interpreting Figure 10's phenomenon, not evidence of representation collapse.
