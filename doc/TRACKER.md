# Current research state

Updated 2026-10-16 19:12 UTC. See the [latest notebook entry](october/october16/2026-10-16-progress-report.md),
the [October 15 notebook](october/october15/2026-10-15-progress-report.md),
the [October 14 notebook](october/october14/2026-10-14-progress-report.md),
the [October 13 notebook](october/october13/2026-10-13-progress-report.md),
the [prior-day notebook](october/october12/2026-10-12-progress-report.md),
the [setup notebook](october/october11/2026-10-11-progress-report.md)
and [dependency plan](../configs/research-plan.json) (revision v2).

- **Immediate graph-baseline subgoal — reproduce T-GRAB Figure 10.** The
  target is `Pdet(2,32)` per-timestep F1, with a GC-LSTM-first DTDG-panel
  milestone and full two-panel reproduction tracked separately. Historical
  replay, source-faithful reconstruction, and corrected prospective
  reconstruction have distinct evidence labels. The converted primary data is
  independently verified. Native engineering calibration job11261627 and its
  independent saved-array audit passed (60 arrays, 27 comparisons, zero
  mismatches); this is not trained-model evidence. Original model checkpoints/
  plot inputs remain unresolved. A 256-update CPU prefix-canary draft has nine
  focused tests, but independent review rejected launch pending an outer
  resource/source/admission wrapper, reliable optimizer-step accounting and the
  frozen timing schedule. Separately, the full train/validation epoch lifecycle
  core passed independent review and seven spy tests; it is not integrated with
  model/data/checkpoint/test handling. The fit lifecycle core independently passed seven spy tests. The native adapter
  remains under revision for enforced seed-0 construction and epoch-keyed rows.
  No Figure 10 retraining has started. Before any fit, the release map requires
  node2a synthetic native checkpoint/forward/loss/gradient parity; the selected
  checkpoint must later pass node2b before test access. The owner
  explicitly authorized Compute Canada retraining for this subgoal without
  additional approval or resource limits on 2026-10-16. Figure 10 is a
  behavioral target; no collapse claim follows from its F1 plot. See both dated
  wrapper review receipts in the latest report.
  Independent local-asset audit and subgoal review passed the target pin and
  execution-gate wording; both report original raw curves/checkpoint absent in
  bounded searched scopes.
  See [Figure 10 reproduction subgoal](auto/figure-10-reproduction-subgoal.md).
  Latest step: Mila job11280346 completed the one-shot runtime-identity capture
  (12 seconds, 1 CPU/2 GiB/zero GPUs). Its records and inventory form a
  hash-consistent chain. Independent review found and accepted a fix for the
  stale preparation schema, which now pins the source identity and exact
  runtime module/distribution map. The first synthetic node2a job (11280406)
  failed before native model construction because supervisor journals and
  bootstrap atomic publication shared one output path. Its failed records are
  preserved. A separate child `fixture/` output path now resolves the
  collision; independent review accepted the exact fix and 30 focused tests
  pass. No node2a parity or Figure 10 training has passed. A read-only audit
  confirms the complete training/replay/test/plot lifecycle is still not
  integrated, so fit code must be completed after node2a before retraining.
  Latest step before the capture: the one-shot node2a runtime-identity capture wrapper passed
  independent review after closing its bootstrap source race; its focused
  review and subsequent Slurm startup/TRES/restart-field corrections passed 16
  focused tests and independent review. Three submitted capture attempts failed
  in 0, 1 and 1 seconds, with no capture artifact or training. They exposed the
  configured cluv job-script-path override, Slurm startup-state race, and
  observed `Restarts` field name. The capture wrapper now passes independent
  review at its current hashes (`eb2744a2…a31cd` / `ba1dfad8…18dad`), backed by
  17 tests; authoritative `Restarts=0`/`Requeue=0` are required while Mila may
  omit `SLURM_RESTART_COUNT`. The same correction is applied to the native
  node2a runner; independent review accepted exact hashes
  (`eae8f330…12b24` / `22266cad…378be`) after scheduler-field tests were added.
  A new capture-only attempt (Mila job11279364) passed source, receipt and
  scheduler admission, then failed before capture because a cgroup ancestor
  lacked `memory.max`. The fix skips ancestors where the kernel exposes no
  memory controller file, while requiring the job leaf files and a finite
  exposed effective limit. Independent review accepted capture/native hashes
  (`c3035baf…e52a6` / `f82792a6…e882b`, `604f4405…1fea` /
  `afbdbad5…b288a`); focused suites pass 24/24. No inventory or model training
  resulted. After tightening the run-root mode to 700, job11279383 reached
  capture-start but failed on a misspelled stop flag before inventory creation.
  The reviewed correction uses the actual signal flag and retains STOP-file
  checks (capture supervisor/test `24b201f2…008ac6` / `2971197d…73b794d`);
  the updated combined suite passes 25/25. The training-readiness audit still
  finds only component drafts and a 256-update diagnostic, not an integrated
  fit/replay runner. Job11279392 then reached the bootstrap and found the
  upstream model files were staged under a separate pinned snapshot rather
  than the project checkout. The exact three wrapper hashes and revision
  `31293102…4edfdb` are now staged on Mila; the capture plan and receipt bind
  that source root/revision/hash set and the supervisor rechecks it at both
  boundaries. Independent review accepted the exact capture supervisor/test
  hashes (`8853e541…45541e` / `9109b388…6c903c`), noting the revision is a
  fixed metadata label while executable identity is enforced by wrapper bytes.
  Figure 10 training remains unstarted.

  Latest update: node2a synthetic zero-update native parity passed on Mila as
  job11280629 and was independently accepted as an engineering gate only. The
  observed-transition metric guard is fixed and reviewed. The pure-data
  six-file selected-checkpoint codec is now implemented and independently
  accepted for integrity/serialization only; 13 focused and 54 combined
  Figure 10 tests pass (three Linux-only skips on macOS). Its create-only,
  manifest-last publication is fail-closed but does not provide atomic
  directory visibility. No scientific selection authority is implemented.
  Production training is still blocked on durable selection transaction,
  pre-step attempt accounting, complete epoch candidate snapshots, binding to
  verified data/runtime/source admissions, and node2b full hidden/cell replay
  parity. No Figure 10 training, curve, or collapse claim exists.
  The paper itself defines Figure 10 as per-timestep link-prediction F1 on
  `Pdet(2,32)`; it interprets within-phase dips and post-change recovery as
  persistence-like behavior. This is a behavioral baseline, not a measurement
  of representation dimension. The prospective representation analysis must
  measure state geometry separately.
  The fit lifecycle now has an independently accepted pre-update attempt
  barrier, exact production-policy enforcement and truthful completed-step
  accounting for invalid returned losses (12 focused synthetic tests). The
  native adapter validates that policy before calling model factories and
  forwards the attempt hook; shortened test policies are confined to the test
  module. The hook only provides durable charging when a production caller
  durably commits before returning. Integration with that journal, complete
  epoch checkpoint candidates and verified data/source receipts remains open.

- **Overnight progress confirmed; checkpoint schema specification accepted — October16.**
  The approved Mila conversion job11273685 ran from02:09:36 to02:11:42 UTC on
  October16 (05:09:36–05:11:42 Qatar), taking126seconds with exit0;
  independent checks covered337920 raw rows,3072
  snapshots and all split/manifest comparisons. Qwen call25 completed at01:15:32
  UTC (04:15:32 Qatar) on the required-history bridge. These establish actual
  overnight engineering and proposal activity, not uninterrupted inference,
  training, or a collapse result. Read-only full closeout has now rehashed
  sources/admission/output inventory and independently decoded the offline W&B
  transaction; a follow-up read-only hash check matched the pinned
  evidence-inventory file. Independent review accepted the narrow offline
  reporting/provenance closeout after107 checks; remote W&B delivery remains
  unverified. Its single approved allocation is consumed and no retry is
  authorized.
  Selected-checkpoint schema revision002 resolved SCHEMA-001 and its exact
  revision review accepted the **specification only**: 54 focused checks passed,
  including all three omitted failure stages. A standard JSON Schema validator
  remains unavailable. The new stdlib canonical JSON utility and its20 focused
  tests pass; independent review accepted only the primitive parser/helpers,
  not artifact semantics, file I/O, codec or native parity. The Gate B readiness
  review found the assigned-order contrast defensible but did not release
  training; it recommends separating the graph baseline/history task, freezing
  interpretation scope, and staging discovery/confirmation admission.
  At13:29 UTC local Qwen PID84719/start/source was verified alive and waiting
  for evidence:30 identity/schema-verified replies,27 tasks,156106 reported
  tokens, zero provider errors, no call in flight. Its newest bounded response
  correctly separates parser fixtures from loader integration but proposes
  inapplicable 503 author-fixture integration and assumes a legacy loader;
  those proposals were rejected. No continuous inference is claimed. See
  [structured status](auto/research-status-2026-10-16.json),
  [schema review](auto/tgrab-selected-checkpoint-schema-independent-review-2026-10-16.md)
  [utility review](auto/checkpoint-json-utility-independent-review-2026-10-16.md),
  [Gate B review](auto/gate-b-readiness-independent-review-2026-10-16.md) and
  [progress report](october/october16/2026-10-16-progress-report.md).

- **S04 serialized validation checker and conversion closeout — October16.**
  Root reran the focused JSON/checker suites on Python3.14:35 tests passed after
  repairing a test witness that relied on pre-3.14 built-in `sum` rounding.
  Independent S04 review accepted serialized-row consistency only. The offline
  conversion closeout verifier passed and the narrow closeout is independently
  accepted. Qwen call32 offered a generic dependency observation, but its
  proposed audit was rejected as stale/mismatched. No numerical experiment,
  training, new submission or collapse finding occurred.

- **Graph release map drafted; corrected Qwen critical-path proposal reviewed — October16.**
  The release map separates original T-GRAB baseline reproduction/reconstruction,
  native checkpoint parity, unchanged vector Gate B, and later graph
  required-history intervention. Independent review accepted the map after
  resolving RELEASE-001's pre-fit/bootstrap and selected-checkpoint sequencing
  ambiguity. Qwen call33
  supported protocol separation and scoped negative interpretation; its proposed
  schema review was stale and its analytic alternative was underspecified. No
  scientific conclusion or training release follows. See
  [release map](auto/graph-baseline-history-release-map-v1.md),
  [independent review](auto/graph-baseline-history-release-map-independent-review-2026-10-16.md),
  [Qwen review](../runs/ops/local-qwen-continuation-20261014T1650Y/review-033.json)
  and current [structured status](auto/research-status-2026-10-16.json).

- **Approved primary conversion completed — observed October16 02:13 UTC.**
  Native cluv submitted the single approved Mila CPU job **11273685** from
  `0fe14aa`; scheduler reports **COMPLETED/0:0 in126seconds**. The worker's
  independent raw-data verifier reports **337920rows,3072snapshots,N100**,
  all three split byte comparisons and the complete manifest comparison passing;
  zero duplicates/markers. Four converted files total2997959bytes on scratch.
  Physical CPU1/2GiB/GPU0, one conversion and one verification attempt;
  the one-submission envelope is consumed, with no retry authorized.
  Final closeout remains pending: source/admission rehash and independent offline
  SDK transaction decoding. Existing receipts show completed reporting through
  step2 and two retained owned identities exited without cleanup errors.
  Mila's master subsequently expired; nativecluv freshconnect at03:39UTC failed
  with network unreachable, before authentication. No job or worker was restarted.
  Local Qwen verified03:35UTC alive/waiting,25replies/22tasks,zeroerrors;
  this does not establish continuous model inference or training.
  [Current structured evidence](auto/research-status-2026-10-16.json).
- **Checkpoint lifecycle/schema progress and Qwen resumed on new evidence — October16 (historical status).**
  The selected-checkpoint terminal-load attempt loophole is repaired and
  independently accepted; 32 focused synthetic tests pass. A closed six-file
  checkpoint schema freezes thirteen artifact tags, twenty-one definitions,
  and all26 catalogued tensors/offsets plus selection transaction rules. Its475
  author checks passed; independent standards validation remains in progress,
  including whether anchored regular expressions reject terminal newlines as
  JSON Schema requires. No codec or native execution exists yet.
  The verified local Qwen worker completed call26/task23 against the new
  conversion evidence: 26 model/schema-verified replies,23 completed tasks,
  134,026 reported tokens,zero provider errors; it returned to
  waiting-for-evidence. Its correct boundary assessment was retained; its
  recommendation was rejected because it assumed consumed training allocations
  and bypassed native checkpoint/parity gates. No compute was authorized.
  [Checkpoint schema](auto/tgrab-selected-checkpoint-schema-v1.md),
  [terminal-load review](auto/graph-terminal-load-attempt-independent-review-2026-10-16.md),
  [Qwen answer audit](../runs/ops/local-qwen-continuation-20261014T1650Y/review-026.json).
- **Schema independent review found one narrow blocker — October16 13:18 UTC.**
  3,255 structural assertions and ECMAScript pattern witnesses passed; the
  schema declares draft2020-12. No standard validator was installed in five
  checked interpreters, so full metaschema conformance remains unclaimed. The
  transaction failure `failed_stage` enum cannot represent failures publishing
  the required immutable intent or applying/verifying GR selection. No codec
  is approved; an author-owned schema/protocol revision and failure fixtures
  are in progress.
  [Independent review](auto/tgrab-selected-checkpoint-schema-independent-review-2026-10-16.md).
- **Conversion+verification launch packet reviewed — October16 06:03 Yerevan (historical).**
  Integrated wrapper passes28fixtures (22base/root+6independent); nativecluv
  controller passes9author+14independent fixtures. Root rechecked final source,
  config and all helper/controller pins. Owner allocation remains absent;
  no submission or primary body parsing occurred. Exact technical config is
  enabled for immutable publication/staging; both external approval and an
  exclusive one-submission claim remain mandatory. Read-only Mila01:55UTC:
  partitionUP, projectqueueempty, scratch9.16GB, exact existing environment.
  Paper now states the independently reviewed motif-exposure limitation;
  build18pages, finalreferencesresolved/nooverfullboxes, changedpages visually
  inspected. No empirical claim added.
  [Job review](auto/graph-data-conversion-verification-job-independent-review-2026-10-16.md),
  [cluv controller review](auto/primary-conversion-cluv-controller-independent-review-2026-10-16.md).
- **Independent raw-data verifier and exact tensor catalog accepted — October16 05:48 Yerevan.**
  Raw-to-artifact verifier passes11author+18reviewer fixtures and independently
  rebuilds every snapshot/manifest without producer helpers. Source-derived
  checkpoint catalog passes477structural checks:26tensors/931332FP32bytes.
  Actual target reduction, native dictionary/load parity and trained claims remain
  unverified. Mila interactive nativecluv access restored01:34UTC after master
  expiry;21installed source files hash/RECORD-verified,15match public source.
  Conversion+verification integration and single-submission controller are under
  review. Technical config is pinned/enabled; a fresh external owner allocation
  receipt is still absent and mandatory. No job submitted.
  [Verifier review](auto/tgrab-primary-data-verifier-independent-review-2026-10-16.md),
  [catalog review](auto/tgrab-selected-tensor-catalog-independent-review-2026-10-16.md).
- **Scientific scope guard and checkpoint direction reviewed — October 16 05:22 Yerevan.**
  Independent source review accepts the registered-boundary exposure counting
  argument: equal full motif exposure would force forbidden control tiling.
  This limits periodicity-only mechanism attribution; frozen Gate B is unchanged.
  [Scope review](auto/ordering-mechanism-scope-independent-review-2026-10-16.md).
  The selected-checkpoint design direction passes independent review, with
  concrete catalog/schema, codec, observer and terminal-load/parity work still
  required. It identifies the actual existing load-before-terminal-flag gap;
  no checkpoint writer/loader or payload was produced.
  [Checkpoint proposal review](auto/tgrab-selected-checkpoint-codec-independent-review-2026-10-16.md).
- **Conversion-job review and actual Qwen continuation — October 16 05:18 Yerevan.**
  The disabled launcher passes16author +16independent preparation fixtures;
  root rechecked final source hashes. Actual SDK/job execution and independent
  primary reduction remain unverified. Nativecluv0.1.5 read-only Mila check at
  01:17UTC found main-cpu UP, no project-name queue matches, existing
  Python3.10.19/W&B0.30.0 metadata,9,162,264,576bytes scratch used below20GiB.
  No allocation was submitted or approved. Qwen completed both newly assigned
  scientific critiques, reaching25verified replies/22tasks/0errors. Whole replies
  were rejected: they confuse input spectra with recoverability and invent a
  native accuracy result. Native review extracts only defensible design limits.
  [Launcher review](auto/graph-data-conversion-job-independent-review-2026-10-16.md),
  [ordering scope assessment](auto/ordering-mechanism-scope-review-2026-10-16.md).
- **Converter and input diagnostics accepted — October 16 05:15 Yerevan.**
  Implemented primary T-GRAB conversion, separate split serialization and a
  lazy one-shot test store; 18 author +19 independent synthetic fixtures pass.
  Added realized motif-exposure, higher-order count, wrapped input covariance
  and Fourier-power diagnostics; 8 author +7 independent fixtures pass.
  Root rechecked all review-pinned source hashes and accepted engineering
  integrity only. Reviews repaired lazy-read authorization and mixed-input
  precision bugs. The primary body remains unparsed; no trained-model result.
  The guarded CPU conversion job is prepared for separate review and fresh
  allocation approval; it has not run. Checkpoint specification is in progress.
  Qwen identity verified at01:13:16UTC,23 replies/0errors, waiting for evidence;
  two distinct scientific critiques were queued at01:14UTC. Completion is not
  yet inferred. [Converter review](auto/tgrab-primary-data-conversion-independent-review-2026-10-16.md),
  [diagnostic review](auto/matched-input-statistics-independent-review-2026-10-16.md).
- **Research goal activated and worker rechecked — October 16 04:51 Yerevan.**
  At the owner's explicit request, the session goal is active: novelty-audited
  study of task-required history loss under matched periodic ordering, starting
  with graphs then a native non-graph modality. Positive collapse findings are
  conditional; preservation and valid negatives remain legitimate outcomes.
  [Durable goal and completion criteria](auto/active-research-goal.md).
  Qwen PID/start/source verified alive at00:51:27UTC:23verified replies,
  20completed proposal tasks,0providererrors, currently waiting for evidence.
  Last reply was October15 06:40:15UTC; continuous overnight inference is not
  established. No new trained-model result or allocation accompanies this check.
  Graph converter/store subsequently passed synthetic engineering review below;
  real conversion and trainer/checkpoint integration remain next.
- **Native graph calibration accepted — October 15 10:34 Yerevan.**
  Owner-approved native cluv job **11261627** completed `0:0` in **74 seconds**
  from reviewed published `3f24d63`: Mila/main-cpu/account mila/QOS normal,
  one CPU, physical 2 GiB cap, zero GPUs. Before submission, actual-reporter
  testing caught and fixed a path-versus-hash metadata mismatch; the older
  approved plan was preserved and superseded without consuming an allocation.
  Exact planned native counts completed; all 27 engineering checks pass.
  Independent reduction verified 60 arrays and 1,337 scalar checks with zero
  mismatches. Source/admission lineage, seven native and one audit offline
  W&B history records, and both retained owned cleanup sets verify. Root accepts
  tiny untrained D12 cell/wrapper engineering integrity only; cloud sync and
  trained-model/Floquet/collapse claims remain unverified. No allocation remains.
  Original primary T-GRAB CSV is now hash-verified on Mila scratch: 6,907,326
  bytes; no local payload, parsing or conversion. A source-anchored trainer
  implementation contract identifies the remaining sealed-store, checkpoint
  and lifecycle work. Qwen alive at 06:32 UTC: 19 verified replies, zero provider
  errors; next training-parity/control reviews supplied. No uninterrupted
  overnight inference or numerical work is inferred from liveness.
  [Results](auto/native-graph-calibration-v2-results-2026-10-15.md),
  [primary payload](auto/tgrab-primary-payload-staging-2026-10-15.md),
  [trainer contract](auto/tgrab-training-runner-implementation-contract-2026-10-15.md).
- **Correction published and staged; analytic review complete — October 15 02:49 Yerevan.**
  Reviewed v2 source `e4ab463` is clean, pinned and hash-verified on Mila.
  The staged plan is explicitly disabled; fresh one-CPU allocation approval
  remains pending. Whole-project scratch is 9,081,705,472 bytes, below 20 GiB.
  The prospective independent v2 verifier is frozen before any v2 outputs;
  it retains 60 arrays, 27 comparisons and the original numerical tolerances.
  An independently accepted one-step folding construction shows required-bit
  loss with unchanged pooled population covariance and no contracting local
  tangent direction on the initial support. This is an analytic measurement
  guard, without an empirical, periodicity or novelty claim.
  Qwen identity/source verified at 22:48:19 UTC: 14 replies, 71,910 reported
  tokens, zero provider errors; it was waiting for evidence. Three new bounded
  analytic questions are queued. Whole answers remain unaccepted.
  [Analytic review](auto/folding-memory-counterexample-independent-review-2026-10-15.md),
  [v2 result-verification contract](auto/native-graph-calibration-v2-result-verifier-2026-10-15.md).
- **Native import failure closed out; correction ready — October 15 02:25 Yerevan.**
  Mila11256293 endedFAILED1:0/23seconds at22:05:26UTC, before all model/
  derivative calls; noNPZ. ExactTemporal0.54/PyG2.6.1 umbrella import uses an
  obsolete module path. All17selectedfailurefiles/34,828bytes verify; durable
  offlineW&B failed summary haszero history and both retained ownedidentities
  reportexit/noerrors. Independent failure review accepts evidenceonly.
  The isolated actual-cell loader andv2integration passengineeringreview;
  15helperfixtures and23v2controltests pass, numericalcode/config unchanged.
  V1 ispreserved. InitialCPUslots4used/0remaining; newsingleMilaCPU1/2GiB/
  tenminute/zeroGPUrequest prepared, awaitingfreshownerallocation approval.
  Qwenalive/sourceverified22:16UTC,13replies/66,705reportedtokens, waiting;
  wholeproposalsunaccepted. Nooriginalgraphtraining orcollapseclaim.
  [Failure](auto/native-graph-calibration-failure-2026-10-15.md),
  [reviewed correction](auto/native-graph-calibration-v2-independent-review-2026-10-15.md),
  [next exact allocation](auto/native-graph-calibration-v2-launch-request-2026-10-15.md).
- **Native graph calibration submitted — October 15 02:05 Yerevan.**
  Nativecluv0.1.5 submitted Mila job11256293 from published15074fb. Scheduler
  confirmsRUNNING/cn-f003/accountmila/QOSnormal/main-cpu/CPU1/Mem2G/GPU0,
  tenminutes/no-requeue. Physicalcgroup2147483648bytes andoneCPU affinity
  verified; offlineSDK0.30 initialized before nativework. Exact60-wheel
  legacyCPUenvironment installed/hash/metadata/pipcheck verified in170seconds;
  wholeproject scratch9,008,236,544bytes<20GiB. Nativeoutcomes stillpending.
  This consumes the final initialCPUslot; noGPU orrecurringbudget added.
  [Frozen launch](auto/native-graph-single-cpu-launch-2026-10-15.md),
  [prospective result verifier](auto/native-graph-calibration-result-verifier-2026-10-15.md).
- **Resumption and resubmission preparation — October 15 01:14 Yerevan.**
  Local Qwen PID/start/source verified alive at21:11UTC, eleven replies; a new
  evidence/task event subsequently produced reply12. No inference or numerical
  work occurred during the preceding gap. Mila/Tamia project queues are empty;
  eight other Alliance targets lack reusable masters. Independent accounting
  confirms one remaining initial CPUcalibration slot:1CPU/2GiB/10minutes,
  zero GPUs, no retry/race. Three native agents prepare exact-cell calibration,
  isolated legacy-compatiblePython3.10 environment and independent review.
  Actual native work and submission await complete immutable readiness.
  [Budget audit](auto/compute-resubmission-envelope-audit-2026-10-15.md),
  [new status](auto/research-status-2026-10-15.json).
- **Local worker refreshed — October 14 13:24 UTC.** PID/start/source
  verification confirms ten replies, 51,176 reported tokens, nine completed
  tasks and zero provider errors. Whole recommendations remain unaccepted;
  later replies contain explicit arithmetic and readiness errors. Native
  implementation/independent review completed the concrete milestones below.
  All nine tasks are consumed. New accepted-result evidence has been published
  to the leaf's separate feed; it remains active for evidence-driven proposals.
  No native timer or cluster allocation is running.
- **Next step complete — October 14 13:15 UTC.** The forty-file archive
  audit and independent reduction are numerically verified: 11,700 comparisons,
  zero mismatches, all 2,280 norm checks pass. All nineteen envelopes contain
  twenty byte-identical rows. Norms decrease across different prefix files;
  those rows do not establish within-file dynamics or learned-memory collapse.
  [Report and verified plot](auto/legacy-floquet-array-audit-results-2026-10-14.md).
  The corrected graph components passed independent engineering review after
  two lifecycle fixes; 28 synthetic tests pass. Native graph execution remains
  blocked on environment/data/native parity/checkpoint/reporting/watchdog work
  and execution admission. [Review](auto/tgrab-reconstruction-implementation-independent-review-2026-10-14.md).
  Local Qwen PID84719 is source/start verified at13:14UTC with seven replies,
  35,935 reported tokens and zero provider errors; three more distinct
  proposal tasks are queued. No new cluster allocation or native graph run.
- **Live local continuation — October 14 12:54 UTC.** A new Qwen proposal
  worker on the owner's Mac is PID/start/source verified, with four exact-model
  replies and 20,384 reported tokens. No provider errors or request in flight.
  Three explicit tasks completed; it now waits for evidence. Repaired prompt
  source `351fa68`; separate STOP file and runtime journal. No new allocation.
  Whole recommendations remain unaccepted. Native agents prepare the independent
  raw reducer and corrected graph components, with a separate implementation
  reviewer. [Actual launch, limits and stop control](auto/local-autoresearch-continuation-2026-10-14.md).
- **Stored-array audit executed — October 14 12:43 UTC.** One admitted CPU
  run from clean published `351fa68` completed in 3.42 seconds. Producer reports
  nineteen envelopes with twenty byte-identical rows each; all 2,280 directed
  norm comparisons pass. Stored six magnitudes range 0.379784–0.407370; exact
  product, residuals, trajectory and original checkpoint remain unknown.
  Independent raw reconciliation is pending. Twenty offline W&B metric records
  and terminal summary are verified; cloud delivery remains pending. No graph
  forward, optimizer or Jacobian execution occurred.
- **Continuation — October 14 12:40 UTC.** Forty legacy result files
  (468,187,088 bytes) are acquired with matching sender/receiver hashes; no
  checkpoints, datasets or large Jacobians were transferred. The numerical
  reader passed independent engineering review. Publish its frozen source,
  then run one local CPU audit capped at 600 seconds with offline W&B.
  [Array protocol](auto/legacy-floquet-array-audit-v1.md),
  [implementation review](auto/legacy-floquet-array-audit-implementation-review-2026-10-14.md),
  [reconstruction specification](auto/tgrab-checkpoint-reconstruction-v1.md).
  Numerical findings and graph training remain pending.
- **Hosting ended — terminal reconciliation at 12:36 UTC.** Allocation
  `11228391` completed `0:0` at 08:53:38 UTC. The guard stopped Qwen at
  08:52:40 after nine replies; the observer finished at its 08:50 deadline
  after 144 checks, maximum gap 300.000441 seconds, zero recorded issues.
  No replacement worker or native timer is running. Historical morning
  liveness checks below retain their original observation times.
- **Original graph reproduction — 2026-10-14, priority update.** The current
  vector studies are not graph replications; no negative T-GRAB replication has
  occurred. The owner-directed Mila TSA search recovered archived analysis
  source and eigenpair/envelope/Jacobian candidates (20 Jacobians, 52.43 GB).
  Headers show eigenvectors `(25600,6)` and envelopes `(20,25600,6)`; no payload
  or checkpoint was loaded. No original model checkpoint was found in the
  enumerated owner TSA scopes. Source/array/checkpoint linkage remains open.
  Three native agents audited claims, source parity and archive recovery;
  seven engineering checks reproduce sampler/evaluation/complex-matvec issues.
  The archived source also repeats envelope time entries, hardcodes 20 steps
  and reloads eigenpairs; its relationship to the older plots is unverified.
  Prioritize `Pdet(2,32)` behavioral parity separately from AB/AABB dynamics,
  before additional vector expansion. [Audit and next steps](auto/tgrab-reproduction-audit-2026-10-14.md).
  No graph model execution, numerical payload audit or new submission occurred.
- **Completed-task inventory — 2026-10-14.** Seven milestones are complete:
  protocol review, forward-engine implementation, the actual 736-call run,
  independent numerical reduction, conditional theory review, W&B reporting
  setup, and morning operational reconciliation. Eleven hash-bound evidence
  files and all 32 condition statuses were checked. See the
  [completed tasks in the notebook](october/october14/2026-10-14-progress-report.md)
  and `completed_tasks` in [structured status](auto/research-status-2026-10-14.json).
  No new experiment was run during this inventory; task competence and the main
  harmful-collapse question remain open.
- **Morning status check — 2026-10-14, historical (09:42–09:44 Yerevan).**
  A fresh read-only check at 05:42:34 UTC verifies existing Mila job `11228391`
  is RUNNING and all five owned worker/supervisor/guard/observer process
  identities, source hashes and job cgroups match. Qwen has nine verified
  replies, 45,061 reported tokens, zero provider errors and no request in flight.
  Its heartbeat is 47.5 seconds old; it is waiting for evidence. Counts have
  not advanced since the last reply at October 13 21:43 UTC. Monitoring continued
  overnight: 106 observations through 05:37:40 UTC, maximum interval
  300.000441 seconds, zero recorded metadata issues. This establishes continued
  monitoring and worker health, not additional overnight model research.
  Native cluv 0.1.5 at 05:44 UTC verifies Mila/Tamia project-name queue queries
  with no matching experiment jobs. Eight other Alliance targets have no
  reusable masters; their queues remain unknown. No new login or submission.
  The last completed numerical work remains the reviewed 32-condition,
  736-call calibration and independent reduction below. No new fit, numerical
  experiment or model request occurred in this status pass. Observer deadline
  is 12:50 Yerevan today; allocation expiry is 12:55:56, without automatic renewal.
  [Structured status](auto/research-status-2026-10-14.json) and
  [dated report](october/october14/2026-10-14-progress-report.md) retain receipts.
- **Completed forward calibration — 2026-10-14, latest completed run.** Published producer
  `edc47ee` completed 32 conditions and 736 native forward calls in 49.66 seconds,
  with zero fitting, targets or derivatives. Independent NumPy reduction from
  published `70b3a05` completed in 41.06 seconds and verified 15,024 comparisons.
  Independent closeout accepts numerical integrity. Fixed-precision B0/C, sham,
  state/cache identity, duplication and repeat checks pass. Four FP32/FP64
  comparisons of signed hidden-intervention logits fail at the known
  GRU/categorical checkpoint: one original history, reused across duplicate
  rows, signs and exact repetitions. Strict all-comparison acceptance is false.
  These new inputs/runtime controls do not repair the old v5 stop or confer
  original-task competence. [Results and limits](auto/cached-forward-v2-results-independent-review-2026-10-14.md).
  Both numerical runs have verified contemporaneous offline W&B records;
  cloud delivery remains pending. The dashboard's verified online smoke is
  [here](https://wandb.ai/mehrab-hamidi-milaquebec5431/representation-collapse/runs/rc-reporting-smoke-20261013t2058z).
  A fresh Qwen task produced reply 9 at 21:43 UTC: nine attempts/replies, 45,061
  tokens, zero recorded provider errors. It missed the mathematical task and
  was rejected; no proposed training or precision change was executed. Native
  agents carried out implementation and independent checks. A fresh reviewer
  completed the written closeout from saved evidence after the original
  reviewer's connection failures; no numerical work was repeated. The
  [conditional bridge derivation](auto/approximate-bridge-audit-2026-10-14.md)
  passed independent static algebra review; it establishes no actual B0/C
  certificate. The last complete process check at 21:29 UTC verified the worker,
  supervisor, guard and five-minute observer; the worker was checked again at
  21:44 UTC. The observer ends October 14 at 08:50 UTC, guard at 08:52:40,
  allocation at 08:55:56 UTC (12:55 Yerevan), with no automatic renewal.
  No new Slurm/GPU submission. Counts remain 32 completed fits plus one
  incomplete fit, 114,200 updates, zero original-eight-query eligible checkpoints.
- **Renewed continuation — 2026-10-14, earlier snapshot.** At October13 20:46 UTC
  (October14 00:46 Yerevan), the original Qwen leaf, supervisor and expiry guard
  are verified alive in existing Mila allocation11228391. Seven replies and
  34,894 reported tokens are unchanged since16:12UTC; the worker is waiting
  for evidence, with zero errors or requests in flight. Liveness does not imply
  additional proposals or experiments. The previous observer ended normally
  at20:00UTC after119 observations; a separately owned observer-only continuation
  launched at20:52UTC through October14 08:50UTC. Two observations300.000158s
  apart verify advancement with no issues. At21:00UTC, a new mathematical task
  produced Qwen reply8: eight attempts/replies, zero errors,39,932reportedtokens.
  Its counterexample was correct; insufficient conditions for finite intervention
  transfer caused rejection of the whole answer. The [algebraic audit](auto/causal-bridge-audit-2026-10-14.md)
  records a counterexample to that recommendation and sufficient scoped map conditions.
  No worker restart, new allocation, GPU job or fitting occurred. The guard
  remains armed for08:52:40UTC; allocation expiry is08:55:56UTC (12:55Yerevan).
  The native callback due16:50UTC missed its deadline and was skipped upon
  resumption. No uninterrupted native coordinator activity is claimed.
  The last accepted numerical result remains372conditions/120,000native calls;
  the cumulative acquisition count remains32complete fits plus1incomplete,
  114,200updates. The [old measurement draft review](auto/post-calibration-measurement-contract-independent-review-2026-10-13.md)
  rejects baseline-output agreement as a causal bridge and requires distinct
  training-map/AD budgets. A new [736-call forward-only contract](auto/post-calibration-measurement-contract-v2-2026-10-14.md)
  and exact disabled config passed [static review](auto/post-calibration-contract-v2-independent-review-2026-10-14.md).
  The additive implementation and one184-call tiny fixture campaign passed
  [independent engineering review](auto/cached-forward-v2-implementation-independent-review-2026-10-14.md).
  A single manifest-bookkeeping correction is independently proven equivalent
  without rerunning native fixtures. Published source`edc47ee` and a hash-bound root admission enabled one
  local CPU run:32conditions/736calls,49.66seconds, zero fitting or targets.
  All72raw file hashes and32offline W&B metric records are verified. Producer
  reports four failed comparisons; independent raw reduction remains pending.
  No new scientific claim is accepted. The owner-authorized
  [W&B project](https://wandb.ai/mehrab-hamidi-milaquebec5431/representation-collapse)
  was created at21:05UTC; a tiny online SDK0.21.0 transport smoke has verified
  history/summary readback. The pinned SDK0.30.0 reporting helper separately
  passes13lifecycle tests and an offline transaction/artifact inspection; its
  remote delivery remains unverified. The initial lazy-object lookup error and
  zero-init failed transport attempt are preserved. [Reporting contract](auto/wandb-reporting.md),
  [structured status](auto/research-status-2026-10-14.json).
  A fresh21:13UTC native-cluv check verifies Mila and Tamia with empty project
  queue matches. The other eight Alliance targets have no reusable masters from
  this controller; their queues are unverified. No fresh Duo or submission.
- **Daytime research — 2026-10-13, historical snapshot.** The old Qwen cohort produced
  16 valid proposals in 19 attempts before three provider errors stopped it at
  00:36 UTC. The 129-snapshot overnight observer and original-host PID absence
  are verified; this was not uninterrupted all-night inference.
  The new **continuous Qwen client is actually running** in existing Mila job
  **11228391**, on cn-f003: leaf PID **3244371**, supervisor **3244355**, separate
  project tmux session and journal. At 16:15 UTC, seven exact-model/schema
  proposals are verified: seven attempts, zero errors, 34,894 reported tokens.
  The latest call completed16:12UTC on a new numerical-stop event; its scientific
  recommendation was rejected after explicit review.
  Multiple recommendations are rejected for missed topics, weakened gates or
  unsupported information-loss inference; native OpenAI review and a local
  prompt repair address these failures. The repair is independently reviewed
  locally and staged through cluv. Automatic approval review rejected the
  specific old-client shutdown; graceful replacement awaits owner confirmation.
  The current client, observer and guard remain active. Live answer-quality
  improvement is unverified. Total-call/runtime caps are null; backoff, signal/STOP controls remain.
  Five task events are consumed, with no scientific answer automatically accepted. A project-only hosting guard will request graceful stop at08:52:40UTC on
  October14. The existing allocation ends
  **October 14 08:55:56 UTC** (12:55 Yerevan); no replacement allocation is automatic.
  Native OpenAI coordinator/subagents prepare and review independent work.
  Both native follow-ups were delivered, but the second scheduled for12:42UTC
  arrived only at16:01UTC (3h19m late). Independent implementation/review
  completed by12:53UTC; continuous coordinator work in the intervening gap
  is unverified. The coordinator resumed integration at16:01UTC. A single best-effort callback
  is now armed for16:50UTC; its timing and future execution are not guaranteed.
  The project's five-minute observer is running outside Qwen evidence roots;
  a fresh16:05UTC certificate verifies72 observations and live owned processes.
  The48 checks since12:03UTC are consecutive, maximum gap300.000416seconds,
  with no metadata issues. Qwen produced no additional proposals during that
  interval and was waiting for evidence. The observer ends20:00UTC.
  Today **32 local CPU fits completed, plus one incomplete fit;
  114,200 optimizer updates** were executed:
  twelve v2 discovery fits fail before body training, and eight fixed 5,000-update
  v3 fits fail full-coordinate competence despite five passing coordinate scores.
  Independent raw/source/native reduction accepts this scoped acquisition audit.
  [V3 results and reproducible plot](auto/prefix-acquisition-v3-results-2026-10-13.md)
  retain every coordinate; neither periodicity harm nor architectural impossibility
  follows. The new fixed-address-v4 diagnostic completed all twelve fixed endpoints: all
  eight q7/q15 singleton gates pass and four q0 gates fail. Independent reduction
  accepts this scoped seed0 result, with q0/q7 paired native-error comparisons.
  Those separate successful weights do not confer original eight-coordinate
  eligibility or isolate lag/capacity causes.
  [V4 results](auto/address-acquisition-v4-results-2026-10-13.md).
  The [v5 proposal](auto/query-averaged-acquisition-proposal-v5.md) retains the
  original task and averages eight QUERY branches; sixteen tiny gradient fixtures
  pass, with no optimizer updates or learned result. Additive implementation and
  actual-size calibration pass independent review; clean source/freeze precede
  four local fits. Actual production stopped in its first fit at4,000updates
  when the frozen QUERY-path consistency check failed. Finite weights and seven
  prior validations are independently verified; all fixed5,000endpoints remain
  unknown and three cells unstarted. No retry, resume or tolerance change occurred.
  [V5 stopped campaign](auto/query-averaged-acquisition-v5-results-2026-10-13.md).
  A separate numerical-chronology proposal passed independent design review;
  implementation and engineering review are complete, published as9cff2d3.
  The isolated local diagnostic completed372conditions/120,000native core
  calls in272.36seconds, exit0, with strict numerical interoperability failing.
  Independent reduction verifies all saved comparisons and accepts the numerical
  audit: all38failed comparisons concern the known trained GRU checkpoint
  in FP32; initialization and null conditions pass.
  [Calibration result and limits](auto/native-chronology-calibration-v1-independent-results-review-2026-10-13.md).
  A separate measurement-contract proposal and scientific review are active;
  further fitting is unadmitted. An
  all-eight R0 pass would still leave GateB R1/R2 untested.
  Reviewed seed-statistics and complete-state matrix-free utilities were published
  in **999d88d**. The current reviewed integration passes **279 source tests**,
  zero skips. Inference assumptions,
  smallest effects and confirmation freeze remain unresolved.
  Native **cluv 0.1.5 verifies Mila plus all nine configured Alliance clusters**
  at 09:25 UTC; matching project queues are empty and all scheduler queries succeed.
  No new Slurm allocation or GPU telemetry. The unchanged GPU canary is unapproved
  and should be reconsidered after the acquisition failures.
  The venue-neutral manuscript now includes the verified zero-body audit and
  endpoint figures: seven-page main text, eighteen pages including appendices.
  [Day plan](auto/day-research-2026-10-13.md),
  [structured status](auto/research-status-2026-10-13.json).
- **Overnight continuation — 2026-10-12, historical.** Fresh heartbeat at 17:00 UTC
  and nine completed model-identity-verified proposal calls confirm continued Qwen
  activity: nine valid reviews, zero recorded errors, 49,965 reported tokens. The
  same original worker and 24-hour/24-attempt ceiling remain; no restart occurred.
  A 600-character project update was consumed by call 10 at 17:24 UTC. The 17:26
  check verifies ten valid reviews, zero errors and 55,482 reported tokens.
  Native cluv 0.1.5 now verifies all nine Alliance targets from Tamia1. Their
  project queues are empty; Mila was independently checked at 17:11 UTC.
  Observer PID **1299159** on Tamia1 polls public metadata every five minutes
  until **2026-10-13 08:00 Asia/Yerevan (04:00 UTC)**. Its journals are outside
  Qwen's evidence roots, so polling cannot cause provider calls. It does not
  submit, restart, cancel or accept scientific claims. A temporary local idle-
  sleep assertion supports the existing transport until the same deadline.
  All **150 source tests pass**, with no skips. Independent review resolved two
  observer metadata bugs. The [GPU guide](auto/gpu-efficiency-adoption-2026-10-12.md)
  is permanently adopted; all six local/Mila/project package files match.
  The [statistics specification](auto/statistics-freeze-proposal-v2.md) is a
  reviewed proposal, with mean-inference/SEI/measurement freeze blockers explicit.
  Fir/Nibi expose suitable account/typed-device availability, but new source and
  offline environments are unprepared. The four-fit/two-GPU-hour canary remains
  unapproved and unlaunched. No new GPU work or harmful-collapse claim.
  [Overnight record](auto/overnight-research-2026-10-12.md).
- **Earlier local integration and route check — 2026-10-12.**
  Tamia's reusable SSH master is absent; current Qwen liveness is **unverified**. The last
  successful check was at 10:07 UTC. Native cluv 0.1.5 confirms no reusable Tamia
  master at 12:48 UTC; a passive SSH check still finds it absent at 13:49 UTC.
  One Duo attempt closed before login completed. A fresh
  authentication response is pending. No duplicate worker, renewed provider
  call by the coordinator or new compute submission was made.
  Local research advanced substantially: a bounded staged trainer, complete
  native h/c phase capture and validation-only checkpoint selection are now
  independently reviewed and published in **af20702**; the submission guard and
  seven regression tests are committed in **9e22c6c**. All **128 tests pass**,
  with no skips. Validation batches are pinned before optimization; repaired
  selection rejects changed manifests and uses uninterrupted QUERY scores.
  The [prospective training protocol](auto/competence-training-v2.md) and
  [four-fit canary](../configs/experiments/competence-canary-v2.json) prepare
  seed 0/LR 0.001 discovery on Fir/Nibi: one GPU/four CPUs/16 GiB/30 minutes per fit,
  at most two simultaneous jobs and two requested GPU-hours total. Target cache/account/typed GPU
  verification and exact owner allocation approval are still required;
  no full-budget training, final LR-grid choice or scientific claim is accepted.
  The [graph source audit](auto/tgrab-upstream-pin-2026-10-12.md) now pins T-GRAB
  code, dataset metadata and GC-LSTM dependency, verifies 23 small source files,
  and resolves upstream AB/AABB notation. Legacy checkpoint/data payload hashes
  remain missing; graph reproduction has not run. The lease-recovery planner
  preserves the existing 24-hour/24-attempt Qwen ceiling across retries.
  [Structured current status](auto/research-status-2026-10-12.json).
- **Earlier restored-access continuation — 2026-10-12, verified then.** Native cluv 0.1.5
  verified all nine configured Alliance targets from Tamia3; Mila was verified
  separately. All ten targets now have an isolated clean source snapshot at
  `ec1c489`; original checkouts were preserved. Tamia CPU job **612745**
  completed `0:0` in 14 seconds, independently reducing the four periodicity
  cells: 64 tangent windows and 1,536 step Jacobians, with no consistency
  errors. A separate review passed 31,264 consistency assertions. The
  [numerical report and reproducible plot](auto/periodicity-independent-results.md)
  show attenuation in both arms and precision-sensitive, nonzero cue
  differences. All four original competence gates failed: **no harmful-collapse
  or certified Floquet claim is accepted**. Manual-query and probe-floor checks
  remain unavailable in this artifact schema.
  The owner explicitly approved a renewed Tamia Qwen run, bounded to 24 hours
  and 24 attempts. PID **524659** and an actual model-identity-verified proposal
  establish participation; the current timestamp, call count and usage are in
  [structured status](auto/research-status-2026-10-12.json). It watches project
  evidence and writes proposals; coordinator review governs scientific use.
  Its uptime depends on the local proxy and encrypted SSH route. No new GPU
  training or racing. Additive coordinate-wise competence assays are committed
  and deployed; a final finite-score overflow repair raises the full suite to **82 tests**,
  without skips. Training
  integration, checkpoint selection and the statistics freeze remain next.
- **Earlier automatic-research check — 2026-10-12 07:52–07:54 UTC, superseded.**
  Automatic research **has run, but is currently stopped**. The Mila Qwen
  worker ran for about 15.2 hours and finished at its 22-call cap on October 11
  at 23:28:39 UTC. Fresh heartbeat/status, PID and process checks agree:
  22 schema-valid proposals, 125,730 reported tokens, exit 0, no live worker.
  Mila's current queue has no matching project jobs. The four training and
  four periodicity jobs retain their independently verified historical
  `COMPLETED/0:0` outcomes; a fresh Alliance scheduler query was unavailable.
  Native cluv 0.1.5 reports no reusable Narval/Rorqual connections from Mila;
  the official Tamia alias requires interactive authentication. No Duo,
  model call, restart or job submission occurred during this check.
  GitHub contains manuscript revision `c7c5f44` and the owner's subsequent
  Overleaf merge `4c0b56e`, now preserved locally. Mila source remains
  `8fce9b0`, so those revisions have not reached that checkout. Both GPU
  batches and the initial worker call allowance are consumed. The next
  scientific work is independent periodicity reduction and competence recovery;
  the completed pilots do not establish harmful collapse.
  [Structured current status](auto/research-status-2026-10-12.json).
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

Next actions: execute and independently reduce the newly frozen native-chronology
diagnostic, preserving every numerical failure; then decide the measurement
contract before preparing any new acquisition fit. Check the new Qwen event
against its assigned scientific question and preserve rejected answers. The
reviewed prompt replacement remains staged pending the specific restart answer.
The native coordinator's delayed callback does not establish continuous work;
cluster monitoring and substantive research are reported separately. Complete
the confirmation statistics freeze and outstanding intervention/sham controls,
then recover the remaining graph data/checkpoint provenance. No broad sweep or
additional GPU training follows automatically.

### Figure 10 reproduction — 2026-10-16

- Runtime identity capture job `11280346` completed and was independently
  checked; it is environment provenance only.
- Node2a job `11280406` failed before model/data work due to a supervisor versus
  bootstrap output-path collision. Job `11280432` failed before model/data
  work because the supervisor passed the project checkout instead of the
  separate pinned T-GRAB source snapshot. Both failed run directories and
  terminal receipts are retained; neither run ID may be reused.
- The upstream-source-root correction was independently reviewed. It pins the
  scratch source path, revision label, and all three wrapper byte hashes, and
  verifies them before and after the fixture. The revision label is provenance
  metadata; wrapper hashes enforce executable identity. It does not verify a
  Git HEAD inside the ignored snapshot.
- Validation: preparation 10/10, supervisor 8/8, bootstrap 15/15 (33 focused
  tests total); Python compilation, shell syntax and diff checks pass. No
  node2a result or representation-collapse finding exists yet.
- The independent fit-readiness audit found no production Figure 10 fit/replay/
  sealed-test/plot pipeline. Do not start a fit or open test data until that
  pipeline is integrated, audited and frozen. See the dated report and
  `doc/auto/tgrab-node2a-job-preparation-2026-10-16.md` for the detailed record.
- **T-GRAB Figure 10 node2a parity — passed as engineering evidence, 2026-10-16.**
  Mila job `11280629` completed in 14 seconds with exit 0, 1 CPU/2 GiB/zero
  GPUs, no restart/requeue. The synthetic fixture verifies forward/loss/gradient
  and mode/dropout parity, both model components and complete hidden/cell state
  round-trip, and crash/reopen transaction handling. It made zero optimizer
  updates and opened no primary, validation, or test payload. Local/remote hashes
  for all six artifacts and all plan/source/runtime bindings match. The receipt
  explicitly sets `scientific_acceptance=false`; this does not reproduce Figure
  10 or support a representation-collapse claim. Run archive:
  `runs/figure10-node2a/node2a-20261016T184800Z-40942e57/` (ignored). The three
  failed infrastructure submissions (11280406, 11280432, 11280586) are retained
  with causes and were not interpreted as model evidence. Next gate remains the
  reviewed T-GRAB fit/replay/held-out metric/plot workflow; the readiness audit
  says that production workflow is not yet integrated.
- The node2a receipt has now received an independent artifact audit: plan/source/
  environment/job bindings, complete transaction manifest and payload byte
  hashes, closed shape/order catalog, full hidden/cell and both-component state
  round-trip, exact parity flags, zero updates, and unopened split flags all
  reconcile. Audit scope is engineering only; the bootstrap itself emits the
  flags, so this is not an independent scientific replication.
- A draft Figure 10 metric reducer was corrected after independent code audit:
  phase-transition flags must be booleans and must equal actual adjacent-graph
  inequality; transition summaries use observed `change_point`. Focused tests
  pass 7/7. Exact changed-file hashes are recorded in the October 16 report.
- Next local implementation gate: versioned object-free selected-inference
  checkpoint codec/transaction and trusted reload, then selected-checkpoint
  replay/node2b before opening test. No fit has run.
