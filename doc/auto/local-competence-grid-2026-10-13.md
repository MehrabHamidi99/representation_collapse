# Frozen competence discovery grid — local CPU, 2026-10-13

The complete v2 seed-0 discovery grid has no eligible checkpoint. All twelve
declared model/task/learning-rate fits were executed and reconciled: two failed
the 100-update fixed-batch gate, and ten passed immediate acquisition but failed
full-prefix R0 at update 1,000. No fit entered the R1/R2 curriculum stages or
experienced a periodic distractor body during training. This is a bounded
acquisition failure under the declared schedule, not periodicity-induced harm.

## Scope and provenance

This completes the local LR-grid acquisition screen of
[competence-recovery-v2](competence-training-v2.md), with frozen learning rates
0.0005, 0.001 and 0.003; GRU/LSTM; categorical/continuous vector memory; and
training seed 0. The existing [four-fit report](local-competence-feasibility-2026-10-13.md)
and its artifacts remain intact. The remaining eight fits ran sequentially
from 08:21:00 to 08:21:53 UTC, in 53.262 seconds. The two loops together used
87.178 seconds of process wall time and 10,200 optimizer updates. Each loop had
its own 20-minute ceiling; neither timed out. All processes returned exit 0 with
their frozen scientific stopping outcome retained.

Every fit used clean archived source
`1add2b6aba948714673f45f1b7854eff6212833d`, the existing project Python
environment and the same unchanged configuration and lockfile. Hidden 128,
encoder 64, batch 32, AdamW/weight decay 0.01, clipping 1, validation every
100 updates, the current-checkpoint boundary gates and all manifest rules were
preserved. Device was CPU with one thread, Torch 2.6.0 and FP32 parameters;
deterministic algorithms were enabled, TF32 disabled and matmul precision
`highest`. No engineering override, extended stage, rescue, replacement fit,
held-out evaluation, GPU job, cluster submission or provider call was introduced.

Configuration SHA-256:
`0ee6582ebb70116660ff67ed57327daf15abf270368ba77c0bbe70a23f4a8368`.
Lockfile SHA-256:
`588147aa66724c075fd44754db3ddf757b249eefcaf94c9154ddc49578c016dc`.
Ignored grid outputs: `runs/day-competence-grid-20261013/`; original LR-0.001
outputs: `runs/day-competence-20261013/`. Independent local Git identity belongs
to the archived snapshot, so concurrent parent-checkout edits cannot change the
recorded source revision.

## All declared outcomes

The following lists retain every learning-rate outcome, in the frozen order
0.0005 / 0.001 / 0.003. Coordinate ranges are descriptive summaries of the
eight individually checked R0 scores; every R0 coordinate failed in both arms.
Full coordinate values and raw logits remain in the aggregate receipt and
hashed per-fit artifacts.

- GRU categorical: fixed-batch failure at 100 / R0 failure at 1,000 / R0 failure
  at 1,000. R0 accuracy ranges were 0.0625–0.3359375 at LR 0.001 and
  0.1328125–0.2421875 at LR 0.003. At LR 0.0005, fixed-batch accuracies for
  coordinates 0–7 were 0.75, 0.75, 0.75, 0.5, 1.0, 0.75, 1.0, 1.0; the
  declared rule therefore stopped the fit before stage-1 completion.
- LSTM categorical: fixed-batch failure at 100 / R0 failure at 1,000 / R0 failure
  at 1,000. R0 accuracy ranges were 0.0234375–0.1640625 at LR 0.001 and
  0.046875–0.1796875 at LR 0.003. At LR 0.0005, fixed-batch accuracies were
  0.75, 0.75, 0.75, 0.75, 1.0, 0.75, 1.0, 1.0; no extension was performed.
- GRU continuous: R0 failure at 1,000 for all three rates. Coordinate
  MSE/(1/3) ranges were 0.964102–1.123059 / 0.961714–1.122007 /
  0.962310–1.125711.
- LSTM continuous: R0 failure at 1,000 for all three rates. Coordinate
  MSE/(1/3) ranges were 0.965359–1.124165 / 0.963648–1.121685 /
  0.961354–1.125078.

The categorical gate requires accuracy at least 0.95 on each coordinate; the
continuous gate requires MSE divided by the fixed population variance 1/3 at
most 0.05. The ten R0-failing fits first passed every coordinate's fixed-batch
diagnostic and fresh IID immediate validation at stage-1 update 500. They each
completed exactly 500 immediate-task updates and 500 R0 updates. The two early
categorical failures retained their 100-update validation arrays, checkpoint and
diagnostic without being treated as missing or successful fits.

R0 has sixteen ordered WRITE updates and the native terminal QUERY, with no
intervening body. Its two arm inputs are identical by construction and their
scores agree. They represent 1,024 paired episode units per evaluation, with
128 per coordinate; they do not provide independent arm replications. All
reported trained-seed units are seed 0. Learning-rate comparisons share that
seed and validation manifests. No across-seed uncertainty, causal mechanism,
cross-modality transfer or universal architecture claim is supported.

## Verification and selection

All seven validation manifests were pinned before each fit's first optimization
update. The same task-specific sampling seeds and batch identities were retained
across the grid. Failed and unreached stages remain explicit. No test score,
geometric statistic, arm contrast, desired effect or historical best checkpoint
entered eligibility or selection.

`runs/day-competence-grid-20261013/verify_grid.py` reconciled all twelve unique
model/task/LR outcomes, source/config/precision identities, contiguous optimizer
step histories, stage exposure counters, all stopping gates and checkpoint
selection. It verified 384 output-file hashes and recomputed 1,216 coordinate
metrics directly from saved native logits/targets across every executed
validation evaluation. The producer recorded 2,432 complete-state chronology
chunks and 19,456 passing adapter checks. GRU states contain 128 hidden
coordinates; LSTM states contain 128 hidden plus 128 cell coordinates, in that
order, at post-WRITE, pre-QUERY and post-QUERY phases.

A second implementation independently loaded the actual checkpoint weights and
ran the native recurrent core/head, without using producer capture, renderer,
competence-evaluation or metric helpers. Across both runs it verified 32
diagnostic/boundary checkpoints, 319,488 logit entries and 24,772,608 complete-
state entries, plus 43,008 validation episode targets against the addressed
WRITE payload and terminal QUERY. The remaining-eight replay additionally
verified all 64 fixed-batch coordinate checks. Native logits and states matched
the saved arrays exactly on this CPU. This is not a repeat-training determinism
test, a GPU parity test or proof of retained information below numerical
resolution. Coordinator/reviewer scientific acceptance remains separate.

The additional-eight native/oracle receipt is
`runs/day-competence-grid-20261013/native-oracle-replay-receipt.json`, SHA-256
`6446f57a7065128ebbd482fada5353394b46ff6b2b155314e250d8c109de9a4a`.
The all-twelve aggregate receipt is
`runs/day-competence-grid-20261013/verification-receipt.json`, SHA-256
`be50cd921497f78cebbaf978c051e5ca042cf69730df9a39890e0041d125a986`.
It binds the original and additional native/oracle receipts, every failed
outcome, per-fit artifacts, coordinate values, manifest IDs and counters.

The discovery grid is complete as a record of bounded attempts; competence
recovery remains unresolved. There is no stage-5 candidate and therefore no
eligible final LR choice or checkpoint. Choosing an LR by the least-bad R0
score would change the frozen selection contract and is not done.

## Next decision

Independent review should accept or reject the artifact reductions and check
the training task/input/target chronology. The input-target oracle and native
replay rule out several simple label/capture errors, while the cause of poor
full-prefix acquisition remains open. A new curriculum or optimization-budget
proposal must be prospective, versioned separately, and preserve these v2
failures. Useful next local controls could distinguish addressing, overwriting,
credit assignment and insufficient training horizon; they are proposals,
not completed mechanisms.

The unchanged four-fit LR-0.001 GPU canary would repeat the already observed
failed discovery slice. It remains unlaunched and unapproved; these CPU results
support reconsidering that exact allocation before spending GPU time. GPU
throughput, per-device utilization and CPU/GPU numerical parity are unknown.
No harmful-collapse, periodicity effect, equivalence or certified-Floquet claim
is accepted from this acquisition screen.
