# Competence-recovery training integration v2

Prospective discovery protocol, 2026-10-12. The
[configuration](../../../../../configs/experiments/competence-recovery-v2.json) versions
the [recovery plan](competence-recovery-plan.md) without altering frozen v1
results. This document grants no compute or model-call budget.

The native GRU/LSTM architecture, 38-channel task, hidden128/encoder64,
AdamW/weight-decay0.01, clipping1 and batch32 stay fixed. The prospective
learning-rate grid is [0.0005,0.001,0.003], seed0, four model/task cells, at
most twelve fits. Selection uses native task loss and short-horizon competence;
it cannot use geometry, arm contrasts, DID, held-out tests or a desired failure.

## Fixed schedule and gates

Stage 1 has 100 updates on a fixed immediate-WRITE/QUERY batch, four examples
per required coordinate, then 400 fresh IID immediate-task updates. At 100,
every coordinate must reach accuracy>=0.95 or MSE/(1/3)<=0.05 on that fixed
batch. This is an overfit diagnostic with four examples, not the 128-episode
validation competence gate. A fixed-batch failure stops the fit at 100; no extra
updates. At 500 the current checkpoint must pass fresh IID validation with 128
independent episodes per required coordinate. Stages 2–4 each have500 updates:
full-prefixR0; alternatingR0/R1; cyclingR0/R1/R2. Stage 5 has1500 updates cycling
R1/R2/R4/R8/R16. Gates use the checkpoint at the declared boundary, rather
than an earlier best score. No stage extension or threshold adjustment.

The500-update stage4 cannot have exactly equal exposure across three horizons:
its fixed order yields 167, 167 and 166 updates. This one-update imbalance is reported
explicitly. Paired forcing arms remain exactly balanced each paired update.
Stage3 has 250 updates per horizon; stage 5 has 300 per horizon.

All seven validation batches are generated and persisted before the first
optimization update. The six selection-horizon hashes and sampling seeds are
caller-pinned and checked again at evaluation and selection; replacing a batch
cannot improve checkpoint eligibility.

Fixed validation manifests are independent of the fixed training batch and
IID training draws. Every required coordinate is retained in the gate. No
aggregate can hide a failed coordinate. Loss uses categorical cross-entropy
or continuous raw MSE; the continuous competence metric uses fixed population
variance1/3, distinct from raw loss. Test scores never enter eligibility or
checkpoint/LR selection. Evaluate every100updates and preserve every stage
failure, timeout and numerical failure.

## Native chronology and candidate selection

Capture complete state after WRITE, before QUERY and after QUERY, with LSTM
hidden and cell coordinates. Native uninterrupted logits define behavior;
segmented/forked outputs only cross-check chronology at declared numerical
tolerances. A tolerance pass does not establish exact state equivalence or
rescue information below arithmetic resolution. Record input/weight/source
hashes, state components, actual flags and the maximum discrepancies.

Only stage 5 native validation checkpoints passing every R0/R1/R2 coordinate
in both arms are eligible. The ranking averages task loss equally over arms
and R1/R2/R4/R8/R16. Exact ties keep the earlier global step, then the frozen
LR grid order. Stage-best snapshots are diagnostics and cannot substitute for
current-boundary gates. A four-fit canary at LR 0.001 is a partial discovery
slice; it cannot establish the final grid-selected LR. All three LR outcomes
per cell must be reconciled before the complete discovery choice.

Pure FP32 training disables CUDA and cuDNN TF32 and records actual backend,
cuDNN, deterministic-algorithm and matmul-precision values. Per-module gradient
and parameter-change norms and input gradients are diagnostic measurements,
not proof of memory. Saved outputs support independent verification with
unique attempt directories, atomic status/config/provenance/history, raw
validation arrays, checkpoints and hash manifests.

## Remaining scientific work

Local tiny runs verify engineering only. These modules do not complete Gate B:
fixed-driver probe clouds/floor robustness, interventions and shams, precision
and noise assays, replicated competent seeds, effect-size/uncertainty freeze,
graph reproduction and native-modality transfer remain separate prerequisites.
No harmful-collapse, certified-Floquet or equivalence claim follows from
successful code execution. Current evidence is in the [tracker](../report/TRACKER.md).

## Cluster canary and continuation limits

The [canary allocation](../../../../../configs/experiments/competence-canary-v2.json)
prepares four seed 0/LR 0.001 fits on Fir/Nibi, at most one GPU/four CPUs/16 GiB/
30 minutes each and two simultaneous jobs. Its total requested ceiling is two
GPU-hours. Source commitment, fresh typed-GPU/account/partition verification,
offline Torch 2.6 cache readiness and exact source/config/resource-bound owner
approval must all precede submission. Target fields are currently explicitly
unverified; preparation is not approval or execution. The
[submission CLI](../../../../../code/ops/submit_competence_recovery.py) defaults to a
zero-remote-action preview and rejects ambiguous prior submissions. Configured
`SBATCH_*`/`SLURM_CLUSTERS` inputs are rejected; inherited inputs are checked in
the same login shell immediately before native `sbatch`. The native cluv 0.1.5
command shape is validated, and the exact guarded command is journaled. This
prevents undocumented array/federation inputs from expanding the canary
allocation before the batch wrapper can inspect it. The rule follows
[Slurm's documented environment precedence](https://slurm.schedmd.com/sbatch.html#SECTION_INPUT-ENVIRONMENT-VARIABLES).

Continuous proposal generation remains within the existing Qwen lease:
24 hours from its original start, 24 total attempts including failures. The
[recovery planner](../../../../../code/ops/qwen_resume_budget.py) requires fresh stopped
process/status evidence and subtracts prior usage and elapsed time before
proposing a recovery. It never launches, submits or grants approval. Unknown
remote liveness cannot authorize a duplicate worker. These bounds do not
provide an unlimited or recurring allowance.
