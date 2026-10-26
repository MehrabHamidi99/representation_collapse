# Initial experimental phase — completed 2026-10-11

**Engineering complete; harmful-collapse evidence inconclusive.** Both
calibration suites pass 10 + 36 checks. All four GPU pilots complete their full
1,500 optimization steps and pass independent artifact verification. Every
selected model fails native short-horizon memory competence. These outcomes
establish the pipeline and an acquisition problem under the frozen protocol.

## Protocol and actual execution

Every job ran clean source `9004a0c114a8b7c5335dca3d5247449fd10b5ffa`,
[initial-phase-v1](../../../../../configs/experiments/initial-phase-v1.json) and
[vector-pilot-v1](../../../../../configs/experiments/vector-pilot-v1.json), with locked
offline Torch 2.6.0+cu124, Python 3.11.13 and NumPy 2.4.6. Native cluv 0.1.5
submitted each cell once. All five jobs finished `COMPLETED`, exit `0:0`:

- Narval calibration **5234514**: 32 seconds; one CPU, 2 GiB, ten-minute limit.
- Narval GRU categorical **5234860**: 243 seconds; A100-SXM4-40GB.
- Narval GRU continuous **5235508**: 228 seconds; A100-SXM4-40GB.
- Rorqual LSTM categorical **22980985**: 52 seconds; H100 MIG `3g.40gb`.
- Rorqual LSTM continuous **22980998**: 32 seconds; H100 MIG `3g.40gb`.

All GPU jobs allocated one device, four CPUs and 16 GiB host memory. Narval
retained its one-hour limit. The existing pending Rorqual job IDs were reduced
from one hour to fifteen minutes and from full H100 to one 40 GiB MIG slice
after native scheduler eligibility checks. Source, seed, training steps and
dependencies stayed fixed. Original requests remain in the submission journal;
amendments have a separate record. No racing, GPU retry or resubmission occurred.

Actual GPU allocation totaled **555 device-seconds (0.15417 device-hours)**,
counting a MIG slice as one device, not a full-GPU equivalent. Actual maximum
concurrency was one, below the cap of two. Both dependency chains were honored.
Elapsed allocation includes environment preparation and cannot establish an
architecture-speed comparison. MIG memory/utilization accounting was unavailable.

All four GPU submissions are consumed. Three bootstrap CPU submissions were
used: Tamia **536757** succeeded, Mila **11190579** failed before calibration
because of inherited uv environment selection, and Narval **5234514** succeeded
with corrected wrappers. The failed attempt supplies no trained-model evidence.

## Measurement and acceptance

The task writes random memory, supplies periodic or matched varying distractors,
then queries memory. Paired arms share cues, queries, horizon, letter/bigram
counts and final motif. Training balances arms and horizons 1, 2, 4, 8 and 16;
checkpoint selection uses validation loss. Held-out behavior uses 256 paired
episodes per reported horizon 1, 2 and 16. Each fixed-driver cloud uses 512
training and 256 test cue histories; two panels retain separate results.
GRU state is all 128 hidden coordinates; LSTM state is all 256 hidden/cell
coordinates. These complete-state capacities are unequal.

Independent verification passes **6/6 artifact directories**: four trained
pilots and two synthetic calibration suites. It checks source/lock/config,
full history, validation selection, checkpoint and array hashes, complete state,
matched stimuli, saved predictions, paired errors, covariance and held-out
real-label/shuffled-label ridge probes. Separate offline reconciliation verifies
UTC Slurm outcomes, accounts, resources, hardware, IDs and actual dependencies.
Engineering completion is true; scientific acceptance remains false.

Tiny variances use zero absolute tolerance and relative tolerance `1e-5`;
eigenvalue absolute tolerance is `1e-12` times recomputed cloud trace. Reproducing
stored arrays does not certify the resolution of float32 dynamics. The existing
source regression suite passes **29/29** tests.

## Results and scientific decision

- Both categorical models select step 300. Accuracy in both arms is
  **5.46875% at R1, 6.25% at R2 and 5.078125% at R16**, against 6.25% chance.
  Periodic-minus-control error contrasts and R16-minus-R2 DID are zero.
- GRU continuous selects step 400. Periodic R1/R2/R16 MSE is
  **0.35293357 / 0.36422871 / 0.33352217**; control differs by `1e-7` or less.
  Raw-recomputed DID is `-9.98762525e-8`.
- LSTM continuous selects step 200. Periodic MSE is
  **0.35247289 / 0.36560507 / 0.33494743**; DID is `-5.91133419e-8`.

Exact saved-target zero-prediction MSE is 0.35329986 / 0.35910623 / 0.33411188.
Native errors are close to this no-memory reference. All models fail even weak
R1/R2 engineering gates: 25% categorical accuracy or continuous MSE at most 0.25
in both arms. Strong research competence therefore also remains unmet.

Very small cue-conditioned variance does not prove harmful collapse. GRU and
continuous LSTM participation ratio increases at R16 while variance shrinks.
All four cells retain R2 held-out probe advantages over shuffled labels despite
poor native readout. R16 advantages are inconsistent. Variations approach
float32 spacing; two categorical LSTM R16 control clouds have 511 distinct
stored rows out of 512, and other clouds have 512. Precision, robust retention
and the source of residual cue information remain unresolved.

The [independent scientific review](initial-phase-scientific-review.md) retains
these counterarguments and exact reductions. One seed, synthetic encodings,
unequal complete-state dimensions and hardware confounding preclude population,
native-modality or architecture-superiority claims. No orbit closure, tangent
spectrum, Floquet multiplier, graph reproduction or causal intervention was
measured. Failed acquisition neither supports nor falsifies harmful periodic
collapse in competent models.

## Evidence and storage

[Committed compact results](../../../../../doc/auto/initial-phase-results.json) preserve reductions,
provenance and hashes. Detailed ignored runtime evidence is:

- `runs/ops/phase1-launches-20261011.json`: immutable original submissions.
- `runs/ops/phase1-execution-amendments-20261011.json`: same-ID resource changes.
- `runs/ops/phase1-status-20261011.json`: native UTC scheduler and provenance.
- `runs/ops/phase1-final-verification-20261011.json`: raw artifact reduction;
  SHA256 `4a71636fa22baba0cc4b20c1b0b8ff0ca339278e159854fe7966b5159e01a4a0`.
- `runs/ops/phase1-scheduler-reconciliation-20261011.json`: engineering decision.
- `runs/ops/phase1-scientific-results-review.json`: separate scientific review.

Verifier SHA256:
`5f092ee5b87c5e407b5271eee97d5a59c66264570d2862df3cc672de0cb8d22b`.
Postprocessing helpers are committed after immutable training source; hashes
identify the versions actually used for acceptance.

Original cluster scratch folders and complete hash-verified copies on Tamia and
Mila retain checkpoints. Durable private archives preserve original run-folder
names below `/project/def-ioannism/mehrab/representation-collapse/evidence/initial-phase-v1-9004a0c/`:
Narval canonical prefix `/lustre06/project/6044649/mehrab/`; Rorqual canonical
prefix `/lustre09/project/6044649/mehrab/`. Archive receipts verify every file
and manifest. No datasets or checkpoints were downloaded into the Mac checkout.

## Next bounded research step

The [competence-recovery protocol](competence-recovery-plan.md) diagnoses WRITE
acquisition, retention and QUERY readout: exact streaming oracle, immediate
WRITE/query, full-prefix zero-body task, then strict native coordinate-wise
R1/R2 competence. Competent models can advance to long-horizon geometry,
precision/robustness, tangent and intervention assays. Preserve all v1 failures.
Further training requires a new frozen protocol and concrete allocation.
The existing bounded actual Qwen worker on Mila remains proposal-only; it
cannot submit jobs or accept claims. Graph-dependent research remains blocked
on reproducible assets and complete-state measurement validation.
