# Initial phase: independent scientific review of all four vector pilots

Date: 2026-10-11. Scope: all four completed full-budget, seed-0 GRU/LSTM ×
categorical/continuous vector pilots. **Scientific decision: inconclusive for
repetition-induced harmful collapse because every selected model fails
short-horizon native competence.** Execution and source-bound artifact
verification succeeded. These are separate decisions.

## Evidence and scope

The [final artifact verification](../../runs/ops/phase1-final-verification-20261011.json),
checked at `2026-10-11T13:33:28.734460+00:00`, independently reduces saved
behavior, cue-conditioned geometry and probe outputs. It passes **6/6 artifact
directories: four trained GPU pilots and two synthetic calibration directories**,
with 11,557 successful checks. Synthetic fixtures are not additional trained
models or modality evidence. Each pilot has a full 1,500-step history; the
reported checkpoint was selected using validation loss. Source commit:
`9004a0c114a8b7c5335dca3d5247449fd10b5ffa`.

Verification JSON SHA-256:
`4a71636fa22baba0cc4b20c1b0b8ff0ca339278e159854fe7966b5159e01a4a0`.
Postprocessing verifier SHA-256:
`5f092ee5b87c5e407b5271eee97d5a59c66264570d2862df3cc672de0cb8d22b`.
Metrics-module SHA-256:
`41a4790b639fb2d37c1169771c08760e7df1ab0a1b7fa75ee2553828c21ebf1e`.
This reviewer inspected check statuses and reductions, then summarized them;
the reviewer did not rerun training or independently reopen remote checkpoints.

The [persisted scheduler snapshot](../../runs/ops/phase1-status-20261011.json),
checked at `2026-10-11T13:39:46.336418+00:00`, separately records every GPU job
COMPLETED with ExitCode `0:0`, using UTC timestamps:

- Narval GRU/categorical `5234860`: 243 seconds, 13:01:18–13:05:21.
- Narval GRU/continuous `5235508`: 228 seconds, 13:08:53–13:12:41.
- Rorqual LSTM/categorical `22980985`: 52 seconds, 13:20:23–13:21:15.
- Rorqual LSTM/continuous `22980998`: 32 seconds, 13:21:16–13:21:48.

The GRUs used one A100 each. The LSTMs used one H100 MIG `3g.40gb` slice
(`nvidia_h100_80gb_hbm3_3g.40gb=1`) each, with the same job IDs amended to a
15-minute limit. This is an explicit allocation deviation from the requested
full H100, not an additional GPU submission. Each job allocated four CPUs and
16 GiB host memory. Hardware, training selection and architecture differ, so
elapsed time is not a controlled architecture-speed comparison.

The [frozen configuration](../../configs/experiments/vector-pilot-v1.json),
[task renderer](../../src/representation_collapse/vector_task.py),
[training/measurement code](../../src/representation_collapse/pilot.py), and
[research plan](periodicity-research-plan.md) define the scientific contracts.
The pilots are synthetic vector memory tasks. They do not constitute graph
reproduction, native vision/audio/text evidence, pretrained-model evidence or
coverage of other recurrent families.

## Native behavior: no model reaches the acquisition gate

Behavior uses 256 matched prefix/query pairs per horizon. Define each arm
contrast as periodic error minus control error, and DID as delta16 minus delta2.
Checkpoint selection uses validation task loss, not these contrasts or geometry.

- **GRU categorical**, selected step 300: both arms have accuracy 5.46875% at
  R1, 6.25% at R2 and 5.078125% at R16. Chance is 6.25%. Delta2, delta16 and
  DID are zero.
- **LSTM categorical**, selected step 300: the same aggregate accuracies and
  zero contrasts occur. Equality of aggregate errors does not establish equal
  predictions or equal internal trajectories between architectures.
- **GRU continuous**, selected step 400: periodic/control MSE is
  0.3529335672/0.3529335672 at R1, 0.3642287132/0.3642285941 at R2 and
  0.3335221724/0.3335221533 at R16. Delta2 is `1.19058328e-7`, delta16 is
  `1.91820755e-8`, and DID is `-9.98762525e-8`.
- **LSTM continuous**, selected step 200: periodic/control MSE is
  0.3524728915/0.3524728915 at R1, 0.3656050692/0.3656049966 at R2 and
  0.3349474319/0.3349474185 at R16. Delta2 is `7.25556980e-8`, delta16 is
  `1.34423561e-8`, and DID is `-5.91133419e-8`.

All four fail even the frozen engineering gates: short-horizon categorical
accuracy at least 0.25 or continuous raw MSE at most 0.25, in both R1/R2 arms.
These weak gates do not replace strict coordinate-wise research competence.
The results establish failed native acquisition under this protocol and budget;
they do not localize the failure to writing, encoding, storage or terminal query.
Small or zero DID at failed acquisition is not an equivalence test and does not
falsify harmful repetition in a competent model. No substantive interpretation
of the tiny continuous contrast is justified here.

The final verifier computes continuous zero-predictor MSE **from saved raw
targets**, not sampler replay: 0.3532998603 at R1, 0.3591062313 at R2 and
0.3341118809 at R16. Native/zero MSE ratios across the two architectures and
arms lie between 0.9976593 and 1.0180973. Thus native errors are close to the
no-memory reference for the exact evaluation samples. The population reference
is 1/3, but its finite-sample variation matters: a lower raw MSE at R16 than R2
cannot by itself demonstrate better memory. This saved-target check supersedes
the earlier source-reconstructed evaluation reference. Historical validation
sampler replay remains an ancillary reconstructed check, not independent raw
validation-target verification.

There is one training seed per model/task cell. Do not pool cells, panels,
cues, coordinates, horizons or timesteps as independent training replicates.
No training-population confidence interval or significance claim follows from
this phase. Successful deterministic CUDA execution and raw-output verification
also do not establish bitwise reproducibility across hardware or repeated
training; no independent training replay was performed.

## Geometry: concentration differs from amplitude and memory

Each cloud fixes its complete driver and nuisance history while varying required
cues. Arm-matched clouds share cue samples. Geometry uses 512 recorded pre-query
states per cloud, with separate held-out test cues for probes. GRU state has
128 hidden coordinates; LSTM complete state concatenates h and c, totaling 256
coordinates. Both use hidden size 128, which does not match complete-state
capacity or parameter count. Report LSTM h and c separately in future assays;
the relative coordinate scales affect covariance PR.

Ranges below span two driver panels and both arms at each horizon:

- **GRU categorical:** R2 total variance `5.45e-15–5.94e-15`, PR `2.45–2.54`;
  R16 variance `1.09e-16–1.38e-16`, PR `29.50–33.58`.
- **GRU continuous:** R2 variance `7.30e-15–7.60e-15`, PR `1.098–1.107`;
  R16 variance `7.86e-17–1.40e-16`, PR `20.74–26.09`.
- **LSTM categorical:** R2 variance `3.09e-16–4.80e-16`, PR `18.23–24.27`;
  R16 variance `1.86e-16–2.76e-16`, PR `11.43–27.26`.
- **LSTM continuous:** R2 variance `5.18e-16–6.28e-16`, PR `6.63–7.03`;
  R16 variance `1.78e-16–2.43e-16`, PR `18.99–22.33`.

The GRUs and continuous LSTM have greater normalized PR at R16 despite smaller
absolute variation. Categorical LSTM ranges overlap and vary across clouds.
There is no universal monotone dimensional-collapse sequence in these sampled
windows. Without WRITE-prefix measurements, these clouds do not establish
how much encoding was first acquired. PR measures linear covariance
concentration; it is not an intrinsic-dimension, task-information or causal
mediation bound. Direct GRU/LSTM PR rankings also confound state coordinates
and their scale conventions.

All cloud variances are nonzero, so `normalized_dimension_defined=true` is
mathematically correct for the stored arrays. Scientific resolution remains a
separate issue. The native recurrence and saved states are float32; subsequent
centering and decomposition are float64. That cast cannot restore information
lost during float32 recurrence.

The post hoc audit reports centered-state RMS divided by stored float32 ULP
RMS as `2.01–2.55` at GRU R2 and `0.255–0.345` at GRU R16. LSTM ratios are
`0.368–0.467` at R2 and `0.250–0.348` at R16. Every cloud has 512 distinct
stored rows except the two categorical LSTM **R16 control** clouds, which each
have 511. An aggregate ratio below one does not mean every feature difference
is unrepresentable, that states are identical, or that residuals are necessarily
roundoff. Rare changes or coordinates with finer spacing can produce many
unique vectors. The two duplicate-row observations alone do not identify which
required targets, if any, share indistinguishable states.

This is stored-resolution evidence, not a measurement of trajectory error,
robust retention or the source of the residuals. No FP64 dynamical replay or
specified-noise assay has been run. Relative numerical ranks are GRU 128 at
R2 and 114–124 at R16, versus LSTM 150–172 at R2 and 130–150 at R16; they do
not establish meaningful signal in every counted mode.

Postprocessing variance comparisons now use absolute tolerance zero and relative
tolerance `1e-5`; covariance eigenvalue absolute tolerance is `1e-12` times
recomputed cloud trace. All six directories pass this scale-sensitive rerun.
This resolves the earlier inappropriate generic absolute tolerance for tiny
variances, while native float32 precision and robustness remain open.

## Conditional probes preserve counterevidence

The real-label R2 probes outperform their corresponding shuffled-label controls
in each observed driver/arm cloud, including both LSTM cells:

- GRU categorical accuracy `20.95–22.75%`, shuffled `6.05–7.13%`.
- LSTM categorical accuracy `14.75–15.53%`, shuffled `6.01–6.79%`.
- GRU continuous MSE `0.23420–0.24086`, shuffled `0.37049–0.38067`.
- LSTM continuous MSE `0.30302–0.31799`, shuffled `0.35749–0.38025`.

These are held-out conditional traces in recorded states. Neither poor native
behavior nor tiny variance demonstrates total absence of cue information at
R2. In particular, the LSTM observations caution against equating an aggregate
sub-ULP variation ratio with no recoverable information.

At R16 the real probes have no consistent advantage over corresponding shuffled
controls: categorical accuracy is `5.81–7.13%` for GRU and `5.71–6.74%` for
LSTM; continuous MSE is `0.33429–0.36548` for GRU and `0.33978–0.36442` for
LSTM. This motivates a finite-horizon conditional-decoding hypothesis for these
checkpoints and sampled drivers. It does not certify nonlinear information
absence or explain the failed native task.

Probes use train-only means/scales, a `1e-8` feature-scale floor, fresh test
cues and separate fits per fixed driver/arm. This avoids using test labels to
fit the decoder, while allowing substantial offset compensation and
amplification unavailable to the selected native head. The design estimates
conditional recovery with a driver-specific decoder, not one decoder shared
across unseen drivers. Feature-floor counts, effective gains and noise-scaled
recovery remain unreported. Decodeability of these stored residuals is not yet
robust native memory. Preserve the R2 counterevidence rather than discarding
all small residuals as numerical noise.

## Next justified work and claims withheld

The four v1 outcomes are complete and should remain frozen. The proposed
[competence-first recovery protocol](competence-recovery-plan.md) starts with a
streaming oracle, immediate single-WRITE/query task, full-prefix zero body and
strict coordinate-wise R1/R2 competence. Its implementation and training need
separate frozen configuration, budget and authorization; this review initiates
none. Capture native states after WRITE, immediately before QUERY and after
QUERY to distinguish acquisition, retention and readout hypotheses.

Before interpreting geometry, propose bounded forensic assays on the existing
checkpoints: report per-feature probe-floor activation/effective gains; check
native fixed-cue replay and batched versus sequential execution; replay fixed
weights in higher precision as a separately labeled numerical assay; test
specified perturbation scales calibrated to meaningful prefix-state variation.
Retain the original float32 results. FP64 replay is neither retraining nor a
replacement native benchmark. These assays should test both whether R2 traces
persist and whether apparent R16 loss survives precision and decoder controls.

No orbit closure, tangent spectrum or causal intervention was measured. This
phase supports an engineering pipeline and an acquisition problem, not a
Floquet mechanism, causal harmful dimension collapse, architecture superiority,
multimodal transfer or a universal recurrent-memory limitation. Competent
retention, useful compression and stable categorical attractors remain valid
alternative future outcomes.
