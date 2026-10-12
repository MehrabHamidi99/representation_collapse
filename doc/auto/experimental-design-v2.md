# Experimental design v2: useful-history loss under controlled periodic forcing

Date: 2026-10-12. Version 2. Status: design document, not an empirical result.

## 0. Status and scope

**Framing.** When does repetition erase what a recurrent network needs?
Useful-history loss under controlled periodic forcing in fixed-weight recurrent
models; "dimension collapse across modalities" is the umbrella, not the claim.

**Primary research question.** At fixed weights and equal retention horizon,
does higher-order repetitive (periodic) forcing of a distractor body cause
excess loss of previously written, task-required information in a recurrent
model, relative to a matched non-periodic body (same symbol counts, cyclic
bigrams, run lengths, start/end context, total updates, final motif, cue,
query and nuisance)? Through which state-transition mechanism (input/state-
dependent gating, noncommuting propagation, cue-dependent basin selection or
merging, saturation, specified noise/precision effects)? When does stable
attractor memory prevent that loss?

**What v2 changes.** This document supersedes the execution order of
[periodicity-research-plan.md](periodicity-research-plan.md) section 3E: the
351-checkpoint streaming/depth screen becomes deferred coverage, not the next
compute request. It integrates Gates A-D of the
[novelty audit](../literature/novelty-review.md) (sections 4-6), the
[competence-recovery ladder](competence-recovery-plan.md), the frozen
[checkpoint-only periodicity protocol](periodicity-analysis-v1.md), the
measurement contract of [research-program.md](research-program.md), the
[graph reproduction protocol](graph-reproduction.md), [source inventory](source-inventory.md),
[agent protocol](agent-protocol.md) and
[artifact contract](../../skills/experiment-loop/references/workflow.md).
The dependency graph is [configs/research-plan.json](../../configs/research-plan.json)
(`representation-collapse-v2`); current state is in [the tracker](../TRACKER.md).
**No compute is authorized by this document**: all GPU submissions in both
recorded envelopes are consumed, and section 8 is an estimate requiring a new
recorded approval envelope.

**Evidence labels.** `established` (standard result); `derived` (elementary
consequence checked by a fixture or script here); `verified-infrastructure`
(synthetic fixtures pass; nothing about trained models); `executed-unverified`
(job completed, independent reduction pending); `preliminary-attributed`
(inherited manuscript statement without recovered provenance); `planned`.

**Non-claims.** Never supported by this design: "first Floquet analysis of RNN
memory"; "universal law of representation collapse"; "negative exponents imply
lost memory"; "periodic input implies periodic hidden orbit"; "PSD peak implies
neutral mode"; "modality-general collapse" from isomorphic renderers.

## 1. Task and intervention, checked against the implementation

Every statement here was checked against
[vector_task.py](../../src/representation_collapse/vector_task.py),
[pilot.py](../../src/representation_collapse/pilot.py),
[periodicity.py](../../src/representation_collapse/periodicity.py),
[vector-pilot-v1.json](../../configs/experiments/vector-pilot-v1.json),
[initial-phase-v1.json](../../configs/experiments/initial-phase-v1.json) and
[periodicity-analysis-v1.json](../../configs/experiments/periodicity-analysis-v1.json),
plus a read-only NumPy scratch script (`PYTHONPATH=src .venv/bin/python`);
line numbers refer to the working tree at commit `8fce9b0`.

### 1.1 Episode structure (verified)

- `INPUT_DIM = 38`: type one-hot `[0:5]` = WRITE, A, B, X, QUERY; coordinate
  one-hot `[5:21]`; categorical value one-hot `[21:37]`; real value `[37]`
  (`vector_task.py:14-16`). No arm or schedule channel exists; body
  observations carry only the A/B/X type channel.
- Episode = 16 WRITE observations, body, one terminal QUERY: length
  `17 + 12qR` (`render`, `vector_task.py:75`). WRITE `k` writes coordinate `k`
  in order `0..15`; coordinates `0..7` are required and `8..15` nuisance
  (`vector_task.py:77-83`; queries drawn from `0..7`, `vector_task.py:108`).
  Categorical memory: 16 equiprobable values; continuous: `Uniform[-1,1]` as
  float32 (`vector_task.py:50-55`). Body and QUERY carry no memory value
  (`tests/test_vector_task.py:39-40`).
- Updates from a required WRITE to the QUERY update, inclusive: 9-16 at R0,
  21-28 at R1, 33-40 at R2, 201-208 at R16 (derived; matches
  [competence-recovery-plan.md](competence-recovery-plan.md)).

### 1.2 Motif library and matched statistics (derived, script-checked)

The five motifs `AXAXAXBXBXBX`, `AXAXBXAXBXBX`, `AXAXBXBXAXBX`, `AXBXAXAXBXBX`,
`AXBXBXAXAXBX` (`vector_task.py:10-13`) are the letter sequences `AAABBB`,
`AABABB`, `AABBAB`, `ABAABB`, `ABBAAB` interleaved with `X`.

- Each has A=3, B=3, X=6; cyclic bigrams AX=XA=XB=BX=3; all runs of length one
  (length `q` after dwell); start `AX`, end `BX`; no proper whole-string period
  (`tests/test_vector_task.py:10-15`). The library is exactly the aperiodic
  subset of the six `(letter X)^6` strings whose letters permute `AAABBB`,
  start with `A` and end with `B`; the sixth candidate `ABABAB` =
  `AXBXAXBXAXBX` has proper period 4 and is excluded.
- Motifs can differ in symbol-level cyclic trigram counts (`AXA`, `BXB`, `AXB`,
  `BXA` are each 1 or 2 depending on the motif) and in letter-level
  (X-skipping) bigram counts (`AA` is 2 for `AAABBB`, 1 otherwise). For R>1
  the arms can differ at trigram order while bigram order is matched, but
  not every paired body differs in trigram counts (periodic motif 1 twice
  versus control motifs 2 then 1 is a counterexample). The treatment is a
  **specified higher-order ordering intervention**; report realized
  higher-order statistics rather than assuming pairwise trigram imbalance.

### 1.3 Arms and pairing (verified)

- Periodic arm: `R` copies of one uniformly drawn motif (`vector_task.py:39`).
  Control arm: `R` independent uniform motif identities, the last replaced by
  the periodic motif; bodies with any proper whole-string period are rejected
  (`vector_task.py:42-46`). R=1 returns identical arms (`vector_task.py:40-41`).
  Pairing: same memory prefix, nuisance, query and total updates; a batch holds
  `pairs` periodic episodes followed by the same `pairs` control episodes with
  duplicated targets (`vector_task.py:104-115`; `tests/test_vector_task.py:32-46`).
- Control statistics from 2,000 scratch draws: the fraction of motif positions
  equal to the periodic motif is 0.50 at R2 (forced by the final match), 0.25
  at R16, 0.21 at R64; the longest run of identical consecutive motifs in a
  control averages 2.5 at R16 (maximum observed 6 of 16). Controls have no
  proper whole-body period but may contain locally repeated motifs and
  repeated substrings, including final motifs.
  The checkpoint-only diagnostic's generator (`fixed_panels`,
  `periodicity.py:167-181`) nests controls across horizons as
  `control[:R-1] + final`, rejecting only if any of R2/R16/R32 tiles.

**Controlled:** symbol counts; symbol-level cyclic bigram counts (whole-body
bigrams also match because every motif junction is `X -> A`); run lengths;
start/end context; dwell; total updates; final motif; memory prefix; nuisance;
query; absence of any arm flag. **Not controlled:** symbol trigram and
letter-bigram counts; input autocovariance and spectrum (the periodic body has
a line spectrum at multiples of 1/12 per update); next-symbol predictability
given long context; number of distinct motif identities; prior exposures of
the final motif (R-1 in the periodic arm, about (R-1)/5 in the control). Dwell
`q` changes period and exposure jointly, so it is a dwell/period condition;
Gate B uses q=1 only. Report realized input- and encoder-level statistics with
every run (section 7).

### 1.4 Model, training and measurement as implemented in v1 (verified)

- Encoder `Linear(38 -> 64)` then `tanh`, jointly trained (`pilot.py:36`): a
  widening, not a bottleneck. Core: one-layer `nn.GRU` or `nn.LSTM`, input 64,
  hidden 128 (`pilot.py:38`). Head `Linear(128 -> 16)` or `Linear(128 -> 1)`
  on the hidden output after the QUERY update (`pilot.py:39,51-53`); the LSTM
  head reads `h` only.
- Complete state: GRU 128; LSTM `[h, c]` packed `h[0:128]`, `c[128:256]`
  (`pilot.py:41-45`; `periodicity.py:27-32`). Budget check: core+head
  parameters <= 1e6 and state <= 1 MiB FP32 (`pilot.py:256`).
- Capture point: immediately before the QUERY (`pre_query_state` runs the core
  on `inputs[:, :-1]`, `pilot.py:47-49`); forked QUERY readout equals the
  uninterrupted forward (`read_query`, `tests/test_pilot.py:32`).
- Training: AdamW, lr 1e-3, weight decay 0.01, clip 1.0, batch 32 = 16 pairs x
  2 arms, deterministic cycle R1, R2, R4, R8, R16 (`pilot.py:262-263`), 1,500
  updates, full-episode BPTT, one queried value per episode; per horizon 300
  updates, 4,800 distinct prefix/query pairs, about 600 queried examples per
  coordinate. Validation: fixed 32 pairs per train horizon every 100 updates;
  selection by minimum mean validation loss with strict `<`, so ties keep the
  earlier checkpoint (`pilot.py:259,274-279`).
- Evaluation: 256 paired episodes per horizon for R in {1, 2, 16};
  `delta = periodic_error - control_error`; `DID = delta16 - delta2`
  (`pilot.py:290-301,332-335`). Engineering gate: accuracy >= 0.25 or raw MSE
  <= 0.25 on arm-mean errors at R1 and R2 (`pilot.py:303-307`).
- Geometry: R in {2, 16}, 2 driver pairs per horizon, 512 training and 256
  held-out probe cues, one fixed nuisance vector shared by both arms
  (`pilot.py:309-315`); `covariance_dimension` (PR, covariance entropy
  dimension, numerical rank at relative tolerance 1e-10, total variance);
  ceiling `min(N-1, d)` applies: 128 for GRU and 256 for joint LSTM state at N=512. Probes: ridge with `alpha * N`
  regularization (alpha 0.01) on train-only standardized features (floor
  1e-8); categorical probes predict 8 x 16 one-hot blocks with per-coordinate
  argmax; identical permuted-label control (`ridge_probe`, `pilot.py:123-148`).

### 1.5 Document-versus-code mismatches found

None changes the v1 outcome; all must be resolved in the v2 configuration.

1. `periodicity-research-plan.md:444` requires at least 1,024 paired episodes
   per primary cell; v1 evaluates 256 pairs per horizon
   (`configs/experiments/vector-pilot-v1.json:26`; `pilot.py:291`).
2. `periodicity-research-plan.md:445-446` starts geometry with eight fixed
   driver pairs; v1 uses 2 per horizon (`vector-pilot-v1.json:27`;
   `pilot.py:309-310`).
3. `periodicity-research-plan.md:324-325` specifies a "64-dimensional
   observation bottleneck"; the encoder maps 38 -> 64 (`pilot.py:36`); already
   flagged in `competence-recovery-plan.md:58-60`.
4. `periodicity-research-plan.md:331-333` requires a three-learning-rate search
   and three seed blocks per cell; v1 ran one learning rate
   (`vector-pilot-v1.json:21`) and seed 0 only, although the config lists
   `seeds: [0, 1, 2]` (`:7`) and says "three diagnostic seeds" (`:42`).
5. `periodicity-research-plan.md:280` lists primary dwell q in {1, 2}; the v1
   validator rejects any dwell other than 1 (`pilot.py:158`).
6. `periodicity-research-plan.md:516-517` and `competence-recovery-plan.md:168-171`
   set coordinate-wise competence at 95% accuracy or normalized MSE <= 0.05;
   v1 encodes 0.25/0.25 on arm-mean errors without coordinate stratification
   (`vector-pilot-v1.json:33-35`; `pilot.py:303-307`).
7. `competence-recovery-plan.md:93-95` (R0 supported by `render` but not by
   `paired_body_ids` or the v1 validator) is confirmed: `render` returns
   `(n, 17, 38)` for empty bodies; `paired_body_ids(rng, 0)` raises
   (`vector_task.py:36-37`); `validate_config` fixes `train_repeats` and
   requires `{1, 2, 16}` (`pilot.py:158-161`). A confirmed gap.
8. `periodicity-analysis-v1.md:57-58` says a control driver "need not be
   nested across horizons"; the implementation does nest control prefixes
   (`periodicity.py:290`). The document understates the structure; no outcome
   depends on it.
9. `periodicity-research-plan.md:233-235` describes d in {1, 4, 8, 16} required
   coordinates; renderer and validator fix 16 written and 8 queried
   (`vector_task.py:65`; `pilot.py:167`). Planned, not implemented.
10. `periodicity-analysis-v1.md:79-80` (FP64 replay casts weights and inputs)
    is confirmed (`periodicity.py:283-284,302`), but the assay is bound to the
    checkpoint-only registry contract (section 7, item 10).

## 2. Staged design

Each gate lists question, inputs, conditions, sample sizes, measurements,
decision rule, falsifier, stop rule, and what completion releases. Edges are
evidence requirements, not requirements that a hypothesis be true.

### Gate A: analytic and measurement falsification

- **Question.** Do the measurement and analytic controls behave as derived, so
  that trained-model measurements can be interpreted? **Inputs.** `metrics.py`,
  `calibrate.py`, `mechanism_calibration.py`, `periodicity.py`; `tests/`.
- **Done (verified-infrastructure).** 10 measurement checks (known-rank
  covariance, degenerate zero variance, diagonal and noncommuting period maps,
  nonnormal transient gain, forced-scalar `a^T` response) and 36 mechanism
  checks in seven families: affine ordering null (equal centered cue clouds at
  equal horizon; offsets can change native error; pooled-body conditioning
  rejected); commuting switched-affine null (equal cue product, order-dependent
  process noise); noncommuting positive construction; isotropic attenuation
  preserves PR; locally contracting forced basins preserve labels; continuous
  neutral memory with contracting nuisance; fresh noise sustains covariance
  without old memory. Pass locally (`runs/calibration/setup-2026-10-11-v4/`,
  ignored) and on Narval CPU job 5234514. Derived ceilings: time-only centered
  covariance of an exact period-P orbit has rank <= P-1; `N_cue` cues have
  centered rank <= `N_cue - 1`.
- **Remaining (planned).** (i) Graph complete-state adapter with state `2NH`,
  node-ordering manifest, chronology test and tiny dense-versus-JVP agreement;
  (ii) intervention and sham operators validated on analytic fixtures (a swap
  moves the readout to the donor cue; an equal-norm rank-k random perturbation
  changes the affine readout by the predicted amount; a sham changes nothing);
  (iii) statistics module validated on planted-effect seed sets (bootstrap
  coverage, exact sign-flip p-values, Holm ordering, TOST); (iv) three-phase
  capture chronology fixture; (v) R0 renderer and streaming-oracle fixtures
  with corruption controls; (vi) D90/D95 and subsampling-curve fixtures;
  (vii) closure tolerances redefined in float32 ULP units at measured state
  scale with FP64 replay as reference: a float32 forced contracting map that
  converges to an exact float32 cycle must have zero residual; sparse and
  dense specified 1-ULP perturbations must reproduce the expected residuals
  and pass/fail decisions under the declared bounds, with bitwise equality
  reported separately
  ([implementation-audit.md](implementation-audit.md) section 5 R1);
  (viii) probe scale-equivariance and floor-accounting fixture (section 5 R2).
- **Decision rule.** Every declared check passes at its stated tolerance.
  **Falsifier.** A failing fixture, or one passing because its tolerance
  exceeds the quantity of interest. **Stop rule.** A failing fixture blocks
  dependent measurements. **Releases.** Gate B measurement code; Gate D adapter.

### Gate B: minimal trained mechanism study

**Question.** In competent fixed-weight GRU and full-state LSTM models on the
vector renderer with q=1, do periodic and matched control bodies differ in
required-information loss at equal horizon within the training support, and
which mechanism accounts for any difference? **Inputs.** A new versioned v2
configuration (the v1 config stays frozen), source commit, `uv.lock`, the
ladder below, frozen test manifests. **Provenance requirements.** Each fit
records, as measured values and not constants, `torch.backends.cuda.matmul.allow_tf32`,
`torch.backends.cudnn.allow_tf32`, `float32_matmul_precision`, the cuDNN
version and whether cuDNN is enabled; `pilot.py` sets neither TF32 flag and
writes `deterministic_algorithms: True` as a literal (`pilot.py:205`), so the
A100/H100 v1 pilots may have used TF32 inside cuDNN RNN kernels
([implementation-audit.md](implementation-audit.md) section 4 item 9, section
5 R8). Gate B decides pure-float32 training (TF32 off) before the first fit
and records it. A selected seed-0 configuration per model/task cell is run twice
and asserts identical weight hashes and identical `deltaR` (repeat-run
determinism check); a mismatch is reported and blocks interpretation of that
cell until explained. The periodicity path already sets and records these
flags (`periodicity.py:239-243,265-267`); the pilot path does not.

**B.0 Competence-first ladder** (from
[competence-recovery-plan.md](competence-recovery-plan.md)). Each fit follows a
fixed maximum schedule; stage transitions happen only at declared boundaries
and only if the preceding gate passes; no extra updates or relaxed thresholds
after a failure.

| Stage | Task | Updates | Gate to advance |
|---|---|---|---|
| 0 | Streaming register oracle on CPU fixtures (explicit memory reference, not a learned model) | none | exact categorical recovery; continuous error < 1e-12 at R0/R1/R2/R16; shuffled payloads/queries reduce agreement with original targets on distinct-value fixtures; random controls can retain chance matches |
| 1 | Single WRITE, immediate QUERY (new constructor) | 500 | fixed-batch overfit, then IID generalization, with recorded gradient and parameter-change norms |
| 2 | All 16 WRITEs, zero body (R0) | 500 | coordinate-wise gate on all 8 required coordinates |
| 3 | R0 and R1, equal exposure | 500 | coordinate-wise gate at R0 and R1 |
| 4 | R0, R1, R2, equal exposure, balanced arms | 500 | coordinate-wise gate at R0, R1, R2 in both arms |
| 5 | R1, R2, R4, R8, R16, equal exposure, balanced arms | 1,500 | selection only; R16 reported against the same gate, pass or fail |

Coordinate-wise gate: accuracy >= 0.95 on each required coordinate, or
normalized MSE = MSE/(1/3) <= 0.05 (raw <= 1/60) on each coordinate, using 128
independent validation episodes per coordinate per horizon per arm. R0 and R1
are shared anchors (arms identical by construction). Discovery grid: hidden
128, encoder 64, AdamW, weight decay 0.01, clip 1.0, batch 32; learning rates
frozen at [0.0005, 0.001, 0.003]; seed 0 for each of the four model/task
cells: at most 12 fits of at most 3,500 updates. Selection: minimum balanced
validation task loss over the trained horizons, eligibility requiring the
short-horizon gates at that checkpoint; ties keep the earlier checkpoint, then
the frozen grid order; neither geometry nor any arm contrast enters selection.
If no candidate qualifies in a cell, report acquisition/optimization
unresolved, stop the grid, and propose a diagnosed branch (encoder pretraining,
query-conditioned decoder, auxiliary prefix supervision) as a separately
labelled protocol. After discovery, freeze the learning rate and schedule and
run seeds 1 and 2 (feasibility pilots); confirmation uses ten fresh seeds and
untouched test manifests (section 5).

**B.1 Conditions.** GRU (state 128) and LSTM (complete state 256); categorical
and continuous targets; q=1; periodic and matched control arms; horizons R0
(diagnostic), R1 (shared anchor), R2 and R16 (primary, within support), R4/R8
(secondary, within support), R32/R64 (extrapolation, labelled separately).

**B.2 Sample sizes per seed.** Behavior: 1,024 paired episodes per evaluated
horizon, 128 per required coordinate, on a frozen test manifest. Geometry: 8
driver pairs x 512 training cues + 256 held-out probe cues per horizon per arm
per phase, nuisance fixed within a cloud and shared across arms; subsampling
convergence at N in {64, 128, 256, 512}. Tangent: 8 driver pairs x 2 arms x
{R2, R16} x {FP32, FP64} = 64 twelve-step windows at the first cue history of
each pair. Interventions: 256 paired episodes per horizon per intervention type
with equal-size random and sham arms. Replication: seeds 0-2, then 10 fresh.

**B.3 Measurements.**

1. *Behavior.* Native error per coordinate and arm; `deltaR` for R in {2, 16};
   `DID`; R1 as anchor; R32/R64 as extrapolation; zero-predictor and
   uniform-classifier references from the saved targets.
2. *Three-phase complete-state capture.* Post-WRITE (after the 16th WRITE),
   pre-QUERY (after the last body update) and post-QUERY, same episodes; LSTM
   `h`, `c` and `[h, c]` separately and jointly with coordinate scales; forked
   evaluation must equal uninterrupted execution.
3. *Cue-conditioned geometry.* Per cloud (fixed driver, nuisance, phase, arm):
   spectrum, PR, covariance entropy dimension, D90/D95, total variance,
   numerical rank at relative tolerance 1e-10, stable rank, the `min(N-1,d)` ceiling (128 for GRU; 256 for joint LSTM at N=512)
   and the subsampling curve; between-driver components kept separate.
4. *Information.* Train-fit ridge probes with shuffled-label controls per
   coordinate. The v1 standardization floor is an absolute constant
   (`np.maximum(x.std(axis=0), 1e-8)`, `pilot.py:128`) while observed
   per-feature std is ~1e-9 (cloud variances ~1e-16 over 128-256 coordinates;
   [implementation-audit.md](implementation-audit.md) section 4 item 16 and
   section 5 R2), so the effective ridge strength was horizon-dependent and the
   v1 R2-versus-R16 probe contrast is confounded. v2 requires: the floor
   declared relative to the cloud scale (for example a fraction of the median
   per-feature std, recorded in the config); per-feature floor-hit counts and
   effective gains `1/max(std, floor)` saved per cloud; a scale-equivariance
   fixture (noise-free linear memory, uniform rescaling 1e-10 to 1e2) in Gate A;
   and an unstandardized-probe sensitivity analysis reported next to the
   standardized one. Also: finite counterfactual cue separation between paired
   histories at each phase; offset-corrected decoding (decoder told the arm)
   as the readout-offset control.
5. *Tangent propagation.* Final 12-step propagator `P = J[11] ... J[0]`:
   singular values, finite-time rates `log(sigma_i)/12` per update, largest
   gain, tangent PR of `sigma^2`, eigenvalues labelled finite-time; dense
   product versus direct autodiff; FP64 JVP and central finite differences at
   eps in {1e-4, 1e-5, 1e-6}. A Floquet label requires input periodicity over
   the whole body, complete-state closure of adjacent 13-state blocks at the
   declared tolerances for every adjacent cycle pair, and Jacobian closure
   between the last two windows; otherwise the object stays a finite-time
   propagator. Closure tolerances are declared in float32 ULP units at the
   measured state scale, not as absolute constants: the v1 FP32 state
   tolerance `state_absolute = 1e-7` (`periodicity-analysis-v1.json:31`,
   applied at `periodicity.py:329`) is a scale-dependent global bound, not
   a universal rounding floor or bitwise-equality test. A single FP32
   coordinate at 0.1 can differ by one ULP (~7.45e-9) and pass the bound.
   An all-coordinate half-ULP norm is a hypothetical aggregate rounding
   scale, not a minimum nonzero difference. v2 reports coordinate ULP
   spacings at measured magnitudes, absolute/relative residuals, the number
   of changed coordinates and bitwise equality separately; any block ULP
   RMS normalization must use an explicitly matched residual normalization.
   FP64 replay is a separate dynamics comparison, not an exact-arithmetic
   reference. Gate A uses exact, sparse and dense ULP-perturbed fixtures
   with analytically specified residuals and tolerance decisions.
6. *Causal interventions with native readout.* (a) Complete-state swap between
   arms at pre-QUERY for the same cue (LSTM swaps `h` and `c`; `h`-only and
   `c`-only swaps are labelled decompositions); (b) same-phase transplant of
   the post-WRITE state into the pre-QUERY slot (body bypass), bounding the
   error attributable to the body; (c) perturbation of the pre-QUERY state
   along the train-fit task subspace (rank 8 or 16 from probe coefficients)
   versus an equal-norm rank-matched random subspace versus a sham re-run;
   (d) rescue by restoring the task-subspace component to its post-WRITE value,
   separating attenuation (rescuable) from erasure. Effects are native error
   changes relative to sham, summarized per seed.
7. *FP64 fixed-weight replay* (separately labelled numerical assay): cast
   weights and inputs, replay from the zero state, repeat behavior, separation,
   closure and tangent measurements; compare with native FP32. Not retraining.
8. *Noise-robustness assays.* Additive isotropic Gaussian noise on the complete
   state (a) after every body update and (b) at pre-QUERY only; scales
   {0.25, 0.5, 1, 2, 4} times the median post-WRITE paired-cue separation per
   coordinate, fixed from discovery seeds before confirmation; LSTM `h` and `c`
   scaled separately; report native error and probe recovery versus scale.

**B.4 Decision rule.** Interpret contrasts only in cells whose selected
checkpoint passes the coordinate-wise gates at R0, R1 and R2 in both arms.
Primary: per-seed paired `delta16` and `DID` with the intervals and Holm
correction of section 5; excess harm at R16 requires the `delta16` interval to
exclude the smallest effect of interest, relative deterioration the same for
`DID`; mechanism attribution requires convergent geometry, information and
intervention evidence (section 4). **Falsifiers.** Equal forgetting in both
arms; an interval excluding the smallest effect (no excess in the tested
models, tasks, horizons and dwell); a readout-offset explanation that equalizes
arms; preserved categorical decoding under contracting local spectra. **Stop
rules.** Competence failure stops the cell; numerical failure stops the fit; no
re-running until a desired sign appears; all cells and failures are reported.
**Releases.** Gate C predictions become testable; Gate D transfer starts from
a frozen prediction, including a null one.

**B.5 Optional commuting-null architecture control (planned; requires a pinned
implementation; not part of the first compute request).** Theory proposition
P2 ([theory-propositions.md](theory-propositions.md), "Commuting
switched-affine null") shows that a transition `s_{t+1} = A_{x_t} s_t +
b_{x_t}` whose matrices depend only on the current body symbol and commute
pairwise gives an arm-independent cue product at matched symbol counts; a
single diagonal selective state-space layer whose transition depends only on
the current body symbol satisfies this, while stacked layers break it because
later layers receive cue-dependent inputs. The control: one pinned diagonal
selective-SSM layer with its complete state declared (per-channel state plus
any auxiliary variables), trained through the same ladder and measured with
the same battery, run only after the GRU and LSTM cells. Prediction (frozen
from P2): the deterministic cue propagator and centered cue-conditioned
geometry agree at matched counts. Offset-compensated recovery should agree
under the same decoder class and no process noise, or matched terminal noise.
Raw `delta16` and `DID` need not vanish: native readout offsets and
order-dependent propagated process noise can change native error without
changing the deterministic cue propagator. A replicated difference in that
propagator or in noiseless centered geometry requires checking commutation,
complete-state accounting, history-dependent transitions and measurement
leakage before interpreting GRU/LSTM contrasts. A raw error difference
alone is not a failure of the commuting null. It is
optional and labelled planned; it needs its own envelope line and a pinned
implementation with a commutation fixture before any run.

### Gate C: theory that adds a prediction

- **Question.** Which prediction distinguishes gated/nonlinear propagation from
  the affine ordering null and the commuting switched-affine null? **Inputs.**
  Analytic fixtures; [theory-propositions.md](theory-propositions.md) (P1-P12,
  drafted 2026-10-12; document artifact, no empirical claim); frozen Gate B
  checkpoints.
- **Discriminating measurements.** (i) Under the affine null, centered
  cue-conditioned clouds and cue derivatives are arm-independent at equal
  horizon; a seed-replicated arm difference in centered cloud geometry or in
  the propagator's action on cue-difference directions refutes an
  affine-equivalent description of the trained model. (ii) The commuting null
  predicts equal final cue products for matched symbol counts; measure state
  dependence of per-step Jacobians (same symbol, different phase or cycle),
  commutator norms, and whether the propagators' action on the task subspace
  differs between arms beyond equal-count commuting products; the optional
  B.5 diagonal selective-SSM control, if run, is the architecture that should
  satisfy this null (P2) and anchors the comparison. (iii) Gating:
  saturation per arm and phase, and whether error differences track saturation
  of cue-carrying units. (iv) Basin memory: finite cue separation bounded away
  from zero while local singular values are below one. (v) Noise channel:
  `(I + H^T Sigma^{-1} H)^{-1}` is a declared Gaussian surrogate only; measured
  recovery under declared noise is the test.
- **Decision rule.** A proposition counts only with assumptions, proof or
  derivation, numerical fixture, counterexample, and a prediction frozen
  before confirmation seeds are inspected. **Falsifier.** No held-out
  improvement over covariance/horizon/norm baselines or no direction-specific
  intervention effect: descriptive diagnostic, not a causal theory. **Stop
  rule.** Standard results are foundations; a restated neutral-eigenvector
  argument or a commutator plot alone does not complete the gate.
  **Releases.** The prediction transferred in Gate D.

### Gate D: graph and cross-modal confirmation

- **Question.** Does a frozen Gate B/C prediction hold in a native temporal
  graph model and in one native non-graph task that requires retaining earlier
  information? **Inputs.** Pinned T-GRAB code commit and dataset revision with
  checksums (leads in [source-inventory.md](source-inventory.md); none pinned
  yet); [graph-reproduction.md](graph-reproduction.md).
- **Order.** (1) Pin commit and dataset revision; record GC-LSTM cell,
  preprocessing, splits, seeds. (2) Implement the full-state GC-LSTM adapter
  with state `2NH` (node-wise `h` and `c`), node-ordering manifest and
  chronology test. (3) Reproduce prediction performance: exhaustive-pair
  per-step F1 averaged over time, change-point metrics, persistence and
  EdgeBank baselines, threshold frozen on train/validation. (4) Resolve
  notation: AB is `Pdet(2,1)`, AABB is `Pdet(2,2)`; identify which condition
  the inherited figures used or record "unresolved". (5) Only then: closure of
  the complete graph state on the deterministic periodic input (burn-in,
  several initial states, period possibly a multiple of the input period),
  tangent windows, cue-conditioned geometry with graph-specific required-memory
  variables (delayed cause/effect lags). (6) One native non-graph sequence task
  with encoder signal-to-noise and external-memory audits (caches, buffers and
  prompt access counted as state).
- **Sample sizes.** Five seed blocks for the graph reproduction (the
  manuscript's replication count), bounded pilot first; native task sizes are
  fixed in a later plan revision.
- **Decision rule.** Transfer is supported only if the frozen prediction holds
  with seed-level uncertainty in the native model; reproduction discrepancies
  are reported without rewriting inherited values. **Falsifier.** Effects
  confined to one renderer, a faulty adapter or unused prefix information.
  **Stop rule.** No dynamics interpretation before prediction performance is
  reproduced; the inherited `|mu| ~ 0.41-0.44` stays preliminary-attributed.
  **Releases.** The deferred broad screen and the mitigation branch, each as a
  new plan revision with its own envelope.

## 3. Measurement battery

Numerical constants in the table describe the implemented v1 reference unless
explicitly marked otherwise. Proposed v2 ULP diagnostics, probe-floor rule and
acceptance thresholds remain subject to an exact hashed freeze; the table does
not silently redefine the frozen v1 reduction tolerances.

| Estimand | Quantity | Sampling unit | Centering / normalization | Tolerance | Fixture or test | Interpretation limits |
|---|---|---|---|---|---|---|
| Covariance geometry | Participation ratio `(sum l)^2 / sum l^2` | 512 cue histories at fixed driver, nuisance, phase, arm; per seed | centered by cloud mean; raw state coordinates; normalized by trace | eigenvalue > 1e-10 x max for numerical rank; variance compared at atol 0, rtol 1e-5 | `metrics.covariance_dimension`; `test_metrics` spectrum/invariance tests; `isotropic_decay_preserves_pr` | ceiling `min(N-1,d)` (128 GRU; 256 joint LSTM at N=512); not intrinsic dimension; changes under anisotropic rescaling; LSTM h/c scales; undefined at zero variance |
| Covariance geometry | Covariance entropy dimension `exp(-sum p log p)` | same | same | same | same | distinct from singular-value entropy rank of Z |
| Covariance geometry | D90 / D95 | same | same | smallest rank reaching 90% / 95% of trace | not implemented (section 7) | threshold-dependent |
| Covariance geometry | Total variance `trace(C)`; per-component h/c | same | uncentered state RMS also reported | atol 0, rtol 1e-5 | `isotropic_variance_decays_exactly` | attenuation is not dimension loss; v1 variation-to-ULP ratios 0.25-2.5 |
| Covariance geometry | Subsampling curve of PR, entropy, D90 at N in {64,128,256,512} | nested subsamples | same | report only | planned | estimator bias at small N |
| Tangent propagation | Singular values of `P_{t,12}`, rates `log sigma_i / 12`, largest gain, tangent PR | one window per driver pair, arm, horizon, precision; first cue history | fixed inputs and weights; complete state; product in replay dtype plus float64 accumulation | product vs direct map FP32 1e-5/1e-4, FP64 1e-10/1e-8; FD atol 1e-8, rel 1e-4 | `periodicity.tangent_window`; `test_periodicity` derivative and ordering tests; nonnormal fixture | local to one history; partial description; spectral radius is not Euclidean contraction |
| Tangent propagation | Floquet multipliers (eigenvalues of the 12-step map) | window on a verified orbit only | as above | state closure FP32 1e-7/1e-5, FP64 1e-10/1e-8; Jacobian closure FP32 1e-6/1e-5, FP64 1e-10/1e-8 | `periodicity.distance`; closure tests; forced-scalar `a^T` fixture | tolerance-qualified; forced discrete orbits need no neutral multiplier; period may be a multiple |
| Tangent propagation | Closure residuals (absolute, relative, max coordinate) over 13-state blocks | all R-1 adjacent cycle pairs | reference scale max norm, floor 1e-12 | as above | same | a pass is approximate-closure-compatible, not an exact orbit |
| Behavior | Categorical error; continuous raw and normalized MSE per coordinate | 1,024 paired episodes per horizon, 128 per coordinate; summarized per seed | none; coordinate-stratified | gates 0.95 / 0.05 | `pilot.evaluate_batch`; `diagnostics.streaming_register_oracle` | one seed is one replicate; equal aggregates do not imply equal predictions |
| Behavior | `delta2`, `delta16`, `DID` | seed | paired within seed | smallest effect of interest (section 5) | `test_tiny_run_complete_artifacts_and_paired_contrast` | positive DID is relative deterioration, not harm at a horizon; DID at chance is uninformative |
| Information | Ridge probe accuracy / MSE vs shuffled labels; floor counts; effective gain | 512 train / 256 test cues per cloud | train-only mean and SD, floor 1e-8; `alpha N` ridge, alpha 0.01 | advantage over control, reported | `pilot.ridge_probe`; `test_probe_recovers_known_linear_memory_and_label_control` | decodability is not native use; linear failure is not absence; decoder is driver-specific |
| Information | Counterfactual cue separation (complete state, h, c) by phase | paired cue histories | raw norms; ratio to closure residual undefined at zero | report | periodicity paired-separation arrays | two histories are not a covariance sample |
| Information | Offset-corrected decoding by arm | same clouds | decoder may know the arm | report | `affine_offset_corrected_*` checks | tests the readout-offset alternative only |
| Causal | Native error change under swap, transplant, task-subspace perturbation vs equal-norm random vs sham | 256 paired episodes per horizon per intervention per seed | perturbation norm and rank matched | effect vs sham with seed CI | planned (section 7) | shows causal use of state directions, not mediation by PR |
| Numerical | FP64 replay of all of the above | same histories | cast weights and inputs | FP64 tolerances above | `periodicity.replay`; `test_native_complete_state_trajectory_and_query` | arithmetic assay; backend differences possible; not retraining |
| Robustness | Error and probe recovery vs noise scale at declared location | paired episodes; scales {0.25..4} x median post-WRITE separation | covariance transformed with coordinates | report curves | planned | noise location must be declared; equal numerical noise is not comparable across architectures |
| Input statistics | Symbol counts, cyclic bigram/trigram counts, letter bigrams, autocovariance per body | each realized body | none | exact match for controlled statistics | `test_library_and_paired_body_statistics`; scratch check | the arms differ at trigram order by design |

## 4. Interpretation matrix

Outcomes are judged per cell on confirmation seeds with the smallest effect of
interest (SEI) and intervals of section 5; "equal" means within `+/- SEI`.

| Outcome pattern | Supports | Refutes or qualifies |
|---|---|---|
| Coordinate-wise gate fails at R0-R2 in any arm (the v1 outcome in all four cells) | acquisition failure | nothing about H1-H4; DID at chance is uninformative |
| Gates pass at R2; `delta16` and `DID` equal; both arms degrade at R16 | shared horizon forgetting | H1 as periodic-specific harm; any universal repetition claim |
| `delta16 > SEI`, `DID > SEI`; periodic arm loses task-subspace variance and probe recovery; task-subspace perturbation and rescue change native error beyond random and sham | H1 harmful collapse through a state mechanism | the affine null as a description of the trained model |
| `delta16 > SEI` but probes and offset-corrected decoding recover the cue equally in both arms | readout-offset effect (named alternative) | erasure of stored information |
| `delta16 > SEI`, probes recover in both arms, task-subspace interventions ineffective, saturation differs by arm | access failure at readout | a storage-loss mechanism |
| `delta16` equal; lower PR or variance in the periodic arm with preserved performance | H2 sufficient compression | collapse as harm; "lower PR implies lost information" |
| Local singular values below one in both arms; closure compatible in the periodic arm; finite cue separation bounded away from zero; labels preserved | stable categorical basin memory (Pals-type) / H3 persistent response | a local-spectrum explanation of forgetting |
| Local contraction; cue separation to zero; probes fail at R16 in both arms | synchronization-type forgetting shared by arms | periodic specificity |
| FP64 replay changes contrasts, separations or closure materially | H4 precision-artifact caveat | an unqualified FP32 geometric interpretation |
| Geometry depends on `N_cue` without convergence, or on pooled drivers | H4 sampling artifact | any dimension claim from that cloud |
| `delta16 < -SEI` (periodic better) | repetition-stabilized memory; admissible opposite effect | harmful repetition in that cell |
| `DID > SEI` with `delta16 <= SEI` | relative deterioration only | excess harm at R16 |
| Closure fails in the periodic arm | finite-time regime | any Floquet multiplier statement |

## 5. Statistics and preregistration

- **Replication unit.** The independent training seed. Episodes, time steps,
  cues, coordinates, driver pairs and horizons within one checkpoint are not
  replicates. Per seed, compute the paired mean error difference over matched
  test episodes per horizon (`deltaR`) and `DID`; analyze these seed-level
  paired effects across seeds.
- **Interval method (default).** Paired bootstrap over seeds, `B = 10,000`
  resamples, percentile intervals on the seed-level effects: no parametric
  shape is imposed, but independent representative seeds and suitable
  moments are required; finite-sample coverage is approximate. Because
  percentile intervals are coarse with 10 seeds, also report a paired
  sign-flip sensitivity test (all `2^10 = 1,024` sign assignments) and a
  paired t-interval. Sign flipping is exact only under justified null
  sign-exchangeability (for example independent seed effects symmetric
  about the tested boundary) or a valid sign-randomization design. A
  zero-mean null alone is insufficient. Declare the shifted SEI null,
  one/two-sided alternative and exchangeability justification before
  freezing; if unjustified, revise the primary test before confirmation; a
  claim requires the primary sign-flip decision (next bullet) and both
  interval methods to agree on SEI exclusion, otherwise the result is
  inconclusive.
- **Registered family and Holm.** Per cell (GRU/LSTM x categorical/continuous)
  two registered primary tests, `delta16` and `DID` against the SEI: eight
  tests. Holm governs the eight-test decision family at level 0.05; the
  SEI-exclusion decision for each test uses the Holm-ordered p-values from the
  paired sign-flip test as primary only after its null assumptions are
  justified in the freeze record; it is not distribution-free for mean
  error differences. For reporting, simultaneous
  intervals are Bonferroni-adjusted paired intervals at level `1 - 0.05/8`
  (percentile bootstrap and t), used as the effect-size report, not as the
  decision device. `delta2` is a registered descriptor, not a claim target. R32/R64, geometry,
  probes, tangent and intervention quantities are secondary and carry no
  significance claims. **Proposed; freeze before confirmation.**
- **Smallest effect of interest (proposed).** Categorical: a paired error
  difference of 0.02 (two percentage points of coordinate-averaged error).
  Continuous: a normalized-MSE difference of 0.02 (raw 1/150). Rationale: per
  seed, 1,024 paired episodes resolve error differences near 0.01 at the 5%
  error level, and 0.02 is 40% of the continuous competence margin. The
  equivalence margin equals the SEI. **Freeze these values and their
  justification before any confirmation-seed held-out result is inspected.**
- **Equivalence versus absence.** "No excess" is claimed only by two one-sided
  tests at the SEI margin (a 90% interval inside `+/- SEI` is the
  unadjusted per-comparison rule). Equivalence claims across the registered
  family require a separately specified multiplicity correction, such as
  Holm on valid TOST p-values; freeze that family and assumptions too. An interval merely
  containing zero is "inconclusive", never "no effect", and equivalence holds
  for the tested models, tasks, horizons and dwell only.
- **Power check before confirmation.** Estimate the seed-level standard
  deviation `s` of `delta16` from the three discovery seeds; with 10 seeds an
  ordinary 95% paired t-interval has half-width `0.7154 s`, whereas the
  proposed 99.375% Bonferroni reporting interval has half-width `1.1215 s`
  (df=9). Use the latter for this precision check. If `1.1215 s > SEI`, record that 10
  seeds are underpowered and request more in a new envelope or state the
  limitation; never lower the SEI after seeing data.
- **Selection rules.** Learning rate and checkpoint by balanced validation task
  loss only; eligibility by the coordinate-wise gates; ties to the earlier
  checkpoint, then the frozen grid order; no selection on geometry, tangent
  spectra or any contrast's sign; identical configuration across seeds.
- **Freezing.** Before any confirmation test manifest is evaluated, write a
  freeze file hashing the configuration, protocol, SEI, interval method,
  family, selection rule and manifest seeds; test manifests come from a
  declared seed offset and are never inspected before the freeze file exists.
  Discovery seeds 0-2 stay discovery evidence.
- **Reporting.** Every fit including non-qualifying fits and numerical
  failures; every cell, horizon, arm, phase and precision; raw per-coordinate
  errors; validation histories; failed branches; freeze-file hash; exact seed
  list. Missing seeds are reported, never dropped.

## 6. Evidence ledger

Statuses follow [research-program.md](research-program.md): planned,
blocked-input, executed, artifact-verified, interpreted, accepted, rejected-claim.

| Node / result | Status | Source commit | Job IDs | Artifacts | Establishes | Does not establish |
|---|---|---|---|---|---|---|
| Graph source audit | artifact-verified (documents) | repository | none | `source-inventory.md`, `research-program.md`, `graph-reproduction.md` | claim ledger; measurement contract | any empirical graph result |
| Measurement calibration (10 checks) | artifact-verified; verified-infrastructure | `9004a0c` (cluster), local v4 | Narval CPU 5234514 | `runs/calibration/setup-2026-10-11-v4/` (ignored); archive `/project/def-ioannism/mehrab/representation-collapse/evidence/initial-phase-v1-9004a0c/narval_5234514/calibration` | estimator behavior on known fixtures | anything about trained models |
| Mechanism calibration (36 checks, 7 families) | artifact-verified; verified-infrastructure | `9004a0c` | Narval CPU 5234514 | same archive, `mechanisms/` | analytic nulls and counterexamples | that trained models follow any fixture |
| Compute bootstrap | executed | bootstrap | Tamia 536757 (exit 0); Mila 11190579 (failed before calibration) | ignored ops journals | wrapper correctness after fix | no scientific content |
| v1 pilot GRU categorical | artifact-verified; interpreted: inconclusive (competence failed) | `9004a0c114a8b7c5335dca3d5247449fd10b5ffa` | Narval 5234860 (243 s, A100) | `/scratch/mehrab/representation-collapse/logs/cluv/narval_5234860/pilot/`; checkpoint sha `f1db7938...`; archives Narval `/lustre06/project/6044649/mehrab/...`, replicas Tamia/Mila | accuracy 5.47/6.25/5.08% at R1/R2/R16 (chance 6.25%); DID 0; R2 probes 20.9-22.8% vs shuffled 6.1-7.1% | harmful collapse; absence of cue information; the R2 > R16 probe contrast, confounded by the absolute `1e-8` feature-scale floor (`pilot.py:128`) active at the observed ~1e-9 feature std until re-reduced from the saved `probe_feature_scale` arrays ([implementation-audit.md](implementation-audit.md) section 4 item 16) |
| v1 pilot GRU continuous | same | same | Narval 5235508 (228 s, A100) | `.../narval_5235508/pilot/`; sha `e35502d2...` | MSE 0.353/0.364/0.334 near zero-predictor 0.353/0.359/0.334; DID -1.0e-7 | same, including the probe-floor confound |
| v1 pilot LSTM categorical | same | same | Rorqual 22980985 (52 s, H100 MIG 3g.40gb) | `.../rorqual_22980985/pilot/`; sha `8752a480...`; archive `/lustre09/project/6044649/mehrab/...` | same aggregate accuracies as GRU; DID 0; R2 probes 14.7-15.5% vs 6.0-6.8% | same, including the probe-floor confound |
| v1 pilot LSTM continuous | same | same | Rorqual 22980998 (32 s) | `.../rorqual_22980998/pilot/`; sha `ec63862d...` | MSE 0.352/0.366/0.335; DID -5.9e-8 | same, including the probe-floor confound |
| v1 verification | artifact-verified | helpers after `9004a0c` | none | `runs/ops/phase1-final-verification-20261011.json` sha `4a71636f...`; verifier sha `5f092ee5...`; [compact results](initial-phase-results.json) | 6/6 directories reproduce from raw arrays | scientific acceptance |
| Periodicity checkpoint diagnostics (4 cells) | executed; hashes verified (28 artifacts, 359,996,711 bytes); reduction pending (`executed-unverified`) | analysis `458bc9cfce474b8e11e433e33bf132faf5032387`; training `9004a0c` | Narval 5254858 (263 s), 5254871 (231 s); Rorqual 22984956 (62 s), 22984963 (52 s) | `/scratch/mehrab/representation-collapse/logs/cluv/<cluster>_<job>/periodicity/` (`inputs.npz`, `trajectories.npz`, `tangents.npz`, `metrics.json`, `manifest.json`); [status](periodicity-analysis-status.json); [launch](periodicity-analysis-launch.json) | worker-reported checks pass: 64 dense windows, 1,536 step Jacobians, 128 assays, 256 adapter passes, 32 FP64 JVP windows, 96 FD checks | closure, contraction or separation conclusions before recomputation from arrays; any Floquet or collapse claim (competence failed); FP32 closure flags as evidence of near-periodicity: the fixed `state_absolute = 1e-7` tolerance is not a bitwise-equality test or a universal ULP floor; independent reduction must report measured coordinate spacings, changed-coordinate counts and residuals, with a separate FP64 dynamics comparison ([implementation-audit.md](implementation-audit.md) section 5 R1) |
| Independent periodicity reduction | planned | n/a | none | dated notebook (pending) | pending | pending |
| Graph assets | blocked-input | none pinned | none | recovery leads only | nothing yet | legacy checkpoint identity |
| Graph reproduction / interpretation | blocked / planned | none | none | none | nothing | the inherited `|mu| ~ 0.41-0.44` (`floquet_analysis/main.tex:502-510`, preliminary-attributed) |
| Literature audit | interpreted (documents) | repository | none | [novelty-review.md](../literature/novelty-review.md), 90 records | broad novelty claim rejected; narrower question retained | any empirical result |
| Qwen worker proposals | not evidence | n/a | Mila PID 2812601 | ignored `runs/ops/phase1-qwen-*.json`; 14 schema-valid reviews by 16:23 UTC 2026-10-11 | the worker ran within its contract | any scientific claim; first proposal exceeded the envelope and used an insufficient stopping rule |
| Competence recovery discovery; mechanism study Gate B; native transfer Gate D | planned | none | none | none | pending | pending |
| Theory Gate C (drafted) | drafted (document artifact; no empirical claim) | repository, 2026-10-12 | none | [theory-propositions.md](theory-propositions.md): P1-P12, 14-row corrections table, 29/29 scratch checks, 34 tests and the 10 + 36 calibration checks rerun | propositions with assumptions, derivations, fixtures, limits and frozen predictions as stated in the document | any empirical result; acceptance of any prediction; Gate C completion before Gate B checkpoints exist |
| Research charter and paper skeleton (drafted) | drafted (document artifacts; no empirical claim) | repository, 2026-10-12 | none | [research-charter.md](research-charter.md); `paper/` (34 pages after the audit caveats were added, builds with 0 errors, 86 cite keys verified, 17 named `\todo` placeholders) | framing, non-claims and manuscript scaffold consistent with section 0 | any result; no placeholder states an unaccepted finding |
| Implementation audit (drafted) | drafted (document artifact; read-only code audit; no empirical claim) | repository, 2026-10-12 | none | [implementation-audit.md](implementation-audit.md): 34/34 tests in 4.4 s (Python 3.11.13, NumPy 2.4.6, torch 2.6.0); 10 and 36 calibration checks reproduced with matching manifest hashes; `uv run` broken locally by hidden `.pth` files, `PYTHONPATH` fallback used | code-level coverage gaps, numerical risk register, prioritized backlog | any empirical result; that the backlog is complete |

No accepted scientific claim exists at the time of writing.

## 7. Design-driven implementation gaps relative to `src/` and `tests/`

This list covers only what the v2 design requires; the prioritized, acceptance-
tested backlog with file:line evidence is
[implementation-audit.md](implementation-audit.md) section 7 and is not
duplicated here.

1. **R0 support — additive constructor implemented.**
   `diagnostics.paired_batch_with_zero` returns identical empty-body arms.
   The frozen v1 constructor and validator stay unchanged; a stage-aware v2
   training validator remains needed.
2. **Single-WRITE immediate-QUERY constructor — implemented.**
   `diagnostics.single_write_episode` keeps the renderer layout and emits two
   observations with no target value in QUERY.
3. **Streaming register oracle and corruption controls — implemented.**
   `diagnostics` supplies eight explicit registers, payload/query controls and
   input validation. These fixtures do not establish learned competence.
   See [implementation and checks](diagnostics-implementation.md).
4. **Coordinate-wise gates and stratified manifests.** The gate uses arm-mean
   errors (`pilot.py:303-307`) and validation is unstratified (`pilot.py:259`).
5. **Staged, gate-conditioned curriculum and learning-rate grid.** The loop is
   a fixed cycle at one learning rate (`pilot.py:262-272`).
6. **Three-phase complete-state capture.** Only `pre_query_state` exists
   (`pilot.py:47-49`); post-QUERY `c` is not exposed by `forward`;
   `periodicity.replay` keeps every state but is single-sample.
7. **Interventions and shams.** No swap, transplant, subspace-perturbation,
   rescue or sham operators exist; `read_query` (`pilot.py:55-57`) and
   `unpack_state` (`periodicity.py:27-32`, single sample) are the only
   building blocks. Analytic-fixture tests belong to Gate A.
8. **Statistics module.** No paired bootstrap, sign-flip, Holm or TOST code.
9. **Geometry additions.** `covariance_dimension` (`metrics.py:12-52`) lacks
   D90/D95, stable rank and subsampling curves; per-component `h`/`c` clouds
   and the between-driver component need helpers.
10. **FP64 replay exists but is bound to the checkpoint-only contract.**
    `periodicity.run` casts weights and inputs (`periodicity.py:283-284,302`)
    and `replay` works at any dtype, but `validate_config` fixes horizons
    `[1, 2, 16, 32]` (`periodicity.py:152-153`) and `run` requires
    `effective_training_steps == 1500` plus the v1 registry
    (`periodicity.py:236`). A generalized entry point is required.
11. **Tangent windows for Gate B.** `tangent_window` and `distance` are
    reusable (`periodicity.py:72-108`); the compatibility flag uses only the
    last adjacent cycle pair (`periodicity.py:329-333`); v2 requires every pair.
12. **Noise-robustness assay.** Not implemented.
13. **Input-statistics artifact.** No per-run report of realized body statistics
    per arm; the existing test covers counts and bigrams only.
14. **Freeze and manifest ledger.** Provenance records source hashes
    (`pilot.py:175-206`); nothing binds SEI, statistics and manifest seeds.
15. **Graph adapter.** No GC-LSTM full-state adapter (`2NH`), node-ordering
    manifest or chronology test exists (Gate D).
16. **Precision provenance and determinism.** `pilot.py` neither sets nor
    records TF32/cuDNN flags and writes `deterministic_algorithms: True` as a
    constant (`pilot.py:205`); `periodicity.py` does both (`periodicity.py:239-243,265-267`).
    Needs: recorded `allow_tf32` (matmul and cuDNN), `float32_matmul_precision`
    and cuDNN version in pilot provenance; a CPU repeat-run determinism test in
    `tests/` and a CUDA pair in the job script asserting identical weight
    hashes (audit section 4 item 9; backlog items 2 and 8).
17. **Probe floor accounting.** Floor relative to cloud scale, floor-hit
    counts, effective gains, scale-equivariance fixture and unstandardized
    sensitivity (B.3 item 4; audit section 5 R2, backlog item 7). The v1
    `probe_feature_scale` arrays in `cue_clouds.npz` allow re-reduction.
18. **ULP-aware closure tolerances.** `distance` (`periodicity.py:95-108`)
    takes absolute and relative bounds only; needs residuals in float32 ULP
    units at measured state scale and a `resolution_limited` flag (B.3 item 5;
    audit section 5 R1 and R3).

## 8. Compute estimate (requires a new approval envelope; nothing submitted)

These are provisional planning extrapolations, not validated runtime or memory
ceilings for the expanded battery. Benchmark one eligible full measurement
battery before pricing a new concrete allocation. The repeat-run allowance
below covers one selected seed-0 configuration per model/task cell (four extra
fits), not all twelve learning-rate candidates; repeating all candidates would
require twelve extra fits and revised costing. Freeze that choice explicitly.

Derived from the verified v1 runtimes: GRU 228-243 s and LSTM 32-52 s per
1,500-update pilot including setup, evaluation and geometry, on one A100
(Narval) and one H100 MIG `3g.40gb` slice (Rorqual); hardware and architecture
are confounded. Ceilings attribute all elapsed time to updates (0.162 s GRU,
0.035 s LSTM per update) and price forward-only validation at one third of a
training episode with no batching gain.

| Item | Fits | Per-fit ceiling | Device-hours ceiling | Requested walltime |
|---|---|---|---|---|
| Discovery grid (4 cells x 3 learning rates, <= 3,500 updates, validation every 100 updates on 10,240 episodes) | 6 GRU + 6 LSTM | GRU ~20 min (567 s updates + 605 s validation); LSTM ~4 min | ~2.4 | 6 x 30 min + 6 x 15 min = 4.5 GPU-h |
| Feasibility seeds 1-2 at the frozen setting | 4 GRU + 4 LSTM | same | ~1.6 | 3.0 GPU-h |
| Confirmation (10 seeds x 4 cells) | 20 GRU + 20 LSTM | same | ~7.9 | 15.0 GPU-h |
| Measurement per checkpoint (three-phase capture, 8-driver geometry, 64 tangent windows, interventions, FP64 replay), scaled from the periodicity diagnostics (263 s GRU, 62 s LSTM for 16 windows) | per eligible checkpoint | GRU ~18 min; LSTM ~4 min | ~7.2 for 40 confirmation checkpoints | 40 x 20 min = 13.3 GPU-h |
| Gate B total | | | ~21 | ~36 GPU-h requested |

The optional B.5 commuting-null control and the repeat-run determinism pairs
are excluded from this table; the determinism pair adds one discovery-seed fit
per cell (about 1 GPU-h at the ceilings above) and B.5 needs its own line once
an implementation is pinned.

v1 peak RSS was below 1.4 GB, so 8 GiB host memory and 4 CPUs per job suffice;
dense 128x128 and 256x256 tangent windows are also feasible on CPU. A concrete
envelope must state cluster, account, submissions, per-job resources, walltime,
concurrency, total ceiling, retry policy and stopping rule before any
submission ([AGENTS.md](../../AGENTS.md)). This is an estimate, not a request.
