# Checkpoint-only periodicity analysis v1

Date: 2026-10-11. Status: bounded post hoc numerical diagnostics; no training.
This protocol analyzes the four seed-0 GRU/LSTM × categorical/continuous
checkpoints from training source
`9004a0c114a8b7c5335dca3d5247449fd10b5ffa`. All four previously failed native
short-horizon competence. The
[final scientific review](initial-phase-scientific-review.md) and
[verified raw-output report](../../../../../runs/ops/phase1-final-verification-20261011.json)
remain the reference outcomes. This analysis does not repair acquisition or
promote those pilots to scientific confirmation.

## Question and bounded design

Do repeated body inputs produce approximately repeating complete-state
trajectories in these fixed models? How do absolute cue separation,
finite-window tangent gains and precision dependence relate to that observation?
Agreement, divergence, expansion, contraction and failure of approximate closure
are all admissible outcomes.

The [frozen analysis configuration](../../../../../configs/experiments/periodicity-analysis-v1.json)
records the four checkpoint hashes and original failed competence before launch. The analysis source is a new commit, distinct from the training commit.
Record both identities; do not claim that new analysis code trained the existing
weights. The bounded design is four checkpoint-only cells, each at most one GPU
for 15 minutes, with at most two concurrent jobs: at most one requested GPU-hour
in total. The application also checks an 840-second internal cap between
trajectories; scheduler termination remains the outer bound. No training, checkpoint selection, automatic retries or additional
allocations are part of this protocol.

Within each cell, use:

- Two fixed driver panels, each with two distinct required-cue histories sharing
  identical nuisance values. Reuse those histories across arms, horizons and
  precision. Freeze the random generation seed and save the literal arrays.
- Repetition counts R1, R2, R16 and R32, with dwell q=1 and 12 body updates per
  motif. Use the existing motif vocabulary and matched varying-order control
  generator; save every motif ID and resulting body sequence.
- Native float32 and fixed-weight float64 dynamical replay. R32 is a newly
  analyzed extension beyond the original evaluated R16, not a training horizon.
- Sixteen dense tangent products per cell: two panels × two arms × two
  precisions × R2/R16, differentiated at the **first cue history only**. Each
  product contains the final 12 body-update Jacobians, with the previous
  12 Jacobians retained for phase-matched comparison. Thus each cell computes
  384 single-step Jacobians and 16 current-window direct-map Jacobians. The
  previous windows are not additional reported products. The second history is
  used for state-separation diagnostics, not a second tangent estimate.

There are 16 body trajectories per cell: two panels × two histories × two arms
× two precisions, each continued through the four measured horizons. If the
implementation instead renders separate episodes per horizon, save that fact
and the exact horizons/driver histories; do not infer a trajectory change from
unmatched histories. Four per-horizon renders of each combination are 64
trajectory evaluations, rather than 16 unique combinations. The implementation
uses these separate renders. Each panel uses an independent fixed cue pair;
sampling and driver seeds are seed+60000 and seed+70000 respectively. The
two periodic panels use different final motifs. A control driver
need not be nested across horizons; its literal sequence fixes what was tested.
Horizon-dependent control changes are an interpretation limit, not independent
repetition replicates.

## State and chronology

Let s_t be the complete recurrent state immediately before body input u_t, and
s_(t+1)=F_theta(s_t,u_t). GRU state has 128 hidden coordinates. LSTM state is
[h,c], with 128 hidden and 128 cell coordinates, giving 256 total. Record the
flattening order and native layer/batch shapes. Encoder features and QUERY
selection are inputs, not persistent recurrent-state coordinates in this model.

Start from the native zero state, process all sixteen WRITE observations, then
process the body. Save the after-WRITE state and same-phase body states at the
specified repetitions. The terminal QUERY is excluded from closure and tangent
products. To check native behavior, fork from the pre-query state, apply the
original QUERY update, and use the original head. The bounded diagnostic
queries coordinate zero only, so it does not reassess eight-coordinate native
competence. Verify this forked output
against uninterrupted native forward execution on an identical episode. Never
substitute a linear probe for this native query/head check.

Float64 replay must cast **the fixed model weights and every input**, then run
WRITE and body recurrence entirely in float64 from the zero state. Casting
already recorded float32 states into float64 is not a high-precision dynamical
assay. Preserve both versions and the arithmetic backend. Float64 here changes
arithmetic, not learned weights or native benchmark semantics. A discrepancy
can involve backend execution as well as rounding; no single cause is presumed.
The renderer preserves the same original float32 cue values in both assays;
float64 does not introduce higher-precision latent values or a new input sample.

If a manual differentiable recurrent-cell implementation is used because a
fused backend cannot expose the required state derivative, compare its one-step
and episode outputs against the native module in each precision. Record raw
errors and declared tolerance. Different formulations may produce nonidentical
float32 rounding even with the same real-valued equations. Native float32
behavior is the reference; never silently replace it with the derivative
adapter's behavior. The implementation saves manual states, native one-step
complete-state trajectories, native fused full-prefix pre-query h/c state,
and native full hidden-output chronology separately. It also saves native and
manual paired-history separations, including separate LSTM h/c norms. Closure
and derivatives are calculated on the manual trajectory; recompute native
closure from the retained native arrays when interpreting native behavior.
The adapter tolerance can exceed the cue separation, so its pass flag does not
establish equality of the two implementations' cue traces.

The analysis disables CUDA matmul and cuDNN TF32 and records these settings.
This is a controlled new numerical assay, not a claim of bitwise replay of the
original training or evaluation arithmetic.

## Forcing and approximate closure

Periodic-arm forcing repeats one declared 12-symbol motif at matched phase.
Verify the realized input vectors, not just the arm name. R1 is an identical-arm
anchor; its inputs and measurements must agree within the declared arithmetic
contract. For R>1, the varying control must have no proper whole-string period
under the existing exact-tiling definition. Motif counts of A/B/X, episode length
and final motif are matched by construction; preserve other matching checks
used by the generator. A finite control can share local repeated substrings,
including a locally repeated final motif. That does not establish a globally
periodic reference system.

Compare the last two 12-step cycles at the same body-input phases, including
cycle boundaries. Adjacent 13-state blocks include both boundaries and must
have equal shape; there are R-1 such comparisons at horizon R. Report maximum coordinate error, L2 errors and scale-aware
relative errors, with their denominator convention. A single pair of close
cycle endpoints does not establish periodic trajectory closure. Save the raw
phase-wise differences; any tolerance result is an additional flag, not a
replacement for those values.

The frozen closure contract uses the global L2/Frobenius norm of a phase-matched
state-array difference. Both bounds must pass: absolute error <= atol **and**
relative error <= rtol, where the reference scale is the larger of the two
state-array norms and 1e-12. FP32 state tolerances are atol=1e-7, rtol=1e-5;
FP64 state tolerances are atol=1e-10, rtol=1e-8. Jacobian-array closure uses
FP32 atol=1e-6, rtol=1e-5 and FP64 atol=1e-10, rtol=1e-8. An additional coordinatewise absolute-plus-relative bound is reported and
required by the helper, alongside both global bounds. The global absolute
bound already implies that coordinatewise bound for finite arrays; it is not
an alternative looser acceptance rule. Report full phase-residual arrays
and their raw magnitudes. R1 has no pair of observed cycles and no closure
estimate; label it unavailable rather than treating an endpoint as another cycle.

Adapter/native equivalence uses elementwise PyTorch comparisons: FP32 atol=1e-6,
rtol=1e-5, FP64 atol=1e-12, rtol=1e-10. Ordered-product/direct-map agreement uses
both global norm bounds: FP32 atol=1e-5, rtol=1e-4, FP64 atol=1e-10, rtol=1e-8.
The same denominator floor is 1e-12. These declared tolerances establish a
numerical check contract, not an empirical trajectory-error guarantee.

Compare closure residual with between-cue state separation where meaningful.
When separation is zero, flag the ratio undefined instead of creating a large
or zero sentinel without explanation. Absolute closure tolerances can greatly
exceed the tiny cue differences observed in the prior phase. Passing them does
not establish robust cue erasure or exact state equality.

A numerical pass is an **approximate closure-compatible observation** within
this window and tolerance. It does not prove an exact periodic orbit, convergence
for all initial conditions, absence of a longer-period response, or periodicity
of the Jacobian sequence. This bounded v1 does not separately certify the latter.
**All tangent products retain the finite-window label**, even when an approximate
closure flag passes. No certified Floquet monodromy or Floquet multiplier is
reported. Failure to close is preserved; do not extend the horizon automatically
until it passes.

## Tangent propagation and numerical checks

Hold the realized body input and weights fixed and compute
J_t = partial F_theta(s_t,u_t)/partial s_t at the first cue-history trajectory.
For column perturbations, the final-cycle product is
Phi = J_(b-1) ... J_a, where b-a=12. Store the actual indices, states, inputs,
Jacobians and product so ordering can be independently recomputed. This is a
local sensitivity to complete-state perturbations, not a cue-input derivative
or a task-readout channel. Derivatives at one history cannot characterize
nonlinear sensitivity throughout the cue distribution.

The saved assay-dtype Jacobians are multiplied both in their original dtype
and with float64 accumulation. Report which product supplies a spectrum: the
implementation uses float64 accumulation of saved Jacobians for both assays,
which does not turn a float32 state/Jacobian replay into float64 dynamics.

Report product eigenvalues (real and imaginary parts), spectral radius,
singular values and largest singular gain separately. Eigenvalue modulus and
singular value answer different questions; a nonnormal product can have
spectral radius below one and finite-window singular gain above one. A small
spectral radius alone does not establish Euclidean contraction, cue-information
loss or monotonically shrinking within-cycle perturbations. Log gain divided
by 12, if reported, is a finite-window rate, not an asymptotic Lyapunov exponent.
Zero singular values require explicit nonfinite-log handling.

Verify full-state derivative shape, chronological multiplication and dense
product recomputation. Compare the ordered twelve-step product with direct
autodifferentiation of that twelve-step map at the same start state and input
sequence; record the frozen absolute/relative agreement test. This checks
chain accumulation and ordering but is not independent verification of the
manual recurrent equations. Native adapter comparisons supply a separate
forward check.

Every float64 tangent window additionally checks a deterministic normalized
state direction, linear in coordinate index from -1 to +1. Save the direction,
autograd JVP, direct-Jacobian direction product and central finite differences
at eps=[1e-4,1e-5,1e-6]. Require each finite difference to match the direct
Jacobian-direction product within both absolute 1e-8 and relative 1e-4 bounds
using the same norm helper; the JVP agreement uses the frozen product tolerances.
There are eight tested FP64 directions per cell, giving 24 central-difference
pairs (48 extra twelve-step forward maps), plus eight JVPs. Float32 production
windows have product/direct-map checks but no finite-difference assay. Local
calibration tests additionally exercise small GRU/LSTM finite differences,
complete cell-state dependence and noncommuting chronological products.

One direction does not validate all derivative directions, and a very small
true directional gain can make relative finite-difference agreement difficult
at the floating-point floor. Preserve such failures; do not loosen a tolerance
or choose an easier direction after inspecting a result. Nonfinite arrays,
failed adapter equivalence or failed derivative checks block numerical
acceptance while preserving diagnostic artifacts. Do not substitute a partial
spectrum or a different cue silently.

## Cue separation and falsifiable predictions

With only two distinct cue histories, their centered covariance has rank at
most one by construction; any normalized nonzero PR is one. It is therefore
not a dimension-collapse estimate. Report raw complete-state difference norms,
per-component h/c norms for LSTM, mean-state scale and absolute two-point
variance only as descriptive paired-history quantities. Keep zero-variance
normalized quantities scientifically undefined, with the implementation's
explicit false flag. Two histories are neither a distributional covariance
sample nor independent trained-model replicates.

Freeze these diagnostic interpretations before inspecting outcomes:

1. **Approximate forced-response synchronization.** If same-phase closure
   residuals and paired-cue separation decrease across repeated forcing,
   preserve that joint observation. Large final-window singular gain, persistent
   history separation or failure to close counters a simple uniformly
   contractive synchronized-response account. A local tangent observation and
   two finite histories cannot prove global synchronization or shared basins.
2. **Precision-sensitive residuals.** If genuine FP64 replay changes the
   magnitude/order of cue differences or closure residuals substantially,
   the original FP32 geometric interpretation needs numerical qualification.
   Similar residuals in both precisions counter the simplest arithmetic-floor
   explanation but do not prove noise-robust information. This assay does not
   retrain a competent model.
3. **Periodicity specificity.** If both arms show comparable finite-window
   contraction and cue attenuation, the observations do not isolate a repeated
   forcing effect. If only one arm does, record its driver/horizon-local
   contrast; order, trajectory and nonlinearity remain potential explanations.
   Never select a panel because its contrast has the preferred sign.
4. **Tangent/behavior dissociation.** Any apparent closure or contraction can
   coexist with failed native query behavior or retained differences between
   histories. Preserve such dissociation. All four inherited acquisition gates
   are false, so further poor behavior cannot by itself identify acquired
   memory lost to repetition. Improved query behavior under numerical replay
   is a numerical diagnostic, not a repaired trained-model acceptance.

No p-value or training-population confidence interval is justified. Two driver
panels, two histories and multiple horizons are repeated diagnostics of one
checkpoint per cell. Report every cell, panel, arm, precision and failure. No
cross-modal conclusion, architecture superiority, causal dimensional mediator,
universal collapse, or negative result about competent recurrent memory follows.

## Artifact acceptance and handoff

A complete handoff preserves the frozen/effective configuration, analysis source
commit and file hashes, training source commit, checkpoint byte hashes,
checkpoint-selected steps, original failed competence, driver/cue arrays,
precision/backend metadata, raw states/Jacobians/products/spectra, numerical
checks, manifest, status, cluster/job identity and scheduler resource record.
Missing/nonfinite/check-failed work is retained with explicit blocked status.
Successful exit, verified artifacts and scientific interpretation are distinct.

Independent review recomputes products and reductions from raw arrays, checks
state/query chronology and FP64 replay, verifies derivative/adapter diagnostics,
and enforces finite-window labels and the two-history sample limit. Check the
actual analysis commit and immutable config against this protocol before launch.
The coordinator owns config/checkpoint bindings and the bounded launch; this
reviewer makes no cluster or model-provider calls.

## Independent prelaunch review

The final source-bound review finds no remaining scientific launch blocker for
this bounded checkpoint-only diagnostic, conditional on the coordinator
committing and deploying the reviewed immutable files. This is readiness to
measure, not scientific acceptance of collapse or a Floquet result. The
[ignored review record](../../../../../runs/ops/periodicity-v1-scientific-prelaunch-review-20261011.json)
contains assertions, limitations and the exact local test output.

Five focused CPU tests passed in 1.635 seconds on the final reviewed source:
complete GRU/LSTM/native query chronology, finite-difference/JVP/full-state
derivatives, noncommuting product order, driver/shape checks and tiny
checkpoint-only artifact/failure integration. These tests exercise calibrated
fixtures, not all four actual GPU checkpoints. Production acceptance still
requires each runtime numerical gate and independent artifact verification.

Reviewed SHA-256 bindings:

- Analysis module: `5ff03bcddc5b0c3810eec8c02b4efbbd86da370567e876eaa150dd446bcc25e8`.
- Focused tests: `81212f2509b4a4f209e9c2e009e67f5f5faa4905d9eb084b08da48b6cef2a70c`.
- Frozen configuration: `ab74585e37dfd06058ed0aec1904b7aaa6d43d5daf70eb3a16ceea2689f97fa0`.

The module/test/config hashes remained unchanged during the final check. The
new analysis commit is assigned by the coordinator after this review; it must
contain these bindings. The reviewer launched no cluster job, trained no model
and made no model-provider call.
