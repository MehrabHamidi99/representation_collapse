# Post-calibration measurement contract — prospective proposal

2026-10-13. **Design only.** No implementation, native forward, checkpoint
loading, label access, fitting, precision change or new execution is admitted.
The coordinator owns implementation, external freezes and scientific acceptance.
This proposal preserves the original failed numerical predicate and the stopped
v5 campaign. It does not retry, resume, extend or rescore that campaign.

## Evidence and what remains unknown

The [independent numerical results review](native-chronology-calibration-v1-independent-results-review-2026-10-13.md)
accepts a complete integrity audit at source
`9cff2d3d2bbaccee1642441b325625969e551f7b`: 372 conditions, 120,000 native
calls, 272.359 engine seconds, zero optimizer updates. Its receipt has SHA-256
`a48031c7c0a32c5691fba45af150c839ff51705df45f300fb0f6318b72b4adc1`.
Strict interoperability is **false**. The independent saved-array reduction
SHA-256 is `f21b877459ca46b5b35fa10a045d93f267ecf8feb0438221622292e1b5d19b43`.

All 38 failed comparisons concern known trained GRU/categorical θ4,000 in FP32:
P2 separately encoded prefix/QUERY logits have 54 violations per repetition,
P3 freshly encoded QUERY from the cached-prefix origin has two, and 34
cross-precision checks have 1,880 aggregate violating entries. The largest
cross-precision discrepancy is P2 logits, `7.857092281682299e-6`.
Primary initialization/null/repeat/chunk checks have no violations; the known
FP64 local checks pass. Entries, repetitions, paths and duplicated arms are
not independent scientific samples. Passing FP64 local checks does not make
FP64 exact truth. Twenty-four origin-before copy assertions lack independently
exported before operands; the review labels that limitation explicitly.

The [v5 stop](query-averaged-acquisition-v5-results-2026-10-13.md) remains an
execution/measurement failure after 4,000 updates: no step-4,000 accepted
evaluation and no fixed step-5,000 endpoint; three cells unstarted. Seven
earlier evaluations passing the original guard failed the complete R0 gate.
Those are valid partial observations, not a selected endpoint. V2/v3 bounded
acquisition failures and v4 singleton successes retain their separate scopes.
Neither a numerical failure nor an incomplete acquisition endpoint establishes
that the architecture cannot retain information.

## Three explicitly different executed maps

Write the computational identity as
`π=(source, θ bytes, task/input layout, dtype, device/backend/build, thread flags,
batch/row order, encoder partition, core partition, cache recipe, head)`.
Finite arithmetic can depend on tensor shape and partition even where the ideal
real-valued recurrence acts independently on rows. Each observable binds π;
different precision or batching is a different executed map, not a silent
implementation detail. Native finite values do not certify information retention.

**B0: canonical behavior.** Preserve the unchanged public `model(inputs)`:
full seventeen-row R0 native forward, including terminal QUERY, FP32, B=64
evaluation chunks and frozen row order/source/backend. No path is selected by
its scores. This is the operational native behavioral estimand for a *new*
prospective acquisition protocol. Logits come from that call; non-mutating core
hooks observe its hidden sequence and returned terminal state. Complete
terminal GRU h or LSTM `(h,c)` is required. The full fused LSTM call does not
expose intermediate c; record it unavailable rather than inventing a trajectory.

**C: complete-state trajectory and cached readout.** For a separately named
state assay, cache `Eθ(X)` from one full-episode encoder call at the declared
batch/length/dtype, then execute **every** WRITE/body/QUERY update as a literal
native one-row core call from zero complete initial state. This is the P4-style
trajectory; h/c are observed at every update. At pre-QUERY state S, define
`RQ(S)=head(core(cached_EQ,S))`. No fresh QUERY encoding is substituted. Keep
the same cache, batch, row order, source, precision, state packing and readout
for baseline, perturbation, zero intervention and sham. Do not concatenate
sham/perturbed examples into a larger batch or normalize each arm differently.
The original cached bulk-sixteen-WRITE prefix (P1) may be a *separate* boundary
diagnostic; it is not interchangeable with the declared one-row C trajectory.

For R0, C has sixteen WRITE steps and one QUERY. A later body assay must declare
the **whole** sequence length/cache recipe and literal one-row execution of
every A/B/X update. Bulk body execution with unobserved intermediate LSTM c
cannot supply those states. Jacobians must be evaluated on the actual declared
C trajectory, never composed on a nominal fused B0 trajectory. No body tangent
or periodicity experiment is admitted by R0 calibration.

**D64: precision sensitivity.** Exactly promote the same stored FP32 θ/input
values to FP64; execute the separately declared B0 or C recipe without training.
Compare matched values and report every discrepancy/failure under the existing
directed bounds. This is a paired numerical sensitivity assay, not FP64 training,
checkpoint repair, an exact reference model or permission to substitute its
predictions for FP32. No precision is selected because its task score is higher.

B0 and C have different purposes. A future B0-only behavioral record must meet
its own frozen identity/finite/repetition and authorized label-oracle checks.
For a C intervention to be interpreted as changing the original task behavior,
its **zero-intervention C readout must match B0 at that record under the original
dtype bound**, and value-preserving sham/copy controls must match the C baseline.
A failure blocks that intervention interpretation; it cannot become memory loss.
Behavioral B0 validity and C-assay validity have separate flags in the *new*
contract. This distinction does not retrospectively change v5, whose combined
guard failed and stopped it. Public fresh-QUERY P3 and the original all-path
interoperability audit remain failed diagnostics, never relabelled passes.

## Minimal new label-free forward checks

Propose eight fixed weight contexts: seed-0 initialization and one predeclared
stored discovery checkpoint for each GRU/LSTM × categorical/continuous cell.
Use known θ4,000 for trained GRU/categorical; use the unchanged v3 **fresh
current step-5,000** checkpoint for the other three cells. The rule chooses
current protocol endpoints/challenge weights, not best scores. Hash-bind every
file/source before loading in any later admitted run. Old failed/partial outcomes
are unchanged and none supplies new eligibility.

Reserve input seed **30019** before generation: 32 independent sixteen-WRITE
histories duplicated into B=64, QUERY `row_id mod 8`, no target arrays. Every
context uses identical task-specific input bytes. FP64 is exact promotion;
both dtypes run twice independently. Required local checks are finite full h/c,
bitwise same-map repetition, B0↔C zero-intervention residuals with B0 as the
reference, fixed-state/cache identity before/after, copy/zero-sham residuals,
and recorded nonzero perturbation norms. The old FP32 `(1e-6,1e-5)` and FP64
`(1e-12,1e-10)` forward bounds stay unchanged. Cross-precision outcomes are
diagnostics; their known failures cannot be forced to pass or used to choose
the primary precision. Export both before/after operands for copy controls.

Freeze one normalized state direction per GRU context and two per LSTM context
(hidden-only and cell-only), with independent direction seed **30029**. Use
±`1e-3` per-row unit-norm raw-state perturbations, preserving complete state
and the same cached QUERY. These are label-free plumbing/resolution probes,
not causal task interventions or a positive-effect requirement. A numerically
indistinguishable perturbation is reported as such, not as absent information.

Exact forward count: eight contexts × two dtypes × two repeats ×
`(one B0 + seventeen C steps)` = **576**; two cached copy/zero-sham QUERY calls
per condition = **64**; twelve state-direction/context combinations × two
dtypes × two repeats × two signs = **96**. Total **736 native-core calls**.
No warmups, extra chunk search, discarded failures or decoder fitting enters.
Cross-chunk portability is not asserted: the canonical chunk is fixed, with
earlier chunk diagnostics retained separately.

## Derivative evidence before tangent claims

Autodiff/JVP differentiates a declared **ideal real-valued surrogate of the
specified native operation graph**. It is not the ordinary derivative of the
piecewise-constant IEEE byte-to-byte execution. Its numerical action must agree
with finite perturbations over a resolvable step range at the chosen θ/state;
forward agreement alone proves neither gradient accuracy nor information loss.

For state derivatives, define `Fθ(S,e)=native_core(e,S)` with complete flattened
state (GRU h; LSTM h then c) and the cached encoded input e held fixed. S is an
independent argument: detaching its historical construction is appropriate for
this **local state partial**. For full parameter gradients, the prefix state
and all encoder/cache operations must instead remain attached to θ. Detaching
h, c or the encoder cache drops genuine parameter-gradient contributions.
Keep the two derivative questions distinct.

Propose these exact, separately frozen calibration stages:

1. **Tiny dense reference:** native hidden width 2/encoder 3, B=1, GRU/LSTM,
   both dtypes, two successive declared inputs WRITE/QUERY: eight primitive
   base cases. Form the full dense state Jacobian from one attached primal and
   one reverse action per output coordinate. For three fixed independent
   direction/cotangent pairs, compare JVP and VJP with dense multiplication,
   check `<w,Jv>=<Jᵀw,v>`, and check central finite differences at four frozen
   ε values. Two-step ordered product is `JQUERY JWRITE`; retain noncommuting,
   singular and nonnormal analytic controls without calling them model findings.
   Cost per base: 1 dense primal +3 JVP +3 VJP +24 central-FD primal calls =31;
   total **248 native calls**, 48 reverse actions.
2. **Actual-width local state actions:** the four fixed trained contexts above,
   both dtypes, two explicitly observed C states (before and after QUERY).
   Probe fixed encoded WRITE/QUERY/A/B/X inputs; A/B/X encodings come from
   declared seventeen-row synthetic templates ending in that type, not task
   labels. There are 80 base cases. Use two state directions for GRU and three
   for LSTM, and independent output cotangents: 200 direction cases. One primal
   per base plus JVP/VJP and eight central-FD calls per direction gives
   **80 +200×10 =2,080 native calls**, 200 reverse actions. These five input
   probes establish finite local compatibility only; actual future body states
   and horizon-specific cached inputs must be recalibrated before body tangents.
3. **Full parameter-gradient plumbing:** the four trained contexts, both
   dtypes, B=32 training geometry. Fix eight QUERY views and independent
   seed-30031 logit cotangents; the scalar proxy is their mean linear output
   projection, with no targets. Compare every encoder/core/head parameter
   gradient of the shared-prefix graph to eight independently recomputed
   **same training-path** branches. Cost 9 shared +16 independent calls per
   base. Check one fixed seed-30037 parameter direction by four central-FD
   pairs, 72 calls per base. Eight bases cost **8×97 =776 native calls**,
   72 reverse actions. No clipping/AdamW/parameter update is performed.

State directions have per-row unit norm in raw h/c coordinates; cotangent
normalization and state ordering are stored. Parameter direction has unit norm
over the complete concatenated parameter vector, with the ordering pinned.
No direction is chosen by a largest observed derivative or a desirable score.
Perturbed parameters are temporary diagnostic arguments; before/after byte
identity of the original θ is checked and no new checkpoint is accepted.

Freeze ε grids before seeing derivatives: FP64 `[1e-3,1e-4,1e-5,1e-6]`, nominal
pair `1e-4/1e-5`; FP32 `[1e-1,3e-2,1e-2,3e-3]`, nominal pair `1e-2/3e-3`.
Raw units/actual represented perturbation are reported. Require both nominal
points to pass, retain the entire error curve, and never choose the best ε
posthoc. Near-zero/unchanged perturbations, saturation and rounding floors are
explicit unresolved cases. No numerical singular-value floor is inserted.

Proposed derivative-only engineering bounds, requiring independent review
before use: AD-versus-dense action and adjoint checks use elementwise FP64
`1e-10 +1e-8*abs(reference)` and FP32 `1e-6 +1e-4*abs(reference)`; central-FD
action/directional-gradient residual is normalized by `max(1,||reference||₂)`
and must be ≤`1e-6` in FP64 and ≤`1e-3` in FP32 at both nominal ε points.
These derivative-unit bounds are **new prospective settings**, not replacements
for the unchanged forward/logit bounds. They do not certify modes below their
resolution, a complete spectrum, leading singular vectors or contraction in
all directions. Failure blocks the derivative claim, not a task verdict.

Unsupported native JVP/VJP/double-backward is a recorded blocking outcome.
No automatic backend switch, hand-coded gate surrogate, dtype change or dense
actual-size Jacobian is allowed. Such a replacement needs its own map/source
definition and independent primal/derivative calibration. Finite-window tangent
products use the complete observed C trajectory in chronological order, inputs
fixed; Floquet terminology additionally requires actual state/Jacobian/input
periodicity and closure. None is established here.

## Proposed finite local budget and subsequent acquisition question

The numerical proposal totals **736 +248 +2,080 +776 =3,840 native-core
invocations**, at most **320 reverse actions** and **224 JVP actions**, with
one shared **600-second** CPU-only clock covering source/input/weight loading,
all stages and finalization. Encoder calls, AD transforms, hashing and reduction
also count toward walltime. Hook-count every attempted primal before execution;
unsupported actions, count/deadline exhaustion or missing conditions yield
incomplete calibration. There are zero GPUs, optimizer updates, providers,
cluster submissions or automatic retries. This is a budget **proposal**, not
admission; independent code/config review and an external freeze must precede it.

A subsequent **new**, separately reviewed acquisition protocol could ask:
at equal initial θ, prefix draws and optimizer-update count, does averaging all
eight QUERY gradients improve original-task acquisition relative to one sampled
QUERY when **both conditions use the same declared training-path graph**?
Keep sixteen ordered WRITEs, query support 0–7, native architecture 128/64,
batch32/16 independent histories, AdamW LR .001/WD .01/clip1, one update after
the normalized loss. Sampled and averaged conditions both use separately encoded
prefix/QUERY training branches (v5-style), preserving original random-selector
RNG draws. B0 scores the endpoints. Its FP32 executed objective need not equal
the full B0 loss exactly; the conditional-mean identity is for the specified
training graph, with real-arithmetic and finite-precision limits explicit.

The smallest possible fit envelope to propose later is two **fresh** GRU/
categorical seed-0 fits, sampled versus averaged, exactly 5,000 updates each,
evaluation every500, 10,000 total updates and600 CPU seconds. Reserve validation
seed **20007** before any new score; the original stratified128 episodes per
coordinate/arm remain. No θ4,000 transfer/resume, earlier best checkpoint,
extra LR or partial-endpoint rescue occurs. Expansion to other cells/seeds
requires a separate declared envelope. Equal prefix/update exposure is not
equal QUERY supervision, compute or walltime; report those counts explicitly.

All eight original categorical accuracies must be ≥.95, or all continuous
MSE/(1/3) scores ≤.05, with the existing four-ULP score-boundary rule. No
average-score substitute, coordinate omission, singleton q15, relaxed gate or
precision-selected result supplies eligibility. Fixed endpoints remain fixed;
partial, numerical-failed, censored and unstarted outcomes remain in the ledger.
R0 success still leaves Gate B R1/R2 competence, independent trained seeds,
causal perturbation/sham evidence and periodicity/body analysis untested.

No option is selected by task performance in this proposal. B0 preserves the
operational behavioral program; C is proposed for explicit complete-state
observability and matched interventions; D64 is a separately labelled sensitivity
assay. Any new combined contract must be accepted and frozen before new fits.
Forward residuals, gradients, tangents, task behavior and information retention
remain distinct estimands. No learned-collapse, capacity-impossibility or
Floquet conclusion is enabled by this design.
