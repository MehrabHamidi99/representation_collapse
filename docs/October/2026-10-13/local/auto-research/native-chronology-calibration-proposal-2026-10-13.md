# Native chronology calibration — prospective numerical protocol

2026-10-13. **Preparation only.** This document admits no model fitting,
resumption, retry, checkpoint selection, competence rescoring, new tolerance,
dataset export, model-service request or cluster operation. The v5 execution
failure remains immutable. Producer and independent reviewer own its raw-array
reduction and current-weight replay; this author read source and failure
metadata only and did not duplicate that replay. The ignored supporting directory
is `runs/native-chronology-calibration-proposal/`.

## Trigger and precise evidence boundary

The separately frozen [v5 protocol](../../../2026-10-26/local/auto-research/query-averaged-acquisition-v5.md) stopped
during GRU/categorical validation after **4,000 returned optimizer updates**.
The traceback identifies `native_read_query_vs_forked_logits` in
`native_state.native_phase_capture`, with maximum absolute discrepancy
`1.4491379261016846e-6` in its first failing 64-row/16-logit chunk. The unchanged
FP32 rule is elementwise
`abs(candidate-reference) <= 1e-6 + 1e-5*abs(reference)`.
The reported 744 changed entries out of 1,024 are **bitwise differences**, not
744 tolerance violations. The maximum absolute discrepancy alone cannot locate
the violating entry or its allowed relative contribution.

Source: `f766df5e792c8a6dedaa3ea17f2c92fd630f067d`. Stable ignored evidence:
`runs/query-acquisition-v5-20261013/experiment/status.json`, `exit.json`,
`launch.json` and `stderr.log`. Metadata records zero completed fits, three
unstarted cells, unknown endpoint competence and a finite censored current
checkpoint at step 4,000. Seven scheduled raw validations exist at steps
500–3,500; no completed step-4,000 validation or step-5,000 endpoint is implied.
The saved current checkpoint is diagnostic, never a selected endpoint.

Exact SHA-256 pins checked here:

- Frozen native capture: `eeadc7a7332045c6209c6b823af2ec35740dbe3eb662dc0de25755e82041fdeb`.
- Native model: `0aa1c337ffd0ab576056e70b4ecb0dfeee073c1d3453e87389447f77f1dfbbd3`.
- Executing acquisition module: `f2af2c73088731e3664b7ca59c54ed9c9f083dc863b83881b20956ffb429a19d`.
- Failure status: `572cdcba7a9458ae4142cbb258a559f5c35a27c70d0efc36558225f99be4785c`.
- Traceback: `73555fcdc5eb2ccaa62b8be1f1f4e7f5dfe39d463ce43d2105b5f64711dcc5c2`.
- Censored current checkpoint: `b57b645f80a8e85a1743385e01df8075706904e122a57923b6216310365f3788`.

The separate independent label-free replay receipt
`runs/review/query-acquisition-v5-results/independent-raw-reduction.json`
(SHA-256 `3440a8d1b35dd7e310c3d691dd472a082c1353b068711bff932ebb4487309d11`)
confirms the producer's full 2,048-row diagnosis: QUERY-only versus full-episode
encoding differs by at most `5.960464477539063e-8`; read-versus-cached logits
differ by at most `1.9073486328125e-6`, with exactly two true tolerance
violations among 32,768 entries. Its 21,002 bitwise differences remain a
different count. The first failing chunk starts at row 960 and contains one
true violation. Cached-fork versus full native 17-update logits and complete
terminal state are bitwise equal in that replay. FP64 read/fork discrepancy is
at most `2.6645352591003757e-15`, with no violations and bitwise-equal QUERY
encodings. These are the other reviewer's checks, not a replay by this author;
no targets were scored. They establish path-dependent residuals at the pinned
weights/inputs, not a uniquely isolated encoder/backend cause, score,
dimension, memory loss or scientific mechanism.
The two full-manifest violations are logit column 4 at rows 1,018/2,042:
paired copies of history ID 1,018, not two independent numerical failures.

## Why equivalent real-arithmetic paths can differ

The current model's pointwise encoder is `Linear(38,64)` followed by `Tanh`;
the recurrent core is one-layer native GRU/LSTM; the head is unchanged `Linear`.
There is no dropout, batch normalization or cross-example mathematical coupling.
Nonetheless, IEEE floating-point addition is not associative. Tensor shape,
strides, batching, operation fusion and library/backend choices can alter
reduction/rounding order. Deterministic algorithms constrain repetition of a
specified computation; they do not require bitwise agreement between different
tensor shapes, batch sizes, dtypes or invocation partitions. TF32 is disabled
here and cannot explain this local CPU failure.

`native_phase_capture` first encodes the full `[B,17,38]` tensor and slices its
cached encoding into sixteen-WRITE and one-QUERY recurrent calls. It also makes
an uninterrupted full native call and invokes `model(inputs)` independently.
Its `read_query` check uses the **same copied complete pre-QUERY state**, but
the public method freshly encodes `[B,1,38]` rather than reusing the last row of
the `[B,17,64]` encoding. A different linear-kernel path, `Tanh` rounding or
layout could therefore perturb the query encoding, with the nonlinear recurrent
transition/head magnifying that difference. This is a source-level hypothesis,
not proof that a particular BLAS/fusion implementation caused the failure.

Other independent axes require isolation: encoding the prefix separately as
`[B,16,38]`; running the core once for 17 rows versus 16+1 or seventeen single
rows; materializing contiguous copies versus strided slices; and evaluating
one example versus B=8/16/32/64. FP64 can reduce rounding but also select a
different kernel; a small FP64 residual is numerical support, not exact truth
or proof that the FP32 task is correct. An FP32 checkpoint converted to FP64
represents its same finite parameter values more precisely; it is not FP64
training or a replacement checkpoint.

## Canonical prediction and complete-state contract

The existing v5 behavioral reference stays **uninterrupted `model(inputs)`**,
including all sixteen WRITEs and the terminal QUERY, in FP32 with fixed
64-row evaluation chunks, fixed row order and original source/backend flags.
It cannot be replaced with the numerically most agreeable fork or dtype.
No labels enter the proposed calibration interface or comparisons.

For GRU, the native sequence output exposes `h` at every update. The canonical
full-core terminal state is its returned `h`; compare phase-prefix `h` against
the full-run output at the same time. For LSTM, the native sequence output
exposes hidden state but **not intermediate cell state**. The full native call
returns complete terminal `(h,c)`. A separate native 16-row prefix call exposes
its own complete post-WRITE `(h,c)`, but its `c` cannot be asserted bitwise equal
to an unexposed intermediate cell inside a fused full 17-row call. Record that
phase probe's distinct invocation path. Never infer cell agreement from hidden
agreement or fabricate intermediate cells from hidden outputs.

Define recorded phase provenance explicitly: post-WRITE/pre-QUERY is the
complete returned native prefix state; terminal behavioral logits and terminal
complete state come from the uninterrupted native full call. The segmented
continuation is a comparison path, with residuals for **both** h and c. This
is not automatic certification of one simultaneously observed fused trajectory.
For a later tangent/causal experiment requiring every intermediate complete
state, an alternative native one-step chronology can expose `(h,c)` each step;
that is a separately named numerical execution contract requiring independent
validation and prospective source/protocol freeze. It cannot rewrite v5.

For any later state intervention/readout comparison, preserve exactly the same
cached encoded QUERY, batch shape, row order, complete-state packing and native
core/head path for perturbed and sham states. If that contract uses cached
encoding rather than the public QUERY-only method, name it explicitly. Changing
encoder shape between intervention arms introduces a second numerical change;
small input-encoding error is not automatically bounded by a state-perturbation
norm. This proposal executes no intervention.

## Proposed label-independent probe matrix

First consume the independent existing failure diagnosis without repeating its
current-weight replay. Keep step-4,000 weights/inputs as an explicitly **known
discovery numerical regression**; they are not unseen confirmation data. A
source repair must pass that regression without scoring its targets or choosing
weights. It cannot transform the old censored execution into a complete fit.

A new calibration run, separately admitted after source review, uses two fixed
synthetic input streams, sampling seeds **30013/30017**, and initialization seeds
**0/17**, all declared before execution. Use native GRU/LSTM × categorical/
continuous architectures 128/64. Generate 64 independent sixteen-WRITE prefixes
per task/stream, terminal addresses `row_id mod 8`; all QUERY payload channels
are zero. Do not construct, pass or persist target/competence arrays. The
calibration's synthetic WRITE values are inputs, not held-out task labels.
Parameter tensors are generated once in FP32, hashed, then exactly promoted
for matched FP64 evaluation; input values are promoted identically. No training,
gradient clipping or optimizer update occurs.

Preserve identical episode IDs and input bytes across chunk sizes **1/8/16/32/64**,
dtypes and paths. Within each chunk compare these fixed alternatives:

1. P0: ordinary full `model(inputs)`/full native core, canonical path.
2. P1: full-episode encoding cached once; native core split 16+1 using its last
   cached encoded row and copied complete prefix state.
3. P2: separately encoded prefix and QUERY, core split 16+1; this isolates the
   actual training/public-method partition from P1.
4. P3: reuse P1's **identical complete prefix state**, changing only the QUERY
   encoding from cached full-episode row to QUERY-only encoder call; this
   isolates query-encoder arithmetic from prefix-state differences.
5. P4: cache the same full encoding and use seventeen one-row native core calls,
   preserving complete state after each update; this probes core partitioning.

P0 obtains its encoded tensor, full-core hidden sequence and **returned terminal
h/c** through non-mutating encoder/core forward hooks on its single
`model(inputs)` call; hooks never add a forward, replace an output or mutate
state. P1 and P4 reuse that exact P0 encoded tensor. P3 reuses P1's complete
post-WRITE state and performs only its fresh QUERY encoding plus one core call;
it never recomputes a prefix. P2 independently encodes/runs its prefix and QUERY.
Per chunk the core-call schedule is therefore P0=1, P1=2, P2=2, P3=1, P4=17.
Each repeat recomputes P0–P4 from the pinned θ/input; no encoded tensor or prefix
state crosses repeats. Hooks count and capture an invocation, never create one.

The following **directed candidate → reference** graph fixes the operand used
in `atol + rtol*abs(reference)`; there is no blanket all-pairs state comparison:

- Terminal logits and complete terminal state: each P1/P2/P3/P4 → P0, same
  θ/input/chunk/dtype. Both GRU h and LSTM h/c are required.
- Encoder outputs: P2's prefix → P0's cached prefix; P2/P3 QUERY → P0's cached
  QUERY. P1/P4 encoding reuse is an identity/hash check, not a fresh encoder
  execution. The comparison establishes residuals, not operation-specific cause.
- Complete state after WRITE 16: P2/P4 → P1. P3's input complete state → P1
  must match bitwise. P1's hidden state → P0's hidden sequence at index 15 is
  separately required for either recurrent architecture.
- **P0 intermediate LSTM c is unobservable**: record null/unavailable with
  reason, never synthesize, fill with P1's cell or make a comparison involving
  it. P0's terminal c is observable and remains required.
- Cross-chunk: each available recorded field at B=1/8/16/32 → **the same path's
  B=64 field**, after restoring original row IDs; compare only matching phases
  and observed components. No comparison changes the recurrent path.
- Cross-dtype: each FP32 observed field → **the same path/chunk's FP64 field**
  with exactly promoted θ/input. Difference/threshold arithmetic is computed
  in FP64 using that FP64 reference. This uses the FP32 `(1e-6,1e-5)` bound;
  all same-dtype edges use their own dtype's unchanged bound.
- Repeat 2 → repeat 1 at otherwise identical configuration must be bitwise
  equal. Finite-value signed-zero equivalence is reported separately and never
  silently replaces the bitwise repeat requirement.

P4 complete prefix/intermediate cells are its own observable native one-step
states. Their availability does not make P0 fused intermediate cells observable.
An edge with an unobserved required operand fails the contract; only the
explicitly unobservable P0 intermediate-cell fields above are excluded, with
the exclusion frozen in advance. All terminal states remain required.

Record encoded prefix/query, native terminal logits/states, complete phase
states, tensor dtype/shape/stride/contiguity, model/source/input/weight hashes,
Torch/Python/NumPy versions, CPU architecture, backend availability/enabled
flags, thread counts, deterministic flags, matmul precision and actual execution
order. Treat layout as a recorded axis; any extra contiguous-copy experiment
is secondary discovery and cannot silently enlarge this frozen primary grid.
Record two exact repetitions of every condition. No row permutation is used in
the primary grid. Missing/nonfinite/truncated conditions remain failures or
censored unknowns; never drop them to obtain a smaller passing matrix.

Proposed calibration ceiling: one local CPU thread, zero GPU/provider/cluster
operations, **600 seconds and 120,000 native-core invocations maximum**. With
the defined sharing, 32 model/task/weight/input/dtype conditions × two repeats
× `(64+8+4+2+1)` chunks × `(1+2+2+1+17)` core calls = **116,288** calls; small
numerical nulls and the known discovery regression have the following exact
remaining schedule:

- Repaired-source **known θ4,000 regression**: the pinned 2,048 GRU/categorical
  input rows, B=64 only, both dtypes, two independently recomputed repeats,
  the same five paths/graph: `32 chunks * 2 dtypes * 2 repeats * 23 = 2,944`
  core calls. Read only input/row-ID fields, never targets or scores. All earlier
  producer/reviewer replay remains immutable. This future regression is within
  this calibration envelope, not authority for new execution here.
- N1 zero-head null: all four model/task cells, θ initialized with seed 0;
  set only head weight **and bias** exactly to zero. Use the first 32 primary
  seed-30013 input histories duplicated into 64 rows, preserving selectors,
  both dtypes, B=64, two repeats, all five paths. Cost
  `4 cells * 2 dtypes * 2 repeats * 23 = 368`. All logits must be zero; both
  h/c and duplicate-row residuals still use the directed graph. This categorical
  tie is deliberate and supplies no unique-argmax certificate.
- N2 zero-recurrence null: all four cells, **all parameters exactly zero** and
  zero initial state; fixed all-zero `[64,17,38]` inputs, both dtypes, B=64,
  two repeats, P0/P1/P4 only. Cost `4 * 2 * 2 * (1+2+17) = 320`. All states
  and logits must be zero; compare observed fields and repeats as above. This
  algebraic native-input null is intentionally outside the semantic WRITE
  renderer and never enters a task scorer or semantic input validator.
- N3 complete-state-copy null: all four cells, unmodified seed-0 parameters,
  N1's same duplicated input rows, both dtypes, B=64, two repeats. Cache full
  encoding, run the 16-row prefix **once**, then four one-QUERY continuations
  using original state, detached value-preserving clone, twice-cloned state
  and explicit contiguous value-preserving clone, always with the same cached
  QUERY and unchanged head. Cost `4 * 2 * 2 * (1+4) = 80`. Input state values,
  component hashes and before/after immutability must match bitwise; compare
  each copied-state terminal/logit result → original-state result under its
  dtype bound. LSTM copies preserve both h and c. No gradients are computed.

Total: **116,288 + 2,944 + 368 + 320 + 80 = 120,000** core calls exactly.
Source/hash checks, hooks, metric reduction and clones consume no core calls.
Any extra native call, profiler warmup, repair retry or enlarged layout probe
is outside this fixed schedule; it cannot borrow unseen budget. Actual calls
are counted at the core invocation boundary, and all stages share one 600-second
clock from preparation through finalization. Preparation/data loading count
toward walltime. Identity/nonfinite failure stops; a residual failure is retained
and the finite remaining diagnostic matrix may complete within the same ceiling,
without any training or failed-cell replacement.
Deadline/call exhaustion means incomplete calibration and no training admission.
This is a proposed calibration budget, not authorization or a completed run.

## Residuals, ULPs and numerical nulls

For every array entry, retain absolute error, signed difference, relative error
where the reference is nonzero, and the exact frozen threshold ratio
`abs(delta)/(atol + rtol*abs(reference))`. Report maximum and quantiles **plus
the count and episode/component IDs of true threshold violations**. Maxima alone
are insufficient. At zero reference label relative error undefined; report
absolute error instead. Summaries separate encoder, h, c and logits, task/query,
path, dtype, chunk and repeat. Paired R0 copies, components and kernel repeats
are numerical checks, not independent scientific samples.

Report ULP distance as the integer distance between monotonically ordered
finite IEEE encodings, with the signed-zero policy explicit and ±0 assigned
distance zero. Compute integer differences without unsigned overflow; NaNs,
infinities and unavailable subnormal handling are explicit failures/unknowns.
Also report local `nextafter` spacing if useful. ULP count is sensitive to
magnitude/subnormals and cannot by itself supply a global behavioral error floor
or justify increasing absolute tolerance.

Numerical nulls: exact repetition of P0 at a fixed chunk/source/θ; identity
copies of complete fork state; fixed zero head (all logits exactly its bias);
duplicate rows kept together as a local consistency check; zero-input/zero-state
native recurrence with explicitly declared parameters; and FP64 versions of the
same finite θ/input. A zero head cannot validate recurrent state, so still
compare h/c. A categorical all-logit tie is an intentional margin-null control,
not a competence example. Native path disagreements and nondeterministic
repeat differences remain separate diagnostics.

## What logit/prediction residuals can bound

Let canonical logits be `z` and alternative logits `z+δ`, with independently
verified coordinate bounds `abs(δj) <= εj`. If canonical winner k is unique,
then its argmax is preserved whenever, for every j≠k,

`zk-zj > εk+εj`.

Indeed the alternative gap is at least `zk-zj-εk-εj`, which is positive.
With uniform ε, a top-two margin strictly greater than `2ε` is sufficient.
Use strict inequalities: overlapping intervals and ties are unresolved even
if one observed alternative happens to predict identically. This certificate
uses only logits/error bounds, not targets. Stable predicted class does not
establish correctness or competence. If comparison against a common fixed
label set were later authorized, at most the fraction of uncertified rows
could change accuracy under the bounds; no such score is computed here.

For scalar regression prediction a, alternative `a+δ`, target y and
`abs(δ)<=ε`, the exact raw squared-error difference is

`(a+δ-y)^2-(a-y)^2 = 2(a-y)δ + δ^2`.

Therefore its absolute value is bounded by `2*abs(a-y)*ε + ε^2`, not ε alone.
Since this task declares y∈[-1,1], a **label-independent** bound is
`2*(abs(a)+1)*ε + ε^2`. Averaging these per-row bounds bounds the absolute raw
MSE difference. Normalization by fixed population variance `1/3` multiplies
the bound by three. A conditional alternative using authorized canonical
MSE M and RMS prediction perturbation e is `2*sqrt(M)*e + e^2`; it is not
evaluated in this proposal. Neither formula relaxes a competence gate or
establishes that any model is near/past it. A future paired error contrast
would require summing the applicable numerical bounds for its constituent
prediction paths; adapter agreement is not a periodicity/collapse estimator.

For the unchanged linear head, in exact arithmetic
`abs(Δzj) <= Σr abs(Wjr)*abs(Δhr)`. Finite head-evaluation error adds its own
residual term. A small hidden discrepancy need not imply small logits when
head weights are large; LSTM cell differences can also alter a subsequent
hidden transition while being invisible to the current head. These are
mechanistic numerical sensitivities, not evidence of lost task information.

## Prefreeze decisions before any new fit

The inherited FP32 `(atol,rtol)=(1e-6,1e-5)` and FP64 `(1e-12,1e-10)` rules
remain unchanged. A primary calibration is operationally accepted only when
all pinned identities/conditions exist, finite complete states are retained,
same-condition repeats are bitwise equal, and **every** required
encoder/path/full-state/logit comparison passes its appropriate frozen bound.
Cross-chunk/cross-precision comparisons use FP32 bounds when either operand is
FP32; comparisons within FP64 use FP64 bounds. A failed condition blocks this
candidate contract. This is a finite calibration-set check, not a universal
bound over all future trained weights/inputs; future guards must still stop.

The prior tiny/actual-size v5 gradient checks used **initial** parameter tensors.
The real-arithmetic conditional gradient identity holds pointwise at smooth
parameters, but those checks do not certify floating-point gradient agreement
at trained θ4,000 or along the optimization trajectory. A forward residual
bound does not bound parameter-derivative residuals. A capture-only path repair
must leave the training computation unchanged. If a new proposal changes the
training encoder/core partition, it additionally needs separately frozen
gradient/JVP/VJP or finite-difference checks at declared fixed diagnostic
weights, with synthetic output cotangents rather than competence labels and
its own bounded calibration scope. It cannot inherit initial-only parity or
silently add this work to the 120,000-call forward matrix.

Prefer an independently reviewed path repair that makes the compared QUERY
encoder operation/source/layout identical, while leaving canonical predictions
unchanged. Merely bypassing `read_query`, dropping its failed check or selecting
a smaller convenient chunk is not acceptance of the public method. If the
public method and cached-encoding adapter intentionally define distinct
numerical paths, label and retain both, test their full residuals and specify
which owns each measurement. Any new distinction/source contract requires
review and cannot claim the old adapter check passed.

There are two distinguishable future choices. An **interoperability contract**
claims agreement among all primary paths and must pass the full strict matrix
above. A **canonical execution contract** instead fixes one explicitly named
encoder/core/batch path for behavior and matched state readouts, while treating
other paths as measured diagnostic alternatives with no equivalence claim.
It may be scientifically defensible when different finite-precision programs
do not agree under the interoperability bound, but this proposal does not admit
it automatically. Its required comparisons, exclusions, state observability,
training/evaluation gradient limits and scientific sensitivity bounds must be
independently reviewed and frozen as a new protocol before fitting. It cannot
retroactively remove v5's failed public-method check or use a path chosen for
better scores. A per-row native recurrence or all-FP64 protocol similarly changes
the numerical contract and requires separate compute/source admission.

If these fixed bounds prove inappropriate, stop and propose a **new numerical
protocol** whose bound has an explicit backward/forward-error rationale and
predeclared behavioral sensitivity requirements. The single observed 1.449e-6
failure, a passing maximum multiplied by a convenient factor or desirable
competence scores cannot justify a replacement tolerance. Separate exploratory
kernel/magnitude diagnosis from unseen label-free numerical confirmation and
from held-out scientific confirmation. Reserve any new numerical confirmation
seed/weights/path matrix before examining it; do not call the already-used
step-4,000 regression unseen evidence.

Admission requires independent accepted failure diagnosis, exact reviewed source
and path contract, complete calibration, all acceptance/stop rules frozen before
new fitting, fresh immutable source/config/receipt and a separately authorized
new fit envelope. V5 stays failed/censored; no resumption or posthoc acceptance.
Independent seeds, periodic body competence, finite-time/Floquet scope and
scientific inference remain separate project gates.

Preparation additionally checked the displayed argmax, raw-MSE and linear-head
inequalities with exact rational arithmetic: 180 scalar squared-error cases,
eight certified logit-interval corners, one equality/tie counterexample and
sixteen linear-head triangle-inequality cases. These **205 analytic fixture
checks** use no native model, task dataset, checkpoint replay or competence
scores. Their ignored script/receipt are `check_bounds.py` and
`analytic-checks.json` in the supporting directory. They check finite examples
of the algebra; the mathematical derivations above supply the general argument.
