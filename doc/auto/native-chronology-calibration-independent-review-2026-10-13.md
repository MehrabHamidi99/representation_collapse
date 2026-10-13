# Independent review of the prospective native chronology calibration

2026-10-13. This review examines the proposal's mathematics, source paths,
observable states and bounded forward-calibration design. It performs **zero
native model calls, zero checkpoint replay and zero optimizer updates**. It
reads no calibration task-label arrays and accepts no future score, competence,
dimension or Floquet result. The proposal author and reviewer have disjoint
write scopes. Independent v5 failure diagnosis belongs to the previously
completed result review, not to this design review.

## Review verdict and authority boundary

**Accepted for implementation preparation only.** No calibration execution,
source repair, canonical-only contract or model fitting is authorized. The
conceptual distinction between strict numerical interoperability and a
separately specified canonical execution contract is appropriate. The proposed
fixed-weight, label-free forward calibration can support a finite numerical
admission check; it cannot guarantee agreement at future trained parameters.
Source implementation, exact comparison graph, call accounting and external
execution admission remain required before a run. No canonical-only contract
or new fitting envelope is admitted by this document.

During review I requested two concrete corrections: an exact directed
comparison/observability graph and explicit budgeting of the known-weight
regression and numerical nulls. Their final definitions and proposal hashes are
bound in the accompanying review receipt. Reviewed proposal SHA-256 is
`9438ec2626b30b3f85aca2f784ed5d9d16bc7af18ccc6ad6216181d36cd83f32`;
supporting `plan-v2.json` is
`bb3fad3c89118476f247320b1af1f5972334163ee6bf7c5122ec7375deeae551`.
The failed v5 attempt, its original
FP32/FP64 tolerances and its missing endpoints remain immutable.

## Source-path and state-observability audit

I checked the ten source/evidence SHA-256 pins in the initial supporting plan;
all matched. The frozen native model uses pointwise `Linear`/`Tanh`, a
one-layer GRU/LSTM and its original linear head, with no dropout or mathematical
cross-example coupling. `forward(inputs)` encodes the full sequence; public
`read_query(query,state)` freshly encodes the one-QUERY tensor. The adapter
instead slices the QUERY encoding from its full-sequence encoding. The proposal
correctly distinguishes this encoding partition from recurrent-core
partitioning and layout effects. It does not isolate one backend or encoder as
the unique cause of the observed residual.

P0 must be a **single** unmodified canonical `model(inputs)` call whose encoder
output and returned full-core terminal state are observed by non-mutating
hooks. Computing the full core again to obtain its state would be an additional
path and native-core invocation. P1 has two core calls; P2 has two; P3 reuses
P1's complete origin and adds one; P4 has seventeen. Complete h/c copies must
preserve source tensors and ordering. The two repetitions independently repeat
their prescribed operations; they cannot reuse an earlier result as an exact
repetition check.

The proposal's LSTM warning is essential and correct. Native full-sequence
output exposes every hidden output but only the **terminal** cell state. A
separate sixteen-WRITE prefix call exposes its own complete h/c; it does not
observe the inaccessible intermediate c inside the fused seventeen-row call.
P0's intermediate LSTM c therefore has an unavailable marker, not a fabricated
value or inferred agreement. Terminal h/c comparisons across native paths are
available; prefix-cell comparisons among P1/P2/P4 concern their explicitly
different invocation paths. One-step P4 exposes its own per-step h/c without
certifying those cells as observations from P0. GRU has the stronger
intermediate complete-state observability because its state is h.
This also matches the locally pinned Torch 2.6 LSTM output contract: sequence
output contains h at each update, while h_n and c_n are terminal states.

Directed comparison references matter: the original relative allowance scales
with the **reference** value. A path/dtype/chunk matrix without direction is
underdefined. Required terminal comparisons must identify the canonical P0 at
matched chunk/dtype; prefix comparisons must name an observable prefix
reference; cross-chunk and cross-precision edges must keep episode IDs aligned.
Unobservable comparisons are recorded as such and never counted as passes.
No convenient path, chunk or result may be selected after observing residuals.

## Budget and prospective data audit

The primary arithmetic is correct under the stated sharing:

`(2 models × 2 tasks × 2 weight seeds × 2 input streams × 2 dtypes)`
`× 2 repeats × (64+8+4+2+1 chunks) × (1+2+2+1+17 core calls)`
`= 32 × 2 × 79 × 23 = 116,288 native-core invocations`.

A full known GRU/categorical step-4,000 regression over 2,048 rows at chunk 64,
two dtypes and two repetitions costs `32 × 2 × 2 × 23 = 2,944`. A precisely
fixed null schedule of 768 calls uses the remaining budget, giving exactly
120,000 calls. Its zero-head five-path schedule costs 368; its zero-input,
zero-parameter P0/P1/P4 schedule costs 320; its complete-state identity/copy
schedule costs 80. Nulls use fixed seeds/inputs/dtypes/repetitions and no labels.
Detached copies in this forward-only identity null check values; they must not
be described as preserving training gradients.
This ceiling leaves no allowance for another comparison forward,
retry, convenience rerun or uncounted hook call. The 600-second limit is a
ceiling, not a measured runtime prediction. Attempted calls must be counted
before invocation and blocked at exhaustion; a timeout or truncated/null-missing
matrix is incomplete and cannot admit fitting. Atomic artifact finalization may
record actual elapsed time without issuing another native call.
All stages count toward the 600 seconds. A matrix whose finalization exceeds
that admitted deadline is incomplete; retaining its artifacts cannot relabel
it as complete. No prior warmup or hidden core invocation is outside the count.

Input streams 30013/30017 and initialization seeds 0/17 are declared before
execution. The 64 histories per task/stream retain the same rows and input
bytes across chunk sizes and paths. Terminal selector `row_id mod 8` is fixed,
and QUERY payloads are zero. The input WRITE payloads are allowed calibration
inputs; no target extraction, target arrays or competence scoring belong in the
interface. FP64 is exact promotion of the same finite FP32 parameters and
inputs, not an independently sampled initialization or FP64 training.

The already inspected step-4,000 regression is **known numerical discovery
evidence**, not unseen confirmation. A candidate source repair needs a new
hash-bound regression result; the old source's passing or failing comparisons
cannot certify edited code. Any later numerical confirmation must freeze its
new seeds/weights/path matrix before inspection. Neither those future results
nor any new scientific confirmation labels are available in this review.

## Mathematical audit and limits

The floating-point discussion is sound: nonassociativity, tensor partitions,
strides and backend paths can produce different finite results despite
deterministic repetition of each specified path. A small FP64 residual is
supporting numerical evidence, not exact truth or proof of a unique cause.
TF32 is disabled and does not explain this local CPU discrepancy. The proposal
correctly separates bitwise changes from true elementwise tolerance violations.

The argmax bound is correct. For a unique canonical winner k,
`z_k-z_j > epsilon_k+epsilon_j` for all competitors makes every alternative gap
strictly positive. The uniform top-two condition is margin >2epsilon. Equality
and ties remain unresolved. These are conditional certificates given verified
error bounds; an observed finite-grid maximum is not a universal future error
bound. Prediction stability establishes no correctness. With common fixed
labels, the fraction of uncertified rows bounds possible accuracy change, but
this proposal computes no such label-dependent score.

The exact regression error difference is
`2(a-y)delta + delta^2`; its label-free bound for y in [-1,1] is
`2(|a|+1)epsilon + epsilon^2`. Averaging bounds absolute raw-MSE change, and
population normalization by 1/3 multiplies that bound by three. The RMS
Cauchy–Schwarz bound `2 sqrt(M) e + e^2` is also valid when its required canonical
MSE and perturbation RMS are legitimately available. Neither formula replaces
a competence threshold. Paired contrasts require bounds for both constituent
prediction errors. The linear-head triangle bound is correct in exact
arithmetic, with a separate finite head-evaluation residual. It does not bound
future LSTM hidden changes from cell discrepancies.

The author's 205 exact-rational fixture checks have the stated arithmetic
counts: 180 squared-error cases, eight logit corners, one tie counterexample
and sixteen head inequalities. I inspected that source and receipt without
rerunning it. The general derivations, rather than these finite examples,
justify the inequalities. ULP summaries require finite ordered IEEE encodings,
explicit signed-zero/subnormal handling and safe integer subtraction. ULP
distance does not supply a global behavioral noise floor.

Initial-parameter v5 gradient checks remain limited to their initial tensors.
Forward agreement at any weight does not bound parameter-gradient, JVP or VJP
residuals. A capture-only source change must preserve training graph and tensor
partition. Changing the training encoder/core path requires a separately
budgeted, frozen gradient/tangent calibration at declared fixed diagnostic
weights using synthetic cotangents, with no competence labels. It cannot inherit
initial-only parity or spend uncounted calls from this forward-only ceiling.

## Exact conditions for subsequent admission

For **strict interoperability**, reviewed code must implement every frozen
directed comparison and available complete-state observable, preserve source
and input/weight hashes, repeat the specified paths, and pass every required
original-tolerance comparison at the primary grid, known-weight regression and
fixed numerical nulls. Same-condition repetitions must satisfy their declared
bitwise rule. Missing or nonfinite observables, a failing required residual,
count/deadline exhaustion or provenance mismatch blocks that candidate. This
is finite-set numerical admission only; future online guards remain active.
The known **unchanged** old-source step-4,000 path comparison already fails this
contract. Preparation approval is not a claim that this source now passes.

For **canonical-only execution**, a separate new proposal must name the exact
behavioral encoder/core/dtype/batch/row path and all phase/readout definitions;
identify which alternatives are diagnostics rather than interoperability
claims; specify LSTM cell observability and scientifically necessary residual
and sensitivity controls; and preserve matched paths for perturbations and
shams. It must independently justify its required comparisons and exclusions
before new fitting. Dropping `read_query`'s old failing check cannot be presented
as its having passed. Replacing canonical predictions, selecting a numerically
convenient chunk, or using a path selected for desirable scores requires an
explicit newly reviewed scientific execution contract. No such contract is
automatically admitted here.

Both routes require reviewed implementation, clean committed source, frozen
configuration and externally authorized numerical execution. Subsequent fitting
requires completed calibration and its independent review, a fresh immutable
fit source/config/receipt, and a separately admitted bounded fit envelope. The
old v5 run stays failed/censored with no endpoint rescoring, resumption or
posthoc competence. Periodic-body competence, causal memory, covariance/tangent
dimension, finite-time/Floquet interpretation and scientific confirmation stay
separate project gates.
