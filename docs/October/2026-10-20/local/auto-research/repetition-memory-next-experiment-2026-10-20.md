# Candidate next experiment: one-bit retention under matched repeated bodies

2026-10-20. **Design candidate only: not preregistered, implemented, accepted for
training, or an empirical result.** This document proposes the smallest useful
memory experiment following the current fixed-checkpoint graph dwell sweep.
It does not modify previous protocols or their failed outcomes.

## Why change the acquisition task?

The [v2 discovery grid](../../../2026-10-13/local/auto-research/local-competence-grid-2026-10-13.md) produced no eligible
checkpoint in twelve fits. Ten learned immediate acquisition but failed the
sixteen-WRITE, zero-body task. Longer [v3 acquisition](../../../2026-10-13/local/auto-research/prefix-acquisition-v3-results-2026-10-13.md)
also failed full-coordinate competence. [V4 singleton tasks](../../../2026-10-13/local/auto-research/address-acquisition-v4-results-2026-10-13.md)
succeeded at addresses 7 and 15 but failed at address 0. These distinct weights
cannot be assembled into an eight-item memory model. The later
[cached-forward audit](../../../2026-10-14/local/auto-research/cached-forward-v2-results-independent-review-2026-10-14.md)
accepted numerical evidence integrity while retaining four failed cross-dtype
checks; it supplied no competent original-task checkpoint.

Accordingly, do not repeat the sixteen-write discovery grid with another
learning rate. Remove write interference and address selection initially:
write one independently sampled binary value once, then require its recovery.
The [graph dwell protocol](repetition-dwell-protocol-2026-10-20.md) changes
prediction timing for a model trained at dwell32; it cannot itself distinguish
loss of required history from mistimed next-graph prediction. This task can.

## Exact task and smallest model

An episode is **WRITE(bit), body, QUERY**. The bit is balanced Bernoulli and
independent of the complete body, length, motif and arm. WRITE carries a type
indicator and signed value −1/+1. Every other observation has value zero.
The five type channels are WRITE, A, B, X, QUERY: six input channels total.
There is no clock, period, arm, seed or target channel, and QUERY always requests
the written bit. There are no other writes or nuisance-memory coordinates.

Start with one native one-layer LSTM, input6/hidden32, a linear hidden-to-two-logit
head, no encoder/dropout/normalization. Complete state is hidden32 plus cell32.
Use one explicitly defined stepwise native LSTM execution for training,
validation, capture, interventions and QUERY; do not substitute differently
partitioned fused execution and assume equivalence. A GRU is a subsequent
replication, not required to make this first task meaningful.

Reuse the five twelve-symbol motifs and validated paired-body construction in
[experimental design v2](../../../2026-10-26/local/auto-research/experimental-design-v2.md), with a new renderer for the
single WRITE. Periodic bodies repeat one motif R times; controls vary motif
identities, preserve the final motif, and reject proper whole-body periods.
Pairs share bit, QUERY, length and all nuisance randomization. As already
registered, symbol counts, bigrams, runs and endpoint context match, while
higher-order ordering, distinct-motif exposure and predictability can differ.
Call the contrast the effect of this specified ordering intervention, not an
isolated universal effect of periodicity.

## Acquisition and selection candidate

Use CPU FP32, deterministic algorithms, one thread, Adam at 0.001, zero weight
decay, batch64, clip global gradient norm1, full-episode backpropagation and
cross-entropy only at QUERY. Draw 32 new cue/body units per batch and render both
arms; balance the two cue values. No supervision of covariance or contraction.

A single discovery seed0 runs these fixed stages, without rescue or extensions:

1. 1,000 updates at R0 (no body), with independent validation at update1,000.
2. 2,000 updates cycling R0/R1/R2, then a boundary validation.
3. 2,000 updates cycling R0/R1/R2/R4/R8, then final validation.

Each stage boundary must achieve at least 95% native accuracy separately for
both cue values at every already introduced short horizon R0/R1/R2 and both
arms. Gate only horizons introduced by that stage. Use 1,024 independent cue/body
units per horizon, balanced by cue, drawn and hashed before training. R0/R1
identical arms count once. No geometry or periodic-control difference selects
weights. The final checkpoint, rather than a retrospective best checkpoint,
is the candidate; failing a boundary ends that fit and remains visible.
Long-horizon accuracy is reported but is not an eligibility gate.

If seed0 passes and engineering checks pass, freeze this configuration and run
five fresh confirmation seeds1–5. Do not tune on their outcomes or substitute
successful seeds for failed ones. If discovery fails, acquisition remains the
next problem and the confirmation campaign does not start.

## Frozen-weight evaluation and controls to freeze

The primary grid is q=1, R={2,4,8,16,32}, hence body horizons24–384 updates.
Use 2,048 fresh independent cue/body units per horizon, balanced cues, paired
arms, unseen validation/test RNG streams. Assign motif identities and controls
independently of cue. Report both-arm absolute accuracy, balanced cross-entropy,
and paired error difference (periodic minus control); R2 and R16 contrast and
its difference-in-differences remain separately named estimands. A positive
DID alone is insufficient if excess error at R16 is absent.

A secondary dwell grid holds body horizon384 fixed: q={1,2,4},
R={32,16,8}. This changes run duration and number of motif repeats together,
while matching total recurrent updates within and across q. It estimates a
specified dwell intervention, not repetition count independently of duration.
Keep this grid descriptive unless multiplicity and minimum meaningful effects
are separately frozen. All horizons include the same WRITE and QUERY updates.

Controls: R0 acquisition; identical-arm R1 rendering; explicit one-bit register
oracle; constant and shuffled-label readouts; all-X body at the same horizon as
an unmatched elapsed-time diagnostic; both cue polarities for every fixed
forcing schedule. QUERY carries no timing-dependent class target, so ordinary
next-symbol phase prediction cannot solve it. Equal-horizon matched arms remove
elapsed time as the explanation for their difference; shared deterioration in
both arms remains generic forgetting.

## What distinguishes information from its use?

Capture complete state after WRITE and immediately before QUERY. At fixed body,
compare opposite-cue states, native QUERY margins and finite differences.
Across independent bodies fit separate train-only linear probes and a small
predeclared nonlinear probe; evaluate on unseen bodies. Probe failure is not
proof that information is absent; successful decoding with poor native accuracy
instead implicates access/readout. One bit gives at most two cue-conditioned
means, so their centered rank is at most one: it cannot establish a broad
capacity collapse from low dimension.

For same-body opposite-cue pairs, exchange the **entire H+C state** immediately
before QUERY. Correct donor-label recovery confirms causal dependence on the
retained state under the native readout. Include exact self-copy and same-cue
independent-copy shams, untouched replay, and a zero-state corruption control.
Whole-state exchange does not identify a specific memory subspace. Any later
partial-state rescue must freeze its train-fitted direction and norm-matched
shams separately. Record FP32 cue-separation scale; assay declared perturbation
scales without treating FP64 as empirical ground truth.

## Inference and capacity boundary

A trained seed is the replication unit; a paired cue/body draw is the
within-seed unit. Time steps, repeated cycles, arm copies and cue flips sharing
a body are dependent. Report per-seed curves, competence failures and paired
within-seed uncertainty. Five seeds permit a small replication study, not
precise population equivalence. Before confirmation, freeze the primary
contrast, seed-level analysis, multiplicity, minimum effect and power rationale;
these are deliberately unresolved here.

Only after reliable one-bit retention should capacity be tested by independently
varying load K={1,2,4,8}, using one vector WRITE of K independent bits and an
unpredictable terminal coordinate query. Require coordinate-wise acquisition
competence and report the largest supported load at a stated accuracy/horizon;
that is operational task capacity, not Shannon capacity. Sequential writes,
graph-native memory and modality transfer are later tasks with separate gates.
Preserved memory or a valid null is an acceptable outcome throughout.
