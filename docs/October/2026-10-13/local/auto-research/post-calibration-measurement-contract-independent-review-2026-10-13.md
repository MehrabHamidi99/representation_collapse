# Post-calibration measurement contract — partial independent review

2026-10-13. **The saved draft is not admitted.** This is a static scientific
review, with no implementation, native forward, checkpoint loading, target
access, fitting, provider request, cluster operation or Git action. The author
encountered a second stream disconnection before a final hash/handoff arrived.
The coordinator instructed no automatic retry loop. This review therefore
binds the saved draft snapshot and returns concrete revisions; it does not
pretend the author completed its handoff.

The reviewed draft is
[post-calibration-measurement-contract-proposal-2026-10-13.md](post-calibration-measurement-contract-proposal-2026-10-13.md).
Its exact snapshot SHA-256, review artifacts and verdict are recorded in
`runs/review/post-calibration-measurement-contract-independent-review/review.json`.
The accepted numerical integrity receipt remains
`a48031c7c0a32c5691fba45af150c839ff51705df45f300fb0f6318b72b4adc1`;
strict interoperability and new trained science remain unaccepted.

## Scientifically defensible parts of the draft

The proposal correctly treats B0 uninterrupted native behavior, C literal
one-row complete-state trajectory, and D64 fixed-weight promotion as different
executed maps. Choosing C for explicit state observability can be defensible
prospectively, with its whole WRITE/body/QUERY chronology, encoder cache,
batch/layout/precision and readout fixed. It cannot relabel the old failed
interoperability check. FP64 replay is a sensitivity assay, not exact truth,
FP64 training or proof of the FP32 task.

The draft preserves full LSTM h+c, explicitly excludes the unobserved fused
intermediate cell, holds the same cached QUERY and native path for sham and
perturbation, and distinguishes local state derivatives from full parameter
gradients. It correctly names autodiff's real-valued surrogate rather than
the IEEE byte map. Dense tiny checks, adjoint/JVP/VJP checks, fixed central-FD
grids, unsupported-backend stops and recalibration at actual future body states
are appropriate requirements. Their finite tested directions cannot certify
all near-null modes or an entire spectrum.

Original query support 0–7, all-eight coordinate gates, normalized continuous
MSE with variance 1/3, and later R1/R2 eligibility are retained. A new numerical
path cannot restore the old v5 fixed endpoints: θ4,000 remains a diagnostic
checkpoint, fixed θ5,000 is unknown and three cells are unstarted. The proposed
fresh sampled-versus-averaged acquisition comparison is an additional future
protocol, not an execution envelope accepted by this review.

## Revisions needed before design admission

**1. Baseline output agreement is not a causal bridge to B0.** The draft
requires the zero-intervention C readout to match B0 before interpreting an
intervention as changing original task behavior. That is a necessary alignment
check, but it is insufficient for complete-state or counterfactual equivalence.
The fused LSTM intermediate cell is unobserved, and a many-to-one readout can
agree at different states with different derivatives. For example, `g(s)=s²`
gives equal baseline outputs at `s=1` and `s=-1`, but opposite derivatives.
The same perturbation therefore has different local effects despite exact
baseline equality.

Scope intervention effects to **the declared C computational program**.
Transferring them to B0 requires an additional independently specified bridge;
the current draft does not supply one. Alternatively, a future primary
mechanism protocol may define its behavior and interventions consistently in C,
with that prospective change named explicitly. Neither option rescues v5.

Even the old forward tolerance does not guarantee equal task decisions. With
two logits `(0, 5e-7)` versus `(5e-7, 0)`, every coordinate passes the FP32
absolute/relative bound while the categorical winner changes. Require exact
baseline prediction agreement or a predeclared margin certificate before
claiming categorical behavior alignment, retaining every ambiguous example.
For continuous outputs `u,v` with `|u-v|≤ε`, raw squared-error change is bounded
by `ε(2|v-y|+ε)`; normalized MSE multiplies this by three. A small logit residual
is not zero task-error change or a causal mechanism certificate.

**2. Name the parameter-gradient training map separately.** Stage 3 uses nine
shared-prefix calls and sixteen independently recomputed calls. These counts
calibrate a separately encoded bulk-prefix training graph, call it **T**; they
do not calibrate the sixteen literal-prefix updates of C. The later acquisition
paragraph does describe the v5-style training graph, but its derivative
certificate must explicitly belong to T. Do not transfer it to C's full
parameter gradients or imply T's finite objective is B0's executed objective.
For local C state actions, holding the complete state and cached input fixed
is appropriate; full T parameter gradients must keep every prefix/encoder
contribution attached. The expectation/variance rationale is for that stated
training graph and its separable real-valued per-example loss, with finite
accumulation limits retained.

**3. Define physical versus compound AD budget units.** Forward arithmetic
checks: 736 + 248 + 2,080 + 776 = **3,840 native primals** for the stated T
recipe. Direct dense/VJP/parameter reverse actions total 48 + 200 + 72 = 320;
JVP actions total 24 + 200 = 224. A reverse-over-reverse JVP can additionally
perform two reverse actions per JVP. If that implementation is allowed, the
physical reverse total can reach **768**, rather than 320, while the declared
forward count stays unchanged. Explicitly freeze the transform backend and
counting unit. If “320” counts only standalone reverse requests, say so and
separately account for compound JVP work under the shared clock. Unsupported
native actions remain blocking outcomes; there is no automatic surrogate or
backend substitution.

**4. Complete the future calibration manifest before execution.** Pin exact
checkpoint paths/hashes, source/build, selected state rows, input templates,
direction/cotangent generator order, packing/scales and the graph to which
each tolerance applies. The existing fixed endpoint selection rule is
principled; “current checkpoint” alone is not an immutable executable pin.
The new derivative thresholds are prospective engineering proposals, not
analytic error guarantees. Give their resolution/acceptance rationale and
explicit unresolved-near-zero rule. Native zero-parameter controls can provide
a useful nonzero derivative reference: GRU has `J=.5I`; LSTM at zero state has
packed `J=[[0,.25I],[0,.5I]]`. A zero head alone cannot certify cell derivatives.
This suggestion adds no unapproved fixture or native call here.

## Bounded next decision

First obtain a completed author handoff and revise the causal scope, T-map
identity and AD budget units. Then review a static configuration for the
736-call label-free forward stage before considering the derivative stages.
The full 3,840-call/600-second plan is not the smallest first decision and is
not admitted. All source/config review, clean publication and external execution
freeze gates remain. A failed B0–C bridge blocks B0-transfer interpretation;
it is not a memory-loss result.

No threshold relaxation, score-selected path/precision, old checkpoint rescue,
body tangent, Floquet classification or fit is authorized. Numerical closure,
complete-state Jacobian closure, competence, independent trained seeds and
causal memory evidence remain distinct requirements. This partial review
provides an actionable next revision, with no accepted new trained claim.
