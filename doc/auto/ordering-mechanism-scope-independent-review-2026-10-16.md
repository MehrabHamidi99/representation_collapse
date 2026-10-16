# Independent review of ordering-mechanism scope

2026-10-16. **Accepted as an analytic design limitation and interpretation
guard.** The coordinator's [ordering-scope assessment](ordering-mechanism-scope-review-2026-10-16.md)
is correct for the frozen registered motif family. No repair or protocol change
is required. This review inspected source, governing documents and Qwen replies
24–25, then computed file hashes only. It ran no constructor, diagnostics,
numerical experiment, model, SDK, provider or cluster operation.

## Counting argument

Let one registered dwell-expanded motif have length L=12q, let M be the final
motif, and divide each body into exactly R registered L-length slots. The source
`paired_body_ids` fills every periodic slot with the same sampled motif and fixes
the control's final slot to that motif. At R>1 it rejects any control whose
rendered whole body has a proper tiling period. `matched_body_audit` measures
the number of exact M matches at indices 0,L,...,(R−1)L, not at every overlapping
substring position.

Write the boundary exposure count as C_M(B)=sum over slots j of the indicator
that chunk j equals M. Each indicator is either zero or one. For periodic
B=M^R, C_M(B)=R. If another R-slot body has C_M=R, every indicator must be one;
therefore every chunk equals M and its entire rendered body is M^R. Because
R>1, L is a proper whole-body tiling period, which the control rule rejects.
This proof does not assume motifs have no shorter internal period: a shorter
period would also be a rejected proper period.

Consequently any accepted current control has C_M at most R−1. Its final motif
is M, so it also has C_M at least one. The impossibility holds separately for
each matched pair, independently of random sampling frequencies or the trained
model. The older design discusses **prior** final-motif exposures, excluding
the matched final slot: subtracting one from both counts gives the equivalent
R−1 versus at most R−2 statement. No contradiction between those conventions
is present.

The proof concerns registered chunk boundaries. It is not a theorem about
arbitrary overlapping substrings, all n-gram matching schemes, continuous-time
periods or every possible experiment. R1 is an identical-arm anchor; R0 is the
zero-body anchor supported elsewhere in the rendering/diagnostic orchestration,
not by `paired_body_ids`, which requires repeats at least one. No R0 execution
support is inferred from that constructor.

## Causal interpretation and proposal critique

Under the existing fixed-weight, matched-horizon, correctly paired and competent
evaluation design, a supported arm contrast estimates the total effect of the
specified body-order intervention. The two arms differ structurally in the
final-motif exposure count and may differ in other higher-order statistics.
Those facts do not invalidate the registered finite contrast; they limit the
claim that one isolated periodicity mechanism has been identified. Calling an
arm-derived descriptor a pre-treatment covariate merely because it is computed
before evaluating the model does not provide independent variation or identify
a controlled direct effect. The coordinator correctly calls for descriptive
reporting and independently justified causal interventions rather than an
unregistered adjustment that changes the estimand.

Qwen24's proposal cannot be adopted as written. Its 1% mismatch tolerance and
unspecified FFT significance rule are unregistered, and neither resolves the
exact current matching impossibility. Wrapped input covariance/FFT describes
the body inputs; the helper excludes WRITE and QUERY and never observes the
model state or cue-conditioned outcomes. Two models with identical bodies can
retain different cue information while returning identical input diagnostics.
Thus those diagnostics alone cannot measure required-history recoverability.
Its asserted universal dependence of a periodicity-only mechanism on local
motif repetition is unsupported. The coordinator's rejection is appropriately
specific, while retaining the useful question of alternative motif exposure.

Qwen25's conceptual next-graph/prefix-recall distinction is valid and already
belongs to the charter. Its claimed high next-graph accuracy in preliminary
untrained calibration has no accepted evidence. The tiny native calibration
verified measurement/implementation integrity, not learned benchmark competence.
A vector GRU calculation is also not an established native graph bridge, and
an unspecified nonlinear mutual-information calculation is not a frozen
recoverability estimator. Rejecting the whole reply while preserving the valid
logical distinction is justified.

A future multi-motif superperiod intervention could preserve primitive-motif
counts while disrupting their superperiod ordering. It would require explicit
definitions of the primitive motif, superperiod, boundary-count unit, context
matching and non-tiling condition. It would not automatically preserve the
number of occurrences of the entire original repeated supermotif or isolate
periodicity from every other higher-order feature. The coordinator explicitly
labels this a new treatment family requiring review/preregistration, which is
the correct scope. No new construction, tolerance, seed rule or Gate B change
is authorized here.

The original T-GRAB next-graph benchmark and a subsequent graph-prefix memory
intervention remain different scientific nodes. A valid graph memory study
must make the varied prefix task-required, establish competent acquisition and
readout, measure complete hidden-plus-cell/node state and use the declared
independent graph/training-seed units and causal shams. Nothing in these Qwen
answers proves a trained retention effect, geometric collapse, Floquet dynamics
or native cross-modal transfer.

## Evidence and acceptance boundary

Inspected inputs were the coordinator assessment, frozen `vector_task.py`,
`input_diagnostics.py`, their accepted descriptive-diagnostic review/protocol,
experimental design v2, research charter, novelty audit and ignored
`runs/qwen/local-continuous-20261014T1650Y/call-0024/review.json` and
`call-0025/review.json`. Exact byte hashes and this review's receipt are under
`runs/review/ordering-scope-20261016/independent/`.

Acceptance is limited to the counting proof, proposal critique and resulting
interpretation guard. It establishes no positive priority/novelty certificate,
empirical mechanism, new statistical freeze or compute authority. The main
research goal remains unfinished.
