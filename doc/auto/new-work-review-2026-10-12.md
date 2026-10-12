# Review of the new documentation and stopped implementation pass

Reviewed on 2026-10-12. Inputs: `eb12ea1` (documentation consolidation),
`dc5e020` (stopped Gate B pass), the working tree and dated October 11 runtime
receipts. This review introduces no trained-model evidence, new allocation or
model-service call. Current runtime status is in
[research-status-2026-10-12.json](research-status-2026-10-12.json).

## What landed

The documentation commit changes 26 files: a research charter, P1–P12 theory
document, experimental design v2, implementation audit, sixteen-node plan,
science-first reading guides and a manuscript skeleton. The framing is now
useful-history loss under a controlled periodic-versus-varying body at fixed
weights and equal horizon. It preserves competence as a prerequisite and
separates covariance geometry, tangent dynamics, and behavior/information.
The 351-checkpoint screen is deferred. The paper retains the four failed
pilots, unreduced periodicity diagnostics and unverified legacy graph evidence
with distinct labels. This is useful planning progress, not a new empirical
finding or a submission-ready paper.

The later commit records that the implementation workers stopped. The only
leftover is an untracked `src/representation_collapse/diagnostics.py` draft.
It contains additive R0 and single-WRITE constructors, an explicit-memory
register oracle and corruption controls. Syntax and eight tiny constructor /
oracle fixture cases pass in this review. It remains untracked, unchanged,
without matching unit tests, versioned training integration or Gate B
acceptance. The rest of the Gate B implementation is absent. This review does
not resume the stopped implementation pass.

## Corrections applied to current documents

- **FP32 closure.** A tolerance pass need not mean bitwise equality. For a
  13×128 FP32 block at 0.1, changing one coordinate by one ULP gives a
  residual of approximately 7.45e-9 and passes the frozen 1e-7 absolute bound.
  A hypothetical all-coordinate half-ULP norm is not a universal nonzero
  difference floor. Current documents now require measured spacings, changed
  coordinates and bitwise equality to be reported separately. Frozen v1
  thresholds are unchanged; FP64 replay is a separate assay, not exact truth.
- **Commuting null.** Equal deterministic cue products and centered noiseless
  geometry do not force raw `delta16` or DID to vanish. Arm-dependent offsets
  and propagated process noise can change native error. The optional SSM
  control now tests the actual cue-channel null with these qualifications.
- **Readout-offset loss.** With control residual `e` and a fixed projected
  offset `Delta`, the MSE gap is `2 E[e]^T Delta + ||Delta||²`. Centering the cue
  alone does not remove the cross term. For random paired offsets, retain
  the expectation of their product. The manuscript statement and proof agree.
- **Approximate continuous memory.** Finite-horizon average MSE competence
  alone does not imply eight noise-resolved local singular values. Exact
  smooth invariant-memory P4 remains valid; a quantitative approximate claim
  needs decoder-gain, noise and local recovery assumptions.
- **Rank, matching and probe scale.** A 512-cue covariance ceiling is
  `min(511,d)`: 128 for GRU, 256 for joint LSTM, not universally 511. Motif
  trigram counts can differ but need not differ in every paired body. The
  probe-floor concern stands; a universal hundredfold horizon change in
  effective regularization was not established and has been removed.
- **Statistics.** Sign flipping is exact only under justified null
  sign-exchangeability or a valid sign-randomization design; zero mean is
  insufficient. The shifted SEI null, sidedness and equivalence-family
  correction must be frozen. The ten-seed 99.375% Bonferroni t-interval has
  half-width approximately 1.1215 seed SD, compared with 0.7154 for an ordinary
  95% interval. Bootstrap coverage with ten seeds remains approximate.
- **Theorem scope.** P1 noise formulas distinguish fixed shared realizations
  from random episode noise. P5 requires nonzero scalar input coupling and
  uniqueness of a periodic or bounded bi-infinite response, rather than all
  bounded forward trajectories; a first harmonic need not exist. P7's upper
  equalities require equal nonzero eigenvalues; equality also at the lower
  bound one requires rank one.

## Remaining design work

The statistics are proposed, not a hashed preregistration. Null assumptions,
the exact decision tests and multiplicity-adjusted equivalence rules must be
resolved before confirmation. Vector calibration readiness should be separated
from the broader graph calibration milestone in the DAG; prediction/statistics
freezes need explicit prerequisites without a Gate B–C cycle. The original
sixteen-node graph is acyclic, but that alone does not establish readiness.

Compute costs are extrapolations, not measured bounds for the expanded battery.
The repeat-run allowance is clarified as four selected seed-0 fits, not all
twelve learning-rate candidates. Benchmark a complete eligible measurement
battery before proposing a new allocation. Confirmation-seed qualification
and missing-seed treatment must be fixed before using competent fits in a
population claim; no undisclosed subset or replacement-seed policy is accepted.

Independent periodicity raw-array reduction, probe-floor accounting, Gate B
implementation and T-GRAB code/data pinning remain outstanding. No gate or
scientific claim is promoted by this review.

## Cluster research worker and publication

The live SSH attempt on October 12 at 01:26 UTC failed before reaching Mila:
the saved control socket was inaccessible and hostname resolution failed in
this restricted session. No current process, scheduler or cluv connection
state could be read. The worker is **currently unverified**, not confirmed
running or stopped. The last verified heartbeat is October 11 at 16:23 UTC:
14 valid Qwen reviews, zero consecutive errors and 76,635 reported tokens;
call0014 consumed the completed periodicity packet. Its contract is 24 hours
and at most 22 attempts; it may since have reached that cap. No restart or
additional budget is inferred. Stopped implementation workers are separate
from this Mila worker.

The four periodicity jobs were independently reconciled as completed `0:0`
on October 11, with 28 artifact hashes verified; raw-array numerical reduction
remains pending. No fresh claim about other cluster jobs is made today. Local
cluv is still 0.1.5. New-document delivery to Mila is unverified. Local HEAD
has two commits beyond the cached `origin/main`; that cached comparison is
not a fresh remote publication check. This session has restricted network and
read-only Git metadata, so review changes are left locally for inspection.

Review receipts, deterministic counterexamples and validation details are in
ignored `runs/ops/20261012-new-work-theory-review.json`,
`20261012-new-work-design-review.json` and
`20261012-investigation-verification.json`. Historical dated reports and
frozen sources remain intact; the October 12 notebook appends this correction.
