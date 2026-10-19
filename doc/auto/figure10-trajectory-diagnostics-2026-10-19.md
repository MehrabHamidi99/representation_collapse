# Frozen descriptive complete-state trajectory diagnostics

Frozen 2026-10-19 before inspecting any actual saved hidden/cell trajectory for
the first corrected Figure 10 reconstruction. The coordinator has observed an
aggregate validation F1 mean; no actual per-phase validation or held-out curve
was used to select these diagnostics. This protocol does not change model
fitting, checkpoint selection, retained data, numerical precision or test-read
allowance. The existing evaluator captures states during its one authorized
held-out pass; analysis reads that retained artifact and performs no model
forward calls or new dataset/weight/test-opener access.

The accepted artifact is exactly 26,214,400 bytes of little-endian float32,
C ordered as [time, component, layer, canonical node ID, feature], shape
[256,2,1,100,128]. Component zero is hidden and component one is cell. Nodes
are IDs 0..99 without relabelling. Time rows correspond to absolute held-out
targets 2816..3071 and contain **s[t] before scoring target G[t]**, after
consuming G[t-1]. Require the explicit evaluation-result JSON hash, its exact
state descriptor and the independent explicit trajectory SHA-256 to agree.
Reject incomplete files, wrong ordering/shape/timing, nonfinite states and
checksum changes. The analysis CLI must enter its mandatory offline W&B
reporter before reading or hashing binary numerical state bytes. Keep binary
and derived state analysis on cluster scratch; only reviewed compact summaries
should be collected into source control.

Analyze three views: the joint flattened H+C state (dimension 25,600), hidden
alone (12,800) and cell alone (12,800). Compute in FP64 while retaining the
source FP32 identity. For every cloud, subtract its own sample mean across
its selected time rows. Use the n-by-n Gram matrix Xc Xc^T/(n-1), whose
nonzero eigenvalues equal those of the sample covariance. This is a
**centered time cloud**, not a population covariance estimate or a recurrent
Jacobian. The full cloud has n=256 and rank ceiling min(n-1,d)=255.

Retain the descending raw Gram eigenvalues. Define a scale-aware spectral
tolerance `64 * eps(float64) * max(n,d) * max_eigenvalue`. Fail if a raw
eigenvalue is smaller than the negative of that tolerance; do not silently
repair material indefiniteness. Eigenvalues within tolerance of zero are
zeroed for the reported normalized spectrum. Report the positive trace
removed by that operation, numerical rank, normalized eigenvalue vector,
participation ratio `1/sum(p_i^2)` and entropy rank
`exp(-sum(p_i * log(p_i)))` using natural logarithms. Both dimension measures
are invariant to a uniform nonzero rescaling of states, within floating-point
limits. Preserve mean squared centered Euclidean norm
`sum(||x_i-mean||^2)/n`, sample-covariance trace
`sum(||x_i-mean||^2)/(n-1)`, mean-state squared norm and maximum absolute
state magnitude separately from normalized dimension. The scale-aware
spectral threshold must never turn a small but nonzero cloud into a claim
of no information merely because its physical variance is small.

For exactly zero centered variance, normalized eigenvalues, participation
ratio and entropy rank are **undefined (JSON null)**, with an explicit
`normalized_dimension_defined=false`. Numerical rank zero is an algebraic
statement about this cloud; it is not zero memory or absence of useful
attractor-coded information. Constant states can still carry information
across independently varied histories, which this one trajectory does not
sample.

Use the same predeclared within-motif windows as the
[behavioral protocol](figure10-within-motif-descriptive-2026-10-19.md):
phase q=t mod32, early q=1..8 and late q=24..31, excluding transition q=0.
For each motif A/B separately, each window has 32 time rows (four correlated
cycles times eight phases); pooled across A/B, each window has 64 rows.
Center each cloud separately. Report all three state views, their absolute
variance, complete normalized spectra and dimensions for both windows.
These are equal-count descriptive comparisons. Do not infer useful-memory
loss from a dimension difference: centering, phase support, motif baseline,
forced orbit geometry and amplitude can change these numbers. Report the
per-cloud rank ceiling (31 for motif-specific windows, 63 for pooled windows)
and do not compare raw ranks across unequal n as if they had equal ceilings.

For period-64 closure, compare every pair s[t] and s[t+64] for t=2816..3007:
192 comparisons covering all 64 phases over three adjacent cycle pairs.
Record exact bitwise equality of the complete H+C state, Euclidean residual,
maximum absolute coordinate difference and relative Euclidean residual
`||b-a|| / max(||a||, ||b||)`. Define relative residual as zero only when
both reference norms are exactly zero. Keep every per-pair residual and
report max/RMS absolute and max relative residuals. Exact bitwise comparison
retains signed-zero differences; numerical residuals may be zero while bits
differ. State closure on this observed finite segment neither certifies
Jacobian periodicity nor establishes a Floquet monodromy operator, stability,
contraction or causal retention. Small residuals remain descriptive without a
new, independently justified physical tolerance.

There is one trained model seed and one graph realization. Timesteps, phases,
nodes and four cycles are correlated and are not independent model or graph
replications. Perform no inferential test, bootstrap confidence interval or
causal/dimension-collapse claim from these diagnostics. Preserve any valid
negative or inconclusive result. Numerical calibration uses synthetic known-
rank, rescaled, zero, cell-only and ordering cases; those tests are
infrastructure evidence and must never be presented as a trained-model
finding.
