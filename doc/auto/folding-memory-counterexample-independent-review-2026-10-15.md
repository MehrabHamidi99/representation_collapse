# Independent review: memory loss by global folding

2026-10-15. **Accepted as an exact one-step analytic counterexample; no
empirical project result or novelty claim is accepted.** No correction to the
author's proof is required. This review used source inspection, analytic
reasoning and one standard-library exact-rational check. There was no sampled
data, NumPy, Torch, model, SDK, provider, network or cluster execution.

The reviewed source is
[folding-memory-counterexample-2026-10-15.md](folding-memory-counterexample-2026-10-15.md),
SHA-256 `e1e26fb0e59a744506a247ace04ba7cacbf49b8efdf79678de43d71ca1895d99`.
It fixes mutually independent M∈{−1,+1}, balanced; A∼Unif[2,3];
U∼Unif[−√19,+√19], initial complete state S=(MA,U), and

\[
F(x,u)=\left(c(x^2-19/3),u\right),\qquad c=\sqrt{285/94}>0.
\]

The task is recovering the previously written M from the available complete
state. The output information boundary includes F(S) and the known fixed map,
and excludes the previous state, an external copy of M or branch identity.
This boundary is essential to the information-loss claim. All distribution
and map choices are fixed in the proof, with no postmeasurement fitting.

## Population moments and covariance

Because A has density one on an interval of length one,

\[
\mathbb E[A^k]=\frac{3^{k+1}-2^{k+1}}{k+1},\quad
\mathbb E[A^2]=\frac{19}{3},\quad
\mathbb E[A^4]=\frac{211}{5},\quad
\operatorname{Var}(A^2)=\frac{211}{5}-\left(\frac{19}{3}\right)^2
=\frac{94}{45}.
\]

E[M]=0 and independence give E[MA]=0, Var(MA)=E[A²]=19/3,
and Cov(MA,U)=0. U has mean zero and variance (√19)²/3=19/3.
For Y=F(S), the first coordinate is c(A²−19/3), so its mean is zero and

\[
\operatorname{Var}(Y_1)=c^2\operatorname{Var}(A^2)
=\frac{285}{94}\frac{94}{45}=\frac{19}{3}.
\]

Y₁ is a function of A alone, hence independent of U. Therefore both pooled,
mean-centered population covariance matrices equal (19/3)I₂. Both eigenvalues
are strictly positive and equal: rank 2, trace 38/3, covariance participation
ratio (2λ)²/(2λ²)=2. With normalized covariance eigenvalues p₁=p₂=1/2,
the natural-log entropy is ln2 and entropy effective rank exp(ln2)=2.
These are covariance-eigenvalue conventions, rather than entropy of the state
distribution or of the task label.

The author's claim about all covariance-only measurements is valid for any
fixed function **solely of these marginal, pooled, centered population matrices**.
It does not assert equality of all statistics that can be described using
covariance. Conditional, lagged or cross-state covariances and alternative
sampling/centering definitions are outside that equality. For example,
Var(S₁|M)=Var(A)=1/12, whereas Var(Y₁|M)=19/3. The pooled covariance proof does
not erase that distinction. It also does not assert equal finite-sample
covariance matrices for a particular draw or a phase covariance from one orbit.

Higher distributional statistics need not match: the initial first coordinate
has zero third moment by sign symmetry, while
E[(A²−19/3)³]=394/945 and hence E[Y₁³]=c³(394/945)>0. This exact scope check
confirms that identical covariance does not imply identical state distributions.

## Local tangent bound and finite-distance folding

The map is globally smooth with

\[
DF(x,u)=\operatorname{diag}(2cx,1),\qquad
DF^{\mathsf T}DF=\operatorname{diag}(4c^2x^2,1).
\]

Every initial state has |x|∈[2,3]. Its first singular value is
2c|x|≥4c>1; the other is exactly1. An exact squared bound is
16c²=2280/47>1. For every real complete-state tangent v=(v₁,v₂),

\[
\|DF(S)v\|_2^2-\|v\|_2^2
=(4c^2x^2-1)v_1^2
\ge\frac{2233}{47}v_1^2\ge0.
\]

Thus there is no contracting local tangent direction at any realized initial
state; the U-only direction is neutral, rather than strictly expanding.
The derivative has rank 2 and nonzero determinant at these initial states.
Each sign branch has a local inverse, with first coordinate
±√(Y₁/c+19/3) and second coordinate U. This is local invertibility, not a
global injectivity claim.

For any fixed a∈[2,3] and admissible u,

\[
F(+a,u)=F(-a,u),\qquad
\|(+a,u)-(-a,u)\|_2=2a>0.
\]

The finite pair separation is therefore mapped to zero. The example is not
a globally distance-preserving or globally noncontracting map. Its local
derivatives do not compare the two separated sign branches. It also makes no
claim that future derivatives remain noncontracting: the output support in
the first coordinate contains zero, where the derivative of another application
of F is singular. No future propagator, Lyapunov estimate, periodic orbit or
Floquet multiplier is verified by this one-step bound.

## Information and Bayes error

Initially A>0 almost surely, so M=sign(S₁) exactly. With the balanced label,
H(M)=1 bit and H(M|S)=0, giving I(M;S)=1 bit. These are discrete-label
conditional entropies; the argument needs no differential entropy of S.

The complete output is Y=(c(A²−19/3),U), a measurable function of (A,U),
which is independent of M. For every measurable output set B and either m,
P(M=m,Y∈B)=P(M=m)P(Y∈B). Consequently P(M=m|Y)=1/2 almost surely,
H(M|Y)=1 bit and I(M;Y)=0. Every measurable decoder g(Y) taking values in
{−1,+1} has error 1/2; this is also the Bayes minimum. Initial Bayes error is 0.
Independent decoder randomization cannot improve that posterior risk.

The paired intervention changes only M and holds A,U fixed; the exact equality
above demonstrates removal of branch information. There is no omitted state
coordinate, stochastic innovation, decoder optimization failure or unknown
readout approximation in this two-coordinate argument. An external record
of the old sign would change the information boundary and the conclusion.

## Verdict and project scope

The proof refutes universal necessity of a covariance-dimension decrease or a
contracting **one-step local complete-state tangent direction on the input
support** for loss of the required bit. It illustrates global folding as a
mechanism invisible to those particular local/covariance summaries. It does
not refute a statement assuming global injectivity, preservation of all finite
pair distances or access to the old state. It does not imply that contraction
cannot cause memory loss, that all contracting dynamics lose useful memory,
or that covariance and tangent diagnostics are uninformative in other settings.

This accepts a source proof and a measurement guard against the coordinator's
identified, unaccepted proposal that contraction is necessary for harmful
memory loss. It accepts no trained-model finding, periodic-body effect,
benchmark reproduction, causal project-collapse mechanism, general Floquet
claim or novelty attribution. This project's empirical claims still require
competence, task-required history controls, matched horizons/body order,
independent replication/paired uncertainty and accepted numerical artifacts.
No compute envelope is created or consumed by this review.

One exact Fraction invocation verified ten declared algebraic checks, including
both fourth/second moments, output variance, PR and the singular-value lower
bound; it additionally checked the nonzero higher-moment scope example. Its
source identity, exact rational results, review hash and zero-execution scope
are frozen in `runs/review/folding-memory-counterexample-20261015/receipt.json`.
The root proof was read-only throughout. No blocking mathematical finding or
source edit is requested.
