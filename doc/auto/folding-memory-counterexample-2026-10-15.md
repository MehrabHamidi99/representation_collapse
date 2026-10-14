# Useful memory can disappear with identical covariance and no local contraction

**Analytic measurement guard, not an empirical finding or novelty claim.**
This counterexample refutes using covariance dimension or local tangent
contraction as a necessary condition for task-required memory loss. It does
not establish a periodicity effect, a recurrent benchmark result or Floquet
behavior. The argument concerns one discrete update of the complete state.

Let independent variables be a balanced binary label M in {-1,+1},
A uniform on [2,3], and U uniform on [-sqrt(19),sqrt(19)]. The previously
written task is to recover M from state S=(M A,U). Initial recovery is exact:
M=sign(S_1). Fix the globally smooth deterministic update

\[
F(x,u)=\bigl(c(x^2-19/3),u\bigr),\qquad c=\sqrt{285/94}.
\]

There is no stochastic innovation, additional forcing, hidden coordinate,
adaptive parameter, fitted decoder or population-dependent tuning after
measurement. The distribution and constant are declared together.

## Complete-state covariance is exactly unchanged

Both initial coordinates have zero mean. Since

\[
\mathbb E[A^2]=19/3,\qquad
\mathbb E[A^4]=211/5,\qquad
\operatorname{Var}(A^2)=94/45,
\]

the output first coordinate has zero mean and variance
c^2 Var(A^2)=(285/94)(94/45)=19/3. U is independent of the first coordinate
both before and after the update, and Var(U)=19/3. Therefore

\[
\operatorname{Cov}(S)=\operatorname{Cov}(F(S))=(19/3)I_2.
\]

Population covariance rank stays two, participation ratio stays two, and
entropy effective rank stays two. Trace, eigenvalues and all measurements
depending only on the centered covariance are identical. This is a statement
about the declared independent population, not a finite-sample estimate or a
covariance across phases along one trajectory.

## Every realized tangent direction is noncontracting at this update

\[
DF(x,u)=\operatorname{diag}(2cx,1).
\]

Every initial |x| lies in [2,3], so the singular values are 2c|x|>1 and 1.
For every tangent vector v, ||DF(S)v||_2 >= ||v||_2. The Jacobian has full
rank and no contracting direction at any realized initial state. This claim
is about this one-step derivative, not all future iterates, a periodic orbit,
a Lyapunov exponent or a Floquet multiplier.

## Useful information nevertheless drops from one bit to zero

F(M A,U)=(c(A^2-19/3),U) depends only on A and U, which are independent of M.
Thus I(M;S)=1 bit, whereas I(M;F(S))=0. Every measurable decoder from the
complete output state has binary Bayes error 1/2, compared with zero before
the update. This is irreversible loss under the stated information boundary,
rather than failure to fit a sufficiently flexible decoder.

The causal paired check holds A and U fixed and changes only the written
label: F(+A,U)=F(-A,U). The update identifies two separated branches. Local
invertibility around each initial state does not imply global injectivity;
infinitesimal derivatives do not compare the finite separation between those
branches. Full covariance can be preserved by nuisance variation while the
task-required bit disappears.

## Consequence for this project's acceptance rules

Task-required recoverability, covariance dimension and tangent dynamics remain
separate estimands. A task-loss finding must not be rejected merely because
covariance dimension is unchanged or a local tangent propagator does not
contract. Conversely, contraction or low dimension alone does not prove
task loss. Distinguish decoder error from complete-state information loss
using the task's information boundary and paired causal controls.

This example supplies no evidence that periodic bodies cause excess loss.
That question still requires competent trained models, fixed weights, equal
retention horizon, matched body-order interventions and paired uncertainty.
It also grants no new compute allocation. The moment identities and proof are
exact algebra; no model run, sampled dataset or numerical campaign was used.
