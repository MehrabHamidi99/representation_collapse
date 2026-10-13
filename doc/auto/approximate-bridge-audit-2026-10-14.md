# Approximate state correspondence: a conditional finite-horizon bound

2026-10-14. Coordinator derivation; independent static algebra review accepted. This is a
standard error-propagation argument, not a novelty claim or a certificate for
the measured B0/C programs.

Let two discrete systems receive the same input sequence:
`b[t+1]=F_B(b[t],u[t])` and `c[t+1]=F_C(c[t],u[t])`. A fixed map `H` sends C
states to B states. Fix norms, an integer horizon `T>=0`, and domains containing
both trajectories, their H images, and every permitted post-intervention
continuation. Assume on those domains:

- `F_B(.,u)` is uniformly L-Lipschitz, with `L>=0`.
- `||F_B(H(c),u)-H(F_C(c,u))|| <= eta` for every permitted state/input.
- Initial mismatch `||b[0]-H(c[0])|| <= e0`.
- The common-output-space readout `g_B` is K-Lipschitz and
  `||g_B(H(c))-g_C(c)|| <= rho`, with `K,eta,e0,rho>=0`.

For `e[t]=||b[t]-H(c[t])||`, adding and subtracting `F_B(H(c[t]),u[t])` gives
`e[t+1] <= L*e[t]+eta`. Induction therefore gives

`e[T] <= L^T*e0 + eta*G(T,L)`,

where `G(0,L)=0`, and for positive T:

- `0<=L<1`: `G(T,L)=(1-L^T)/(1-L)`.
- `L=1`: `G(T,L)=T`.
- `L>1`: `G(T,L)=(L^T-1)/(L-1)`.

Equivalently `G(T,L)=sum(L^j,j=0..T-1)`; the zero-step state multiplier is1,
including when L=0. The final prediction discrepancy is at most
`K*(L^T*e0+eta*G(T,L))+rho`. For L<1 this bounds accumulated transition error;
it says nothing by itself about which task information either system retains.

## Interventions require an additional distinction

Suppose paired interventions occur after state time tau and before the next
transition, with `0<=tau<=T`. If **actual post-intervention mismatch** is
directly bounded by delta, replace the initial error by delta and the remaining
horizon by `T-tau`: final discrepancy is bounded by
`K*(L^(T-tau)*delta+eta*G(T-tau,L))+rho`.

An **intervention correspondence defect** is different. If
`||I_B(H(c))-H(I_C(c))||<=delta_I`, one additionally needs an L_I-Lipschitz
bound for I_B on the relevant pre-intervention states. Then

`e[tau+] <= L_I*e[tau-]+delta_I`,

so a sufficient final bound is
`K*(L^(T-tau)*(L_I*(L^tau*e0+eta*G(tau,L))+delta_I)
     +eta*G(T-tau,L))+rho`.

The defect alone does not imply `e[tau+]<=e[tau-]+delta_I`. For example, with
H the identity and both interventions `I(s)=10*s`, the defect is zero while
every nonzero mismatch increases tenfold. Domain coverage must still hold
after either intervention; otherwise the propagation assumptions do not apply.

For a difference of intervened and baseline predictions, the cross-system
effect discrepancy is bounded by the sum of the two trajectory prediction
bounds, by the triangle inequality. The two rho terms do not automatically
cancel. This is a conditional deterministic bound, not a statistical interval.

## What this does not establish

No constants, global domain coverage or H have been certified for the actual
B0/C measurement programs. Agreement at finitely many baseline outputs, a
matching local Jacobian, or a local Hessian condition does not supply these
uniform assumptions; see the [exact-bridge audit](causal-bridge-audit-2026-10-14.md).
B0's fused intermediate LSTM cell state remains unobserved. A Lipschitz
assumption must itself be justified for the domain and finite implementation
being discussed; a real-arithmetic surrogate does not establish an ordinary
derivative of an IEEE rounding map. This bound confers no original-task
competence, intervention transport certificate, useful-information loss,
dimension collapse, periodic orbit or Floquet conclusion.

## Qwen participation and disposition

The original cluster worker completed call9 at October13 21:43:52UTC, with
the exact requested model identity,5129reportedtokens and16543requestcharacters.
The complete1215-character mathematical task and490-character context note
were present. Its answer did not derive the requested bound. It instead
proposed new training, FP64 selection and significance-dependent stopping.
The coordinator rejects the whole answer as nonresponsive; FP64 alone is no
stability certificate, and outcome-dependent stopping is not a substitute for
the project's frozen inference plan. No proposed job or retry was executed.
The derivation above is the coordinator's replacement, not an attributed Qwen
result. Qwen participation receipt SHA-256:
`d0c5d23fd237f57ef12a28a8d64a15e2bb700cf96e81c297b815b320a6baa820`.
