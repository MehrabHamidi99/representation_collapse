# Baseline agreement and intervention transfer — scope audit

This is an algebraic check, not a learned-model finding or a novelty claim.
No model weights, targets, native forwards or optimizer updates are used.
It supports the C-only interpretation in the
[prospective measurement contract](post-calibration-measurement-contract-v2-2026-10-14.md).

## A correct counterexample and an insufficient conclusion

Qwen call8 completed October13 21:00:30 UTC on an explicit mathematical task:
eight replies in eight attempts, zero errors, 39,932 cumulative reported tokens.
The call used5,038tokens and the requested model identity. The seven previous
call directories remain byte-identical. This establishes additional proposal
activity, not scientific acceptance.

The reply correctly evaluates the readout `g(s)=s²` at states +1 and -1.
Their outputs both equal1, but the same additive perturbation ε changes the
outputs by `2ε+ε²` and `-2ε+ε²`. At ε=.1 these are .21 and -.19.
Baseline output agreement alone therefore does not establish equal finite
intervention effects.

The reply then proposes equal pointwise Jacobians, or a perturbation in the
null space of the second derivative, as sufficient conditions. These are
insufficient for finite interventions. Consider instead

`g(s)=(s²-1)³`.

At both s=+1 and s=-1, the output, first derivative and second derivative
all equal zero. After the same additive ε, the changes are

`g(1+ε)=(2ε+ε²)³`, and `g(-1+ε)=(-2ε+ε²)³`.

Their difference is `16ε³+12ε⁵`, which is nonzero for every nonzero real ε.
At ε=1/10 the changes are exactly9261/1000000 and -6859/1000000.
Equal first derivatives concern first-order local responses; they do not
control all finite remainders. A pointwise zero Hessian is insufficient too.
Readout linearity is not a necessary condition for transfer when complete
maps and interventions correspond correctly. The whole Qwen answer is
therefore rejected; only its stated scalar counterexample is retained.

## Sufficient conditions with an explicit domain

Let C and B have complete state spaces, transitions `F_C(s,x)` and `F_B(z,x)`,
and readouts `q_C` and `q_B`. Fix an input chronology and a domain containing
the initial states and every state reached by the permitted transitions and
interventions. A sufficient one-way correspondence is a map H satisfying

- Initial state: `z₀=H(s₀)`.
- Every permitted transition: `F_B(H(s),x)=H(F_C(s,x))`.
- Readout: `q_B(H(s))=q_C(s)`.
- Every paired intervention: `I_B(H(s))=H(I_C(s))`.

The domain must remain closed under the operations being compared. Induction
through the common chronology then preserves `z=H(s)`, including interventions,
so both baseline and intervened readouts agree. Bijectivity is not needed for
this one-way sufficient statement; reverse transfer needs its own conditions.
H and the readouts may be nonlinear. Using the same numerical additive vector
on both sides is valid only if it satisfies the intervention equality; a
coordinate transformation generally requires a transformed intervention.

These identities are assumptions or statements to prove over the declared
domain. Finitely many matching outputs do not establish them. Approximate
identities require separate propagated error bounds and decision margins;
this audit supplies no such numerical certificate. In particular, B0's
unobserved intermediate LSTM cell supplies no complete-state correspondence.
The proposed finite C perturbation measurements remain about C alone.

The actual request, response, usage, immutable-call comparison, exact rational
checks and coordinator verdict are retained under ignored
`runs/ops/qwen-causal-bridge-20261013T2100Z/`. No experimental action is taken
from the rejected recommendation.
