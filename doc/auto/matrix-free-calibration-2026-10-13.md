# Complete-state matrix-free calibration

2026-10-13. Status: local engineering calibration passes; independent review
and coordinator acceptance remain separate. This advances the measurement gate
in [graph reproduction](graph-reproduction.md). It supplies no trained-model,
harmful-collapse, rank, covariance-dimension or Floquet evidence.

The new [operator](../../src/representation_collapse/tangent_operator.py) applies
the complete-state finite-time derivative through PyTorch 2.6 JVP/VJP transforms.
The [tests](../../tests/test_tangent_operator.py) compare it against tiny dense
references. Existing dense measurements in `metrics.py` and `periodicity.py`,
native-state capture and inherited manuscript/source snapshots are unchanged.
No environment changes, cluster allocations, training or model-service calls
were made for this calibration.

## Domain and chronology

`StateLayout` names every persistent component and records component order and
shape. Each component flattens in contiguous row-major order. For a core graph
LSTM, a declared layout with `h` then `c`, each `[layers,nodes,hidden]`, has
dimension `2*layers*nodes*hidden`. Node identities/order belong in the caller's
trajectory manifest. Persistent prediction caches or other buffers require an
explicit state component, or a documented justification for holding them fixed;
this API cannot discover missing state. A pooled readout is not a substitute
for complete state.

The caller supplies pure deterministic steps `F_t(state_flat, fixed_input)`,
L+1 recorded states and L inputs. The operator snapshots inputs/states, verifies
two bitwise-equal primal replays, and checks each derivative action's primal
against the next recorded state with caller-declared absolute/relative
tolerances. These checks detect broken recorded trajectories and observable
stochastic replay; they do not prove functional purity or state completeness.
Different callables and different graph inputs can be used at each update.

Column perturbations propagate chronologically as
`P=J[L-1] ... J[0]`. The transpose applies each VJP from update L-1 down to 0,
giving `P.T=J[0].T ... J[L-1].T`. Inputs are held fixed. Teacher-forced derivatives
are conditional on the recorded exogenous sequence. A closed-loop analysis
must place generated-input dependencies inside the differentiated step and
declare all persistent state. The task readout remains outside this operator
unless it is itself persistent; readout derivatives and behavioral effects
require separate assays.

Functions that close over parameters must pass `frozen_tensors`, or pass their
modules as `evaluation_modules`. Module guards verify evaluation mode of all
children, module/parameter/buffer identity, and unchanged tensor values before
and after actions. Guarded parameters receive no accumulated `.grad`. Callables
that omit these guards still have a caller-fixed-weight obligation; constructor
success does not certify it. This module does not switch modes or precision.
It rejects inference mode; ordinary `torch.no_grad()` callers are supported.

## One candidate singular pair

`estimate_leading_singular_gain` performs bounded power iteration on `P.T P`
with exactly one caller-supplied nonzero initial vector or explicit local RNG
seed. The original vector, final right/left directions, requested/converged pair
counts, every iteration's gain/residual, tolerance and termination status are
retained. The local seeded CPU generator leaves the global Torch RNG unchanged.

The gain is `sigma=||P v||_2` for a unit right vector and `u=P v/sigma`. Stopping
uses the relative right singular equation residual
`||P.T u-sigma*v||_2/sigma`, evaluated using normalized actions and log scales.
The left singular equation holds by construction at the working precision.
Each JVP/VJP rescales between updates; `log_gain` and, for L>0,
`log_gain_per_update` remain available when the gain exceeds the Python float
range. This is a finite-window rate, not a long-horizon Lyapunov estimate.

A small residual establishes a stationary singular pair at the numerical
tolerance. It does not establish that the pair is leading: an initial vector
orthogonal to the leading subspace can converge to a smaller singular pair.
The implementation always sets `leading_pair_certified=false`,
`spectrum_complete=false`, `all_directions_contraction_certified=false` and
`floquet_certified=false`. It cannot supply tangent effective dimension from one
pair, infer eigenvalue contraction from a singular gain, or certify periodic
state/Jacobian closure.

Zero actions retain an observed directional-zero status, which cannot distinguish
true nullspace from underflow at the working dtype. A zero initial direction
does not certify a zero operator. Nonfinite derivatives, adjoint breakdown,
unsupported autodiff actions and maximum-iteration termination preserve failure
status; no arbitrary singular-value floor is inserted. A later failed iteration
clears stale gain/residual values while retaining earlier history. Stepwise
rescaling cannot recover precision already lost inside a differentiated step
or between small coordinates, and does not make unsupported upstream kernels
differentiable.

## CPU acceptance checks

Command:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_tangent_operator.py -v
```

All **24 tests pass**, with no skips, in **1.439 seconds** of unittest time on
Python 3.11.13 / Torch 2.6.0, CPU, one Torch thread. This timing describes tiny
test execution and is not GPU throughput evidence.

The nonlinear fixture contains one layer, two coupled nodes, width two and both
h/c, giving eight coordinates over three updates. Its sigmoid/tanh recurrence
is explicitly constructed for calibration; it is not upstream GC-LSTM. Fixed
graph inputs differ by step, and cross-node Jacobian entries are checked to be
nonzero. The following numbers use initialization seed 812:

- FP64: candidate gain **0.30401381268055816**, equal to the dense SVD reference
  at the displayed precision; **184 iterations** at residual tolerance 1e-10,
  final relative singular residual **9.154732846904646e-11**. Maximum absolute
  JVP and VJP errors are both **1.3877787807814457e-17**. The adjoint inner-product
  error is **3.469446951953614e-18**. Central full-rollout finite differences at
  epsilon 1e-6 differ from JVP by at most **1.9791675243130413e-11**.
- FP32: candidate gain **0.3040137592447721**, dense SVD reference
  **0.30401375889778137**; **68 iterations** at residual tolerance 1e-5, final
  residual **9.978565151363538e-6**. Maximum absolute JVP/VJP errors are
  **1.1175870895385742e-8** / **7.450580596923828e-9**. The measured adjoint
  inner-product difference is zero at FP32 precision.

The FP64 fixture requires more than the API's default 100-iteration cap for the
stated 1e-10 tolerance; its test explicitly freezes a 200-iteration cap. A
plausible gain alone does not override residual-based nonconvergence.

Other checks include the existing GRU/LSTM gate-equation dense window reference
in both precisions; noncommuting ordering against `metrics.ordered_product`;
rotation and uniform contraction; singular and zero maps; initial nullspace;
a subleading stationary pair; bounded nonconvergence; adjoint breakdown; an
actual infinite derivative at a finite primal; unsupported autodiff failure;
frozen state/input snapshots; evaluation/weight/module-identity mutation; and
strict JSON diagnostics with no NaN/Infinity values. Three updates at scale
1e-200 or 1e200 demonstrate finite logged gains despite unscaled-product
underflow/overflow. A nonnormal matrix with eigenvalues 0.5 has singular gain
above four, preserving the eigen/singular distinction. Dense Jacobian APIs are
mocked to fail while production matvecs and iteration still pass.

Source SHA-256:
`18ba61ed6e4cf7b14cb9bdd0417b03d7cfd697ad75593ffc2470d3c52d4a25b1`.
Test SHA-256:
`6b764993a8dfb2cbd3e6b3c583b14319e9f6a202f0f9f93cfc43c50bcde16592`.

## Remaining graph gate

Production storage scales with the trajectory and guarded weights, not a dense
d-by-d Jacobian. Each iteration still traverses the full window in both
directions, copies fixed inputs and performs weight/buffer checks. Large graph
performance, GPU telemetry and numerical stability need measured profiling;
this CPU fixture does not establish allocation sizing.

The pinned upstream graph cell and its Chebyshev/PyG operations still require a
complete-state adapter, derivative-rule compatibility checks and comparison
against an actual tiny upstream dense recurrence. Dataset payload, historical
checkpoint/condition and cached evaluation chronology remain separate recovery
blockers. After those gates, trajectory replay and numerical tolerances must be
frozen for the chosen task/forcing regime before held-out learned-model
analysis. The operator supplies no scientific acceptance by itself.
