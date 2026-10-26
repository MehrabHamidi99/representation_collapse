# Research program: recurrent representation dimension and dynamics

Version 1, 2026-10-11. This is a prospective protocol, not an empirical result.
The initial evidence is inherited graph manuscripts and figures; see
[source inventory](source-inventory.md). Recover and reproduce that evidence
before making a claim about recurrent models across modalities.

## Question and competing explanations

Under what task, training, and input conditions do recurrent models lose
representational dimensions or sensitivity to past information? Does that loss
explain a measured failure on information that the task actually requires?
Analyze three distinct objects: state activity covariance, tangent propagation,
and task-relevant behavior. Agreement is a hypothesis to test, not a definition.

- **H1, harmful collapse:** fewer reliably measured activity/tangent directions
  accompany reduced retention of independently varied past information, and an
  intervention on those directions changes task behavior.
- **H2, sufficient compression:** dimension falls while the relevant information
  and performance remain intact. A compact representation may solve the task.
- **H3, input synchronization:** repeated forcing removes initial-state
  sensitivity while continuously injecting task information. A contracting
  model may still have a persistent periodic response.
- **H4, measurement artifact:** apparent collapse follows short recordings,
  repeated identical inputs, pooling, saturating amplitudes, graph symmetries,
  numerical precision, or feature normalization rather than lost computation.

Freeze the main comparison and uncertainty method before viewing held-out
outcomes. Select models on validation task performance; do not select on a
desired dimension or multiplier unless that is an explicitly separate study.

## Exact discrete dynamics first

Define the model's complete persistent state and step convention:

\[
s_{t+1}=F_\theta(s_t,x_t),\qquad
J_t=\partial_sF_\theta(s_t,x_t),\qquad
P_{t,L}=J_{t+L-1}\cdots J_t.
\]

Differentiate at fixed external inputs and parameters. For an LSTM include
hidden and cell state; for a graph model include every coupled node and layer;
for architectures with persistent buffers include those buffers or explicitly
state why they are fixed. A node-wise diagonal block does not represent the
full graph transition. Record teacher forcing versus a closed-loop rollout.
In a closed loop, derivatives through generated inputs belong in the Jacobian.

Writing `F(s,x)-s` as a vector field defines an Euler surrogate at step one;
its exact ODE flow is not generally `F`. A continuous model is optional and
needs its own approximation or embedding assumptions. No ODE embedding is
required for the discrete period-map analysis below. This corrects the proposed
conversion in `floquet_analysis/main.tex:175–210` without editing the source.

## When Floquet terminology applies

For a deterministic period `T`, verify both `x[t+T]=x[t]` and a recurrent orbit
`s*[t+T]=s*[t]`. Use the observed orbit's period, which may be a multiple of the
input period. Record burn-in, multiple initializations, relative/absolute cycle
residuals, and phase-wise Jacobian differences over successive cycles. Select
closure tolerances from numerical calibration, not from a desired conclusion.

Then `J[t+T]=J[t]` and the one-period tangent map is

\[
M_t=J_{t+T-1}\cdots J_t.
\]

Its eigenvalues are discrete Floquet multipliers. They describe local
perturbation evolution once per cycle, not the input/output frequency spectrum.
The period-map construction is standard cycle stability analysis; see
[Cvitanović et al., ChaosBook, Cycle stability](https://chaosbook.org/chapters/stability.pdf).
The following consequences are direct deductions for this project's maps:

- `|mu|<1` denotes modal contraction per period at the verified orbit. It does
  not imply monotone contraction within each period. Nonnormal products can
  amplify perturbations transiently; measure singular values and direct kicks.
- `|mu|=1` alone does not prove an oscillatory clock or bounded response.
  Defective unit-modulus blocks can grow polynomially. A multiplier of `+1`
  need not oscillate. Singular/defective maps need not admit a full eigenbasis.
- Forced discrete periodic orbits need not have a neutral multiplier. The
  neutral time-shift direction familiar from autonomous continuous flows is
  not an automatic property of a recurrence with fixed external phase.
- Repeating the same graph does not make its Jacobian constant: the state
  can differ between repeats. For AABB use `J3 J2 J1 J0`; replace it with
  `JB^2 JA^2` only if the equalities of the relevant Jacobians are verified.
- For a one-step rotation with frequency `m/T`, raising that operator to `T`
  gives a one-period multiplier `1`. Thus AABB does not predict four
  fourth-root-of-unity eigenvalues of its *one-period* map. If a nonzero
  multiplier is parametrized as `exp(alpha*T + i*omega*T)`, then
  `alpha=log|mu|/T` and `omega=(arg(mu)+2*pi*k)/T`; the branch is ambiguous.
- Spectral peaks of a trajectory are not equivalent to neutral tangent modes.
  A 3D PCA projection does not identify a clock or exclude one.

A useful falsification fixture is the analytically solvable forced scalar
recurrence `s[t+1]=a*s[t]+b*x[t]`, with `0<|a|<1` and periodic nonconstant
`x`. It has an attracting periodic response and multiplier `a^T`. A measured
periodic peak with no neutral multiplier is therefore compatible with correct
input-driven computation. This is a mathematical control, not an experimental
finding about GC-LSTM.

If cycle closure fails, label the object a finite-window tangent product and
report the residual. Do not average states to manufacture an orbit. For fresh
stochastic draws from a periodic distribution, use finite-time/conditional
Lyapunov analysis; replaying a fixed draw block defines a separate forcing
experiment and does not establish natural stochastic periodicity.

## Nonperiodic and stochastic sequences

For any finite window report singular values `sigma_i(P[t,L])`, amplification,
and `ell_i(t,L)=log(sigma_i(P[t,L]))/L` with units of inverse update. These are
finite-time rates. Repeated QR orthogonalization supplies long-horizon
Lyapunov estimates, with burn-in, horizon, orthogonalization interval,
precision and convergence checks. A partial tangent basis estimates only a
partial spectrum. Under identical replayed inputs, call the rates conditional
on that forcing; input-perturbation sensitivity is a different measurement.
Input-driven RNN Lyapunov methods and their distinction from autonomous
dynamics are developed by [Vogt et al.](https://arxiv.org/abs/2006.14123).

Raw long products can underflow or overflow; prefer JVP/VJP operations,
scaled products and QR/SVD methods. Check tiny cases with full Jacobians,
noncommuting known products and finite differences. Report zero singular
values as rank deficiency/negative-infinite rates at the stated precision;
do not silently replace them with arbitrary floors. Leading eigenpairs do
not constitute the full spectrum or prove all directions contract.

## Dimension measurement contract

For a specified representation `z`, form centered rows `Z` and covariance
`C=Z^T Z/(N-1)`. Record the sampling axis, centering reference, normalization,
window, phase balance, independent sequences, seeds, and feature dimension.
For nonzero `C` with eigenvalues `lambda_i>=0`, report:

- Participation ratio: `D_PR=(sum lambda)^2/sum(lambda^2)`.
- Covariance entropy dimension: `D_entropy=exp(-sum p_i log p_i)`,
  `p_i=lambda_i/sum lambda`. State that this uses covariance eigenvalues.
- `D90` and `D95`: smallest ranks explaining 90% and 95% of variance.
- Stable rank of `Z`: `||Z||_F^2/||Z||_2^2`, absolute variance `trace(C)`,
  spectrum and numerical rank at declared tolerances.

Participation ratio summarizes variance distribution, not the number of
independent task variables; covariance spectra also differ from recurrence
eigenvalue spectra. See [Hu and Sompolinsky, §2.2](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010327).
Entropy rank based on singular values of `Z` is a different quantity from
entropy rank of `C`; retain the naming convention from
[Roy and Vetterli](https://core.ac.uk/download/pdf/147929764.pdf).
For an all-zero representation report zero variance and undefined normalized
dimension. The local reference implementation may serialize numeric zero as
a sentinel only together with `normalized_dimension_defined=false`; that value
is not a defined normalized dimension and must be excluded from ratios or
comparisons requiring one. Retain unclipped eigenvalues
and diagnose materially negative covariance eigenvalues before any numerical
roundoff clipping.

Record hidden/cell separately and jointly; measure graph node-wise,
whole-graph and pooled readouts separately. A metric after pooling cannot
characterize the full state. On an exactly repeated orbit of period `T`,
time-only centered covariance has rank at most `T-1`, irrespective of network
width: repeated cycles increase rows without creating new independent states.
Use different sequences, phases and initial conditions to measure other axes.

Define tangent effective dimension separately, e.g. PR of
`sigma_i(P)^2`, with both absolute amplification and spectrum. A tangent map
can have small effective rank yet retain high rank numerically, or contract
uniformly without changing effective rank. Preserve both descriptions.

Compare matched sample counts and widths; plot subsampling curves. Different
unit rescalings can change covariance spectra, so report raw-state and declared
standardized sensitivity analyses. Global scalar amplitude decay alone does
not lower normalized PR. Probe lagged independent task variables with
train/validation/test separation; decodability does not establish causal use.

## Controls, interventions and inference

Start with trained versus untrained recurrent models, persistence, memoryless
input/readout baselines, a validated sufficient-memory/clock control, and
uniformly contracting versus nonnormal analytic fixtures. Match data and
sampling wherever possible. Include task-required memory manipulations,
sequence/time-order shuffles preserving appropriate marginals, phase-balanced
sampling, and initial-state synchronization under identical inputs.

For interventions, compare matched-norm perturbations along leading/trailing
state PCs, singular vectors, and random directions; include sham intervention
and readout sensitivity. Distinguish direct input ablation from state ablation.
Fixed/noisy input during a gap changes the forcing; report it as a new regime,
not as the original task. Evaluate retention on independently varied past
variables that cannot be recovered from the current input. PCA compression
must use training-fit components and be evaluated on untouched sequences.

Independent training/data realizations are replication units. Time steps from
one orbit are not independent replicates. Report paired seed effects and
uncertainty; use sequence/block methods for temporal data with a justified
block rule. Record unsuccessful seeds, missing runs and stopping decisions.
Exploratory correlation between dimension and performance is not a causal
claim. A general collapse claim requires measurement robustness, independent
replication, task-relevant information loss, and convergent intervention
evidence; otherwise use the narrower terms covariance concentration,
contraction, or synchronization.

## Milestones and modality expansion

1. **Graph recovery:** pin upstream code/data; obtain the legacy checkpoint or
   reconstruct GC-LSTM with deviations logged. Follow
   [graph reproduction](graph-reproduction.md).
2. **Measurement calibration:** verify known contracting, rotating, singular
   and nonnormal maps; validate the full-state graph adapter and output
   chronology on CPU before reserving accelerator resources.
3. **Graph replication:** bounded seed pilot, then separately reviewed
   counting, memorization and delayed-memory experiments. An absent or
   contradicted preliminary effect is an accepted outcome of replication.
4. **Matched cross-modal synthetic tasks:** encode the same latent periodic,
   delayed-copy and multi-variable memory task as graph snapshots, rendered
   frames, audio features, token streams and multivariate series. Hold latent
   difficulty constant and vary encoder modality separately. Synthetic
   agreement supports this controlled family, not arbitrary real data.
5. **Independent modality pilots:** recurrent vision (ConvLSTM/GRU on frame
   sequences), audio (LSTM/GRU on declared waveform/features), text (LSTM/GRU
   and explicitly specified recurrent state-space models), and time-series
   forecasting (LSTM/GRU/state-space baselines). Start one model and one task
   per modality, with resource requests justified by graph/calibration evidence.
6. **External validity and mechanism:** add natural data with chronological or
   source/group-disjoint splits and dataset-specific metrics, then test a
   targeted retention intervention against capacity/regularization controls.

For vision, keep sampling rate and object identity/occlusion histories; for
audio, disjoint speakers/sources and frame/hop units; for text, disjoint
documents/templates, token horizon and attention access; for time-series,
chronological splits, forecast horizon and known seasonal leakage controls.
Do not equate an attention model's cached history with a fixed recurrent
state, or assume a state-space implementation has a time-invariant Jacobian.
Each adapter must specify the complete state and update map. Use exact
periodic replay for Floquet studies; natural sequences usually require
finite-time analysis. Model/data choices remain prospective until pinned.

## Agent work and scientific acceptance

The coordinating researcher assigns bounded tasks with file ownership,
question, source/artifact inputs, output contract, resource envelope and
acceptance criteria. Suitable roles are source-reproduction, numerical
measurement, modality pilot, infrastructure, and an independent skeptical
reviewer. Parallelize independent preparation or runs; keep shared checkpoint
and data dependencies explicit. Agents can propose changes to the protocol;
the coordinator versions the protocol before running changed conditions.

Distinguish `planned`, `blocked-input`, `executed`, `artifact-verified`,
`interpreted`, `accepted`, and `rejected-claim`. A zero exit code advances only
execution status. A reviewer checks provenance, sample definitions, controls,
numerical validation, uncertainty and whether conclusions match scope.
Record both support and refutation. Neither a worker nor the coordinator may
raise budget caps, silently drop failed seeds, or repeat until a positive result.

Every accepted result references code revision and local patch hash, immutable
config, environment lock/container identity, model/data hashes, split manifest,
seeds, complete state shape, rollout mode, numerical settings, raw spectra,
metric definitions, uncertainty method, job/run IDs, resource usage, and reviewer
decision. Append evidence and decisions to the dated notebook; keep the tracker
current. The external template's cluster quotas and model-routing authorizations
belong to its project and are not inherited here.
