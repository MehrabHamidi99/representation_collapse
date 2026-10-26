# Research programme: periodic forcing, recurrent memory, and representation collapse

Date: 2026-10-11. Status: research plan documented; experiments have not been
run. This document records the research path developed with the project owner.
Experiment-specific configurations, acceptance checks, and compute envelopes
must be frozen before execution.

**Literature reassessment, later on 2026-10-11:** the
[novelty audit](../literature/novelty-review.md) identifies direct antecedents
and recommends a small mechanism study before this plan's broad screen.
Its analytic ordering nulls, noise assumptions, horizon-specific arm contrasts
alongside the interaction, and cue-sample rank ceiling must be incorporated
when freezing the implementation. The breadth below remains proposed coverage;
the audit does not constitute a new executable configuration or compute approval.

Related documents: [measurement foundations](research-program.md),
[graph reproduction](graph-reproduction.md), [source inventory](source-inventory.md),
[compute operations](compute-plan.md), and [current state](../report/TRACKER.md).

## 1. Scientific objective and starting point

**The central question is: when does repetition remove information that a
recurrent model needs, and what dynamical mechanism explains that loss?**

The programme will connect three separately measured quantities:

- **Representation geometry:** how many directions the observed states occupy.
- **Memory and behaviour:** whether the state retains information required to
  answer correctly.
- **Dynamics:** how perturbations and differences between histories propagate
  through recurrent updates.

These quantities need not move together. Periodic forcing can stabilize
distinct memory-carrying attractors: published RNN experiments already
demonstrate retained memories in locally contracting, phase-locked cycles.
Repetition-dependent attractor multiplicity is also established prior work.
The contribution must therefore explain the conditions separating forgetting,
useful compression, and stable memory.
[Pals, Macke and Barak, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10868787/),
[Ashwin and Ceni, 2023](https://arxiv.org/abs/2309.04728)

### Locked research choices

- Study fixed model weights during inference first.
- Prioritize mechanism and theory, supported by broad experiments.
- Begin with matched synthetic tasks across modalities.
- Study both continuous-valued and categorical memory.
- Cover classical RNNs, gated recurrence, state-space and matrix-memory models,
  and looped transformers.
- Treat temporal recurrence and repeated transformer depth as separate
  experimental axes.
- Establish controlled compact-model results before native and pretrained
  validation.

### Current evidence

The repository contains preliminary graph manuscripts and figures, research
infrastructure, and numerical calibration. It does not yet contain a reproduced
trained-model collapse result. Recover the original experiment code, data
configuration, checkpoint, and raw spectra before treating the inherited
GC-LSTM observations as verified.

The source audit must resolve the AB/AABB task ambiguity, complete graph-state
definition, and multiplier computation. Preserve the inherited manuscripts;
write corrections and new findings separately.

### Questions to preregister

1. Does periodic repetition produce additional forgetting compared with
   nonperiodic inputs matched for horizon and low-order statistics?
2. Is that forgetting explained by attenuation of task-relevant directions,
   concentration into fewer directions, or movement between memory attractors?
3. Which memory mechanisms remain robust under repetition?
4. Does the mechanism transfer across renderings of the same latent task?
5. How does persistent access to the original input change the interpretation
   of convergence in looped transformers?

The novelty review must also cover recent work on memory geometry and
temporally correlated inputs before making theoretical priority claims.
[Temporal superposition, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/bfcb583d225b1db8d3ca2331f6785774-Abstract-Conference.html),
[Temporal correlations in linear RNNs, 2026 preprint](https://arxiv.org/abs/2609.00420)

## 2. Theoretical programme

### A. Establish the discrete dynamical framework

For the complete recurrent state,

$$
s_{t+1}=F_\theta(s_t,x_t),\qquad
J_t=\partial_sF_\theta(s_t,x_t),
$$

define

$$
\Phi_{b,a}=J_{b-1}\cdots J_a.
$$

Use $\Phi$ as a finite-time propagator on arbitrary trajectories. Call a period
product a Floquet monodromy matrix only after verifying periodic forcing and
closure of the reference state trajectory. Allow responses with period $kP$,
multiple attracting responses, and failure to reach a periodic orbit.

For a verified period-$P$ orbit, report multiplier rates as
$\log|\mu|/P$ per recurrent update. Do not infer neutral directions from a
periodic output, or infer equal Jacobians from equal consecutive inputs.

### B. Separate contraction, dimensional concentration, and driven variability

Develop exact affine examples before nonlinear claims:

- A stable periodically forced affine system has an attracting periodic
  response.
- With independent cycle innovations, its stationary covariance satisfies a
  discrete Lyapunov equation.
- Covariance rank depends on which directions the input/noise excites, as well
  as the dynamics.
- Uniform contraction $M=aI$ reduces amplitude without reducing normalized
  participation ratio.
- An isolated dominant mode can concentrate propagated perturbations into one
  direction even when its magnitude grows.
- Nonnormal dynamics can amplify perturbations transiently despite asymptotic
  stability.

These examples become executable counterexamples to invalid interpretations.

Also formalize the sampling limitation: an exactly repeated period-$P$
trajectory has centered time-only covariance rank at most $P-1$. This bound
alone says nothing about retained task information.

### C. State precisely when continuous memory requires dimension

Suppose a smooth state encoding $E(u)$ permits exact smooth recovery of a
continuously varying $d$-dimensional memory:

$$
g(E(u))=u.
$$

Then $\operatorname{rank}DE(u)\ge d$.

If those encodings form a smooth family of states preserved by the same cycle
map,

$$
\mathcal P(E(u))=E(u),
$$

then

$$
D\mathcal P\,DE=DE.
$$

Consequently, at least $d$ independent tangent directions have multiplier $+1$.

State the restrictions prominently: this concerns exact continuous memory on
a preserved manifold. It does not apply directly to finite categorical codes,
approximate storage, or finite-horizon performance. Categorical memories can
occupy distinct, strictly attracting states or cycles.

### D. Connect finite-horizon memory to a noise-resolved task channel

For an exact affine Gaussian calibration model,

$$
z=Hu+\eta,\quad
u\sim\mathcal N(0,I),\quad
\eta\sim\mathcal N(0,\Sigma),
$$

derive

$$
\operatorname{Cov}(u\mid z)
=(I+H^\top\Sigma^{-1}H)^{-1}.
$$

During storage without cue reinjection, factor the channel as $H=C\Phi G$:
cue injection, recurrent propagation, and observation/readout.

Measure the complete-state channel and trained-output channel separately. A
weak task head does not establish that the state forgot. For nonlinear models,
tangent-channel measurements are local sensitivity diagnostics; validate them
against finite cue changes and actual behaviour.

The theory should predict retention from task-channel singular values, their
alignment with the readout, and a specified noise scale—not from covariance
participation ratio alone.

### E. Account for input reinjection in depth loops

For

$$
z_{\ell+1}=G_\ell(z_\ell,x),
$$

input sensitivity obeys

$$
S_{\ell+1}=J_\ell S_\ell+B_\ell.
$$

Thus contracting sensitivity to the initial state can coexist with strong
sensitivity to an input supplied at every loop. At a suitable tied fixed point,

$$
S_*=(I-J_*)^{-1}B_*.
$$

Use $z_{\ell+1}=az_\ell+(1-a)x$ as a calibration: initial-state influence
vanishes while the final representation preserves $x$. This distinction is
also central to equilibrium models.
[Deep Equilibrium Models](https://arxiv.org/abs/1909.01377)

**Theoretical acceptance:** each proposition needs explicit assumptions, a
proof, a numerical fixture, a counterexample showing its limits, and a
corresponding empirical prediction. Standard results serve as foundations;
novelty claims require independent literature review.

## 3. Experimental programme

### A. Build one latent task with several renderers

Each episode contains:

1. Sixteen WRITE observations carrying independently sampled memory values.
2. A distractor body whose repetition structure is manipulated.
3. A terminal query selecting a previously written value.

The primary task queries eight designated coordinates; the other eight provide
nuisance information. Sensitivity studies use $d\in\{1,4,8,16\}$ while retaining
the same prefix length.

Run two linked tasks:

- **Categorical:** sixteen equiprobable values per coordinate.
- **Continuous:** values sampled uniformly from $[-1,1]$.

Render categorical episodes as vectors, temporal graphs, images, audio frames,
and tokens. Render continuous episodes as vectors, graphs, images, and audio.
Hard-token experiments remain finite-information experiments; continuous-memory
lower bounds will not be applied to them.

Use identical latent episode manifests across modalities. Validate that the
renderer preserves the intended information. Pretrain each observation encoder
on latent-observation recovery, reuse its initialization across architectures,
and then apply the same end-to-end training protocol.

Cue values, query indices, nuisance variables, and horizons must be independent.
The distractor and query must never reveal the required answer.

### B. Use a controlled repetition intervention

Use the following primitive twelve-symbol motifs:

- `AXAXAXBXBXBX`
- `AXAXBXAXBXBX`
- `AXAXBXBXAXBX`
- `AXBXAXAXBXBX`
- `AXBXBXAXAXBX`

They have identical symbol counts, cyclic bigram counts, run lengths, and
start/end context. Expand each symbol into $q$ identical observations.

- **Periodic arm:** repeat one selected motif.
- **Nonperiodic arm:** vary motifs while matching the final motif; reject
  accidental proper whole-body periods.
- Pair the arms by memory prefix, nuisance variables, total updates, and query.
- Provide no arm flag or schedule identifier.

This intervention changes higher-order repeatedness while matching specified
lower-order statistics. It does not isolate repetition from every aspect of
predictability.

Primary settings:

- $q\in\{1,2\}$, giving temporal forcing periods $P=12q$.
- Train on $R\in\{1,2,4,8,16\}$, with balanced periodic and nonperiodic arms.
- Treat $R=1$ as a shared anchor: the two arms are identical.
- Evaluate $R=32,64$ separately as horizon extrapolation.

The primary behavioural contrast is:

$$
\Delta_E=
(E_{\mathrm{periodic},16}-E_{\mathrm{periodic},2})
-
(E_{\mathrm{nonperiodic},16}-E_{\mathrm{nonperiodic},2}).
$$

Positive $\Delta_E$ indicates additional deterioration under the specified
repetition intervention. Deterioration shared by both arms is ordinary
horizon-dependent forgetting.

Add secondary tests of matched total horizon with different periods, input
amplitude, phase shifts, and fresh stochastic rendering noise. Fresh noise
defines a stochastic-forcing experiment; it cannot inherit ordinary Floquet
assumptions.

### C. Cover the architecture families explicitly

The mandatory streaming registry contains:

- Tanh RNN and orthogonal RNN.
- GRU and LSTM.
- sLSTM and mLSTM.
- S4D and Mamba2.
- RWKV7 and GatedDeltaNet.
- A block-recurrent transformer with explicitly bounded memory.

A shared-block depth-looped transformer is a separate entry. Modern families
use pinned primary implementations and documented state semantics.
[xLSTM](https://arxiv.org/abs/2405.04517),
[Mamba2](https://arxiv.org/abs/2405.21060),
[RWKV7](https://arxiv.org/abs/2503.14456),
[GatedDeltaNet](https://arxiv.org/abs/2412.06464)

This provides broad family coverage; it does not imply coverage of every
published recurrent variant. Missing or failed adapters remain visible
milestones.

For the controlled study, use a 64-dimensional observation bottleneck, a
one-million-parameter ceiling for core plus readout, and a one-MiB
persistent-state ceiling per sequence in FP32. Resolve valid widths from a
declared grid before examining performance. Report actual parameters,
complete-state size, FLOPs, and runtime; these budgets are not interchangeable.

Use three pilot seed blocks. Apply an equal three-learning-rate search per
cell, select by balanced validation task loss, and freeze configurations before
confirmation. Default training uses AdamW, gradient clipping, full
backpropagation through the episode, and a common episode budget. Any
truncated-backpropagation experiment is a separately labelled intervention.

### D. Separate two depth-loop experiments

1. **State-carried memory:** encode the WRITE prefix into latent state once.
   Subsequent loops can access only tag-independent context.
2. **Persistent-access control:** every loop can reread the original
   tag-containing input.

For the controlled depth experiment, one tied block application is one loop,
and each loop reads a complete motif context. Repeating a fixed context
therefore has forcing period **one loop**, not twelve loops. Varying context
motifs supplies the matched nonperiodic control.

Train represented loop counts before evaluating unseen depths. Record all
depth embeddings, masks, mutable caches, and direct input-to-output paths.

A growing append-only cache changes the state dimension and retains old
entries explicitly. Analyse it separately from bounded-state recurrence.

After these experiments pass, add a vector-task hybrid that varies temporal
horizon and inner loop count independently. Use pretrained looped models,
such as Huginn, for later external validation of depth behaviour.
[Huginn recurrent-depth study](https://arxiv.org/abs/2502.05171)

### E. Stage the breadth instead of launching the entire matrix

The streaming pilot comprises:

- Eleven families × five categorical renderers × three seeds:
  **165 checkpoints**.
- Eleven families × four continuous renderers × three seeds:
  **132 checkpoints**.

Both depth regimes add **54 checkpoints**, giving **351 pilot checkpoints**,
excluding tuning, confirmation, native models, and targeted ablations.

Run vector tasks first, then remaining renderers in bounded batches. Expensive
tangent analysis and interventions concentrate initially on the confirmation
subset. Pending cells remain recorded rather than disappearing from the study.

### F. Reconnect to graph research and natural inputs

Recover and pin the original [T-GRAB code](https://github.com/alirezadizaji/T-GRAB)
and [dataset](https://huggingface.co/datasets/Gilestel/T-GRAB). Reproduce GC-LSTM
prediction performance before interpreting its dynamics. Report change-point
performance alongside aggregate F1 and persistence baselines.

Then evaluate native candidates:

- Graphs: GC-LSTM, T-GCN, EvolveGCN, and DCRNN for traffic inputs.
- Vision: ConvGRU, ConvLSTM, and PredRNN.
- Audio: causal convolutional encoders with GRU/LSTM and state-space recurrence.
- Text: recurrent language models and looped transformers from the registry.

“Best” means strongest held-out validation performance within the declared
budget. Selection must not favour architectures because they exhibit collapse.

Natural-input validation uses repeated snippets from
[METR-LA](https://github.com/liyaguang/DCRNN),
[CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html),
[Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands),
and [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext).
Preserve source-level splits before constructing episodes.

These are natural-input stress tests with controlled memory demands. Keep
them distinct from ordinary forecasting, classification, or language-modelling
performance. A checkpoint never trained to retain an arbitrary cue cannot
establish harmful forgetting merely by failing that new task.

## 4. Measurement, testing, and scientific acceptance

### Complete-state interface

Add a common adapter contract providing initialization, a pure recurrent step,
task readout, state packing/unpacking, and metadata describing every persistent
component.

Include hidden and cell states, convolution histories, state-space buffers,
matrix memories, bounded attention caches, and evolving weights where relevant.
For a one-layer node-state graph LSTM, the state dimension is $2NH$, not $2H$.

The interface must also distinguish external context from recurrent state and
define when the query updates the state. Primary memory measurements occur
immediately before the query.

### Geometry and information measurements

For each fixed driver history, vary required memory cues and construct a
separate state cloud. Hold irrelevant cue coordinates and rendering nuisance
fixed within that cloud. Average cloud-level measurements afterward.

This prevents different nonperiodic input histories from artificially
inflating the control covariance.

Report:

- Covariance participation ratio, entropy effective rank, variance-explained
  dimensions, total variance, and spectra.
- Complete-state and readout-visible representations separately.
- Between-history distances and finite counterfactual cue differences.
- Held-out linear and nonlinear probe performance.
- Actual query accuracy or normalized MSE.
- Continuous cue-to-state and cue-to-output sensitivities.

Covariance dimension measures linear spread, not intrinsic manifold dimension.
Zero variance has undefined normalized dimension. Fit probes and projections
on separate data; probe failure alone does not prove information absence.

Use at least 1,024 paired behavioural episodes per primary cell. Geometry
starts with eight fixed driver pairs and 512 cue histories per driver, with
sample-size convergence checks on selected cells.

### Dynamical measurements

- Verify orbit closure separately for different histories and initializations.
- Compute Floquet spectra only for verified periodic responses.
- Otherwise use finite-time singular values and conditional Lyapunov estimates.
- Examine task-relevant propagation as well as unrestricted tangent directions.
- Preserve absolute amplification alongside normalized tangent dimension.
- Label partial spectra explicitly; leading eigenvalues cannot certify global
  contraction.

Noise robustness assays use calibration-fixed scales and locations. Transform
the noise covariance consistently when changing state coordinates; equal
numerical noise across arbitrary architectures is not automatically comparable.

### Causal checks

Use complete-state swaps between matched histories, same-phase early/late
state transplants, and task-sensitive subspace perturbations.

Include sham swaps and random subspaces matched for rank and perturbation
norm. Swapping LSTM state means swapping both $h$ and $c$.

These interventions test causal use of stored information and possible rescue.
They do not, by themselves, prove that covariance dimension is the causal
mediator.

### Numerical acceptance tests

Require:

- Sequential versus equivalent batched execution.
- Reset, checkpoint restoration, deterministic replay, and chronology checks.
- Tiny FP64 dense products agreeing with JVP/VJP implementations.
- Adjoint dot-product checks and finite-difference step-size sweeps.
- Noncommuting, singular, rotating, nonnormal, contracting, and multistable
  fixtures.
- Randomized-estimator convergence and iterative-solver residual checks.
- Preservation of linearity when stabilizing long products.

Calibrate useful compression, uniform amplitude decay, categorical multistable
memory, continuous neutral memory, and depth-input reinjection as explicit
controls.

### Statistical acceptance

Three-seed pilots estimate feasibility and variability. Confirmation uses ten
fresh seed blocks and untouched test manifests.

Confirm the LSTM anchor and validation-selected strongest streaming model for
each renderer/task type. Confirm both depth regimes on vector tasks and
categorical text, plus one predeclared null or opposite-effect vector case
selected from pilot validation results.

Use paired seed-level effects and confidence intervals; episodes and time
steps are not independent training replicates. Apply Holm correction across
registered primary comparisons. Report all pilot cells and failures.

Interpret outcomes distinctly:

- Lower dimension with preserved performance: useful compression.
- Lower amplitude with stable normalized dimension: attenuation.
- Memory failure without dimension reduction: forgetting without demonstrated
  dimensional concentration.
- Contracting but distinguishable memory attractors: stable retention.
- Additional behavioural loss, task-relevant geometric loss, and consistent
  intervention evidence: supported harmful-collapse mechanism.

Short-horizon competence is required before attributing deterioration to
repetition. Default gates are 95% categorical accuracy or continuous normalized
MSE at most 0.05. Failed competence gates trigger bounded
implementation/optimization diagnosis and remain reported.

## 5. Execution milestones and deliverables

### Milestone 1 — Foundations and provenance

Complete the graph-source audit, theorem/counterexample document, task
generator, state contracts, and numerical fixtures. Graph asset recovery and
synthetic calibration proceed in parallel; missing legacy checkpoints do not
block independent synthetic work.

### Milestone 2 — Vector mechanisms

Run categorical and continuous vector pilots across the full registry.
Establish task competence, measure runtime and memory, and test whether
geometry, behaviour, and tangent propagation agree or diverge.

### Milestone 3 — Cross-modal coverage

Execute the remaining controlled cells. Analyse whether effects survive
changes in rendering and encoder, while retaining the same latent task and
paired histories.

### Milestone 4 — Confirmation and interventions

Freeze hypotheses and selected configurations, run fresh-seed confirmation,
and test memory-use and rescue interventions. Treat valid null results as
completed scientific outcomes.

### Milestone 5 — Native and pretrained validation

Reproduce the graph benchmark, evaluate native modality models, and test
selected pretrained recurrent systems. Separate frozen evaluation, supervised
adaptation, and full fine-tuning.

Update the research protocol, evidence DAG, tracker, and skills to encode these
stages. Each run records immutable source/environment/data/checkpoint
references, complete-state semantics, raw measurements, diagnostics, resource
usage, and deviations.

Use bounded subagent assignments for theory, model/measurement implementation,
and independent scientific review. The coordinator integrates evidence;
successful execution and scientific acceptance remain separate decisions.

Use cluv for Mila, Tamia, and subsequently verified Alliance clusters. First
profile one training-and-analysis cell, then prepare concrete resource
envelopes with job counts, walltimes, storage, retry limits, and stopping rules.
Submit only within an approved envelope; racing requires approval for every
candidate allocation. No GPU-hour estimate will be presented as established
before profiling.

**Completion criterion:** a reproducible account of when repetition preserves,
compresses, or destroys required information, with mathematical conditions,
calibrated measurements, controlled cross-modal evidence, and clearly bounded
claims.
