---
name: floquet-analysis
description: Design, implement, or review Floquet, finite-time tangent, Lyapunov, and dimension measurements for recurrent models in this repository, including graph-state adapters and cross-modal replication.
---

# Recurrent dynamics and representation dimension

Read [the research program](../../doc/auto/research-program.md) for the
measurement contract. For legacy graph experiments read
[graph reproduction](../../doc/auto/graph-reproduction.md); for provenance or
claims inherited from figures read [source inventory](../../doc/auto/source-inventory.md).

Choose the complete persistent state and output chronology before writing
derivatives. GC-LSTM graph states include coupled nodes, hidden/cell state and
all layers; the draft's `2H` shorthand is not sufficient unless its actual
architecture has a graph-level state of that shape. Record flattening order,
teacher-forced versus closed-loop mode, and inputs held fixed for derivatives.

Use discrete recurrence Jacobians directly. Verify input, state and Jacobian
periodicity before calling a product a Floquet monodromy matrix. If closure
fails or observations are fresh stochastic samples, use finite-window products
and conditional Lyapunov estimates. An Euler surrogate `F-s` does not establish
an exact continuous-time embedding.

Measure state covariance spectra separately from tangent eigen/singular spectra
and task behavior. Report PR, explicit entropy convention, variance-explained
dimensions, absolute variance, sample counts and normalization. All-zero
covariance has undefined normalized dimension; numeric zero in local JSON is
only a sentinel accompanied by `normalized_dimension_defined=false`. A repeated period-`T` orbit
has centered time-only rank at most `T-1`; that alone is not harmful collapse.

Calibrate step ordering, JVP/VJP, dense tiny-state products, finite differences,
precision and iterative-solver residuals with noncommuting, singular, rotating,
contracting and nonnormal fixtures before scientific use. A partial spectrum
supports only a partial claim. Preserve numerical failures and raw spectra.

Avoid these inherited draft inferences: periodic PSD peak implies a neutral
tangent mode; all multipliers below one imply lost clock or monotone within-cycle
decay; AABB forces fourth-root eigenvalues of the one-period map; repeated input
means equal Jacobians; a PCA shape identifies an internal mechanism. Use the
forced scalar contracting recurrence from the research program as a counterexample
control, and test retained independently varied history rather than relying on
current-input phase recognition.

State hypotheses and falsification controls before held-out analysis. Match
sample counts and seed blocks; do not count cycle steps as independent replicates.
Fit PCA/probes on training data and evaluate untouched sequences. Separate
information decodability from causal task use with matched perturbation/sham
controls. Scope conclusions to the validated model/task/forcing regime.

Workers return immutable config/code/environment/data/checkpoint references,
state shape, raw metrics/spectra, closure and numerical diagnostics, run/job IDs,
resource usage, deviations and a proposed interpretation. Independent scientific
review checks those artifacts before accepting a claim. Verified negative
evidence is a valid outcome. Follow the repository's experiment and cluv workflow
for compute; this skill neither grants cluster resources nor transfers quotas
from the external template. Preserve inherited manuscripts and figures.
