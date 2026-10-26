# Query-averaged acquisition v5: prospective local discovery protocol

Prepared 2026-10-13. This is a separately admitted acquisition intervention on
the **original** sixteen-WRITE, random-terminal-QUERY task. Preparation,
zero-update calibration and reduced fixtures authorize no production fitting.
The coordinator must commit reviewed source, create a hash-bound execution
freeze, and admit the four-fit run separately. Scientific acceptance remains
false until independent result reduction and coordinator review.

## Question and interpretation

At a common initial parameter tensor and common sequence of independently drawn
sixteen-value memories, can replacing one sampled QUERY loss with its exact mean
over all eight required QUERY coordinates improve acquisition of the original
task? The public model, encoder, head, WRITE chronology, target distribution,
native complete state and coordinate competence gates stay unchanged. This
changes QUERY supervision exposure and removes conditional QUERY sampling noise
at fixed weights; it does not promise the same optimizer trajectory.

For deterministic recurrence, additive per-query loss, and a uniformly sampled
terminal QUERY independent of the memory, the eight-loss mean equals the
conditional expected sampled-QUERY loss at fixed parameters. Differentiation
through the complete sixteen-WRITE graph gives the corresponding conditional
expected parameter gradient. The conditional QUERY-gradient covariance becomes
zero for this exact average; randomness in memories remains. These fixed-weight
identities do not imply equal clipped gradients across sampled optimizer steps,
equal AdamW trajectories, equal walltime, or an empirical learning improvement.

The v4 singleton-address results motivated this intervention but provide no
original-task eligibility. All eight original coordinates must pass on both
paired R0 arms. A v5 success would be a seed-0 discovery candidate, not a
population guarantee, a confirmation result, or evidence of periodicity harm.

## Frozen training contract

The prospective configuration is
`configs/experiments/query-averaged-acquisition-v5.json`. It declares the four
GRU/LSTM × categorical/continuous cells, each fresh seed 0, hidden width 128,
encoder width 64, batch 32 paired arms from 16 independently sampled memories.
Initial tensors must equal the four pinned v3 fresh initial checkpoints from
model/data source `999d88d8679c0f353c6bb7fb4f5ab8433a7e730c`; exact file and tensor
hashes are in the configuration. Optimizer state is empty for every fit.

Every fit uses new AdamW with LR 0.001 and weight decay 0.01. Each step computes
the sixteen-WRITE prefix **once with its graph connected**, clones its complete
native recurrent state into eight independent one-QUERY branches, applies the
unchanged native head, and averages their native losses. Each branch starts from
the same post-WRITE state. A branch never starts from another branch's state.
For LSTM both hidden and cell tensors retain their gradient paths. There is one
backward call, one global gradient-norm clip at 1, and one AdamW update after the
loss mean. Categorical loss is native cross entropy. Continuous training loss
is **raw MSE**, with no population-variance normalization in backpropagation.

The public differentiable helper is
`average_query_loss(model, prefix, queries, targets)`. Shapes are respectively
`[32,16,38]`, `[8,32,38]`, `[8,32]`. It returns the mean loss and graph-connected
query logits, complete post-query states, eight raw losses and complete prefix
state. Inputs and model are CPU; categorical targets are integer class labels,
continuous targets share the native floating dtype. No diagnostic state-capture
helper participates in the training graph.

There are exactly 5,000 updates per cell, evaluated at every 500 updates, for a
maximum 20,000 optimizer updates. Committed-source freeze checks precede the
engine clock. The common 1,200-second engine deadline covers runtime source
recording, reference preflight, prehashing, helper calibration, baseline replay,
training and native evaluation. Failure-artifact finalization preserves the
realized elapsed time without admitting another optimizer update. CPU threads
are 1; training is pure FP32, Torch
2.6.0, deterministic algorithms, TF32 disabled, highest float32 matmul precision.
The current step-5,000 checkpoint is the endpoint. Early passes are descriptive
only. There is no early success stop, best-checkpoint selection, new LR, rescue,
budget extension, replacement fit, held-out evaluation, or periodic body.

## Paired data and prospective measurement

Training regenerates the exact original v3 base stream using seed 2026101301.
It draws the same sixteen payloads **and consumes the original sampled QUERY
draws** at every batch. Those QUERY draws remain provenance and do not select
the averaged loss. All eight QUERY views are derived without consuming further
randomness: each is a value-free terminal QUERY one-hot selector 0 through 7;
its target comes from the corresponding WRITE. A full original render and
target oracle rejects changed event/address features, wrong targets, payload
leakage, wrong chronology or non-identical paired prefixes.

Before any optimizer step the runner hashes every original base batch and its
rolling v3 stream prefix against the pinned manifests, derives all eight QUERY
views and targets, and hashes the entire prospective query stream. Actual
training regenerates each batch and verifies both original and derived hashes.
The shared prefix is reused computationally within a step; this changes neither
the memories nor their original order. Both architectures receive identical
task-specific histories, targets and query-view hashes.

Validation is a new original-constructor R0 manifest, seed 20005, with 128
independent memory draws per coordinate per arm: 1,024 independent episode
pairs, represented as 2,048 arm rows. A pair's arms are identical at R0 and are
not counted as independent replicates. The unmodified original scorer requires
accuracy ≥0.95 for every categorical coordinate, or MSE/(1/3) ≤0.05 for every
continuous coordinate, on both arms, with its existing four-ULP comparison
allowance. Native evaluation includes the terminal QUERY update, captures
post-WRITE, pre-QUERY and post-QUERY complete states, and verifies uninterrupted
logits against the phase capture. LSTM packing is hidden then cell, width 256;
GRU packing width is 128. The frozen validation arrays and identities are
verified at every scheduled evaluation.

The four pinned v3 **fresh step-5,000** checkpoints are replayed without training
on these exact new validation histories. They are not selected again. All eight
coordinate comparisons are episode-paired using the same targets and memories;
categorical contrasts use error indicators, continuous contrasts use squared
error divided by population variance 1/3. The report preserves 128 individual
differences per coordinate and their mean. No CI or p-value is computed in this
single-seed discovery. Historical seed-20001 v3 scores may be described
separately and are never pooled with the new paired contrasts.

## Calibration and unequal compute exposure

Before fitting, each actual 128/64/batch32 helper is checked in FP32 and FP64
using the same initial FP32 tensors and inputs represented in each dtype.
Its native logits, complete post-QUERY states, loss and **every parameter
gradient** must match the arithmetic mean of eight separate uninterrupted
seventeen-row native forwards. Each parameter tensor must have a nonzero
reference gradient. Elementwise bounds are `1e-6 + 1e-5*abs(reference)` in FP32
and `1e-12 + 1e-10*abs(reference)` in FP64. FP32 versus FP64 gradients must also
meet the declared FP32 bound. Raw gradients, logits, complete states, inputs,
targets and checks are retained in ignored calibration artifacts.

Negative calibration controls detach the prefix, chain QUERY branches,
omit LSTM cell state, or omit the factor 1/8. Every applicable control must be
detectably wrong under the same bounds. These controls never train. An
independent reviewer additionally checks actual-size native gradients and FP64
finite differences; its evidence is distinct from producer calibration.

Each v5 update exposes 16 independently sampled memory histories, 32 arm
instances and 256 QUERY-branch loss instances, versus 32 sampled-QUERY loss
instances in v3. The nominal recurrence-row count is 32×(16+8)=768 versus
32×17=544, a ratio 24/17, **not** a measured FLOP, memory, walltime or speed ratio.
Eight-query losses and their backpropagation also cost extra work. The primary
discovery comparison matches optimizer updates and prefix-memory exposure; it
does not match QUERY supervision or compute. All realized counts, walltime and
useful optimizer updates per second are reported. No efficiency claim follows
from nominal operation counts.

## Admission, artifacts and failure preservation

Production CLI has no fixture or budget override:

```sh
PYTHONPATH=src /absolute/project/.venv/bin/python \
  -m representation_collapse.query_acquisition \
  --config configs/experiments/query-averaged-acquisition-v5.json \
  --freeze-receipt /absolute/project/runs/ops/query-acquisition-v5-freeze.json \
  --output /absolute/project/runs/query-acquisition-v5-20261013/experiment
```

Before output creation or weight loading, the runner requires an external
`query_acquisition_v5_freeze` receipt declaring local authorization, exact clean
committed source SHA, config path/hash and an accepted independently reviewed
implementation/config hash plus review receipt hash. Executed source and config
bytes must equal their committed Git objects. The coordinator owns this receipt
and actual execution admission. An isolated clean archived source with copied
pinned ignored references allows coordination documents to change elsewhere.
No cluster submission or provider call is part of v5.

An output directory is create-only. Atomic manifests bind source/lock hashes,
precision, original reference hashes, validation and query-stream identities,
calibration results, baseline native arrays, every scheduled checkpoint/history
and endpoint scalar and episode-level contrasts. A deadline preserves partial
history and a clearly censored **current-weight** checkpoint at the exact number
of optimizer steps that returned, even between scheduled evaluations. A
returned update producing nonfinite weights is counted before the finite check,
preserved with an explicit nonfinite flag, and marked unusable. An optimizer
step that itself raises is not fabricated as completed. Such artifacts carry
unknown endpoint competence, no selected checkpoint and no resume authorization.
Completed earlier fits and unstarted cells remain explicit. No failure is
discarded or replaced.

The public `run_fixture` requires an explicit `FixtureBudget` that reduces at
least update count, fit count or walltime below the production envelope. It
retains actual model dimensions, original data semantics and calibration but
tags the scorer split as `fixture`, with zero trained-seed evidence units and
no validation eligibility. Fixtures test infrastructure only. Tests cover the
original oracle and RNG, all-query/full-native gradients, source freeze,
single clipping/update, paired population units, actual censored/nonfinite
steps, create-only paths and completion despite an early descriptive pass.

## Decision after independently verified execution

Report every fixed endpoint, scheduled trajectory, coordinate metric and paired
baseline difference, including negative outcomes and numerical/time failures.
Any original-task candidate must meet all eight original coordinate gates;
none inherits v4 singleton eligibility. Follow-up seeds, confirmation assays,
equal-QUERY-supervision controls, compute-matched comparisons and later body
experiments require separate prospective protocols. Acquisition alone supports
no conclusion about dimension collapse, tangent contraction, harmful memory
loss, causal lag/capacity, stable cycles, Floquet multipliers or modality-wide
replication.
