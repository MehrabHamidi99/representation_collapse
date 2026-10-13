# Query-averaged acquisition v5 — independent admission review

2026-10-13. The prospective design and exact implementation pass independent
review with no remaining blocker. This admits preparation of the declared
four-fit local discovery run after clean source commit and an external
coordinator freeze. It is not empirical acquisition evidence; scientific
acceptance remains false and belongs to the coordinator.

## Why the estimator targets the original objective

The original batch has sixteen independent WRITE histories duplicated into
32 identical R0 arm rows. Given the entire prefix batch, its sampled gradient
is `(1/16) Σ_i g_i,Q_i`, with a uniform independent selector for each history.
Per-example separability gives conditional mean
`(1/16) Σ_i (1/8) Σ_q g_i,q`. Averaging eight reference batches, each having a
common query address, produces this same value. There is no need to enumerate
all `8^16` selector combinations.

The native Linear/Tanh encoder, GRU/LSTM core and Linear head operate separately
on each row; there is no batch-coupled normalization or stochastic dropout.
Cross-entropy and raw MSE are additive per-example losses. Consequently the
global common-query branch average equals the conditional expectation under
the original independently sampled selectors, for both loss and the complete
parameter gradient. LSTM hidden and cell state retain their paths through all
sixteen WRITEs. The result concerns fixed weights and a uniform selector
independent of the prefix. Conditional QUERY-noise covariance is removed under
finite second moments; memory sampling remains random.

An independent FP64 check exhausts all 64 combinations of two independently
selected queries, using two histories replicated into batch32, for all four
model/task cells. Losses, per-row native logits and gradients match the
eight-branch average; maximum gradient error is 6.245e-17.
The extension to sixteen histories follows from the algebra above, not an
exhaustive numerical enumeration. Clipping and AdamW are nonlinear; neither
clipped-update parity nor equal optimization trajectories follows.

## Exact source and numerical review

Accepted implementation SHA-256:
`f2af2c73088731e3664b7ca59c54ed9c9f083dc863b83881b20956ffb429a19d`.

Accepted configuration SHA-256:
`d6d89b431942c8e786f1ac169d5fee09ef51e8f85f2030cee1ec8bcdd9d4289e`.

Accepted tests SHA-256:
`f925e4b474bfdd8ef86f69253ad423a19050a4a5b4a9a3e74db4e4a88ea679ad`.

Accepted protocol SHA-256:
`cc626ed99e3c5c38385fc1a3f7ba1594ed89b5526b57c6b05d77551a710707c5`.

At hidden128/encoder64/batch32, an independently written audit compares the
actual training helper against eight separate native seventeen-update
trajectories in FP32 and FP64: eight cases, all 64 parameter-tensor checks,
and complete native post-QUERY states. Each of eight QUERY branches starts
from the same graph-connected complete WRITE origin; the prefix is processed
once and never detached or chained. All checks meet the frozen elementwise
tolerances. Thirty-two independent FP64 per-tensor central finite differences
with step `1e-5` pass; maximum derivative error is
`4.187737e-11`.
These are zero-update synthetic calibrations.

Direct RNG reconstruction checks both original 5,000-batch streams and both
derived all-query streams: 10,000 base and 10,000 all-query batches. The old
QUERY draws are consumed, all WRITE payloads match, all eight targets come from
their addressed original payloads, and sampled-query metadata remains intact.
Every batch and rolling hash matches. Eighteen pinned reference/stream files
also match their configured identities.

Independent raw reduction of the producer's final actual-size calibration
checks eight cases, 64 parameter tensors, twelve manifest files and 28 negative
controls. Detachment, sequential QUERY chaining, omitted cell and omitted mean
normalization are detected. FP32 versus FP64 maximum gradient difference is
`1.074220e-8`, within the
declared FP32 bound. Producer evidence is distinct from this reviewer’s checks.

All eleven focused tests pass, zero skips, in 6.996 seconds. Temporary test
outputs were redirected into the reviewer’s ignored scope. Five returned
optimizer steps occurred only in explicitly reduced infrastructure fixtures;
they are not production fits or empirical GateB evidence. Thirteen independent
configuration/full-budget rejection checks pass. Controlled bad-coordinate
fixtures fail the all-eight gate even when a pooled score would pass.

## Admission and measurement boundaries

Production requires an external freeze binding this review, exact code/config
hashes and clean committed source before output creation or weight loading.
The fixed envelope is four fresh seed-0 fits at 5,000 updates each, at most
20,000 returned optimizer updates and a 1,200-second engine deadline. The
reviewed source uses one mean-loss backward pass, one global clip and one new
AdamW step per update. Empty optimizer state and pinned initial tensors remain
required. First passes are descriptive and do not select replacement weights.

The protocol was clarified during review: committed-source freeze checks
precede the engine clock. Runtime source recording, reference preflight,
prehashing, calibration, baseline replay, fitting and evaluation are timed.
Failure-artifact finalization records elapsed time and grants no further
optimizer updates. Deadline/nonfinite paths preserve current parameters and
actual completed-step counts, with endpoint competence unknown and no resume
authority.

The original coordinate scorer remains intact: all eight categorical
accuracies must reach 0.95, or every continuous MSE/(1/3) must be at most 0.05,
on both arms. New validation seed20005 supplies 128 independent histories per
coordinate, 1,024 paired histories per task; its 2,048 arm rows are duplicate
copies. Task-specific histories are shared across architectures, checkpoints
and pinned baseline reevaluations. The four original v3 fresh step5,000
baselines are reevaluated on the exact same new histories and supported QUERY
coordinates0–7, without optimization or selection. Episode-level paired native
errors retain population units; old validation scores remain separate.

Equal optimizer updates and prefix histories do not mean equal supervision or
compute. Each update evaluates 256 QUERY-loss rows versus 32 in v3; nominal
recurrence rows are 768 versus544. The ratio24/17 is an operation count, not a
FLOP, speed or memory result. One discovery seed supplies no population
guarantee, uncertainty interval or independent trained replication.

A verified all-eight **R0 acquisition** pass would release only that
prerequisite. V5 does not test overall **GateB R0/R1/R2 eligibility**. It also
tests no periodicity forcing, dimension loss, tangent contraction, causal
memory intervention or Floquet closure. New results require a separate
independent raw/native reduction and coordinator acceptance.

The hash-bound admission receipt is
`runs/review/query-acquisition-v5/implementation-review.json`; executable
independent audits and raw calibration evidence live beside it. Earlier
attempts are preserved. No author source, configuration, tests, prior results
or scientific protocol was changed by this reviewer.

