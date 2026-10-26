# Local competence feasibility — 2026-10-13

Four actual CPU fits passed immediate acquisition but failed the full-prefix R0
gate. Each stopped at update 1,000 under the frozen rule. No fit reached a
periodic distractor body, stage 5, or an eligible selected checkpoint. This
narrows the present bottleneck to prefix-memory acquisition under this training
schedule; it provides no test of excess harm from periodicity.

## Execution and fixed controls

The coordinator authorized four sequential local feasibility fits, with a
20-minute total ceiling and no engineering overrides. They ran from
2026-10-13 08:14:44 to 08:15:18 UTC, taking 33.915 seconds including interpreter
startup, manifest construction, evaluation and artifact writing. All four
processes returned exit 0, with their scientific stopping outcomes retained.
There were 4,000 actual optimizer updates. No cluster submission, GPU
allocation, provider call, rescue fit, or held-out evaluation was made.

Source revision: `1add2b6aba948714673f45f1b7854eff6212833d`. A `git archive`
snapshot lives in ignored `runs/day-competence-20261013/source/`; an independent
local Git index/object store provides that snapshot's identity without changing
the parent checkout. All fit provenance records report that exact revision and
a clean snapshot. The existing project Python environment was used with
`PYTHONPATH` pointing to the archived source. Python 3.11.13, Torch 2.6.0,
CPU device, one Torch/BLAS thread, and FP32 parameters were recorded. TF32 was
disabled, deterministic algorithms enabled, and matmul precision set to
`highest`. These are CPU timings, not measurements of GPU throughput,
occupancy, utilization or VRAM.

The [prospective v2 protocol](../../../2026-10-26/local/auto-research/competence-training-v2.md) and
[frozen configuration](../../../../../configs/experiments/competence-recovery-v2.json)
were applied unchanged: hidden 128, encoder 64, batch 32, AdamW with weight
decay 0.01, gradient clipping 1, seed 0 and learning rate 0.001. The declared
five-stage schedule was available, with all subsequent stages contingent on
the current boundary checkpoint. Configuration SHA-256:
`0ee6582ebb70116660ff67ed57327daf15abf270368ba77c0bbe70a23f4a8368`.
Lockfile SHA-256:
`588147aa66724c075fd44754db3ddf757b249eefcaf94c9154ddc49578c016dc`.

## Boundary results

At update 100, all eight required coordinates passed the fixed-batch diagnostic
in every fit. Categorical accuracy was 1.0 on each coordinate's four examples.
Continuous population-normalized MSE also passed every coordinate. Encoder,
recurrent core and output head all had nonzero recorded parameter changes.
That diagnostic is an overfit check, not independent validation competence.

At update 500, all eight coordinates passed fresh IID immediate-WRITE/QUERY
validation: 128 episodes per coordinate. GRU and LSTM categorical accuracies
were 1.0 throughout. The largest coordinate MSE/(1/3) was 0.000098845 for GRU
and 0.000603555 for LSTM, below the fixed 0.05 threshold. All fits advanced
to stage 2 using that current checkpoint.

At update 1,000, all eight R0 coordinates failed in both arms. Each arm had
128 validation episodes per coordinate, 1,024 paired episode units total.
R0 contains 16 WRITE updates followed by the native terminal QUERY; its
periodic distractor body has length zero. The two arm inputs and their observed
coordinate scores are identical here, so they are not independent replications.
Metrics below list coordinates 0 through 7 in order:

- GRU categorical, accuracy: 0.3359375, 0.1171875, 0.0625, 0.140625,
  0.0703125, 0.1015625, 0.0625, 0.09375. Fit wall time: 8.464 seconds.
- LSTM categorical, accuracy: 0.1640625, 0.0234375, 0.0546875, 0.1171875,
  0.0703125, 0.0390625, 0.0390625, 0.0625. Fit wall time: 9.002 seconds.
- GRU continuous, MSE/(1/3): 0.990439, 1.005578, 0.961714, 1.073715,
  1.076535, 1.044142, 1.030976, 1.122007. Fit wall time: 7.285 seconds.
- LSTM continuous, MSE/(1/3): 0.992708, 1.009010, 0.963648, 1.077428,
  1.072117, 1.046614, 1.027850, 1.121685. Fit wall time: 9.009 seconds.

The categorical acceptance threshold is 0.95 on every coordinate; the continuous
threshold is 0.05 after normalization by the fixed Uniform[-1,1] variance 1/3.
Each fit's outcome is `stopped_coordinate_gate_failure`, with exactly 500
immediate-stage and 500 full-prefix R0 updates. No historical best checkpoint
replaced the failing boundary. No candidate was selected, and stages 3–5 were
not executed. The result is one seed/LR discovery slice per cell, not a completed
three-LR discovery grid or a replicated trained-seed result.

## State, manifests and independent array reduction

All seven validation manifests were constructed and persisted at step 0, before
the first optimizer update. Sampling seeds were 10000 for immediate validation
and 10001–10006 for R0/R1/R2/R4/R8/R16. Sampling and input identity are shared
between architectures within a task, not between categorical and continuous
tasks. The verification receipt retains every manifest identity. The R0 array
identity is `58eb5bafe37ac36b95f0caa580a462f0bc42370e6a3b5573b6b02555bd2c76f7`
for categorical and
`7fec9441df0e72b8b7a47755c4108869468956565cb16fa4bc3da3e0b55255b3`
for continuous validation. Unreached horizons were persisted for provenance;
their validation outcomes were not evaluated.

Behavior is read from uninterrupted native terminal-QUERY logits. Raw arrays
preserve complete state after WRITE, before QUERY, and after QUERY: 128 hidden
coordinates for GRU; 256 coordinates packed hidden then cell for LSTM. At R0,
post-WRITE and pre-QUERY arrays are equal because the body has zero updates.
Producer chronology reports comprise 960 chunk reports and 7,680 individual
adapter checks, all passing their declared absolute 1e-6 plus relative 1e-5
criterion. The maximum comparison discrepancy across categorical reports was
1.90735e-6 and across continuous reports 1.19209e-7. A tolerance pass does not
establish bitwise equivalence, preservation of arbitrarily small information,
or a Floquet cycle.

The separate reduction script
`runs/day-competence-20261013/verify_artifacts.py` verified 140 output-file
hashes, the archived source/config identities, 1,000 contiguous update records
per fit, all stage counters and stopping outcomes, and 96 coordinate metrics
recomputed directly from raw native logits/targets. Each coordinate had the
required 128 evaluation episodes. This reduction checks producer chronology
reports and state shapes. A second, independent implementation replayed the
100-update diagnostic and both stage-boundary checkpoints directly through the
archived native recurrent core and output head, without calling the producer's
capture, rendering, metric or competence-evaluation helpers. It checked 12
checkpoints, 139,264 logit entries and 9,437,184 complete-state entries against
the saved arrays, plus 16,384 episode targets against the addressed WRITE payload
and terminal QUERY. All passed. Coordinator/reviewer acceptance remains separate.
No covariance, tangent, intervention or periodicity result was inferred.

Ignored receipt:
`runs/day-competence-20261013/verification-receipt.json`, SHA-256
`3a4cedd957a7acbaa1883c984ad0438e9322140abd6eb1a28160d7a1af03d214`.
The receipt binds runner/source/lock/config identities, per-fit provenance,
metrics, complete history, validation IDs, manifests, timings and raw reductions.
The ignored runner refuses duplicate attempts or overwriting existing fit paths.
Native/input-target oracle receipt:
`runs/day-competence-20261013/native-oracle-replay-receipt.json`, SHA-256
`5d5b4230c98c55a981faddea09d5023d1bf1934444bbfd08b1668fe1113191e9`.

## Decision and remaining work

These results preserve a valid optimization/acquisition failure. They show that
immediate-query learning is working under the tested settings, while the
specified 500-update full-prefix R0 training stage does not produce competent
memory at its required boundary. The mechanism of that failure remains open;
neither an architectural impossibility nor periodicity-induced collapse follows.

The unchanged seed-0/LR-0.001 four-fit GPU canary would target this same failed
discovery slice. Before requesting GPU time, independently review these artifacts
and the full-prefix task/input/target chronology, then decide prospectively
whether to complete the already specified LR grid locally or propose a new
versioned acquisition curriculum. Preserve this v2 failure; do not extend its
stage budget or relabel a rescue as the original experiment. GPU-specific
numerics and efficiency remain unmeasured, and no GPU resource approval is
conferred by this local run.
