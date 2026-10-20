# T-GCN production training-prefix diagnostic, 2026-10-20

Prospective protocol for `tgrab_tgcn_training_prefix_v1`. Freeze and review
this protocol, worker and admission source closure before the first production
training-prefix execution. This is one isolated 256-update engineering
diagnostic; it is not a complete epoch, competence test, training-seed cohort,
full-fit cost estimate, Figure 10 reproduction or representation-collapse result.

## Question and prerequisite

Can the calibrated native production T-GCN and link head execute targets
**1–256** of the already verified primary Pdet(2,32) training sequence with
correct causal chronology, finite loss/gradients/parameters, exactly one
optimizer step per target, durable pre-update charges and complete reporting?
No validation or held-out measurements may influence this decision.

The prerequisite is the independently accepted
[T-GCN synthetic native calibration](tgcn-native-calibration-protocol-2026-10-20.md):
job **11332092**, source
`e0e6c20a72a98134aa1f2a2450f9eeb011718ab2`, accepted result SHA-256
`23b68f8c78e0f4b1fd00959c812463318cb687d8b9b95404c709c6ade5001619`.
Its compact acceptance receipt SHA-256 is
`45ef2dd57b60ed0746967f042eaf8574915b53b612434628736775974a0054e2`;
kind is `tgcn_native_calibration_compact_acceptance_v1`, status
`accepted_synthetic_execution`, with 248 verified checks. The receipt correctly
states `training_admitted=false`: it verifies a prerequisite, while this new
protocol and its separate concrete admission define the prefix work. Neither
receipt nor this prefix protocol admits a subsequent full fit or test pass.

Bind the actual accepted receipt and original calibration result by their
external hashes. Reject missing/failed acceptance, a different architecture,
source/policy substitution or unreviewed runtime. No native calibration retry
or synthetic result recomputation is part of this job.

## Executed model and unchanged data identity

Use one native TGCN layer, **100 one-hot input features/nodes, hidden width 128**,
CPU float32 and the calibrated public link head. Complete persistent state is
**H[1,100,128]**, next-target timestamp and ordered node IDs 0–99. There is no
LSTM cell component and no GC-LSTM state/checkpoint coercion. Each GCN gate uses
`improved=False`, `cached=False`, `add_self_loops=False`, `normalize=True` and
source-to-target flow. All graph caches remain empty. Native parameter counts
are 137,472 recurrent plus 49,537 head parameters, 187,009 total.

Freeze the same source pins accepted by calibration:

- Backend `tgrab_tgcn_backend.py`:
  `0b03219543c0cce16e77c69cf7350a403e2d492134742bb1ec23aaf03dea512f`.
- Public T-GRAB commit: `31293102b065feef7817ff86211427d7594edfdb`.
  Wrapper `model/tgcn.py`:
  `028a85f0527833a7e62099e8bb826ac144df29738ee120795d205d96a8c6f78c`.
  Base `model/node_emb.py`:
  `4be4450730f33949e0354348b3bd8f69ba0434e4aa911abec0fc5f798e8c2d29`.
  Head `model/link_predictor.py`:
  `cca12cfc73b8660c92236f530504da08ba0b606ec13a1a681ffabcae43b7ca86`.
- Native `temporalgcn.py`:
  `1f417ad10b1fb3ed6af24731efb0e80b614495a3e4df3fc16254e631c3c8819e`.
  PyG `gcn_conv.py`:
  `036b464769ad1d98501c9fc18ff3132b507890b2c5cb41b3e6f9e1f22d4f57f6`.

Use the same verified primary conversion and semantic training loader as the
GC-LSTM prefix. The conversion receipt below identifies the prior independent
verification lineage; this job authenticates its already accepted exact
converted manifest and training payload, without repeating conversion or
claiming a new conversion-receipt audit. These identities remain data/conversion
identities, not an executed GC-LSTM architecture contract:

- Raw public input SHA-256:
  `5a19f9e7949668473bc61cb59b6215f25f093ea0ee2d45b5e143338ded5523e3`.
- Conversion/numerical template SHA-256:
  `22bef6bf781c5bfb5493dd5e03169ea0e658b2463ccaa9ba09189af454508f1f`.
- Independently verified converted manifest SHA-256:
  `a6ff50183de570ff17cfbe92b3f1976bac9d2d85557ecc11379e24b2c737d1ce`.
- Conversion verification receipt SHA-256:
  `9e279518ab997e8a7596a3a23f67fd3d9ea0362d761d814149fb5eec8e88e2c8`.
- Frozen conversion protocol SHA-256:
  `003e6b01d3b090841f623f75ed0648a397fe8afdaeb49dc382fb69a3f0e05584`.

The verified manifest includes training [0,2560), validation [2560,2816) and
test [2816,3072) metadata. Read and validate **only the training payload**;
reading the entire training file for its established hash/semantic integrity
is allowed. Numerical work uses only training snapshots 0–256. Validation and
test payload reads, callbacks, evaluation capabilities and metrics are forbidden.
Do not use `graph_data.open_train_validation`, which eagerly loads validation
as well as training. Use the existing pinned train-only split loader with
verified manifest, guard and canonical snapshot factory. Manifest-only hashes
or interval metadata for other splits do not authorize their payload access.

## Fixed initialization, optimization and state chronology

Use the locked native environment, exact CPython 3.10.19 and dependency versions
accepted by calibration; verify interpreter/environment receipt, environment
lock and native wheel manifest. Source closure binds the actual worker,
backend, loader, sampler, reporting, optimizer-step witness, journal helpers,
launcher/supervisor and this protocol. No dependency installation or source
substitution within the attempt is allowed.

Seed Python, NumPy and Torch with **0 before constructing models**; create a
separate `numpy.random.default_rng(1)` for negative destinations. Construct the
calibrated recurrent wrapper then the public head, then a fresh Adam optimizer.
Unlike the tiny calibration's gradient-control fixture, **do not fill head
parameters with 0.1 and do not reset to dropout seed 73**. Production parameters
use the actual seed-0 native initialization. Record initialization hashes,
RNG observations, constructor order and actual model/tensor inventory.

One CPU numerical thread, no AMP and CPU float32 are mandatory. Torch intra-op
and inter-op threads, OMP, MKL, OpenBLAS, NumExpr and vecLib limits are one.
Enable deterministic algorithms without warn-only fallback; cuDNN benchmark
is false and its deterministic flag true. Preserve public head dropout **0.2**
in training mode. Fresh recurrent/head parameters and optimizer are required;
load no weights, optimizer state, saved state or saved RNG from another run.

Adam settings: learning rate **1e-4**, betas **(0.9,0.999)**, epsilon **1e-8**,
weight decay **0**, amsgrad **false**. Parameter order is deterministic:
recurrent component first, then head, sorting named parameters within each
component. This order must agree with the accepted backend inventory.

Start at positive-zero complete H and next-target timestamp zero. For each
`t=1,…,256`, in order:

1. Check ownership, STOP, deadline and all available data/source boundaries.
2. Detach incoming H; consume **G[t−1]** exactly once at the current weights.
   Require next-target timestamp `t`, complete H shape and canonical node IDs.
3. Sample one negative per positive target edge using the seed-1 generator,
   preserving each positive edge's source. Destinations are sampled **with
   replacement** from nodes excluding that source and **all** positive target
   neighbors. Reject diagonal/out-of-range/full-positive collisions or a
   source with no admissible negative. Preserve the native canonical positive
   pair order; no resampling or data-dependent retry is allowed.
4. Durably publish/fsync the update intent, including sequence, consumed and
   target timestamps, before entering the loss/update call. The first intent
   must precede production optimizer step one; a failure after this barrier
   leaves a charged attempt.
5. Score G[t] **before consuming G[t]**. Loss is mean positive BCE plus mean
   negative BCE, with equally sized pools; do not average both pools into a
   differently scaled combined loss. Backpropagation window is one, with one
   optimizer step on recurrent and head parameters. Require finite loss,
   gradients and updated parameters.
6. Persist the returned-call outcome and observed optimizer-step status,
   then report the live update. Carry the resulting H value to the next
   iteration; do not reset at motif boundaries or replay the predecessor
   again after the parameter update.

A step witness must observe exactly one entered and returned `optimizer.step`;
a finite loss alone does not prove a step occurred. Distinguish a returned
backend call, a confirmed step, not-started and unknown mutation statuses.
Reconcile the durable intent/outcome records on every terminal path. Missing,
torn or uncommitted outcomes remain charged/unknown; never promote visible
bytes after a failed fsync into confirmed durable completion. A returned call
is counted before checking its diagnostic loss or later reporting I/O.

The successful prefix has **256 attempted, returned and confirmed updates**,
targets 1–256 and consumed inputs 0–255, with **zero complete epochs**. At its
end, G[256] has been scored but not consumed. Any retained state descriptor is
at next-target 256 after consumption of G[255]; its value predates the final
weight update and is not an inference-ready fixed-weight replay state. No
checkpoint, selected candidate, optimizer/RNG resume artifact or inference
admission may be published. Diagnostic hashes/compact summaries are allowed;
no weight or complete state payload is required to leave process memory.

## Durable reporting, failure and independent acceptance

Enter explicit offline W&B in project `representation-collapse` before any
training-payload access, model construction or numerical work. A missing or
failed reporter prevents work. Record source/architecture/data/calibration
identity and seeds at initialization, each chronological loss/update as it is
resolved, resource observations and the terminal summary/artifact. Keep
monotonic reporting steps distinct from optimizer counters. Do not upload
raw datasets, weights, binary states, credentials or provider configuration.

Success requires scheduler `COMPLETED 0:0`, successful worker and supervisor
cleanup, exactly 256 paired durable intents/outcomes, exactly 256 confirmed
single optimizer steps, 256 finite nonnegative losses, exact timestamp and
node-order chronology, no validation/test/selection/checkpoint activity, and a
finished offline reporter with retained real transaction history. Independently
recompute counts and mean loss from the retained updates, verify journal
hashes/predecessors and result/report artifact identities, and decode actual
W&B history/config/artifact/terminal records including CRC framing. A metadata
audit does not independently rerun native training or verify live parameter
values. Remote dashboard delivery remains unknown until verified separately.

A short, failed or interrupted prefix reports its actual completed/confirmed
counts and unresolved charges. It does not pass the 256-update gate and must
not be relabeled as a complete epoch or resumed. Throughput, if reported,
describes only this prefix and its measured runtime scope. The shared loop
`elapsed_seconds` and `updates_per_second` include its reset and 256 updates,
but exclude native imports, training-file integrity checks and model
construction. Scheduler elapsed time describes the job separately. Neither
measure is a full-fit projection or a GPU sizing argument.

## One-job execution envelope

One cluv submission on **Mila**, account **mila**, QOS **normal**, partition
**main-cpu**: **one node, one task, one CPU, 2 GiB, zero GPUs, thirty minutes**.
Use a unique scratch run/attempt and sibling STOP path. The independent source
admission pins exact argv, fresh output paths, accepted calibration receipt and
established conversion-manifest identity, source closure, interpreter, actual scheduler/cgroup
resources and job-derived clocks. Queue waiting does not consume the worker
clock. Work stops by **1740 s**, supervisor by **1760 s**, shell by **1770 s**,
within Slurm's **1800 s** ceiling.

STOP, deadline, signal, identity/source/data/reporting failure, nonfinite
values, missing journal durability or invalid step accounting terminates the
attempt. **No automatic retry, requeue, race, alternate seed, resume or second
submission** is admitted here. Reconcile ambiguous scheduler state before any
further action; preserve all partial evidence and do not change other projects'
jobs or shared infrastructure.

After independent prefix acceptance, a separately reviewed full-fit runner,
T-GCN-specific checkpoint/catalog/hidden-state schema, complete-epoch
validation and validation-only selection protocol are still required. No
held-out opener is available from this diagnostic. No F1 competence, covariance
dimension, periodic-state/Jacobian, Floquet, causal-memory or cross-modal claim
follows from a passing training-prefix execution.
