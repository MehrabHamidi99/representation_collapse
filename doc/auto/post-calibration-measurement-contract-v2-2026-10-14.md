# Post-calibration measurement contract v2 — static proposal

2026-10-14 (local project date). **Execution is not authorized.** This document
and `configs/experiments/cached-state-forward-calibration-v2.json` propose only
one 736-call, label-free forward calibration. They are not an implementation,
an execution freeze, a fit envelope, a restart, or scientific acceptance. The
coordinator owns admission after independent review. The October 13 draft and
its partial review remain unchanged; this revision addresses their four
specified objections. No inherited manuscript/active TeX include supplies a
claim here; all inherited snapshots remain untouched.

## Prior evidence and the question

The [independent v1 results review](native-chronology-calibration-v1-independent-results-review-2026-10-13.md)
accepted integrity of 372 completed conditions/120,000 native calls at source
`9cff2d3d2bbaccee1642441b325625969e551f7b`, with zero optimizer updates.
The accepted receipt is
`a48031c7c0a32c5691fba45af150c839ff51705df45f300fb0f6318b72b4adc1`.
Strict interoperability failed: all 38 failed comparisons involved the known
trained GRU/categorical checkpoint in FP32. This does not establish collapse,
competence, causal memory loss, a kernel cause, or FP64 ground truth. The 24
copy-origin before operands not separately exported in v1 motivate mandatory
before/after exports here, without changing the historical evidence.

V5 remains stopped at 4,000 updates; its fixed 5,000 endpoints are unknown and
three cells were unstarted. Earlier score gates/outcomes remain unchanged.
No rescoring of v5 inputs, selection of a better path/precision, retry, resume,
training, or checkpoint repair is proposed. The question is whether the
specified C program and its copies/perturbations can be observed reproducibly
at fixed challenge weights. Failure is retained as numerical evidence.

## Four different map identities

Every map binds source hashes, exact weight values, input bytes and layout,
encoder/core partitions, batch/row order, dtype, native binary/build, device,
thread flags and head. Shape and partition are part of finite execution.

**B0, unchanged public behavior:** call the pinned `VectorMemoryModel.forward`
once on `[64,17,38]`: full encoder, one native bulk 17-update core invocation,
head on the last returned hidden output. Model is in eval/no-grad mode with no
recurrent state carried between records. Non-mutating hooks export encoder
output, all 17 hidden outputs, and the returned complete terminal state from
this same invocation. No second native call may recover a hook omission.
B0 means this prospective pinned configuration; matching historical v1's
numerical predictions is not assumed (its recorded inter-op setting was 10,
whereas this one-CPU proposal explicitly uses inter-op 1).

**C, complete observed one-row trajectory:** make one separately named full
encoder call on the same `[64,17,38]` tensor; preserve the complete resulting
cache `E` with shape `[64,17,64]`. Start `S0` with exact zero complete state.
For t=0..16, call the original native core on `E[:,t:t+1,:]` and an independent
contiguous copy of `St`, then take the returned complete state as `S(t+1)`.
Sixteen ordered WRITEs precede one QUERY; there is no body, reset, fresh QUERY
encoding, bulk-prefix substitution, or concatenation of intervention batches.
The slice has the full-cache stride (including batch stride 17*64); do not
silently make a contiguous encoded-input copy. Capture actual strides.
`RQ_C(S)=head(core(E[:,16:17,:],copy(S)).output[:,0,:])`.
The C baseline is the already performed seventeenth call at `S16`.

For GRU, complete state is h `[1,64,128]`; packed row order is h[0]. For LSTM
it is `(h,c)`, each `[1,64,128]`; packed `[64,256]` is h then c with increasing
feature index, no feature scaling, centering or standardization. `S16` is the
complete post-sixteenth-WRITE/pre-QUERY intervention origin; `S17` is the
complete post-QUERY terminal state. All rows 0..63 are selected; none is chosen
by response. B0's returned terminal h/c can be compared with C S17. B0's fused
intermediate LSTM c is **unobserved**: no complete B0 S16 exists in this record.
Its intermediate h is observable through the hidden sequence, not a complete
LSTM state. Missing c remains null with an explicit reason.

**T, deferred bulk-prefix training graph:** separately encode sixteen WRITEs,
execute one bulk-prefix core call at training B=32, and branch each separately
encoded QUERY from that attached complete prefix. One shared eight-QUERY graph
has nine core invocations; eight independently recomputed same-T branches have
sixteen. T is distinct from the sixteen literal WRITE updates in C and the
uninterrupted seventeen-step B0 graph. T does not execute in this proposal.
A full-parameter derivative certificate for T belongs only to T and retains
all encoder/cache/prefix contributions attached to theta. It certifies neither
C parameter gradients nor the B0 finite objective. A conditional-mean gradient
argument concerns T's stated separable real-valued per-example objective;
finite-precision accumulation still requires calibration.

**D64, paired numerical sensitivity:** promote exactly the stored FP32 weights
and FP32 input values to FP64, then execute the same named B0/C recipes. Do not
train, recover extra precision, or select FP64 by task behavior. Native forward
backend is the pinned PyTorch 2.6.0 CPU/arm64/Accelerate build in the config;
its Python native RNN source and local native binaries are hash-bound. There
is no backend/device/precision fallback. A mismatch blocks execution until a
separate reviewed revision; it cannot silently switch the map.

## C effects and behavior agreement have separate meanings

All nonzero perturbation effects here concern **C only**. Equality of B0 and C
baseline logits, even bitwise equality, cannot establish state or counterfactual
equivalence: a many-to-one readout can agree at different complete states and
have different derivatives (`s^2` at +1 versus -1 is a simple example).
B0's unobserved intermediate LSTM c makes the missing bridge explicit.
No B0-transfer claim is permitted by this stage. A future bridge must define
and independently validate a complete-state correspondence and commuting
continuations/intervention transformations over its declared domain, with
uncertainty/error bounds. Tests at finitely many baseline outputs alone do
not prove such a bridge. Alternatively a future mechanism experiment may
prospectively define both its behavior and interventions in C. Neither choice
repairs v5 or confers competence.

Elementwise forward agreement also does not certify task decisions. Categorical
logits `(0,5e-7)` and `(5e-7,0)` can satisfy the old FP32 bounds and switch argmax.
For each categorical row, export both exact computed argmaxes (first index on
ties), all ties, and a separate label-free margin certificate. With B0 reference
logits z, candidate C logits w, `b_j=atol+rtol*abs(z_j)` and unique B0 winner a,
certify only if all directed residuals pass **and**, for every j != a,
`z_a-z_j > b_a+b_j`. Use strict greater-than; equality/ties are uncertified.
Also record the actual signed differences and margins. Exact prediction
agreement is a recorded baseline decision fact, not accuracy or a causal bridge.
A tolerance pass without equal predictions or a valid certificate supplies no
categorical behavior alignment. No label is needed or allowed for this check.

For continuous B0 output v, C output u and a future target y, if `|u-v|<=e`,
then `|(u-y)^2-(v-y)^2| <= e*(2*|v-y|+e)`; normalized MSE divides by 1/3 and
thus multiplies this bound by 3. The label-free task-range bound for every
`y in [-1,1]` is `3*e*(2*(|v|+1)+e)`. Report its mean over all 64 rows using
actual `e=abs(u-v)`; also report the conditional bound with e replaced by the
forward tolerance only for rows that pass it. These bounds do not say the
change is zero, certify a .05 gate, or supply actual MSE. Actual MSE and target
residuals are undefined because targets are prohibited. No clipping of outputs
or dropping ambiguous rows is allowed. These are arithmetic bounds on saved
values; no new forward or label query enters.

## Fixed contexts, inputs and directions

The config pins all eight existing checkpoint files by path, byte length and
SHA-256, verified here by file hashing only. Four are seed-0 `fresh_seed0`
initial checkpoints from v3, giving stored initial bytes rather than a new RNG
initialization. The fixed challenge checkpoints are v5 GRU/categorical current
step 4,000 and the v3 fresh step-5,000 endpoints for GRU/continuous,
LSTM/categorical and LSTM/continuous. They are selected by the predeclared
endpoint/challenge rule, never by score; none is declared competent. No archive
was deserialized by this author. Later loading must read only the pinned model
state field, validate FP32 shapes, and forbid optimizer/target/score access.

Use validation-input seed **30019**; this is a label-free synthetic calibration
stream, not validation task scoring. Reset an independent NumPy PCG64 generator
to that same seed for each task. Draw exactly 32 histories: categorical
`integers(0,16,size=(32,16),dtype=int64)` or continuous
`uniform(-1,1,size=(32,16)).astype(float32)` in C row-major order. Concatenate
the 32-row array with its identical copy, yielding row IDs 0..63 with independent
history ID `row_id mod 32`. Set QUERY to `row_id mod 8`; all 64 bodies are empty.
Render using the pinned vector-task layout: type[0:5], coordinate[5:21],
categorical value[21:37], real value[37]. WRITE address t at time t=0..15;
QUERY at time16 has no value payload. Do not construct any targets, target
indices from the payload, paired_batch object, score, loss, or decoder.
Generate once per task; both architectures, contexts, repeats and dtypes reuse
those input bytes (FP64 exact promotion). No existing validation .npz is read.

Directions use a separate single NumPy PCG64 stream seeded **30029**. Follow
config context order (GRU categorical init/challenge; GRU continuous
init/challenge; LSTM categorical init/challenge; LSTM continuous init/challenge).
Draw one `(32,128)` float64 standard-normal array for each GRU h direction;
for each LSTM context draw h then c arrays in that order. Normalize each row by
its float64 raw Euclidean norm and duplicate rows 0..31 into rows32..63. Store
these canonical directions before dtype casting; use the same directions for
both dtypes and repeats, with no response-dependent redraw or normalization.
Casting incurs representational unit-norm error: export actual direction norms,
require finite positive norms and `abs(norm-1)<=1e-6`, and record errors rather
than asserting exact floating-point norm 1. A zero/nonfinite norm stops the
stage, with no extra draw. h-only has zero c displacement; c-only has zero h.
There are 12 direction/context combinations and no joint h+c direction here.
For each, form `copy(S16) + sign*dtype(1e-3)*dtype(direction)`, signs + then -;
record both requested and represented complete-state displacements. The raw
state unit is fixed; no whitening, standardization or arm-specific scaling.

## Exact 736-call registry and exports

There are 32 conditions: 8 contexts, dtype order FP64 then FP32, repeat1 then
repeat2. A repeat reloads the same pinned weights and resets state/cache; it
reuses input/direction bytes and is a numerical repetition, not another seed.
For each condition execute in this exact order:

1. One B0 native bulk-core call and seventeen C one-row calls: 18.
2. One copy QUERY from an untouched clone of C S16; one zero-displacement QUERY
   from a separate `S16 + zeros_like(S16)` clone, both with the same cached E16
   and same readout: 2. Zero-displacement is not zero-state replacement.
3. The h+ and h- QUERY calls; for LSTM also c+ and c-, each from a fresh S16
   clone with the same cache, batch, dtype and head: 2 for GRU, 4 for LSTM.

Thus `8*2*2*18=576`, copy/zero `8*2*2*2=64`, and signed-direction
`12*2*2*2=96`, totaling **736 physical native-core forward invocations**.
There are sixteen 22-call GRU conditions and sixteen 24-call LSTM conditions.
Encoder invocations are separately counted (2 per condition, 64 total); head
invocations are 224 total (one B0 and 1 baseline+2 sham+2/4 signed C readouts
per condition). Neither is a native-core call; both consume the shared clock.
Hooks count attempted calls immediately before native execution, including a
call that raises. Completed counts increment only after successful returns.
Record per-call index, map, context, dtype, repeat, step/arm, input/state hashes,
start/end clock, and completion/exception. A call after 736 is prohibited.
No warmup, retry, extra chunk, hidden guard call, derivative, zero-parameter
fixture, separate prefix diagnostic or AD transform is included.

Export complete C states S0..S17, per-step hidden outputs, full cache, B0 hidden
sequence and terminal h/c, B0 and every C arm's logits and returned terminal
complete state. For **each** C call and each sham/perturbation call export
independent before/after snapshots of the supplied complete-state operands
and cache. Also export untouched origin S16 before/after the whole arm set,
arm constructed input states, canonical directions and represented changes.
Each snapshot has independent saved bytes, component names, shape, dtype,
stride, row/feature IDs and hashes; a runtime equality assertion alone is
insufficient. Export FP32 base weights' identity and promotion checks as compact
metadata, never upload raw weights. Hooks must not alter outputs, state, mode
or RNG. Any missing component is explicit missing evidence, never inferred zero.

Required directed comparisons use float64 arithmetic on saved finite operands:
`abs(candidate-reference) <= atol+rtol*abs(reference)` with FP32
`(atol=1e-6,rtol=1e-5)` and FP64 `(1e-12,1e-10)`. There is no threshold relaxation.
Report every violated index, both operands, absolute differences and bounds:

- C baseline logits/terminal h/c/hidden sequence versus their corresponding B0
  values, B0 reference. No B0 intermediate-c comparison exists.
- Copy and zero-displacement logits/returned terminal h/c versus C baseline,
  C baseline reference. These diagnose only the same C continuation.
- All same-map repeat2 observables versus repeat1 require exact dtype, shape
  and raw-byte identity; signed zeros are distinct for this purpose.
- Before/after operand/cache/origin identity requires exact bytes; duplication
  checks compare corresponding halves with the same dtype bound.
- FP32 observables versus matching FP64 observables use the FP32 bound and
  FP64 reference. This is sensitivity only, with every failed edge retained.

No nonzero perturbation is required to change a logit. Export each finite
signed effect relative to the C baseline and its actual state displacement;
logit equality is `unresolved_at_this_perturbation`, not absent information.
If a represented perturbation is zero, record `represented_zero` and no effect
ratio. Norm/gain ratios with zero denominators, missing operands, overflow or
nonfinite values are null with a named reason and count; never zero sentinels
or fabricated infinities. Nonfinite native output stops safely and leaves the
registry incomplete. Failed numerical comparisons remain records and do not
choose a new path. Numeric dimension/tangent/competence summaries are absent.

## Prospective execution gate and reporting

One local CPU, no GPUs, one inter-op/intra-op thread and a **single 600-second
wall clock**, beginning before process imports, source/checkpoint hash checks,
W&B initialization and input generation, through final artifact writes. A
launcher must enforce the external deadline; the engine checks remaining time
before every native call and at stage boundaries. Reserve the last 10 seconds
for finalization: launch no new core call at elapsed >=590 seconds. Missing
conditions/calls, deadline, owner stop, dependency/hash mismatch or exception
produce an incomplete result with attempted/completed registry; no automatic
retry or resume. W&B service/process overhead consumes the same envelope.

Initialize W&B in explicit **offline** mode before generating sweep inputs or
any native call, using the owner-authorized project **representation-collapse**
and a unique run ID. Entity/authentication may remain unknown for this durable
local offline stage: use `entity=null`, `sync_pending=true`, and record that
remote delivery is unverified. Missing dashboard authentication is not a
scientific prerequisite for local offline calibration. No credential
lookup/inference, provider request or online sync occurs. Require the exact
SDK pin **wandb==0.30.0**, its reviewed lock/installed identity and a successful
offline lifecycle smoke receipt at the implementation/execution freeze;
failure to initialize durable offline tracking blocks the sweep. Log
config/source/checkpoint hashes, input/direction identities, per-condition
counts/residual summaries, CPU time/RSS/elapsed time, failures and completion
live with monotonic steps. Retain offline directory and run ID. Compact
metadata manifests only: no raw datasets, full weights, credentials or endpoint
configuration are uploaded. Offline receipt is not dashboard delivery; any
later authenticated sync requires its own receipt. Prior runs are not
retroactively described as W&B-tracked.

The config's `execution_authorized=false` is normative. Before any execution,
require independent static acceptance, separately reviewed additive
implementation and bounded fixtures, clean source publication, exact argv,
unique output/attempt path, W&B dependency/offline initialization evidence and an
external hash-bound coordinator admission/freezing record. No actual argv or
implementation exists yet. This design grants no code execution through a
config toggle. All W&B/source/binary/lock identities must be reverified at
admission; runtime mismatch does not permit installation or map substitution
inside this stage. There are zero fits, optimizer updates, targets, gradients,
JVPs, providers, cluster operations and task scores.

## Deferred derivatives — no current admission

No derivative stage or fit appears in the executable-stage config. A later
proposal must calibrate the exact graph it names: local C state partials hold
complete S and cached e fixed; full C parameter derivatives keep every C
encoder and literal-step dependency attached; full T parameter derivatives
keep every bulk-prefix and branch contribution attached. An AD certificate
concerns the real-valued surrogate of that operation graph, not the ordinary
derivative of a quantized IEEE byte map, another map, or all unresolved modes.

Budget units must be separate: physical native primals, standalone reverse
requests, compound JVP requests, and physical internal reverse passes. The
older combined draft's arithmetic was 3,840 native primals, 320 standalone
reverse requests and 224 compound JVP requests; a reverse-over-reverse JVP
would additionally spend two reverse passes each, making **768 physical reverse
passes**. None of these derivative allowances is adopted here. The current
AD backend is explicitly **none**, with all derivative counters zero. Any
future AD proposal must pin one actual transform/backend and its physical
accounting before admission; unsupported native operations stop it without
surrogate, backend or dtype fallback. Prospective derivative tolerances and
near-zero resolution remain for that separately reviewed proposal, not implicit
acceptance thresholds in this forward stage.

No finite-time tangent, covariance dimension, Floquet, causal-memory or trained
collapse conclusion is enabled. The next decision is independent review of
this completed static forward-only candidate; admission remains with the root
coordinator.

The original [Gate B competence contract](experimental-design-v2.md) remains
unchanged for any later scientific experiment: sixteen ordered WRITEs, QUERY
support 0–7, every one of the eight required coordinates at categorical
accuracy >=0.95 or continuous MSE/(1/3) <=0.05, with the existing four-ULP
score-boundary rule. R0 success alone is insufficient: R0/R1/R2 competence in
both arms, independent trained seeds and the original paired inference gates
remain required. This forward stage computes no such score and grants no new
eligibility. V5's fixed endpoints stay unknown and its three unstarted cells
remain unstarted; no retry, resume, checkpoint selection or fit is admitted.
