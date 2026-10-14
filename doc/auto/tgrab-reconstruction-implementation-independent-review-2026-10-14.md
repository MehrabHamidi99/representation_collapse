# Independent review of T-GRAB reconstruction contract components

2026-10-14. **Component engineering accepted after two lifecycle repairs;
native integration, runnable training and scientific acceptance remain absent.**
This review concerns the final frozen module and its synthetic fixtures. It
does not accept a reconstructed checkpoint, graph experiment, trained-model
result or collapse claim. The primary condition remains Pdet(2,32); AB and
AABB are distinct, unexecuted conditions.

Reviewed identities:

- `src/representation_collapse/graph_reconstruction.py`:
  `89894f8bd3e8574b5b57c777353fe1aac3a1f123f792467a009b04b8a6f9bcdb`.
- `tests/test_graph_reconstruction.py`:
  `9b1f55a5f3e57ac8d48914500eada783071f7820b41a63f06a52c766eba89eb2`.
- Author implementation report:
  `6436b1e9a4408fbbf829e2132a1ac18d83d1079c5ea38d3c3c1bd4451d678912`.
- Frozen specification/config:
  `698a3150566e47a1866308f7b9782ab2b2b1f4878b6592acb978972cb9f1107b` /
  `22bef6bf781c5bfb5493dd5e03169ea0e658b2463ccaa9ba09189af454508f1f`.

The author's initial and additive repair receipts, every bound file and the
saved raw repair-test output were independently hash-checked. Initial source,
tests, report and receipt remain preserved. Review evidence is under
`runs/review/tgrab-reconstruction-implementation-independent-20261014/`;
the final receipt distinguishes component acceptance from native readiness.

## Two defects found and repaired

The initial module `b61f32bd…` allowed `training_epoch` after `test_once` had
opened the held-out sequence, although validation was already forbidden. It
also checked the pilot budget before selected-checkpoint loading but opened
test data without checking again after that load. A load reaching the walltime
cap therefore still invoked the test opener before the next replay boundary
stopped execution.

One independent, three-node spy invocation on the preserved exact initial
source confirmed both defects: two fake update events were accepted after
test, and the opener was called after loading reached the cap. This used no
native model, optimizer, real graph payload or SDK. An earlier import-path
failure and a subsequent source-freeze refusal occurred before any fixture;
they are retained as preparation failures, not graph failures.

The final source rejects training after test opens, checks the walltime/owner
stop after selected loading before opening test, and checks again after the
opener before reset/replay. A failed opener is terminal because `test_opened`
is set before calling it. The source diff contains only these agreed guards.
Three new spy methods cover terminal training and both walltime and owner-stop
events during loading/opening. The saved final command reports **28 tests
passing in 0.101 seconds**, with all 28 named successes and exit code zero in
the preserved raw tool result. The reviewer inspected that evidence and did
not repeat the campaign. No blocking defect remains in the reviewed contract
components.

## Contract findings

The guarded construction path calls seeding before fresh recurrent/head and
Adam construction. Parameter order is deterministic: named recurrent
parameters followed by named head parameters. The dedicated negative RNG is
separate from global NumPy draws. Initialization repetition, actual dropout
streams and native optimizer/RNG round trips still need native evidence.

The prepared native path imports the exact public MultiLayerGCLSTM,
LinkPredictor and NodeEmbeddingModel text pins from commit
`31293102b065feef7817ff86211427d7594edfdb`. It requires the actual Temporal
v0.54.0 GC-LSTM cell from `46abd39abec0cd8de23c0e489a617c320c3fc5e8`, with
source SHA-256
`c3568d2c1ba8926ab43e11c840213f90ebe782ad158dd76809d1dc1c255797af`.
There is no replacement dense cell or independently chosen convolution. The
source link head retains ordered source/destination concatenation and the
unusual ReLU-before-linear sequence, dropout and sigmoid. These conclusions
come from source comparison; the wrappers and native backend were not executed.

Complete-state layouts include hidden and cell, all layers, ascending node IDs
and features. Flattening is hidden then cell, layer, node, feature; L1/N100/H128
gives dimension 25600. The native adapter preserves both components and asserts
that derived output `z` equals the last-layer hidden state. Synthetic shape,
ordering and flattening fixtures support this contract, rather than native
graph-state parity.

Training resets each epoch, consumes only the predecessor before each target's
loss/update, and detaches before the next recurrent consume. The final training
target is scored without an unnecessary subsequent recurrent update. Evaluation
freshly resets and replays its conditioning prefix in evaluation mode, then
scores from complete state and fixed candidate IDs before consuming each target
once. The scoring API receives no current-target graph. Evaluation performs no
optimizer update; selected recurrent and head weights must be loaded before
test opens. Full native prefix lengths and split intervals remain an integration
gate because the controller deliberately accepts tiny fixture sequences.

The sampler checks the full positive set, same-source alignment and diagonal
exclusion, samples admissible destinations uniformly with replacement, permits
negative duplicates and fails finitely on saturated rows. Metrics use the fixed
lexicographic ordered off-diagonal pair set: 9900 for 100 nodes, threshold
probability >=0.5 and zero-division=0. Primary aggregation is time-mean F1.
Actual change points compare consecutive graphs; phase expectations alone do
not verify distinct patterns or conversion. Boundary phases and eight test
transitions are synthetic contract checks.

Selection accepts only complete-epoch validation, uses strict F1 improvement
and keeps the earliest tie. Strict relative loss improvement and plateau
patience are separate from selection. Budget counters cover owner stop and
finite epoch/update/walltime boundaries; incomplete epochs are not counted as
complete. These helpers do not provide a physical outer watchdog: a hung native
call, checkpoint I/O or test opener cannot be preempted by a boundary check.
A reviewed launcher and deadline cleanup are required before the 3600-second
entire-pilot bound can be accepted as an execution guarantee.

## Remaining admission and evidence gates

Preflight remains `ready=false`, and launch remains disabled. Current metadata
has Torch2.6.0/NumPy2.4.6, whereas the source environment requests
Torch2.0.0/NumPy1.24.4; PyG, Temporal and the graph extensions are missing.
Installed cell identity, compiled-extension compatibility and the locked graph
environment are unverified. The unconditional
`native_checkpoint_and_reporting_runner_not_implemented` blocker prevents
native factory construction even with matching dependency metadata. The
selected-checkpoint loader explicitly raises instead of deserializing weights.

Native recurrent/head/loss-gradient/dropout/state parity, actual data and
CSV/NPZ conversion, full native split enforcement, durable selected/resume/RNG
artifacts, full epoch/patience/selection orchestration, fail-closed W&B and a
physical stop supervisor remain required. Independent scientific protocol
review and a fresh exact compute envelope remain separate gates. This review
does not set the frozen configuration's readiness flags or authorize training,
cluster jobs, derivative/Floquet work, or automatic retry/resume.

Reviewer execution counts: one completed two-case synthetic spy invocation;
zero native graph/head calls, real optimizer steps, generator calls, dataset or
checkpoint payload reads, SDK/provider runs, cluster operations and campaign
reruns. Coordinator tracker/notebook integration belongs to root.
