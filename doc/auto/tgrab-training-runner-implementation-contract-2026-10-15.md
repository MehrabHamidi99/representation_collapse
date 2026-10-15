# Implementation contract for the next T-GRAB training runner

2026-10-15. **Specification of missing implementation, not an implemented or
admitted trainer.** This task performed source/document inspection, signature
extraction and hashing only. It executed no native/numerical/SDK/data-generation
work, downloaded no payload and made no remote or submission call. Accepted
code, configuration and protocols were read-only. Root's separately approved
native calibration does not authorize a training canary or fit.

The controlling scientific settings remain the
[reconstruction specification](tgrab-checkpoint-reconstruction-v1.md), SHA
`698a3150566e47a1866308f7b9782ab2b2b1f4878b6592acb978972cb9f1107b`,
and `configs/graph/tgrab-reconstruction-v1.json`, SHA
`22bef6bf781c5bfb5493dd5e03169ea0e658b2463ccaa9ba09189af454508f1f`.
The [runner-readiness plan](tgrab-reconstruction-runner-readiness-2026-10-15.md),
SHA `a81376c27044340b5c69068027191a26377a16c44ecc9c4862dc8f3d0063ff1d`,
is a preparation input. Its proposed competence cutoffs are **not** accepted
source settings and are not inherited by this contract. No new competence,
convergence, gradient-parity or numerical-error threshold is invented here.
The only performance threshold implemented below is the frozen source link
decision probability >=0.5; loss tolerance/patience/caps are separately fixed.

## Existing source and reporter interoperability

References marked GR point to
`src/representation_collapse/graph_reconstruction.py`, SHA
`89894f8bd3e8574b5b57c777353fe1aac3a1f123f792467a009b04b8a6f9bcdb`.
Line anchors below refer to these exact bytes, not to a hypothetical trainer.

- [GR:161](../../src/representation_collapse/graph_reconstruction.py#L161)
  actually defines `preflight(config_path: Path, source_root: Path,
  readiness: Readiness=Readiness(), *, probe=dependency_probe) -> dict`.
  [GR:190](../../src/representation_collapse/graph_reconstruction.py#L190)
  requires six typed/config-bound receipts; line218 unconditionally blocks
  the missing native checkpoint/reporting runner. Launch is false in the config.
- [GR:501](../../src/representation_collapse/graph_reconstruction.py#L501)
  defines `training_epoch(graphs, *, expected_updates=2559) -> float`;
  [GR:556](../../src/representation_collapse/graph_reconstruction.py#L556)
  defines `validation(training_prefix, targets) -> list[dict]`;
  [GR:561](../../src/representation_collapse/graph_reconstruction.py#L561)
  defines `test_once(selection, training_and_validation, open_test) -> list[dict]`.
  These are causal components accepting synthetic sequences, not a real split
  store, epoch orchestrator, checkpoint codec or physical launcher.
- [GR:703](../../src/representation_collapse/graph_reconstruction.py#L703)
  defines `build_native_backend(config_path, source_root,
  readiness=Readiness(), *, device="cpu")`; it currently always refuses.
  [GR:699](../../src/representation_collapse/graph_reconstruction.py#L699)
  explicitly refuses `load_selected(reference)` rather than loading a file.

**Historical reporting incompatibility found in this source audit:** the inspected
v2 calibration engine, SHA
`d4ecd1d240978beb2746a12b2d68f2b8df5a22ec53cb3821b09a2b3f3f907f72`,
places `module_path`, `config_path` and `protocol_path` in reporter `identities`
at [original-v2:503](../../runs/review/native-graph-reporting-path-fix-20261015/review-original-v2-source.py#L503).
But [reporting.py:64](../../src/representation_collapse/reporting.py#L64)
requires every identity value to be a lowercase 40/64-character hex digest,
and [reporting.py:110](../../src/representation_collapse/reporting.py#L110)
validates that before SDK initialization. Path-valued entries therefore fail
before Torch/helper/model construction. The author/coordinator was alerted;
this task did not patch or run that engine and does not infer scheduler state.
A separately versioned, reviewed correction must keep only hashes in
`identities`, and put permitted compact relative paths in reporter config or
local manifests. The training runner needs a dependency-free metadata-schema
fixture covering the **actual dictionaries passed to the actual reporter**.
Earlier import-order/AST spies did not check this cross-component contract.

Root subsequently repaired that integration, independently reviewed it in
`runs/review/native-graph-reporting-path-fix-20261015/review.json`, SHA
`3d4cc972b1268917ef49fff8f13d3024ab345725b35a8dfc364b5eb50c58f043`,
and published corrected engine SHA
`eecb96df752fbb5842685cc423e67494758785792302d7c46fa24e2642741156`
at commit `3f24d63153fdc49a028a59b9cce35120a519e67c`. The original source remains
preserved at the linked path. The finding is resolved for that corrected engine;
the new training runner still needs its own actual-metadata compatibility check.
An initial static handoff check refused the moving local engine rather than
bind old findings to new source bytes; both versions are distinguished here.

Root reports calibration job 11261627 completed 0:0 in 74 s with all 27 producer
checks passing and the fixed native/AD counts. Its independent numerical
verification is pending at this handoff. This is coordinator-supplied runtime
status, not an independently accepted result in this document, and supplies no
N100 training/data/checkpoint parity or training allocation authority.

## Immutable inputs and unresolved acquisition

Public T-GRAB commit: `31293102b065feef7817ff86211427d7594edfdb`.
Temporal 0.54.0 commit: `46abd39abec0cd8de23c0e489a617c320c3fc5e8`.
Installed cell must equal SHA
`c3568d2c1ba8926ab43e11c840213f90ebe782ad158dd76809d1dc1c255797af`.
Public wrapper pins are GR:34–38; use their exact source and actual ChebConv,
never a substitute recurrent cell. Local public fixtures below are rooted at
`runs/graph/upstream-pin-20261012/source/`:

- `model/gclstm.py`, SHA `86069a32fced5f09c60cd795e4f0abd82674dafae250ef3673ac8c3650504609`:
  [lines10–16](../../runs/graph/upstream-pin-20261012/source/model/gclstm.py#L10)
  define `GCLSTMParam(in_channels,out_channels,K=1)` and
  `MultiLayerGCLSTM(num_units,gclstm_param)`;
  [line38](../../runs/graph/upstream-pin-20261012/source/model/gclstm.py#L38)
  accepts `(X,edge_index,edge_feat,hs,cs)` and returns `(h1,c1,h)` at line51.
  The forward annotation is a single tensor but the actual return is a tuple.
- `model/link_predictor.py`, SHA `cca12cfc73b8660c92236f530504da08ba0b606ec13a1a681ffabcae43b7ca86`:
  [line5](../../runs/graph/upstream-pin-20261012/source/model/link_predictor.py#L5)
  defines `LinkPredictor(hidden_dim,num_layers,p_dropout=0.2)`;
  line20 defines `forward(x1,x2)`. Preserve ordered concatenation, ReLU before
  each hidden Linear, dropout and sigmoid. `model/node_emb.py` SHA is
  `4be4450730f33949e0354348b3bd8f69ba0434e4aa911abec0fc5f798e8c2d29`.
- `dataset/DTDG/graph_generation/periodicity.py`, SHA
  `754cd90de5a3ba0887d73df29391cc62933c479e89f5e5d6960d07b6bd542e54`:
  [line26](../../runs/graph/upstream-pin-20261012/source/dataset/DTDG/graph_generation/periodicity.py#L26)
  is `PeriodicGenerator(args)`, not a function returning native snapshots.
  Lines56–70 define repeated pattern order and total length; lines125–134
  define chronological masks; lines137–148 seed pattern-specific ER graphs;
  lines224–228 insert the empty-step self-loop marker before sparse conversion.
- `dataset/DTDG/graph_generation/graph_generator.py`, SHA
  `447028d7c436bcc160c294ed10febe7dd372465d042531335b123c17aaf5e1f0`:
  [line17](../../runs/graph/upstream-pin-20261012/source/dataset/DTDG/graph_generation/graph_generator.py#L17)
  is `nx_undirected_graph_to_sparse(G,return_edge_feat=False)`; lines26–39
  append reverse edges and preserve supplied weights/default one.
  [line83](../../runs/graph/upstream-pin-20261012/source/dataset/DTDG/graph_generation/graph_generator.py#L83)
  defines `GraphGenerator.create_data(args_dict) -> None`; lines104–114 write
  `data.npz` fields src,dst,t,edge_feat,num_nodes,train_mask,val_mask,
  test_inductive_mask,test_mask. It is not the public CSV parser.
- `dataset/DTDG/torch_dataset/link_pred/node_feat_static.py`, SHA
  `de5cbc2996c797df5982ff3e6ba6cd94c59c4da63921f3b8d8185e9b6f602234`:
  [line13](../../runs/graph/upstream-pin-20261012/source/dataset/DTDG/torch_dataset/link_pred/node_feat_static.py#L13)
  constructor takes `(data_loc,data_name,split="train",node_feat,
  node_feat_dim=1,to_dense=True,return_edge_info=False)`.
  Its base loads/sorts the entire NPZ before split filtering
  ([base:25](../../runs/graph/upstream-pin-20261012/source/dataset/CTDG/torch_dataset/link_pred/node_feat_static.py#L25),
  lines46–86); the DTDG loader enumerates only observed timestamps at lines33–45
  and repeats the first split-local graph as its predecessor at lines53–65.
  Its dense return at line85 has **six** items despite the shorter annotation.
  Therefore do not use this loader unchanged for sealed test access, missing
  timestamps or boundary conditioning.

Data metadata revision is `603388f332c7756f90bc1702b26053c2bc834654`.
The existing frozen `runs/graph/upstream-pin-20261012/hf-tree.json`, SHA
`93fbd4b15f026a1f6821ca277ab24eee0aea9c7b9cec8fa27d0501756ae71225`,
already includes the primary file metadata:

- Path: `periodicity/periodicity_det_(2, 32)_100N_40tr_4vl_4ts_pER(0.01).csv`.
- Declared bytes: 6907326; declared LFS SHA-256:
  `5a19f9e7949668473bc61cb59b6215f25f093ea0ee2d45b5e143338ded5523e3`.
- Git blob OID: `796cff0d9290a13998f0a64bd924be256cde1fcb`.

`hf-metadata.json`, SHA
`c6b519445defa651d88e4ff5247498e13d96d6830d3bf51cd184954cc4421e75`,
binds that repository revision. This task checked those **metadata** records,
not the actual CSV payload. Payload acquisition/body-byte SHA and size matching,
CSV columns/types, parsed timestamps/weights and conversion parity remain
**unverified**. Root can stage the exact revision blob on scratch within its
existing storage envelope; no local dataset payload or download occurred here.
Do not substitute a generated replacement for that available metadata lead.
For a separately chosen newly generated route,
declare it prospective and bind the public generator file/import closure,
entrypoint/config, NetworkX/NumPy/Python/Torch versions and all seeds before
generation. The source's full generator imports matplotlib and another
negative-generator module; their import closure is not established by the
isolated cell loader. The frozen CPU lock includes NetworkX 3.0 and NumPy 1.24.4;
it is not by itself evidence of a complete runnable generator environment.
No generator is part of training startup: generation resets global RNGs at
graph_generator.py:52–56 and periodicity.py:143–145.
The pinned legacy shell
[fixed_er/_main.sh:12](../../runs/graph/upstream-pin-20261012/source/scripts/dataset/periodicity/fixed_er/_main.sh#L12)
calls `T-GRAB.dataset.DTDG.graph_generation.run periodicity`, maps positional
node/topology/train/validation/test/probability/pattern arguments and fixes
seed 12345 at line15. Its Python 3.8/module/environment/visualization assumptions
are historical source, not a prepared command for this environment. That
generator `run.py` is absent from the bounded local fixture; entrypoint/import
closure and generation flags must be acquired/pinned before calling it.

Legacy checkpoint, separate head weights, runtime/RNG history, exact data and
producer linkage remain null as in config:34–57. A bounded archive search is
not global checkpoint absence. Any new fit must be labeled corrected prospective
reconstruction; similarity to a graph figure does not identify old checkpoint
bytes. Keep AB/Pdet(2,1), AABB/Pdet(2,2) and primary Pdet(2,32) separate.

## Dataset store to implement

Implement a new verified store, not a wrapper around split-local predecessor
tuples. Its public contracts are **proposed new interfaces**, not existing API:

```text
read_manifest(path, expected_sha256, guard) -> VerifiedDatasetManifest
open_train_validation(manifest, guard) -> TrainingValidationStore
store.train() -> immutable Sequence[Snapshot]             # t=0..2559
store.validation() -> immutable Sequence[Snapshot]        # t=2560..2815
store.sealed_test_opener(finalized_selection) -> lazy one-shot iterator
```

Conversion must seal 3072 timestamps, N=100, node IDs 0..99, float32 identity X,
unit edge weights, two distinct nonempty undirected off-diagonal patterns,
32 A/32 B repetition, and exact 40/4/4 periods. Check every actual boundary and
unit-weight/reverse-edge alignment. If a marker is found, record/remove only
the identified source marker, preserve its timestamp, then reject the candidate
if its off-diagonal pattern is empty. No target/feature repair or dropped step
is implicit. Reject duplicate raw edges with conflicting weights; deduplicate
identical edges explicitly. Require one allowed nonedge destination for every
positive source row before optimization.

The existing [Snapshot](../../src/representation_collapse/graph_reconstruction.py#L278)
stores timestamp,nodes,positive_pairs,node_ids; it has no edge-weight array
and permits unsorted edge tuples. Thus this unit-weight candidate must verify
weights in conversion and record their hashes outside Snapshot. Freeze positive
edges in lexicographic(u,v) order, then create immutable tuples. This explicit
prospective serialization choice fixes negative-RNG alignment and FP32 reduction
order; it is not recovered historical edge ordering. Preserve this same order
in native edge_index, weights, loss pairs and checkpoint provenance.

Store separate train/validation and sealed test payloads; eager loading of the
combined NPZ by the upstream dataset is incompatible with this access policy.
Independent conversion verification may bind test bytes/schema/patterns to a
sealed manifest; the fitting process must not deserialize test labels during
training or selection. Test access is an enforceable capability, not a split
name passed to a loader that first reads all rows.

The manifest must bind origin/revision/raw bytes; converter revision/hash;
generator route/versions/seeds or explicit non-generation status; parsed column
mapping/dtypes; node and edge order; feature/weight encoding; patterns and
empty-marker decision; each timestamp's ordered snapshot hash; split payload
hashes; exact intervals; and validation of all preconditions. Record file-byte
SHA separately from a canonical semantic array/snapshot digest: a serialization
container's metadata is not graph semantics. No dataset payload currently
satisfies this gate. Acquisition belongs to root's recorded scratch staging
authority; generation is a separate explicit prospective choice.

## Native construction, state and seed contract

Preserve L=1/N=100/input=100/H=128/K=2 and the exact public head with two hidden
layers/dropout 0.2. One-hot does not mean one channel:
[utils/node_feat.py:20](../../runs/graph/upstream-pin-20261012/source/utils/node_feat.py#L20)
uses `torch.eye(num_nodes).float()` regardless of the feature-dimension default.
Retain the head's source train/eval semantics and no recurrent dropout. Use
FP32, Adam lr 1e-4/betas (0.9,0.999)/eps 1e-8/decay 0/amsgrad false, no AMP,
clipping, batching across time or unreviewed numerical optimization.

Seed before model and optimizer construction using the actual
[fresh_objects signature](../../src/representation_collapse/graph_reconstruction.py#L575):
`fresh_objects(seed, *, seed_all, model_factory, optimizer_factory)`.
Seed 0 covers Python, global NumPy and Torch; record deterministic flags and
postinitialization RNG state, without an extra reseed before dropout training.
Use fresh objects in a fresh worker. Construct the separate NumPy negative
Generator at seed 1 and keep it independent of global NumPy and generation.
Model seed is not an independent graph realization. Adam parameter order is
sorted named recurrent parameters followed by sorted named head parameters,
with alias rejection ([GR:729](../../src/representation_collapse/graph_reconstruction.py#L729)).

The old guarded factory imports the public Temporal umbrella at GR:714–715
and the original metadata probe compares literal base versions at GR:184–186.
Do not enable it by changing booleans. A new reviewed integration must bind
the exact CPU wheel build tags/lock and the approved isolated import route,
while preserving base source versions. The CPython 3.10 Linux-x86_64 fixture
environment is explicitly separate from mainproject Python >=3.11 installation.
Immutable CPU manifest/lock SHA are
`67e27963f2166f8a7e26dba301c9020d2bd7d4c012010fb859e6e43b15ce9842` /
`ea2f58d2a12f5d8562b31d5fa7b5c646121784867c134afd4a5cf6a8108b8c06`.

The actual helper API is
[pinned_graph_import.py:128](../../src/representation_collapse/pinned_graph_import.py#L128),
`load_pinned_graph_modules(*,source_root,installed_cell_path,cell_sha256,
wrapper_sha256,guard) -> PinnedGraphModules`; source SHA
`2f14b3683f67d33f4a5f6325b5b204aee3a086e1fad1966acb9810a302d04344`.
Verify helper bytes before execution, installed path from distribution metadata,
fresh namespaces, exact source/class identity and
`source_release_hashes_match is True`. Preserve provenance that full Temporal
API is unverified ([helper:233](../../src/representation_collapse/pinned_graph_import.py#L233)).
Only enter this route after successful mandatory reporting and Torch import.
Its calibration acceptance never establishes N=100 training-gradient/checkpoint
parity, generator compatibility or a source-matched benchmark result.

The exact state records are
[StateLayout](../../src/representation_collapse/graph_reconstruction.py#L225)
and [CompleteState](../../src/representation_collapse/graph_reconstruction.py#L247):
H and C each [1,100,128], next_target_t, canonical node_ids. Joint D=25600,
flatten hidden then cell, layer,node,feature, all ascending. Do not feed a
flattened array where the backend expects CompleteState. The NumPy flattening
helper is a synthetic reference, not a native serializer.

## Lifecycle and exact chronology

The new runner must implement the following phases durably; these names are a
proposed state machine, not an API claimed to exist:

1. **prepared → admitted → reporting_initialized → constructed.** Verify all
   pinned config/source/environment/data/review/admission inputs and physical
   ownership. Create no native model before reporter entry. Record fresh model,
   head and optimizer/RNG identities. No dataset generation or implicit resume.
2. **training_epoch(e).** Enforce full train timestamps 0..2559 and exactly 2559
   updates for an eligible epoch; reset H/C zero. For t=1..2559: check budget,
   detach prior state, consume G[t−1] once with its matching weights to produce
   s[t], score target G[t] from s[t], calculate loss/backward/update. Do not
   consume G[t] before that loss or consume G[2559] merely to complete training.
   This matches GR:515–526. Gradients span the one predecessor consume only.
   Training states were produced under changing parameter values; they are not
   fixed-weight trajectories for tangent/Floquet analysis.
3. **epoch_weights_frozen → validation(e).** Snapshot both epoch-final models,
   reset zero and replay G[0]..G[2559] in eval/no-grad. For targets 2560..2815,
   use fixed pair IDs to score first, then consume the target once. Scoring
   receives state and pairs only, never current-target graph/labels/weights
   ([GR:545](../../src/representation_collapse/graph_reconstruction.py#L545)).
   The next training epoch resets independently; no validation cache seeds it.
4. **validation_complete(e) → selection_update.** Only complete, finite,
   integrity-verified validation can select. Persist the candidate weights and
   complete metrics before atomically changing the selected pointer. A partial
   validation cannot select, even when its training epoch finished. Record
   complete training loss exactly once; evaluate the current complete epoch
   before normal patience termination when the work budget permits.
5. **normal_training_stop → selection_finalized.** Apply training-loss patience
   and epoch/update/time caps independently of validation-score maximization.
   Any integrity/reporting failure or partial unfinished finalization leaves
   the selected result unaccepted and test unavailable. Cap termination can
   preserve a previous verified selection, but test may proceed only inside
   the remaining admitted physical work time; no extension is implied.
6. **selected_loaded → prefix_replayed → test_opened → terminal.** Reload both
   components from the selected reference; verify payload parity and budget/STOP
   after load. Fresh eval/no-grad replay train+validation 0..2815. Then score
   each test target2816..3071 before one consume. Once test opening is attempted,
   all future training/validation/test attempts are terminally forbidden, even
   if opening/reading fails. Preserve GR:564–571's lifecycle repairs.

Existing `test_once` invokes `open_test()` before `_evaluate` replays the
conditioning prefix. To support the stronger sealed-store lifecycle, return a
lazy one-shot iterator from that callback: no test payload I/O until its first
`next`, after conditioning and load-parity checks. Test capability use is still
marked terminal before callback invocation. Alternatively implement/review an
explicit replay-before-open method; do not claim the current helper already
does that. Validate every split interval/length in the new orchestrator;
`_evaluate` only verifies contiguous chronology and nonempty outputs.

Backend interface remains the actual [ContractBackend:468](../../src/representation_collapse/graph_reconstruction.py#L468):
training/evaluation context managers; reset; detach(state); consume(state,
snapshot); score_pairs(state,pairs); loss_and_update(state,target,negatives);
load_selected(reference). Proposed monitoring callbacks may observe returned
complete states and counters; they must add no model/head forward or state
advance. Existing consume asserts z equals last-layer H at GR:659–663.
Store state phase, next target, last input/weights, node order, model reference
and derived-z timestamp; the wrapper's z is not an independent advancing cache.

## Loss, negative sampler, metrics and stopping

Use the existing [sample_negatives:315](../../src/representation_collapse/graph_reconstruction.py#L315)
and [validate_negatives:305](../../src/representation_collapse/graph_reconstruction.py#L305)
semantics: one same-source destination per positive, uniformly from all current
off-diagonal nonedges, replacement/duplicate negatives allowed. Test against
the whole integer positive set, not Tensor identity or a recursive subset.
Empty positives/saturated source rows fail finitely. Pass ordered positives
and matching negatives to the source head separately; do not merge their mean
BCE into one pooled mean. This explicitly corrects upstream sampler behavior
at `train/DTDG/link_pred/trainer.py:194–219`.

The actual [native loss method:674](../../src/representation_collapse/graph_reconstruction.py#L674)
zeros gradients, scores positive and negative pairs in two head calls, adds
their separately averaged BCE, checks finite loss/gradients, calls Adam.step,
then checks finite parameters. Preserve those semantics; source BCE and head
construction are `train/DTDG/link_pred/trainer.py:68–80,287–301`.
Do not add source-style all-pair training F1 computation: it would add an
unbudgeted third head call and a different dropout draw sequence.

Maintain separate durable counters for loss/update attempts (charged before
entry), optimizer.step entries, step returns, and successful backend returns.
The current helper increments budget.updates only after backend return
(GR:522–523); Adam can already have changed weights before a post-step error
at GR:694–696. Treat such a failure as an attempted/possibly committed update,
stop, preserve it and never retry invisibly. Count every native cell/head entry
before invocation with the frozen L1 schedule; record partial counts on failure.

Eligible epoch loss is the arithmetic mean of 2559 successful update losses.
Use actual [LossPatience.observe_complete_epoch:382](../../src/representation_collapse/graph_reconstruction.py#L382):
strict relative improvement `(best-current)/max(abs(best),1e-12)>1e-6`, else
increment; stop after 100 consecutive nonimproving complete epochs. Maxima remain
200 complete epochs, 511800 attempted updates,3600 entire-pilot seconds.
The old boundary budget begins when PilotBudget is constructed (GR:448–452);
a new launcher must bind its elapsed clock to the supervisor's pre-import
start, not reset time after imports/reporter. Normal epoch/update caps must
stop training while permitting only already-admitted final evaluation inside
remaining time. Partial epochs cannot update selection or loss patience.

Use [ValidationOnlySelection.consider:415](../../src/representation_collapse/graph_reconstruction.py#L415)
and finalize:430 semantics: complete-epoch validation only, strictly increasing
epoch number, strict maximal time-mean F1, earliest ties; finalize exactly once.
Test never affects stopping/selection. No threshold is selected from held-out
performance and no perfect-test-F1 early stop is inherited from upstream.

Metrics use [snapshot_metrics:347](../../src/representation_collapse/graph_reconstruction.py#L347)
and [time_mean_f1:369](../../src/representation_collapse/graph_reconstruction.py#L369):
9900 ordered off-diagonal pairs, probability >=0.5, zero_division=0, each target's
confusion counts/F1 before arithmetic time averaging. Never use sampled-negative
or pooled-count F1 as the primary metric. Store source/target t and phases,
actual graph inequality and per-direction/phase curves. Primary test has 256
targets/eight transitions/248 nontransitions after conversion proves distinct
patterns; target phases 0/32 and source phases 63/31. Persistence uses the same
predecessor and mask. Exact EdgeBank definition and untrained reference schedule
remain separately unresolved; neither may silently add a fit or forward pass.

## Checkpoint codec and load-parity work to implement

The actual [CheckpointReference:395](../../src/representation_collapse/graph_reconstruction.py#L395)
has only epoch and three SHA identities. It is not a codec or trusted path map.
Implement a versioned data-only checkpoint format with regular bounded files,
primitive JSON metadata and tensor arrays loaded without object pickle
(for an NPZ implementation, `allow_pickle=False`, object dtypes forbidden).
Do not assume undocumented Torch deserialization flags for the legacy runtime.
Final serialization encoding, tensor/RNG tags and size limits must be frozen
and independently reviewed before payload writing/loading; no codec exists now.

Required selected manifest: recurrent and head payload file hashes plus tensor
name/dtype/shape/byte-order/semantic identities; parameter order; exact wrapper/
cell/helper/config/environment/data/feature/node-order hashes; epoch; complete
validation rows and selected F1; reset/dropout/precision/deterministic settings;
initialization/RNG provenance; and the fixed-weight validation terminal-state
hash s[2816]. Hash the finalized manifest externally, not circularly inside
itself. `checkpoint_sha256` identifies that manifest; recurrent/head SHA identify
the separately declared component payloads. Create CheckpointReference only
after durable writes/fsync/hash verification; atomically publish selected
pointer only after valid validation and strict improvement. Retain failures
and lineage instead of overwriting them as later success.

Resume is a different, explicitly unselected artifact: both components, Adam
state/group-to-named-parameter mapping, Python/NumPy/negative-Generator/Torch
RNG states, complete H/C, next target/last input, modes, epoch/update attempts/
step counters, patience, selection state, and original deadline/STOP lineage.
Numeric tensors and tagged primitive RNG records must round-trip exactly;
unknown tags/shapes/objects fail closed. Automatic resume is disabled. Training
H/C may reflect earlier weights, so record its actual construction chronology
rather than claim it is the selected fixed-weight validation state.

The loader must verify trusted manifest/payload hashes before decoding, names/
shapes/dtypes, exact architecture, state layout, device/environment and source
identity; restore both recurrent and head, set eval, and assert both resulting
state_dict identities. Do not restore optimizer/RNG/training state when loading
selected weights for inference. No filename-only cache or implicit old run.
Assert semantic byte equality for codec round trips. Native forward/dropout/
loss-gradient and resume parity need a separately finite engineering fixture
with reviewed tolerances/counts; tiny AD calibration does not supply those.

For a full selected checkpoint, reuse already-budgeted final conditioning:
its state after replay0..2815 must match the saved selected-validation terminal
state s[2816] under identical frozen CPU/eval conditions. This adds no cell/head
call and checks complete-state replay/load parity before test payload access.
Any additional pre/post-load forward comparison must have explicit fixture
counts; it cannot be hidden inside the canary or final evaluation schedule.

## Reporting, process boundary and durable evidence

Use `ExperimentReporter(output, *, project, config, identities, entity=None,
mode="offline", run_id=None)` at
[reporting.py:95](../../src/representation_collapse/reporting.py#L95), SHA
`443c9e89d2eba954ee60cc35e015fe61196fbec2f50331067540b4a8259e8184`.
Pin SDK 0.30.0/offline project representation-collapse; no disabled/fallback mode.
Before allocation, exercise the exact compact metadata mappings against the
source schema without entering the SDK: config/metrics are explicit scalar
mappings, identities are hashes only, each mapping <=64 entries/16 KiB, and
prohibited key patterns/object payloads are refused as documented at lines56–83.
Paths are invalid identity values; permitted compact public relative paths can
be scalar config values. Keep absolute local paths in the local manifest.
Put arrays/data/weights in local artifacts, not reporter scalar payloads.
`log(metrics, *, step)` requires strictly increasing nonnegative steps
([line201](../../src/representation_collapse/reporting.py#L201)); record phase
and epoch/update as separate scalar fields, not reused step counters.
New output/run ID and no SDK resume; reporting failure is terminal.

A stdlib-only launcher must start monotonic time and deadline/owned-child
journal before NumPy/Torch/SDK imports, initialize reporter successfully,
then import GR (which imports NumPy at line27), validate data payloads and
construct models. Source/config/receipt text checks can precede reporter;
model/data/array payload hashing and numerical work cannot. Register SDK
children before SDK import, cap CPU/numerical/Go threads 1, bind actual Slurm
job/CPU affinity and kernel memory cgroup. RLIMIT_AS is not physical RSS.
Reuse ownership helper only with its exact SHA
`8e1f388d4f5ce72e0bfd9a4b5a4b1f8c98f66f2512841d96875822bfd9bffe09`;
actual signatures are `process_identity(pid)` at line18,
`install_spawn_journal(path)` at 139, `OwnedProcesses.refresh(timeout=0.5)`
at 171 and `cleanup(deadline=None)` at 202. No process outside this run's
PID/start ownership set may be killed. An in-process budget cannot preempt
hung native/I/O work; kernel allocation and outer physical supervisor are required.

Durable outputs: immutable argv/config/admission and source manifests; environment
and conversion receipts; initialization and RNG identities; progress plus all
attempt counters; every validation/test metric row; selected/resume manifests;
complete-state records with phase/weight provenance; source/semantic/file SHA
manifest; offline W&B history/summary/finish receipt; supervisor/child cleanup
and actual allocation/RSS/status. Atomic success requires all mandated outputs,
not an exit code alone. Offline durability does not prove cloud delivery.
Whole-project scratch usage plus conservative new output/environment/cache
reservation must remain within 20 GiB; no data/checkpoints/cache into the Mac repo.
State capture must observe existing forwards; covariance/tangent analysis and
dimension-based selection remain outside this training contract.

## Finite throughput canary and stopping

The readiness proposal is **one new, separately configured and admitted CPU
canary**, not a fit or a native-calibration retry: CPU 1/physical 2 GiB/600 s,
work deadline 590 s plus 10 s cleanup inside the cap, zero GPUs, one submission,
no retry/requeue/race/resume. These are proposed maxima, not authority or a
promise the schedule completes. Future exact source/config/argv/account/
partition/output/STOP/admission hashes must exist before approval/submission.

Use the same primary graph/model/head/loss/seed/negative chronology, but only
targets 1..256 and at most 256 loss/update attempts. Implement a separate partial
training-window path that **never increments complete epochs or invokes
LossPatience/selection**. Do not call `training_epoch(...,expected_updates=256)`:
the existing synthetic-flexible helper increments budget.epochs at GR:528 and
would mislabel a partial 40-period epoch as complete.

After the last update, freeze partial-fit weights; if the deadline permits,
fresh eval replay 0..2559 and score/consume 256 validation targets once. These
metrics establish throughput/integrity only. Test opener remains inaccessible,
selected-checkpoint count is zero, and any saved checkpoint is explicitly
unselected partial fit. Do not add an initial untrained replay or extra metric
head calls to this canary.

Successful maxima: 256 training cells/512 heads; 2816 evaluation cells/256 heads;
total **3072 cell entries, 768 head entries**, 256 backward/update attempts,
one wrapper/contained-cell/head/Adam construction each. No dense/JVP/VJP/FD,
eigensolver, covariance or additional native-parity fixture entry. Count API
entries separately from internal reverse kernels. Evaluation emits at most
2534400 pair probabilities; positive/negative training counts are actual data
counts, not guesses from edge probability. A partial run retains actual counts
and never claims the expected successful totals.

Record first 32 updates as warmup and remaining 224 as at most 14 consecutive
16-update timing windows, plus startup, sampling, backward/Adam, replay/head,
checkpoint/reporting and cleanup phases. Record actual pairs/update, useful
updates/s and physical RSS; missing/partial windows stay missing. Retain all
failures. Stop at first count/time/memory/STOP/nonfinite/integrity/reporting
bound; no resource increase or retry follows. Fit-cost forecasts must use this
N=100 workload, not N3 calibration timing. A GPU request requires a separately
pinned CUDA/extension/device packet and measured matched useful throughput;
no GPU allocation is proposed ready here.

A later fresh full fit keeps frozen 200 complete epochs/511800 attempts/3600
entire-pilot seconds. For count planning only, per epoch is 5375 cells/5374
heads including full validation; at 200 epochs plus one final test replay,
maxima are 1078072 cells/1075056 heads. Initial untrained reference is not in
those totals and remains an explicit optional protocol decision with extra
2816 cells/256 heads. No full-fit convergence or completion-time claim follows.

## Acceptance and critical path

**Engineering readiness:** verified immutable primary store/serialization;
exact import/build/schema/reporter metadata; fresh seed/object/RNG contracts;
causal full-split spies; adversarial sampler/metric fixtures; strict patience/
selection/terminal-test lifecycle; attempted-step accounting; data-only codec
and full H+C/load parity; finite native loss/dropout/checkpoint checks; physical
supervisor/owned cleanup; immutable reviewed runnable config and fresh admission.
Native fixtures need their own frozen counts and tolerances before execution;
this document ran none. Do not bypass the existing unconditional preflight
blocker or original disabled config. Implement a versioned new runner and config
referencing the unchanged canonical candidate SHA
`dea0e7684f52a0a53cba5072e77778d22995cb0b0e30f3352a6e349b89628092`.

**Scientific acceptance:** pending a separately reviewed protocol with explicit
competence/controls/replication rules. Source probability threshold/loss patience
are not competence criteria. A throughput canary or N3 calibration cannot
establish trained task competence, historical byte parity, orbit closure,
Floquet contraction or harmful collapse. Keep all time/transition curves and
negative/failed outcomes; do not tune for a decline, desired dimension or sign.
One seed on fixed patterns supplies no seed-level uncertainty. Covariance,
tangent behavior and task-required recoverability remain separate estimands.

The shortest useful implementation sequence is: (1) preserve the accepted
reporter repair and independently reconcile the calibration, then validate
the new trainer's own actual metadata mappings;
(2) obtain/freeze primary payload+conversion/import closure; (3) implement
trusted dual-component/RNG codec and sealed store; (4) implement full/partial
orchestrators with counters and reporting/physical supervisor; (5) complete
dependency-free review followed by separately admitted finite native parity;
(6) publish the exact disabled canary packet for owner budget review. No
training CLI, data payload, checkpoint codec or launch authority is created by
this contract. The root coordinator owns integration and any future execution.
