# Selected-inference checkpoint codec proposal — October 16

**Proposed specification only; no codec or checkpoint has been implemented,
written, loaded or admitted.** Root must independently review and freeze this
format, its architecture catalog and integration contract before implementation.
This proposal grants no model, numerical, SDK, dataset or compute execution.
It covers the primary CPU/FP32 selected-inference checkpoint only. Resume,
optimizer serialization and RNG restoration are deferred to a separate format.

## Source anchors and existing limits

The governing [trainer contract, checkpoint section](tgrab-training-runner-implementation-contract-2026-10-15.md#checkpoint-codec-and-load-parity-work-to-implement)
requires both model components, safe data-only tensors, complete selected
validation terminal state, validation-only publication and trusted hash checks.
Its inspected SHA256 is
`91745076bbcb4c963530980b6b58471a9f184835b72fc0cc4fcde82573a93d73`.

[graph_reconstruction.py:395](../../src/representation_collapse/graph_reconstruction.py#L395),
SHA256 `89894f8bd3e8574b5b57c777353fe1aac3a1f123f792467a009b04b8a6f9bcdb`,
defines actual `CheckpointReference(epoch, checkpoint_sha256, recurrent_sha256,
link_head_sha256)`. It contains neither paths nor decoding authority.
[ValidationOnlySelection:409](../../src/representation_collapse/graph_reconstruction.py#L409)
requires increasing complete-epoch validation and strict F1 improvement; ties
retain the earlier epoch. `finalize()` is one-shot. These public mutable fields
are cooperative state, not proof of durable selection history.

[StateLayout:226](../../src/representation_collapse/graph_reconstruction.py#L226)
and [CompleteState:248](../../src/representation_collapse/graph_reconstruction.py#L248)
define separate hidden/cell tensors and canonical node IDs. The native backend
uses `(1,100,128)` FP32 tensors and resets both to zero at
[line633](../../src/representation_collapse/graph_reconstruction.py#L633).
[load_selected:699](../../src/representation_collapse/graph_reconstruction.py#L699)
currently raises unconditionally. The backend's old guarded factory remains
blocked; this proposal neither enables it nor modifies its preflight/config.

The [public wrapper:20](../../runs/graph/upstream-pin-20261012/source/model/gclstm.py#L20)
SHA256 `86069a32fced5f09c60cd795e4f0abd82674dafae250ef3673ac8c3650504609`
contains one actual GC-LSTM unit for this candidate. The
[public head:5](../../runs/graph/upstream-pin-20261012/source/model/link_predictor.py#L5),
SHA256 `cca12cfc73b8660c92236f530504da08ba0b606ec13a1a681ffabcae43b7ca86`,
has Linear/ReLU/Dropout layers and a sigmoid output. The inspected
[cell:58](../../runs/graph/upstream-pin-20261012/gclstm-v0.54.0.py#L58)
SHA256 `c3568d2c1ba8926ab43e11c840213f90ebe782ad158dd76809d1dc1c255797af`
declares gate matrices/biases and ChebConv submodules. The actual PyG ChebConv
state-dictionary catalog has **not** been independently inspected in this task.

The frozen reconstruction JSON SHA256 is
`22bef6bf781c5bfb5493dd5e03169ea0e658b2463ccaa9ba09189af454508f1f`;
its canonical candidate identity remains
`dea0e7684f52a0a53cba5072e77778d22995cb0b0e30f3352a6e349b89628092`.
The approved isolated importer
[pinned_graph_import.py:128](../../src/representation_collapse/pinned_graph_import.py#L128)
SHA256 `2f14b3683f67d33f4a5f6325b5b204aee3a086e1fad1966acb9810a302d04344`
provides class/source provenance without changing cell/wrapper bytes. Its tiny
engineering acceptance is not N100 training or checkpoint/load parity.

## Proposed data-only file format and strict bounds

Choose a versioned, uncompressed binary format with canonical JSON descriptors:
`kind=tgrab_selected_inference_v1`. This avoids pickle, arbitrary object tags,
archive members and decompression. Do not call `torch.save`, `torch.load`,
NumPy object loading, dynamic class import or executable reconstruction hooks.
The reader's metadata/hash phase can use stdlib only; native tensor copying
belongs to the separately admitted integration, after reporting initialization.

A fresh checkpoint directory contains exactly six regular files:
`recurrent.f32le`, `link-head.f32le`, `validation-terminal.f32le`,
`validation.json`, `provenance.json`, `manifest.json`. No directories, links,
devices, sockets or undeclared files are accepted. All files are single-link,
bounded and opened without following symlink components. Compare descriptor
identity/size before and after read to refuse replacement or concurrent writes.
Incremental checks must honor the caller's shared deadline/STOP guard.

Every tensor uses finite IEEE754 binary32, little endian, C-contiguous order,
with no header, padding or stride metadata inside its payload. The manifest
describes each component's tensors in lexicographic native state-dictionary
name order. A descriptor contains exactly `name,dtype,shape,order,offset,nbytes,
data_sha256,semantic_sha256`; dtype is exactly `float32_le`, order exactly `C`.
Offsets start at zero and advance by exactly the preceding tensor's nbytes;
the final end equals file length. Reject overlap, gaps, trailing bytes, missing
or repeated names, aliasing, sparse/quantized/meta tensors, unsupported dtype,
nonfinite values and lossy conversion. Snapshot detached CPU values without a
forward/backward or reseed; FP32 CPU input values must already satisfy policy.
Negative zero and subnormals retain their exact bits; normalization is forbidden.

`data_sha256` hashes that tensor's exact payload bytes. `semantic_sha256` hashes
the ASCII bytes `tgrab-tensor-v1` followed by one zero byte, canonical descriptor metadata
`{component,name,dtype,shape,order,nbytes}` excluding hashes/offset, followed by
the exact tensor bytes. This separates semantic identity from whole-file SHA.
The canonical metadata serialization is ASCII JSON, sorted object keys,
separators `,`/`:`, no whitespace, with exactly one terminal newline. All JSON
files use that same encoding. Reject duplicate/unknown keys, non-ASCII strings,
nonfinite constants, nesting beyond 8 container levels and bools in integer
fields. Strings are <=256 bytes and object mappings <=64 keys; separately
bounded descriptor/validation lists have their exact declared lengths. JSON lists
preserve declared order; descriptor names never choose a filesystem path.

Proposed hard maxima: recurrent 8 MiB, head 1 MiB, validation JSON 256 KiB,
provenance JSON 64 KiB, manifest 256 KiB, total bundle 12 MiB; at most 128 model
tensors, each <=1 MiB. Model tensors have rank 1 or 2, positive dimensions <=256,
and exact name/shape/count from the separately trusted architecture catalog.
The terminal-state payload is **exactly 102,400 bytes**, not merely below a cap:
hidden then cell, each 12,800 values/51,200 bytes, shape `(1,100,128)`. The state
flattening order is component(hidden,cell), layer, ascending node ID 0..99,
feature 0..127, with joint dimension 25,600. No independent cached `z` is stored;
its provenance says it derives from last-layer hidden at the same timestamp.

Before implementation acceptance, freeze a source-derived full catalog of
both components' expected native state-dictionary keys, shapes, parameter vs
persistent-buffer classification and lexicographic serialization order. The
catalog is a trusted input hash, never learned from a payload. Direct source
anchors establish `units.0.W_{i,f,c,o}` shape `(100,128)`,
`units.0.b_{i,f,c,o}` shape `(1,128)` and gate ChebConv submodules. The head's
source establishes `seq.1.weight/bias` shapes `(128,256)/(128,)`,
`seq.3.weight/bias` `(128,128)/(128,)` and `seq.5.weight/bias` `(1,128)/(1,)`.
Exact ChebConv descendant keys and shapes remain a **freeze blocker** pending
inspection of the pinned PyG2.6.1 source/wheel. Do not guess keys from another
version, accept a wildcard namespace, omit persistent buffers, or let a file
declare a replacement catalog. Unexpected valid native keys require review.

## Required manifest, validation and primitive provenance

The manifest requires an exact versioned schema with these named sections:
`kind,schema_version,artifact_role,epoch,components,terminal_state,validation,
provenance,identities,architecture,semantics`. Each nested record has a closed
key set; the detailed schema must be frozen before implementation, and
extensions require a new format version. In particular:

- `components` has exactly `recurrent,link_head`; each record has
  `filename,bytes,sha256,tensors`, with descriptors defined above.
- `terminal_state` has `filename,bytes,sha256,tensors,next_target_t,
  last_consumed_t,construction,epoch,recurrent_sha256,link_head_sha256`.
- `validation` has `filename,bytes,sha256,split,interval,row_count,time_mean_f1`.
- `provenance` has `filename,bytes,sha256`.
- `architecture` has `layers,num_nodes,input_channels,hidden_channels,
  chebyshev_order,laplacian_normalization,convolution_bias,recurrent_dropout,
  head_layers,head_dropout,dtype,device,state_shape,node_ids,flatten_order`.
- `semantics` has `node_order,positive_edge_order,features,weights,
  prediction_cache,reset,validation_chronology` with the exact policies below.

`artifact_role=selected_inference_candidate` denotes a durable validated
candidate, not proof that it won selection. Only a trusted committed selection
receipt/pointer confers selected status. The manifest contains no self-hash.

`components` records both fixed payload basenames, exact sizes/file SHA256 and
the full ordered tensor descriptors. `terminal_state` does the same for hidden
and cell and records `next_target_t=2816`, `last_consumed_t=2815`,
`construction=fresh_zero_eval_replay_0_through_2815`, selected epoch and both
component payload hashes. These are the epoch-final fixed weights used for the
entire validation replay. A training carry with earlier-weight history cannot
be substituted. Hidden/cell hashes bind both their values and state chronology.

`validation.json` contains `kind=tgrab_complete_validation_rows_v1`,
`split=validation`, epoch, ordered rows and `time_mean_f1`. Exactly 256 rows cover
timestamps 2560..2815 once, in order. Each has the existing `snapshot_metrics`
fields: target/source timestamp and phase, expected phase transition,
change-point flag, pairs, TP/FP/FN/TN, precision, recall, F1 and accuracy.
Require source timestamp=t-1, period 64/block 32, 9,900 ordered offdiagonal pairs,
nonnegative integer confusion counts summing to 9,900 and finite metrics in [0,1].
Reconcile fractions, zero-division 0 and arithmetic time-mean F1 with the frozen
existing metric formulas. No pooled F1 substitution, sampled evaluation pairs,
test rows, probabilities or raw labels belong here. The manifest binds this
file's hash, interval, row count and exact scalar selection score. A complete
epoch is 1..200 and reflects 2,559 completed updates; attempt/completion counters
are recorded separately and never inferred from a filename.

`provenance.json` is primitive-only JSON, containing run/attempt/job identifiers,
source commit and file hashes, exact executed configuration and candidate SHA,
actual environment/lock/artifact-manifest hashes and versions, model seed 0,
independent negative Generator seed 1, seeding-before-construction receipt,
initial recurrent/head semantic identities and pinned initialization RNG
provenance-record hashes (including postinitialization RNG identities). Refer
to actual retained initialization records; do not invent an RNG state from its
seed. These hashes preserve lineage; the inference loader never restores RNG.

It additionally records actual initialization/construction order, successful
training epoch/update and attempted counters, fixed-weight validation replay
chronology, reset-to-zero policy, sorted recurrent-then-head optimizer parameter
name order, deterministic flags, precision FP32/AMP false, CPU device, model/head
train-vs-eval/dropout policy and complete module-mode receipt hash. No optimizer
state, tensor object, environment mapping, credentials or executable object is
accepted. Public paths may be local-only receipt fields, never hash identities.

`identities` binds raw-data SHA, converted manifest SHA, train/validation/test
split file and semantic SHA from that trusted converted manifest, conversion
protocol/converter hashes, node-order/feature/weight semantic hashes, recurrent
wrapper/head/cell/importer source hashes, architecture-catalog SHA, checkpoint
codec/protocol hashes, trainer source and reporting/admission lineage. Binding
the sealed test split's already-declared hash reads no test payload. No fallback
to a filename, another run, an unverified old checkpoint or a matching seed.

Architecture/semantics require L1/N100/input100/H128/K2, symmetric Laplacian,
convolution bias enabled, no recurrent dropout and exact public head/dropout 0.2;
state shape/order above; canonical nodes and lexicographic positive edge order;
float32 identity features and aligned unit weights. Exact CPU environment/build
tags and class/source identities are part of compatibility. This version
rejects GPU/device migration and mixed precision; those require separate policy.

## Durable publication and proposed typed interfaces

The following are **proposed new APIs**, not existing calls:

```text
write_inference_candidate(new_directory, recurrent_named_tensors,
    head_named_tensors, terminal_complete_state, complete_validation_rows,
    primitive_provenance, trusted_catalog, guard) -> DurableCheckpointCandidate

verify_inference_candidate(manifest_path, expected_manifest_sha256,
    expected_recurrent_sha256, expected_head_sha256,
    trusted_architecture_and_run_identity, guard) -> VerifiedInferenceCheckpoint

commit_validation_improvement(candidate, actual_ValidationOnlySelection,
    trusted_previous_selection_receipt, new_receipt_path, pointer_path, guard)
    -> CommittedSelectionReceipt

load_verified_for_inference(verified_checkpoint, actual_native_backend,
    trusted_expected_identity, guard) -> InferenceLoadReceipt
```

Verified/durable result types are frozen, factory-only data records. A strict
boundary validates exact built-in integer epoch, lowercase 64-hex digests and
the actual GR reference class; constructing a similar object is insufficient.
`CheckpointReference.checkpoint_sha256` is the externally computed final manifest
SHA; recurrent/head SHA identify their respective complete payload files. A
trusted root admission/selection receipt maps those identities to the exact
scratch directory, expected architecture catalog and dataset/run lineage.

Write the first five payload/metadata files exclusively in a new private
directory, fsync each, reopen/read/hash-verify them, then exclusively publish
the manifest only after every prerequisite holds. Fsync the directory. A
published manifest never has an incomplete dependency. Only then construct the
actual `CheckpointReference`. Retain incomplete directories/failure receipts;
they are not eligible candidates. Never overwrite an existing bundle or reuse
an attempt after failure. No model call is needed for persistence/hashing.

For selection, verify complete validation, score and epoch progression against
the trusted prior committed history. Strictly larger F1 wins; equal scores
retain the earlier reference. Use actual `consider(reference,f1,
split="validation",epoch_complete=True)` semantics and record every decision,
including non-improvements, in immutable chained receipts. Successful improvement
publication consists of a durable new selection receipt and atomic replacement
of a small `selected.json` pointer referencing that receipt and the verified
manifest/reference; fsync the pointer's directory. Serialize this operation
under the single trainer's ownership. The immutable receipt chain supplies
history that the mutable GR selection object alone cannot establish.

If persistence fails after mutable selection changes, stop terminally; retain
the previous committed pointer and failed transaction. Do not roll fields back
and continue, or treat an orphan receipt as committed selection. Readers accept
only the latest trusted committed pointer/receipt pair; orphan candidates and
orphan receipts are historical partial evidence. Finalization similarly creates
a distinct immutable finalized receipt bound to the selected reference, score
and last completed validation epoch. It does not modify checkpoint manifests.
Bound the pointer to 4 KiB and each immutable selection receipt to 16 KiB;
admit at most 200 complete-epoch decisions plus one finalization receipt. The
trusted controller must reserve space for retained candidates and the complete
receipt chain within its separately approved whole-project scratch budget.

## Trusted load and the sealed lazy test boundary

Before tensor decoding or native copying, verify admission, committed finalized
selection history, trusted manifest SHA, all closed metadata schemas, source/
architecture/environment/data identities, every file length/SHA and every
descriptor bound/semantic digest. Reject an unsupported tag/version, unexpected
tensor, dtype/shape mismatch or nonfinite bit pattern before mutating a model.
Decode only the frozen float32 bytes. Native loading restores **both** actual
recurrent and head components with strict full-key matching and exact parameter/
buffer order; verify both resulting state-dictionary semantic identities.
Failure after any partial model mutation is terminal, with no scoring/test read.

Put both components and every submodule in evaluation mode; assert dropout
inactive, gradients disabled for inference, CPU/FP32 state and exact model/class
identity. This loading API neither constructs an optimizer nor restores RNG,
patience, training state or the saved terminal H/C as the initial test state.
The saved H/C is the parity reference. Test starts from fresh zero H/C and the
already-budgeted fixed-weight replay 0..2815; compare the resulting full H+C
semantic bytes and next-target timestamp with saved s[2816] before opening any
test bytes. This adds zero cell/head calls. A mismatch fails closed; any proposed
relaxation requires a separately reviewed native parity policy. No tolerance
or task-competence threshold is proposed here.

Current [CausalController.test_once:561](../../src/representation_collapse/graph_reconstruction.py#L561)
loads selected weights, consumes the callback attempt, then `_evaluate` replays
the prefix before requesting a target. The actual
[lazy store opener:521](../../src/representation_collapse/graph_data.py#L521),
source SHA256 `498e60f7be048411c253e4b8cb63e4b907973f969c397bfdfacde949389f057e`,
defers test payload read until that target request. It validates a captured
selection reference/summary and never proves checkpoint durability itself.
`_evaluate` currently discards its terminal validation state and has no
pre-first-target parity hook. Therefore integration still needs an explicit
read-only state observer after existing validation consumes and a one-shot
parity gate after final replay, before iterator advancement. The gate can wrap
the lazy first-target request; it must inspect the actual existing replay state,
not execute another replay or inspect test labels. This proposal implements
neither hook. Current GR marks `test_opened` only after `load_selected` and the
following `budget.check_boundary()` both return; a load or that intervening
budget failure leaves its flag false. The sealed opener cannot consume an
attempt it has not yet received. Therefore the future orchestrator must charge
its terminal selected-test evaluation/load attempt before selected load, through
a reviewed wrapper/state-machine transition. This stronger failure rule is a
required integration decision, not an existing helper guarantee. Parity/read
failure after opener invocation already consumes the store's one-shot capability.

## Decisions to freeze and acceptance still required

Root must freeze raw f32le encoding, domain/canonicalization, bounds, exact closed
schemas, full trusted native tensor catalog, selection transaction/receipt
schema, typed interfaces and the observer/parity-hook ownership before payload
implementation. Exact primitive initialization record tags/hashes must reference
the actual future trainer's recorded initialization; they cannot be fabricated
by the codec. The catalog's unknown ChebConv keys are a concrete blocker.

Future dependency-free codec checks should cover byte round trips, endian and
negative-zero preservation, malformed offsets/shapes/counts, links/hardlinks,
unknown tags/keys, nonfinite values, component swaps, hash/source/epoch mismatch,
publication interruption and stale/mutated selection receipts. They are proposed
checks, not tests run here. A separately finite, reported and admitted native
fixture must verify real state-dictionary/name/shape load, both-component
identity, existing-forward state capture and load/eval/dropout parity. Its
counts, tolerances and resource envelope remain unresolved; tiny AD calibration
does not supply them. No native or numerical payload work occurred for this
proposal. Trained competence, reproducibility of historical weights and harmful
representation-collapse findings remain unverified.
