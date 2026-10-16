# Independent review of the selected-checkpoint codec proposal

2026-10-16. **The design direction is accepted; implementation readiness is
not accepted.** The reviewed proposal explicitly leaves its complete tensor
catalog, detailed schemas and integration hooks unfrozen. These remain concrete
blocking prerequisites, rather than evidence of a functioning codec.

Reviewed [proposal](tgrab-selected-checkpoint-codec-proposal-2026-10-16.md):
21,196 bytes, SHA-256
`bfa4548956c23ef6e3819ed015053ac68e939fa076b2ae1675cd582c61368f8f`.
Independent criteria were recorded before author handoff at
`runs/review/selected-checkpoint-codec-20261016/independent/acceptance-criteria.json`,
SHA-256 `45c4e307eca1bf356ff2bac41fa35a7a4dfc47710d32c57fce1fc36685a21e6d`.
This review inspected source and hashes only. It created/read no checkpoint
payload or dataset, imported no NumPy/Torch/SDK, and executed no numerical,
model, remote, provider or cluster work. No tests or parity runs were performed.

## Source-grounded checks

All eight author-recorded source hashes match the current files. In particular,
the actual GR remains
`89894f8bd3e8574b5b57c777353fe1aac3a1f123f792467a009b04b8a6f9bcdb`,
the sealed store remains
`498e60f7be048411c253e4b8cb63e4b907973f969c397bfdfacde949389f057e`,
and the reconstruction configuration remains
`22bef6bf781c5bfb5493dd5e03169ea0e658b2463ccaa9ba09189af454508f1f`.
The existing native `load_selected` still raises unconditionally. No accepted
source, native factory or launch gate was enabled by this proposal or review.

The actual `CheckpointReference` has epoch and three SHA fields; it supplies
neither a trusted path map nor a decoding implementation. The proposed external
trusted manifest SHA plus separate recurrent/head payload SHA mapping addresses
this gap explicitly. Internally supplied checksums and matching filenames alone
cannot confer trust. The schema binds complete state dictionaries, including
persistent buffers, rather than only named trainable parameters or one model
component. The source-supported head keys are correctly `seq.1`, `seq.3` and
`seq.5`; gate input matrices/biases are grounded in the pinned cell source.
Unknown ChebConv descendants are explicitly withheld pending source inspection.

The six-file raw binary32/little-endian/C-order proposal avoids object pickle,
archive members, dynamic construction tags and decompression. Fixed basenames,
closed primitive metadata, byte/count/rank/dimension bounds, contiguous offsets,
finite-value checks, stable single-link regular files and guarded reads define
a plausible bounded decoder contract. Per-tensor domain-separated semantic
hashes bind names, shapes, dtype/order and exact bytes separately from file SHA.
Negative zero and subnormals are preserved. This is a proposed encoding policy,
not a successfully exercised encoder, parser or adversarial-security boundary.

The selected-state reference includes both H and C with shape `(1,100,128)`,
canonical node order, hidden-then-cell flattening, `next_target_t=2816`,
`last_consumed_t=2815`, epoch and both component identities. It distinguishes
the fixed-weight validation terminal state from training carry generated under
changing weights. The inference loader restores neither terminal state as an
initial test state nor optimizer/RNG state; initialization/RNG hashes retain
provenance only. Resume remains a separate unfinished format.

The publication plan preserves immutable candidates and failed transactions,
writes/hashes/fsyncs prerequisites before publishing a manifest/reference, then
uses strict validation-only improvement and earliest ties for a final atomic
selected pointer. Immutable decision/finalization receipts must provide trusted
history; GR's mutable selection object cannot prove it. Failed persistence after
an in-memory selection change must terminate rather than silently roll back
and continue. Crash and concurrent-reader behavior still require implementation
fixtures and are not validated by this source review.

## Exact test-access chronology

The actual `_evaluate` first resets and consumes the entire supplied conditioning
prefix, then advances the target iterator. The sealed store delays test payload
I/O until that first advance. However, `_evaluate` currently discards its final
state and has no replay-parity hook. The proposal correctly requires a new
observer to capture validation s[2816] and a new gate to inspect the already
produced final-replay H/C before any test target request. Comparing those bytes
and chronology adds no cell/head calls; performing an extra replay would.

The actual `test_once` calls `load_selected`, then checks the budget, and only
after both return sets `test_opened=True`. Thus load or intervening budget
failure currently leaves that flag false. The proposal correctly labels charging
a terminal selected-load/evaluation attempt **before loading** as a required
future orchestrator repair. The lazy store cannot consume a callback it has not
received. A partial component load, load-parity failure, deadline failure or
pre-test replay mismatch must render that future attempt unusable, with no
scoring, test read or automatic retry. These are required integration guarantees,
not existing API behavior.

## Required next gates

1. Inspect pinned PyG2.6.1 ChebConv source/wheel and freeze the full trusted
   recurrent/head catalog: every key, shape, dtype, parameter/buffer classification,
   alias policy and serialization order. Do not infer an accepted native catalog
   from the tiny D12 calibration or accept keys declared by a payload itself.
2. Freeze every nested manifest, validation, provenance, initialization-record,
   selection-receipt and pointer schema, primitive tags, exact trust roots and
   transaction rules. The proposal names these requirements but does not supply
   all detailed closed schemas. No payload writer/reader is approved by merely
   accepting this design direction.
3. Implement and independently verify the object-free codec, durable publication,
   strict two-component loader and selection history; exercise malformed metadata,
   bit-preserving round trips, component swaps, partial writes and failed loads.
4. Implement/review observer ownership, terminal load-attempt charging and the
   one-shot parity gate on the actual final replay before lazy test advancement.
   Previously captured tuples and mutable/public Python objects remain cooperative
   boundaries, not revocable capabilities or same-user security.
5. Prepare a separately finite native parity fixture and reporting/admission
   envelope. Real N100 key/shape loading, both state-dictionary identities,
   eval/dropout behavior, existing-forward capture, native forward/loss-gradient
   parity and any later resume parity remain unexecuted. Counts, tolerances and
   resources require their own reviewed protocol; this review adds none.

The proposal introduces no competence cutoff, scientific tolerance, task change,
new model authorization or compute allowance. Actual primary conversion, complete
training/checkpoint integration, competence and representation-collapse findings
remain outside this acceptance. The independent receipt and source inspection
record are in `runs/review/selected-checkpoint-codec-20261016/independent/`.
