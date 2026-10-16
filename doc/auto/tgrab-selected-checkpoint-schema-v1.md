# Selected-inference checkpoint schema v1 — October 16

**Specification frozen for independent review; no codec or native execution is
implemented or authorized.** The machine-readable registry is
[tgrab-selected-checkpoint-schema-v1.json](../../configs/graph/tgrab-selected-checkpoint-schema-v1.json).
Its `$defs` close every object and require every listed key. Root `oneOf` accepts
exactly thirteen artifact tags; there are no arbitrary extension dictionaries,
object deserialization tags, optional hidden payloads or implicit defaults.
Unknown keys/tags/versions fail. The registry follows the accepted
[codec direction](tgrab-selected-checkpoint-codec-proposal-2026-10-16.md),
[independent review](tgrab-selected-checkpoint-codec-independent-review-2026-10-16.md)
and [complete source-derived tensor catalog](../../configs/graph/tgrab-selected-tensor-catalog-v1.json).
The registry's `x-source-pins` records exact inspected source hashes. Its own
SHA is supplied externally by root admission, avoiding a self-hash cycle.

## Encoding, bounds and external trust

Checkpoint metadata uses ASCII JSON with sorted object keys, compact comma/colon
separators and exactly one terminal newline. Reject duplicate keys, nonfinite
numbers, non-ASCII strings, bools used as integers and integer fields decoded as
floats. Maximum instance depth is eight containers, object size 64 keys and
string length 256 bytes. Integer tokens have at most 10 decimal digits and number
tokens at most 64 characters before conversion; these metadata bounds do not
specify the separately trusted RNG evidence's internal representation.
Parse real calendar-valid UTC timestamps, rather than
checking their character pattern alone. These bounds apply to metadata instances, not to the
schema document itself. Schema validation must be combined with the fifteen
mandatory `x-contract.mandatory_semantic_rules` S01–S15: JSON Schema alone cannot
establish file hashes, cross-file equality, chronological history or runtime
ownership. The stdlib author checker is a bounded fixture checker, not a
production implementation of JSON Schema or a codec.

All paths inside metadata are relative to root's externally trusted run
directory; the restricted path pattern forbids absolute paths, leading dots,
traversal, shell syntax and Unicode. Exact bundle basenames are constants.
Files must be bounded, single-link regular files, without symlink components;
check descriptor/path identity, length and timestamps before/after reads.
Verify externally trusted SHA before decoding payloads. A file's own checksum,
`owner` string, selected score or schema-valid fields never confer authority.
Root's independently verified admission map identifies the exact run directory,
schema/catalog/config/source/environment/data identities and finalized history
head SHA. No filename cache, matching seed or other run is a fallback.
Admission's manifest relative path must equal its bundle directory followed by
`manifest.json`; all pointer/receipt references must resolve to those same
trusted identities rather than choose another path.

The `trusted_admission` artifact is a **record shape**, not a permission grant.
It binds run/attempt IDs, source commit, external schema SHA, fixed catalog SHA,
actual CheckpointReference fields, bundle directory/manifest relative path,
finalized selection-head path/size/SHA and environment/data/config/trainer/codec
identities. Root must obtain trust through its existing approval/readiness
process. This schema neither supplies compute approval nor a model run budget.
`scientific_acceptance=false` and `automatic_resume=false` are constants.

## Six-file bundle

The directory contains exactly these files, with no extra member or directory:

- `recurrent.f32le`: exactly 733,184 bytes, twenty parameter tensors.
- `link-head.f32le`: exactly 198,148 bytes, six parameter tensors.
- `validation-terminal.f32le`: exactly 102,400 bytes, hidden then cell.
- `validation.json`: at most 262,144 bytes.
- `provenance.json`: at most 65,536 bytes.
- `manifest.json`: at most 262,144 bytes; total bundle cap 12 MiB remains.

Binary tensors are finite IEEE754 binary32, little endian and C order, with no
header, archive, pickle, compression, padding, gaps, overlap, trailing bytes or
lossy casting. Negative zero/subnormal bits are retained. Every descriptor has
exactly `name,dtype,shape,order,offset,nbytes,data_sha256,semantic_sha256`.
The registry expands all 26 exact catalog entries, ordered lexicographically
within each component, with constant names/shapes/offsets/byte counts. The catalog
SHA is fixed to `2dcaff589cdfd48f2d5426fabab3dae8f229dde7b2bff27ab7ed7d1815db4268`.
Source-derived no-buffer/alias/order expectations still require actual native
observation; no native state dictionary has been established by these schemas.

`manifest` has exactly the proposal's root keys: kind/schema_version,
artifact_role, epoch, components, terminal_state, validation, provenance,
identities, architecture and semantics. `artifact_role` is
`selected_inference_candidate`: durable validation is insufficient to prove
winning selection. Both component records have filename/bytes/SHA/tensors.
Terminal state additionally binds epoch, both weight-file hashes,
next_target_t 2816, last_consumed_t 2815 and
`fresh_zero_eval_replay_0_through_2815`. Its two descriptors are hidden offset 0
and cell offset 51,200, each shape `(1,100,128)`/51,200 bytes. Never substitute
training carry produced under changing weights, cached z, pooled nodes or hidden
alone. Test initializes H/C to zero; saved terminal H/C is a parity reference.

Architecture constants are L1/N100/input100/H128/K2, symmetric normalization,
bias enabled, FP32 CPU, no recurrent dropout, two hidden head layers/dropout 0.2,
canonical node IDs and hidden-then-cell/layer/node/feature order. Semantics fix
lexicographic positive pairs, identity features, aligned unit weights,
same-timestamp derived cache and score-before-target-consume validation.
GPU migration, mixed precision, alternate architecture or encoding requires a
new independently reviewed version.

The manifest binds raw primary SHA, converted-store manifest/file/semantic SHA
for all three splits, converter/protocol identity, reconstruction and executed
config, trainer/codec/schema/catalog, environment and admission/reporting lineage.
Copying the already trusted test split's declared hash reads no test labels.
Node/feature/weight **implicit encoding** digests are defined, not observed
numerical tensor hashes: SHA256(ASCII domain + zero byte + canonical descriptor).
The registry's `implicit_encoding_descriptors` specifies the exact domains and
values: node IDs 0..99; float32_le C-order identity shape 100×100; and float32_le
unit value string `1.0`, aligned with lexicographic positive pairs. Actual backend
feature/weight consistency remains a future native check.

Tensor semantic SHA is SHA256(`tgrab-tensor-v1` ASCII + zero byte + canonical
`{component,name,dtype,shape,order,nbytes}` + exact tensor bytes). Component labels
are recurrent/link_head/terminal_state. Component semantic SHA hashes canonical
ordered `[name,semantic_sha256]` pairs with `tgrab-component-v1` + zero byte.
File-byte SHA remains separate. No manifest contains its own final SHA.

## Validation and provenance

`validation` kind is `tgrab_complete_validation_rows_v1`, with exactly
kind/schema_version, split, epoch, rows, time_mean_f1. Its 256 row schemas freeze
timestamps 2560..2815, source t−1, phases modulo 64 and transition/change flags at
phases 0/32. This relies on the independently verified eligible primary store's
two distinct patterns; the schema does not establish raw-data compatibility.
Each row closes the actual GR metric fields: timestamps/phases, expected
transition/change flag, pairs, TP/FP/FN/TN, precision/recall/F1/accuracy.
Pairs=9,900; confusion counts are strict nonnegative integers summing to 9,900.
Recompute existing formulas with zero-division 0 and time mean
`math.fsum(row_f1)/256`; do not replace it with pooled F1. Exact same-runtime
scalar agreement is an integrity check, not a new scientific tolerance or
competence cutoff. No probabilities, labels or test metrics are serialized.

`provenance` kind `tgrab_selected_inference_provenance_v1` closes run/attempt/job
IDs, UTC time, source commit and fixed-role source hashes, executed/candidate/
reconstruction config identities, environment, initialization, counters,
runtime policy, optimizer parameter order, primitive module-version mapping,
validation chronology and admission/reporting identities. Runtime values are
required future observations; the fixture values are explicitly synthetic.
Known source/catalog/config/version constants are pinned; actual trainer/codec,
run/config/environment/data references must agree with external admission.

Environment fixes CPython 3.10.19/Linux-x86_64 and the already pinned exact native
CPU distribution versions, lock and wheel-manifest identities. Runtime policy
requires FP32 CPU/AMPfalse, deterministic algorithms without warn-only fallback,
one CPU thread, frozen dropout/mode/reset semantics and a complete module-mode
receipt SHA. The version side mapping contains exact component-local module
names with inherited Module version 1; it is primitive metadata used to reconstruct
trusted load metadata, not arbitrary OrderedDict attributes or extra tensor keys.
Actual native compatibility remains unverified.

Initialization fixes model seed 0, negative seed 1, seeding before construction,
construction order seed_all→recurrent→head→optimizer→negative_generator, and
initial component semantic digests. Three closed references have distinct
constant roles: seeding_before_models, post_model_construction_rng and
negative_generator_initialization. Each reference has exactly role,
encoding=`primitive_json`, safe relative_path, byte size 1..262,144 and SHA.
These are references to actual independently trusted retained initialization
evidence, not invented RNG states inferred from seeds. This node freezes the
reference envelope only; it neither defines RNG restoration nor implements the
separate initialization recorder. The inference loader verifies required
evidence identity and never restores it. Missing/untrusted/nonprimitive evidence
fails, and any future recorder's own state serialization requires separate review.

Candidate epoch equals its complete/validation epoch counters, both 1..200.
Completed and attempted updates both equal epoch×2,559 for a successful complete
candidate. A later failed attempt does not rewrite this immutable provenance.
Selected-load and test-payload attempts are zero when the candidate is written.
Optimizer order equals the catalog's sorted recurrent names then sorted head
names. Optimizer state is excluded and RNG restoration forbidden.

## Selection history and one atomic commit point

External metadata is outside the six-file bundle. Selection receipt/intent/
failure cap is 16 KiB each; pointer/head cap 4 KiB. At most 200 consecutive epoch
decisions plus one finalization are permitted. Each schema requires the shared
run/attempt/time/schema/run-admission identifiers. SHA references close role,
relative_path, size and SHA; references are verified before parsing their closed
tagged content. These metadata records are not reporter upload payloads.

`decision` records transaction_id, sequence, epoch, candidate reference/path/
validation SHA/F1, previous receipt/selected/best, selected-or-retained decision,
resulting selected/best/last_epoch and finalized=false. Sequence=epoch=last_epoch
runs 1..E with no gaps. First decision has null predecessors and selects even
F1=0. Later selections require strictly larger F1; ties preserve the earlier
reference. Check every predecessor/score/reference against the committed chain,
not mutable GR fields. The winning checkpoint's epoch may be earlier than E.

`finalization` occurs once at sequence E+1, binds the last decision, selected
reference/best score/last_epoch E and finalized=true. It records the actual stop
reason from the closed list loss_patience/complete_epoch_cap/deadline/owner_stop/
incomplete_epoch/error. This list records history, not scientific eligibility or
authority to evaluate. Both decision/finalization have
test_authorized_by_this_receipt=false. Whether a particular stopped/failed fit
may receive a later test admission remains the frozen training policy/root's
scientific decision; this schema must not turn an error into a successful fit
or resume a terminal attempt. No new stopping threshold is introduced.

`selected_pointer` binds stage selected/finalized, selected reference, manifest
relative path, the improvement/finalization receipt, best score and last_epoch.
Publish a new immutable pointer object and update `selected.json` only for
strict improvement or finalization. Retained decisions leave the pointer bytes
unchanged. Preserve every earlier immutable pointer object.

`selection_head` names sequence, current decision/finalization receipt,
selected-pointer object, transaction intent and finalized flag. Its mutable
`selection-head.json` is the **sole atomic commit anchor** and advances after
every committed decision, including ties/nonimprovements. This supplies complete
history without pretending a tie selected a new checkpoint.

For each transaction, prepare hashes without cycles: receipt→previous receipt;
pointer→receipt; intent→next receipt/pointer and prior head/pointer; head→intent,
receipt,pointer. Intent binds operation select/retain/finalize, whether alias
changes (false exactly for retain), prior hashes and next references. It contains
no next-head SHA. First prior head/pointer are null; later ones match the actual
committed head. Serialize under one trainer's ownership; automatic retry=false.

Durable publication order is: exclusively write/fsync/hash-verify the five
bundle dependencies; publish manifest then fsync directory; construct actual
CheckpointReference; write immutable intent; publish immutable receipt and, if
needed, immutable pointer object; replace/fsync selected alias only when needed;
finally atomically replace/fsync the history head. On retain, reuse prior pointer.
Apply the actual GR consider/finalize semantics only after its prerequisites
are durable, check the result against the prepared strict-improvement decision,
and treat any persistence failure after in-memory mutation as terminal. A
mutated GR selection alone never replaces a committed receipt/head.
An orphan receipt/intent/pointer is uncommitted evidence. A temporary alias/head
mismatch fails closed; no inference reader may adopt the new alias by filename.
Old immutable pointer objects preserve previous committed state. Readers require
the external finalized-head hash and matching head/receipt/pointer/alias tuple.

`transaction_failure` records transaction ID/sequence, optional intent/candidate,
prior committed head SHA, exact stage, commit-state tag and error class only.
The closed `failed_stage` tags cover every required transaction operation:
`payload`, `manifest`, `transaction_intent`, `selection_apply`,
`selection_verify`, `receipt`, `pointer_object`, `selected_alias`,
`head_replace`, and `head_fsync`. `transaction_intent` includes exclusive intent
publication, its file/directory fsync and verification before receipt publication;
its optional intent reference may be null when publication never completed.
`selection_apply` covers an exception applying actual GR consider/finalize after
its prerequisites are durable; `selection_verify` covers checking its result
against the prepared decision. Both happen before the history head commit and
remain terminal on failure, including a failure after in-memory selection mutation.
Other publication tags likewise include their operation's write/fsync/verification;
no catch-all or unknown stage is permitted.
Failure before head replacement is not_committed; failure during/after head
replacement/fsync is uncertain_after_head_replace until independently reconciled.
It is terminal with no retry/rollback-and-continue. Preserve partial directories,
all prior committed metadata and uncertainty. File existence or exit0 alone
cannot establish durable commit. Immutable records never embed their own SHA.

## Load, existing replay and terminal test-attempt events

`load_receipt` requires the selected reference, catalog/native-observation
identity, both post-load component semantic digests, strict match, all submodules
eval, gradients/dropout inactive and CPU FP32. It asserts no optimizer/RNG or
terminal-state-as-start restore and zero forward calls for loading. Hash/schema
checks must finish before decoding or model mutation. Load failure after partial
copy is terminal. Native observation, parameter/buffer/mode/alias/source checks
remain separately admitted engineering requirements.

`replay_receipt` observes the already-budgeted zero-state replay 0..2815, binds
the load receipt and saved terminal file, and requires full hidden/cell semantic
and byte equality with next target 2816/last consumed 2815 before any test payload
read. Its extra parity-check cell/head call counts are zero. Never execute an
extra replay, restore saved H/C as initial state, relax equality silently or
advance a target iterator to obtain this check.

`test_attempt` events distinguish consumed evaluation/load attempt from actual
lazy payload read. The now-repaired GR source is pinned in `x-source-pins`:
eligibility/initial-budget rejection does not charge; after eligibility it sets
test_attempted before selected load. Its inspected source SHA256 is
`1e8212b9589709d1d4654d0ebfe98179a41ea5a307dbc2083fe84f06c1aead13`.
All later failures forbid training,
validation and retesting. Its test_opened flag records callback invocation,
which may still have read no test bytes. Native loading is still blocked and
state-observer/parity integration remains unimplemented.

One attempt permits at most 8 immutable events: charged→both_components_loaded→
replay_verified→test_read_attempted→completed, or failed from any charged phase.
Event index/previous-event SHA form a consecutive chain; selected/finalized-head
identity remains fixed. attempt_consumed=true and retry_allowed=false always.
Load receipt is present after load, replay receipt before payload-attempt event,
and payload_read_attempted=false through replay verification. Error class is
nonnull exactly for failed; terminal-event flag is true exactly for completed/
failed. Charge before loading; record payload attempt immediately before the
first lazy iterator advance, after parity. A failure cannot reopen either gate.
Schemas or receipts confer no same-user security against deliberate mutation.

## Verification scope and remaining gates

The author generator/checker under
`runs/review/selected-checkpoint-schema-20261016/author/` uses only stdlib JSON,
hashing and synthetic metadata. Checks cover closed-key/tag/type handling, all 26
fixed offsets, epoch/state bounds, forbidden RNG/optimizer restoration, typed
initialization roles/path safety, cross-file metadata identities, existing metric
formulas on fictional counts, strict tie retention and pre-parity test-read
rejection. No tensor/data payload, source module import, model, SDK, native,
remote, provider or scheduler operation occurred. These fixtures do not establish
a production parser, file race handling, crash consistency, native loading or
scientific validity.

Independent schema review is required before implementation. Future codec and
transaction tests must enforce all S01–S15 semantic rules, bounded stable file
I/O and real crash boundaries; independently verify actual runtime/source/
catalog/mode/alias identities and complete-state parity under a separately
finite reported allocation. Resume format, trained competence and collapse
findings remain outside this specification.
