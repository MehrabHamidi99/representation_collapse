# Independent review of selected-checkpoint schema v1

2026-10-16. **The specification review is complete, with one narrow blocker
before schema acceptance: transaction-intent publication failures have no exact
failure-stage tag. No codec, loader, transaction implementation, native execution
or test admission is approved.** The remainder of the reviewed design is
consistent with the accepted source-derived catalog and prior design direction.

Reviewed [registry](../../configs/graph/tgrab-selected-checkpoint-schema-v1.json):
849,021 bytes, SHA256
`0177c710c0f3541be50c565126f083ef89dac27f97924c97f0d80154c0cc9857`.
Reviewed [mapping](tgrab-selected-checkpoint-schema-v1.md): SHA256
`d855b56c4e15327ce2a62fe98cfd0fa4836b7c0f9467821b69f031615a76a09f`.
The author receipt and all six registry source pins match current bytes, including
the repaired GR source
`1e8212b9589709d1d4654d0ebfe98179a41ea5a307dbc2083fe84f06c1aead13`.
The prior proposal/review remains historical and correctly describes the earlier
load-before-charge gap; the new schema explicitly pins the repair.

## Blocking finding SCHEMA-001

JSON pointer: `#/$defs/transaction_failure/properties/failed_stage/enum`.
The allowed values are `payload`, `manifest`, `receipt`, `pointer_object`,
`selected_alias`, `head_replace`, and `head_fsync`.

The mapping separately requires an immutable transaction intent to be written
before the receipt. An exclusive intent write, fsync or publication can fail.
An otherwise well-formed failure record with `failed_stage: "transaction_intent"`
is rejected by this enum; none of its current values truthfully names that
operation. The mapping requires the **exact stage**, so silently calling this a
receipt failure would weaken its own audit contract. The required actual GR
consider/finalize application and result verification also need explicit stage
coverage or an unambiguous documented mapping.

The minimum next step is an author-owned, newly hash-pinned schema revision
adding the intent-publication stage, defining coverage for selection apply/verify,
and testing failure metadata for those operations. Preserve the old author
receipt and registry hash. Do not implement a codec or introduce a catch-all
error message to work around the closed enum. This is a specification
completeness defect, not evidence of a failure in a running checkpoint writer.

## Checks completed

The independent metadata audit passes **3,255 structural assertions**. It resolves
all local references, checks all 21 definitions and 13 unique root tags, requires
every object key and closes additional properties/items, verifies all 15 mandatory
semantic rules are present, and rehashes the author/source bindings. The registry
already declares the draft-2020-12 dialect; there is no missing-dialect finding.

Every tensor descriptor matches the separately frozen independent catalog:
20 recurrent plus six head tensors, 931,332 FP32 bytes, exact names/shapes,
lexicographic order and contiguous offsets. Full hidden/cell state is separately
102,400 bytes, two `(1,100,128)` arrays at offsets 0 and 51,200. All 256 validation
row descriptors fix timestamps, phases, transition flags and 9,900 candidate
pairs correctly. Candidate/config, environment lock/wheel and optimizer-order
bindings agree with their retained metadata. These are metadata comparisons,
not observations of tensor values or a native state dictionary.

The review also inspected external trust, S01–S15 cross-file obligations,
strict validation-only improvement/tie retention, the sole atomic history-head
commit point, retained-pointer behavior, uncertain post-head failures, bounded
paths/JSON/files and initialization references. Hash/schema-valid records remain
insufficient authority. Selection/finalization receipts explicitly do not
authorize testing; error/stop labels cannot establish scientific eligibility.
One-shot charging precedes selected load, complete-state replay parity precedes
lazy test advancement, and later failures cannot reopen the attempt.

The initialization envelope intentionally references separately trusted primitive
records; their recorder and internal state schemas remain future reviewed work.
The transaction protocol and same-user cooperation are a specification, not
tested crash consistency or an adversarial security boundary.

## Validator and parser limits

Neither `jsonschema` nor `fastjsonschema` is installed in the project interpreter
or the four other inspected local interpreters. **No full standards-validator or
metaschema-validation run is claimed.** Unavailability is not evidence that the
registry is syntactically invalid. The independent stdlib audit is a deliberately
bounded metadata checker, not a replacement production JSON Schema engine.

Actual V8/ECMAScript regex tests accept the intended SHA/path/name/revision/error
witnesses and reject all 25 trailing LF/CR/CRLF/U+2028/U+2029 mutations. The initial
regex concern was corrected: Python `re.search` end-anchor behavior differs from
these ECMA results. It is not a frozen-pattern blocker. Future Python consumers
must retain explicit full-match/parser checks and portability fixtures, alongside
strict integer-token, finite-number, canonical-byte, duplicate-key, ASCII,
calendar-UTC, size/depth/path and stable-file validation. Draft-2020-12 must be
bound explicitly; older-draft `prefixItems` behavior is insufficient.

Before production implementation acceptance, run an actual standards validator,
repair SCHEMA-001, implement and independently test all mandatory semantic/file
rules, then separately review codec, initialization recording, observers,
terminal load/parity and a finite native admission. No payload, dataset,
NumPy/Torch/SDK import, model, remote, provider or scheduler call occurred.
The independent checker, ECMA matrix and receipt are retained under
`runs/review/selected-checkpoint-schema-20261016/independent/`.

## Addendum — revision 002: SCHEMA-001 resolved

2026-10-16. **The exact repaired specification is accepted.** This supersedes
the blocking specification status above while preserving the original finding.
No codec, parser, native execution, transaction implementation or test admission
is approved by this acceptance.

The repaired registry is 849,118 bytes, SHA256
`db2baaa64f891533f5c2c36229dd250733f675a0a2045d0436fc93d3145f2acc`;
the mapping SHA256 is
`959287af66b18ae3796ef809b0e71071b576e912f2705920ef8db582c27084ee`.
Author revision receipt SHA256 is
`6853f6f7ddb8c1a613c57c68bbd333d3e6ef6ab1bd32c1cee380c503e59687b3`.
The original registry, mapping and author receipt remain preserved.

All **54 focused independent checks pass**. Reverting only
`/$defs/transaction_failure/properties/failed_stage/enum` reproduces the original
registry's complete bytes and SHA exactly; there are no unrelated schema changes.
The enum now contains exactly the original seven stages plus
`transaction_intent`, `selection_apply` and `selection_verify`. All ten intended
tags are admitted and the seven tested unknown/nonstring values remain excluded.
The author test-source/result bindings verify its 503-check metadata receipt.

The prose explicitly includes exclusive intent write/publication, file/directory
fsync and verification before the receipt; an unpublished intent may have a null
reference. GR selection application and result verification occur after durable
prerequisites and before the history-head commit. Failures, including after
in-memory mutation, remain terminal. The sole head anchor, uncertain post-head
replacement, preservation and no-retry requirements are unchanged.

This re-review is limited to SCHEMA-001 and its exact revision proof. It did not
repeat all 21 definitions or install/run a standards validator. The earlier
validator limitation, ECMA correction and all codec/native/semantic/runtime
acceptance gates remain. Distinct immutable receipt:
`runs/review/selected-checkpoint-schema-20261016/independent/receipt-revision002.json`,
SHA256 `be7d7f251d55f4216dbd4a031cc03124fde13174564df94eb2b84210f52b431b`.
