# Independent confirmation of selected-checkpoint schema revision 002

2026-10-16. **Specification revision 002 is accepted within the narrow scope
already recorded in the [schema review addendum](tgrab-selected-checkpoint-schema-independent-review-2026-10-16.md).**
SCHEMA-001 is resolved. This confirms metadata/specification acceptance only;
it approves no parser, codec, transaction implementation, runtime, native loading
or experiment admission.

The exact checked artifacts are:

- Registry: 849,118 bytes, SHA256
  `db2baaa64f891533f5c2c36229dd250733f675a0a2045d0436fc93d3145f2acc`.
- [Mapping specification](../../../2026-10-26/local/auto-research/tgrab-selected-checkpoint-schema-v1.md): 19,930 bytes,
  SHA256 `959287af66b18ae3796ef809b0e71071b576e912f2705920ef8db582c27084ee`.
- Existing independent report including its revision-002 addendum: SHA256
  `3a5e2a03f37b5f154df66f7bd55379824fafa0039bb435cf56a27cdc5b1f9203`.
- Immutable author revision receipt: SHA256
  `6853f6f7ddb8c1a613c57c68bbd333d3e6ef6ab1bd32c1cee380c503e59687b3`.
- Immutable independent revision receipt: SHA256
  `be7d7f251d55f4216dbd4a031cc03124fde13174564df94eb2b84210f52b431b`.

All new receipt file-size/hash bindings, six source pins and both prior-receipt
links verify against actual retained files. The retained author 503-check result
and independent 54-check receipt bind this exact registry. Those earlier checks
were inspected, not rerun or counted as new execution here.

Independently replacing only
`/$defs/transaction_failure/properties/failed_stage/enum` with its original value
reproduces the preserved 849,021-byte revision-001 registry **byte for byte**, SHA256
`0177c710c0f3541be50c565126f083ef89dac27f97924c97f0d80154c0cc9857`.
The sole schema change adds `transaction_intent`, `selection_apply` and
`selection_verify` to the original seven failure stages. The new prose assigns
intent write/publication/fsync/verification and actual GR application/result
verification explicitly, with terminal failures before the history-head commit.
The original blocker and its receipts remain preserved.

Neither `jsonschema` nor `fastjsonschema` is available in the inspected project
interpreter. No standards-validator/metaschema validation was run or installed.
This confirmation uses stdlib JSON, source inspection and hashes; it does not
repeat the complete 21-definition audit. All mandatory semantic/file checks,
canonical parser, codec, initialization recording, crash consistency, runtime
ownership, native observation and load/replay parity remain separate work.
No checkpoint/dataset payload, model, SDK, provider, remote or cluster operation
occurred. The immutable confirmation receipt is under
`runs/review/selected-checkpoint-schema-revision002-confirmation-20261016/independent/`.
