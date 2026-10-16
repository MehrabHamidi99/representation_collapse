# Independent review of checkpoint JSON primitives

2026-10-16. **Accepted within the narrow primitive-utility contract.** No
concrete defect was found in the reviewed source or adversarial cases. This
does not accept an artifact validator, full JSON Schema implementation, file
reader, checkpoint codec, native loader or runtime integration.

Exact reviewed hashes:

- `src/representation_collapse/checkpoint_json.py`:
  `3ee8d359249359b1e0f89e09491949117c3050690a31127486c39caea50dfe7a`.
- `tests/test_checkpoint_json.py`:
  `bca6eafee720b78bf0ba8fd8b9ccabde3064095cbd48c9d37bd42de4ff92fbe9`.

The reviewer inspected only this implementation/test pair, checked Python3.10
AST compatibility and its stdlib-only imports, and replayed the focused command:

```text
PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_checkpoint_json.py -v
Ran 20 tests in 0.006s
OK
```

Both exact hashes were unchanged before and after the run. No broader suite,
dependency installation, numerical/model/SDK work or cluster operation ran.
The test's registry snapshot is specification metadata, not checkpoint data.

## Accepted checks

The encoder admits only exact built-in primitive types, checks strings and
keys before serialization, limits nesting/keys/numeric values and bounds output
as it is emitted, including the terminal LF. Unsupported subclasses and cyclic
containers cannot bypass those limits. The decoder bounds byte input and scans
container depth before decoding, rejects duplicate keys after escape decoding,
limits numeric tokens before conversion and rejects nonfinite results. Exact
re-encoding rejects alternate whitespace, ordering, escapes, numeric spellings,
underflowed tokens, extra documents and missing/additional terminal newlines.
Depth counting correctly ignores escaped quotes and bracket characters in strings.

The focused cases include boundary depth eight/nine, object keys 64/65, ASCII
strings 256/257 bytes, 10-digit integer limits, 64-character numeric-token limits,
byte-cap boundaries, malformed JSON, decoded duplicate aliases, nonfinite
constants/overflow, negative floating zero and finite extreme/subnormal floats.
Strict integer-field validation excludes bools/floats, while the closed-record
helper checks the exact caller-provided key set. Primitive operations have no
file-I/O path. The separate schema-byte helper checks the accepted registry's
exact 849,118-byte snapshot and SHA, with its own bound distinct from the 256 KiB
metadata-instance ceiling.

## Explicit limits

Parsing intentionally permits a canonical primitive record with an unknown
artifact tag or a boolean in an otherwise integer-named field. Callers must
apply the appropriate closed artifact shape, `require_int`, field/list limits,
semantic relationships and authority checks. `require_keys` returns the original
dict and does not recursively validate it. Parsing does not automatically call
the separate schema-pin check or authenticate any record.

There is no JSON Schema engine/metaschema validation, calendar/path validation,
cross-file identity, stable-file I/O, tensor handling, transactional publication
or checkpoint loading here. Concurrent mutation and deliberately modified Python
state are not security guarantees. Python3.10 syntax inspection is not execution
in the pinned Linux3.10 cluster environment. Those integration and runtime gates
remain separate; no scientific result follows from these metadata fixtures.

The retained log, exact before/after hashes and immutable independent receipt
are under `runs/review/checkpoint-json-utility-independent-20261016/`.
