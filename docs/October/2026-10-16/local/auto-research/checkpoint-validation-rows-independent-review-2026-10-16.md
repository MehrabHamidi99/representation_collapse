# Independent review of S04 validation-row consistency

2026-10-16. **Accepted for serialized validation consistency only.** No concrete
defect was found against the frozen `$defs.validation` and S04 contract. This
acceptance does not establish that the serialized counts came from data or a
model, or confer validation-result authority.

Exact reviewed identities:

- `src/representation_collapse/checkpoint_validation.py`: SHA256
  `26af742bd2d96c0d4b8c84e757fb0255f6e85fd0aca710ca28324200be542fde`.
- `tests/test_checkpoint_validation.py`: SHA256
  `fb77fe6b09343479f6afd28c81877cca688b90f074c0945b82e410678b941592`.
- Registry: `db2baaa64f891533f5c2c36229dd250733f675a0a2045d0436fc93d3145f2acc`;
  human specification: `959287af66b18ae3796ef809b0e71071b576e912f2705920ef8db582c27084ee`.
- GR scalar formulas: `1e8212b9589709d1d4654d0ebfe98179a41ea5a307dbc2083fe84f06c1aead13`;
  accepted JSON primitives: `3ee8d359249359b1e0f89e09491949117c3050690a31127486c39caea50dfe7a`.

## Checks and result

The checker closes all six top-level fields and 15 row fields, fixes kind/split/
version and epoch1..200, and requires exactly 256 ordered built-in rows. Target
timestamps are 2560..2815, predecessors 2559..2814, and source/target phases use
their respective timestamps modulo64. Both boundary flags are exact booleans at
target phases0/32, including the first validation boundary. This matches the
primary two-distinct-pattern schema; it does not independently verify the graphs.

Count fields are exact built-in integers in0..9900 and sum to9900. Metrics admit
only finite built-in numbers in[0,1], excluding booleans and subclasses. Numeric
integer endpoints0/1 are permitted, consistently with the schema's number type.
Precision, recall, F1 and accuracy use the existing GR arithmetic order and
zero-division0 conventions. Mean F1 is exactly `math.fsum(row_f1)/256`, with no
tolerance or pooled-count substitution. Range/type checks precede float coercion,
so an enormous caller-owned integer is rejected without an OverflowError.

Only the focused command ran, with the final replay after the author repaired
the mean-comparison test witness:

```text
PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_checkpoint_validation.py -v
Ran 15 tests in 0.098s
OK
```

The originally assigned tests at SHA256
`2471ced124802ab8d14de6e982c87f985be95e26562e9a2af7f06038382a6df9`
also passed 15 tests in0.099s. Their initial log is retained. During closeout the
author changed only the mean witness to heterogeneous fictional counts and an
explicit left-to-right floating reduction, avoiding reliance on version-dependent
built-in `sum` behavior. Restoring only that witness in memory reproduces the
original test file's exact SHA. The checker did not change, and the revised
tests passed the final focused replay above. No production defect or tolerance
change was introduced.

These synthetic cases cover every pinned row descriptor, first/last timestamps,
missing/unknown fields, reordered/duplicated/missing rows, strict field/container
types, both boundary offsets, negative/oversized counts, nonfinite and enormous
numbers, one-floating-step formula errors, zero-division branches, exact mean
versus explicit-left-fold/pooled alternatives and no mutation/I/O. The GR comparison evaluates
only its pinned scalar expressions/helpers; it does not import GR or NumPy.
Reviewed source/test/dependency/schema/formula hashes were unchanged before and
after execution. Python3.10 AST inspection passed; this is not a target-cluster
runtime observation.

## Remaining gates

Input is an already-decoded mutable tree. The function neither parses canonical
bytes nor checks file sizes/hashes, schema-pin execution, cross-artifact identity,
admission or the provenance of counts. A self-consistent fictional perfect row
is not evidence of competence. Standard JSON Schema/metaschema validation,
stable-file readers, other semantic rules, codecs, transactions and native/runtime
integration remain separate gates. No primary data, tensors, model, SDK, remote
operation or dependency installation was used; no code/schema/test or git state
was modified by this review.

The retained focused log, exact hashes and immutable receipt are under
`runs/review/checkpoint-validation-rows-independent-20261016/`.
