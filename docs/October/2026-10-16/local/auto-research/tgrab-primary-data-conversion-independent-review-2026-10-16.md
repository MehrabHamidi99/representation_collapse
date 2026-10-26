# Independent primary-data converter and sealed-store review

2026-10-16. **Engineering acceptance for the source below, with synthetic
fixtures only.** No primary CSV body was parsed, no original graph generator
was invoked, and no model, SDK, remote command or allocation was used by this
review. The actual dataset remains acquired but unconverted at this handoff.

## Reviewed identity and acceptance

- `src/representation_collapse/graph_data.py`: SHA-256
  `498e60f7be048411c253e4b8cb63e4b907973f969c397bfdfacde949389f057e`.
- Author tests, `tests/test_graph_data.py`: SHA-256
  `2bd25a7ea7edae23d948139e1a340735543b3ac951e0cb9f38c2947c9afb1ea3`.
- Independent tests, `tests/test_graph_data_independent.py`: SHA-256
  `709559510f93f3cc98d902987bc2c883f3518d56927fdd6ac28a795595c21944`.
- [Conversion protocol](../../../2026-10-26/local/auto-research/tgrab-primary-data-conversion-v1.md): SHA-256
  `003e6b01d3b090841f623f75ed0648a397fe8afdaeb49dc382fb69a3f0e05584`.

The controlling acceptance criteria were derived first from the
[training-runner contract](../../../2026-10-15/local/auto-research/tgrab-training-runner-implementation-contract-2026-10-15.md)
and [reconstruction specification](../../../2026-10-26/local/auto-research/tgrab-checkpoint-reconstruction-v1.md), plus
the immutable public generator/parser source. The supplied header-only receipt
confirmed eight column names, not their actual token/body compatibility.
The accepted existing `graph_reconstruction.py` remains unchanged at SHA-256
`89894f8bd3e8574b5b57c777353fe1aac3a1f123f792467a009b04b8a6f9bcdb`.

Final command:

```sh
PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p 'test_graph_data*.py' -v
```

It passed **49 tests in 0.463 seconds**: 18 converter-author fixtures,
19 independent fixtures, and 12 concurrently added job-preparation fixtures
matched by the wildcard. This acceptance concerns the **37 converter/store
fixtures** and reviewed converter source. The 12 preparation tests passed but
do not constitute an independent job/launcher review here. Inspection confirms
their subprocess, scheduler and supervisor cases used mocks; they made no
actual scheduler, SDK or worker call. An earlier run of the first 16 independent
fixtures passed in 0.314 seconds before the additional marker/failure-bound
coverage was added. The final source hash was rechecked after the final run.

## Findings resolved before acceptance

The first inspected implementation checked finalized selection when reserving
and calling the opener, then returned a lazy generator without another gate at
its first `next()`. A caller could invalidate the current public selection
between callback and actual test access. This was identified by source review;
no failing target execution was performed. The author centralized the actual
typed selection/reference validation and now rechecks captured reference,
last epoch, score type/value and finalization at reservation, callback and first
test access. Independent fixtures invalidate finalization, epoch, score and
selected reference before first access and confirm refusal **before any test
read**, with the attempt still consumed.

The marker-only failure originally gave a generic empty-positive error without
durable timestamp/count evidence. At the reviewer's request, failed conversion
now records bounded partial parsing counts, identified source-marker timestamps,
and an explicit timestamp in the marker-resolution error. The independent
fixture uses two identical node-zero markers at timestamp 5 and verifies both
marker rows, one deduplication and retained timestamp 5, with no published
manifest. No marker is reinterpreted as a task edge or silently repaired.

## What the evidence covers

The independent suite builds a deliberately sparse synthetic fixture with the
full **N=100, 3,072 timestamps, 32 A/32 B and 40/4/4** dimensions. It remains
`tgrab_fixture_converted_v1`, including at primary dimensions. A separately
written canonical JSON and domain-separated SHA implementation checks all
3,072 snapshot semantic hashes and all three split semantic hashes, as well as
the [0,2560), [2560,2816), [2816,3072) boundaries. This tests the fixed dimension
and serialization contracts; it does not reproduce the ER generator or verify
the primary pattern edge sets.

Token tests reject near-integer fractions, enormous IDs, exponent notation,
nonfinite weights, near-one decimal weights and float-like masks without
float conversion or integer truncation. Graph checks cover late pattern-order
changes, reverse-edge/full-mask negative feasibility, mask boundaries, identical
duplicates and source-marker policy. The author suite additionally verifies
missing timestamps, distinct/nonempty patterns and ordinary lexical/schema
variants. Identity features and aligned weights are **implicit float32 encoding
declarations**; these fixtures do not instantiate numerical feature tensors.

Manifest metadata returns a detached deep copy; altering nested returned
patterns or split names cannot change the verified byte snapshot. Split access
checks the manifest again. A payload changed to another valid graph with its
file-byte hash and length deliberately rebound is still rejected against the
unchanged semantic digest. File SHA and semantic SHA therefore enforce distinct
checks. Trusted expected manifest/source hashes remain the caller/supervisor's
responsibility; supplying a newly fabricated trusted hash is outside this gate.

File fixtures cover symlink ancestors, hardlinks, traversal, FIFO refusal with
nonblocking opens, the actual manifest ceiling before payload reads, and inode
replacement during a read even when replacement contents are identical. A
simulated validation-file write failure preserves the already written training
file and failure record without publishing a manifest; a subsequent attempt at
the same directory cannot overwrite it. Pending manifests are ineligible.
Power-loss behavior and malicious concurrent same-UID filesystem manipulation
were not tested and are not claimed.

Train/validation opening succeeds with the test file removed while independent
`lstat` and `os.open` spies forbid even probing its path. Actual test access
requires the exact existing GR selection/reference types with consistent current
finalized fields. The callback consumes the one-shot capability before errors,
then returns an iterator with no test-file I/O until first `next()`. Invalid
current selection, failed reads, guard failures and failed snapshot factories
cannot reopen the capability. Train/validation methods refuse after callback
use; no partial test target escapes a failing full-split decode.

## Scope and remaining gates

The gate is an audited cooperative API, not a Python or OS security boundary.
GR's mutable public selection cannot prove that `finalize()` was historically
executed, that complete validation actually ran, or that checkpoint persistence
and reload parity succeeded. Public pattern metadata makes repeated test labels
inferable. Previously retained tuples cannot be revoked, and deliberate private
state mutation or opening files independently remains outside the guarantee.
The future trusted trainer/codec must supply selection provenance, verify both
selected model components, replay the prefix before requesting the first test
target, and enforce terminality across the complete lifecycle.

Acceptance does **not** establish actual CSV lexical compatibility, successful
primary conversion, independent primary-data reduction, native N100 training,
checkpoint parity, competence, representation collapse, periodic Jacobians or
Floquet behavior. Those remain separate unfinished goal nodes. The physical
reporter/supervisor integration and a fresh allocation are also separate gates.
The review receipt is in ignored
`runs/review/graph-data-20261016/independent/receipt.json`.
