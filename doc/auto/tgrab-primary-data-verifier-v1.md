# Independent primary-conversion verifier v1

Frozen on 2026-10-16 **before verifier fixture execution or target CSV parsing**.
This is an independent stdlib-only oracle for the already frozen
[conversion protocol](tgrab-primary-data-conversion-v1.md), not a new parser
policy, scientific threshold, execution approval or trained-model result.

## Acceptance fixed before execution

The verifier must independently read/parse every raw CSV row, reconstruct all
timestamp edge sets and counts, validate the complete fixed candidate, and
rebuild the exact canonical split and manifest bytes from the published
serialization contract. Compare every actual split byte string and the entire
actual manifest against that independently reconstructed oracle. Separately
verify the caller's trusted raw/manifest SHA and size, converter SHA/revision
and fixed reconstruction-config SHA. Self-consistent artifact hashes without
agreement with the raw source are insufficient.

Production identity is the published primary CSV: 6,907,326 bytes, SHA256
`5a19f9e7949668473bc61cb59b6215f25f093ea0ee2d45b5e143338ded5523e3`,
Hugging Face `Gilestel/T-GRAB` revision
`603388f332c7756f90bc1702b26053c2bc834654` and path
`periodicity/periodicity_det_(2, 32)_100N_40tr_4vl_4ts_pER(0.01).csv`.
The reconstruction config must hash to
`22bef6bf781c5bfb5493dd5e03169ea0e658b2463ccaa9ba09189af454508f1f`.
Converter identity is explicitly supplied by the trusted caller, then compared
to the reconstructed manifest; it is not inferred from an artifact's claim.

Primary N100/3,072 timestamps/32 A then 32 B/40-4-4 periods and exact masks are
mandatory. Preserve all timestamps, canonical integer nodes, lexicographic
edges, unit weights, bidirectionality, distinct nonempty patterns and at least
one offdiagonal nonedge for each positive source. Identical duplicate raw edges
must be counted and removed; conflicting weight/mask rows must fail. Only the
source node-zero marker is identified, and marker-only/mixed-marker timestamps
remain ineligible rather than silently repaired. The verifier independently
implements lossless decimal integral tokens with optional zero fractional
suffix, exact unit tokens and True/False/1/0 booleans, with no float conversion.

Exact canonical artifact comparison binds all manifest fields: origin/raw,
converter/config, non-generation status, lexical/marker policies, implicit
float32 feature/weight encoding, node/edge order, counts, two patterns, every
timestamp semantic hash, split file size/SHA/interval and split semantic hashes.
Canonical JSON and both semantic-hash domains are implemented independently;
no producer helper is imported or called. No general untrusted artifact JSON
decoder is needed: expected canonical bytes are reconstructed from trusted
policy plus independently validated raw rows, and byte equality rejects unknown
keys, duplicate keys, type substitutions, noncanonical encoding and nonfinite
artifact constants. Exact comparison also verifies every converted snapshot.

## API and receipt

```python
verify_primary_conversion(raw_path, manifest_path, *, expected_raw_sha256,
    expected_raw_bytes, expected_manifest_sha256, expected_converter_sha256,
    expected_converter_revision, config_path, guard, fixture_only=False,
    spec=PRIMARY_SPEC) -> dict
```

The module is `src/representation_collapse/graph_data_verifier.py`.
`VerificationSpec` is its own frozen typed spec with nodes, block and
train/validation/test periods; defaults are the primary values. Production
requires those exact defaults and fixed raw identity. Explicit fixture mode
allows smaller specs/raw identities, requires a visibly fixture-kind converted
manifest, and returns a distinct fixture verification kind. A synthetic fixture
cannot become primary-eligible even when its dimensions equal the primary.

The primitive receipt has production
`kind=tgrab_primary_conversion_independent_verification_v1` or fixture
`kind=tgrab_fixture_conversion_independent_verification_v1`, `status=verified`,
raw SHA/bytes, manifest SHA, converter SHA/revision, config SHA, contemporaneous
verifier source-file SHA, spec, actual raw/count/comparison totals, all three
split hashes, `snapshot_count`, `split_count`, `primary_eligible`,
`independent_of_producer=true`, and `scientific_acceptance=false`. No invented
comparisons are credited; receipt counters describe completed checks. On any
failure raise `VerificationError` (or propagate a caller guard failure), with
no success receipt or output file written. The caller owns durable failure
recording and the hash of the resulting successful receipt.

## Bounds, stability and authority

Input/manifest/split/source reads require stable bounded single-link regular
files, no symlink at any path component, no traversal and nonblocking/no-follow
opens. Raw ceiling is 16 MiB/500,000 rows, each field at most 32 characters;
manifest/config/verifier source at most 2 MiB and each split at most 64 MiB.
Every read checks descriptor/path device, inode, size, timestamps and link/type
identity before/after. Final identity checks also reject ordinary changes to
previously read inputs during the whole verification operation. This is not
a security sandbox against deliberately malicious same-UID filesystem races.
Only the published basename `manifest.json` is eligible; split basenames are
fixed by this independent contract, never chosen from untrusted metadata.

`guard()` is mandatory and must raise on STOP/deadline/physical violations.
It is called before reads/chunks, each CSV row/timestamp, split work and final
checks. The outer supervisor owns physical memory/time and immutable executed
source provenance; this helper's source-file digest alone cannot prove which
cached Python bytecode was executed. Mandatory reporting must already be
initialized by any admitted caller before raw hashing or verification.

This API reads the sealed test payload as an independent data-verification
phase, **outside training/selection**. It does not grant a fitting process
early test access. No feature tensor, native model, generator, SDK or numerical
campaign is initialized; the function has no CLI, submit, resume or write path.

## Prospective fixture and remaining gates

Author fixtures must cover successful independently reconstructed semantic
agreement, explicit fixture identity, duplicate/raw-order handling, exact
lossless lexical and graph constraints, fully rehashed corruption that still
disagrees with raw semantics, complete manifest identity, fixed source/config
pins, bounded/link/stability checks and guard failures. Fixtures may use the
producer to create artifacts under test, but the verifier oracle must never
reuse producer code. The independent reviewer will add separate adversarial
fixtures before source acceptance.

Actual primary compatibility, independent target reduction, physical job and
reporter integration, and future training/checkpoint competence remain separate
gates. This protocol freezes acceptance without asserting they have run.
