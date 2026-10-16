# Primary T-GRAB conversion and sealed split protocol v1

Frozen prospectively on 2026-10-16 before any target CSV body parsing. This
implements the dataset node of the [trainer contract](tgrab-training-runner-implementation-contract-2026-10-15.md),
not a trainer, scientific result or execution allocation. Only the already
inspected 70-byte header informed this implementation. The primary body remains
on Mila scratch. Author verification uses small synthetic CSV fixtures only.

## Identity and candidate

Dataset: Hugging Face `Gilestel/T-GRAB`, revision
`603388f332c7756f90bc1702b26053c2bc834654`, path
`periodicity/periodicity_det_(2, 32)_100N_40tr_4vl_4ts_pER(0.01).csv`.
Raw file must have exactly 6,907,326 bytes and SHA-256
`5a19f9e7949668473bc61cb59b6215f25f093ea0ee2d45b5e143338ded5523e3`.
Frozen reconstruction config SHA-256 is
`22bef6bf781c5bfb5493dd5e03169ea0e658b2463ccaa9ba09189af454508f1f`.
The public generator/wrapper pins and historical limitations remain in the
trainer contract; this converter never generates data or repairs its targets.

The production candidate is N=100, node IDs 0..99, 3,072 timestamps, two distinct
nonempty offdiagonal undirected patterns, exact repeated 32 A / 32 B ordering,
40 training periods, four validation and four test periods. Half-open splits
are train [0,2560), validation [2560,2816), test [2816,3072). All timestamp rows
must have the exact split masks and false inductive-test masks. Timestamp
presence is checked explicitly; no missing timestamp is synthesized or dropped.
An explicitly requested `fixture_only=True` can use a smaller typed DatasetSpec;
its manifest always says `tgrab_fixture_converted_v1` and cannot satisfy primary
identity, including when fixture dimensions happen to equal the primary spec.

## Lexical and graph semantics

The header is exactly `src,dst,t,edge_feat,train_mask,val_mask,test_mask,test_inductive_mask`.
Input is ASCII, Python strict CSV parsing, exactly eight fields per record.
CSV quoting is parsed normally; lexical rules apply to the decoded field token.
All fields are at most 32 characters, at most 500,000 records. Numeric IDs and
timestamps use decimal digits with no leading zero except zero itself, and may
have a decimal point followed only by zeros (`0`, `12`, `12.00`). Signs,
whitespace, exponents, nonzero fractions and nonfinite strings are rejected.
No float conversion/truncation or precision rounding occurs. Boolean fields
accept exactly `True`, `False`, `1`, `0`. Unit weights accept exactly `1` or
`1.` followed by one or more zeros. Other edge weights, including nonfinite or
almost-one decimal values, fail before graph construction. This prospective
policy is frozen before inspecting actual target tokens; a target incompatibility
is a retained conversion failure, never a reason for silent parser broadening.

Every positive edge is in bounds and has its reverse. Every positive source
has at least one offdiagonal nonedge destination against the **full** positive
set, establishing feasibility of the frozen same-source sampler. Identical raw
edge rows deduplicate explicitly, with counts; conflicting unit-weight/mask
duplicates fail their field/split checks. Edges sort lexicographically by
(source,destination), a prospective serialization/RNG alignment decision, not
recovered historical order. Identity float32 node features and aligned float32
unit edge weights are implicit encodings recorded in the manifest, never
materialized numerically by conversion.

Only node-zero self-loops are identified source empty-step markers. Other
self-loops fail. A marker alongside offdiagonal edges fails because that is not
the source empty-graph convention. Marker-only timestamps are retained while
the marker is removed, and then rejected as an empty offdiagonal candidate.
Consequently a successful manifest has zero marker rows; a failure retains
its files, bounded partial counts, source-marker timestamps and the explicit
empty-after-marker-resolution reason without publishing a valid manifest. Duplicate and marker
counts are separate during parsing, and successful raw_rows equals unique
offdiagonal occurrences plus identical duplicates.

## Serialization, publication and bounds

`src/representation_collapse/graph_data.py` imports only the standard library.
The converter API is:

```python
convert_primary_csv(raw_path, output_dir, *, expected_raw_sha256,
                    expected_raw_bytes, config_path, converter_revision,
                    guard, spec=PRIMARY_SPEC, fixture_only=False)
```

`guard()` is mandatory, must raise on stop/physical violation, and belongs to the caller's physical supervisor, stop
control, reporting lifecycle and compute authority. The caller initializes
mandatory reporting **before** conversion/raw hashing. The converter checks
guards before read chunks, records, timestamp validation and publication. It
has no CLI, process launcher, W&B initialization or implicit model import.

The raw exact expected size is checked before reading and bounds every read;
the absolute raw ceiling is 16 MiB. A split is at most 64 MiB and a manifest at
most 2 MiB. Files are stable bounded regular files with one link; symlinks,
symlink path components, traversal components, hardlinks, FIFOs, changing inode,
size, mtime or ctime fail closed. File opens use O_NOFOLLOW/O_NONBLOCK and
before/after identity checks. This detects ordinary accidental changes; it is
not an OS sandbox against malicious same-UID filesystem races.

Output directory must be fresh, with an existing parent. There is no overwrite,
resume or repair. Three distinct exclusive files `train.json`, `validation.json`,
`test.json` contain canonical ASCII JSON with sorted object keys, compact
separators and a final newline. Snapshot nodes/timestamps are actual integers;
pairs are sorted unique integer lists. Files are flushed/fsynced. Only after
all successful validation/writes is `manifest.pending.json` atomically linked
without overwrite to `manifest.json`, then the pending name removed and
directory fsynced. Failure before publication leaves partials and, where the
filesystem permits, an exclusive bounded `failure.json` recording partial parse
counts and at most 3,072 marker timestamps. Existing output is
refused untouched; the caller retains the refused attempt journal.

The manifest pins origin, exact raw size/SHA, caller-supplied immutable converter
revision, actual converter source SHA, config SHA, fixed schema/spec/masks,
features/weights/orders, marker policy, duplicate counts, two full pattern
pair sets, every timestamp's semantic hash and each split's filename, interval,
size, file SHA and semantic SHA. Revision is a caller lineage assertion; actual
source SHA is independently computed, and the supervisor must compare both
against the admitted immutable release. File-byte SHA is **separate** from graph
semantics. A snapshot semantic SHA is SHA256 of domain
`tgrab-snapshot-semantics-v1\0` plus canonical JSON of timestamp,nodes,pairs,
node order, float32 identity feature encoding and float32 aligned unit weights.
A split semantic SHA hashes `tgrab-split-semantics-v1\0` plus its chronological
list of snapshot semantic hashes. Timestamps are part of semantics, so equal
pattern contents at two times have different snapshot hashes. Raw row order
changes raw byte identity while preserving canonical split bytes/semantics.

## Sealed store and lifecycle

```python
manifest = read_manifest(path, expected_sha256, guard)
store = open_train_validation(manifest, guard, snapshot_factory=None)
train = store.train()
validation = store.validation()
open_test = store.sealed_test_opener(finalized_selection)
targets = open_test()  # reserves/consumes attempt, no test payload I/O
first_target = next(targets)  # verifies and opens test payload now
```

Manifest verification requires canonical JSON, unique keys, finite values,
exact field sets, strict scalar types, fixed primary identity or explicit
fixture identity, complete semantic/pattern/count consistency and exact split
paths/intervals. VerifiedDatasetManifest is frozen and holds canonical bytes;
`metadata` returns a detached copy. Manifest and split files are rechecked on
split access. No test file is opened, statted or hashed by reading the manifest
or opening train/validation. Manifest pattern metadata itself describes test
input patterns; these public repeated patterns are not concealed secrets.

Train/validation return tuples. The default snapshot factory lazily imports
the actual frozen GR Snapshot class; module import and conversion remain
stdlib-only. An explicit factory supports dependency-free fixtures; production
integration must use the actual immutable Snapshot contract and pin that
factory/backend. The store never constructs NumPy/Torch features, models or
optimizers. Synthetic typed-integration fixtures import GR/NumPy only to check
actual Snapshot, CheckpointReference and ValidationOnlySelection interoperability.

Test authorization rejects duck types/subclasses and requires exact actual GR
ValidationOnlySelection and selected CheckpointReference, valid hash fields,
finite in-range validation score, finalized=True and coherent epoch fields.
One closure can be reserved per store. It captures the selected immutable
reference and current validation-score/epoch summary, and refuses changed
selection identity/finalization/summary both when called and immediately before
the lazy first test read.
Calling it terminally consumes the attempt **before** guard/read errors. It
returns a lazy one-shot generator; test bytes are not read until the first
target request, after the audited controller's selected checkpoint reload and
training+validation prefix replay. Initial read verifies the whole bounded test
split before yielding individual immutable snapshots. A failed read/generator
cannot be reopened, and store train()/validation() refuse after test attempt.

These are cooperative API guarantees, not security against deliberately
malicious Python or same-user file access. GR's public mutable selection cannot
prove that finalize() was historically called or fields were never edited.
Previously retained train/validation tuples cannot be revoked. The audited
trainer must enforce complete validation/selected checkpoint persistence,
reload parity, no training after test and provenance of selection history.
A future checkpoint/selection receipt is still required; this module does not
invent that capability or claim historical lifecycle proof.

## Acceptance and remaining work

Author suite: `PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_graph_data.py -v`.
Fixtures cover deterministic serialization, explicit duplicate counts, raw
reordering vs semantics, strict lexical policy, markers/empties/missing times,
two-block patterns, masks/reverse edges/node bounds/full negative feasibility,
failure preservation, path/link/bounds/tamper checks, JSON/schema integrity,
detached metadata, actual typed Snapshot/selection gates, lazy sealed access,
failed-open consumption and terminal train/validation access. These are
engineering fixtures, not numerical campaigns or primary-data verification.

Before target execution: independent review, immutable published source/config
binding, reporter integration, exact guarded physical allocation and independent
conversion reduction are required. No CSV target body was parsed, remote command
run, model/SDK called or job submitted by this implementation task. Actual
payload lexical/graph compatibility, N100 trainer/checkpoint parity, competence,
collapse and Floquet claims remain unverified.
