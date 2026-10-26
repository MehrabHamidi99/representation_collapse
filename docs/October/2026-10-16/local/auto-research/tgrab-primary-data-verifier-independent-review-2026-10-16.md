# Independent engineering review of primary conversion verifier

2026-10-16. **Accepted as a prospective independent conversion oracle.**
There are no remaining blocking findings in the reviewed source. This accepts
the implementation and fixture evidence, not an actual primary-data reduction,
compute approval, task competence or scientific result.

The verifier was authored by a different agent from the producer. This reviewer
previously authored `graph_data.py`, so this is **not complete personnel
independence** from the producer. To address shared-assumption risk, acceptance
criteria were frozen before the verifier handoff, a positive artifact oracle
was independently assembled from the published serialization contract without
either module's helpers, and genuine producer outputs were adversarially
corrupted with their own hashes rebound. Root retains integration/scientific
acceptance authority.

## Reviewed identities and prior criteria

- `src/representation_collapse/graph_data_verifier.py`: SHA256
  `14c08e45ba8b0928792322a3cd4d7fff640aaa649e44e7e43dae6596ed5c486b`.
- `tests/test_graph_data_verifier.py`: SHA256
  `b9c45b3497b0ed458da211d9b15acaa2670f6eb401eb6507b5160de48abf91aa`.
- [Verifier protocol](../../../2026-10-26/local/auto-research/tgrab-primary-data-verifier-v1.md): SHA256
  `98433521eedf5dc86c30e54ab08d82b9bacb6fcdd3e37e94f1febfc099eff06c`.
- Producer `src/representation_collapse/graph_data.py` remains frozen at
  `498e60f7be048411c253e4b8cb63e4b907973f969c397bfdfacde949389f057e`.
- Reconstruction config remains frozen at
  `22bef6bf781c5bfb5493dd5e03169ea0e658b2463ccaa9ba09189af454508f1f`.

Review criteria were recorded before author API/source handoff in
`runs/review/graph-data-verifier-20261016/independent/acceptance-criteria.json`,
SHA256 `e8dd80a3a44e4514c40917890b7be2aff9df55066541847259d0e86006e603eb`.
They derive from the frozen conversion protocol, reconstruction config and
pinned public generator's reverse-edge, unit-weight, mask and marker behavior.
Neither the accepted producer nor author verifier/protocol was edited here.

## Independence and coverage

The verifier imports only stdlib modules and never imports or calls the
producer. Its own frozen `VerificationSpec`, bounded reader, textual integral
and boolean parsing, edge-set reconstruction and canonical semantic/JSON
encoding form the oracle. It works from independently parsed **every raw CSV
row**, not the producer's declared patterns, counts, semantic hashes or split
reader. The lexical implementation checks decimal strings directly and never
rounds through a float; it is separate from the producer's regular-expression
checks.

Raw validation covers exact header/ASCII/field and row bounds; integral tokens
with optional zero fractions; exact unit weights and masks; node/time bounds;
identical duplicate counts; complete observed timestamps; node-zero marker
resolution followed by empty-candidate refusal; reverse edges; full-positive
same-source negative feasibility; and two distinct nonempty patterns in the
registered repeated block order. No missing step or target is repaired.

The oracle rebuilds every chronological snapshot, both semantic-hash domains,
all three canonical split byte strings and the **entire** expected manifest.
Each actual split must equal the independently rebuilt bytes. The complete
actual manifest must equal the independently rebuilt canonical manifest, in
addition to matching its caller-supplied trusted SHA. This covers all source,
origin, config, policy, feature/weight/order, count, pattern, timestamp, split,
file-byte and semantic identities without trusting an artifact JSON decoder.
Unknown keys, duplicate keys, noncanonical encodings, bool/int substitutions and
nonfinite artifact constants cannot equal the expected canonical bytes.

Production requires the exact primary raw size/hash and N100/3072/32A32B/40-4-4
specification. Fixture mode requires its own explicit flag, manifest kind and
receipt kind, and cannot become primary-eligible. Expected converter SHA and
revision are explicit trusted-caller inputs: this helper checks them against
the reconstructed manifest but does not independently discover a release or
authorize an arbitrary caller-selected source. The admitted coordinator must
bind those identities to the actual immutable executed producer release.

## Independent fixtures and corruption tests

All **18 new independent fixtures pass in 0.073 seconds**:

```text
PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_graph_data_verifier_independent.py -v
```

The positive fixture is built directly from literal small graph snapshots,
manually specified protocol fields and independent stdlib serialization/hash
functions, with no producer or verifier helper reuse. A separate fixture invokes
the actual producer once, changes a converted validation graph, then recomputes
split byte hashes, snapshot semantic hashes, split semantic hashes and the
trusted manifest hash. Verification still rejects the result against the raw
CSV. A stronger all-pattern forgery changes all relevant payloads and manifest
pattern declarations, and is also rejected. These demonstrate raw-to-artifact
agreement beyond internal hash consistency.

Fully self-consistent raw/artifact fixtures with identical two patterns or a
saturated positive source row are refused, so a plausible canonical serialization
does not bypass the graph preconditions. Token fixtures accept lossless zero
fractions and the exact four boolean spellings; reject nonzero fractions too
small for reliable float rounding, near-unit weights, leading zeros, signs,
exponents, whitespace and nonfinite strings. Duplicate rows must agree with
independently recomputed counts. Missing/reversed edges, masks, node bounds,
markers, late pattern changes, altered provenance/types/features, snapshot
timestamps and malformed JSON are covered.

Path fixtures refuse symlinks, hardlinks, FIFOs, traversal and an incorrect raw
size bound. Guard failures before and during verification cannot return success
or write artifacts. A distinct OS-read/guard fixture mutates the original raw
file **after its initial read and descriptor checks**, while cached raw bytes
and all artifacts still agree. The final whole-operation stability check detects
that mutation and refuses success.

All **11 author fixtures were independently rerun and pass in 0.183 seconds**:

```text
PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_graph_data_verifier.py -v
```

Those include a full N100/3,072-timestamp synthetic fixture, visibly fixture-kind
and primary-ineligible, checking every timestamp and exact split boundary.
This synthetic dimension check is not a primary dataset run or native-model
calibration. Hashes of the final source/tests/protocol were rechecked after
execution and are pinned in the independent receipt.

## I/O, lifecycle and acceptance limits

The verifier is read-only and has no CLI, submit/resume path, model, feature
tensor, SDK lifecycle or implicit generation. Reads require bounded single-link
regular files with no symlink/traversal path component and O_NOFOLLOW/O_NONBLOCK.
Descriptor and path device/inode/mode/link/size/mtime/ctime are checked before
and after reads; all seven inputs (config, raw, manifest, three splits, verifier
source) are checked again at completion. Guards cover chunks, rows, timestamps,
splits and final checks. Only the fixed published manifest and split basenames
are accepted. These protect ordinary integrity mistakes and concurrent changes;
they do not establish an OS security boundary against malicious same-UID code.

The contemporaneous verifier source hash is receipt provenance, not proof of
which cached bytecode executed. Root's immutable source/admission mechanism and
physical supervisor must establish that separately. The caller must initialize
mandatory reporting before target hashing/verification and retain durable
success/failure receipts. No helper return value establishes actual scheduler
outcome, physical memory enforcement, process cleanup or dashboard delivery.

Independent data verification necessarily reads the sealed test payload outside
training/selection. It grants a fitting process no early test access. Actual
target conversion and this independent reduction must be integrated into a
reviewed, newly owner-approved job before target evidence is accepted. This
review used only synthetic fixtures, source/metadata inspection and hashing;
no primary target body, model/numerical campaign, SDK/provider, remote command
or cluster job was used. Trained benchmark competence, retention/collapse,
Floquet and cross-modal scientific claims remain unverified.
