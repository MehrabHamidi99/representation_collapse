# Independent review: isolated native graph CPU environment

2026-10-15. **Accepted for bounded preparation, with native readiness false.**
This is an independent read-only source/provenance/dependency review. It does
not certify a completed install, native imports, ABI compatibility, W&B runtime,
calibration results, training or scientific claims. No package installation,
compiler, native model, SDK/provider, network or cluster call was performed by
this reviewer. Root retains preparation and compute admission ownership.

## Frozen review scope

- Preparation script `code/ops/prepare_native_graph_environment.py`, SHA-256
  `98103ab0cc82ded41e1474ed40aedfff50558b7a1480660110c377c8d997b738`.
- Wheel manifest `environments/graph-native-cpu/wheel-manifest.json`, SHA-256
  `67e27963f2166f8a7e26dba301c9020d2bd7d4c012010fb859e6e43b15ce9842`.
- Hash lock `environments/graph-native-cpu/requirements.lock`, SHA-256
  `ea2f58d2a12f5d8562b31d5fa7b5c646121784867c134afd4a5cf6a8108b8c06`.
- Current preparation protocol `docs/October/2026-10-15/local/auto-research/native-graph-environment-2026-10-15.md`,
  SHA-256 `264d6a6a98e65d9a5e401bbe7c160549812aeca318491f2b0e27b2e007bb65a6`.

The author's initial handoff receipt is
`runs/review/native-graph-environment-20261015/handoff-receipt.json`, SHA-256
`e7026bcf1398585c4be03d4b4ee561f384010bdb12992b9ec0a6ef1e5dfb50bb`.
Its original protocol hash is preserved; the subsequent editorial provenance
amendment is bound separately by
`editorial-provenance-amendment-001/receipt.json`, SHA-256
`97f75e5ac809248611cb223ac720047b4e2887882570c4dced4cbbceb2e8582c`.
The amendment attributes Python3.10/pure packaging to coordinator engineering
within the existing staging scope, rather than inventing new owner approval.
Script, manifest and lock remain unchanged across that amendment.

## Independent dependency and artifact checks

An independent offline parser recomputed **60 artifacts**, **314764667 bytes**
and **90 active dependency constraints** from hash-bound METADATA. It evaluates
the CPython3.10.19/Linux-x86_64 environment and activates extras transitively,
including the requests extra of opentelemetry-exporter-http-transport. Every
selected wheel has a compatible CP310/abi3/pure-Python tag, satisfies its own
Requires-Python marker, and satisfies all active version edges. The hash-lock
lines exactly match the manifest, and no NVIDIA/triton payload is present.
Saved evidence: `runs/review/native-graph-environment-independent-20261015/metadata-closure-result.json`.
This check imported only metadata-parsing tools, not Torch/PyG/NumPy/W&B.

The eight native source versions remain literal: Torch2.0.0, PyG2.6.1,
Temporal0.54.0, NumPy1.24.4, cluster1.6.3, scatter2.1.2, sparse0.6.18 and
spline-conv1.2.2. Official CPU local build tags are retained in wheel/runtime
identities: Torch2.0.0+cpu and the four +pt20cpu extensions. W&B0.30.0 is a
declared reporting addition. The separate legacy-compatible CPython3.10
environment is an explicit source-import arrangement, not installation of the
main project's package with Python>=3.11 requirements. Target-specific W&B
transitives are resolved separately; the root project lock is preserved.

The reviewer independently rehashed the four received PyG extension wheels
and the derived Temporal wheel. Remaining 55 full wheel bodies, including
Torch's 195412802-byte wheel and W&B, were not locally rehashed because they
have not been acquired in this preparation evidence. Their official declared
hash/size and received metadata remain pins for future staging. A resolved
metadata closure does not substitute for full-body verification or import
compatibility. The script refuses a stage/install mismatch in actual bytes
or SHA-256 and performs all downloads before the numerical worker.

For Temporal, independently verified source archive SHA-256
`8e9f94751a6656f49d68c3dc32ae4067221ad886e66649897b859b7770a522d6`,
48092 bytes, 70 regular/directory entries and 332794 expanded bytes. All 52
source package files match the derived wheel byte-for-byte. ZIP CRC inspection,
purelib/py3-none-any tags and absence of compiled payloads pass. The cell is
the exact pinned SHA-256
`c3568d2c1ba8926ab43e11c840213f90ebe782ad158dd76809d1dc1c255797af`.
The 86746-byte wheel SHA-256 is
`fb5f0bc2c9ef9e7a23a82d53b07490bc8a99adb5e55120559fe5412a8ca70893`.
The small setup source imports setuptools and invokes setup/find_packages;
the author preserves build-tool identities, unchanged source and dependency
metadata. No build was repeated by this review. This is one documented pure
packaging derivation, not an official prepublished wheel and not a compiled
source-build exception. Evidence is `temporal-derivation-result.json`.

## Preparation guards and resolved findings

The default check mode is read-only. Mutation modes require a separately pinned
root admission binding script/manifest, mode, unique owned scratch attempt,
walltime, storage cap, uv and interpreter bytes. Target checks retain Linux
x86_64/glibc>=2.28, uv0.9.18 and an explicitly staged CPython3.10 executable
under project scratch. No managed interpreter download, main-project install,
global shim or root environment mutation occurs in this script.

Stage/install use only reviewed wheel artifacts. HTTPS redirects are checked
through the same official-host, no-userinfo/query/fragment/unsafe-port policy.
Local Temporal copying is exact-size/prehash checked through a no-follow file
descriptor, streamed under byte/deadline/STOP guards, checked for mutation,
then checked again at the target. Existing mutation parents are inspected with
lstat and must be owned nonsymlink directories. A new attempt path cannot be
reopened or resumed. ZIP members reject unsafe paths, links and nonregular
types. All compiled inputs remain prebuilt wheels; Cython is a pinned Temporal
dependency but no compiler is invoked.

Offline installation fixes `--no-config --offline`, `--no-index --find-links`,
`--no-build --no-deps --require-hashes`, explicit Python and copy linking,
followed by `uv pip check`. The canonical lock SHA is bound in the manifest.
Final verification reads distribution metadata and exact cell source text,
not native imports. Receipt labels remain installed-metadata verified with
native smoke pending; no successful process exit promotes scientific acceptance.

During review the author repaired three concrete scope/identity gaps:
unchecked scratch/environments parents, an unbounded local copy before
size/hash refusal, and insufficient redirect checks. A lifecycle follow-up
reserved five seconds before the external whole-attempt cap, polls STOP during
commands at most one second apart, reaps at most two seconds and preserves
the original failure across a process-exit/kill race. The protocol explicitly
requires a root-owned external deadline of 1800 seconds; this source-level
review does not certify an outer supervisor that has not run.

The initial graph-subtree storage census excluded the earlier project baseline.
That finding was repaired in the final frozen script. The census now covers
the **whole project scratch**, counts physical st_blocks*512 for files,
directories and symlinks, deduplicates hardlinked inodes, never follows links,
refuses errors and stops at 200000 entries. Reservation includes the complete
current baseline, all compressed wheel bytes, three expanded copies and 1GiB
overhead, with a 20GiB project cap. No other project's symlink target is scanned.
The prior script and distinct storage regression remain preserved.

Root's prior read-only full-project storage receipt records 6338379776 allocated
bytes at 2026-10-14 21:21:54 UTC:
`runs/ops/status-20261015T0115Y/storage-platform.json`, SHA-256
`d51dad5e58ad1e23a8c7d29175eac7d04fb04103078ad3117c518f271b3482b1`.
Root later staged Python3.10.19 (27.8MiB compressed, approximately 100MiB
expanded), under its separate toolchain receipt SHA-256
`a67a8eb1e90bfc52848103cd6467c1b4838c8f650c9abbb607da89aa29af88de`.
Those are supplied operational records, not fresh cluster measurements by this
reviewer. The repaired census includes both the old baseline and new toolchain;
expanded-wheel reservation is calculated from verified downloaded ZIP headers
before installation. Root's independent full-project post-stage measurement
remains required to reconcile actual storage; no final expanded-byte total is
invented here.

Saved author evidence shows nine initial stdlib guards passing, one separate
fake-process lifecycle regression passing and one whole-project storage
regression passing. The review inspected these recorded checks and their
source rather than repeating them. The initial fixture path refusal and prior
source snapshots are retained; no failed native experiment is inferred.

## Acceptance and remaining gates

Accept the final source, immutable wheel/metadata lock, single pure Temporal
derivation and bounded preparation design. Source-level review has no remaining
blocking correction at the hashes above. Exact admission, official full-wheel
acquisition/hash checks, actual scratch/resource reconciliation, offline install
and pip-check/installed metadata/cell verification are still future root work.
Native ABI/import behavior and SDK initialization remain unverified. The separate
calibration must use its reviewed supervision and initialize pinned offline
reporting before native constructors/calls. This environment review neither
admits that campaign nor authorizes another allocation or retry.

The independent receipt binds this report, the final source/lock/protocol,
author evidence and independent metadata/derivation checks. Scientific
acceptance remains false; original graph checkpoint/data/producer linkage and
trained reconstruction remain outside this accepted engineering scope.
