# Native GC-LSTM engineering calibration v2

2026-10-15. **Frozen prospective preparation; native work not executed or
admitted.** This is a tiny CPU engineering assay of the exact pinned GC-LSTM,
not reconstruction training, task competence, orbit closure, Floquet evidence,
or harmful-collapse evidence. The original reconstruction configuration remains
disabled and unchanged. There is one seed, one architecture and fixed literal
fixtures; no search, training, optimizer, checkpoint, dataset or retry.

## V1 import failure and the isolated import scope

The coordinator reports that v1 Mila job 11256293 failed with 1:0 after 23 s with
`No module named torch_geometric.utils.to_dense_adj`. Temporal 0.54.0's umbrella
initializers import unrelated modules that do not match PyG 2.6.1's API. All
v1 constructor/native-cell/AD counters were zero; the SDK/reporting attempt is
separate from those numerical counts. No GC-LSTM mathematics was executed,
no raw calibration NPZ was produced, and no numerical acceptance followed.
The failed submission consumes the last original analytic CPU slot; original
remaining submissions are zero. A v2 rerun needs a fresh owner compute envelope.
Neither this code nor its disabled config authorizes a retry.

V1 engine/config/protocol remain immutable. The engine copy is
`src/representation_collapse/native_graph_calibration_v2.py`, protocol ID
`native_graph_calibration_v2`, config path
`configs/graph/native-graph-calibration-v2.json`, and this v2 protocol path is
`doc/auto/native-graph-calibration-v2.md`. All fixtures, architecture, tolerances,
precisions, ledger limits, thread/cgroup/deadline bounds and SDK 0.30.0 policy
remain exactly equal to v1. `_numerics` and runtime dependency/resource preflight
are unchanged. There are no added numerical API/constructor/forward entries.

The sole import-path deviation uses reviewed helper
`src/representation_collapse/pinned_graph_import.py`, SHA-256
`2f14b3683f67d33f4a5f6325b5b204aee3a086e1fad1966acb9810a302d04344`.
It bypasses full Temporal package initializers and loads the exact installed
`torch_geometric_temporal/nn/recurrent/gc_lstm.py` plus unchanged public
node/recurrent/head source files into fresh owned namespaces. It constructs
no model. This is **standalone exact-cell and wrapper import parity**, not
full Temporal package API compatibility, trainer compatibility, native math
success, trained competence or original benchmark/checkpoint reproduction.
No distribution version, cell mathematics, source wrapper or fixture is changed.
The actual ChebConv still comes from the pinned PyG distribution.

Runtime preflight discovers the installed cell path through distribution
metadata and verifies its source bytes. Inside the initialized mandatory offline
reporter, after Torch has loaded, v2 rechecks the helper SHA and compiles those
verified source bytes into a fresh private helper namespace without cached
bytecode. It passes the metadata-derived path, immutable cell/wrapper pins and
live deadline/STOP guard to keyword-only
`load_pinned_graph_modules(source_root, installed_cell_path, cell_sha256,
wrapper_sha256, guard)`. The helper refuses any existing Temporal or private
wrapper namespace; guarded source execution must preserve module/class/path/
method-file identities. Failures roll back only helper-owned modules.

Before any ledger constructor entry, v2 requires the typed bundle's
`provenance['source_release_hashes_match'] is True`, actual public
`recurrent.GCLSTM is cell_type`, returned wrapper/parameter/head class identity,
and their exact source locations/hashes. Fixture pins accepted by helper unit
spies cannot be used in production. Its JSON provenance records that Temporal
initializers were not executed, full package API was not verified, and native
math/source was not modified. V2 saves that provenance in worker status and
calibration summary, along with the helper SHA and exact v2 source/config/
protocol paths. An import failure leaves numerical counters zero and remains
an engineering failure rather than a finding about the graph recurrence.

Root admission must use `protocol_id=native_graph_calibration_v2`, exact new
supervisor/worker argv, module/config/protocol hashes, and include the helper
path/hash in `source_files`. Helper execution additionally checks its frozen
config pin; an admission without the helper source entry is refused before
source execution. No old v1 admission, source identity or completed test receipt
is relabeled as v2. Root owns job wrappers and exact fresh allocation admission;
this implementation task edits no launcher or scheduler configuration.

## Complete state, sources and forcing

Use the actual `torch_geometric_temporal` GC-LSTM v0.54.0 at commit
`46abd39abec0cd8de23c0e489a617c320c3fc5e8`, installed cell source SHA-256
`c3568d2c1ba8926ab43e11c840213f90ebe782ad158dd76809d1dc1c255797af`.
Use public T-GRAB commit `31293102b065feef7817ff86211427d7594edfdb`:
`model/gclstm.py` (`86069a32fced5f09c60cd795e4f0abd82674dafae250ef3673ac8c3650504609`),
`model/link_predictor.py` (`cca12cfc73b8660c92236f530504da08ba0b606ec13a1a681ffabcae43b7ca86`),
and `model/node_emb.py` (`4be4450730f33949e0354348b3bd8f69ba0434e4aa911abec0fc5f798e8c2d29`).
The local source fixtures are under
`runs/graph/upstream-pin-20261012/source/`; the cell reference text is
`runs/graph/upstream-pin-20261012/gclstm-v0.54.0.py`. A supervised cluster run
must stage these exact wrapper bytes or fail. No replacement cell or alternate
graph convolution is accepted. Installed versions, cell bytes and source paths
are checked before construction; the imported class is checked again.

Architecture: N=3, input=3, one layer, H=2, K=2, symmetric normalized ChebConv,
bias enabled, CPU. Source wrapper/head wiring uses default float32; full-state
AD/finite differences use exact conversion of those initialized modules to
float64. Full state is hidden then cell, each `[1,3,2]`,
flattened layer/node/feature in ascending canonical node IDs 0,1,2: **D=12**.
Seed 0 is set before wrapper/cell/head construction with source default
float32 dtype. This is a declared tiny untrained engineering setting, not
trained checkpoint parity. Initial FP32 parameter bytes are hashed only after
reporting initializes; no weights are emitted. No parameter tuning or
alternative seed follows a failure. The pinned model wrapper and base
NodeEmbeddingModel contain no feature `.float()` cast; the separate training
NodeFeatGenerator does. The caller supplies dtype-matched identity features
without modifying any source wrapper.

Features are the 3×3 identity. Literal graph A is the bidirectional path
0–1–2; graph B is the bidirectional edge 0–2. All input weights are one.
Initial state, normalized direction and cotangent are the fixed explicit
numbers in `configs/graph/native-graph-calibration-v2.json`. Differentiate
only complete h/c at fixed external graph, features and initialized parameters.
No hidden-state pooling or graph/node diagonal approximation is used.

## Predeclared checks and tolerances

1. Compare the exact cell (`wrapper.units[0]`) and actual public wrapper at
   identical complete state/input, including h, c and `z == h_last`. This is
   the same cell object, so it checks wrapper wiring/state ordering rather than
   two independently implemented cell equations. Compare source head forward
   in eval mode with sigmoid of its own source sequential output; this is
   structural consistency, not trained-head or training-dropout parity.
2. Form dense J0 at `(s0,A)` and J1 at `(s1=F_A(s0),B)` with non-vectorized
   `torch.autograd.functional.jacobian`. Compare each step's matrix-free JVP
   and VJP against dense products. Compare the direct two-step derivative,
   direct two-step JVP/VJP and composed step JVP/VJP with **J1 @ J0** and its
   transpose. No per-step state is advanced by the derivative API.
3. Central finite differences at exactly epsilon 1e−3, 1e−4, 1e−5 compare
   one-step and two-step directional derivatives. Record every residual;
   all three fixed epsilons must pass, without selecting a favorable epsilon
   after seeing results. No monotone error decrease is assumed at roundoff.
   A complex tangent control extends the fixed real two-step JVP by separately
   applying it to real and imaginary directions and recombining them. Check
   agreement with the dense complex extension and homogeneity under i. Native
   states remain real float64. Never cast a complex tangent to real or feed a
   complex state through the nonlinear GC-LSTM. Four added JVP API entries
   account for eight native cell forwards; no archive arrays are involved.
4. Relabel node coordinates by new-to-old permutation `[2,0,1]`: permute
   feature **rows** and h/c nodes, map edges by the inverse permutation, and
   check forward and Jacobian equivariance `J_perm = Q J0 Q.T`. Feature columns
   retain their original semantic values; replacing features with a fresh
   identity after row relabeling would be a changed input. Check ordered-pair
   head outputs under the same node relabeling.
5. An explicit analytic noncommuting, nonnormal control uses the two 2×2
   shears A=[[1,1],[0,1]] and B=[[1,0],[1,1]]. Chronological B@A is
   [[1,1],[1,2]], whereas A@B is [[2,1],[1,1]]. Direction [1,2] gives [3,5]
   versus [5,3]. A's eigenvalues are analytically both one and its spectral
   norm must exceed 1.2, exposing nonnormal transient amplification. This
   infrastructure control is independent of the actual GC-LSTM findings.
   Also retain native `||J1 J0−J0 J1||_F` and a fixed-direction wrong-order
   error, each required >1e−10 to demonstrate realized native order sensitivity.
   If this native condition fails, record the negative sensitivity finding;
   do not replace weights/graphs/directions or infer input-order equivalence.
6. Fresh zero-state chronology uses A,B,A: consume input0, score target1,
   consume input1, score target2, consume input2. Scores receive only current
   complete state and the six fixed ordered off-diagonal pair IDs. The head
   cannot see the target graph through this interface. Record all states,
   events, source/target timestamps and native hook counts.

Default-FP32 public wrapper/head wiring tolerance is elementwise
`atol=1e−6 + rtol=1e−5*abs(reference)`. FP64 forward/permutation tolerance is
`atol=1e−12 + rtol=1e−10*abs(reference)`. AD/dense comparisons use
`atol=1e−10 + rtol=1e−8*abs(reference)`. Each finite-difference comparison
uses `atol=5e−6 + rtol=5e−5*abs(reference)`. Store raw references/candidates,
absolute errors, threshold violations, finite flags and relative norm errors;
failed comparisons stay in the output. Overall engineering acceptance requires
all declared checks and exact finite successful-call counts. A failed assay
does not establish a scientific failure or harmful collapse.

## Finite work and supervision

The sole successful schedule has **50 actual native cell forward hooks**, five
source head forward hooks and six head sequential forward hooks (five inside
source head plus one explicit structural reference). Constructors: one wrapper,
one cell contained in it, one head. AD API calls: four dense Jacobians, nine
JVPs and five VJPs. Dense Jacobians are 12×12, with 12 output reverse sweeps
each; the API counters do not claim to count Torch's internal differentiation
kernels. There are 12 finite-difference function evaluations: six single-step
and six two-step evaluations, contributing 18 of the 50 cell forwards.
There is one analytic shear control. Count attempted API/constructor/forward
entries before work, refuse overrun, retain partial counts on failure, and
require exact counts on success. No eigensolver or parameter grid is present.

The supervised module has a stdlib-only import surface. Supervisor monotonic
time begins before worker spawn. One CPU affinity and numerical thread settings
are imposed; target memory is 2GiB. Linux kernel memory cgroup limit <=2GiB is
required and bound to the actual Slurm allocation. RSS diagnostics retain
physical-memory units. No RLIMIT_AS cap is used: virtual address-space limits
are not physical RSS limits and can prevent Torch/Go SDK mappings. The worker
arms its import-inclusive deadline before NumPy, Torch, source or SDK imports.
Numerical/work time is 590 seconds from the supervisor's start, reserving ten
seconds for reporting and cleanup; physical supervisor cap is 600 seconds.
The source-checked ownership helper is `code/ops/legacy_floquet_audit.py`, SHA
`8e1f388d4f5ce72e0bfd9a4b5a4b1f8c98f66f2512841d96875822bfd9bffe09`.
It registers Python-spawned SDK descendants immediately and checks PID/start
identities for cleanup. STOP, timeout, parent failure and success all trigger
owned descendant reconciliation. No detached worker/service may be left live.
Unsupported platform/cgroup/resource guarantees fail before native execution.

Mandatory reporting uses existing `ExperimentReporter`, source SHA
`443c9e89d2eba954ee60cc35e015fe61196fbec2f50331067540b4a8259e8184`,
and SDK **wandb0.30.0**, explicitly offline in project representation-collapse.
Source/config/environment text identities may be read before reporting; all
native constructors, fixture tensors, model/array payload hashing and numerical
work follow successful reporter initialization. Initialize the spawn journal
before SDK import. Keep the worker alarm active through reporter finish and
final receipt; the supervisor still enforces the full 600-second deadline.
Offline durability is not cloud-delivery evidence. A reporting failure stops
the calibration and cannot silently disable W&B.

## Admission, environment and outputs

No native/SDK campaign or package install is authorized by this packet.
Coordinator publication, independent implementation review, a hash-bound
Linux CPU environment manifest, mandatory reporter readiness and exact root
admission are required. Admission binds config/module/helper/reporter/wrapper
hashes, source commit, exact supervisor argv, unique output/STOP path, one
attempt, resource limits and finite counts, with training=false and
scientific_acceptance=false. The graph environment is separate from the old
vector environment; no root lock or original reconstruction config is changed.
Use a separately isolated CPython3.10/Linux-x86_64 environment for the legacy
binary dependency closure. Import these explicit source modules via PYTHONPATH;
do not install the main project package, whose Python requirement is >=3.11.
This environment deviation is declared before any numerical work.
Required exact distributions are enumerated in the config, including explicit
Torch2.0.0+cpu and PyG extension +pt20cpu build tags rather than conflating
wheel build identity with public-version notation. Temporal installed source
must match the exact cell pin. Packaging the official pure-Python Temporal
sdist, if separately authorized, grants neither a native call nor a job.

Outputs: supervisor receipt and worker log; immutable worker config/admission
identities; dependency/source/environment and resource diagnostics; progress
and exact count records; raw tiny Jacobians, derivative vectors, state/head
outputs and permutation indices in an NPZ with SHA-256; all comparison records
in calibration JSON; terminal success/failure/timeout state; durable offline
W&B directory and reporting receipt; owned spawn/cleanup reconciliation.
No model weights or checkpoints are emitted. New unique attempt paths are
mandatory; existing output refuses resume/retry. The native finite-time
two-step products are not called Floquet maps because no orbit/Jacobian
periodicity or closure is measured.

Prepared supervised argv form, after coordinator fills immutable absolute
paths and exact admission (this is a runnable parser, not authorization):

```text
PYTHONPATH=<published-source>/src <isolated-env>/bin/python -m representation_collapse.native_graph_calibration_v2 --root <published-source> --config <published-source>/configs/graph/native-graph-calibration-v2.json --source-root <pinned-upstream-source> --admission <root-admission.json> --output <new-scratch-attempt>
```

The engine/test evidence and source/config/protocol hashes will be frozen in
`runs/review/native-graph-calibration-v2-20261015/receipt.json` before handoff.
Only dependency-free engineering spies are allowed in the preparation task.

## Preparation checks and unresolved runtime gates

The focused dependency-free command is
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_native_graph_calibration_v2.py' -v`.
Its saved outcome and all final source/test/config/protocol identities belong
in `runs/review/native-graph-calibration-v2-20261015/receipt.json`.
The single focused invocation passed **23 tests in 0.087 seconds**, exit0.
Checks include unchanged v1 numerical/preflight ASTs and every numerical,
environment and resource config field; helper bytes/fresh namespace, required
admission pin, Torch/reporting ordering, returned release/class/source identity
and limited-scope provenance. Source ASTs parse with Python3.10 syntax mode.
These spies/AST checks do not import Torch, NumPy, PyG or W&B, construct a native
model, run its AD/finite differences or execute the supervisor. Python3.10 AST
parsing is syntax evidence; this preparation is not actual target CPython3.10
or extension execution. Old v1 tests/source/config/protocol/receipts stay immutable.

Operational readiness remains false until independent v2 review, publication,
actual target environment/source staging, offline reporter readiness and fresh
hash-bound root admission. The original reconstruction config is disabled and
unchanged. Data conversion and trained checkpoints remain outside this assay.
The isolated-import deviation repairs only a proposed source-loading path; its
actual native execution and full Temporal package compatibility are unverified.
