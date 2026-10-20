# T-GCN native adapter calibration, 2026-10-20

Status: prospective calibration contract. Freeze this document and the reviewed
implementation/source closure before the first native cluster execution.
Passing local stub/synthetic unit tests is not an accepted native calibration.
The coordinator must independently verify the actual job artifacts before
admitting a T-GCN fit. This protocol authorizes no empirical training, test-split
access, held-out analysis, model-seed sweep or representation-collapse claim.

## Question and model boundary

Does the hidden-only adapter implement the pinned public T-GRAB T-GCN wrapper
and link head faithfully on deterministic synthetic inputs, including graph
changes, edge weights, causal state chronology, gradients, one isolated
optimizer update, strict component loading and complete-state capture?

The [source-readiness audit](tgcn-reconstruction-readiness-2026-10-20.md) fixes
upstream commit `31293102b065feef7817ff86211427d7594edfdb`. Use the actual
`TGCN` cell from `torch-geometric-temporal==0.54.0`; **TGCN2 is not interchangeable**.
The public wrapper explicitly sets **`improved=False`, `cached=False`,
`add_self_loops=False`** for every layer. The last flag overrides the dependency
default. Each of the three GCN gates must retain those settings, normalization
and source-to-target flow. No normalized-adjacency cache may carry graph
history. Check both `_cached_edge_index` and `_cached_adj_t` where present.

The reconstruction architecture is one layer, 100 one-hot node features and
hidden width 128. Complete persistent state is **H[1,100,128]**, next-target
timestamp and canonical ordered node IDs 0–99. There is **no LSTM cell tensor**.
For the tiny calibration, explicitly substitute four nodes/input features and
hidden width three; do not label that fixture the production architecture.
The unchanged public head has dropout 0.2, three Linear maps, two ReLU hidden
stages and a sigmoid output. Its constructor's `num_layers=2` is not a count of
all Linear maps. The source-derived production counts are 137,472 recurrent
and 49,537 link-head parameters, 187,009 total; native enumeration must verify
them rather than treating the algebraic count as execution evidence.

## Frozen source and runtime identities

Pin and report the actual bytes used, not just import names:

- Public `model/tgcn.py`: SHA-256
  `028a85f0527833a7e62099e8bb826ac144df29738ee120795d205d96a8c6f78c`.
- Public `model/node_emb.py`: SHA-256
  `4be4450730f33949e0354348b3bd8f69ba0434e4aa911abec0fc5f798e8c2d29`.
- Public `model/link_predictor.py`: SHA-256
  `cca12cfc73b8660c92236f530504da08ba0b606ec13a1a681ffabcae43b7ca86`.
- Dependency `temporalgcn.py`: SHA-256
  `1f417ad10b1fb3ed6af24731efb0e80b614495a3e4df3fc16254e631c3c8819e`.
- Installed PyG 2.6.1 `torch_geometric/nn/conv/gcn_conv.py`: SHA-256
  `036b464769ad1d98501c9fc18ff3132b507890b2c5cb41b3e6f9e1f22d4f57f6`
  (10,415 bytes). Independently extracted from the locked public wheel only
  after verifying its full SHA-256
  `8faeb353f9655f7dbec44c5e0b44c721773bdfb279994da96b9b8b12fd30f427`.
  This matches the coordinator's read-only installed-file hash; no native
  import/execution was used for that independent source check.

The submitted admission must additionally pin the backend, calibration CLI,
launcher/supervisor, this protocol, source-derived tensor catalog/fixture
specifications, and actual installed GCNConv source identities. Retain the
public acquisition inventory and immutable source URLs. Import the pinned
wrapper in an isolated namespace; reject preloaded collisions or unexpected
classes. Authenticate the actual TGCN/GCNConv types and source paths used by the
constructed modules. Package-version agreement alone does not prove source
parity.

Use the existing locked CPU environment: CPython 3.10.19, Torch 2.0.0+cpu,
NumPy 1.24.4, PyG 2.6.1, temporal 0.54.0 and W&B 0.30.0, with the established
native wheel versions. Environment lock SHA-256 is
`ea2f58d2a12f5d8562b31d5fa7b5c646121784867c134afd4a5cf6a8108b8c06`;
wheel manifest SHA-256 is
`67e27963f2166f8a7e26dba301c9020d2bd7d4c012010fb859e6e43b15ce9842`.
Record effective versions, interpreter identity and dependency source hashes.
Do not install, rebuild, substitute or change precision within this job.

Require CPU float32 and one Torch intra-op/inter-op thread, with OMP, MKL,
OpenBLAS, NumExpr and vecLib thread variables set to one. Enable deterministic
algorithms without warn-only fallback, disable cuDNN benchmarking and retain
the deterministic policy. Seed Python, NumPy and Torch with 0 before model
construction; use the separately declared Torch seed 73 for paired dropout in
the synthetic optimizer fixture. Construct two independent tiny model sets
after resetting all three seeds to 0 each time; require identical component
bytes and post-construction Torch RNG bytes. Record initial component hashes
and post-construction RNG hash, then reset all three seeds to 0 again before
the main adapter/reference fixtures. Six model sets are constructed in total:
two repeated-initialization sets, adapter, independent reference, fresh reload
and production inventory. Record initialization/RNG observations and constructor order; no external provider or
credential lookup is needed.

## Deterministic fixtures and numerical predicates

All graphs and labels are generated in memory. No converted benchmark data,
train/validation/test arrays, native weights or previously measured state arrays
may be opened. Tiny node features are identity X=I4 in canonical node order.
All input edges are directed entries of bidirectional undirected fixtures,
without diagonal edges. Preserve and report the actual ordered edge/weight
pairs passed to both branches; canonicalizing edges must not detach weights
from their corresponding pairs.

Use A, the bidirectional cycle 0–1–2–3–0; B, the bidirectional star with center 0
and leaves 1,2,3; and C, A's topology with nonuniform weights: 0↔1 has weight
0.5, 1↔2 has 1.5, 2↔3 has 2.0 and 3↔0 has 0.75. A and B have unit weights.
Uniform scaling of all GCN weights can cancel through normalization and is
not a valid substitute for C. An empty-edge control, when supported by the
native input API, must retain the same four nodes and feature width.

The implementation must retain a fixture descriptor with graph topology,
ordered weights, timestamps, node IDs, feature identity, seeds and fixed loss
pair lists. The loss target D is the bidirectional matching with canonical
positive pairs `(0,1),(1,0),(2,3),(3,2)` and fixed source-preserving negatives
`(0,2),(1,3),(2,0),(3,1)`. B remains a graph-change control, not a loss target:
its center has no source-preserving negative. The loss/update fixture warms A
at timestamp 0 under no-grad, detaches the
nonzero carried H in both branches, then consumes B at timestamp 1 with
gradients enabled and scores D at timestamp 2 before D is consumed. Require
at least one nonzero reset-gate parameter gradient. For this training-only
fixture, set every head parameter to positive 0.1 in both branches before the
step, avoiding an accidental dead-ReLU head that would make recurrent gradient
parity vacuous. Earlier evaluation parity uses the original seed-0 head.
Record this deterministic override; it changes no architecture or empirical
model and is not a claim about random-initialized training performance.

The primary parity branch uses separately constructed pinned native wrapper and
head instances, with identical starting component tensors. Compare the adapter
against explicit direct native calls with the same graph order, hidden input,
module modes, loss pair order and dropout RNG input. Freshly seed or restore
only the recorded **synthetic fixture RNG** between branches; this is a paired
calibration control, not restoration of a trained model's saved RNG/state.

For these identical CPU FP32 operation paths, require exact scalar equality
and elementwise `max_abs_difference == 0` for forward state, head scores,
positive BCE plus equally sized negative BCE, all parameter gradients and all
post-update parameters. Use the same deterministic parameter order and Adam
settings: learning rate 1e-4, betas (0.9,0.999), epsilon 1e-8, weight decay zero,
amsgrad false. Check optimizer state equivalence where exposed. No tolerance
fallback or switch to FP64 may turn a failed primary parity gate into success.
This single isolated synthetic optimizer step is an infrastructure check,
not an empirical training run or a benchmark fit.

Byte-preservation gates use little-endian contiguous float32 bytes and preserve
signed zero: positive-zero reset, component capture/load and complete H capture
must agree exactly. Numerical elementwise equality and byte identity are
reported separately; neither is inferred from a small norm. All compared
values must be finite. A graph-effect gate requires a strictly nonzero maximum
absolute difference, not a performance or statistical significance threshold.
If the implementation adds an independent dense-equation oracle, its operation
order and a separate tolerance must be frozen before execution; it cannot
replace or relax the exact primary native parity gate.

## Required acceptance gates

1. **Source, layout and inventory.** Verify every pin before native use. Reject
   TGCN2, wrong layer count, wrong widths, incorrect GCN flags, populated caches,
   non-CPU/non-FP32 tensors, unexpected tensor names/shapes or aliased parameter
   storage. Enumerate production 100×128 recurrent/head parameters and compare
   the source-derived catalog with native state-dictionary order, names, shapes,
   byte counts and module-version metadata. Record the explicit tiny layout.
2. **Zero, carry and chronology.** Reset yields positive-zero H and next target
   zero, with no `cell` attribute or placeholder payload. Consume exactly one
   graph advances the timestamp once; the output embedding equals H's last
   layer. Input H is not mutated. Nonzero carry agrees with the direct native
   path. Detach preserves value/order/time and removes autograd history. Wrong
   timestamps/node order/layout stop before model mutation or scoring.
3. **Self-loop policy and dynamic edges.** Inspect all three gates for
   `add_self_loops=False`, `cached=False`, `improved=False`; verify caches remain
   empty before and after graph calls. From the same zero H, run A, then B, then
   A again with identical weights: A outputs must agree exactly and A/B outputs
   must differ. C versus A must differ, proving nonuniform edge weights reach
   the actual convolution. Compare all to the direct branch. With empty edges,
   changing X while holding H fixed must not introduce self-loop feature
   messages; record this control or explicitly leave it unverified if the
   adapter cannot express empty-edge snapshots.
4. **Forward/readout/loss/gradient/update parity.** Verify complete H, ordered
   off-diagonal pair scores, deterministic positive/negative loss and all
   gradients/post-Adam tensors by the exact predicates above. Head training mode
   activates dropout 0.2 with paired fixture RNG; evaluation disables dropout
   and gradients across all submodules. Contexts restore prior modes.
5. **Strict dual-component load.** Capture detached contiguous CPU FP32 recurrent
   and head state dictionaries without mutation. Load both components strictly
   into fresh native instances, prechecking both complete catalogs and storage
   before mutating either model; reject missing/extra keys, wrong shape/dtype,
   nonfinite tensors and aliasing. A deliberately malformed head must not
   partially load the recurrent component. Verify every post-load payload byte.
6. **Complete-state capture and replay.** Capture every H coordinate with
   timestamp, node order and shape. Mutating captured clones must not mutate
   live state or change scores. Replaying the same finite synthetic graph
   prefix from fresh positive zero after strict dual-component reload must
   reproduce the full terminal H exactly. This is synthetic prefix replay,
   not the 2,816-snapshot trained-model replay gate; no real benchmark prefix
   or held-out split is authorized here.
7. **Durable reporting and termination.** Enter explicit offline W&B before
   importing/construction of numerical/native models. Publish source/environment,
   fixture/catalog, completed gate results and raw comparison errors, resource
   observations and final acceptance inputs as compact metadata. A failure
   preserves its bounded exception/gate reason; comparisons not reached or
   recorded remain unknown rather than zero. A failed gate is terminal; preserve
   its failure record and never continue into a fit. Successful worker exit
   alone is insufficient without reporter finish and independently reconciled
   scheduler/supervisor/result evidence.

A gate omitted by implementation remains unverified. Report which predicates
ran and which are prospective; do not silently mark the full native adapter
accepted from a smaller subset. Unit tests with fake tensor/model objects are
local engineering evidence only. No passing result implies empirical competence,
historical-run parity, periodic H/Jacobians, Floquet multipliers or memory loss.

## One-job execution envelope and stopping rule

One fresh cluv submission on **Mila**, account **mila**, QOS **normal**, partition
**main-cpu**: **one node, one task, one CPU, 2 GiB RAM, zero GPUs, ten minutes**.
Use the actual checked cluv version and one immutable source checkout; record
project directory, revision, command, interpreter and source closure in the
admission. The root coordinator owns submission and independent acceptance.
No cluster action is delegated by this document.

The outer wrapper reserves cleanup within Slurm's 600 seconds: at most 540
seconds numerical work, an independent supervisor stop by 560 seconds, shell
ceiling 570 seconds. Use a unique run/attempt scratch directory under this
project and a run-specific STOP path. Stop on source/identity/reporting/parity
failure, STOP, deadline, signal, resource-envelope violation or nonfinite values.
No automatic retry, requeue, race, resume or second submission is permitted in
this envelope. Reconcile ambiguous scheduler outcomes before considering any
new action; preserve a spent attempt rather than replacing its evidence.

Outputs stay on project scratch and include exact actual source/runtime/job
identity, gate results and raw maximum errors, fixture/catalog hashes, compact
state/component byte descriptors, reporter receipt and final manifest. Raw
benchmark data, model weights and binary state arrays are not uploaded. The
independent audit may use compact output, source hashes and actual offline
protobuf history; a metadata audit does not itself independently execute the
native numerical operations. Keep remote W&B delivery unknown until verified.

After a verified native pass, the next separate protocol must define the
model-specific T-GCN checkpoint/catalog/complete-hidden schema, corrected
chronology, fit/selection/stopping contract and first seed. This calibration
neither reuses GC-LSTM H+C checkpoint semantics nor admits a training launch.
