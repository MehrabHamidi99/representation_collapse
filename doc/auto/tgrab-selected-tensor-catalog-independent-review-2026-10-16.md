# Independent source review of selected-inference tensor catalog

2026-10-16. **Accepted as the exact expected source-derived catalog for this
candidate.** No blocking catalog finding remains. This does not observe a native
state dictionary, tensor values, aliases, a checkpoint or load parity, and does
not authorize model or compute execution.

Reviewed catalog: `configs/graph/tgrab-selected-tensor-catalog-v1.json`, SHA256
`2dcaff589cdfd48f2d5426fabab3dae8f229dde7b2bff27ab7ed7d1815db4268`.
Reviewed [source derivation](tgrab-selected-tensor-catalog-2026-10-16.md), SHA256
`4a4690f2df7214628f3c5f05c72dfce9804661318a214acd0700a5361a8ab171`.
This reviewer made no catalog, author-script or inspected-source changes.

## Independent criteria and provenance

Criteria were frozen before the complete author-catalog handoff in
`runs/review/selected-tensor-catalog-20261016/independent/acceptance-criteria.json`,
SHA256 `1137e95783438a37b227b6ad5281c0fe3a6eb12d787ecff402f059f77063fce2`.
An independently constructed expected table of all names/shapes/orders/offsets
was also recorded before that handoff in `independent/expected-catalog.json`,
SHA256 `3ea0df45d14208567707d7215fbcf8c5773beaeb4057f5a0668e418dbc07079f`.
The author generator was inspected as source, never executed by this reviewer.

Source inspection used the pinned public recurrent/head/node wrappers and
Temporal cell; actual installed PyG2.6.1/Torch2.0.0+cpu source packets; Inspector,
template loader, propagate/collect templates; and selected initializer source.
All **21 installed source files, 384,318 bytes**, match their local source packet
hashes and sizes. Root's acquisition receipts record actual distribution RECORD
matches and single-link installed files; this reviewer verified those receipt
hashes and the received source bytes, without a new remote query. All **15**
public-versus-installed source equalities were independently recomputed from
the local files and match exactly.

Original wheel bodies were not rehashed in this task. Previously verified wheel
manifest/lock lineage and the root-recorded installed RECORD observations are
distinct inputs; a local source hash is not a new proof of wheel installation.
The catalog correctly retains that distinction and excludes local Torch2.6
sources from the Torch2.0.0+cpu target.

## Names, shapes and classification

The L1/N100/input100/H128/K2 candidate is unchanged. Temporal constructs input,
forget, candidate-cell and output gates in that order. Each gate has its own
W of shape (100,128), b of shape (1,128), and a ChebConv acting on the hidden
features. The wrapper retains exactly one cell under `units.0`.

Each ChebConv constructs two biasless PyG Linear children with weights
(128,128), plus its own bias (128,). Therefore each gate contributes exactly
`units.0.conv_g.bias`, `units.0.conv_g.lins.0.weight` and
`units.0.conv_g.lins.1.weight`, for g=i/f/c/o. The eight child Linear bias slots
are registered as None and are omitted by the actual PyG `_save_to_state_dict`
override; they are not zero-length tensors. Positive known input channels select
the initialized branch, so the lazy-parameter/initialization-hook branch is
outside the candidate.

The selected resolver maps `add` to SumAggregation. SumAggregation inherits
Aggregation's constructor, whose deterministic flag is an ordinary boolean;
neither contributes parameters or persistent buffers. Inspector is an ordinary
class. MessagePassing creates metadata, empty hook mappings and None explanatory
attributes, plus the aggregation child. None are additional tensor keys.
Its default decomposition setting is unchanged. ChebConv does not override the
abstract edge-update method, so the edge-updater template branch is unselected.
The inspected propagate/collect templates and loader install functions and
NamedTuple argument structure, not new model tensor attributes.

The head has exactly:

- `seq.1.weight` (128,256) and `seq.1.bias` (128,).
- `seq.3.weight` (128,128) and `seq.3.bias` (128,).
- `seq.5.weight` (1,128) and `seq.5.bias` (1,).

ReLU and Dropout contribute no tensors. The temporary local ModuleList is not
retained as another head attribute, so it does not create a second persistent
namespace. No selected class adds an extra-state override or persistent buffer.
Backend identity features and complete H/C are external to these two model
state dictionaries and remain separate codec fields.

Initializer inspection closes a potential structural gap: selected PyG
glorot/zeros paths fill existing tensor data; selected Torch Linear initialization
uses in-place uniform fills through its inspected fan helpers. These paths do
not resize, replace or register model attributes. This observation does not
measure initialization values, numerical finiteness, RNG consumption or native
kernel behavior.

## Native, serialization and optimizer order

Pinned Torch `state_dict` first saves a module's own parameters/buffers and then
recurses over registered children. It does not follow the global chronology of
every allocation in the constructor. Thus all eight direct GCLSTM parameters
appear first in i/f/c/o W,b order, followed by convolution children in i/f/c/o
order. Each convolution's own bias appears before its `lins.0` and `lins.1`
weights even though the children were constructed first. The head's native
order is seq.1 weight/bias, seq.3 weight/bias, seq.5 weight/bias.

Serialization independently sorts component-local ASCII names: direct W/b gate
suffixes c/f/i/o, then convolution groups c/f/i/o; head bias before weight in
each layer. Every catalog entry's zero-based native/serialization index and
contiguous payload offset matches the independent pre-handoff table. The
optimizer sequence remains sorted recurrent names with backend prefix
`node_emb`, followed by sorted head names with `link_pred`; globally sorting both
components together would give a different, incorrect optimizer order.

Exact source-derived expected totals match independently:

- Recurrent: **20 parameter tensors, 183,296 values, 733,184 bytes**.
- Head: **6 parameter tensors, 49,537 values, 198,148 bytes**.
- Combined: **26 tensors, 232,833 values, 931,332 bytes**.

All shapes have rank one or two, positive integer dimensions at most 256 and
individual payloads at most 131,072 bytes. Offsets have no gap/overlap/padding.
Exact component lengths/counts are tighter than the codec's 8 MiB recurrent,
1 MiB head, 128-tensor and 1 MiB-per-tensor ceilings. The catalog does not use
those ceilings to admit extra keys. Hidden-plus-cell terminal bytes remain the
separate exact 102,400-byte field of the codec proposal.

Torch state-dictionary `_metadata` is version side metadata, not another tensor
key or an extra-state tensor. The catalog correctly requires trusted primitive
version handling rather than arbitrary object serialization; the future codec
must implement that compatibility decision explicitly.

## Runtime boundary and outcome

Fresh source declarations support a no-alias **expectation**, not an observed
storage proof. Runtime must verify exact class/source identity, all keys/shapes,
parameter/buffer classification, native traversal, sorted serialization, actual
CPU FP32 dense finite tensors, object identity and storage non-overlap. External
global registration or save/load hooks, lazy/custom inputs, pruning,
parametrization, wrappers or module replacement can change the mapping; their
absence is a required runtime precondition, not established by a static table.

The complete source expectation can therefore be frozen while actual
`native_state_dict_observed`, `native_load_verified`, `execution_authorized`
and `scientific_acceptance` correctly remain false. The stated source scope is
selected constructor/registration/structural initializer/state-dictionary/template
closure, not complete package import, arbitrary native execution or scientific
numerical closure.

The independent structural comparison performed **477 checks**, all passing,
including every entry field against the prior independent expectation, source
and receipt pins, public/installed equality, exact totals, component bounds,
None slots, alias policies and runtime flags. Its receipt is
`runs/review/selected-tensor-catalog-20261016/independent/structural-comparison.json`,
SHA256 `a4242277f355f2e94addf42566d11fae3b0bf14bea1832cee20e6166d4c103bd`.
These are stdlib AST/hash and exact integer metadata checks, not tensor-value
comparisons or native model results.

No Torch/NumPy/SDK import, model construction, state_dict call, dataset/checkpoint
read, remote operation or submission occurred in this review. Actual native
catalog observation, both-component safe persistence/load identity, complete-state
replay parity, trainer integration and competence remain future admitted gates.
No representation-collapse, Floquet, benchmark-reproduction or cross-modal claim
is accepted here.
