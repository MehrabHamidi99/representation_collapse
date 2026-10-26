# Selected-checkpoint tensor catalog — October 16

This source-only catalog node follows the
[selected-inference codec proposal](tgrab-selected-checkpoint-codec-proposal-2026-10-16.md)
and its [independent review](tgrab-selected-checkpoint-codec-independent-review-2026-10-16.md).
It freezes a complete **expected source-derived catalog** for independent
review. It does not implement a codec, construct a model, inspect numerical
payloads or observe a native state dictionary. Source-catalog acceptance does
not admit production checkpoint work or establish native parity.

The target is the unchanged primary L=1/N=100/input=100/H=128/K=2 model, CPU FP32,
one pinned public MultiLayerGCLSTM and its distinct LinkPredictor. Component
names in the catalog are `recurrent` and `link_head`; actual backend keys are
`node_emb` and `link_pred`. Native component-local tensor names must retain
their exact spelling; these component labels do not rename model attributes.
Recurrent state hidden/cell and backend identity features are not model
state-dictionary entries and remain separately specified by the codec proposal.

## Inspected sources and provenance

Local read-only searches covered the repository's ignored `runs/`, local `.venv`,
environment metadata, local uv cache and accessible temporary cache files.
No pinned PyG 2.6.1 wheel or installed source was found. The local `.venv` has
Torch 2.6.0, not the target 2.0.0+cpu; those sources are explicitly excluded.
One inaccessible daemon temporary directory was not searched. No source body
was downloaded or retrieved remotely by this catalog worker.

The inspected immutable wrapper, node base, head and Temporal cell are recorded
with SHA256 and AST assignment evidence in the ignored author receipt directory
`runs/review/selected-tensor-catalog-20261016/author/`. They establish the direct
gate input matrices/biases, wrapper unit prefix and the exact head layers.
They establish direct gate declarations and head layout, rather than inherited
registration/state-dictionary behavior on their own.

Root subsequently supplied bounded source excerpts from its already verified
scratch environment:
`/network/scratch/m/mehrab.hamidi/representation-collapse/environments/graph-native-cpu-20261015-calibration01/venv/lib/python3.10/site-packages/`.
No remote access or source retrieval was performed by this worker.

The acquired PyG members include `torch_geometric/nn/conv/cheb_conv.py`,
`torch_geometric/nn/dense/linear.py`, `torch_geometric/nn/conv/message_passing.py`,
`torch_geometric/nn/resolver.py`, `torch_geometric/resolver.py`,
`torch_geometric/nn/aggr/__init__.py`, and the selected aggregation constructor
sources (`base.py`, `basic.py`; `multi.py` is acquired but not selected).
The acquired Torch 2.0 members are `torch/nn/modules/module.py`, `container.py`,
`linear.py`, `activation.py`, `dropout.py`, and `torch/nn/parameter.py`.
Supplemental Inspector, template loader, propagate/collect template and
initializer source pins are recorded in the final JSON. These close the selected
constructor registration/initialization and generated-method paths; they do
not establish the entire import or scientific numerical execution closure.

The frozen wheel manifest is
`environments/graph-native-cpu/wheel-manifest.json`, SHA256
`67e27963f2166f8a7e26dba301c9020d2bd7d4c012010fb859e6e43b15ce9842`;
requirements lock SHA256
`ea2f58d2a12f5d8562b31d5fa7b5c646121784867c134afd4a5cf6a8108b8c06`.
The exact PyG wheel is `torch_geometric-2.6.1-py3-none-any.whl`, 1,135,632 bytes,
SHA256 `8faeb353f9655f7dbec44c5e0b44c721773bdfb279994da96b9b8b12fd30f427`.
The target Torch wheel is `torch-2.0.0+cpu-cp310-cp310-linux_x86_64.whl`,
195,412,802 bytes, SHA256
`7c9e668462cd0813fc4baaeaa137efce2516e73cf381120846edf45d4ec21174`.
Installed-source receipt
`runs/review/selected-tensor-catalog-20261016/source-acquisition-attempt002/receipt.json`,
SHA256 `708142700461cf8dca622d0c55336aff96abb7c8b073a0ea1f06014489af72c9`,
binds 15 files/328,147 bytes to actual Torch 2.0.0+cpu/PyG 2.6.1 distribution RECORD
hashes and single-link regular files. Every source byte/hash and Python AST is
checked locally without import. Root's supplemental source receipts are also
independently checked and pinned in the JSON.

The earlier public packet receipt SHA256 is
`6d2871643fa92c660d7cc4cf7402bf9b9ee8c6bbbaedca27e11304fe29c0953b`:
its PyG source members came from the hash-verified exact frozen wheel, and its
six Torch sources from official release commit
`c263bd43e8e8502d4726643bc6fd046f0130ac0e`. All 15 public source hashes exactly
equal installed source hashes. This closes that release-versus-installed source
uncertainty for these files. Root did not rehash original installed wheel bodies
in this source pass; prior environment acceptance supplies their archive lineage.
Version strings or RECORD hashes alone do not prove original archive hashes.

## Exact tensor and persistent-state derivation

Pinned Temporal source creates gates in input/forget/cell/output order. Each
gate directly declares a fresh W `(100,128)` and b `(1,128)`. The public wrapper
registers exactly one cell under `units.0`; the node base adds no parameters.

PyG ChebConv defaults to `add` aggregation and creates K=2 biasless PyG Linear
children under `lins.0` and `lins.1`, each with weight `(128,128)`. It then declares
its own `(128,)` bias. PyG Linear's positive input dimension selects initialized
Parameter construction; its `_save_to_state_dict` override writes weight and
only non-None bias. Hence each gate adds `conv_g.bias`, `conv_g.lins.0.weight`
and `conv_g.lins.1.weight`. The JSON expands g=i/f/c/o literally. Its eight
`lins.*.bias` None slots are not tensors and must not be encoded as empty arrays.

The exact resolver maps `add` to SumAggregation. Its inherited Aggregation
constructor adds a deterministic boolean, with no parameter/buffer. Inspector
is an ordinary class. MessagePassing adds this aggregation child, metadata,
empty hooks and None explainability attributes; generated templates attach
functions rather than tensor state. Default decomposed_layers=1 adds nothing.
No selected class introduces `get_extra_state` or persistent-buffer entries.

The inspected PyG glorot/constant/zeros paths fill supplied tensor data in place;
the selected biasless Linear reset delegates to glorot without replacing its
parameter. The pinned Torch Linear reset invokes kaiming_uniform_/uniform_,
whose inspected paths fill the same tensor without resizing or registering a
new attribute. These are structural source observations only. No initialization
values, native kernels, RNG consumption or numerical equality were measured.
Both initializer source files and the collect template are hash-pinned alongside
the other installed-source members in the final JSON.

The head uses `seq.1.weight/bias` shapes `(128,256)/(128,)`,
`seq.3.weight/bias` `(128,128)/(128,)` and `seq.5.weight/bias` `(1,128)/(1,)`.
Torch Linear registers weight then bias. ReLU/Dropout add no tensors; the local
temporary ModuleList is not retained as a separate head attribute. Backend
identity features and H/C/cache remain outside model state dictionaries.

Exact source-derived totals are **20 recurrent parameter tensors, 183,296 values,
733,184 bytes**, and **6 head parameter tensors, 49,537 values, 198,148 bytes**.
The combined mapping has **26 tensors, 232,833 values, 931,332 bytes** at 4 bytes
per value. Expected persistent-buffer and extra-state key counts are zero.
These are expected source quantities, not observed native results.

## Catalog policy to retain

Each final entry must declare exact component-local name, integer shape,
`parameter` or `persistent_buffer` classification, native registration/traversal
index, lexicographic serialization index, dtype `float32_le`, C order and exact
element/byte counts. Expected names/shapes are trusted independently of payloads.
No wildcard namespace, extra/missing key, dtype cast or omitted buffer is
accepted. Check aliasing explicitly against the actual future model; no shared
parameter object or overlapping storage is permitted by this target catalog.

Native state-dictionary traversal order and serialization order are different
fields. The former follows pinned Torch module recursion/registration
and the latter by ASCII lexicographic sorting. The optimization contract's
sorted named recurrent parameters followed by sorted named head parameters is
recorded separately. Never substitute an observed payload's ordering for any of
these independently frozen orders. Native `state_dict` emits own parameters
before child modules: recurrent W_i,b_i,W_f,b_f,W_c,b_c,W_o,b_o precede the four
convolution children in i/f/c/o order. Each convolution's own bias precedes its
lins.0/1 weights even though the child modules were constructed first. Head
native order is seq.1 weight/bias, seq.3 weight/bias, seq.5 weight/bias.

Serialization instead sorts names lexicographically: direct W and b suffixes
in c/f/i/o order, then convolution groups in c/f/i/o order; head bias precedes
weight within each linear layer. Every native index, serialization index and
contiguous payload offset is explicit in the JSON. Torch state-dictionary
`_metadata` is primitive class-version side metadata, not another tensor key;
the codec must handle trusted version metadata explicitly, never pickle it.

Model tensors have rank 1 or 2, positive dimensions <=256 and individual size
<=131,072 bytes. The accepted codec proposal's wider ceilings remain recurrent
8 MiB, head 1 MiB, at most 128 model tensors, each <=1 MiB and bundle 12 MiB.
Exact catalog counts, shapes, byte lengths and contiguous offsets are tighter
constraints; the wider ceilings do not permit extra keys or padding.

Fresh Parameter/Linear declarations establish an expected no-alias construction;
actual parameter identity/storage remains unobserved. Runtime admission must
reject shared objects/overlapping storage and verify exact class/source identity,
full dictionary keys, parameter/buffer classification and FP32 CPU layout.
The expectation assumes no external global parameter/module/buffer registration
hooks, no state-dictionary save/load hooks, no extra state, lazy inputs, pruning,
parametrization, quantization or module replacement. Pinned Module supports such
runtime alterations; source inspection does not prove their absence later.
Recheck source bytes in the actual future worker. No tolerance or numerical
condition is inferred from this catalog.

Native observation remains a separate future engineering gate: construct only
under an explicit reported/count-bounded admission, compare actual full keys,
names/shapes/dtypes/parameter-buffer classification, alias identities, traversal
and serialization order to this source-derived catalog, then test both-component
load identity and complete-state parity. This source audit supplies no native
state-dictionary observation, load result, N100 parity, task competence or
representation-collapse evidence.
