# T-GRAB upstream pin and graph recovery audit

Audit: 2026-10-12, 13:07 UTC. This is a bounded source/metadata inspection,
not graph training, dataset validation or recovery of the inherited figures.
The machine-readable pins are in
[`configs/graph/tgrab-upstream-pin.json`](../../../../../configs/graph/tgrab-upstream-pin.json).
The inherited snapshots are preserved. The
[graph reproduction protocol](../../../2026-10-26/local/auto-research/graph-reproduction.md) records this follow-up.

## Immutable references and evidence

- T-GRAB `main` resolves to **`31293102b065feef7817ff86211427d7594edfdb`**.
  Public `git ls-remote --heads` and the [commit API](https://api.github.com/repos/alirezadizaji/T-GRAB/commits/31293102b065feef7817ff86211427d7594edfdb)
  agree; the commit date is June 24, 2026. The recursive Git tree is not
  truncated. Twenty-three small source files, totaling 143,542 bytes, were
  fetched at that revision and checked against the tree's Git blob IDs. Their
  local SHA-256 values are recorded in the pin file. The ignored receipts live
  in `runs/graph/upstream-pin-20261012/`.
- Hugging Face metadata reports dataset revision
  **`603388f332c7756f90bc1702b26053c2bc834654`**. The pinned
  [file tree API](https://huggingface.co/api/datasets/Gilestel/T-GRAB/tree/603388f332c7756f90bc1702b26053c2bc834654?recursive=true)
  includes both deterministic `(2,1)` and `(2,2)` CSV entries. Metadata gives
  their sizes and LFS SHA-256 declarations. **No CSV payload was downloaded or
  locally checksum-verified.** The CSV listing does not establish equivalence
  to a legacy `data.npz` or a training checkpoint.
- T-GRAB's [license file](https://github.com/alirezadizaji/T-GRAB/blob/31293102b065feef7817ff86211427d7594edfdb/LICENSE)
  specifies Apache License 2.0. The dataset's pinned
  [card](https://huggingface.co/datasets/Gilestel/T-GRAB/blob/603388f332c7756f90bc1702b26053c2bc834654/README.md)
  declares `apache-2.0`; its tree has no separate license file. These are
  provider declarations.

No package installation, cluster operation, provider/model call, training,
weights download or repository clone occurred. Public ref/API requests and
small source/card fetches were the only external operations.

The pinned [preferred citation](https://github.com/alirezadizaji/T-GRAB/blob/31293102b065feef7817ff86211427d7594edfdb/citation.cff)
lists the 2026 ACM SIGKDD paper and DOI `10.1145/3770854.3785699`. This is
verified repository metadata; the publisher DOI page was inaccessible to the
browsing tool in this audit. The existing 2025 preprint citation should be
reconciled separately before submission; no bibliography file was changed here.

## Actual recurrent model and complete state

[`model/gclstm.py`](https://github.com/alirezadizaji/T-GRAB/blob/31293102b065feef7817ff86211427d7594edfdb/model/gclstm.py)
stacks `torch_geometric_temporal.nn.recurrent.GCLSTM` units. Each consumes node
features, ordered edge endpoints/weights and its previous hidden/cell state;
the next unit consumes the preceding unit's hidden output. The wrapper returns
stacked hidden states, stacked cell states and the last unit's hidden output.
The persistent recurrent state is consequently **all nodes, both hidden and
cell components, and every unit**. A frozen adapter should document its
flattening order; a suitable declared convention is component → layer → node
ID → feature. That convention has not yet been implemented or validated here.

[`requirements.txt`](https://github.com/alirezadizaji/T-GRAB/blob/31293102b065feef7817ff86211427d7594edfdb/requirements.txt)
pins Torch 2.0.0, NumPy 1.24.4, PyG 2.6.1 and PyG Temporal 0.54.0. The latter's
public tag `v0.54.0` resolves to
**`46abd39abec0cd8de23c0e489a617c320c3fc5e8`**; its pinned
[GC-LSTM cell](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/46abd39abec0cd8de23c0e489a617c320c3fc5e8/torch_geometric_temporal/nn/recurrent/gc_lstm.py)
uses Chebyshev graph convolutions of the hidden state in the gates, a cell
update and a hidden update. The default Laplacian normalization is symmetric.
Thus nodes can be dynamically coupled; isolated per-node Jacobians cannot
stand in for the full tangent map. Inspection of this tagged source does not
verify which package bytes produced the legacy checkpoint.

The [trainer defaults](https://github.com/alirezadizaji/T-GRAB/blob/31293102b065feef7817ff86211427d7594edfdb/train/DTDG/gclstm.py)
are one unit, width 100 and filter size 1, whereas the inspected
[periodicity launch wrapper](https://github.com/alirezadizaji/T-GRAB/blob/31293102b065feef7817ff86211427d7594edfdb/scripts/task/link_pred/periodicity/DTDG/_gclstm.sh)
sets filter size 2 and receives width/unit count as arguments. The larger
[dispatch script](https://github.com/alirezadizaji/T-GRAB/blob/31293102b065feef7817ff86211427d7594edfdb/scripts/task/link_pred/periodicity/slurm.sh)
contains a one-unit, width-128 branch, but currently enables DyGFormer only.
These are source configurations, **not recovered legacy GC-LSTM parameters**.
For that branch's `L=1,N=100,H=128`, the derived joint state dimension is
`2LNH=25,600`; width 100 would give 20,000. Matrix-free differentiation and a
tiny dense reference are required before interpreting full-state spectra.

## Node order, data format and periodic construction

The pinned [base loader](https://github.com/alirezadizaji/T-GRAB/blob/31293102b065feef7817ff86211427d7594edfdb/dataset/CTDG/torch_dataset/link_pred/node_feat_static.py)
loads `data.npz`, reads its explicit `num_nodes`, sorts edges by time and applies
stored split masks. It does not relabel node IDs. The
[DTDG loader](https://github.com/alirezadizaji/T-GRAB/blob/31293102b065feef7817ff86211427d7594edfdb/dataset/DTDG/torch_dataset/link_pred/node_feat_static.py)
builds dense adjacency/features by direct endpoint indexing and groups
chronological timestamps. A manifest must retain the exact node-ID mapping,
including isolated nodes; row count cannot be inferred from observed endpoints
alone. Its time indexing assumes regularly spaced snapshots and needs a fixture
for empty/missing times and short splits.

[`utils/node_feat.py`](https://github.com/alirezadizaji/T-GRAB/blob/31293102b065feef7817ff86211427d7594edfdb/utils/node_feat.py)
constructs one-hot features as an identity matrix in node-index order. The
[graph conversion](https://github.com/alirezadizaji/T-GRAB/blob/31293102b065feef7817ff86211427d7594edfdb/dataset/DTDG/graph_generation/graph_generator.py)
duplicates undirected edges in both directions and writes NPZ arrays and masks.
This format differs from the public CSV artifacts; a CSV-to-NPZ conversion is
not yet pinned or validated. Its behavior, edge ordering, duplicate handling,
node count and masks must be checked against the acquired payloads.

The actual [periodicity generator](https://github.com/alirezadizaji/T-GRAB/blob/31293102b065feef7817ff86211427d7594edfdb/dataset/DTDG/graph_generation/periodicity.py#L56-L60)
expands `(k,n)` by repeating each pattern ID `n` times. Therefore `(2,1)` is
`1,2` (AB), and `(2,2)` is `1,1,2,2` (AABB). Their period lengths are two and
four updates. In `fixed_er`, each pattern reseeds Python/NumPy with
`12345+pattern`, so snapshots repeat for a fixed pattern. In `sbm`, community
assignment is pattern dependent but graph edges remain stochastic. Empty
graphs receive a node-zero self-loop to retain their time step. Train/validation/
test masks use complete-period counts. This resolves the **upstream naming
semantics**; the legacy Floquet checkpoint's actual condition is still unknown.

The pin JSON contains exact URLs for both AB/AABB dataset files at the fixed
Hugging Face revision, their metadata-declared sizes/LFS digests, and an explicit
`payload_downloaded=false`/`locally_verified_sha256=null` status. Filenames and
generator logic cannot prove that those payload bytes follow the intended
construction. No graph generation was executed.

## Prediction chronology and evaluation caveats

The [GC-LSTM trainer adapter](https://github.com/alirezadizaji/T-GRAB/blob/31293102b065feef7817ff86211427d7594edfdb/train/DTDG/link_pred/gclstm.py)
initializes `[L,N,H]` hidden/cell tensors, detaches them at training-window
boundaries and advances them from the supplied graph. Its link readout uses the
last unit's node embeddings. The separate
[link head](https://github.com/alirezadizaji/T-GRAB/blob/31293102b065feef7817ff86211427d7594edfdb/model/link_predictor.py)
concatenates source/destination embeddings and uses a nonlinear MLP, dropout
and sigmoid. A replay must set evaluation mode and recover the head parameters,
not merely the recurrent weights.

The [DTDG link trainer](https://github.com/alirezadizaji/T-GRAB/blob/31293102b065feef7817ff86211427d7594edfdb/train/DTDG/link_pred/trainer.py)
trains by advancing the previous graph and scoring the current graph. Evaluation
normally scores cached `z` first, then consumes the current graph. However, if
`z` is absent, the [observed branch](https://github.com/alirezadizaji/T-GRAB/blob/31293102b065feef7817ff86211427d7594edfdb/train/DTDG/link_pred/trainer.py#L387-L405)
has this timeline: consume current target graph → score that graph → consume it
again. Standard in-training evaluation may already carry `z`; the branch does
not establish leakage on every evaluated checkpoint. Training resets hidden/cell
state each epoch; its first previous graph can
come from the last evaluation, while previous edge features come from the loader
batch. These paths need explicit chronology/alignment tests. The source uses
chronological singleton batches, Adam, BCE, equal-count sampled negative links
and window detachment; the inspected shell chooses window size one. Negative
collision/duplicate handling is not independently validated. Training-loss
patience and readout state transitions must be reconstructed, not assumed.

The recurrent **core** state is hidden/cell `2LNH`. The end-to-end prediction
pipeline additionally accesses cached `z`, the previous graph and split/start
chronology. An adapter must verify when `z` equals the last hidden output, rather
than presume that every pipeline entry point has that identity. Archive those
cache/driver fields with native prediction receipts.

The [periodicity evaluator](https://github.com/alirezadizaji/T-GRAB/blob/31293102b065feef7817ff86211427d7594edfdb/train/DTDG/link_pred/periodicity/trainer.py)
scores all ordered off-diagonal node pairs, thresholds probabilities at 0.5,
uses zero F1 for zero-division cases and averages metrics over snapshots. It
also [stops when cached **test** F1 equals one](https://github.com/alirezadizaji/T-GRAB/blob/31293102b065feef7817ff86211427d7594edfdb/train/DTDG/link_pred/periodicity/trainer.py#L74-L78). The
[base trainer](https://github.com/alirezadizaji/T-GRAB/blob/31293102b065feef7817ff86211427d7594edfdb/train/trainer.py)
evaluates test splits after validation improvements and saves weights then.
Therefore copying the upstream training driver unchanged would allow test
information into stopping. Preserve this behavior when documenting historical
reproduction; a new scientific experiment must separately freeze a validation-only
selection/stopping protocol and disclose the deviation. No selection behavior
has been empirically reproduced here.

The [checkpoint utility](https://github.com/alirezadizaji/T-GRAB/blob/31293102b065feef7817ff86211427d7594edfdb/utils/early_stopping.py)
saves model-component state dictionaries, rather than a complete optimizer/RNG/
recurrent-state run snapshot. Its decreasing-loss comparison treats sufficiently
small nondecreases as continuation; patience semantics require a fixture. The
shell's epoch ceiling is 100,000, patience 100 and default learning rate is
`1e-4`; these settings and compiled dependency hashes remain unverified for the
legacy run. A bounded future allocation is still required before training.

## Assets still required for legacy acceptance

The upstream source is available; the inherited empirical figures remain
unverified. Recover or explicitly mark absent:

1. The exact AB/AABB condition, raw dataset bytes/revision, NPZ/CSV conversion,
   pattern IDs, split masks, graph-generation seed and SHA-256 manifest.
2. The original recurrent **and** link-head checkpoint bytes and hashes,
   selected epoch/run ID, command, dimensions, filter order, features and seed
   roles; model state dictionaries alone do not recover training history.
3. Exact installed Python/Torch/PyG/PyG-Temporal/compiled extension versions and
   hashes, CUDA/cuDNN, device and numerical/determinism flags. The upstream
   requirements file is not an environment lock or binary receipt.
4. Initial/reset hidden/cell states, node ordering, readout-cache chronology,
   train/evaluation carry-over and graph/edge-feature alignment.
5. Full-state raw trajectories at aligned graph phases and per-step metrics,
   including burn-in, complete periods, seeds, driver graphs and retained-history
   controls. Save hidden, cell and joint state without node pooling.
6. The actual plotting/tangent analysis source, window definitions, Jacobian
   ordering, derivative validation, solver convergence, raw spectra and hashes
   that produced `floquet_analysis/figs/interpret/` and its six reported modes.
7. Immutable job IDs, resource/runtime/exit evidence and archived run manifests.

The next ready **local** step is a versioned complete-state graph adapter and
tiny chronology/derivative fixtures against these pinned sources. Payload
acquisition/conversion and asset recovery remain separate gates. Pinning source
does not release graph training or support a representation-collapse claim.
