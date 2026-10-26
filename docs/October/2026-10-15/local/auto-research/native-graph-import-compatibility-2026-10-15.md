# Standalone pinned native graph imports — 2026-10-15

This protocol prepares an import compatibility deviation for a separately reviewed v2 engine. It does not modify the frozen v1 engine, protocol, configuration, installed environment, package versions, public source files or native mathematical operations. No new submission, retry or execution is authorized by this document.

The coordinator reports that Mila job 11256293 failed `1:0` in 23 seconds with `ModuleNotFoundError: No module named 'torch_geometric.utils.to_dense_adj'` during Temporal 0.54.0's umbrella import with pinned PyG 2.6.1. Reported constructor, cell-forward and AD counters were zero. This implementation task did not query the cluster or independently certify scheduler results. The failed job consumes its original slot; a renewed allocation remains a separate coordinator/owner decision.

## Compatibility scope

`pinned_graph_import.py` loads the installed `torch_geometric_temporal/nn/recurrent/gc_lstm.py` directly under its full qualified name, `torch_geometric_temporal.nn.recurrent.gc_lstm`. That source imports Torch, `torch_geometric.nn.ChebConv`, and PyG initialization functions. It does not need unrelated Temporal package initializers. The importer creates visibly marked minimal parent packages for `torch_geometric_temporal`, `.nn`, and `.nn.recurrent`, exporting only the exact verified `GCLSTM` class at the location used by the unchanged public wrapper.

These parent packages use an empty module search path, `loader=None`, a declared standalone-import spec origin, and `__pinned_standalone_graph__=True`. No Temporal `__init__.py` is executed. They do not expose the full Temporal API or imply compatibility of its unrelated graph convolutions, datasets, signals or benchmarks. This is a deliberate package-import deviation, not a recovered historical environment or full package parity result.

The public `model/node_emb.py`, `model/gclstm.py`, and `model/link_predictor.py` files are executed byte-for-byte under private package `_pinned_graph_standalone_model`. Their package initializer is bypassed as well. The public wrapper retains its original `from torch_geometric_temporal.nn.recurrent import GCLSTM` statement and binds the exact installed class. No class implementation, signature, constructor, forward method, ChebConv operation, gate, tensor operation or AD API is replaced.

The release pins remain:

- Installed native cell SHA-256: `c3568d2c1ba8926ab43e11c840213f90ebe782ad158dd76809d1dc1c255797af`; Temporal source commit reference `46abd39abec0cd8de23c0e489a617c320c3fc5e8`.
- Public `model/gclstm.py`: `86069a32fced5f09c60cd795e4f0abd82674dafae250ef3673ac8c3650504609`.
- Public `model/link_predictor.py`: `cca12cfc73b8660c92236f530504da08ba0b606ec13a1a681ffabcae43b7ca86`.
- Public `model/node_emb.py`: `4be4450730f33949e0354348b3bd8f69ba0434e4aa911abec0fc5f798e8c2d29`.

The existing isolated environment gate must remain unchanged: metadata verifies every native/reporting distribution version and CPU build tag, installed cell location/hash, environment artifact-manifest/requirements identities and physical execution limits before loading. Torch remains `2.0.0+cpu`, PyG 2.6.1, Temporal 0.54.0, NumPy 1.24.4, the four exact pt20cpu extensions, and mandatory W&B 0.30.0. This importer does not install packages, patch metadata, alter the wheel lock, substitute an approximate cell or certify binary ABI compatibility.

## Integration contract

Importing `representation_collapse.pinned_graph_import` uses only the standard library. The coordinator's new v2 worker must hash verify this helper before importing it, initialize its mandatory reporter, import the verified Torch distribution, then call:

```python
bundle = load_pinned_graph_modules(
    source_root=public_source_root,
    installed_cell_path=runtime_metadata["installed_cell_path"],
    cell_sha256=config["sources"]["cell_sha256"],
    wrapper_sha256=config["sources"]["wrapper_files"],
    guard=worker_deadline_and_STOP_guard,
)
if bundle.provenance["source_release_hashes_match"] is not True:
    raise CalibrationError("standalone import does not match literal release pins")
recurrent, head_module = bundle.recurrent, bundle.head
receipt["standalone_import_provenance"] = bundle.provenance_dict()
```

The production v2 configuration/admission must freeze the literal release pins above and require `source_release_hashes_match:true` **before any constructor or numerical call**. The generic hash checker accepts explicit alternate hashes solely to permit pure fake-file unit fixtures; those are not a production admission. Distribution and installed-cell metadata gates remain mandatory and independent of this provenance flag. A nonempty `sys.modules['torch']` is a runtime prerequisite, not proof of reporter initialization; the v2 caller owns and records the reporter-before-import invariant.

`PinnedGraphModules` is a frozen typed bundle containing the cell, recurrent-wrapper, head and node modules; `cell_type`, `wrapper_type`, `parameter_type`, and `head_type`; and immutable provenance mappings. `provenance_dict()` returns a plain JSON-serializable copy. The wrapper class is `MultiLayerGCLSTM`, parameter class `GCLSTMParam`, head class `LinkPredictor`, and actual cell class `GCLSTM`. No instance is created by loading.

The guard is required. It executes before and after bounded source reads, before and after each module execution, and before returning. Source files must be canonical regular files, at most 64 KiB each. The cell path must end in the installed distribution's exact four-component source path; the wrapper mapping must contain exactly the three declared relative files. Source bytes, SHA-256, device/inode/size/mtime/ctime identity, module paths/spec/loaders, class module/name/qualname/source file, declared forward-method code filenames, and exact wrapper cell identity are checked. All four source files are rechecked at the end.

The loader is a `SourceFileLoader` subclass using `spec_from_file_location` and `module_from_spec`. It compiles the already checked source bytes directly with the original source filename. It never reads or writes a bytecode cache, preventing a stale timestamp/size-matched `.pyc` from replacing verified source. It neither adds package search paths nor starts deferred package discovery.

Every preexisting canonical Temporal module/submodule or private wrapper namespace is refused before publication, including a namespace created by a prior successful call. Existing modules are preserved unchanged. The interface therefore requires a fresh worker process and cannot be used to conceal a failed earlier Temporal umbrella import. A second load in the same worker is refused; no implicit reload or retry exists.

On any exception, STOP or deadline, rollback removes only module entries and parent attributes that the helper itself created and that still have the same object identity. Foreign/concurrently replaced entries are preserved and cause failure. Unexpected foreign modules appearing inside the isolated prefixes cause refusal; they are not silently deleted. Other Torch/PyG imports are dependencies of the exact cell source and are not rewritten or removed by this helper. Module execution is synchronous; the caller's import-inclusive supervisor retains hard deadline and owned-descendant authority.

## Frozen implementation and fixture evidence

The helper is `src/representation_collapse/pinned_graph_import.py`, SHA-256 `2f14b3683f67d33f4a5f6325b5b204aee3a086e1fad1966acb9810a302d04344`. Final tests are `tests/test_pinned_graph_import.py`, SHA-256 `10e4a6294057cb83b7f1813557a0c06f34633720553bca47c2a0ab4fda2dd2b6`. Both parse with Python 3.10 syntax.

The focused command was:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover \
  -s tests -p 'test_pinned_graph_import.py' -v
```

Fifteen synthetic tests passed in 0.077 seconds. They use only temporary fake source files, a fake `torch` module, plain Python classes, and a stdlib import spy. Every fake package initializer deliberately raises, proving it is skipped. Tests cover exact class binding and JSON provenance, preexisting parent/descendant/private namespaces, repeat calls, changed hashes, extra/unsafe paths, absent Torch, import exception rollback, source mutation during execution, forged class/module/method provenance, foreign wrapper cell binding, injected foreign namespace preservation, STOP rollback, dependency-free helper import, and a poisoned timestamp-based bytecode cache.

The bytecode fixture initially verified a fake method's returned string. It was refined to inspect code constants without calling even that fake forward method; only the changed focused fixture was rerun and passed in 0.004 seconds. The actual helper source remained unchanged. No real Torch, PyG, Temporal, NumPy, SDK or installed native source was imported or executed by these tests. No model constructor, native forward/AD call, package installation, compiler build, provider or cluster call occurred. The suite was not broadened or rerun wholesale after this fixture-only refinement.

The v1 engine/config/protocol remain untouched. Actual standalone native imports, ABI readiness, comparison outcomes and scientific acceptance are still **unverified**. Independent helper review and an immutable v2 protocol/config/engine must precede any separately authorized allocation. Preserve the job 11256293 failure and its consumed budget; this source preparation grants no automatic retry and no training, checkpoint, dataset, graph-performance or collapse claim.
