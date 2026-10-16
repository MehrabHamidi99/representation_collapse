"""Synthetic, zero-update native checkpoint bootstrap parity protocol.

This module contains no model import and never accepts a graph/data path. A
caller supplies already-constructed pinned models and an observation callback;
the callback is responsible for native forward/loss/backward and must return
the captured values described by :func:`assert_parity`. The on-disk format is
plain JSON plus bounded little-endian FP32 payloads (never pickle/torch.load).
It is an engineering fixture, not a Figure 10 result or training admission.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import base64
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import stat
import sys
import tempfile
import types
import zipfile
from typing import Any, Callable, Mapping

import numpy as np

from . import graph_reconstruction as gr
from . import pinned_graph_import as pinned_import


CATALOG_SHA256 = "2dcaff589cdfd48f2d5426fabab3dae8f229dde7b2bff27ab7ed7d1815db4268"
CONFIG_SHA256 = "22bef6bf781c5bfb5493dd5e03169ea0e658b2463ccaa9ba09189af454508f1f"
CATALOG_PATH = Path("configs/graph/tgrab-selected-tensor-catalog-v1.json")
CONFIG_PATH = Path("configs/graph/tgrab-reconstruction-v1.json")
MAX_BUNDLE_BYTES = 12_582_912
STATE_SHAPE = (1, 100, 128)
COMPONENTS = ("node_emb", "link_pred")
SCHEMA = "tgrab-node2a-synthetic-bootstrap-v1"
LOCK_PATH = Path("environments/graph-native-cpu/requirements.lock")
WHEEL_MANIFEST_PATH = Path("environments/graph-native-cpu/wheel-manifest.json")
LOCK_SHA256 = "ea2f58d2a12f5d8562b31d5fa7b5c646121784867c134afd4a5cf6a8108b8c06"
WHEEL_MANIFEST_SHA256 = "67e27963f2166f8a7e26dba301c9020d2bd7d4c012010fb859e6e43b15ce9842"
GRAPH_RECONSTRUCTION_SHA256 = "1e8212b9589709d1d4654d0ebfe98179a41ea5a307dbc2083fe84f06c1aead13"
PINNED_GRAPH_IMPORT_SHA256 = "2f14b3683f67d33f4a5f6325b5b204aee3a086e1fad1966acb9810a302d04344"
MAX_MANIFEST_BYTES = 262_144
_RUNTIME_TOKEN = object()
_SIMULATE_EVIDENCE_FAILURE = object()


class BootstrapError(ValueError):
    """Fail-closed catalog, serialization, parity or transaction failure."""


@dataclass(frozen=True)
class Node2AExecutionResult:
    """In-process, runtime-authenticated node2a evidence; cannot be supplied as a mapping."""
    evidence_sha256: str
    transaction_manifest_sha256: str
    transaction_payload_sha256: str
    runtime_environment: Mapping[str, Any]
    environment_identity_sha256: str
    source_identity: Mapping[str, Any]
    _token: Any


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pinned_json(path: Path, digest: str) -> tuple[dict, bytes]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise BootstrapError(f"pinned regular file required: {path}")
    raw = _read_regular_bounded(path, MAX_MANIFEST_BYTES)
    if _sha(raw) != digest:
        raise BootstrapError(f"release input hash mismatch: {path}")
    try:
        value = json.loads(raw)
    except Exception as exc:
        raise BootstrapError(f"invalid pinned JSON: {path}") from exc
    if type(value) is not dict:
        raise BootstrapError(f"pinned JSON object required: {path}")
    return value, raw


def _read_regular_bounded(path: Path, limit: int) -> bytes:
    path = Path(path)
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
            raise BootstrapError(f"bounded single-link regular file required: {path}")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except OSError as exc:
        raise BootstrapError(f"cannot open bounded regular file: {path}") from exc
    try:
        opened = os.fstat(fd)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                or opened.st_size > limit or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)):
            raise BootstrapError("bounded input identity changed before read")
        chunks, size = [], 0
        while True:
            chunk = os.read(fd, min(65536, limit + 1 - size))
            if not chunk:
                break
            chunks.append(chunk); size += len(chunk)
            if size > limit:
                raise BootstrapError("bounded input exceeded size limit")
        after = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or size != opened.st_size:
            raise BootstrapError("input changed during bounded read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _import_runtime_module(module_name: str):
    """Import required native modules but categorically refuse Temporal init."""
    if module_name == pinned_import.TEMPORAL or module_name.startswith(pinned_import.TEMPORAL + "."):
        raise BootstrapError("Temporal package imports are forbidden; use pinned_graph_import")
    return importlib.import_module(module_name)


def verify_runtime_environment(lock_path: Path = LOCK_PATH,
                               wheel_manifest_path: Path = WHEEL_MANIFEST_PATH) -> dict:
    """Verify locked CPython/platform and every locked installed distribution."""
    lock = _read_regular_bounded(lock_path, 1_000_000)
    wheels = _read_regular_bounded(wheel_manifest_path, 20_000_000)
    if _sha(lock) != LOCK_SHA256 or _sha(wheels) != WHEEL_MANIFEST_SHA256:
        raise BootstrapError("runtime lock/wheel-manifest hashes differ from frozen CPU environment")
    try:
        manifest = json.loads(wheels)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BootstrapError("invalid pinned wheel manifest") from exc
    if (manifest.get("requirements_sha256") != LOCK_SHA256
            or manifest.get("target", {}).get("implementation") != "CPython"
            or manifest.get("target", {}).get("python_major_minor") != [3, 10]
            or manifest.get("target", {}).get("resolution_python_patch") != "3.10.19"
            or manifest.get("target", {}).get("os") != "Linux"
            or manifest.get("target", {}).get("machine") != "x86_64"):
        raise BootstrapError("wheel manifest target/lock binding mismatch")
    if tuple(sys.version_info[:3]) != (3, 10, 19) or platform.python_implementation() != "CPython" or platform.system() != "Linux" or platform.machine() != "x86_64":
        raise BootstrapError("pinned CPython 3.10.19 / Linux x86_64 runtime required")
    expected = {}
    for line in lock.decode("utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^ ;]+)\s+--hash=sha256:[0-9a-f]{64}$", line)
        if match:
            expected[re.sub(r"[-_.]+", "-", match.group(1)).lower()] = match.group(2)
        elif line.strip() and not line.startswith("#"):
            raise BootstrapError("unrecognized line in frozen environment lock")
    if not expected:
        raise BootstrapError("empty frozen environment lock")
    observed = {}
    for package, version in sorted(expected.items()):
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise BootstrapError(f"locked distribution missing: {package}") from exc
        if actual != version:
            raise BootstrapError(f"locked distribution mismatch: {package}={actual}, expected {version}")
        observed[package] = actual
    package_modules = {
        "torch": ("torch", "torch._C", "torch.nn", "torch.nn.functional", "torch.optim"),
        "numpy": ("numpy", "numpy.core._multiarray_umath"),
        "torch-geometric": ("torch_geometric", "torch_geometric.nn"),
        "torch-cluster": ("torch_cluster",), "torch-scatter": ("torch_scatter",),
        "torch-sparse": ("torch_sparse",), "torch-spline-conv": ("torch_spline_conv",),
        "torch-geometric-temporal": (pinned_import.CELL_MODULE,),
    }
    installed_files = {}
    prefix = Path(sys.prefix).resolve()
    for package, module_names in package_modules.items():
        dist = importlib.metadata.distribution(package)
        for module_name in module_names:
            if module_name == pinned_import.CELL_MODULE:
                continue  # Loaded by pinned_graph_import; never import Temporal initializers.
            try:
                _import_runtime_module(module_name)
            except Exception as exc:
                raise BootstrapError(f"pinned native module import failed: {module_name}") from exc
        record_entries = {str(item): item for item in (dist.files or ())}
        record_path = Path(dist.locate_file(dist._path / "RECORD")).resolve()
        if prefix not in record_path.parents or not record_path.is_file():
            raise BootstrapError(f"distribution RECORD outside active venv: {package}")
        record_sha = _sha(_read_regular_bounded(record_path, 16_000_000))
        for module_name in module_names:
            module = sys.modules.get(module_name)
            if module is None:
                raise BootstrapError(f"required standalone/native module not loaded: {module_name}")
            module_path_value = getattr(module, "__file__", None)
            if not module_path_value:
                raise BootstrapError(f"active native module has no file identity: {module_name}")
            module_path = Path(module_path_value).resolve()
            if prefix not in module_path.parents or not module_path.is_file():
                raise BootstrapError(f"active module outside pinned venv: {module_name}")
            try:
                relative = module_path.relative_to(Path(dist.locate_file("")).resolve()).as_posix()
                record = record_entries[relative]
            except (ValueError, KeyError) as exc:
                raise BootstrapError(f"active module missing from installed RECORD: {module_name}") from exc
            if record.hash is None or record.hash.mode != "sha256":
                raise BootstrapError(f"active module lacks a SHA-256 RECORD entry: {module_name}")
            module_sha = _sha(_read_regular_bounded(module_path, 512_000_000))
            expected_digest = base64.urlsafe_b64decode(record.hash.value + "=" * (-len(record.hash.value) % 4)).hex()
            if module_sha != expected_digest:
                raise BootstrapError(f"active module differs from installed RECORD: {module_name}")
            installed_files[module_name] = {"path": str(module_path), "sha256": module_sha,
                                            "distribution": package, "record_sha256": record_sha}
    if not installed_files:
        raise BootstrapError("no active native package module identities found")
    return {"python": sys.version, "python_version": list(sys.version_info[:3]),
            "platform": platform.platform(), "machine": platform.machine(), "prefix": str(prefix),
            "lock_sha256": _sha(lock), "wheel_manifest_sha256": _sha(wheels),
            "installed_versions": observed, "installed_module_files": installed_files}


def load_approved_environment_identity(path: Path, expected_sha256: str,
                                      source_identity: Mapping[str, Any] | None = None) -> tuple[dict, str]:
    """Load an independently pinned runtime inventory and match active files."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or ""):
        raise BootstrapError("approved environment identity SHA-256 is required")
    raw = _read_regular_bounded(path, MAX_MANIFEST_BYTES)
    if _sha(raw) != expected_sha256:
        raise BootstrapError("approved environment identity digest mismatch")
    try:
        document = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BootstrapError("invalid approved environment identity JSON") from exc
    if (type(document) is not dict or set(document) != {"schema", "identity"}
            or document.get("schema") != "tgrab-node2a-environment-identity-v1"
            or type(document.get("identity")) is not dict):
        raise BootstrapError("approved environment identity schema mismatch")
    actual = verify_runtime_environment()
    if source_identity is not None:
        actual["source_identity"] = dict(source_identity)
    if document["identity"] != actual:
        raise BootstrapError("active Python/package/module/RECORD identities differ from approved inventory")
    return actual, expected_sha256


def _source_identity(source_root: Path, cell_path: Path) -> dict:
    helper_path = Path(pinned_import.__file__).resolve()
    graph_path = Path(gr.__file__).resolve()
    wrappers = {relative: _sha(_read_regular_bounded(Path(source_root) / relative, 64_000))
                for relative in gr.WRAPPER_SHA256}
    if wrappers != gr.WRAPPER_SHA256:
        raise BootstrapError("pinned wrapper source identity mismatch")
    cell_sha = _sha(_read_regular_bounded(cell_path, pinned_import.MAX_SOURCE_BYTES))
    graph_sha = _sha(_read_regular_bounded(graph_path, 4_000_000))
    helper_sha = _sha(_read_regular_bounded(helper_path, 1_000_000))
    if (cell_sha != gr.CELL_SHA256 or graph_sha != GRAPH_RECONSTRUCTION_SHA256
            or helper_sha != PINNED_GRAPH_IMPORT_SHA256):
        raise BootstrapError("pinned native source identity mismatch")
    return {"wrappers": wrappers, "cell_sha256": cell_sha,
            "graph_reconstruction_sha256": graph_sha,
            "pinned_graph_import_sha256": helper_sha}


def _load_pinned_sources(source_root: Path):
    """Load the reviewed standalone GC-LSTM path, bypassing Temporal initializers."""
    loader_path = Path(pinned_import.__file__).resolve()
    if _sha(_read_regular_bounded(loader_path, 1_000_000)) != PINNED_GRAPH_IMPORT_SHA256:
        raise BootstrapError("standalone graph import helper differs from reviewed source pin")
    import torch
    cell_path = Path(importlib.metadata.distribution("torch-geometric-temporal").locate_file(
        "torch_geometric_temporal/nn/recurrent/gc_lstm.py")).resolve()
    if _sha(_read_regular_bounded(cell_path, pinned_import.MAX_SOURCE_BYTES)) != pinned_import.PINNED_CELL_SHA256:
        raise BootstrapError("installed standalone GC-LSTM source pin mismatch")
    try:
        modules = pinned_import.load_pinned_graph_modules(source_root=Path(source_root),
            installed_cell_path=cell_path, cell_sha256=pinned_import.PINNED_CELL_SHA256,
            wrapper_sha256=gr.WRAPPER_SHA256, guard=lambda: None)
    except Exception as exc:
        raise BootstrapError("standalone pinned graph import failed") from exc
    if not modules.provenance["source_release_hashes_match"]:
        raise BootstrapError("standalone source release identities differ from pins")
    if _sha(_read_regular_bounded(loader_path, 1_000_000)) != PINNED_GRAPH_IMPORT_SHA256:
        raise BootstrapError("standalone graph import helper changed during loading")
    return torch, modules, cell_path


def write_environment_identity(path: Path, source_root: Path) -> str:
    """Capture an inventory for independent review; does not authorize execution."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise BootstrapError("environment identity output must be new and absent")
    _, _, cell_path = _load_pinned_sources(source_root)
    source_before = _source_identity(source_root, cell_path)
    identity = verify_runtime_environment()
    identity["source_identity"] = source_before
    if _source_identity(source_root, cell_path) != source_before:
        raise BootstrapError("pinned source changed during environment identity capture")
    raw = json.dumps({"schema": "tgrab-node2a-environment-identity-v1", "identity": identity},
                     sort_keys=True, indent=2).encode() + b"\n"
    if len(raw) > MAX_MANIFEST_BYTES:
        raise BootstrapError("environment identity exceeds bounded receipt size")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    _fsync_directory(path.parent)
    return _sha(raw)


def load_release_contract(catalog_path: Path = CATALOG_PATH,
                          config_path: Path = CONFIG_PATH) -> tuple[dict, dict, dict]:
    """Read and hash-check the exact frozen catalog/config at runtime."""
    catalog, catalog_raw = _pinned_json(catalog_path, CATALOG_SHA256)
    config, config_raw = _pinned_json(config_path, CONFIG_SHA256)
    if catalog.get("kind") != "tgrab_selected_tensor_catalog_v1" or catalog.get("status") != "source_derived_expected_catalog_frozen_for_independent_review":
        raise BootstrapError("unexpected selected tensor catalog identity/status")
    if catalog.get("native_load_verified") is not False or catalog.get("native_state_dict_observed") is not False:
        raise BootstrapError("catalog must remain a source expectation until native observation")
    gr.validate_frozen_config(config)
    if config.get("candidate", {}).get("condition") != "primary_Pdet_2_32":
        raise BootstrapError("bootstrap requires the frozen primary Pdet(2,32) configuration")
    if catalog.get("target", {}).get("hidden_channels") != 128 or catalog.get("total_model_numel") != 232833:
        raise BootstrapError("catalog does not encode the frozen GC-LSTM 233k parameter candidate")
    return catalog, config, {"catalog_sha256": _sha(catalog_raw), "config_sha256": _sha(config_raw)}


def _tensor_bytes(value: Any, torch: Any, descriptor: Mapping[str, Any]) -> bytes:
    if not isinstance(value, torch.Tensor):
        raise BootstrapError("state_dict value is not a native tensor")
    if (str(value.dtype) != "torch.float32" or str(value.device) != "cpu"
            or value.layout != torch.strided or not value.is_contiguous()
            or list(value.shape) != descriptor["shape"]
            or not torch.isfinite(value).all()):
        raise BootstrapError(f"native tensor violates frozen catalog: {descriptor['name']}")
    return value.detach().contiguous().numpy().astype("<f4", copy=False).tobytes(order="C")


def observe_catalog(torch: Any, models: Mapping[str, Any], catalog: Mapping[str, Any]) -> dict[str, bytes]:
    """Assert actual component state_dicts match every frozen tensor descriptor."""
    if set(models) != set(COMPONENTS):
        raise BootstrapError("both and only node_emb/link_pred components are required")
    payloads: dict[str, bytes] = {}
    all_object_ids, storage_ranges = set(), {}
    for key in COMPONENTS:
        spec = catalog["components"]["recurrent" if key == "node_emb" else "link_head"]
        model = models[key]
        state = model.state_dict()
        expected = {item["name"]: item for item in spec["tensors"]}
        try:
            named_parameters = list(model.named_parameters(remove_duplicate=False))
            named_buffers = list(model.named_buffers(remove_duplicate=False))
        except TypeError as exc:
            raise BootstrapError("native module must expose non-deduplicating named registries") from exc
        parameters = dict(named_parameters)
        buffers = dict(named_buffers)
        if len(parameters) != len(named_parameters) or len(buffers) != len(named_buffers):
            raise BootstrapError(f"{key} duplicate registered parameter/buffer names")
        parameter_names = {name for name, item in expected.items() if item["classification"] == "parameter"}
        buffer_names = {name for name, item in expected.items() if item["classification"] == "buffer"}
        if set(parameters) != parameter_names or set(buffers) != buffer_names:
            raise BootstrapError(f"{key} named parameter/buffer classification differs from catalog")
        if spec.get("persistent_buffer_count") != len(buffer_names):
            raise BootstrapError(f"{key} persistent buffer count differs from catalog")
        identities = [id(value) for value in parameters.values()]
        if len(identities) != len(set(identities)):
            raise BootstrapError(f"{key} parameter objects alias across names")
        for tensor in list(parameters.values()) + list(buffers.values()):
            identity = id(tensor)
            if identity in all_object_ids:
                raise BootstrapError("native tensor object aliases across components")
            all_object_ids.add(identity)
            storage = tensor.untyped_storage()
            storage_key = (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))
            start = int(tensor.storage_offset()) * int(tensor.element_size())
            end = start + int(tensor.numel()) * int(tensor.element_size())
            for prior_start, prior_end in storage_ranges.get(storage_key, []):
                if max(start, prior_start) < min(end, prior_end):
                    raise BootstrapError("native tensor storage intervals overlap")
            storage_ranges.setdefault(storage_key, []).append((start, end))
        registry = {**parameters, **buffers}
        if list(state.keys()) != spec["native_state_dict_order"]:
            raise BootstrapError(f"{key} native state_dict key/order mismatch")
        if set(state) != set(expected) or len(state) != spec["tensor_count"]:
            raise BootstrapError(f"{key} native state_dict catalog membership mismatch")
        for name in spec["serialization_order"]:
            item = expected[name]
            value = state[name]
            registered = registry[name]
            if (value.untyped_storage().data_ptr() != registered.untyped_storage().data_ptr()
                    or value.storage_offset() != registered.storage_offset()
                    or tuple(value.shape) != tuple(registered.shape)
                    or tuple(value.stride()) != tuple(registered.stride())):
                raise BootstrapError(f"state_dict value is not the registered tensor view: {key}.{name}")
            if item["classification"] != "parameter":
                raise BootstrapError("bootstrap catalog contains a non-parameter tensor")
            raw = _tensor_bytes(value, torch, item)
            if len(raw) != item["nbytes"] or len(raw) != item["numel"] * 4:
                raise BootstrapError(f"tensor byte count mismatch: {key}.{name}")
            payloads[f"{key}.{name}"] = raw
    if sum(map(len, payloads.values())) != catalog["total_model_nbytes"]:
        raise BootstrapError("observed component byte total differs from catalog")
    return payloads


def actual_rng_snapshot(torch: Any) -> dict:
    """Capture actual Python, NumPy and Torch CPU RNG states and their digest."""
    py = random.getstate()
    nr = np.random.get_state()
    tr = bytes(torch.get_rng_state().cpu().numpy().tobytes())
    value = {
        "python": {"version": py[0], "state": list(py[1]), "gauss": py[2]},
        "numpy": {"algorithm": nr[0], "state": nr[1].astype("<u4").tolist(),
                  "position": int(nr[2]), "has_gauss": int(nr[3]), "cached_gaussian": float(nr[4])},
        "torch_cpu_hex": tr.hex(),
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"state": value, "sha256": _sha(canonical)}


def seeded_fresh_models(seed: int, torch: Any,
                        factory: Callable[[], Mapping[str, Any]]) -> tuple[dict[str, Any], dict]:
    """Seed all frozen CPU streams before construction and bind actual RNG state."""
    if type(seed) is not int or seed != 0:
        raise BootstrapError("Figure 10 bootstrap model seed must be exactly zero")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    models = dict(factory())
    if set(models) != set(COMPONENTS):
        raise BootstrapError("factory must create both pinned components")
    return models, actual_rng_snapshot(torch)


def _state_payload(state: gr.CompleteState, torch: Any) -> dict[str, bytes]:
    state.validate(gr.StateLayout(1, 100, 128, tuple(range(100))))
    payload = {}
    for name, value in (("hidden", state.hidden), ("cell", state.cell)):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or str(value.device) != "cpu" or tuple(value.shape) != STATE_SHAPE
                or not torch.isfinite(value).all()):
            raise BootstrapError(f"complete {name} state violates Figure 10 layout")
        payload[f"state.{name}"] = value.detach().contiguous().numpy().astype("<f4", copy=False).tobytes()
    return payload


def _serialize(payloads: Mapping[str, bytes], catalog: Mapping[str, Any],
               rng: Mapping[str, Any], state: gr.CompleteState) -> tuple[bytes, dict]:
    ordered = []
    offset = 0
    for component, catalog_key in (("node_emb", "recurrent"), ("link_pred", "link_head")):
        spec = catalog["components"][catalog_key]
        for name in spec["serialization_order"]:
            full = f"{component}.{name}"
            raw = payloads[full]
            ordered.append({"name": full, "offset": offset, "nbytes": len(raw), "sha256": _sha(raw),
                            "shape": next(t["shape"] for t in spec["tensors"] if t["name"] == name),
                            "dtype": "float32_le"})
            offset += len(raw)
    for name in ("state.hidden", "state.cell"):
        raw = payloads[name]
        ordered.append({"name": name, "offset": offset, "nbytes": len(raw), "sha256": _sha(raw),
                        "shape": list(STATE_SHAPE), "dtype": "float32_le"})
        offset += len(raw)
    if offset > MAX_BUNDLE_BYTES:
        raise BootstrapError("bootstrap bundle exceeds hard size bound")
    blob = b"".join(payloads[item["name"]] for item in ordered)
    manifest = {"schema": SCHEMA, "catalog_sha256": CATALOG_SHA256,
                "config_sha256": CONFIG_SHA256, "rng": rng,
                "model_seed": 0, "optimizer_updates": 0,
                "components": list(COMPONENTS), "complete_state": {"shape": list(STATE_SHAPE), "components": ["hidden", "cell"], "next_target_t": state.next_target_t, "node_ids": list(state.node_ids)},
                "entries": ordered, "payload_sha256": _sha(blob), "payload_bytes": len(blob),
                "primary_payload_opened": False, "validation_payload_opened": False, "test_payload_opened": False}
    return blob, manifest


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_transaction(destination: Path, blob: bytes, manifest: Mapping[str, Any],
                      *, crash_before_publish: bool = False) -> dict:
    """Publish immutable bundle atomically; incomplete staging is never readable."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise BootstrapError("transaction destination must be new and absent")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        payload_path = stage / "payload.f32le"
        with payload_path.open("xb") as stream:
            stream.write(blob); stream.flush(); os.fsync(stream.fileno())
        manifest_path = stage / "manifest.json"
        encoded = json.dumps(dict(manifest), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if len(encoded) > MAX_MANIFEST_BYTES or len(blob) > MAX_BUNDLE_BYTES:
            raise BootstrapError("transaction files exceed hard byte limits")
        with manifest_path.open("xb") as stream:
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
        _fsync_directory(stage)
        if crash_before_publish:
            raise RuntimeError("injected crash before transaction publish")
        os.replace(stage, destination)
        _fsync_directory(destination.parent)
        return {"manifest_sha256": _sha(encoded), "payload_sha256": _sha(blob)}
    except BaseException:
        # Simulated/real crash staging is intentionally left behind for recovery audit.
        raise


def read_transaction(source: Path, catalog: Mapping[str, Any], *, expected_manifest_sha256: str,
                     expected_payload_sha256: str) -> tuple[bytes, dict]:
    source = Path(source)
    if source.is_symlink() or not source.is_dir():
        raise BootstrapError("published transaction directory required")
    if {entry.name for entry in source.iterdir()} != {"manifest.json", "payload.f32le"}:
        raise BootstrapError("transaction directory contents are not the exact closed bundle")
    manifest_path, payload_path = source / "manifest.json", source / "payload.f32le"
    if any(p.is_symlink() or not p.is_file() for p in (manifest_path, payload_path)):
        raise BootstrapError("published transaction is incomplete")
    raw_manifest = _read_regular_bounded(manifest_path, MAX_MANIFEST_BYTES)
    if _sha(raw_manifest) != expected_manifest_sha256:
        raise BootstrapError("transaction manifest differs from independent in-memory hash")
    try:
        manifest = json.loads(raw_manifest)
    except Exception as exc:
        raise BootstrapError("transaction manifest is invalid JSON") from exc
    payload_size = payload_path.stat().st_size
    if payload_size > MAX_BUNDLE_BYTES:
        raise BootstrapError("transaction payload exceeds hard bound")
    _validate_closed_manifest(manifest, payload_size)
    expected = _expected_entry_layout(catalog)
    actual = [(entry["name"], entry["nbytes"], entry["shape"]) for entry in manifest["entries"]]
    if actual != expected:
        raise BootstrapError("transaction descriptor order/shape/size differs from pinned tensor catalog")
    blob = _read_regular_bounded(payload_path, MAX_BUNDLE_BYTES)
    if _sha(blob) != expected_payload_sha256:
        raise BootstrapError("transaction payload differs from independent in-memory hash")
    if manifest.get("schema") != SCHEMA or manifest.get("payload_bytes") != len(blob) or manifest.get("payload_sha256") != _sha(blob):
        raise BootstrapError("published transaction integrity failure")
    cursor = 0
    for entry in manifest["entries"]:
        start, end = entry["offset"], entry["offset"] + entry["nbytes"]
        if start != cursor or end > len(blob) or _sha(blob[start:end]) != entry["sha256"]:
            raise BootstrapError("payload entry offset/hash invalid")
        cursor = end
        floats = np.frombuffer(blob[start:end], dtype="<f4")
        if not np.isfinite(floats).all():
            raise BootstrapError("nonfinite float in serialized model/state payload")
    if cursor != len(blob):
        raise BootstrapError("unreferenced payload bytes")
    return blob, manifest


def _validate_closed_manifest(manifest: Any, payload_size: int) -> None:
    keys = {"schema", "catalog_sha256", "config_sha256", "rng", "model_seed", "optimizer_updates",
            "components", "complete_state", "entries", "payload_sha256", "payload_bytes",
            "primary_payload_opened", "validation_payload_opened", "test_payload_opened"}
    if type(manifest) is not dict or set(manifest) != keys:
        raise BootstrapError("transaction manifest keys are not the exact closed schema")
    if (manifest["schema"] != SCHEMA or manifest["catalog_sha256"] != CATALOG_SHA256
            or manifest["config_sha256"] != CONFIG_SHA256 or type(manifest["model_seed"]) is not int
            or manifest["model_seed"] != 0 or type(manifest["optimizer_updates"]) is not int
            or manifest["optimizer_updates"] != 0 or manifest["components"] != list(COMPONENTS)
            or manifest["primary_payload_opened"] is not False
            or manifest["validation_payload_opened"] is not False or manifest["test_payload_opened"] is not False):
        raise BootstrapError("transaction manifest fixed protocol fields mismatch")
    state = manifest["complete_state"]
    if (type(state) is not dict or set(state) != {"shape", "components", "next_target_t", "node_ids"}
            or type(state["shape"]) is not list or any(type(x) is not int for x in state["shape"])
            or state["shape"] != list(STATE_SHAPE)
            or type(state["components"]) is not list or state["components"] != ["hidden", "cell"]
            or type(state["next_target_t"]) is not int or state["next_target_t"] < 0
            or type(state["node_ids"]) is not list or any(type(x) is not int for x in state["node_ids"])
            or state["node_ids"] != list(range(100))):
        raise BootstrapError("complete state descriptor mismatch")
    _validate_rng_manifest(manifest["rng"])
    entries = manifest["entries"]
    if type(entries) is not list or len(entries) != 28:
        raise BootstrapError("transaction must contain exactly 26 model tensors and H/C")
    names, cursor = set(), 0
    for item in entries:
        if type(item) is not dict or set(item) != {"name", "offset", "nbytes", "sha256", "shape", "dtype"}:
            raise BootstrapError("payload descriptor is not the exact closed schema")
        name, offset, nbytes, shape = item["name"], item["offset"], item["nbytes"], item["shape"]
        if (type(name) is not str or name in names or type(offset) is not int or offset != cursor
                or type(nbytes) is not int or nbytes <= 0 or type(shape) is not list or not 1 <= len(shape) <= 3
                or any(type(dim) is not int or dim <= 0 for dim in shape)
                or math.prod(shape) > MAX_BUNDLE_BYTES // 4
                or nbytes != math.prod(shape) * 4 or type(item["dtype"]) is not str or item["dtype"] != "float32_le"
                or type(item["sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
            raise BootstrapError("payload descriptor value/type/offset mismatch")
        names.add(name); cursor += nbytes
    if (cursor != payload_size or type(manifest["payload_bytes"]) is not int
            or manifest["payload_bytes"] != payload_size):
        raise BootstrapError("manifest payload length mismatch")
    for key in ("catalog_sha256", "config_sha256", "payload_sha256"):
        if type(manifest[key]) is not str or not re.fullmatch(r"[0-9a-f]{64}", manifest[key]):
            raise BootstrapError(f"invalid digest field: {key}")


def _expected_entry_layout(catalog: Mapping[str, Any]) -> list[tuple[str, int, list[int]]]:
    expected = []
    for component, catalog_key in (("node_emb", "recurrent"), ("link_pred", "link_head")):
        spec = catalog["components"][catalog_key]
        by_name = {item["name"]: item for item in spec["tensors"]}
        for name in spec["serialization_order"]:
            item = by_name[name]
            expected.append((f"{component}.{name}", item["nbytes"], item["shape"]))
    expected.extend((f"state.{name}", int(np.prod(STATE_SHAPE)) * 4, list(STATE_SHAPE))
                    for name in ("hidden", "cell"))
    return expected


def _validate_rng_manifest(snapshot: Mapping[str, Any]) -> None:
    if (type(snapshot) is not dict or set(snapshot) != {"state", "sha256"}
            or type(snapshot.get("state")) is not dict):
        raise BootstrapError("transaction lacks actual RNG state")
    value = snapshot["state"]
    py, nr = value.get("python"), value.get("numpy")
    if (set(value) != {"python", "numpy", "torch_cpu_hex"}
            or type(py) is not dict or set(py) != {"version", "state", "gauss"}
            or type(py["version"]) is not int or type(py["state"]) is not list
            or any(type(x) is not int for x in py["state"])
            or (py["gauss"] is not None and (type(py["gauss"]) not in (int, float) or not math.isfinite(py["gauss"])))
            or type(nr) is not dict or set(nr) != {"algorithm", "state", "position", "has_gauss", "cached_gaussian"}
            or type(nr["algorithm"]) is not str or type(nr["state"]) is not list
            or any(type(x) is not int for x in nr["state"])
            or type(nr["position"]) is not int or type(nr["has_gauss"]) is not int
            or type(nr["cached_gaussian"]) not in (int, float) or not math.isfinite(nr["cached_gaussian"])
            or type(value["torch_cpu_hex"]) is not str
            or not re.fullmatch(r"(?:[0-9a-f]{2})+", value["torch_cpu_hex"])):
        raise BootstrapError("RNG state primitive schema mismatch")
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if snapshot.get("sha256") != _sha(raw):
        raise BootstrapError("transaction RNG state digest does not match captured state")


def restore_transaction(torch: Any, models: Mapping[str, Any], state: gr.CompleteState,
                        source: Path, catalog: Mapping[str, Any], *,
                        expected_manifest_sha256: str,
                        expected_payload_sha256: str) -> dict:
    """Strictly restore tensor bytes to existing native objects; never constructs or loads data."""
    blob, manifest = read_transaction(source, catalog, expected_manifest_sha256=expected_manifest_sha256,
                                      expected_payload_sha256=expected_payload_sha256)
    if manifest.get("catalog_sha256") != CATALOG_SHA256 or manifest.get("config_sha256") != CONFIG_SHA256:
        raise BootstrapError("transaction release contract pins mismatch")
    _validate_rng_manifest(manifest.get("rng"))
    expected_layout = _expected_entry_layout(catalog)
    actual_layout = [(item.get("name"), item.get("nbytes"), item.get("shape")) for item in manifest["entries"]]
    if actual_layout != expected_layout or any(item.get("dtype") != "float32_le" for item in manifest["entries"]):
        raise BootstrapError("transaction entry schema/order differs from exact selected tensor catalog")
    meta = manifest.get("complete_state", {})
    if meta.get("shape") != list(STATE_SHAPE) or meta.get("next_target_t") != state.next_target_t or meta.get("node_ids") != list(state.node_ids):
        raise BootstrapError("complete-state chronology/order does not match initialized receiver")
    expected = observe_catalog(torch, models, catalog)
    del expected
    state_bytes = _state_payload(state, torch)
    del state_bytes
    cursor = 0
    for component, catalog_key in (("node_emb", "recurrent"), ("link_pred", "link_head")):
        spec = catalog["components"][catalog_key]
        current = models[component].state_dict()
        loaded = {}
        for name in spec["serialization_order"]:
            item = next(t for t in spec["tensors"] if t["name"] == name)
            nbytes = item["nbytes"]
            array = np.frombuffer(blob[cursor:cursor+nbytes], dtype="<f4").copy().reshape(item["shape"])
            loaded[name] = torch.from_numpy(array)
            cursor += nbytes
        models[component].load_state_dict(loaded, strict=True)
    for name in ("state.hidden", "state.cell"):
        raw = blob[cursor:cursor + int(np.prod(STATE_SHAPE)) * 4]
        tensor = torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy().reshape(STATE_SHAPE))
        if name.endswith("hidden"):
            state.hidden.copy_(tensor)
        else:
            state.cell.copy_(tensor)
        cursor += len(raw)
    if cursor != len(blob):
        raise BootstrapError("restore did not consume complete payload")
    return manifest


def restore_rng_snapshot(torch: Any, snapshot: Mapping[str, Any]) -> None:
    """Restore the complete Python/NumPy/Torch CPU state captured above."""
    try:
        value = snapshot["state"]
        py = value["python"]
        random.setstate((int(py["version"]), tuple(int(x) for x in py["state"]), py["gauss"]))
        nr = value["numpy"]
        np.random.set_state((nr["algorithm"], np.asarray(nr["state"], dtype=np.uint32),
                             int(nr["position"]), int(nr["has_gauss"]), float(nr["cached_gaussian"])))
        raw = bytes.fromhex(value["torch_cpu_hex"])
        torch.set_rng_state(torch.tensor(list(raw), dtype=torch.uint8))
        if actual_rng_snapshot(torch)["sha256"] != snapshot["sha256"]:
            raise BootstrapError("restored RNG state digest mismatch")
    except BootstrapError:
        raise
    except Exception as exc:
        raise BootstrapError("cannot restore actual RNG snapshot") from exc


def execute_roundtrip_parity(*, torch: Any, models: Mapping[str, Any], optimizer: Any,
                             state: gr.CompleteState, catalog: Mapping[str, Any],
                             hashes: Mapping[str, str], destination: Path,
                             observe_native: Callable[[Mapping[str, Any], gr.CompleteState], Mapping[str, Any]],
                             crash_before_publish: bool = True) -> dict:
    """Run native observations around a real dual-model/H+C serialize/restore.

    ``observe_native`` must execute the pinned backend's synthetic graph
    forward, sigmoid/BCE loss, backward, and report ordered finite gradients,
    module train/eval flags, and dropout train/eval observations. It receives no
    graph/data pathname. The harness rejects optimizer.step, serializes both
    observed components plus H/C, injects an unpublished-crash probe, perturbs
    parameters/state, restores, replays identical RNG, then requires exact
    parameter immutability and observation parity. No optimizer update path is
    exposed or called.
    """
    if dict(hashes) != {"catalog_sha256": CATALOG_SHA256, "config_sha256": CONFIG_SHA256}:
        raise BootstrapError("runtime catalog/config identities differ from frozen contract")
    observed_payloads = observe_catalog(torch, models, catalog)
    initial_state = _state_payload(state, torch)
    observed_payloads.update(initial_state)
    rng = actual_rng_snapshot(torch)
    blob, manifest = _serialize(observed_payloads, catalog, rng, state)

    original_step = getattr(optimizer, "step", None)
    if not callable(original_step):
        raise BootstrapError("native optimizer must expose step for zero-update interception")
    step_attempts = []
    def forbidden_step(*args, **kwargs):
        step_attempts.append(True)
        raise BootstrapError("optimizer.step is forbidden during node2a bootstrap")

    instance_dict = getattr(optimizer, "__dict__", None)
    if not isinstance(instance_dict, dict):
        raise BootstrapError("optimizer must expose instance attributes for reversible step interception")
    had_instance_step = "step" in instance_dict
    original_instance_step = instance_dict.get("step")
    try:
        optimizer.step = forbidden_step
    except Exception as exc:
        raise BootstrapError("cannot intercept native optimizer.step safely") from exc
    try:
        before_weights = observe_catalog(torch, models, catalog)
        before = dict(observe_native(models, state))
        if step_attempts:
            raise BootstrapError("native callback attempted an optimizer update")
        if observe_catalog(torch, models, catalog) != before_weights or _state_payload(state, torch) != initial_state:
            raise BootstrapError("reference observation mutated initialized parameters or H/C")
        # Simulate a process interruption after durable staging but before the
        # atomic directory publish. The uncommitted directory must be ignored.
        if crash_before_publish:
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".node2a-crash-probe-", dir=destination.parent) as probe:
                interrupted = Path(probe) / "interrupted"
                try:
                    write_transaction(interrupted, blob, manifest, crash_before_publish=True)
                except RuntimeError:
                    pass
                else:
                    raise BootstrapError("crash-injection path unexpectedly published")
                try:
                    read_transaction(interrupted, catalog, expected_manifest_sha256="0" * 64,
                                     expected_payload_sha256="0" * 64)
                except BootstrapError:
                    pass
                else:
                    raise BootstrapError("unpublished crash transaction was readable")
        transaction_hashes = write_transaction(destination, blob, manifest)
        # Deliberately corrupt every registered tensor and both state tensors;
        # restore must overwrite all registered payloads exactly.
        with torch.no_grad():
            for model in models.values():
                for parameter in model.parameters():
                    parameter.add_(1.0)
        state.hidden.add_(1.0); state.cell.add_(1.0)
        restore_transaction(torch, models, state, destination, catalog,
                            expected_manifest_sha256=transaction_hashes["manifest_sha256"],
                            expected_payload_sha256=transaction_hashes["payload_sha256"])
        after_weights = observe_catalog(torch, models, catalog)
        if before_weights != after_weights:
            raise BootstrapError("dual-model serialized parameters did not restore bitwise")
        restore_rng_snapshot(torch, rng)
        after = dict(observe_native(models, state))
        if step_attempts:
            raise BootstrapError("native callback attempted optimizer.step after restore")
        if observe_catalog(torch, models, catalog) != before_weights or _state_payload(state, torch) != initial_state:
            raise BootstrapError("restored parity observation mutated parameters or H/C")
        before["optimizer_updates"] = after["optimizer_updates"] = 0
        parity = assert_parity(before, after)
        return {"status": "synthetic_native_parity_pass", "parity": parity,
                "rng": rng, "transaction_crash_reopen": bool(crash_before_publish),
                "transaction_manifest_sha256": transaction_hashes["manifest_sha256"],
                "transaction_payload_sha256": transaction_hashes["payload_sha256"],
                "optimizer_updates": 0, "primary_payload_opened": False,
                "validation_payload_opened": False, "test_payload_opened": False}
    except BaseException:
        raise
    finally:
        try:
            if had_instance_step:
                instance_dict["step"] = original_instance_step
            else:
                instance_dict.pop("step", None)
        except BaseException as cleanup_error:
            # Do not mask the primary science/runtime failure with cleanup.
            if sys.exc_info()[0] is None:
                raise BootstrapError("failed to restore optimizer.step descriptor") from cleanup_error


def run_pinned_native_fixture(source_root: Path, output: Path, *,
                              environment_identity_path: Path, environment_identity_sha256: str,
                              catalog_path: Path = CATALOG_PATH,
                              config_path: Path = CONFIG_PATH) -> dict:
    """Standalone synthetic node2a runtime using the pinned native wrappers.

    This deliberately bypasses the disabled training factory only to construct
    the frozen model for this finite zero-update engineering check. It requires
    the isolated pinned CPU environment and exact installed Temporal cell; no
    graph/data/checkpoint payload is opened. Call only under a separately
    reviewed runtime envelope.
    """
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise BootstrapError("runtime output path must be new and absent")
    torch, pinned_modules, cell_path = _load_pinned_sources(source_root)
    source_base = _source_identity(source_root, cell_path)
    environment, environment_identity_sha256 = load_approved_environment_identity(
        environment_identity_path, environment_identity_sha256, source_base)
    catalog, _, hashes = load_release_contract(catalog_path, config_path)
    source_root = Path(source_root).absolute()
    recurrent, head = pinned_modules.recurrent, pinned_modules.head
    source_identity = {**source_base,
                       "runtime_imported_files": environment["installed_module_files"],
                       "source_root": str(source_root),
                       "bootstrap_module_sha256": _sha(Path(__file__).read_bytes())}

    def factory():
        return {"node_emb": recurrent.MultiLayerGCLSTM(1, recurrent.GCLSTMParam(100, 128, 2)).to(dtype=torch.float32, device="cpu"),
                "link_pred": head.LinkPredictor(128, num_layers=2, p_dropout=0.2).to(dtype=torch.float32, device="cpu")}
    models, rng = seeded_fresh_models(0, torch, factory)
    named = [(f"{component}.{name}", param) for component in COMPONENTS
             for name, param in sorted(models[component].named_parameters())]
    expected_names = list(catalog["named_optimizer_parameter_order"])
    if [name for name, _ in named] != expected_names:
        raise BootstrapError("actual native named parameters differ from frozen optimizer catalog")
    optimizer = torch.optim.Adam([param for _, param in named], lr=1e-4,
        betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False)
    backend = gr.NativeGCLSTMBackend(torch, models, optimizer, device="cpu")
    snapshot = gr.Snapshot(0, 100, ((0, 1), (1, 0)), tuple(range(100)))
    negatives = gr.sample_negatives(snapshot, np.random.default_rng(1))
    state = backend.reset()
    baseline_weights = observe_catalog(torch, models, catalog)

    def observe_native(_models, current_state):
        with backend.training_mode():
            train_flags = [bool(module.training) for model in models.values() for module in model.modules()]
            evolved = backend.consume(current_state, snapshot)
            pos = backend._probabilities(evolved, snapshot.positive_pairs)
            neg = backend._probabilities(evolved, negatives)
            loss = torch.nn.functional.binary_cross_entropy(pos, torch.ones_like(pos)) + torch.nn.functional.binary_cross_entropy(neg, torch.zeros_like(neg))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = {}
            for component in COMPONENTS:
                for name, parameter in sorted(models[component].named_parameters()):
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise BootstrapError(f"missing/nonfinite native gradient: {component}.{name}")
                    gradients[f"{component}.{name}"] = parameter.grad.detach().cpu().numpy().copy()
            # More than one edge probes independent train-mode dropout masks;
            # eval mode must disable dropout and gradients.
            probe_pairs = tuple((0, v) for v in range(2, 100))
            train_dropout = backend._probabilities(evolved, probe_pairs).detach().cpu().numpy().copy()
        with backend.evaluation_mode():
            eval_flags = [bool(module.training) for model in models.values() for module in model.modules()]
            eval_dropout = backend._probabilities(evolved, probe_pairs).detach().cpu().numpy().copy()
        if not all(train_flags) or any(eval_flags):
            raise BootstrapError("native train/eval module modes were not applied consistently")
        if np.array_equal(train_dropout, eval_dropout):
            raise BootstrapError("train/eval dropout probe did not distinguish the pinned head mode")
        forward = np.concatenate((evolved.hidden.detach().cpu().numpy().ravel(),
                                  evolved.cell.detach().cpu().numpy().ravel(),
                                  pos.detach().cpu().numpy().ravel(), neg.detach().cpu().numpy().ravel()))
        return {"forward": forward, "loss": float(loss.detach().cpu()), "gradients": gradients,
                "train_modes": train_flags, "eval_modes": eval_flags,
                "dropout_train": train_dropout, "dropout_eval": eval_dropout,
                "optimizer_updates": 0}

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.node2a-", dir=output.parent))
    try:
        result = execute_roundtrip_parity(torch=torch, models=models, optimizer=optimizer,
            state=state, catalog=catalog, hashes=hashes, destination=stage / "transaction",
            observe_native=observe_native)
        if result["rng"]["sha256"] != rng["sha256"]:
            raise BootstrapError("RNG state changed between deterministic initialization and transaction capture")
        if observe_catalog(torch, models, catalog) != baseline_weights:
            raise BootstrapError("native parameters changed during no-update bootstrap")
        final_environment = verify_runtime_environment()
        final_environment["source_identity"] = source_base
        if (final_environment != environment
                or _source_identity(source_root, cell_path) != source_base
                or _sha(Path(__file__).read_bytes()) != source_identity["bootstrap_module_sha256"]):
            raise BootstrapError("runtime source/environment identity changed during fixture")
        evidence = {"schema": "node2a-synthetic-bootstrap-evidence-v1", "result": result,
                    "initialization_rng_state_sha256": rng["sha256"],
                    "transaction_rng_state_sha256": result["rng"]["sha256"],
                    "catalog_sha256": hashes["catalog_sha256"], "config_sha256": hashes["config_sha256"],
                    "runtime_environment": environment, "source_identity": source_identity}
        evidence_raw = json.dumps(evidence, sort_keys=True, indent=2).encode() + b"\n"
        evidence_hash = _sha(evidence_raw)
        execution = Node2AExecutionResult(evidence_hash,
            result["transaction_manifest_sha256"], result["transaction_payload_sha256"],
            environment, environment_identity_sha256, source_identity, _RUNTIME_TOKEN)
        receipt = _publish_complete_bundle(stage, output, evidence_raw=evidence_raw,
            execution=execution, catalog=catalog,
            manifest_sha256=result["transaction_manifest_sha256"],
            payload_sha256=result["transaction_payload_sha256"])
        return receipt
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-environment-identity", type=Path,
                        help="capture inventory for independent review; does not run model code")
    parser.add_argument("--environment-identity", type=Path,
                        help="independently reviewed runtime inventory required for execution")
    parser.add_argument("--environment-identity-sha256",
                        help="reviewed SHA-256 of --environment-identity")
    parser.add_argument("--source-root", type=Path,
                        help="regular pinned T-GRAB source snapshot directory")
    parser.add_argument("--output", type=Path,
                        help="new run directory for synthetic node2a artifacts")
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)
    if args.capture_environment_identity is not None:
        if any((args.output, args.environment_identity, args.environment_identity_sha256)):
            parser.error("identity capture is exclusive with native execution arguments")
        if not args.source_root:
            parser.error("identity capture requires --source-root for standalone pinned imports")
        digest = write_environment_identity(args.capture_environment_identity, args.source_root)
        print(json.dumps({"environment_identity_sha256": digest}, sort_keys=True))
        return 0
    if not args.source_root or not args.output or not args.environment_identity or not args.environment_identity_sha256:
        parser.error("native execution requires source, output, reviewed environment identity and its SHA-256")
    receipt = run_pinned_native_fixture(args.source_root, args.output,
                                        environment_identity_path=args.environment_identity,
                                        environment_identity_sha256=args.environment_identity_sha256,
                                        catalog_path=args.catalog, config_path=args.config)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def assert_parity(before: Mapping[str, Any], after: Mapping[str, Any], *, atol: float = 0.0, rtol: float = 0.0) -> dict:
    """Compare callback observations from native pre/post restore runs.

    Required keys: ``forward``, ``loss``, ``gradients``, ``train_modes``,
    ``eval_modes``, ``dropout_train``, ``dropout_eval``, ``optimizer_updates``.
    Gradient mappings retain parameter names/order. Exact parity is default;
    numerical tolerance must be explicitly justified by the native runtime.
    """
    keys = ("forward", "loss", "gradients", "train_modes", "eval_modes", "dropout_train", "dropout_eval")
    if any(key not in before or key not in after for key in keys):
        raise BootstrapError("native parity callback omitted required observation")
    if before["optimizer_updates"] != 0 or after["optimizer_updates"] != 0:
        raise BootstrapError("bootstrap parity must execute zero optimizer updates")
    if before["train_modes"] != after["train_modes"] or before["eval_modes"] != after["eval_modes"]:
        raise BootstrapError("module mode/dropout configuration changed across restore")
    report = {}
    for key in ("forward", "loss", "gradients", "dropout_train", "dropout_eval"):
        left, right = before[key], after[key]
        if key == "gradients":
            if list(left) != list(right):
                raise BootstrapError("gradient parameter ordering changed")
            names = list(left)
            lvals = [left[n] for n in names]; rvals = [right[n] for n in names]
        else:
            names = [key]; lvals = [left]; rvals = [right]
        digest_left = hashlib.sha256()
        digest_right = hashlib.sha256()
        max_delta = 0.0
        elements = 0
        for name, a, b in zip(names, lvals, rvals):
            aa, bb = np.asarray(a), np.asarray(b)
            if aa.shape != bb.shape or not np.isfinite(aa).all() or not np.isfinite(bb).all():
                raise BootstrapError(f"nonfinite/shape-mismatched parity output: {name}")
            delta = np.abs(aa - bb)
            if not np.all(delta <= atol + rtol * np.abs(bb)):
                raise BootstrapError(f"native parity mismatch: {name}")
            digest_left.update(np.ascontiguousarray(aa).tobytes())
            digest_right.update(np.ascontiguousarray(bb).tobytes())
            max_delta = max(max_delta, float(delta.max(initial=0.0)))
            elements += int(aa.size)
        report[key] = {"passed": True, "atol": atol, "rtol": rtol,
                       "reference_sha256": digest_left.hexdigest(),
                       "restored_sha256": digest_right.hexdigest(),
                       "max_absolute_difference": max_delta, "elements": elements}
    return report


def make_receipt(execution: Node2AExecutionResult) -> dict:
    """Derive the pass receipt only from this process's completed runtime result."""
    if not isinstance(execution, Node2AExecutionResult) or execution._token is not _RUNTIME_TOKEN:
        raise BootstrapError("authenticated native execution result required; raw pass mappings are rejected")
    hashes = (execution.evidence_sha256, execution.transaction_manifest_sha256,
              execution.transaction_payload_sha256)
    if any(type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes):
        raise BootstrapError("execution evidence/artifact digest invalid")
    environment = execution.runtime_environment
    if (environment.get("lock_sha256") != LOCK_SHA256
            or environment.get("wheel_manifest_sha256") != WHEEL_MANIFEST_SHA256
            or environment.get("machine") != "x86_64"):
        raise BootstrapError("execution lacks the exact verified environment identity")
    if not re.fullmatch(r"[0-9a-f]{64}", execution.environment_identity_sha256 or ""):
        raise BootstrapError("execution lacks the independently pinned environment identity digest")
    source = execution.source_identity
    if (source.get("wrappers") != gr.WRAPPER_SHA256 or source.get("cell_sha256") != gr.CELL_SHA256
            or source.get("graph_reconstruction_sha256") != GRAPH_RECONSTRUCTION_SHA256
            or source.get("pinned_graph_import_sha256") != PINNED_GRAPH_IMPORT_SHA256
            or source.get("runtime_imported_files") != environment.get("installed_module_files")
            or not re.fullmatch(r"[0-9a-f]{64}", source.get("bootstrap_module_sha256", ""))):
        raise BootstrapError("execution lacks exact source/executable identities")
    environment_hash = _sha(json.dumps(environment, sort_keys=True, separators=(",", ":")).encode())
    source_hash = _sha(json.dumps(source, sort_keys=True, separators=(",", ":")).encode())
    return {"schema": "node2a-synthetic-bootstrap-pass-v1", "status": "pass",
            "catalog_sha256": CATALOG_SHA256, "config_sha256": CONFIG_SHA256,
            "evidence_sha256": execution.evidence_sha256,
            "transaction_manifest_sha256": execution.transaction_manifest_sha256,
            "transaction_payload_sha256": execution.transaction_payload_sha256,
            "runtime_environment_sha256": environment_hash, "source_identity_sha256": source_hash,
            "approved_environment_identity_sha256": execution.environment_identity_sha256,
            "both_model_components_roundtripped": True, "complete_hc_state_roundtripped": True,
            "forward_loss_gradient_parity": True, "mode_dropout_parity": True,
            "transaction_crash_reopen": True, "optimizer_updates": 0,
            "primary_payload_opened": False, "validation_payload_opened": False,
            "test_payload_opened": False, "scientific_acceptance": False}


def _publish_complete_bundle(stage: Path, output: Path, *, evidence_raw: bytes,
                             execution: Node2AExecutionResult, catalog: Mapping[str, Any],
                             manifest_sha256: str, payload_sha256: str,
                             fail_before_receipt: bool = False) -> dict:
    """Verify all artifacts, then atomically publish one complete output directory."""
    stage, output = Path(stage), Path(output)
    try:
        evidence_path = stage / "bootstrap-result.json"
        with evidence_path.open("xb") as stream:
            stream.write(evidence_raw); stream.flush(); os.fsync(stream.fileno())
        expected_evidence_hash = _sha(evidence_raw)
        read_transaction(stage / "transaction", catalog,
                         expected_manifest_sha256=manifest_sha256,
                         expected_payload_sha256=payload_sha256)
        if _sha(_read_regular_bounded(evidence_path, MAX_MANIFEST_BYTES)) != expected_evidence_hash:
            raise BootstrapError("evidence artifact hash mismatch before publication")
        if (not isinstance(execution, Node2AExecutionResult)
                or execution.evidence_sha256 != expected_evidence_hash
                or execution.transaction_manifest_sha256 != manifest_sha256
                or execution.transaction_payload_sha256 != payload_sha256):
            raise BootstrapError("authenticated execution does not bind staged evidence and transaction")
        if fail_before_receipt:
            raise BootstrapError("injected evidence-to-receipt publication failure")
        receipt = make_receipt(execution)
        receipt_raw = json.dumps(receipt, sort_keys=True, indent=2).encode() + b"\n"
        receipt_path = stage / "node2a-receipt.json"
        with receipt_path.open("xb") as stream:
            stream.write(receipt_raw); stream.flush(); os.fsync(stream.fileno())
        receipt = json.loads(_read_regular_bounded(receipt_path, MAX_MANIFEST_BYTES))
        if (receipt.get("status") != "pass" or receipt.get("evidence_sha256") != expected_evidence_hash
                or receipt.get("transaction_manifest_sha256") != manifest_sha256
                or receipt.get("transaction_payload_sha256") != payload_sha256):
            raise BootstrapError("receipt does not bind verified evidence and transaction")
        if {entry.name for entry in stage.iterdir()} != {"transaction", "bootstrap-result.json", "node2a-receipt.json"}:
            raise BootstrapError("published bundle has an unexpected member inventory")
        if {entry.name for entry in (stage / "transaction").iterdir()} != {"manifest.json", "payload.f32le"}:
            raise BootstrapError("published transaction has an unexpected member inventory")
        _fsync_directory(stage)
        if output.exists() or output.is_symlink():
            raise BootstrapError("runtime output path appeared before atomic publish")
        os.replace(stage, output)
        _fsync_directory(output.parent)
        return receipt
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
