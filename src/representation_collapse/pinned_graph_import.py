"""Import only hash-pinned GC-LSTM and public wrappers, without Temporal init.

Module import is stdlib-only. Runtime loading belongs inside an initialized
reporter after Torch import and under the worker's deadline/STOP guard. This
does not provide the full torch-geometric-temporal package API.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.machinery
import importlib.util
import inspect
from pathlib import Path
import re
import stat
import sys
from types import MappingProxyType, ModuleType
from typing import Any, Callable, Mapping

PINNED_CELL_SHA256 = "c3568d2c1ba8926ab43e11c840213f90ebe782ad158dd76809d1dc1c255797af"
PINNED_WRAPPER_SHA256 = MappingProxyType({
    "model/gclstm.py": "86069a32fced5f09c60cd795e4f0abd82674dafae250ef3673ac8c3650504609",
    "model/link_predictor.py": "cca12cfc73b8660c92236f530504da08ba0b606ec13a1a681ffabcae43b7ca86",
    "model/node_emb.py": "4be4450730f33949e0354348b3bd8f69ba0434e4aa911abec0fc5f798e8c2d29",
})
TEMPORAL = "torch_geometric_temporal"
CELL_MODULE = TEMPORAL + ".nn.recurrent.gc_lstm"
WRAPPER_NAMESPACE = "_pinned_graph_standalone_model"
MAX_SOURCE_BYTES = 64 * 1024


class PinnedImportError(ValueError):
    """Source, namespace, or class provenance could not be verified."""


@dataclass(frozen=True)
class PinnedGraphModules:
    cell: ModuleType
    recurrent: ModuleType
    head: ModuleType
    node: ModuleType
    cell_type: type
    wrapper_type: type
    parameter_type: type
    head_type: type
    provenance: Mapping[str, Any]

    def provenance_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy for the worker's immutable receipt."""
        def copy(value):
            return {key: copy(item) for key, item in value.items()} if isinstance(value, Mapping) else value
        return copy(self.provenance)


@dataclass(frozen=True)
class _Source:
    path: Path
    data: bytes
    sha256: str
    identity: tuple[int, int, int, int, int]


def _read_source(path: Path, expected: str, guard: Callable[[], None]) -> _Source:
    guard()
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise PinnedImportError("immutable source SHA256 required")
    path = path.absolute()
    if path.resolve() != path:
        raise PinnedImportError("source path must be canonical without symbolic links")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_SOURCE_BYTES:
        raise PinnedImportError("bounded regular Python source required")
    with path.open("rb") as stream:
        data = stream.read(MAX_SOURCE_BYTES + 1)
    after = path.lstat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    digest = hashlib.sha256(data).hexdigest()
    if identity(before) != identity(after) or len(data) != before.st_size or digest != expected:
        raise PinnedImportError("source bytes or file identity changed")
    guard()
    return _Source(path, data, digest, identity(after))


def _verify_source(source: _Source, guard: Callable[[], None]) -> None:
    observed = _read_source(source.path, source.sha256, guard)
    if observed.identity != source.identity:
        raise PinnedImportError("source identity changed during import")


class _VerifiedSourceLoader(importlib.machinery.SourceFileLoader):
    """Compile the checked source bytes; never trust or write cached bytecode."""
    def __init__(self, name: str, source: _Source):
        super().__init__(name, str(source.path))
        self.source = source

    def get_code(self, fullname: str):
        if fullname != self.name:
            raise PinnedImportError("unexpected source loader module name")
        return self.source_to_code(self.source.data, str(self.source.path))


def _package(name: str) -> ModuleType:
    package = ModuleType(name)
    package.__package__ = name
    package.__loader__ = None
    package.__path__ = []
    package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    package.__spec__.origin = "hash-pinned standalone GC-LSTM namespace; package init bypassed"
    package.__pinned_standalone_graph__ = True
    return package


def _class(module: ModuleType, name: str, source: _Source, method: str | None = None) -> type:
    value = getattr(module, name, None)
    if (not isinstance(value, type) or value.__name__ != name or value.__qualname__ != name
            or value.__module__ != module.__name__ or Path(inspect.getfile(value)).absolute() != source.path):
        raise PinnedImportError("class source/module provenance mismatch: " + name)
    if method is not None:
        function = value.__dict__.get(method)
        if isinstance(function, property):
            function = function.fget
        if not inspect.isfunction(function) or Path(function.__code__.co_filename).absolute() != source.path:
            raise PinnedImportError("class method provenance mismatch: " + name)
    return value


def load_pinned_graph_modules(*, source_root: Path, installed_cell_path: Path,
                              cell_sha256: str, wrapper_sha256: Mapping[str, str],
                              guard: Callable[[], None]) -> PinnedGraphModules:
    """Load exact source files once in a fresh worker, constructing no models.

    Pins must come from the caller's immutable admitted configuration/manifest.
    Refuse every preexisting Temporal or private wrapper namespace, leaving
    those modules untouched. A second call also requires a fresh worker.
    No package initializer or alternate cell is executed. The caller retains
    mandatory reporter, dependency versions, seeding and numerical authority.
    """
    if not callable(guard):
        raise PinnedImportError("deadline/STOP guard required")
    guard()
    if not isinstance(sys.modules.get("torch"), ModuleType):
        raise PinnedImportError("Torch must already be imported inside initialized reporting")
    root = Path(source_root).absolute()
    cell_path = Path(installed_cell_path).absolute()
    if root.resolve() != root or not root.is_dir():
        raise PinnedImportError("canonical public source root required")
    if tuple(cell_path.parts[-4:]) != (TEMPORAL, "nn", "recurrent", "gc_lstm.py"):
        raise PinnedImportError("installed GC-LSTM distribution source path required")
    if set(wrapper_sha256) != set(PINNED_WRAPPER_SHA256):
        raise PinnedImportError("exact three public wrapper source identities required")
    sources = {"cell": _read_source(cell_path, cell_sha256, guard)}
    for suffix, relative in (("node", "model/node_emb.py"), ("recurrent", "model/gclstm.py"),
                             ("head", "model/link_predictor.py")):
        sources[suffix] = _read_source(root / relative, wrapper_sha256[relative], guard)
    for name in tuple(sys.modules):
        if name == TEMPORAL or name.startswith(TEMPORAL + ".") or name == WRAPPER_NAMESPACE or name.startswith(WRAPPER_NAMESPACE + "."):
            raise PinnedImportError("preexisting Temporal/wrapper namespace; fresh worker required")
    created: dict[str, ModuleType] = {}
    attributes: list[tuple[ModuleType, str, Any]] = []

    def publish(name: str, module: ModuleType) -> None:
        if name in sys.modules:
            raise PinnedImportError("namespace appeared during guarded import")
        sys.modules[name] = module
        created[name] = module

    def attach(parent: ModuleType, name: str, value: Any) -> None:
        if name in parent.__dict__:
            raise PinnedImportError("unexpected existing package attribute")
        setattr(parent, name, value)
        attributes.append((parent, name, value))

    def load(name: str, source: _Source) -> ModuleType:
        guard()
        _verify_source(source, guard)
        loader = _VerifiedSourceLoader(name, source)
        spec = importlib.util.spec_from_file_location(name, source.path, loader=loader)
        if spec is None:
            raise PinnedImportError("source module spec unavailable")
        module = importlib.util.module_from_spec(spec)
        publish(name, module)
        loader.exec_module(module)
        guard()
        if (sys.modules.get(name) is not module or module.__spec__ is not spec or module.__loader__ is not loader
                or spec.origin != str(source.path)
                or Path(module.__file__).absolute() != source.path or module.__package__ != name.rpartition(".")[0]):
            raise PinnedImportError("executed module provenance changed")
        _verify_source(source, guard)
        return module

    try:
        parents = {}
        for name in (TEMPORAL, TEMPORAL + ".nn", TEMPORAL + ".nn.recurrent", WRAPPER_NAMESPACE):
            parents[name] = _package(name)
            publish(name, parents[name])
        attach(parents[TEMPORAL], "nn", parents[TEMPORAL + ".nn"])
        attach(parents[TEMPORAL + ".nn"], "recurrent", parents[TEMPORAL + ".nn.recurrent"])
        cell = load(CELL_MODULE, sources["cell"])
        cell_type = _class(cell, "GCLSTM", sources["cell"], "forward")
        attach(parents[TEMPORAL + ".nn.recurrent"], "gc_lstm", cell)
        attach(parents[TEMPORAL + ".nn.recurrent"], "GCLSTM", cell_type)
        node = load(WRAPPER_NAMESPACE + ".node_emb", sources["node"])
        attach(parents[WRAPPER_NAMESPACE], "node_emb", node)
        _class(node, "NodeEmbeddingModel", sources["node"], "out_dimension")
        recurrent = load(WRAPPER_NAMESPACE + ".gclstm", sources["recurrent"])
        attach(parents[WRAPPER_NAMESPACE], "gclstm", recurrent)
        wrapper_type = _class(recurrent, "MultiLayerGCLSTM", sources["recurrent"], "forward")
        parameter_type = _class(recurrent, "GCLSTMParam", sources["recurrent"])
        if recurrent.GCLSTM is not cell_type:
            raise PinnedImportError("public wrapper did not bind the exact installed GC-LSTM")
        head = load(WRAPPER_NAMESPACE + ".link_predictor", sources["head"])
        attach(parents[WRAPPER_NAMESPACE], "link_predictor", head)
        head_type = _class(head, "LinkPredictor", sources["head"], "forward")
        for source in sources.values():
            _verify_source(source, guard)
        for name, module in created.items():
            if sys.modules.get(name) is not module:
                raise PinnedImportError("owned module identity changed during import")
        if any((name == TEMPORAL or name.startswith(TEMPORAL + ".") or name == WRAPPER_NAMESPACE or name.startswith(WRAPPER_NAMESPACE + "."))
               and name not in created for name in tuple(sys.modules)):
            raise PinnedImportError("unexpected foreign module appeared in isolated namespace")
        for module, name, source, method, expected in (
                (cell, "GCLSTM", sources["cell"], "forward", cell_type),
                (recurrent, "MultiLayerGCLSTM", sources["recurrent"], "forward", wrapper_type),
                (recurrent, "GCLSTMParam", sources["recurrent"], None, parameter_type),
                (head, "LinkPredictor", sources["head"], "forward", head_type)):
            if _class(module, name, source, method) is not expected:
                raise PinnedImportError("class identity changed during import")
        if recurrent.GCLSTM is not cell_type:
            raise PinnedImportError("public wrapper cell identity changed during import")
        guard()
        provenance = MappingProxyType({
            "import_scope": "standalone exact GC-LSTM plus public wrapper/node/head only",
            "Temporal_package_initializers_executed": False,
            "full_Temporal_package_API_verified": False,
            "native_math_or_source_modified": False,
            "source_release_hashes_match": cell_sha256 == PINNED_CELL_SHA256 and dict(wrapper_sha256) == dict(PINNED_WRAPPER_SHA256),
            "cell_module": CELL_MODULE, "wrapper_namespace": WRAPPER_NAMESPACE,
            "sources": MappingProxyType({name: MappingProxyType({"path": str(source.path), "sha256": source.sha256,
                "bytes": len(source.data)}) for name, source in sources.items()}),
            "classes": MappingProxyType({"cell": cell_type.__module__ + "." + cell_type.__qualname__,
                "wrapper": wrapper_type.__module__ + "." + wrapper_type.__qualname__,
                "parameter": parameter_type.__module__ + "." + parameter_type.__qualname__,
                "head": head_type.__module__ + "." + head_type.__qualname__}),
        })
        return PinnedGraphModules(cell, recurrent, head, node, cell_type, wrapper_type, parameter_type, head_type, provenance)
    except BaseException:
        # Never remove/overwrite an existing or concurrently replaced foreign module.
        for parent, name, value in reversed(attributes):
            if parent.__dict__.get(name) is value:
                delattr(parent, name)
        for name, module in reversed(tuple(created.items())):
            if sys.modules.get(name) is module:
                del sys.modules[name]
        raise
