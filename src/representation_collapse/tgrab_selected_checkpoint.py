"""Strict, data-only codec for the frozen T-GRAB selected checkpoint v1.

This module performs no model construction, tensor-library import, or object
deserialization. Trust must still be supplied by an external admission record;
validating a candidate bundle is an integrity check, not selection authority.
"""
from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import math
import os
import re
import stat
import struct
from pathlib import Path
from typing import Any, Mapping


SCHEMA_SHA256 = "db2baaa64f891533f5c2c36229dd250733f675a0a2045d0436fc93d3145f2acc"
CATALOG_SHA256 = "2dcaff589cdfd48f2d5426fabab3dae8f229dde7b2bff27ab7ed7d1815db4268"
# This is the canonical frozen reconstruction-v1 configuration identity, not a
# digest of the fitted recurrent/head payloads. Those bytes have separate hashes.
FROZEN_CANDIDATE_IDENTITY_SHA256 = "dea0e7684f52a0a53cba5072e77778d22995cb0b0e30f3352a6e349b89628092"
SCHEMA_PATH = Path("configs/graph/tgrab-selected-checkpoint-schema-v1.json")
CATALOG_PATH = Path("configs/graph/tgrab-selected-tensor-catalog-v1.json")
FILES = (
    "recurrent.f32le", "link-head.f32le", "validation-terminal.f32le",
    "validation.json", "provenance.json", "manifest.json",
)
_HEX = re.compile(r"^[0-9a-f]{64}$")


class CheckpointError(ValueError):
    """Input is malformed, inconsistent, unsafe, or outside the frozen v1."""


def _repo_root() -> Path:
    # The installed package is under src/representation_collapse.
    return Path(__file__).resolve().parents[2]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    try:
        raw = json.dumps(value, ensure_ascii=True, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("ascii") + b"\n"
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CheckpointError("value cannot be represented as canonical v1 JSON") from exc
    _check_json_limits(value)
    return raw


def _check_json_limits(value: Any, depth: int = 0) -> None:
    if isinstance(value, (dict, list)):
        if depth >= 8:
            raise CheckpointError("JSON nesting exceeds eight containers")
        if isinstance(value, dict):
            if len(value) > 64 or any(not isinstance(k, str) for k in value):
                raise CheckpointError("JSON object key count/type exceeds v1")
            for k, v in value.items():
                _check_text(k)
                _check_json_limits(v, depth + 1)
        else:
            for v in value:
                _check_json_limits(v, depth + 1)
    elif isinstance(value, str):
        _check_text(value)
    elif isinstance(value, float) and not math.isfinite(value):
        raise CheckpointError("nonfinite JSON number")
    elif isinstance(value, int) and not isinstance(value, bool) and len(str(abs(value))) > 10:
        raise CheckpointError("integer exceeds frozen token bound")


def _check_text(value: str) -> None:
    try:
        data = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise CheckpointError("non-ASCII JSON string") from exc
    if len(data) > 256:
        raise CheckpointError("JSON string exceeds 256 bytes")


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise CheckpointError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _parse_int(token: str) -> int:
    if len(token.lstrip("-")) > 10:
        raise CheckpointError("integer token exceeds ten digits")
    return int(token)


def _parse_float(token: str) -> float:
    if len(token) > 64:
        raise CheckpointError("number token exceeds 64 characters")
    number = float(token)
    if not math.isfinite(number):
        raise CheckpointError("nonfinite JSON number")
    return number


def _bad_constant(token: str) -> None:
    raise CheckpointError(f"invalid JSON constant {token}")


def parse_canonical_json(raw: bytes, *, max_bytes: int) -> Any:
    if type(raw) is not bytes or len(raw) > max_bytes:
        raise CheckpointError("JSON file is not bytes or exceeds its frozen maximum")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise CheckpointError("canonical JSON requires exactly one final newline")
    try:
        text = raw.decode("ascii")
        value = json.loads(text, object_pairs_hook=_pairs_no_duplicates,
                           parse_int=_parse_int, parse_float=_parse_float,
                           parse_constant=_bad_constant)
    except CheckpointError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CheckpointError("invalid ASCII JSON") from exc
    _check_json_limits(value)
    if _canonical(value) != raw:
        raise CheckpointError("JSON bytes are not canonical sorted compact v1")
    return value


def _schema_and_catalog() -> tuple[dict[str, Any], dict[str, Any]]:
    root = _repo_root()
    schema_raw = (root / SCHEMA_PATH).read_bytes()
    catalog_raw = (root / CATALOG_PATH).read_bytes()
    if _sha(schema_raw) != SCHEMA_SHA256 or _sha(catalog_raw) != CATALOG_SHA256:
        raise CheckpointError("frozen schema/catalog bytes differ from pinned v1")
    schema = json.loads(schema_raw)
    catalog = json.loads(catalog_raw)
    if schema.get("x-source-pins", {}).get(str(CATALOG_PATH)) != CATALOG_SHA256:
        raise CheckpointError("schema does not pin the expected tensor catalog")
    return schema, catalog


def _schema_validate(value: Any, spec: Mapping[str, Any], schema: Mapping[str, Any], path: str = "$", depth: int = 0) -> None:
    if depth > 16:
        raise CheckpointError("schema recursion exceeds safety bound")
    if "$ref" in spec:
        ref = spec["$ref"]
        if not ref.startswith("#/$defs/"):
            raise CheckpointError("unsupported external schema reference")
        name = ref.removeprefix("#/$defs/")
        _schema_validate(value, schema["$defs"][name], schema, path, depth + 1)
    if "oneOf" in spec:
        matches = 0
        for option in spec["oneOf"]:
            try:
                _schema_validate(value, option, schema, path, depth + 1)
                matches += 1
            except CheckpointError:
                pass
        if matches != 1:
            raise CheckpointError(f"{path}: expected exactly one schema alternative")
    expected = spec.get("type")
    if expected:
        good = {
            "object": lambda x: isinstance(x, dict),
            "array": lambda x: isinstance(x, list),
            "string": lambda x: isinstance(x, str),
            "boolean": lambda x: type(x) is bool,
            "integer": lambda x: type(x) is int,
            "number": lambda x: type(x) in (int, float) and type(x) is not bool and math.isfinite(x),
            "null": lambda x: x is None,
        }.get(expected)
        if good is None or not good(value):
            raise CheckpointError(f"{path}: expected {expected}")
    if "const" in spec and value != spec["const"]:
        raise CheckpointError(f"{path}: constant mismatch")
    if "enum" in spec and value not in spec["enum"]:
        raise CheckpointError(f"{path}: enum mismatch")
    if type(value) in (int, float) and not isinstance(value, bool):
        if "minimum" in spec and value < spec["minimum"]:
            raise CheckpointError(f"{path}: below minimum")
        if "maximum" in spec and value > spec["maximum"]:
            raise CheckpointError(f"{path}: above maximum")
    if isinstance(value, str):
        if len(value.encode("ascii", "strict")) > spec.get("maxLength", 256):
            raise CheckpointError(f"{path}: string too long")
        if "pattern" in spec and re.fullmatch(spec["pattern"], value) is None:
            raise CheckpointError(f"{path}: string pattern mismatch")
        if spec.get("format") == "date-time":
            try:
                dt = _datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
                if dt.tzinfo is None or dt.utcoffset() != _datetime.timedelta(0):
                    raise ValueError
            except ValueError as exc:
                raise CheckpointError(f"{path}: invalid UTC timestamp") from exc
    if isinstance(value, dict):
        props = spec.get("properties", {})
        if any(k not in value for k in spec.get("required", [])):
            raise CheckpointError(f"{path}: required key missing")
        if spec.get("additionalProperties") is False and any(k not in props for k in value):
            raise CheckpointError(f"{path}: unknown key")
        for key, child in value.items():
            if key in props:
                _schema_validate(child, props[key], schema, f"{path}.{key}", depth + 1)
    if isinstance(value, list):
        if len(value) < spec.get("minItems", 0) or len(value) > spec.get("maxItems", 1_000_000):
            raise CheckpointError(f"{path}: array length outside schema")
        if spec.get("items") is False and len(value) != len(spec.get("prefixItems", [])):
            raise CheckpointError(f"{path}: array length differs from prefix schema")
        for i, child in enumerate(value):
            subspec = spec.get("prefixItems", [])[i] if i < len(spec.get("prefixItems", [])) else spec.get("items")
            if isinstance(subspec, dict):
                _schema_validate(child, subspec, schema, f"{path}[{i}]", depth + 1)


def validate_metadata(raw: bytes, *, expected_kind: str | None = None, max_bytes: int | None = None) -> dict[str, Any]:
    schema, _ = _schema_and_catalog()
    if max_bytes is None:
        max_bytes = schema["x-contract"]["six_bundle_files"]["manifest.json"]["max_bytes"]
    value = parse_canonical_json(raw, max_bytes=max_bytes)
    if not isinstance(value, dict):
        raise CheckpointError("metadata root must be an object")
    if expected_kind is not None and value.get("kind") != expected_kind:
        raise CheckpointError("metadata kind mismatch")
    errors = 0
    match = None
    for option in schema["oneOf"]:
        try:
            _schema_validate(value, option, schema)
            errors += 1
            match = option
        except CheckpointError:
            pass
    if errors != 1:
        raise CheckpointError("metadata does not match exactly one closed v1 artifact schema")
    return value


def _finite_f32(payload: bytes) -> None:
    if len(payload) % 4:
        raise CheckpointError("float32 payload length is not divisible by four")
    for (value,) in struct.iter_unpack("<f", payload):
        if not math.isfinite(value):
            raise CheckpointError("tensor contains NaN or infinity")


def tensor_semantic_sha256(component: str, name: str, shape: list[int], payload: bytes) -> str:
    if component not in ("recurrent", "link_head", "terminal_state"):
        raise CheckpointError("unknown tensor component")
    if not isinstance(name, str) or not isinstance(shape, list):
        raise CheckpointError("invalid tensor name/shape")
    _finite_f32(payload)
    metadata = {"component": component, "name": name, "dtype": "float32_le",
                "shape": shape, "order": "C", "nbytes": len(payload)}
    return _sha(b"tgrab-tensor-v1\0" + _canonical(metadata) + payload)


def encode_tensor_bytes(payload: bytes, shape: list[int]) -> bytes:
    """Validate and retain an already-serialized C-order float32-le tensor."""
    if type(payload) is not bytes or not isinstance(shape, list) or not shape:
        raise CheckpointError("tensor must be exact bytes with a nonempty shape")
    if any(type(d) is not int or d < 1 or d > 256 for d in shape):
        raise CheckpointError("tensor dimensions violate the frozen rank/dimension bounds")
    if len(shape) not in (1, 2) or math.prod(shape) * 4 != len(payload):
        raise CheckpointError("tensor byte length does not match frozen float32 shape")
    _finite_f32(payload)
    return payload


def decode_tensor_bytes(payload: bytes, shape: list[int]) -> bytes:
    """Validate the data-only tensor representation without changing any bit."""
    return encode_tensor_bytes(payload, shape)


def component_semantic_sha256(descriptors: list[dict[str, Any]]) -> str:
    pairs = [[d["name"], d["semantic_sha256"]] for d in descriptors]
    return _sha(b"tgrab-component-v1\0" + _canonical(pairs))


def _verify_component(component: str, descriptor: dict[str, Any], payload: bytes, catalog: dict[str, Any]) -> None:
    expected = catalog["components"][component]
    if len(payload) != expected["nbytes"] or descriptor.get("filename") != expected["payload_filename"]:
        raise CheckpointError(f"{component}: file size/name differs from catalog")
    if descriptor.get("bytes") != len(payload) or descriptor.get("sha256") != _sha(payload):
        raise CheckpointError(f"{component}: file digest/length mismatch")
    tensors = descriptor.get("tensors")
    if not isinstance(tensors, list) or len(tensors) != expected["tensor_count"]:
        raise CheckpointError(f"{component}: tensor descriptor count mismatch")
    cursor = 0
    for item, frozen in zip(tensors, expected["tensors"], strict=True):
        keys = {"name", "dtype", "shape", "order", "offset", "nbytes", "data_sha256", "semantic_sha256"}
        if set(item) != keys:
            raise CheckpointError("tensor descriptor has unknown/missing fields")
        if (item["name"], item["dtype"], item["shape"], item["order"], item["offset"], item["nbytes"]) != (
            frozen["name"], "float32_le", frozen["shape"], "C", frozen["payload_offset"], frozen["nbytes"]
        ):
            raise CheckpointError(f"{component}: descriptor differs from frozen tensor catalog")
        if item["offset"] != cursor or item["nbytes"] != math.prod(item["shape"]) * 4:
            raise CheckpointError("tensor offsets/shape byte counts are not contiguous")
        end = cursor + item["nbytes"]
        chunk = payload[cursor:end]
        _finite_f32(chunk)
        if item["data_sha256"] != _sha(chunk):
            raise CheckpointError("tensor data hash mismatch")
        if item["semantic_sha256"] != tensor_semantic_sha256(component, item["name"], item["shape"], chunk):
            raise CheckpointError("tensor semantic hash mismatch")
        cursor = end
    if cursor != len(payload):
        raise CheckpointError("tensor descriptors leave trailing bytes")


def _read_stable_at(dir_fd: int, name: str, limit: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(name, flags, dir_fd=dir_fd)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
            raise CheckpointError(f"{name}: not a bounded single-link regular file")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(fd)
        path_st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        identity = lambda st: (st.st_dev, st.st_ino, st.st_mode, st.st_nlink, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if identity(before) != identity(after) or identity(after) != identity(path_st):
            raise CheckpointError(f"{name}: replaced or modified while reading")
        if len(data) != after.st_size or len(data) > limit:
            raise CheckpointError(f"{name}: length changed or exceeds bound")
        return data
    finally:
        os.close(fd)


def _open_bundle_directory(path: Path) -> int:
    if not path.is_absolute():
        path = Path.cwd() / path
    if ".." in path.parts:
        raise CheckpointError("bundle path traversal is forbidden")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path.anchor or "/", flags)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise CheckpointError("bundle path contains a symlink or non-directory component")


def read_candidate(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and verify a complete immutable six-file candidate bundle."""
    schema, catalog = _schema_and_catalog()
    root = Path(path)
    if not root.is_absolute():
        root = Path.cwd() / root
    if root.name in ("", ".", "..") or ".." in root.parts:
        raise CheckpointError("invalid bundle directory path")
    pfd = _open_bundle_directory(root.parent)
    entry_before = os.stat(root.name, dir_fd=pfd, follow_symlinks=False)
    if not stat.S_ISDIR(entry_before.st_mode):
        os.close(pfd)
        raise CheckpointError("bundle entry is not a directory")
    fd = os.open(root.name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=pfd)
    try:
        opened = os.fstat(fd)
        if (entry_before.st_dev, entry_before.st_ino) != (opened.st_dev, opened.st_ino):
            raise CheckpointError("bundle directory replaced during open")
        entries = os.listdir(fd)
        if set(entries) != set(FILES) or len(entries) != len(FILES):
            raise CheckpointError("bundle must contain exactly the six frozen files")
        maxes = schema["x-contract"]["six_bundle_files"]
        payloads = {}
        for name in FILES:
            lim = maxes[name]["max_bytes"] if isinstance(maxes[name], dict) else maxes[name]
            payloads[name] = _read_stable_at(fd, name, lim)
        if sum(map(len, payloads.values())) > schema["x-contract"]["bundle_max_bytes"]:
            raise CheckpointError("bundle exceeds frozen aggregate size limit")
        manifest = validate_metadata(payloads["manifest.json"], expected_kind="tgrab_selected_inference_v1")
        validation = validate_metadata(payloads["validation.json"], expected_kind="tgrab_complete_validation_rows_v1",
                                       max_bytes=maxes["validation.json"]["max_bytes"])
        provenance = validate_metadata(payloads["provenance.json"], expected_kind="tgrab_selected_inference_provenance_v1",
                                       max_bytes=maxes["provenance.json"]["max_bytes"])
        _verify_component("recurrent", manifest["components"]["recurrent"], payloads["recurrent.f32le"], catalog)
        _verify_component("link_head", manifest["components"]["link_head"], payloads["link-head.f32le"], catalog)
        terminal = manifest["terminal_state"]
        terminal_bytes = payloads["validation-terminal.f32le"]
        if len(terminal_bytes) != 102400 or terminal.get("filename") != "validation-terminal.f32le" or terminal.get("bytes") != 102400 or terminal.get("sha256") != _sha(terminal_bytes):
            raise CheckpointError("terminal state size/name/hash mismatch")
        if len(terminal.get("tensors", [])) != 2:
            raise CheckpointError("terminal state requires hidden and cell descriptors")
        for item, name, off in zip(terminal["tensors"], ("hidden", "cell"), (0, 51200), strict=True):
            raw = terminal_bytes[off:off + 51200]
            if set(item) != {"name", "dtype", "shape", "order", "offset", "nbytes", "data_sha256", "semantic_sha256"}:
                raise CheckpointError("terminal tensor descriptor field set mismatch")
            if (item["name"], item["dtype"], item["shape"], item["order"], item["offset"], item["nbytes"]) != (name, "float32_le", [1, 100, 128], "C", off, 51200):
                raise CheckpointError("terminal hidden/cell descriptor mismatch")
            _finite_f32(raw)
            if item["data_sha256"] != _sha(raw) or item["semantic_sha256"] != tensor_semantic_sha256("terminal_state", name, [1, 100, 128], raw):
                raise CheckpointError("terminal tensor digest mismatch")
        for meta_name, obj in (("validation.json", validation), ("provenance.json", provenance)):
            section = manifest["validation"] if meta_name == "validation.json" else manifest["provenance"]
            if section["filename"] != meta_name or section["bytes"] != len(payloads[meta_name]) or section["sha256"] != _sha(payloads[meta_name]):
                raise CheckpointError(f"manifest {meta_name} digest/length mismatch")
        _validate_validation_score(manifest, validation)
        _validate_cross_identities(manifest, validation, provenance)
        entry_after = os.stat(root.name, dir_fd=pfd, follow_symlinks=False)
        if (entry_after.st_dev, entry_after.st_ino) != (opened.st_dev, opened.st_ino):
            raise CheckpointError("bundle directory replaced during verification")
        return {"manifest": manifest, "validation": validation, "provenance": provenance,
                "recurrent": payloads["recurrent.f32le"], "link_head": payloads["link-head.f32le"],
                "terminal_state": terminal_bytes, "files_sha256": {k: _sha(v) for k, v in payloads.items()}}
    finally:
        os.close(fd)
        os.close(pfd)


def _validate_validation_score(manifest: dict[str, Any], validation: dict[str, Any]) -> None:
    rows = validation.get("rows")
    if not isinstance(rows, list) or len(rows) != 256:
        raise CheckpointError("validation must contain exactly 256 rows")
    fs = []
    for row in rows:
        tp, fp, fn, tn = (row[k] for k in ("tp", "fp", "fn", "tn"))
        pairs = row["pairs"]
        if pairs != 9900 or tp + fp + fn + tn != pairs:
            raise CheckpointError("validation confusion counts do not sum to 9900")
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        # Match graph_reconstruction.snapshot_metrics operation order exactly.
        f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        a = (tp + tn) / pairs
        if (row["precision"], row["recall"], row["f1"], row["accuracy"]) != (p, r, f, a):
            raise CheckpointError("validation metrics differ from frozen zero-division formulas")
        fs.append(f)
    mean = math.fsum(fs) / 256
    if validation["time_mean_f1"] != mean or manifest["validation"]["time_mean_f1"] != mean:
        raise CheckpointError("validation time-mean F1 mismatch")
    if validation["epoch"] != manifest["epoch"]:
        raise CheckpointError("candidate epoch mismatch across validation metadata")


def _validate_cross_identities(manifest: dict[str, Any], validation: dict[str, Any], provenance: dict[str, Any]) -> None:
    ids = manifest["identities"]
    expected = {
        "schema_sha256": SCHEMA_SHA256,
        "tensor_catalog_sha256": CATALOG_SHA256,
        "admission_sha256": provenance["admission_sha256"],
        "executed_config_sha256": provenance["executed_config_sha256"],
        "reconstruction_config_sha256": provenance["reconstruction_config_sha256"],
        "reporting_identity_sha256": provenance["reporting_identity_sha256"],
    }
    for key, value in expected.items():
        if ids[key] != value:
            raise CheckpointError(f"cross-file identity mismatch: {key}")
    epoch = manifest["epoch"]
    counters = provenance["counters"]
    if validation["epoch"] != epoch or counters["complete_epochs"] != epoch or counters["validation_epochs"] != epoch:
        raise CheckpointError("candidate epoch differs across manifest, validation and provenance")
    updates = epoch * 2559
    if counters["completed_updates"] != updates or counters["attempted_updates"] != updates:
        raise CheckpointError("candidate update counters do not match complete-epoch protocol")
    terminal = manifest["terminal_state"]
    for key in ("recurrent_sha256", "link_head_sha256"):
        component = "recurrent" if key == "recurrent_sha256" else "link_head"
        if terminal[key] != manifest["components"][component]["sha256"]:
            raise CheckpointError("terminal state is not bound to candidate component bytes")
    if manifest["validation"]["row_count"] != len(validation["rows"]):
        raise CheckpointError("validation row count differs from manifest")


def write_candidate(path: str | os.PathLike[str], files: Mapping[str, bytes]) -> dict[str, Any]:
    """Publish a complete candidate at a new directory, manifest written last.

    The caller must supply all six already-canonical/encoded files. Failed writes
    leave an incomplete, non-reusable directory and are never resumed/overwritten.
    """
    if set(files) != set(FILES) or any(type(v) is not bytes for v in files.values()):
        raise CheckpointError("writer requires exactly six byte payloads")
    target = Path(path)
    if not target.is_absolute():
        target = Path.cwd() / target
    parent = target.parent
    pfd = _open_bundle_directory(parent)
    name = target.name
    if name in ("", ".", "..") or "/" in name:
        os.close(pfd)
        raise CheckpointError("invalid candidate directory basename")
    dfd = None
    try:
        os.mkdir(name, 0o700, dir_fd=pfd)
        dfd = os.open(name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0), dir_fd=pfd)
        created_identity = os.fstat(dfd)
        maxes = _schema_and_catalog()[0]["x-contract"]["six_bundle_files"]
        order = [n for n in FILES if n != "manifest.json"] + ["manifest.json"]
        try:
            for filename in order:
                data = files[filename]
                lim = maxes[filename]["max_bytes"] if isinstance(maxes[filename], dict) else maxes[filename]
                if len(data) > lim:
                    raise CheckpointError(f"{filename} exceeds frozen maximum")
                if filename.endswith(".json"):
                    parse_canonical_json(data, max_bytes=lim)
                    if filename == "manifest.json":
                        validate_metadata(data, expected_kind="tgrab_selected_inference_v1")
                fd = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=dfd)
                try:
                    view = memoryview(data)
                    while view:
                        count = os.write(fd, view)
                        if count <= 0:
                            raise CheckpointError("short write while publishing bundle")
                        view = view[count:]
                    os.fsync(fd)
                finally:
                    os.close(fd)
            os.fsync(dfd)
            os.fsync(pfd)
        finally:
            os.close(dfd)
            dfd = None
        current = os.stat(name, dir_fd=pfd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (created_identity.st_dev, created_identity.st_ino):
            raise CheckpointError("published bundle directory replaced before verification")
        verified = read_candidate(target)
        current = os.stat(name, dir_fd=pfd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (created_identity.st_dev, created_identity.st_ino):
            raise CheckpointError("published bundle directory replaced during verification")
        return verified
    finally:
        if dfd is not None:
            os.close(dfd)
        os.close(pfd)

