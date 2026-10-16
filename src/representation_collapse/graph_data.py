"""Strict prospective primary T-GRAB conversion and cooperative sealed split API.

Importing this module uses only the standard library. No CLI, model, reporting,
training, or implicit data generation is provided. See the frozen protocol.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat


class DataContractError(ValueError):
    pass


HEADER = ("src", "dst", "t", "edge_feat", "train_mask", "val_mask", "test_mask", "test_inductive_mask")
RAW_SHA256 = "5a19f9e7949668473bc61cb59b6215f25f093ea0ee2d45b5e143338ded5523e3"
RAW_BYTES = 6907326
CONFIG_SHA256 = "22bef6bf781c5bfb5493dd5e03169ea0e658b2463ccaa9ba09189af454508f1f"
ORIGIN = {"repository": "Gilestel/T-GRAB", "revision": "603388f332c7756f90bc1702b26053c2bc834654",
          "path": "periodicity/periodicity_det_(2, 32)_100N_40tr_4vl_4ts_pER(0.01).csv"}
MAX_FILE = 64 * 1024 * 1024
MAX_MANIFEST = 2 * 1024 * 1024
MAX_RAW = 16 * 1024 * 1024
MAX_ROWS = 500000
_INTEGER = re.compile(r"(?:0|[1-9][0-9]*)(?:\.0+)?\Z")
_UNIT = re.compile(r"1(?:\.0+)?\Z")
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_REV = re.compile(r"[0-9a-f]{40}\Z")
_SPLITS = ("train", "validation", "test")
_TOKEN = object()


def _fail(message):
    raise DataContractError(message)


def _int(value, name, low=0, high=1000000):
    if type(value) is not int or not low <= value <= high:
        _fail(f"invalid {name}")
    return value


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical_json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _keys(obj, expected):
    if type(obj) is not dict or set(obj) != set(expected):
        _fail("unexpected JSON keys")


def _json(data):
    def pairs(entries):
        result = {}
        for key, value in entries:
            if key in result:
                _fail("duplicate JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(data, object_pairs_hook=pairs,
                           parse_constant=lambda _: _fail("nonfinite JSON value"))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise DataContractError("invalid bounded JSON") from exc
    if canonical_json(value) != data:
        _fail("JSON is not canonical")
    return value


def _check_guard(guard):
    if not callable(guard):
        _fail("explicit physical/stop guard required")
    guard()


def _safe_path(path):
    if ".." in Path(path).parts:
        _fail("path traversal refused")
    path = Path(os.path.abspath(os.fspath(path)))
    # Do not resolve a symlink into apparently harmless input.
    for ancestor in reversed((path, *path.parents)):
        try:
            mode = ancestor.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            _fail("symlink path component refused")
    return path


def _read(path, limit, guard):
    _check_guard(guard)
    path = _safe_path(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise DataContractError("cannot open regular input") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
            _fail("input type, hardlink or size bound")
        chunks = []
        size = 0
        while True:
            _check_guard(guard)
            chunk = os.read(fd, min(65536, limit + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > limit:
                _fail("input grew beyond bound")
        after = os.fstat(fd)
        at_path = path.lstat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if identity(before) != identity(after) or identity(after) != identity(at_path) or size != before.st_size:
            _fail("input changed during read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _write_new(path, data):
    path = _safe_path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class DatasetSpec:
    nodes: int = 100
    block: int = 32
    train_periods: int = 40
    validation_periods: int = 4
    test_periods: int = 4

    def __post_init__(self):
        _int(self.nodes, "nodes", 3, 100)
        _int(self.block, "block", 1, 32)
        for name, maximum in (("train_periods", 40), ("validation_periods", 4), ("test_periods", 4)):
            _int(getattr(self, name), name, 1, maximum)

    @property
    def intervals(self):
        a = self.train_periods * self.block * 2
        b = a + self.validation_periods * self.block * 2
        return {"train": [0, a], "validation": [a, b], "test": [b, b + self.test_periods * self.block * 2]}

    def as_dict(self):
        return {name: getattr(self, name) for name in ("nodes", "block", "train_periods", "validation_periods", "test_periods")}


PRIMARY_SPEC = DatasetSpec()


def _spec(value):
    _keys(value, PRIMARY_SPEC.as_dict())
    return DatasetSpec(**value)


def _semantic(snapshot):
    # Explicitly encodes implicit X and weights, including their dtype/order.
    return _digest(b"tgrab-snapshot-semantics-v1\0" + canonical_json({
        **snapshot, "node_order": "ascending_integer_ids", "features": "float32_identity",
        "weights": "float32_unit_aligned_with_pairs"}))


def _validate_pairs(pairs, nodes):
    if type(pairs) is not list or not pairs or len(pairs) > nodes * (nodes - 1):
        _fail("invalid or empty positive pairs")
    previous = None
    edges = set()
    for pair in pairs:
        if type(pair) is not list or len(pair) != 2:
            _fail("invalid pair")
        u, v = pair
        _int(u, "source", 0, nodes - 1); _int(v, "destination", 0, nodes - 1)
        edge = (u, v)
        if u == v or (previous is not None and edge <= previous):
            _fail("pairs must be unique sorted offdiagonal edges")
        previous = edge
        edges.add(edge)
    for u, v in edges:
        if (v, u) not in edges:
            _fail("missing reverse edge")
    for u in {u for u, _ in edges}:
        if not any(v != u and (u, v) not in edges for v in range(nodes)):
            _fail("positive source has no feasible negative")
    return edges


def _lex_int(text):
    if len(text) > 32 or not _INTEGER.fullmatch(text):
        _fail("noncanonical integral CSV token")
    return int(text.split(".")[0])


def _lex_bool(text):
    if text not in ("True", "False", "1", "0"):
        _fail("invalid boolean CSV token")
    return text in ("True", "1")


def convert_primary_csv(raw_path, output_dir, *, expected_raw_sha256, expected_raw_bytes,
                        config_path, converter_revision, guard, spec=PRIMARY_SPEC, fixture_only=False):
    """Convert once into a fresh directory; failures retain files, no manifest.

    This executes a deterministic conversion only when explicitly called by an
    admitted supervisor. Fixture mode cannot produce a primary-kind manifest.
    """
    if type(spec) is not DatasetSpec or type(fixture_only) is not bool:
        _fail("typed specification and fixture mode required")
    if not fixture_only and spec != PRIMARY_SPEC:
        _fail("only frozen primary spec permitted outside fixture mode")
    if not isinstance(expected_raw_sha256, str) or not _HEX.fullmatch(expected_raw_sha256):
        _fail("invalid raw SHA")
    _int(expected_raw_bytes, "raw size", 1, MAX_RAW)
    if not fixture_only and (expected_raw_sha256 != RAW_SHA256 or expected_raw_bytes != RAW_BYTES):
        _fail("primary raw identity mismatch")
    if not isinstance(converter_revision, str) or not _REV.fullmatch(converter_revision):
        _fail("converter source revision required")
    _check_guard(guard)
    output = _safe_path(output_dir)
    try:
        output.mkdir(mode=0o700, parents=False, exist_ok=False)
    except OSError as exc:
        raise DataContractError("fresh output directory required; existing output preserved") from exc
    counts = None
    marker_ts = set()
    try:
        config_data = _read(config_path, MAX_MANIFEST, guard)
        if _digest(config_data) != CONFIG_SHA256:
            _fail("frozen reconstruction config changed")
        raw = _read(raw_path, expected_raw_bytes, guard)
        if len(raw) != expected_raw_bytes or _digest(raw) != expected_raw_sha256:
            _fail("raw bytes identity mismatch")
        try:
            text = raw.decode("ascii")
        except UnicodeError as exc:
            raise DataContractError("ASCII CSV required") from exc
        rows = csv.reader(io.StringIO(text, newline=""), strict=True)
        if tuple(next(rows, ())) != HEADER:
            _fail("primary CSV header mismatch")
        total = spec.intervals["test"][1]
        by_t = [dict() for _ in range(total)]
        counts = {"raw_rows": 0, "identical_duplicate_rows": 0, "source_marker_rows": 0,
                  "unique_offdiagonal_edges": 0}
        for row in rows:
            _check_guard(guard)
            counts["raw_rows"] += 1
            if counts["raw_rows"] > MAX_ROWS or len(row) != len(HEADER) or any(len(v) > 32 for v in row):
                _fail("CSV row/count/field bound")
            u, v, t = map(_lex_int, row[:3])
            if u >= spec.nodes or v >= spec.nodes or t >= total:
                _fail("node or timestamp outside frozen bounds")
            if not _UNIT.fullmatch(row[3]):
                _fail("candidate requires exact unit edge weights")
            masks = tuple(map(_lex_bool, row[4:]))
            expected_masks = tuple(spec.intervals[s][0] <= t < spec.intervals[s][1] for s in _SPLITS) + (False,)
            if masks != expected_masks:
                _fail("row masks do not match exact chronological split")
            if u == v:
                if u != 0:
                    _fail("only node-zero source marker is identified")
                counts["source_marker_rows"] += 1
                marker_ts.add(t)
            pair = (u, v)
            if pair in by_t[t]:
                counts["identical_duplicate_rows"] += 1
            else:
                by_t[t][pair] = 1
        snapshots = []
        patterns = []
        for t, pairs in enumerate(by_t):
            _check_guard(guard)
            if not pairs:
                _fail("missing timestamp; no implicit synthesis")
            if t in marker_ts:
                if set(pairs) != {(0, 0)}:
                    _fail("source empty marker mixed with offdiagonal edges")
                # Timestamp retained, source marker resolved, candidate rejected.
                pairs = {}
                _fail(f"empty offdiagonal candidate at timestamp {t} after source marker resolution")
            ordered = [list(p) for p in sorted(pairs)]
            _validate_pairs(ordered, spec.nodes)
            pattern = 0 if t % (spec.block * 2) < spec.block else 1
            if t in (0, spec.block):
                patterns.append(ordered)
            if ordered != patterns[pattern]:
                _fail("not exact repeated two-block pattern ordering")
            snapshots.append({"timestamp": t, "nodes": spec.nodes, "positive_pairs": ordered})
            counts["unique_offdiagonal_edges"] += len(ordered)
        if patterns[0] == patterns[1]:
            _fail("two distinct patterns required")
        split_info = {}
        semantic_hashes = [_semantic(s) for s in snapshots]
        for split, (start, end) in spec.intervals.items():
            _check_guard(guard)
            payload = canonical_json({"kind": "tgrab_snapshot_split_v1", "split": split,
                                      "snapshots": snapshots[start:end]})
            if len(payload) > MAX_FILE:
                _fail("split size bound")
            filename = split + ".json"
            _write_new(output / filename, payload)
            split_info[split] = {"filename": filename, "sha256": _digest(payload), "bytes": len(payload),
                                 "interval": [start, end], "semantic_sha256": _digest(
                                     b"tgrab-split-semantics-v1\0" + canonical_json(semantic_hashes[start:end]))}
        converter = _read(Path(__file__), MAX_MANIFEST, guard)
        manifest = {"kind": "tgrab_primary_converted_v1" if not fixture_only else "tgrab_fixture_converted_v1",
                    "schema_version": 1, "spec": spec.as_dict(), "origin": ORIGIN if not fixture_only else {"fixture_only": True},
                    "raw": {"sha256": expected_raw_sha256, "bytes": expected_raw_bytes},
                    "converter": {"revision": converter_revision, "sha256": _digest(converter)},
                    "config_sha256": CONFIG_SHA256, "generation_performed": False,
                    "csv_policy": "ascii_fixed_header_integral_digits_optional_zero_fraction_bool_True_False_1_0_unit_1_optional_zero_fraction_v1",
                    "features": "float32_identity", "weights": "float32_unit_aligned_with_pairs",
                    "edge_order": "lexicographic_source_destination", "node_order": "ascending_integer_ids",
                    "marker_policy": "only_source_node_zero_empty_marker_removed_timestamp_retained_then_empty_candidate_rejected",
                    "counts": counts, "pattern_pairs": patterns, "snapshot_semantic_sha256": semantic_hashes,
                    "splits": split_info}
        data = canonical_json(manifest)
        if len(data) > MAX_MANIFEST:
            _fail("manifest size bound")
        _check_guard(guard)
        _write_new(output / "manifest.pending.json", data)
        # Atomic no-overwrite publication. No reader accepts pending files.
        os.link(output / "manifest.pending.json", output / "manifest.json", follow_symlinks=False)
        os.unlink(output / "manifest.pending.json")
        directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return output / "manifest.json"
    except BaseException as exc:
        if not (output / "manifest.json").exists():
            try:
                _write_new(output / "failure.json", canonical_json({"kind": "conversion_failed_v1", "error_type": type(exc).__name__,
                                                                     "message": str(exc)[:512], "partial_counts": counts,
                                                                     "source_marker_timestamps": sorted(marker_ts)}))
            except OSError:
                pass
        raise


@dataclass(frozen=True, init=False)
class VerifiedDatasetManifest:
    path: Path
    sha256: str
    _bytes: bytes
    _token: object

    def __init__(self, path, sha256, data, token):
        if token is not _TOKEN:
            _fail("use read_manifest")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "sha256", sha256)
        object.__setattr__(self, "_bytes", data)
        object.__setattr__(self, "_token", token)

    @property
    def metadata(self):
        return _json(self._bytes)


def read_manifest(path, expected_sha256, guard):
    if not isinstance(expected_sha256, str) or not _HEX.fullmatch(expected_sha256):
        _fail("expected manifest hash required")
    path = _safe_path(path)
    if path.name != "manifest.json":
        _fail("only published manifest accepted")
    data = _read(path, MAX_MANIFEST, guard)
    if _digest(data) != expected_sha256:
        _fail("manifest SHA mismatch")
    m = _json(data)
    _keys(m, ("kind", "schema_version", "spec", "origin", "raw", "converter", "config_sha256", "generation_performed",
              "csv_policy", "features", "weights", "edge_order", "node_order", "marker_policy", "counts",
              "pattern_pairs", "snapshot_semantic_sha256", "splits"))
    if m["kind"] not in ("tgrab_primary_converted_v1", "tgrab_fixture_converted_v1") or type(m["schema_version"]) is not int or m["schema_version"] != 1:
        _fail("manifest kind/schema")
    spec = _spec(m["spec"])
    _keys(m["raw"], ("sha256", "bytes")); _keys(m["converter"], ("revision", "sha256"))
    if not isinstance(m["raw"]["sha256"], str) or not _HEX.fullmatch(m["raw"]["sha256"]):
        _fail("raw hash")
    _int(m["raw"]["bytes"], "raw bytes", 1, MAX_RAW)
    if not isinstance(m["converter"]["revision"], str) or not _REV.fullmatch(m["converter"]["revision"]) or not isinstance(m["converter"]["sha256"], str) or not _HEX.fullmatch(m["converter"]["sha256"]):
        _fail("converter identity")
    if m["kind"] == "tgrab_primary_converted_v1":
        if spec != PRIMARY_SPEC or canonical_json(m["origin"]) != canonical_json(ORIGIN) or canonical_json(m["raw"]) != canonical_json({"sha256": RAW_SHA256, "bytes": RAW_BYTES}):
            _fail("primary identity")
    elif canonical_json(m["origin"]) != canonical_json({"fixture_only": True}):
        _fail("fixture origin")
    expected_policy = {"config_sha256": CONFIG_SHA256, "generation_performed": False,
                       "csv_policy": "ascii_fixed_header_integral_digits_optional_zero_fraction_bool_True_False_1_0_unit_1_optional_zero_fraction_v1",
                       "features": "float32_identity", "weights": "float32_unit_aligned_with_pairs",
                       "edge_order": "lexicographic_source_destination", "node_order": "ascending_integer_ids",
                       "marker_policy": "only_source_node_zero_empty_marker_removed_timestamp_retained_then_empty_candidate_rejected"}
    for key, value in expected_policy.items():
        if type(m[key]) is not type(value) or m[key] != value:
            _fail("manifest conversion policy mismatch")
    _keys(m["counts"], ("raw_rows", "identical_duplicate_rows", "source_marker_rows", "unique_offdiagonal_edges"))
    for name, value in m["counts"].items():
        _int(value, name, 0, MAX_ROWS)
    c = m["counts"]
    if c["source_marker_rows"] != 0 or c["raw_rows"] != c["identical_duplicate_rows"] + c["unique_offdiagonal_edges"]:
        _fail("inconsistent successful conversion counts")
    patterns = m["pattern_pairs"]
    if type(patterns) is not list or len(patterns) != 2 or patterns[0] == patterns[1]:
        _fail("distinct pattern pair sets required")
    for pattern in patterns:
        _validate_pairs(pattern, spec.nodes)
    expected_unique = (len(patterns[0]) + len(patterns[1])) * spec.block * sum((spec.train_periods, spec.validation_periods, spec.test_periods))
    if c["unique_offdiagonal_edges"] != expected_unique:
        _fail("pattern edge counts mismatch")
    hashes = m["snapshot_semantic_sha256"]
    if type(hashes) is not list or len(hashes) != spec.intervals["test"][1]:
        _fail("semantic snapshot count")
    for t, digest in enumerate(hashes):
        expected = _semantic({"timestamp": t, "nodes": spec.nodes, "positive_pairs": patterns[(t // spec.block) % 2]})
        if digest != expected:
            _fail("semantic snapshot hash mismatch")
    _keys(m["splits"], _SPLITS)
    for split, interval in spec.intervals.items():
        record = m["splits"][split]
        _keys(record, ("filename", "sha256", "bytes", "interval", "semantic_sha256"))
        if record["filename"] != split + ".json" or type(record["interval"]) is not list or any(type(v) is not int for v in record["interval"]) or record["interval"] != interval:
            _fail("split path or interval mismatch")
        _int(record["bytes"], "split bytes", 1, MAX_FILE)
        if not isinstance(record["sha256"], str) or not _HEX.fullmatch(record["sha256"]):
            _fail("split hash")
        if record["semantic_sha256"] != _digest(b"tgrab-split-semantics-v1\0" + canonical_json(hashes[interval[0]:interval[1]])):
            _fail("semantic split mismatch")
    return VerifiedDatasetManifest(path, expected_sha256, data, _TOKEN)


def _snapshot_factory():
    from .graph_reconstruction import Snapshot
    return Snapshot


def _load_split(manifest, split, guard, factory):
    if _read(manifest.path, MAX_MANIFEST, guard) != manifest._bytes:
        _fail("published manifest changed after verification")
    m = manifest.metadata
    record = m["splits"][split]
    data = _read(manifest.path.parent / record["filename"], record["bytes"], guard)
    if len(data) != record["bytes"] or _digest(data) != record["sha256"]:
        _fail("split bytes identity mismatch")
    payload = _json(data)
    _keys(payload, ("kind", "split", "snapshots"))
    start, end = record["interval"]
    if payload["kind"] != "tgrab_snapshot_split_v1" or payload["split"] != split or type(payload["snapshots"]) is not list or len(payload["snapshots"]) != end - start:
        _fail("split schema/count")
    result = []
    for t, snapshot in zip(range(start, end), payload["snapshots"]):
        _check_guard(guard)
        _keys(snapshot, ("timestamp", "nodes", "positive_pairs"))
        if type(snapshot["timestamp"]) is not int or snapshot["timestamp"] != t or type(snapshot["nodes"]) is not int or snapshot["nodes"] != m["spec"]["nodes"]:
            _fail("snapshot timestamp/node mismatch")
        _validate_pairs(snapshot["positive_pairs"], snapshot["nodes"])
        if _semantic(snapshot) != m["snapshot_semantic_sha256"][t]:
            _fail("snapshot semantics mismatch")
        result.append(factory(timestamp=t, nodes=snapshot["nodes"],
                              positive_pairs=tuple(tuple(p) for p in snapshot["positive_pairs"]),
                              node_ids=tuple(range(snapshot["nodes"]))))
    return tuple(result)


def _selection_state(selection):
    from .graph_reconstruction import CheckpointReference, ValidationOnlySelection
    if type(selection) is not ValidationOnlySelection or type(selection.selected) is not CheckpointReference:
        _fail("actual selection and checkpoint reference types required")
    reference = selection.selected
    if type(reference.epoch) is not int or any(type(v) is not str or not _HEX.fullmatch(v) for v in (reference.checkpoint_sha256, reference.recurrent_sha256, reference.link_head_sha256)):
        _fail("strict checkpoint reference fields required")
    reference.__post_init__()
    if selection.finalized is not True or type(selection.last_epoch) is not int or selection.last_epoch < reference.epoch:
        _fail("selection not validly finalized")
    if type(selection.best_f1) not in (int, float) or not 0 <= selection.best_f1 <= 1:
        _fail("invalid validation score")
    return reference, (selection.last_epoch, type(selection.best_f1), selection.best_f1)


class TrainingValidationStore:
    """Cooperative API isolation, not same-user security or immutable selection history."""
    def __init__(self, manifest, guard, factory, train, validation, token):
        if token is not _TOKEN:
            _fail("use open_train_validation")
        self.manifest, self.guard, self.factory = manifest, guard, factory
        self._train, self._validation = train, validation
        self._test_reserved = False
        self._test_attempted = False

    def train(self):
        if self._test_attempted:
            _fail("training access is terminal after test attempt")
        _check_guard(self.guard)
        return self._train

    def validation(self):
        if self._test_attempted:
            _fail("validation access is terminal after test attempt")
        _check_guard(self.guard)
        return self._validation

    def sealed_test_opener(self, selection):
        reference, summary = _selection_state(selection)
        _check_guard(self.guard)
        if self._test_reserved:
            _fail("test capability already reserved")
        self._test_reserved = True
        used = False

        def verify_current_selection():
            current, current_summary = _selection_state(selection)
            if current is not reference or current_summary != summary:
                _fail("selection changed after capability creation")

        def open_once():
            nonlocal used
            if used:
                _fail("test open already attempted")
            used = True  # A guard/read/parse/factory failure consumes the attempt.
            self._test_attempted = True
            verify_current_selection()
            _check_guard(self.guard)
            def lazy_test():
                # GR opens the callback before its fixed-weight prefix replay.
                # Delay all test payload I/O until evaluation requests a target.
                verify_current_selection()
                yield from _load_split(self.manifest, "test", self.guard, self.factory)
            return lazy_test()
        return open_once


def open_train_validation(manifest, guard, *, snapshot_factory=None):
    if type(manifest) is not VerifiedDatasetManifest or manifest._token is not _TOKEN:
        _fail("verified manifest required")
    _check_guard(guard)
    factory = _snapshot_factory() if snapshot_factory is None else snapshot_factory
    if not callable(factory):
        _fail("snapshot factory required")
    train = _load_split(manifest, "train", guard, factory)
    validation = _load_split(manifest, "validation", guard, factory)
    return TrainingValidationStore(manifest, guard, factory, train, validation, _TOKEN)
