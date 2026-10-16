"""Independent, read-only primary CSV/conversion oracle; standard library only.

This module deliberately imports no producer helper. An admitted caller owns
reporting, executed-source provenance, physical guards and durable receipts.
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


class VerificationError(ValueError):
    pass


RAW_SHA256 = "5a19f9e7949668473bc61cb59b6215f25f093ea0ee2d45b5e143338ded5523e3"
RAW_BYTES = 6907326
CONFIG_SHA256 = "22bef6bf781c5bfb5493dd5e03169ea0e658b2463ccaa9ba09189af454508f1f"
MAX_RAW = 16 * 1024 * 1024
MAX_METADATA = 2 * 1024 * 1024
MAX_SPLIT = 64 * 1024 * 1024
MAX_ROWS = 500000
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_FIELDS = ("src", "dst", "t", "edge_feat", "train_mask", "val_mask", "test_mask", "test_inductive_mask")


def _require(condition, reason):
    if not condition:
        raise VerificationError(reason)


def _integer(value, name, lower, upper):
    _require(type(value) is int and lower <= value <= upper, "invalid " + name)
    return value


def _guard(guard):
    _require(callable(guard), "raising deadline/physical guard is required")
    guard()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(record):
    return (json.dumps(record, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


@dataclass(frozen=True)
class VerificationSpec:
    nodes: int = 100
    block: int = 32
    train_periods: int = 40
    validation_periods: int = 4
    test_periods: int = 4

    def __post_init__(self):
        _integer(self.nodes, "node count", 3, 100)
        _integer(self.block, "block length", 1, 32)
        _integer(self.train_periods, "training periods", 1, 40)
        _integer(self.validation_periods, "validation periods", 1, 4)
        _integer(self.test_periods, "test periods", 1, 4)

    def record(self):
        return {name: getattr(self, name) for name in
                ("nodes", "block", "train_periods", "validation_periods", "test_periods")}

    def intervals(self):
        period = 2 * self.block
        train = period * self.train_periods
        validation = train + period * self.validation_periods
        return (("train", 0, train), ("validation", train, validation),
                ("test", validation, validation + period * self.test_periods))


PRIMARY_SPEC = VerificationSpec()


def _path(path):
    _require(".." not in Path(path).parts, "traversal path refused")
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        for component in reversed((absolute, *absolute.parents)):
            _require(not stat.S_ISLNK(component.lstat().st_mode), "symlink path component refused")
    except OSError as exc:
        raise VerificationError("input path is unavailable") from exc
    return absolute


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _read(path, maximum, guard, observations):
    _guard(guard)
    path = _path(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise VerificationError("cannot open bounded regular input") from exc
    try:
        initial = os.fstat(descriptor)
        _require(stat.S_ISREG(initial.st_mode) and initial.st_nlink == 1 and
                 initial.st_size <= maximum, "file type/link/size bound")
        parts = []
        length = 0
        while True:
            _guard(guard)
            chunk = os.read(descriptor, min(65536, maximum + 1 - length))
            if not chunk:
                break
            parts.append(chunk)
            length += len(chunk)
            _require(length <= maximum, "file grew beyond size bound")
        final = os.fstat(descriptor)
        _require(_identity(initial) == _identity(final) == _identity(path.lstat()) and
                 length == initial.st_size, "input changed during read")
        observations.append((path, _identity(final)))
        return b"".join(parts)
    except OSError as exc:
        raise VerificationError("input identity became unavailable") from exc
    finally:
        os.close(descriptor)


def _stable_at_end(observations, guard):
    for path, identity in observations:
        _guard(guard)
        _require(_identity(_path(path).lstat()) == identity, "input changed during verification")


def _csv_integer(token):
    # Work with decimal strings, never binary floats or narrowing casts.
    pieces = token.split(".")
    _require(len(pieces) <= 2, "invalid integral CSV token")
    whole = pieces[0]
    _require(bool(whole) and whole.isascii() and whole.isdecimal() and
             (whole == "0" or not whole.startswith("0")), "invalid integral CSV token")
    if len(pieces) == 2:
        _require(bool(pieces[1]) and set(pieces[1]) == {"0"}, "nonzero or invalid integral fraction")
    return int(whole)


def _csv_boolean(token):
    _require(token in ("True", "False", "1", "0"), "invalid CSV mask token")
    return token == "True" or token == "1"


def _raw_oracle(data, spec, guard):
    try:
        text = data.decode("ascii")
    except UnicodeError as exc:
        raise VerificationError("ASCII source CSV required") from exc
    intervals = spec.intervals()
    total = intervals[-1][2]
    edge_sets = [set() for _ in range(total)]
    markers = set()
    counts = {"raw_rows": 0, "identical_duplicate_rows": 0,
              "source_marker_rows": 0, "unique_offdiagonal_edges": 0}
    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        _require(tuple(next(reader, ())) == _FIELDS, "source CSV header mismatch")
        for row in reader:
            _guard(guard)
            counts["raw_rows"] += 1
            _require(counts["raw_rows"] <= MAX_ROWS and len(row) == 8 and
                     all(len(token) <= 32 for token in row), "CSV row/field count bound")
            source, destination, timestamp = (_csv_integer(v) for v in row[:3])
            _require(source < spec.nodes and destination < spec.nodes and timestamp < total,
                     "source node/timestamp outside candidate bounds")
            unit = row[3].split(".")
            _require(unit == ["1"] or (len(unit) == 2 and unit[0] == "1" and
                     bool(unit[1]) and set(unit[1]) == {"0"}), "nonunit source weight")
            observed_masks = [_csv_boolean(v) for v in row[4:]]
            expected_masks = [start <= timestamp < end for _, start, end in intervals] + [False]
            _require(observed_masks == expected_masks, "source split masks mismatch")
            if source == destination:
                _require(source == 0, "unidentified source self loop")
                markers.add(timestamp)
                counts["source_marker_rows"] += 1
            pair = (source, destination)
            if pair in edge_sets[timestamp]:
                counts["identical_duplicate_rows"] += 1
            edge_sets[timestamp].add(pair)
    except csv.Error as exc:
        raise VerificationError("malformed bounded CSV") from exc
    for timestamp, edges in enumerate(edge_sets):
        _guard(guard)
        _require(bool(edges), "missing source timestamp")
        if timestamp in markers:
            _require(edges == {(0, 0)}, "source marker mixed with positive edges")
            raise VerificationError(f"empty candidate at timestamp {timestamp} after source marker resolution")
        degrees = {}
        for source, destination in edges:
            _require((destination, source) in edges, "missing source reverse edge")
            degrees[source] = degrees.get(source, 0) + 1
        _require(all(degree < spec.nodes - 1 for degree in degrees.values()),
                 "positive source has no feasible offdiagonal nonedge")
        counts["unique_offdiagonal_edges"] += len(edges)
    patterns = [[list(pair) for pair in sorted(edge_sets[t])] for t in (0, spec.block)]
    _require(patterns[0] != patterns[1], "two distinct nonempty source patterns required")
    snapshots = []
    semantic = []
    for timestamp, edges in enumerate(edge_sets):
        _guard(guard)
        pairs = [list(pair) for pair in sorted(edges)]
        _require(pairs == patterns[(timestamp // spec.block) % 2], "source two-block periodic ordering mismatch")
        snapshot = {"timestamp": timestamp, "nodes": spec.nodes, "positive_pairs": pairs}
        snapshots.append(snapshot)
        semantic.append(_sha(b"tgrab-snapshot-semantics-v1\0" + _json_bytes({
            **snapshot, "node_order": "ascending_integer_ids", "features": "float32_identity",
            "weights": "float32_unit_aligned_with_pairs"})))
    return snapshots, semantic, patterns, counts


def verify_primary_conversion(raw_path, manifest_path, *, expected_raw_sha256,
                              expected_raw_bytes, expected_manifest_sha256,
                              expected_converter_sha256, expected_converter_revision,
                              config_path, guard, fixture_only=False, spec=PRIMARY_SPEC):
    """Reconstruct every artifact independently from trusted raw/policy identities.

    Fixture mode is explicit and cannot return a primary-eligible receipt.
    The function writes nothing and does not import/call the producer.
    """
    _require(type(spec) is VerificationSpec and type(fixture_only) is bool, "typed spec/fixture flag required")
    for value in (expected_raw_sha256, expected_manifest_sha256, expected_converter_sha256):
        _require(type(value) is str and _SHA.fullmatch(value) is not None, "trusted SHA256 required")
    _require(type(expected_converter_revision) is str and
             _REVISION.fullmatch(expected_converter_revision) is not None, "trusted converter revision required")
    _integer(expected_raw_bytes, "raw bytes", 1, MAX_RAW)
    if not fixture_only:
        _require(spec == PRIMARY_SPEC and expected_raw_sha256 == RAW_SHA256 and
                 expected_raw_bytes == RAW_BYTES, "fixed primary identity required")
    _guard(guard)
    manifest_path = _path(manifest_path)
    _require(manifest_path.name == "manifest.json", "only published manifest accepted")
    observations = []
    config = _read(config_path, MAX_METADATA, guard, observations)
    _require(_sha(config) == CONFIG_SHA256, "frozen reconstruction config mismatch")
    raw = _read(raw_path, expected_raw_bytes, guard, observations)
    _require(len(raw) == expected_raw_bytes and _sha(raw) == expected_raw_sha256, "raw byte identity mismatch")
    actual_manifest = _read(manifest_path, MAX_METADATA, guard, observations)
    _require(_sha(actual_manifest) == expected_manifest_sha256, "trusted manifest SHA mismatch")
    snapshots, semantic, patterns, counts = _raw_oracle(raw, spec, guard)
    split_records = {}
    compared = 0
    for name, start, end in spec.intervals():
        _guard(guard)
        expected_payload = _json_bytes({"kind": "tgrab_snapshot_split_v1", "split": name,
                                        "snapshots": snapshots[start:end]})
        _require(len(expected_payload) <= MAX_SPLIT, "rebuilt split size bound")
        actual_payload = _read(manifest_path.parent / (name + ".json"), len(expected_payload), guard, observations)
        _require(actual_payload == expected_payload, "converted " + name + " snapshots/encoding disagree with raw oracle")
        split_records[name] = {"filename": name + ".json", "sha256": _sha(actual_payload),
                               "bytes": len(actual_payload), "interval": [start, end],
                               "semantic_sha256": _sha(b"tgrab-split-semantics-v1\0" + _json_bytes(semantic[start:end]))}
        compared += end - start
    expected_manifest = {
        "kind": "tgrab_fixture_converted_v1" if fixture_only else "tgrab_primary_converted_v1",
        "schema_version": 1, "spec": spec.record(),
        "origin": {"fixture_only": True} if fixture_only else {
            "repository": "Gilestel/T-GRAB", "revision": "603388f332c7756f90bc1702b26053c2bc834654",
            "path": "periodicity/periodicity_det_(2, 32)_100N_40tr_4vl_4ts_pER(0.01).csv"},
        "raw": {"sha256": expected_raw_sha256, "bytes": expected_raw_bytes},
        "converter": {"revision": expected_converter_revision, "sha256": expected_converter_sha256},
        "config_sha256": CONFIG_SHA256, "generation_performed": False,
        "csv_policy": "ascii_fixed_header_integral_digits_optional_zero_fraction_bool_True_False_1_0_unit_1_optional_zero_fraction_v1",
        "features": "float32_identity", "weights": "float32_unit_aligned_with_pairs",
        "edge_order": "lexicographic_source_destination", "node_order": "ascending_integer_ids",
        "marker_policy": "only_source_node_zero_empty_marker_removed_timestamp_retained_then_empty_candidate_rejected",
        "counts": counts, "pattern_pairs": patterns, "snapshot_semantic_sha256": semantic,
        "splits": split_records}
    expected_manifest_bytes = _json_bytes(expected_manifest)
    _require(len(expected_manifest_bytes) <= MAX_METADATA and actual_manifest == expected_manifest_bytes,
             "complete converted manifest disagrees with independent raw/policy oracle")
    verifier_source = _read(Path(__file__), MAX_METADATA, guard, observations)
    _stable_at_end(observations, guard)
    _guard(guard)
    return {"kind": "tgrab_fixture_conversion_independent_verification_v1" if fixture_only else
                    "tgrab_primary_conversion_independent_verification_v1",
            "status": "verified", "primary_eligible": not fixture_only,
            "raw_sha256": expected_raw_sha256, "raw_bytes": len(raw),
            "manifest_sha256": _sha(actual_manifest), "converter_sha256": expected_converter_sha256,
            "converter_revision": expected_converter_revision, "config_sha256": _sha(config),
            "verifier_source_sha256": _sha(verifier_source), "spec": spec.record(),
            "snapshot_count": len(snapshots), "split_count": len(split_records), **counts,
            "converted_snapshot_comparisons": compared, "semantic_snapshot_comparisons": len(semantic),
            "split_byte_comparisons": len(split_records), "complete_manifest_byte_comparisons": 1,
            "split_hashes": split_records, "stable_input_files_checked": len(observations),
            "independent_of_producer": True, "scientific_acceptance": False}
