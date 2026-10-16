"""Independent verifier fixtures; reviewer authored producer (see review).

The literal artifact oracle below uses neither producer nor verifier helpers.
One separate test corrupts outputs from the actual producer under test.
"""
from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from representation_collapse import graph_data_verifier as verifier

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/graph/tgrab-reconstruction-v1.json"
CONFIG_SHA = "22bef6bf781c5bfb5493dd5e03169ea0e658b2463ccaa9ba09189af454508f1f"
CONVERTER_SHA = hashlib.sha256((ROOT / "src/representation_collapse/graph_data.py").read_bytes()).hexdigest()
REVISION = "a" * 40


def encode(value):
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def semantic(snapshot):
    content = {"timestamp": snapshot["timestamp"], "nodes": snapshot["nodes"], "positive_pairs": snapshot["positive_pairs"],
               "node_order": "ascending_integer_ids", "features": "float32_identity", "weights": "float32_unit_aligned_with_pairs"}
    return digest(b"tgrab-snapshot-semantics-v1\0" + encode(content))


class IndependentVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.base = Path(self.temp.name)
        self.raw = self.base / "raw.csv"
        self.folder = self.base / "converted"; self.folder.mkdir()
        self.manifest_path = self.folder / "manifest.json"
        self.spec = verifier.VerificationSpec(nodes=4, block=2, train_periods=1, validation_periods=1, test_periods=1)
        self.intervals = {"train": [0, 4], "validation": [4, 8], "test": [8, 12]}
        self.patterns = [[[0, 1], [1, 0]], [[2, 3], [3, 2]]]
        self.snapshots = [{"timestamp": t, "nodes": 4, "positive_pairs": deepcopy(self.patterns[(t // 2) % 2])} for t in range(12)]
        self.rows = []
        for snapshot in self.snapshots:
            t = snapshot["timestamp"]
            for u, v in snapshot["positive_pairs"]:
                self.rows.append([str(u), str(v), str(t), "1.0", str(t < 4), str(4 <= t < 8), str(8 <= t), "False"])
        self.write_raw()
        self.manifest = {
            "kind": "tgrab_fixture_converted_v1", "schema_version": 1,
            "spec": {"nodes": 4, "block": 2, "train_periods": 1, "validation_periods": 1, "test_periods": 1},
            "origin": {"fixture_only": True}, "raw": {"sha256": digest(self.raw.read_bytes()), "bytes": self.raw.stat().st_size},
            "converter": {"sha256": CONVERTER_SHA, "revision": REVISION}, "config_sha256": CONFIG_SHA,
            "generation_performed": False,
            "csv_policy": "ascii_fixed_header_integral_digits_optional_zero_fraction_bool_True_False_1_0_unit_1_optional_zero_fraction_v1",
            "features": "float32_identity", "weights": "float32_unit_aligned_with_pairs", "edge_order": "lexicographic_source_destination",
            "node_order": "ascending_integer_ids",
            "marker_policy": "only_source_node_zero_empty_marker_removed_timestamp_retained_then_empty_candidate_rejected",
            "counts": {"raw_rows": 24, "identical_duplicate_rows": 0, "source_marker_rows": 0, "unique_offdiagonal_edges": 24},
            "pattern_pairs": deepcopy(self.patterns), "snapshot_semantic_sha256": [], "splits": {}}
        self.write_payloads()
        self.guards = 0

    def tearDown(self):
        self.temp.cleanup()

    def guard(self):
        self.guards += 1

    def write_raw(self, rows=None):
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("src", "dst", "t", "edge_feat", "train_mask", "val_mask", "test_mask", "test_inductive_mask"))
        writer.writerows(self.rows if rows is None else rows)
        self.raw.write_bytes(stream.getvalue().encode("ascii"))

    def write_payloads(self):
        self.manifest["snapshot_semantic_sha256"] = [semantic(s) for s in self.snapshots]
        for split, (start, end) in self.intervals.items():
            data = encode({"kind": "tgrab_snapshot_split_v1", "split": split, "snapshots": self.snapshots[start:end]})
            (self.folder / (split + ".json")).write_bytes(data)
            self.manifest["splits"][split] = {"filename": split + ".json", "sha256": digest(data), "bytes": len(data),
                "interval": [start, end], "semantic_sha256": digest(b"tgrab-split-semantics-v1\0" + encode(self.manifest["snapshot_semantic_sha256"][start:end]))}
        self.save_manifest()

    def save_manifest(self):
        self.manifest_path.write_bytes(encode(self.manifest))

    def bind_raw(self):
        self.manifest["raw"] = {"sha256": digest(self.raw.read_bytes()), "bytes": self.raw.stat().st_size}
        self.save_manifest()

    def verify(self, **changes):
        kwargs = {"expected_raw_sha256": digest(self.raw.read_bytes()), "expected_raw_bytes": self.raw.stat().st_size,
                  "expected_manifest_sha256": digest(self.manifest_path.read_bytes()), "expected_converter_sha256": CONVERTER_SHA,
                  "expected_converter_revision": REVISION, "config_path": CONFIG, "guard": self.guard,
                  "fixture_only": True, "spec": self.spec}
        kwargs.update(changes)
        return verifier.verify_primary_conversion(self.raw, self.manifest_path, **kwargs)

    def test_independent_literal_positive_oracle(self):
        result = self.verify()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["kind"], "tgrab_fixture_conversion_independent_verification_v1")
        self.assertTrue(result["independent_of_producer"])
        self.assertFalse(result["primary_eligible"])
        self.assertFalse(result["scientific_acceptance"])
        self.assertEqual(result["snapshot_count"], 12)
        self.assertEqual(result["split_count"], 3)
        self.assertGreater(self.guards, 24)

    def test_producer_helpers_are_never_imported_or_called(self):
        import ast
        tree = ast.parse(Path(verifier.__file__).read_text())
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imported.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
        self.assertEqual(imported, {"__future__", "csv", "dataclasses", "hashlib", "io", "json", "os", "pathlib", "re", "stat"})
        import builtins
        original = builtins.__import__
        def refuse(name, *args, **kwargs):
            if name.endswith("graph_data") or name in ("numpy", "torch", "wandb"):
                raise AssertionError("producer/numerical/SDK import forbidden")
            return original(name, *args, **kwargs)
        with mock.patch.object(builtins, "__import__", side_effect=refuse):
            self.assertEqual(self.verify()["status"], "verified")

    def test_real_producer_corruption_with_rebound_hashes_rejected_against_raw(self):
        from representation_collapse import graph_data as producer
        output = self.base / "actual-producer"
        actual_manifest = producer.convert_primary_csv(self.raw, output,
            expected_raw_sha256=digest(self.raw.read_bytes()), expected_raw_bytes=self.raw.stat().st_size,
            config_path=CONFIG, converter_revision=REVISION, guard=self.guard,
            spec=producer.DatasetSpec(4, 2, 1, 1, 1), fixture_only=True)
        self.folder = output; self.manifest_path = actual_manifest
        self.manifest = json.loads(actual_manifest.read_bytes())
        self.snapshots[5]["positive_pairs"] = [[0, 2], [2, 0]]
        # Change file bytes, all semantic hashes and the trusted manifest hash.
        # A merely internally consistent verifier would incorrectly accept.
        self.write_payloads()
        with self.assertRaises(verifier.VerificationError):
            self.verify()

    def test_all_forged_pattern_semantics_and_payloads_rejected(self):
        self.patterns[0] = [[0, 2], [2, 0]]
        self.manifest["pattern_pairs"] = deepcopy(self.patterns)
        for snapshot in self.snapshots:
            if snapshot["timestamp"] % 4 < 2:
                snapshot["positive_pairs"] = deepcopy(self.patterns[0])
        self.write_payloads()
        with self.assertRaises(verifier.VerificationError):
            self.verify()

    def test_lexical_integral_zero_fractions_and_exact_boolean_tokens(self):
        for row in self.rows:
            row[:3] = [value + ".000" for value in row[:3]]
            row[3] = "1.000"
            row[4:] = ["1" if value == "True" else "0" for value in row[4:]]
        self.write_raw(); self.bind_raw()
        self.assertEqual(self.verify()["status"], "verified")

    def test_lexical_nonfinite_near_unit_fraction_exponent_and_whitespace_rejected(self):
        original = deepcopy(self.rows)
        for column, value in ((0, "01"), (0, "-0"), (1, "1.00000000000000001"), (2, "1e0"), (2, " 0"),
                              (3, "0.99999999999999999"), (3, "nan"), (3, "Infinity"), (4, "true"), (7, "False ")):
            with self.subTest(column=column, value=value):
                rows = deepcopy(original); rows[0][column] = value
                self.write_raw(rows); self.bind_raw()
                with self.assertRaises(verifier.VerificationError):
                    self.verify()

    def test_raw_duplicates_must_match_manifest_counts(self):
        self.write_raw(self.rows + [self.rows[0]]); self.bind_raw()
        with self.assertRaises(verifier.VerificationError):
            self.verify()
        self.manifest["counts"].update(raw_rows=25, identical_duplicate_rows=1); self.save_manifest()
        self.assertEqual(self.verify()["status"], "verified")

    def test_raw_timestamp_reverse_mask_marker_and_pattern_invariants(self):
        variants = [self.rows[2:], self.rows[1:]]
        mask = deepcopy(self.rows); mask[0][4] = "False"; variants.append(mask)
        out = deepcopy(self.rows); out[0][0] = "4"; variants.append(out)
        marker = [r[:] for r in self.rows if r[2] != "0"]; marker.append(["0", "0", *self.rows[0][2:]]); variants.append(marker)
        mixed = deepcopy(self.rows); mixed.append(["0", "0", *self.rows[0][2:]]); variants.append(mixed)
        late = deepcopy(self.rows); late[-2][0:2] = ["0", "2"]; late[-1][0:2] = ["2", "0"]; variants.append(late)
        for rows in variants:
            with self.subTest(rows=rows[:2]):
                self.write_raw(rows); self.bind_raw()
                with self.assertRaises(verifier.VerificationError):
                    self.verify()

    def test_positive_source_negative_feasibility_checked_from_raw(self):
        rows = deepcopy(self.rows)
        for t in range(12):
            if t % 4 < 2:
                template = next(r for r in self.rows if r[2] == str(t))
                for u, v in ((0, 2), (2, 0), (0, 3), (3, 0)):
                    rows.append([str(u), str(v), *template[2:]])
        self.write_raw(rows); self.bind_raw()
        self.patterns[0] = [[0, 1], [0, 2], [0, 3], [1, 0], [2, 0], [3, 0]]
        self.manifest["pattern_pairs"] = deepcopy(self.patterns)
        for snapshot in self.snapshots:
            if snapshot["timestamp"] % 4 < 2:
                snapshot["positive_pairs"] = deepcopy(self.patterns[0])
        self.manifest["counts"].update(raw_rows=48, unique_offdiagonal_edges=48)
        self.write_payloads()
        with self.assertRaises(verifier.VerificationError):
            self.verify()

    def test_identical_two_patterns_rejected_even_when_every_artifact_matches_raw(self):
        self.patterns[1] = deepcopy(self.patterns[0]); self.manifest["pattern_pairs"] = deepcopy(self.patterns)
        for snapshot in self.snapshots:
            snapshot["positive_pairs"] = deepcopy(self.patterns[0])
        rows = deepcopy(self.rows)
        for row in rows:
            row[0] = {"2": "0", "3": "1"}.get(row[0], row[0])
            row[1] = {"2": "0", "3": "1"}.get(row[1], row[1])
        self.write_raw(rows); self.bind_raw(); self.write_payloads()
        with self.assertRaises(verifier.VerificationError):
            self.verify()

    def test_fixture_cannot_be_promoted_to_primary(self):
        with self.assertRaises(verifier.VerificationError):
            self.verify(fixture_only=False)
        self.manifest["kind"] = "tgrab_primary_converted_v1"; self.save_manifest()
        with self.assertRaises(verifier.VerificationError):
            self.verify()

    def test_converter_revision_hash_config_and_semantic_identity_rejected(self):
        original = deepcopy(self.manifest)
        mutations = [lambda m: m["converter"].update(revision="b" * 40), lambda m: m["converter"].update(sha256="0" * 64),
                     lambda m: m.update(config_sha256="0" * 64), lambda m: m["snapshot_semantic_sha256"].__setitem__(11, "0" * 64),
                     lambda m: m["splits"]["test"].update(semantic_sha256="0" * 64), lambda m: m["counts"].update(raw_rows=True),
                     lambda m: m["spec"].update(nodes=True), lambda m: m.update(features="float64_identity")]
        for mutate in mutations:
            self.manifest = deepcopy(original); mutate(self.manifest); self.save_manifest()
            with self.assertRaises(verifier.VerificationError):
                self.verify()

    def test_split_timestamp_order_and_duplicate_schema_changed_with_bytehash_rebound(self):
        file = self.folder / "validation.json"
        payload = json.loads(file.read_bytes()); payload["snapshots"][0]["timestamp"] = 5
        data = encode(payload); file.write_bytes(data)
        self.manifest["splits"]["validation"].update(sha256=digest(data), bytes=len(data)); self.save_manifest()
        with self.assertRaises(verifier.VerificationError):
            self.verify()

    def test_duplicate_json_nonfinite_and_noncanonical_payload_rejected(self):
        original = self.manifest_path.read_bytes()
        for malformed in (b'{"kind":"x","kind":"y"}\n', b'{"bad":NaN}\n', original.rstrip(b"\n")):
            self.manifest_path.write_bytes(malformed)
            with self.assertRaises(verifier.VerificationError):
                self.verify()

    def test_symlink_hardlink_fifo_traversal_and_file_bounds(self):
        file = self.folder / "test.json"; old = file.read_bytes(); file.unlink(); file.symlink_to(self.raw)
        with self.assertRaises(verifier.VerificationError):
            self.verify()
        file.unlink(); file.write_bytes(old)
        alias = self.base / "raw-hardlink"; os.link(self.raw, alias)
        with self.assertRaises(verifier.VerificationError):
            self.verify()
        alias.unlink()
        original_raw = self.raw
        self.raw = self.base / ".." / self.base.name / "raw.csv"
        with self.assertRaises(verifier.VerificationError):
            self.verify()
        self.raw = original_raw
        with self.assertRaises(verifier.VerificationError):
            self.verify(expected_raw_bytes=self.raw.stat().st_size - 1)
        file.unlink(); os.mkfifo(file)
        with self.assertRaises(verifier.VerificationError):
            self.verify()

    def test_guard_failure_preserves_all_inputs_without_writes(self):
        before = {str(p.relative_to(self.base)): p.read_bytes() for p in self.base.rglob("*") if p.is_file()}
        def stop():
            raise RuntimeError("synthetic owner stop")
        with self.assertRaises(RuntimeError):
            self.verify(guard=stop)
        after = {str(p.relative_to(self.base)): p.read_bytes() for p in self.base.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_mid_verification_guard_failure_is_not_success(self):
        calls = []
        def stop_later():
            calls.append(None)
            if len(calls) == 9:
                raise RuntimeError("synthetic deadline during verification")
        with self.assertRaises(RuntimeError):
            self.verify(guard=stop_later)
        self.assertEqual(len(calls), 9)

    def test_previously_read_raw_mutation_is_detected_at_operation_end(self):
        original_read = os.read
        inode = self.raw.stat().st_ino
        finished_raw = False
        mutated = False
        def read(descriptor, size):
            nonlocal finished_raw
            data = original_read(descriptor, size)
            if not data and os.fstat(descriptor).st_ino == inode:
                finished_raw = True
            return data
        def guard():
            nonlocal mutated
            if finished_raw and not mutated:
                content = self.raw.read_bytes()
                self.raw.write_bytes(content.replace(b"0,1,0,", b"1,1,0,", 1))
                mutated = True
        with mock.patch.object(os, "read", side_effect=read):
            with self.assertRaisesRegex(verifier.VerificationError, "changed during verification"):
                self.verify(guard=guard)
        self.assertTrue(mutated)


if __name__ == "__main__":
    unittest.main()
