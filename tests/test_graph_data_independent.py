"""Independent synthetic/adversarial store review; no target or model work."""
from dataclasses import dataclass
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from representation_collapse import graph_data as gd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/graph/tgrab-reconstruction-v1.json"


@dataclass(frozen=True)
class Record:
    timestamp: int
    nodes: int
    positive_pairs: tuple
    node_ids: tuple


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False) + "\n").encode("ascii")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def snapshot_digest(t, nodes, pairs):
    return sha(b"tgrab-snapshot-semantics-v1\0" + canonical({
        "timestamp": t, "nodes": nodes, "positive_pairs": [list(p) for p in pairs],
        "node_order": "ascending_integer_ids", "features": "float32_identity",
        "weights": "float32_unit_aligned_with_pairs"}))


class IndependentGraphDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.base = Path(self.temp.name)
        self.spec = gd.DatasetSpec(4, 2, 1, 1, 1)
        self.guard_calls = 0
        self.counter = 0

    def tearDown(self):
        self.temp.cleanup()

    def guard(self):
        self.guard_calls += 1

    def rows(self, spec=None):
        spec = spec or self.spec
        result = []
        for t in range(spec.intervals["test"][1]):
            pattern = ((0, 1), (1, 0)) if (t // spec.block) % 2 == 0 else ((2, 3), (3, 2))
            masks = [str(lo <= t < hi) for lo, hi in spec.intervals.values()] + ["False"]
            result.extend([[str(u), str(v), str(t), "1", *masks] for u, v in pattern])
        return result

    def convert(self, rows=None, *, spec=None, changes=None):
        self.counter += 1
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("src", "dst", "t", "edge_feat", "train_mask", "val_mask", "test_mask", "test_inductive_mask"))
        writer.writerows(self.rows(spec) if rows is None else rows)
        raw = self.base / f"input-{self.counter}.csv"
        raw.write_bytes(stream.getvalue().encode("ascii"))
        options = dict(expected_raw_sha256=sha(raw.read_bytes()), expected_raw_bytes=raw.stat().st_size,
                       config_path=CONFIG, converter_revision="d" * 40, guard=self.guard,
                       spec=spec or self.spec, fixture_only=True)
        options.update(changes or {})
        return gd.convert_primary_csv(raw, self.base / f"converted-{self.counter}", **options)

    def manifest(self):
        path = self.convert()
        return gd.read_manifest(path, sha(path.read_bytes()), self.guard)

    def store(self):
        return gd.open_train_validation(self.manifest(), self.guard, snapshot_factory=Record)

    def selection(self):
        from representation_collapse.graph_reconstruction import CheckpointReference, ValidationOnlySelection
        s = ValidationOnlySelection()
        r = CheckpointReference(2, "a" * 64, "b" * 64, "c" * 64)
        s.consider(r, 0.75, split="validation", epoch_complete=True)
        self.assertIs(s.finalize(), r)
        return s

    def test_primary_dimensions_and_boundaries_have_independent_semantic_oracle(self):
        # Full primary dimensions, explicitly synthetic/fixture-labelled, sparse
        # deterministic edges: not the primary CSV or a generator reproduction.
        self.assertEqual(gd.PRIMARY_SPEC.as_dict(), {"nodes": 100, "block": 32,
                         "train_periods": 40, "validation_periods": 4, "test_periods": 4})
        path = self.convert(spec=gd.PRIMARY_SPEC)
        m = gd.read_manifest(path, sha(path.read_bytes()), self.guard)
        meta = m.metadata
        self.assertEqual(meta["kind"], "tgrab_fixture_converted_v1")
        self.assertEqual(meta["origin"], {"fixture_only": True})
        self.assertEqual([meta["splits"][s]["interval"] for s in ("train", "validation", "test")],
                         [[0, 2560], [2560, 2816], [2816, 3072]])
        expected = [snapshot_digest(t, 100, ((0, 1), (1, 0)) if t % 64 < 32 else ((2, 3), (3, 2)))
                    for t in range(3072)]
        self.assertEqual(meta["snapshot_semantic_sha256"], expected)
        for split, (start, end) in (("train", (0, 2560)), ("validation", (2560, 2816)), ("test", (2816, 3072))):
            self.assertEqual(meta["splits"][split]["semantic_sha256"],
                             sha(b"tgrab-split-semantics-v1\0" + canonical(expected[start:end])))
        store = gd.open_train_validation(m, self.guard, snapshot_factory=Record)
        self.assertEqual((len(store.train()), len(store.validation())), (2560, 256))
        for t in (0, 31, 32, 63, 64, 2559):
            self.assertEqual(store.train()[t].positive_pairs,
                             ((0, 1), (1, 0)) if t % 64 < 32 else ((2, 3), (3, 2)))
        self.assertEqual((store.validation()[0].timestamp, store.validation()[-1].timestamp), (2560, 2815))

    def test_saturated_positive_source_rejected_against_complete_pattern(self):
        rows = self.rows()
        for t in range(12):
            if t % 4 < 2:
                masks = next(r[4:] for r in rows if r[2] == str(t))
                rows.extend([[str(u), str(v), str(t), "1", *masks]
                             for u, v in ((0, 2), (2, 0), (0, 3), (3, 0))])
        with self.assertRaisesRegex(gd.DataContractError, "feasible negative"):
            self.convert(rows)

    def test_precision_near_one_fraction_and_huge_integer_are_not_truncated(self):
        for col, token in ((0, "1.00000000000000000000001"), (0, "99999999999999999999999999999999"),
                           (1, "3.00000000000000001"), (2, "2e0"),
                           (3, "1.0000000000000000000001"), (3, "NaN"), (4, "1.0")):
            rows = self.rows(); rows[0][col] = token
            with self.subTest(col=col, token=token), self.assertRaises(gd.DataContractError):
                self.convert(rows)

    def test_late_pattern_mismatch_cannot_publish_even_when_all_rows_valid(self):
        rows = self.rows()
        for row in rows:
            if row[2] == "11":
                row[:2] = ["0", "1"] if row[0] == "2" else ["1", "0"]
        with self.assertRaisesRegex(gd.DataContractError, "pattern ordering"):
            self.convert(rows)
        output = self.base / "converted-1"
        self.assertFalse((output / "manifest.json").exists())
        self.assertTrue((output / "failure.json").is_file())

    def test_source_marker_resolution_failure_retains_timestamp_and_counts(self):
        rows = [r for r in self.rows() if r[2] != "5"]
        rows.extend([["0", "0", "5", "1", "False", "True", "False", "False"]] * 2)
        with self.assertRaises(gd.DataContractError):
            self.convert(rows)
        output = self.base / "converted-1"
        failure = json.loads((output / "failure.json").read_bytes())
        self.assertEqual(failure["source_marker_timestamps"], [5])
        self.assertEqual(failure["partial_counts"]["source_marker_rows"], 2)
        self.assertEqual(failure["partial_counts"]["identical_duplicate_rows"], 1)
        self.assertIn("timestamp 5", failure["message"])
        self.assertFalse((output / "manifest.json").exists())

    def test_split_publication_failure_retains_ineligible_partial_attempt(self):
        original = gd._write_new
        def write(path, data):
            if Path(path).name == "validation.json":
                raise OSError("synthetic full filesystem")
            return original(path, data)
        with mock.patch.object(gd, "_write_new", side_effect=write), self.assertRaises(OSError):
            self.convert()
        output = self.base / "converted-1"
        self.assertTrue((output / "train.json").is_file())
        self.assertTrue((output / "failure.json").is_file())
        self.assertFalse((output / "manifest.json").exists())
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        # Refused fresh-directory re-entry must leave the previous failure untouched.
        raw = self.base / "input-1.csv"
        with self.assertRaises(gd.DataContractError):
            gd.convert_primary_csv(raw, output, expected_raw_sha256=sha(raw.read_bytes()),
                                   expected_raw_bytes=raw.stat().st_size, config_path=CONFIG,
                                   converter_revision="d" * 40, guard=self.guard, spec=self.spec, fixture_only=True)
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_manifest_and_train_validation_never_probe_missing_test_payload(self):
        path = self.convert()
        test = path.parent / "test.json"
        test.unlink()
        # Fail even an attempted lstat/open, not just deserialization or hashing.
        real_lstat, real_open = Path.lstat, os.open
        def lstat(p, *a, **kw):
            if Path(p) == test:
                raise AssertionError("test stat before authorization")
            return real_lstat(p, *a, **kw)
        def opened(p, *a, **kw):
            if Path(p) == test:
                raise AssertionError("test open before authorization")
            return real_open(p, *a, **kw)
        with mock.patch.object(Path, "lstat", lstat), mock.patch.object(os, "open", side_effect=opened):
            manifest = gd.read_manifest(path, sha(path.read_bytes()), self.guard)
            store = gd.open_train_validation(manifest, self.guard, snapshot_factory=Record)
            self.assertEqual(len(store.train()), 4)
            self.assertEqual(len(store.validation()), 4)

    def test_test_payload_is_lazy_and_all_public_access_is_terminal_after_callback(self):
        store = self.store(); selection = self.selection()
        opener = store.sealed_test_opener(selection)
        with mock.patch.object(gd, "_read", side_effect=AssertionError("eager test read")):
            iterator = opener()
        for action in (store.train, store.validation, opener, lambda: store.sealed_test_opener(selection)):
            with self.assertRaises(gd.DataContractError):
                action()
        self.assertEqual([row.timestamp for row in iterator], [8, 9, 10, 11])
        self.assertEqual(list(iterator), [])

    def test_selection_gate_is_rechecked_at_first_actual_test_read(self):
        for field, value in (("finalized", False), ("last_epoch", 0), ("best_f1", float("nan")),
                             ("selected", None)):
            store = self.store(); selection = self.selection()
            iterator = store.sealed_test_opener(selection)()
            setattr(selection, field, value)
            with self.subTest(field=field), mock.patch.object(gd, "_read", side_effect=AssertionError("unauthorized test read")):
                with self.assertRaises(gd.DataContractError):
                    next(iterator)
            with self.assertRaises(gd.DataContractError):
                store.train()

    def test_selection_changed_after_reservation_is_refused_and_consumed(self):
        store = self.store(); selection = self.selection()
        opener = store.sealed_test_opener(selection)
        from representation_collapse.graph_reconstruction import CheckpointReference
        selection.selected = CheckpointReference(3, "e" * 64, "f" * 64, "0" * 64)
        with mock.patch.object(gd, "_read", side_effect=AssertionError("unexpected test read")):
            with self.assertRaises(gd.DataContractError):
                opener()
        with self.assertRaises(gd.DataContractError):
            opener()

    def test_wrong_selection_types_and_corrupt_current_fields_cannot_reserve(self):
        from representation_collapse.graph_reconstruction import ValidationOnlySelection
        class Derived(ValidationOnlySelection):
            pass
        store = self.store()
        invalid = [Derived(), ValidationOnlySelection()]
        for field, value in (("finalized", 1), ("last_epoch", True), ("best_f1", True),
                             ("best_f1", float("inf")), ("last_epoch", 1)):
            s = self.selection(); setattr(s, field, value); invalid.append(s)
        for selection in invalid:
            with self.subTest(selection=selection), self.assertRaises(gd.DataContractError):
                store.sealed_test_opener(selection)
        # Invalid gate attempts do not consume the one genuinely valid reservation.
        self.assertEqual(next(store.sealed_test_opener(self.selection())()).timestamp, 8)

    def test_failed_test_factory_consumes_attempt_and_yields_no_partial_targets(self):
        store = self.store(); opener = store.sealed_test_opener(self.selection())
        calls = []
        def bad_factory(**kwargs):
            calls.append(kwargs["timestamp"])
            if kwargs["timestamp"] == 9:
                raise RuntimeError("synthetic decode adapter failure")
            return Record(**kwargs)
        store.factory = bad_factory
        iterator = opener()
        with self.assertRaisesRegex(RuntimeError, "decode adapter failure"):
            next(iterator)
        self.assertEqual(calls, [8, 9])
        self.assertEqual(list(iterator), [])
        with self.assertRaises(gd.DataContractError):
            opener()

    def test_semantic_mismatch_rejected_even_with_rebound_valid_file_byte_hash(self):
        manifest = self.manifest(); meta = manifest.metadata
        path = manifest.path.parent / "train.json"
        payload = json.loads(path.read_bytes())
        payload["snapshots"][0]["positive_pairs"] = [[2, 3], [3, 2]]
        data = canonical(payload); path.write_bytes(data)
        meta["splits"]["train"].update(sha256=sha(data), bytes=len(data))
        manifest.path.write_bytes(canonical(meta))
        rebound = gd.read_manifest(manifest.path, sha(manifest.path.read_bytes()), self.guard)
        with self.assertRaisesRegex(gd.DataContractError, "semantics mismatch"):
            gd.open_train_validation(rebound, self.guard, snapshot_factory=Record)

    def test_detached_manifest_nested_mutation_cannot_change_verified_bytes(self):
        manifest = self.manifest()
        detached = manifest.metadata
        detached["pattern_pairs"][0][0][0] = 99
        detached["splits"]["train"]["filename"] = "test.json"
        self.assertEqual(manifest.metadata["pattern_pairs"][0][0], [0, 1])
        self.assertEqual(gd.open_train_validation(manifest, self.guard, snapshot_factory=Record).train()[0].positive_pairs,
                         ((0, 1), (1, 0)))

    def test_symlink_ancestor_and_fifo_manifest_are_refused_without_blocking(self):
        manifest = self.manifest()
        alias = self.base / "alias"; alias.symlink_to(manifest.path.parent, target_is_directory=True)
        with self.assertRaises(gd.DataContractError):
            gd.read_manifest(alias / "manifest.json", manifest.sha256, self.guard)
        fifo_dir = self.base / "fifo"; fifo_dir.mkdir()
        fifo = fifo_dir / "manifest.json"; os.mkfifo(fifo)
        with self.assertRaises(gd.DataContractError):
            gd.read_manifest(fifo, "0" * 64, self.guard)

    def test_strict_manifest_scalars_policy_and_pending_publication(self):
        variants = [("schema_version", True), ("generation_performed", 0),
                    ("config_sha256", "0" * 64), ("kind", "tgrab_primary_converted_v1")]
        for key, value in variants:
            manifest = self.manifest(); meta = manifest.metadata; meta[key] = value
            data = canonical(meta); manifest.path.write_bytes(data)
            with self.subTest(key=key), self.assertRaises(gd.DataContractError):
                gd.read_manifest(manifest.path, sha(data), self.guard)
        manifest = self.manifest()
        pending = manifest.path.with_name("manifest.pending.json")
        pending.write_bytes(manifest.path.read_bytes())
        with self.assertRaises(gd.DataContractError):
            gd.read_manifest(pending, manifest.sha256, self.guard)

    def test_exact_size_precheck_and_manifest_regular_file_size_bound(self):
        manifest = self.manifest()
        with mock.patch.object(gd.os, "read", side_effect=AssertionError("oversized read attempted")):
            with self.assertRaises(gd.DataContractError):
                gd._read(manifest.path, manifest.path.stat().st_size - 1, self.guard)

    def test_input_replacement_during_read_is_detected_before_return(self):
        manifest = self.manifest()
        path = manifest.path.parent / "train.json"
        original = os.read
        replaced = False
        def read(fd, count):
            nonlocal replaced
            data = original(fd, count)
            if not replaced:
                replaced = True
                replacement = path.with_name("replacement.json")
                replacement.write_bytes(path.read_bytes())
                replacement.replace(path)
            return data
        with mock.patch.object(gd.os, "read", side_effect=read), self.assertRaisesRegex(gd.DataContractError, "changed during read"):
            gd._read(path, gd.MAX_FILE, self.guard)
        self.assertTrue(replaced)

    def test_actual_manifest_ceiling_refuses_before_payload_read(self):
        folder = self.base / "oversized"; folder.mkdir()
        path = folder / "manifest.json"
        with path.open("wb") as stream:
            stream.truncate(gd.MAX_MANIFEST + 1)
        with mock.patch.object(gd.os, "read", side_effect=AssertionError("oversized manifest read")):
            with self.assertRaises(gd.DataContractError):
                gd.read_manifest(path, "0" * 64, self.guard)


if __name__ == "__main__":
    unittest.main()
