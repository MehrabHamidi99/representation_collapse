"""Synthetic data-contract fixtures only; never opens the primary dataset."""
from dataclasses import dataclass
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from representation_collapse import graph_data as gd


CONFIG = Path(__file__).resolve().parents[1] / "configs/graph/tgrab-reconstruction-v1.json"


@dataclass(frozen=True)
class FixtureSnapshot:
    timestamp: int
    nodes: int
    positive_pairs: tuple
    node_ids: tuple


class GraphDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.base = Path(self.temp.name)
        self.spec = gd.DatasetSpec(4, 2, 1, 1, 1)
        self.raw = self.base / "fixture.csv"
        self.rows = []
        for t in range(12):
            pairs = ((0, 1), (1, 0)) if t % 4 < 2 else ((2, 3), (3, 2))
            for u, v in pairs:
                self.rows.append([str(u), str(v), str(t), "1.0", str(t < 4), str(4 <= t < 8), str(t >= 8), "False"])
        self.guard_calls = 0

    def tearDown(self):
        self.temp.cleanup()

    def guard(self):
        self.guard_calls += 1

    def write(self, rows=None, header=gd.HEADER):
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(self.rows if rows is None else rows)
        self.raw.write_text(stream.getvalue(), encoding="ascii")

    def convert(self, rows=None, output="converted", **changes):
        self.write(rows)
        kwargs = dict(expected_raw_sha256=hashlib.sha256(self.raw.read_bytes()).hexdigest(),
                      expected_raw_bytes=self.raw.stat().st_size, config_path=CONFIG,
                      converter_revision="a" * 40, guard=self.guard, spec=self.spec, fixture_only=True)
        kwargs.update(changes)
        return gd.convert_primary_csv(self.raw, self.base / output, **kwargs)

    def verified(self):
        path = self.convert()
        return gd.read_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest(), self.guard)

    def changed_manifest(self, manifest, mutate):
        value = manifest.metadata
        mutate(value)
        manifest.path.write_bytes(gd.canonical_json(value))
        return gd.read_manifest(manifest.path, hashlib.sha256(manifest.path.read_bytes()).hexdigest(), self.guard)

    def test_stdlib_import_does_not_import_numpy_or_model(self):
        result = subprocess.run([sys.executable, "-B", "-c", "import sys; from representation_collapse import graph_data; assert not any(k.split('.')[0] in ('numpy','torch','wandb') for k in sys.modules)"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_deterministic_conversion_semantics_and_counts(self):
        first = self.convert(self.rows + [self.rows[0]])
        second = self.convert(self.rows + [self.rows[0]], output="again")
        self.assertEqual(first.read_bytes(), second.read_bytes())
        m = gd.read_manifest(first, hashlib.sha256(first.read_bytes()).hexdigest(), self.guard)
        data = m.metadata
        self.assertEqual(data["kind"], "tgrab_fixture_converted_v1")
        self.assertEqual(data["counts"], {"raw_rows": 25, "identical_duplicate_rows": 1, "source_marker_rows": 0, "unique_offdiagonal_edges": 24})
        self.assertNotEqual(data["splits"]["train"]["sha256"], data["splits"]["train"]["semantic_sha256"])
        self.assertNotEqual(data["snapshot_semantic_sha256"][0], data["snapshot_semantic_sha256"][1])
        self.assertTrue(self.guard_calls > 24)

    def test_raw_order_only_changes_file_identity(self):
        one = self.convert()
        a = json.loads(one.read_bytes())
        two = self.convert(list(reversed(self.rows)), output="reverse")
        b = json.loads(two.read_bytes())
        self.assertNotEqual(a["raw"]["sha256"], b["raw"]["sha256"])
        self.assertEqual(a["snapshot_semantic_sha256"], b["snapshot_semantic_sha256"])
        self.assertEqual(a["splits"], b["splits"])

    def test_integral_zero_fraction_and_boolean_policy(self):
        rows = [row[:] for row in self.rows]
        for row in rows:
            row[:3] = [v + ".00" for v in row[:3]]
            row[4:] = ["1" if v == "True" else "0" for v in row[4:]]
        self.convert(rows)
        for column, token in ((0, "01"), (0, "-0"), (1, "1.5"), (2, "1e0"), (2, " 1"), (3, "nan"), (3, "inf"), (3, "0.99999999999999999"), (4, "true"), (7, "False ")):
            rows = [row[:] for row in self.rows]
            rows[0][column] = token
            with self.subTest(column=column, token=token), self.assertRaises(gd.DataContractError):
                self.convert(rows, output=f"bad-{column}-{hashlib.sha256(token.encode()).hexdigest()[:5]}")

    def test_primary_cannot_be_relabelled_fixture(self):
        with self.assertRaises(gd.DataContractError):
            self.convert(fixture_only=False)
        with self.assertRaises(gd.DataContractError):
            self.convert(spec=gd.PRIMARY_SPEC, fixture_only=False)

    def test_exact_weight_node_mask_and_full_negative_guards(self):
        variants = []
        for index, value in ((0, "4"), (1, "4"), (2, "12"), (3, "2"), (4, "False"), (7, "True")):
            rows = [r[:] for r in self.rows]; rows[0][index] = value; variants.append(rows)
        rows = [r[:] for r in self.rows]
        for t in range(12):
            if t % 4 < 2:
                for u, v in ((0, 2), (2, 0), (0, 3), (3, 0)):
                    template = next(r for r in rows if r[2] == str(t))
                    rows.append([str(u), str(v), *template[2:]])
        variants.append(rows)
        for i, rows in enumerate(variants):
            with self.subTest(i=i), self.assertRaises(gd.DataContractError):
                self.convert(rows, output=f"case-{i}")

    def test_missing_timestamp_reverse_pattern_and_markers_reject(self):
        variants = [self.rows[2:], self.rows[1:]]
        same = [r[:] for r in self.rows]
        for r in same:
            r[0], r[1] = ("0", "1") if r[0] in ("0", "2") else ("1", "0")
        variants.append(same)
        mixed = [r[:] for r in self.rows]; mixed.append(["0", "0", *self.rows[0][2:]]); variants.append(mixed)
        marker = [r[:] for r in self.rows if r[2] != "0"]
        marker.append(["0", "0", *self.rows[0][2:]]); variants.append(marker)
        other = [r[:] for r in self.rows]; other[0][0:2] = ["1", "1"]; variants.append(other)
        wrong = [r[:] for r in self.rows]; wrong[2][0:2] = ["2", "3"]; variants.append(wrong)
        for i, rows in enumerate(variants):
            with self.subTest(i=i), self.assertRaises(gd.DataContractError):
                self.convert(rows, output=f"case-{i}")
            self.assertFalse((self.base / f"case-{i}" / "manifest.json").exists())
            self.assertTrue((self.base / f"case-{i}" / "failure.json").is_file())

    def test_failed_output_preserved_and_not_overwritten(self):
        with self.assertRaises(gd.DataContractError):
            self.convert(expected_raw_sha256="0" * 64)
        old = (self.base / "converted" / "failure.json").read_bytes()
        with self.assertRaises(gd.DataContractError):
            self.convert()
        self.assertEqual(old, (self.base / "converted" / "failure.json").read_bytes())

    def test_marker_failure_retains_timestamp_and_counts(self):
        rows = [r[:] for r in self.rows if r[2] != "0"]
        rows.extend([["0", "0", *self.rows[0][2:]]] * 2)
        with self.assertRaisesRegex(gd.DataContractError, "timestamp 0 after source marker resolution"):
            self.convert(rows)
        failure = json.loads((self.base / "converted" / "failure.json").read_bytes())
        self.assertEqual(failure["source_marker_timestamps"], [0])
        self.assertEqual(failure["partial_counts"]["source_marker_rows"], 2)
        self.assertEqual(failure["partial_counts"]["identical_duplicate_rows"], 1)
        self.assertFalse((self.base / "converted" / "manifest.json").exists())

    def test_train_validation_do_not_read_or_require_test(self):
        manifest = self.verified()
        test = manifest.path.parent / "test.json"
        test.unlink()
        calls = []
        original = gd._read
        def read(path, *args):
            calls.append(Path(path).name)
            return original(path, *args)
        with mock.patch.object(gd, "_read", side_effect=read):
            store = gd.open_train_validation(manifest, self.guard, snapshot_factory=FixtureSnapshot)
        self.assertNotIn("test.json", calls)
        self.assertEqual(tuple(s.timestamp for s in store.train()), (0, 1, 2, 3))
        self.assertEqual(tuple(s.timestamp for s in store.validation()), (4, 5, 6, 7))
        self.assertEqual(store.train()[0].positive_pairs, ((0, 1), (1, 0)))

    def selection(self):
        from representation_collapse.graph_reconstruction import CheckpointReference, ValidationOnlySelection
        selection = ValidationOnlySelection()
        selection.consider(CheckpointReference(1, "a" * 64, "b" * 64, "c" * 64), 0.5, split="validation", epoch_complete=True)
        selection.finalize()
        return selection

    def test_actual_snapshot_and_selection_success_laziness_and_terminal_access(self):
        from representation_collapse.graph_reconstruction import Snapshot
        manifest = self.verified()
        store = gd.open_train_validation(manifest, self.guard)
        self.assertIs(type(store.train()[0]), Snapshot)
        selection = self.selection()
        opener = store.sealed_test_opener(selection)
        calls = []
        original = gd._read
        def read(path, *args):
            calls.append(Path(path).name)
            return original(path, *args)
        with mock.patch.object(gd, "_read", side_effect=read):
            iterator = opener()
            self.assertEqual(calls, [])
            self.assertEqual(next(iterator).timestamp, 8)
            self.assertIn("test.json", calls)
        self.assertEqual(tuple(s.timestamp for s in iterator), (9, 10, 11))
        for function in (opener, store.train, store.validation, lambda: store.sealed_test_opener(selection)):
            with self.assertRaises(gd.DataContractError):
                function()

    def test_failed_lazy_test_read_consumed(self):
        manifest = self.verified()
        store = gd.open_train_validation(manifest, self.guard, snapshot_factory=FixtureSnapshot)
        opener = store.sealed_test_opener(self.selection())
        (manifest.path.parent / "test.json").unlink()
        iterator = opener()
        with self.assertRaises(gd.DataContractError):
            next(iterator)
        with self.assertRaises(gd.DataContractError):
            opener()
        with self.assertRaises(gd.DataContractError):
            store.train()

    def test_reject_ducktyped_unfinalized_mutated_selection(self):
        from types import SimpleNamespace
        from representation_collapse.graph_reconstruction import ValidationOnlySelection
        store = gd.open_train_validation(self.verified(), self.guard, snapshot_factory=FixtureSnapshot)
        for invalid in (SimpleNamespace(finalized=True, selected="fake"), ValidationOnlySelection()):
            with self.assertRaises(gd.DataContractError):
                store.sealed_test_opener(invalid)
        selection = self.selection()
        opener = store.sealed_test_opener(selection)
        selection.finalized = False
        with self.assertRaises(gd.DataContractError):
            opener()

    def test_selection_mutation_between_callback_and_first_next(self):
        store = gd.open_train_validation(self.verified(), self.guard, snapshot_factory=FixtureSnapshot)
        selection = self.selection()
        opener = store.sealed_test_opener(selection)
        iterator = opener()
        selection.finalized = False
        calls = []
        original = gd._read
        def read(path, *args):
            calls.append(Path(path).name)
            return original(path, *args)
        with mock.patch.object(gd, "_read", side_effect=read):
            with self.assertRaises(gd.DataContractError):
                next(iterator)
        self.assertEqual(calls, [])
        with self.assertRaises(gd.DataContractError):
            opener()
        with self.assertRaises(gd.DataContractError):
            opener()

    def test_manifest_immutable_copy_and_disk_replacement_rejected(self):
        manifest = self.verified()
        copy = manifest.metadata; copy["splits"]["train"]["filename"] = "test.json"
        self.assertEqual(manifest.metadata["splits"]["train"]["filename"], "train.json")
        manifest.path.write_bytes(b"{}\n")
        with self.assertRaises(gd.DataContractError):
            gd.open_train_validation(manifest, self.guard, snapshot_factory=FixtureSnapshot)

    def test_manifest_paths_nonfinite_duplicate_and_semantics_fail_closed(self):
        mutations = [lambda m: m["splits"]["train"].update(filename="../test.json"),
                     lambda m: m["splits"]["train"].update(interval=[False, 4]),
                     lambda m: m["counts"].update(raw_rows=True),
                     lambda m: m["snapshot_semantic_sha256"].__setitem__(0, "0" * 64),
                     lambda m: m["converter"].update(revision="bad"),
                     lambda m: m.update(generation_performed=0)]
        for i, mutate in enumerate(mutations):
            path = self.convert(output=f"manifest-{i}")
            manifest = gd.read_manifest(path, hashlib.sha256(path.read_bytes()).hexdigest(), self.guard)
            with self.subTest(i=i), self.assertRaises(gd.DataContractError):
                self.changed_manifest(manifest, mutate)
        for i, data in enumerate((b'{"a":1,"a":2}\n', b'{"a":NaN}\n')):
            folder = self.base / f"json-{i}"; folder.mkdir(); path = folder / "manifest.json"; path.write_bytes(data)
            with self.assertRaises(gd.DataContractError):
                gd.read_manifest(path, hashlib.sha256(data).hexdigest(), self.guard)

    def test_symlink_hardlink_traversal_and_split_tamper_reject(self):
        manifest = self.verified()
        train = manifest.path.parent / "train.json"
        original = train.read_bytes(); train.unlink(); train.symlink_to(manifest.path.parent / "test.json")
        with self.assertRaises(gd.DataContractError):
            gd.open_train_validation(manifest, self.guard, snapshot_factory=FixtureSnapshot)
        train.unlink(); train.write_bytes(original + b" ")
        with self.assertRaises(gd.DataContractError):
            gd.open_train_validation(manifest, self.guard, snapshot_factory=FixtureSnapshot)
        import os
        hard = self.base / "hard-manifest.json"; os.link(manifest.path, hard)
        with self.assertRaises(gd.DataContractError):
            gd.read_manifest(manifest.path, manifest.sha256, self.guard)
        with self.assertRaises(gd.DataContractError):
            gd.read_manifest(manifest.path.parent / ".." / "converted" / "manifest.json", manifest.sha256, self.guard)

    def test_guard_failure_consumes_test_attempt(self):
        store = gd.open_train_validation(self.verified(), self.guard, snapshot_factory=FixtureSnapshot)
        opener = store.sealed_test_opener(self.selection())
        def stop():
            raise RuntimeError("owner stop")
        store.guard = stop
        with self.assertRaises(RuntimeError):
            opener()
        with self.assertRaises(gd.DataContractError):
            opener()


if __name__ == "__main__":
    unittest.main()
