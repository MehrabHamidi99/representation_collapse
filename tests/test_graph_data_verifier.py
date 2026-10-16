"""Synthetic producer/verifier fixtures only; never opens primary target data."""
import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from representation_collapse import graph_data as producer
from representation_collapse import graph_data_verifier as verifier


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/graph/tgrab-reconstruction-v1.json"
SOURCE = ROOT / "src/representation_collapse/graph_data.py"


def digest(data):
    return hashlib.sha256(data).hexdigest()


class GraphDataVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.base = Path(self.temp.name)
        self.calls = 0
        self.counter = 0
        self.spec = verifier.VerificationSpec(4, 2, 1, 1, 1)

    def tearDown(self):
        self.temp.cleanup()

    def guard(self):
        self.calls += 1

    def rows(self, spec):
        result = []
        for t in range(spec.intervals()[-1][2]):
            edges = ((0, 1), (1, 0)) if (t // spec.block) % 2 == 0 else ((2, 3), (3, 2))
            masks = [str(start <= t < end) for _, start, end in spec.intervals()] + ["False"]
            result.extend([[str(u), str(v), str(t), "1.00", *masks] for u, v in edges])
        return result

    def write_raw(self, path, rows):
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("src", "dst", "t", "edge_feat", "train_mask", "val_mask", "test_mask", "test_inductive_mask"))
        writer.writerows(rows)
        path.write_bytes(stream.getvalue().encode("ascii"))

    def fixture(self, *, rows=None, spec=None):
        self.counter += 1
        spec = spec or self.spec
        raw = self.base / f"raw-{self.counter}.csv"
        self.write_raw(raw, self.rows(spec) if rows is None else rows)
        manifest = producer.convert_primary_csv(raw, self.base / f"converted-{self.counter}",
            expected_raw_sha256=digest(raw.read_bytes()), expected_raw_bytes=raw.stat().st_size,
            config_path=CONFIG, converter_revision="e" * 40, guard=self.guard,
            fixture_only=True, spec=producer.DatasetSpec(**spec.record()))
        kwargs = dict(expected_raw_sha256=digest(raw.read_bytes()), expected_raw_bytes=raw.stat().st_size,
                      expected_manifest_sha256=digest(manifest.read_bytes()),
                      expected_converter_sha256=digest(SOURCE.read_bytes()), expected_converter_revision="e" * 40,
                      config_path=CONFIG, guard=self.guard, fixture_only=True, spec=spec)
        return raw, manifest, kwargs

    def test_verifier_has_no_producer_or_numerical_import_or_call(self):
        tree = ast.parse(Path(verifier.__file__).read_text())
        allowed = {"__future__", "csv", "dataclasses", "hashlib", "io", "json", "os", "pathlib", "re", "stat"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name in allowed for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("eval", "exec", "__import__")
                             for n in ast.walk(tree)))

    def test_fixture_receipt_counts_and_no_output_writes(self):
        raw, manifest, kwargs = self.fixture()
        before = {str(p.relative_to(self.base)): p.read_bytes() for p in self.base.rglob("*") if p.is_file()}
        result = verifier.verify_primary_conversion(raw, manifest, **kwargs)
        self.assertEqual(result["kind"], "tgrab_fixture_conversion_independent_verification_v1")
        self.assertFalse(result["primary_eligible"])
        self.assertFalse(result["scientific_acceptance"])
        self.assertTrue(result["independent_of_producer"])
        for key, value in (("snapshot_count", 12), ("split_count", 3), ("raw_rows", 24),
                           ("unique_offdiagonal_edges", 24), ("identical_duplicate_rows", 0),
                           ("converted_snapshot_comparisons", 12), ("semantic_snapshot_comparisons", 12),
                           ("complete_manifest_byte_comparisons", 1), ("stable_input_files_checked", 7)):
            self.assertEqual(result[key], value)
        self.assertEqual(before, {str(p.relative_to(self.base)): p.read_bytes() for p in self.base.rglob("*") if p.is_file()})
        json.dumps(result, allow_nan=False)

    def test_full_primary_dimensions_stay_visibly_synthetic_and_check_every_step(self):
        raw, manifest, kwargs = self.fixture(spec=verifier.PRIMARY_SPEC)
        receipt = verifier.verify_primary_conversion(raw, manifest, **kwargs)
        self.assertEqual(receipt["snapshot_count"], 3072)
        self.assertEqual(receipt["converted_snapshot_comparisons"], 3072)
        self.assertEqual(receipt["raw_rows"], 6144)
        self.assertFalse(receipt["primary_eligible"])
        self.assertEqual(receipt["split_hashes"]["train"]["interval"], [0, 2560])
        self.assertEqual(receipt["split_hashes"]["validation"]["interval"], [2560, 2816])
        self.assertEqual(receipt["split_hashes"]["test"]["interval"], [2816, 3072])

    def test_duplicate_reorder_and_lossless_zero_fraction_tokens(self):
        rows = self.rows(self.spec)
        rows.append(rows[0][:])
        rows.reverse()
        for row in rows:
            row[:3] = [token + ".000" for token in row[:3]]
            row[4:] = ["1" if token == "True" else "0" for token in row[4:]]
        raw, manifest, kwargs = self.fixture(rows=rows)
        result = verifier.verify_primary_conversion(raw, manifest, **kwargs)
        self.assertEqual(result["raw_rows"], 25)
        self.assertEqual(result["identical_duplicate_rows"], 1)
        self.assertEqual(result["unique_offdiagonal_edges"], 24)

    def test_rebound_complete_manifest_cannot_change_counts_provenance_or_types(self):
        for change in (lambda m: m["counts"].update(raw_rows=25, identical_duplicate_rows=1),
                       lambda m: m["converter"].update(revision="a" * 40),
                       lambda m: m.update(schema_version=True),
                       lambda m: m.update(features="float64_identity"),
                       lambda m: m["origin"].update(unexpected=True)):
            raw, manifest, kwargs = self.fixture()
            m = json.loads(manifest.read_bytes()); change(m)
            manifest.write_bytes((json.dumps(m, sort_keys=True, separators=(",", ":")) + "\n").encode())
            kwargs["expected_manifest_sha256"] = digest(manifest.read_bytes())
            with self.assertRaisesRegex(verifier.VerificationError, "complete converted manifest"):
                verifier.verify_primary_conversion(raw, manifest, **kwargs)

    def test_payload_semantics_corruption_fails_even_after_rebinding_own_hash(self):
        raw, manifest, kwargs = self.fixture()
        payload_path = manifest.parent / "test.json"
        payload = json.loads(payload_path.read_bytes())
        payload["snapshots"][-1]["positive_pairs"] = [[0, 1], [1, 0]]
        data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        payload_path.write_bytes(data)
        m = json.loads(manifest.read_bytes()); m["splits"]["test"].update(bytes=len(data), sha256=digest(data))
        manifest.write_bytes((json.dumps(m, sort_keys=True, separators=(",", ":")) + "\n").encode())
        kwargs["expected_manifest_sha256"] = digest(manifest.read_bytes())
        with self.assertRaisesRegex(verifier.VerificationError, "converted test snapshots"):
            verifier.verify_primary_conversion(raw, manifest, **kwargs)

    def test_invalid_raw_lexical_fields_are_rejected_before_artifact_agreement(self):
        raw, manifest, kwargs = self.fixture()
        for column, token in ((0, "1.00000000000000001"), (1, "-0"), (2, "1e0"),
                              (3, "1.00000000000000001"), (3, "NaN"), (4, "1.0")):
            rows = self.rows(self.spec); rows[0][column] = token; self.write_raw(raw, rows)
            kwargs.update(expected_raw_sha256=digest(raw.read_bytes()), expected_raw_bytes=raw.stat().st_size)
            with self.subTest(column=column, token=token), self.assertRaises(verifier.VerificationError):
                verifier.verify_primary_conversion(raw, manifest, **kwargs)

    def test_invalid_raw_graph_and_masks_are_not_repaired(self):
        raw, manifest, kwargs = self.fixture()
        variants = []
        variants.append(self.rows(self.spec)[2:])
        variants.append(self.rows(self.spec)[1:])
        bad_mask = self.rows(self.spec); bad_mask[0][4] = "False"; variants.append(bad_mask)
        marker = [r for r in self.rows(self.spec) if r[2] != "0"]
        marker.append(["0", "0", "0", "1", "True", "False", "False", "False"]); variants.append(marker)
        dense = self.rows(self.spec)
        for t in range(12):
            if t % 4 < 2:
                mask = next(r[4:] for r in dense if r[2] == str(t))
                dense.extend([[str(u), str(v), str(t), "1", *mask]
                              for u, v in ((0, 2), (2, 0), (0, 3), (3, 0))])
        variants.append(dense)
        for i, rows in enumerate(variants):
            self.write_raw(raw, rows)
            kwargs.update(expected_raw_sha256=digest(raw.read_bytes()), expected_raw_bytes=raw.stat().st_size)
            with self.subTest(variant=i), self.assertRaises(verifier.VerificationError):
                verifier.verify_primary_conversion(raw, manifest, **kwargs)

    def test_wrong_hashes_revision_config_and_fixture_promotion_fail(self):
        for key, value in (("expected_raw_sha256", "0" * 64), ("expected_raw_bytes", 1),
                           ("expected_manifest_sha256", "0" * 64), ("expected_converter_sha256", "0" * 64),
                           ("expected_converter_revision", "0" * 40), ("fixture_only", False)):
            raw, manifest, kwargs = self.fixture(); kwargs[key] = value
            with self.subTest(key=key), self.assertRaises(verifier.VerificationError):
                verifier.verify_primary_conversion(raw, manifest, **kwargs)
        raw, manifest, kwargs = self.fixture()
        changed = self.base / "config.json"; changed.write_bytes(CONFIG.read_bytes() + b" ")
        kwargs["config_path"] = changed
        with self.assertRaisesRegex(verifier.VerificationError, "config mismatch"):
            verifier.verify_primary_conversion(raw, manifest, **kwargs)

    def test_links_pending_manifest_and_growth_are_rejected(self):
        raw, manifest, kwargs = self.fixture()
        alias = self.base / "alias"; alias.symlink_to(manifest.parent, target_is_directory=True)
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_primary_conversion(raw, alias / "manifest.json", **kwargs)
        hard = self.base / "hard.csv"; os.link(raw, hard)
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_primary_conversion(raw, manifest, **kwargs)
        hard.unlink()
        pending = manifest.with_name("manifest.pending.json"); pending.write_bytes(manifest.read_bytes())
        with self.assertRaises(verifier.VerificationError):
            verifier.verify_primary_conversion(raw, pending, **kwargs)
        original = verifier._raw_oracle
        def mutate(data, spec, guard):
            value = original(data, spec, guard)
            raw.write_bytes(raw.read_bytes() + b"\n")
            return value
        with mock.patch.object(verifier, "_raw_oracle", side_effect=mutate), self.assertRaisesRegex(verifier.VerificationError, "changed during verification"):
            verifier.verify_primary_conversion(raw, manifest, **kwargs)

    def test_raising_guard_cannot_return_success_or_write_outputs(self):
        raw, manifest, kwargs = self.fixture()
        before = {p: p.read_bytes() for p in self.base.rglob("*") if p.is_file()}
        def stop():
            raise RuntimeError("owner stop")
        kwargs["guard"] = stop
        with self.assertRaisesRegex(RuntimeError, "owner stop"):
            verifier.verify_primary_conversion(raw, manifest, **kwargs)
        self.assertEqual(before, {p: p.read_bytes() for p in self.base.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
