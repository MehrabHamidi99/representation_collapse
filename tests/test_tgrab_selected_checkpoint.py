import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from representation_collapse import tgrab_selected_checkpoint as codec


ROOT = Path(__file__).resolve().parents[1]


def catalog_descriptors(component, payload):
    catalog = json.loads((ROOT / codec.CATALOG_PATH).read_text())
    out = []
    for entry in catalog["components"][component]["tensors"]:
        start = entry["payload_offset"]
        part = payload[start:start + entry["nbytes"]]
        out.append({
            "name": entry["name"], "dtype": "float32_le", "shape": entry["shape"],
            "order": "C", "offset": start, "nbytes": len(part),
            "data_sha256": hashlib.sha256(part).hexdigest(),
            "semantic_sha256": codec.tensor_semantic_sha256(component, entry["name"], entry["shape"], part),
        })
    return {"filename": catalog["components"][component]["payload_filename"],
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "tensors": out}


def schema_sample(spec, registry):
    if "$ref" in spec:
        return schema_sample(registry["$defs"][spec["$ref"].split("/")[-1]], registry)
    if "const" in spec:
        return json.loads(json.dumps(spec["const"]))
    if "enum" in spec:
        return spec["enum"][0]
    if "oneOf" in spec:
        return schema_sample(spec["oneOf"][0], registry)
    kind = spec.get("type")
    if kind == "object":
        return {k: schema_sample(v, registry) for k, v in spec["properties"].items()}
    if kind == "array":
        return [schema_sample(v, registry) for v in spec["prefixItems"]]
    if kind == "integer":
        return spec.get("minimum", 0)
    if kind == "number":
        return float(spec.get("minimum", 0))
    if kind == "boolean":
        return False
    if kind == "null":
        return None
    if kind == "string":
        pattern = spec.get("pattern", "")
        if "{64}" in pattern:
            return "a" * 64
        if "{40}" in pattern:
            return "b" * 40
        if "00:00" in pattern:
            return "2026-10-16T02:00:00+00:00"
        if pattern.startswith("^[1-9]"):
            return "1"
        if pattern.startswith("^[0-9]"):
            return "1"
        if "(/" in pattern:
            return "fixture/record.json"
        return "fixture"
    raise AssertionError(f"unhandled schema sample {spec}")


def synthetic_valid_bundle():
    schema = json.loads((ROOT / codec.SCHEMA_PATH).read_text())
    manifest = schema_sample(schema["$defs"]["manifest"], schema)
    validation = schema_sample(schema["$defs"]["validation"], schema)
    provenance = schema_sample(schema["$defs"]["provenance"], schema)
    epoch = 1
    manifest["epoch"] = validation["epoch"] = epoch
    provenance["counters"].update(complete_epochs=epoch, validation_epochs=epoch,
                                  attempted_updates=2559, completed_updates=2559)
    for row in validation["rows"]:
        row.update(tp=0, fp=0, fn=2, tn=9898, precision=0.0, recall=0.0,
                   f1=0.0, accuracy=9898 / 9900)
    validation["time_mean_f1"] = 0.0
    identities = manifest["identities"]
    for key in ("admission_sha256", "executed_config_sha256", "reconstruction_config_sha256", "reporting_identity_sha256"):
        identities[key] = provenance[key]
    identities["schema_sha256"] = codec.SCHEMA_SHA256
    identities["tensor_catalog_sha256"] = codec.CATALOG_SHA256
    catalog = json.loads((ROOT / codec.CATALOG_PATH).read_text())
    recurrent = b"\0" * catalog["components"]["recurrent"]["nbytes"]
    link_head = b"\0" * catalog["components"]["link_head"]["nbytes"]
    manifest["components"]["recurrent"] = catalog_descriptors("recurrent", recurrent)
    manifest["components"]["link_head"] = catalog_descriptors("link_head", link_head)
    terminal_bytes = b"\0" * 102400
    terminal = manifest["terminal_state"]
    terminal.update(bytes=len(terminal_bytes), sha256=hashlib.sha256(terminal_bytes).hexdigest(), epoch=epoch,
                    recurrent_sha256=manifest["components"]["recurrent"]["sha256"],
                    link_head_sha256=manifest["components"]["link_head"]["sha256"])
    terminal["tensors"] = []
    for name, offset in (("hidden", 0), ("cell", 51200)):
        chunk = terminal_bytes[offset:offset + 51200]
        terminal["tensors"].append({"name": name, "dtype": "float32_le", "shape": [1, 100, 128],
                                    "order": "C", "offset": offset, "nbytes": len(chunk),
                                    "data_sha256": hashlib.sha256(chunk).hexdigest(),
                                    "semantic_sha256": codec.tensor_semantic_sha256("terminal_state", name, [1, 100, 128], chunk)})
    validation_bytes = codec._canonical(validation)
    provenance_bytes = codec._canonical(provenance)
    manifest["validation"].update(bytes=len(validation_bytes), sha256=hashlib.sha256(validation_bytes).hexdigest(), time_mean_f1=0.0)
    manifest["provenance"].update(bytes=len(provenance_bytes), sha256=hashlib.sha256(provenance_bytes).hexdigest())
    files = {
        "recurrent.f32le": recurrent,
        "link-head.f32le": link_head,
        "validation-terminal.f32le": terminal_bytes,
        "validation.json": validation_bytes,
        "provenance.json": provenance_bytes,
        "manifest.json": codec._canonical(manifest),
    }
    return files


class CanonicalJsonTests(unittest.TestCase):
    def test_canonical_roundtrip_and_closed_artifact_schema(self):
        value = {"kind": "not-a-frozen-kind", "schema_version": 1}
        with self.assertRaises(codec.CheckpointError):
            codec.validate_metadata(codec._canonical(value))
        good_order = b'{"a":1,"z":2}\n'
        self.assertEqual(codec.parse_canonical_json(good_order, max_bytes=20), {"a": 1, "z": 2})

    def test_duplicate_whitespace_newline_unicode_nonfinite_and_token_bounds_reject(self):
        bad = [b'{"a":1,"a":2}\n', b'{"a": 1}\n', b'{"a":1}', b'{"a":"\xc3\xa9"}\n',
               b'{"a":NaN}\n', b'{"a":12345678901}\n', b'{"a":1e999}\n']
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(codec.CheckpointError):
                codec.parse_canonical_json(raw, max_bytes=100)

    def test_schema_and_catalog_are_hash_pinned(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / codec.SCHEMA_PATH).parent.mkdir(parents=True, exist_ok=True)
            (root / codec.SCHEMA_PATH).write_bytes(b"{}")
            (root / codec.CATALOG_PATH).parent.mkdir(parents=True, exist_ok=True)
            (root / codec.CATALOG_PATH).write_bytes(b"{}")
            with mock.patch.object(codec, "_repo_root", return_value=root):
                with self.assertRaisesRegex(codec.CheckpointError, "pinned"):
                    codec._schema_and_catalog()

    def test_schema_candidate_digest_is_config_identity_not_weight_digest(self):
        config = json.loads((ROOT / "configs/graph/tgrab-reconstruction-v1.json").read_text())
        canonical_candidate = json.dumps(config["candidate"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        identity = hashlib.sha256(canonical_candidate).hexdigest()
        schema = json.loads((ROOT / codec.SCHEMA_PATH).read_text())
        declared = schema["$defs"]["provenance"]["properties"]["candidate_sha256"]["const"]
        self.assertEqual(identity, codec.FROZEN_CANDIDATE_IDENTITY_SHA256)
        self.assertEqual(declared, identity)


class TensorCodecTests(unittest.TestCase):
    def test_exact_bytes_roundtrip_preserves_negative_zero_and_subnormal(self):
        payload = bytes.fromhex("00000080 01000000 0000803f ffff7f7f")
        self.assertEqual(codec.decode_tensor_bytes(codec.encode_tensor_bytes(payload, [4]), [4]), payload)

    def test_wrong_shape_length_rank_dimension_and_nonfinite_reject(self):
        for payload, shape in [(b"\0" * 4, [2]), (b"\0" * 4, [1, 1, 1]), (b"\0" * 4, [257]),
                               (b"\0" * 4, [True]), (bytes.fromhex("0000807f"), [1])]:
            with self.subTest(shape=shape), self.assertRaises(codec.CheckpointError):
                codec.encode_tensor_bytes(payload, shape)

    def test_validation_metric_f1_uses_frozen_operation_order(self):
        row = {"pairs": 9900, "tp": 1, "fp": 2, "fn": 3, "tn": 9894,
               "precision": 1 / 3, "recall": 1 / 4,
               "f1": 2 / 7, "accuracy": 9895 / 9900}
        rows = [row] + [{"pairs": 9900, "tp": 0, "fp": 0, "fn": 0, "tn": 9900,
                         "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 1.0} for _ in range(255)]
        validation = {"epoch": 1, "rows": rows, "time_mean_f1": math.fsum([2 / 7] + [0.0] * 255) / 256}
        manifest = {"epoch": 1, "validation": {"time_mean_f1": validation["time_mean_f1"]}}
        codec._validate_validation_score(manifest, validation)
        row["f1"] = 2 * row["precision"] * row["recall"] / (row["precision"] + row["recall"])
        with self.assertRaises(codec.CheckpointError):
            codec._validate_validation_score(manifest, validation)

    def test_frozen_catalog_descriptor_exactness_and_hashes(self):
        catalog = json.loads((ROOT / codec.CATALOG_PATH).read_text())
        for component in ("recurrent", "link_head"):
            frozen = catalog["components"][component]
            payload = b"\0" * frozen["nbytes"]
            descriptor = catalog_descriptors(component, payload)
            codec._verify_component(component, descriptor, payload, catalog)
            changed = json.loads(json.dumps(descriptor))
            changed["tensors"][0]["offset"] += 4
            with self.assertRaises(codec.CheckpointError):
                codec._verify_component(component, changed, payload, catalog)
            changed = json.loads(json.dumps(descriptor))
            changed["tensors"][0]["shape"][0] += 1
            with self.assertRaises(codec.CheckpointError):
                codec._verify_component(component, changed, payload, catalog)
            changed = json.loads(json.dumps(descriptor))
            changed["tensors"][0]["unexpected"] = 1
            with self.assertRaises(codec.CheckpointError):
                codec._verify_component(component, changed, payload, catalog)

    def test_truncation_overlap_nonfinite_and_wrong_data_hash_reject(self):
        catalog = json.loads((ROOT / codec.CATALOG_PATH).read_text())
        payload = b"\0" * catalog["components"]["recurrent"]["nbytes"]
        descriptor = catalog_descriptors("recurrent", payload)
        with self.assertRaises(codec.CheckpointError):
            codec._verify_component("recurrent", descriptor, payload[:-1], catalog)
        for mutation in ("overlap", "data_hash", "nan"):
            changed = json.loads(json.dumps(descriptor))
            data = payload
            if mutation == "overlap":
                changed["tensors"][1]["offset"] = 0
            elif mutation == "data_hash":
                changed["tensors"][0]["data_sha256"] = "0" * 64
            else:
                data = bytes.fromhex("0000807f") + payload[4:]
                changed["sha256"] = hashlib.sha256(data).hexdigest()
                changed["tensors"][0]["data_sha256"] = hashlib.sha256(data[:51200]).hexdigest()
            with self.subTest(mutation=mutation), self.assertRaises(codec.CheckpointError):
                codec._verify_component("recurrent", changed, data, catalog)


class FilesystemDefenseTests(unittest.TestCase):
    def test_complete_bundle_roundtrip_and_exact_six_file_contract(self):
        files = synthetic_valid_bundle()
        with tempfile.TemporaryDirectory() as td:
            bundle = Path(td).resolve() / "candidate"
            written = codec.write_candidate(bundle, files)
            read = codec.read_candidate(bundle)
            self.assertEqual(read["recurrent"], files["recurrent.f32le"])
            self.assertEqual(read["link_head"], files["link-head.f32le"])
            self.assertEqual(read["terminal_state"], files["validation-terminal.f32le"])
            self.assertEqual(set(bundle.iterdir()), {bundle / name for name in codec.FILES})
            self.assertEqual(written["files_sha256"], read["files_sha256"])
            with self.assertRaises(FileExistsError):
                codec.write_candidate(bundle, files)

    def test_symlink_and_hardlink_files_reject(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "real").write_bytes(b"ok")
            (root / "sym").symlink_to(root / "real")
            os.link(root / "real", root / "hard")
            dfd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                with self.assertRaises(OSError):
                    codec._read_stable_at(dfd, "sym", 10)
                with self.assertRaises(codec.CheckpointError):
                    codec._read_stable_at(dfd, "hard", 10)
            finally:
                os.close(dfd)

    def test_directory_symlink_component_and_concurrent_replacement_reject(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "target"
            target.mkdir()
            (target / "payload").write_bytes(b"first")
            (root / "link").symlink_to(target, target_is_directory=True)
            with self.assertRaises(codec.CheckpointError):
                codec._open_bundle_directory(root / "link")
            dfd = os.open(target, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            original_read = os.read
            did_replace = False
            def replacing_read(fd, count):
                nonlocal did_replace
                data = original_read(fd, count)
                if not did_replace:
                    did_replace = True
                    (target / "replacement").write_bytes(b"other")
                    os.replace(target / "replacement", target / "payload")
                return data
            try:
                with mock.patch.object(codec.os, "read", side_effect=replacing_read):
                    with self.assertRaisesRegex(codec.CheckpointError, "replaced or modified"):
                        codec._read_stable_at(dfd, "payload", 10)
            finally:
                os.close(dfd)

    def test_writer_requires_exact_file_set_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td).resolve()
            with self.assertRaises(codec.CheckpointError):
                codec.write_candidate(parent / "new", {"manifest.json": b"{}\n"})
            (parent / "exists").mkdir()
            with self.assertRaises(FileExistsError):
                codec.write_candidate(parent / "exists", {name: b"" for name in codec.FILES})


if __name__ == "__main__":
    unittest.main()
