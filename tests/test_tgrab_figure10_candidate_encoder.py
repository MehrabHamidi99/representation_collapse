from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from representation_collapse import tgrab_selected_checkpoint as codec  # noqa: E402
from representation_collapse.tgrab_figure10_candidate_encoder import (  # noqa: E402
    CandidateEncodingError,
    encode_epoch_candidate,
)
from tests.test_tgrab_selected_checkpoint import schema_sample  # noqa: E402


class TensorFixture:
    def __init__(self, data, *, dtype="torch.float32", device="cpu"):
        self.data = data
        self.shape = data.shape
        self.dtype = dtype
        self.device = device

    def numpy(self):
        return self.data


def _inputs():
    schema = json.loads((ROOT / codec.SCHEMA_PATH).read_text())
    catalog = json.loads((ROOT / codec.CATALOG_PATH).read_text())
    epoch = 1

    provenance = schema_sample(schema["$defs"]["provenance"], schema)
    provenance["source_sha256"]["schema"] = codec.SCHEMA_SHA256
    provenance["counters"].update(complete_epochs=epoch, validation_epochs=epoch,
                                  attempted_updates=2559, completed_updates=2559,
                                  selected_load_attempts=0, test_payload_read_attempts=0)
    identities = {
        "executed_config_sha256": provenance["executed_config_sha256"],
        "schema_sha256": codec.SCHEMA_SHA256,
        "codec_source_sha256": provenance["source_sha256"]["codec"],
        "trainer_source_sha256": provenance["source_sha256"]["trainer"],
        "environment_receipt_sha256": provenance["environment"]["environment_receipt_sha256"],
        "converted_manifest_sha256": "a" * 64,
        "train_file_sha256": "b" * 64,
        "validation_file_sha256": "c" * 64,
        "test_file_sha256": "d" * 64,
        "train_semantic_sha256": "e" * 64,
        "validation_semantic_sha256": "f" * 64,
        "test_semantic_sha256": "0" * 64,
        "node_order_semantic_sha256": "76433a9aa16a685cb0e3b5744bd8c94fc520115eab962e4d7213c95ffd28e2cd",
        "feature_encoding_semantic_sha256": "d9fb41856b23844dd56bd8f849fc065a4cea397abb6bede2a737de2b0290a94a",
        "weight_encoding_semantic_sha256": "afff57dd0a1965eb0cbeb406a07c5bb6d13ed2676f64c09c4eaf8483e7c3c659",
        "conversion_receipt_sha256": "1" * 64,
        "admission_sha256": provenance["admission_sha256"],
        "reporting_identity_sha256": provenance["reporting_identity_sha256"],
        "tensor_catalog_sha256": codec.CATALOG_SHA256,
        "raw_sha256": "5a19f9e7949668473bc61cb59b6215f25f093ea0ee2d45b5e143338ded5523e3",
        "reconstruction_config_sha256": provenance["reconstruction_config_sha256"],
        "converter_sha256": "498e60f7be048411c253e4b8cb63e4b907973f969c397bfdfacde949389f057e",
        "conversion_protocol_sha256": "003e6b01d3b090841f623f75ed0648a397fe8afdaeb49dc382fb69a3f0e05584",
    }
    provenance["source_sha256"]["codec"] = identities["codec_source_sha256"]
    provenance["source_sha256"]["trainer"] = identities["trainer_source_sha256"]
    provenance["environment"]["environment_receipt_sha256"] = identities["environment_receipt_sha256"]

    rows = []
    row_specs = schema["$defs"]["validation"]["properties"]["rows"]["prefixItems"]
    for i, spec in enumerate(row_specs):
        row = schema_sample(spec, schema)
        t = 2560 + i
        row.update(target_timestamp=t, source_timestamp=t - 1,
                   target_phase=t % 64, source_phase=(t - 1) % 64,
                   pairs=9900, tp=0, fp=0, fn=0, tn=9900,
                   precision=0.0, recall=0.0, f1=0.0, accuracy=1.0)
        rows.append({"epoch": epoch, **row})

    def tensor(shape, fill=0):
        return TensorFixture(np.full(shape, fill, dtype=np.float32))

    components = {}
    for component in ("recurrent", "link_head"):
        spec = catalog["components"][component]
        state = {}
        descriptors = {item["name"]: item for item in spec["tensors"]}
        for name in spec["native_state_dict_order"]:
            item = descriptors[name]
            state[item["name"]] = tensor(tuple(item["shape"]))
        components[spec["backend_key"]] = state
    evidence = SimpleNamespace(
        record=SimpleNamespace(epoch=epoch, attempted_updates=2559, completed_updates=2559,
                               optimizer_steps_confirmed=2559, validation_mean_f1=0.0),
        terminal_state=SimpleNamespace(next_target_t=2816, node_ids=tuple(range(100)),
                                        hidden=tensor((1, 100, 128)), cell=tensor((1, 100, 128))),
        model_state_dicts=components,
        validation_rows=tuple(rows),
    )
    return evidence, provenance, {"identities": identities}


class CandidateEncoderTests(unittest.TestCase):
    def test_exact_bundle_preserves_fp32_bits_and_read_candidate_accepts(self):
        evidence, provenance, manifest = _inputs()
        # Check signed zero and the smallest positive subnormal survive unchanged.
        first = evidence.model_state_dicts["node_emb"]["units.0.W_c"]
        first.data.flat[0] = np.frombuffer(bytes.fromhex("00000080"), dtype="<f4")[0]
        first.data.flat[1] = np.frombuffer(bytes.fromhex("01000000"), dtype="<f4")[0]
        files = encode_epoch_candidate(evidence, provenance=provenance, manifest=manifest)
        self.assertEqual(set(files), set(codec.FILES))
        self.assertEqual(files["recurrent.f32le"][:8], bytes.fromhex("00000080 01000000"))
        with tempfile.TemporaryDirectory() as td:
            target = Path(td).resolve() / "candidate"
            codec.write_candidate(target, files)
            parsed = codec.read_candidate(target)
        self.assertEqual(parsed["recurrent"][:8], bytes.fromhex("00000080 01000000"))
        self.assertEqual(parsed["validation"]["time_mean_f1"], 0.0)

    def test_provenance_and_manifest_identity_omissions_reject(self):
        evidence, provenance, manifest = _inputs()
        bad = copy.deepcopy(provenance)
        del bad["source_sha256"]["trainer"]
        with self.assertRaises(CandidateEncodingError):
            encode_epoch_candidate(evidence, provenance=bad, manifest=manifest)
        bad_manifest = copy.deepcopy(manifest)
        del bad_manifest["identities"]["validation_file_sha256"]
        with self.assertRaises(CandidateEncodingError):
            encode_epoch_candidate(evidence, provenance=provenance, manifest=bad_manifest)

    def test_wrong_counters_or_cross_identity_reject(self):
        evidence, provenance, manifest = _inputs()
        bad = copy.deepcopy(provenance)
        bad["counters"]["completed_updates"] = 2558
        with self.assertRaises(CandidateEncodingError):
            encode_epoch_candidate(evidence, provenance=bad, manifest=manifest)
        bad_manifest = copy.deepcopy(manifest)
        bad_manifest["identities"]["admission_sha256"] = "9" * 64
        with self.assertRaises(CandidateEncodingError):
            encode_epoch_candidate(evidence, provenance=provenance, manifest=bad_manifest)

    def test_schema_identity_must_match_provenance_schema_source_pin(self):
        evidence, provenance, manifest = _inputs()
        provenance["source_sha256"]["schema"] = "9" * 64
        with self.assertRaisesRegex(CandidateEncodingError, "schema_sha256"):
            encode_epoch_candidate(evidence, provenance=provenance, manifest=manifest)

    def test_device_dtype_shape_and_catalog_tampering_reject(self):
        for attribute, value in (("device", "cuda:0"), ("dtype", "torch.float16")):
            evidence, provenance, manifest = _inputs()
            setattr(evidence.model_state_dicts["node_emb"]["units.0.W_c"], attribute, value)
            with self.subTest(attribute=attribute), self.assertRaises(CandidateEncodingError):
                encode_epoch_candidate(evidence, provenance=provenance, manifest=manifest)
        evidence, provenance, manifest = _inputs()
        del evidence.model_state_dicts["node_emb"]["units.0.W_c"]
        with self.assertRaises(CandidateEncodingError):
            encode_epoch_candidate(evidence, provenance=provenance, manifest=manifest)
        evidence, provenance, manifest = _inputs()
        original = evidence.model_state_dicts["node_emb"]["units.0.W_c"]
        original.data = np.zeros((100, 127), dtype=np.float32)
        original.shape = original.data.shape
        with self.assertRaises(CandidateEncodingError):
            encode_epoch_candidate(evidence, provenance=provenance, manifest=manifest)

    def test_score_order_and_terminal_chronology_tampering_reject(self):
        evidence, provenance, manifest = _inputs()
        evidence.validation_rows[0]["target_timestamp"] = 2561
        with self.assertRaises(CandidateEncodingError):
            encode_epoch_candidate(evidence, provenance=provenance, manifest=manifest)
        evidence, provenance, manifest = _inputs()
        evidence.validation_rows[0]["f1"] = 0.25
        with self.assertRaises(CandidateEncodingError):
            encode_epoch_candidate(evidence, provenance=provenance, manifest=manifest)
        evidence, provenance, manifest = _inputs()
        evidence.terminal_state.next_target_t = 2815
        with self.assertRaises(CandidateEncodingError):
            encode_epoch_candidate(evidence, provenance=provenance, manifest=manifest)


if __name__ == "__main__":
    unittest.main()
