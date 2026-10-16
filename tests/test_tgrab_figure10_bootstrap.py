import json
import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np

from src.representation_collapse import graph_reconstruction as gr
from src.representation_collapse import tgrab_figure10_bootstrap as boot


class Tensor:
    dtype = "torch.float32"
    device = "cpu"
    layout = "strided"

    def __init__(self, array):
        self.array = np.asarray(array, dtype="<f4")
        self.shape = self.array.shape

    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return self.array
    def copy_(self, other): self.array[...] = other.array
    def add_(self, value): self.array[...] += value; return self
    def is_contiguous(self): return self.array.flags.c_contiguous
    def storage_offset(self):
        base = self.array
        while isinstance(getattr(base, "base", None), np.ndarray): base = base.base
        return (self.array.__array_interface__["data"][0] - base.__array_interface__["data"][0]) // 4
    def numel(self): return self.array.size
    def element_size(self): return 4
    def stride(self):
        strides = []
        running = 1
        for size in reversed(self.array.shape): strides.insert(0, running); running *= size
        return tuple(strides)
    def untyped_storage(self): return Storage(self.array)


class Storage:
    def __init__(self, array):
        base = array
        while isinstance(getattr(base, "base", None), np.ndarray): base = base.base
        self.base = base
    def data_ptr(self): return int(self.base.__array_interface__["data"][0])
    def nbytes(self): return int(self.base.nbytes)


class ByteTensor:
    dtype = "torch.uint8"
    def __init__(self, value): self.array = np.asarray(value, dtype=np.uint8)
    def cpu(self): return self
    def numpy(self): return self.array


class FakeTorch:
    Tensor = Tensor
    strided = "strided"
    float32 = "torch.float32"
    uint8 = "torch.uint8"
    rng = b"initial-fake-rng"

    @staticmethod
    def isfinite(value): return np.isfinite(value.array)
    @staticmethod
    def from_numpy(array): return Tensor(array)
    @staticmethod
    def tensor(value, dtype=None): return ByteTensor(value)
    @staticmethod
    def get_rng_state(): return ByteTensor(np.frombuffer(FakeTorch.rng, dtype=np.uint8).copy())
    @staticmethod
    def set_rng_state(value): FakeTorch.rng = value.array.tobytes()
    @staticmethod
    def no_grad(): return nullcontext()


class FakeModel:
    def __init__(self, tensors):
        self.values = {k: Tensor(np.zeros(v["shape"], dtype="<f4")) for k, v in tensors.items()}
        self.buffers = {}
    def state_dict(self): return self.values
    def named_parameters(self, remove_duplicate=True): return iter(self.values.items())
    def named_buffers(self, remove_duplicate=True): return iter(self.buffers.items())
    def parameters(self): return list(self.values.values())
    def load_state_dict(self, values, strict=True):
        if strict and set(values) != set(self.values): raise ValueError("key set")
        for key, value in values.items(): self.values[key] = Tensor(value.array.copy())


def fixture_models(catalog):
    models = {}
    for comp, key in (("node_emb", "recurrent"), ("link_pred", "link_head")):
        spec = catalog["components"][key]
        by_name = {item["name"]: item for item in spec["tensors"]}
        models[comp] = FakeModel({name: by_name[name] for name in spec["native_state_dict_order"]})
    return models


def tensor_payload(catalog, torch):
    models = fixture_models(catalog)
    values = boot.observe_catalog(torch, models, catalog)
    values["state.hidden"] = np.arange(np.prod(boot.STATE_SHAPE), dtype="<f4").reshape(boot.STATE_SHAPE).tobytes()
    values["state.cell"] = np.full(boot.STATE_SHAPE, 2, dtype="<f4").tobytes()
    return models, values


def parity():
    return {"forward": [0.25, 0.75], "loss": 1.2,
            "gradients": {"node_emb.w": [0.1, 0.2], "link_pred.w": [0.3]},
            "train_modes": [True, True, True], "eval_modes": [False, False, False],
            "dropout_train": [0.1, 0.9], "dropout_eval": [0.5, 0.5], "optimizer_updates": 0}


def fixture_rng():
    state = {"python": {"version": 3, "state": [1, 2, 3], "gauss": None},
             "numpy": {"algorithm": "MT19937", "state": [0, 1], "position": 1,
                       "has_gauss": 0, "cached_gaussian": 0.0},
             "torch_cpu_hex": "0102"}
    digest = boot._sha(json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    return {"state": state, "sha256": digest}


class BootstrapTests(unittest.TestCase):
    def test_release_catalog_and_config_are_pinned_and_validate(self):
        catalog, config, hashes = boot.load_release_contract()
        self.assertEqual(hashes, {"catalog_sha256": boot.CATALOG_SHA256, "config_sha256": boot.CONFIG_SHA256})
        self.assertEqual(catalog["total_model_numel"], 232833)
        self.assertEqual(config["candidate"]["condition"], "primary_Pdet_2_32")

    def test_catalog_observation_and_roundtrip_of_both_components_and_hc(self):
        catalog, _, hashes = boot.load_release_contract()
        models, payloads = tensor_payload(catalog, FakeTorch)
        source_state = gr.CompleteState(Tensor(np.ones(boot.STATE_SHAPE)), Tensor(np.full(boot.STATE_SHAPE, 3)), 1, tuple(range(100)))
        blob, manifest = boot._serialize(payloads, catalog, fixture_rng(), source_state)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bundle"
            tx = boot.write_transaction(path, blob, manifest)
            observed_blob, observed_manifest = boot.read_transaction(path, catalog,
                expected_manifest_sha256=tx["manifest_sha256"],
                expected_payload_sha256=tx["payload_sha256"])
            self.assertEqual(observed_blob, blob)
            self.assertEqual(observed_manifest["optimizer_updates"], 0)
            for model in models.values():
                for value in model.values.values(): value.array.fill(9)
            source_state.hidden.array.fill(-1)
            source_state.cell.array.fill(-1)
            restored = boot.restore_transaction(FakeTorch, models, source_state, path, catalog,
                expected_manifest_sha256=tx["manifest_sha256"],
                expected_payload_sha256=tx["payload_sha256"])
            self.assertEqual(restored["catalog_sha256"], hashes["catalog_sha256"])
            self.assertTrue(np.all(source_state.cell.array == 2))
            self.assertTrue(np.array_equal(source_state.hidden.array, np.arange(np.prod(boot.STATE_SHAPE), dtype="<f4").reshape(boot.STATE_SHAPE)))
            self.assertTrue(all(np.all(v.array == 0) for model in models.values() for v in model.values.values()))

    def test_catalog_observation_rejects_aliases_overlapping_storage_and_unexpected_buffers(self):
        catalog, _, _ = boot.load_release_contract()
        models = fixture_models(catalog)
        names = list(models["node_emb"].values)
        models["node_emb"].values[names[1]] = models["node_emb"].values[names[0]]
        with self.assertRaises(boot.BootstrapError): boot.observe_catalog(FakeTorch, models, catalog)

        models = fixture_models(catalog)
        names = [name for name, value in models["node_emb"].values.items() if value.shape == (100, 128)]
        self.assertGreaterEqual(len(names), 2)
        shared = np.zeros(128 * 100 + 1, dtype="<f4")
        models["node_emb"].values[names[0]] = Tensor(shared[:128 * 100].reshape(100, 128))
        models["node_emb"].values[names[1]] = Tensor(shared[1:1 + 128 * 100].reshape(100, 128))
        with self.assertRaises(boot.BootstrapError): boot.observe_catalog(FakeTorch, models, catalog)

        models = fixture_models(catalog)
        models["node_emb"].buffers["unexpected"] = Tensor(np.zeros((1,), dtype="<f4"))
        with self.assertRaises(boot.BootstrapError): boot.observe_catalog(FakeTorch, models, catalog)

        models = fixture_models(catalog)
        names = list(models["node_emb"].values)
        original_state_dict = models["node_emb"].state_dict
        def mismatched_state_dict():
            result = dict(original_state_dict())
            result[names[0]] = Tensor(np.zeros(result[names[0]].shape, dtype="<f4"))
            return result
        models["node_emb"].state_dict = mismatched_state_dict
        with self.assertRaises(boot.BootstrapError): boot.observe_catalog(FakeTorch, models, catalog)

    def test_interrupted_transaction_is_not_reopenable_then_valid_publish_is(self):
        catalog, _, _ = boot.load_release_contract()
        _, payloads = tensor_payload(catalog, FakeTorch)
        blob, manifest = boot._serialize(payloads, catalog, fixture_rng(), gr.CompleteState(Tensor(np.zeros(boot.STATE_SHAPE)), Tensor(np.zeros(boot.STATE_SHAPE)), 1, tuple(range(100))))
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / "bundle"
            with self.assertRaises(RuntimeError): boot.write_transaction(dest, blob, manifest, crash_before_publish=True)
            with self.assertRaises(boot.BootstrapError): boot.read_transaction(dest, catalog, expected_manifest_sha256="0"*64, expected_payload_sha256="0"*64)
            tx = boot.write_transaction(dest, blob, manifest)
            reread, _ = boot.read_transaction(dest, catalog, expected_manifest_sha256=tx["manifest_sha256"], expected_payload_sha256=tx["payload_sha256"])
            self.assertEqual(reread, blob)

    def test_transaction_reader_rejects_untrusted_mutations_and_oversize_before_read(self):
        catalog, _, _ = boot.load_release_contract()
        _, payloads = tensor_payload(catalog, FakeTorch)
        state = gr.CompleteState(Tensor(np.zeros(boot.STATE_SHAPE)), Tensor(np.zeros(boot.STATE_SHAPE)), 1, tuple(range(100)))
        blob, manifest = boot._serialize(payloads, catalog, fixture_rng(), state)
        mutations = ("extra_key", "duplicate_entry", "wrong_component", "offset", "truncated", "nonfinite")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "bundle"
                boot.write_transaction(path, blob, manifest)
                raw_manifest = json.loads((path / "manifest.json").read_text())
                raw_payload = bytearray((path / "payload.f32le").read_bytes())
                if mutation == "extra_key": raw_manifest["unexpected"] = True
                elif mutation == "duplicate_entry": raw_manifest["entries"][1]["name"] = raw_manifest["entries"][0]["name"]
                elif mutation == "wrong_component": raw_manifest["components"] = ["node_emb"]
                elif mutation == "offset": raw_manifest["entries"][1]["offset"] += 1
                elif mutation == "truncated": raw_payload.pop()
                elif mutation == "nonfinite":
                    raw_payload[:4] = np.asarray([np.nan], dtype="<f4").tobytes()
                    raw_manifest["entries"][0]["sha256"] = boot._sha(bytes(raw_payload[:raw_manifest["entries"][0]["nbytes"]]))
                raw_manifest["payload_bytes"] = len(raw_payload)
                raw_manifest["payload_sha256"] = boot._sha(bytes(raw_payload))
                (path / "payload.f32le").write_bytes(raw_payload)
                manifest_bytes = json.dumps(raw_manifest, sort_keys=True, separators=(",", ":")).encode()
                (path / "manifest.json").write_bytes(manifest_bytes)
                with self.assertRaises(boot.BootstrapError):
                    boot.read_transaction(path, catalog,
                        expected_manifest_sha256=boot._sha(manifest_bytes),
                        expected_payload_sha256=boot._sha(bytes(raw_payload)))

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bundle"; path.mkdir()
            (path / "manifest.json").write_bytes(b"{}")
            with (path / "payload.f32le").open("wb") as stream: stream.truncate(boot.MAX_BUNDLE_BYTES + 1)
            with self.assertRaises(boot.BootstrapError):
                boot.read_transaction(path, catalog, expected_manifest_sha256=boot._sha(b"{}"), expected_payload_sha256="0"*64)
            oversized_meta = Path(temp) / "oversized-meta"; oversized_meta.mkdir()
            (oversized_meta / "manifest.json").write_bytes(b" " * (boot.MAX_MANIFEST_BYTES + 1))
            (oversized_meta / "payload.f32le").write_bytes(b"")
            with self.assertRaises(boot.BootstrapError):
                boot.read_transaction(oversized_meta, catalog, expected_manifest_sha256="0"*64, expected_payload_sha256="0"*64)

    def test_receipt_publication_failure_leaves_no_pass_artifact(self):
        catalog, _, _ = boot.load_release_contract()
        _, payloads = tensor_payload(catalog, FakeTorch)
        state = gr.CompleteState(Tensor(np.zeros(boot.STATE_SHAPE)), Tensor(np.zeros(boot.STATE_SHAPE)), 1, tuple(range(100)))
        blob, manifest = boot._serialize(payloads, catalog, fixture_rng(), state)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); stage = root / "stage"; stage.mkdir(); output = root / "final"
            tx = boot.write_transaction(stage / "transaction", blob, manifest)
            evidence_raw = b'{"evidence":true}'
            environment = {"lock_sha256": boot.LOCK_SHA256, "wheel_manifest_sha256": boot.WHEEL_MANIFEST_SHA256,
                           "machine": "x86_64", "installed_module_files": {}}
            source = {"wrappers": boot.gr.WRAPPER_SHA256, "cell_sha256": boot.gr.CELL_SHA256,
                      "graph_reconstruction_sha256": boot.GRAPH_RECONSTRUCTION_SHA256, "runtime_imported_files": {},
                      "pinned_graph_import_sha256": boot.PINNED_GRAPH_IMPORT_SHA256,
                      "bootstrap_module_sha256": "e"*64}
            execution = boot.Node2AExecutionResult(boot._sha(evidence_raw), tx["manifest_sha256"],
                tx["payload_sha256"], environment, "f"*64, source, boot._RUNTIME_TOKEN)
            boot._publish_complete_bundle(stage, output, evidence_raw=evidence_raw, execution=execution,
                catalog=catalog, manifest_sha256=tx["manifest_sha256"], payload_sha256=tx["payload_sha256"])
            self.assertEqual({p.name for p in output.iterdir()},
                             {"transaction", "bootstrap-result.json", "node2a-receipt.json"})
            self.assertEqual({p.name for p in (output / "transaction").iterdir()},
                             {"manifest.json", "payload.f32le"})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); stage = root / "stage"; stage.mkdir(); output = root / "final"
            tx = boot.write_transaction(stage / "transaction", blob, manifest)
            with self.assertRaises(boot.BootstrapError):
                boot._publish_complete_bundle(stage, output, evidence_raw=b'{"evidence":true}',
                    execution=boot.Node2AExecutionResult(boot._sha(b'{"evidence":true}'),
                        tx["manifest_sha256"], tx["payload_sha256"], {}, "f"*64, {}, None), catalog=catalog,
                    manifest_sha256=tx["manifest_sha256"], payload_sha256=tx["payload_sha256"],
                    fail_before_receipt=True)
            self.assertFalse(output.exists())
            self.assertFalse(list(root.rglob("node2a-receipt.json")))

    def test_parity_and_receipt_require_zero_updates_and_no_data_payloads(self):
        obs = parity()
        report = boot.assert_parity(obs, obs)
        self.assertTrue(report["forward"]["passed"])
        with self.assertRaises(boot.BootstrapError): boot.make_receipt({"parity": report, "status": "pass"})
        forged = boot.Node2AExecutionResult("a"*64, "b"*64, "c"*64,
            {"lock_sha256": boot.LOCK_SHA256, "wheel_manifest_sha256": boot.WHEEL_MANIFEST_SHA256, "machine": "x86_64"},
            "f"*64,
            {"wrappers": boot.gr.WRAPPER_SHA256, "cell_sha256": boot.gr.CELL_SHA256,
             "bootstrap_module_sha256": "d"*64}, None)
        with self.assertRaises(boot.BootstrapError): boot.make_receipt(forged)
        obs["optimizer_updates"] = 1
        with self.assertRaises(boot.BootstrapError): boot.assert_parity(obs, obs)

    def test_pass_receipt_rejects_changed_pinned_import_helper_identity(self):
        environment = {"lock_sha256": boot.LOCK_SHA256,
                       "wheel_manifest_sha256": boot.WHEEL_MANIFEST_SHA256,
                       "machine": "x86_64", "installed_module_files": {}}
        source = {"wrappers": boot.gr.WRAPPER_SHA256, "cell_sha256": boot.gr.CELL_SHA256,
                  "graph_reconstruction_sha256": boot.GRAPH_RECONSTRUCTION_SHA256,
                  "pinned_graph_import_sha256": boot.PINNED_GRAPH_IMPORT_SHA256,
                  "runtime_imported_files": {}, "bootstrap_module_sha256": "a"*64}
        execution = boot.Node2AExecutionResult("b"*64, "c"*64, "d"*64,
            environment, "e"*64, source, boot._RUNTIME_TOKEN)
        self.assertEqual(boot.make_receipt(execution)["source_identity_sha256"],
                         boot._sha(json.dumps(source, sort_keys=True, separators=(",", ":")).encode()))
        changed_source = dict(source); changed_source["pinned_graph_import_sha256"] = "f"*64
        changed = boot.Node2AExecutionResult("b"*64, "c"*64, "d"*64,
            environment, "e"*64, changed_source, boot._RUNTIME_TOKEN)
        with self.assertRaisesRegex(boot.BootstrapError, "source/executable identities"):
            boot.make_receipt(changed)

    def test_seeded_models_bind_actual_rng_snapshot_not_callback_claim(self):
        class RngTorch:
            class Backend: pass
            backends = type("B", (), {"cudnn": type("C", (), {})()})()
            @staticmethod
            def manual_seed(seed): random_seed.append(seed)
            @staticmethod
            def use_deterministic_algorithms(value): deterministic.append(value)
            @staticmethod
            def get_rng_state(): return Tensor(np.frombuffer(b"seedstate", dtype=np.uint8))
        import random
        random_seed, deterministic = [], []
        models, rng = boot.seeded_fresh_models(0, RngTorch, lambda: {"node_emb": object(), "link_pred": object()})
        self.assertEqual(set(models), set(boot.COMPONENTS))
        self.assertEqual(random_seed, [0])
        self.assertTrue(deterministic[-1])
        self.assertEqual(len(rng["sha256"]), 64)
        with self.assertRaises(boot.BootstrapError): boot.seeded_fresh_models(7, RngTorch, lambda: {})

    def test_parity_detects_forward_gradient_mode_and_dropout_differences(self):
        base = parity()
        changed = parity(); changed["gradients"]["node_emb.w"][0] += 1
        with self.assertRaises(boot.BootstrapError): boot.assert_parity(base, changed)
        changed = parity(); changed["eval_modes"] = [True]
        with self.assertRaises(boot.BootstrapError): boot.assert_parity(base, changed)

    def test_execution_harness_intercepts_step_and_exercises_restore_and_crash_path(self):
        catalog, _, hashes = boot.load_release_contract()
        models = fixture_models(catalog)
        state = gr.CompleteState(Tensor(np.zeros(boot.STATE_SHAPE)), Tensor(np.zeros(boot.STATE_SHAPE)), 1, tuple(range(100)))
        class Optimizer:
            def step(self): raise AssertionError("must be intercepted")
        optimizer = Optimizer()
        observed_calls = []
        def observe(_models, _state):
            observed_calls.append(True)
            return parity()
        with tempfile.TemporaryDirectory() as temp:
            result = boot.execute_roundtrip_parity(torch=FakeTorch, models=models, optimizer=optimizer,
                state=state, catalog=catalog, hashes=hashes, destination=Path(temp) / "published",
                observe_native=observe)
            self.assertEqual(result["status"], "synthetic_native_parity_pass")
            self.assertEqual(result["optimizer_updates"], 0)
            self.assertEqual(len(observed_calls), 2)
            self.assertTrue((Path(temp) / "published" / "manifest.json").is_file())
            self.assertEqual({p.name for p in (Path(temp) / "published").iterdir()},
                             {"manifest.json", "payload.f32le"})
            self.assertEqual(list(Path(temp).glob(".node2a-crash-probe-*")), [])
            self.assertNotIn("step", optimizer.__dict__)
            sentinel = lambda: None
            optimizer.step = sentinel
            boot.execute_roundtrip_parity(torch=FakeTorch, models=models, optimizer=optimizer,
                state=state, catalog=catalog, hashes=hashes,
                destination=Path(temp) / "preserved-instance-step", observe_native=observe)
            self.assertIs(optimizer.__dict__["step"], sentinel)
            with self.assertRaises(boot.BootstrapError):
                boot.execute_roundtrip_parity(torch=FakeTorch, models=models, optimizer=optimizer,
                    state=state, catalog=catalog, hashes=hashes,
                    destination=Path(temp) / "failed",
                    observe_native=lambda _m, _s: (optimizer.step(), parity())[1])
            self.assertIs(optimizer.__dict__["step"], sentinel)

    def test_pinned_environment_fails_closed_and_cli_import_exposes_receipt_functions(self):
        module = importlib.import_module("src.representation_collapse.tgrab_figure10_bootstrap")
        self.assertTrue(callable(module.assert_parity))
        self.assertTrue(callable(module.make_receipt))
        with tempfile.TemporaryDirectory() as temp:
            bad_lock = Path(temp) / "requirements.lock"
            bad_lock.write_text("torch==0 --hash=sha256:" + "0"*64 + "\n")
            with self.assertRaises(boot.BootstrapError): boot.verify_runtime_environment(bad_lock)
        # A same-minor but different patch release is outside the frozen lock.
        with patch.object(boot.sys, "version_info", (3, 10, 18, "final", 0)):
            with self.assertRaisesRegex(boot.BootstrapError, "3.10.19"):
                boot.verify_runtime_environment()

    def test_approved_environment_inventory_requires_external_digest_and_exact_identity(self):
        identity = {"python_version": [3, 10, 19], "installed_module_files": {"torch._C": {"sha256": "a"*64}}}
        raw = json.dumps({"schema": "tgrab-node2a-environment-identity-v1", "identity": identity},
                         sort_keys=True).encode()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "environment-identity.json"; path.write_bytes(raw)
            digest = boot._sha(raw)
            with self.assertRaisesRegex(boot.BootstrapError, "digest mismatch"):
                boot.load_approved_environment_identity(path, "0"*64)
            with patch.object(boot, "verify_runtime_environment", return_value=identity):
                observed, bound_digest = boot.load_approved_environment_identity(path, digest)
            self.assertEqual(observed, identity)
            self.assertEqual(bound_digest, digest)
            with patch.object(boot, "verify_runtime_environment", return_value={**identity, "python_version": [3, 10, 18]}):
                with self.assertRaisesRegex(boot.BootstrapError, "differ from approved"):
                    boot.load_approved_environment_identity(path, digest)

    def test_capture_binds_and_rechecks_loader_source_after_environment_inventory(self):
        source = {"wrappers": dict(boot.gr.WRAPPER_SHA256), "cell_sha256": boot.gr.CELL_SHA256,
                  "graph_reconstruction_sha256": boot.GRAPH_RECONSTRUCTION_SHA256,
                  "pinned_graph_import_sha256": boot.PINNED_GRAPH_IMPORT_SHA256}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "identity.json"
            with patch.object(boot, "_load_pinned_sources", return_value=(None, None, Path("cell"))), \
                 patch.object(boot, "_source_identity", side_effect=[source, source]), \
                 patch.object(boot, "verify_runtime_environment", return_value={"python_version": [3, 10, 19]}):
                digest = boot.write_environment_identity(output, Path("source"))
            identity = json.loads(output.read_text())["identity"]
            self.assertEqual(identity["source_identity"]["pinned_graph_import_sha256"],
                             boot.PINNED_GRAPH_IMPORT_SHA256)
            self.assertEqual(digest, boot._sha(output.read_bytes()))

            output = Path(temp) / "mutated.json"
            changed = {**source, "pinned_graph_import_sha256": "f"*64}
            with patch.object(boot, "_load_pinned_sources", return_value=(None, None, Path("cell"))), \
                 patch.object(boot, "_source_identity", side_effect=[source, changed]), \
                 patch.object(boot, "verify_runtime_environment", return_value={"python_version": [3, 10, 19]}):
                with self.assertRaisesRegex(boot.BootstrapError, "changed during environment identity capture"):
                    boot.write_environment_identity(output, Path("source"))
            self.assertFalse(output.exists())

    def test_runtime_import_guard_refuses_temporal_package_initializers(self):
        with patch.object(boot.importlib, "import_module") as importer:
            for name in ("torch_geometric_temporal", "torch_geometric_temporal.nn",
                         "torch_geometric_temporal.nn.recurrent"):
                with self.assertRaisesRegex(boot.BootstrapError, "forbidden"):
                    boot._import_runtime_module(name)
            importer.assert_not_called()
        completed = subprocess.run([sys.executable, "-m", "src.representation_collapse.tgrab_figure10_bootstrap", "--help"],
                                   check=False, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--source-root", completed.stdout)


if __name__ == "__main__":
    unittest.main()
