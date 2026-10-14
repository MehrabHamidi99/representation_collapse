"""Pure fake source/import-spy fixtures; no actual Torch/PyG/Temporal imports."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest

from representation_collapse import pinned_graph_import as importer


class PinnedImportFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.original = dict(sys.modules)
        self.assertFalse(any(name == importer.TEMPORAL or name.startswith(importer.TEMPORAL + ".") for name in self.original))
        sys.modules["torch"] = ModuleType("torch")
        self.source = self.root / "public"; (self.source / "model").mkdir(parents=True)
        self.cell = self.root / "installed/torch_geometric_temporal/nn/recurrent/gc_lstm.py"
        self.cell.parent.mkdir(parents=True)
        self.cell.write_text("class GCLSTM:\n    def forward(self):\n        return 'new'\n")
        (self.source / "model/node_emb.py").write_text("class NodeEmbeddingModel:\n    @property\n    def out_dimension(self):\n        return 2\n")
        (self.source / "model/gclstm.py").write_text("from torch_geometric_temporal.nn.recurrent import GCLSTM\nfrom .node_emb import NodeEmbeddingModel\nclass GCLSTMParam:\n    pass\nclass MultiLayerGCLSTM(NodeEmbeddingModel):\n    def forward(self):\n        return GCLSTM\n")
        (self.source / "model/link_predictor.py").write_text("class LinkPredictor:\n    def forward(self):\n        return None\n")
        for path in (self.cell.parent / "__init__.py", self.cell.parent.parent / "__init__.py", self.cell.parent.parent.parent / "__init__.py", self.source / "model/__init__.py"):
            path.write_text("raise RuntimeError('umbrella or wrapper package initializer executed')\n")

    def tearDown(self):
        for name in list(sys.modules):
            if name not in self.original:
                del sys.modules[name]
        sys.modules.update(self.original)
        self.temp.cleanup()

    def pins(self):
        return {relative: hashlib.sha256((self.source / relative).read_bytes()).hexdigest() for relative in importer.PINNED_WRAPPER_SHA256}

    def load(self, guard=lambda: None, **changes):
        arguments = {"source_root": self.source, "installed_cell_path": self.cell,
                     "cell_sha256": hashlib.sha256(self.cell.read_bytes()).hexdigest(),
                     "wrapper_sha256": self.pins(), "guard": guard}
        arguments.update(changes)
        return importer.load_pinned_graph_modules(**arguments)

    def assert_no_owned_namespace(self):
        self.assertFalse(any(name == importer.TEMPORAL or name.startswith(importer.TEMPORAL + ".")
            or name == importer.WRAPPER_NAMESPACE or name.startswith(importer.WRAPPER_NAMESPACE + ".") for name in sys.modules))

    def test_helper_import_is_stdlib_only(self):
        result = subprocess.run([sys.executable, "-B", "-c", "import sys;import representation_collapse.pinned_graph_import;assert not any(k in ('torch','numpy','wandb') or k.startswith('torch_geometric') for k in sys.modules)"],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_direct_sources_skip_all_initializers_and_return_exact_types(self):
        value = self.load()
        self.assertIs(value.cell_type, value.cell.GCLSTM)
        self.assertIs(value.recurrent.GCLSTM, value.cell_type)
        self.assertEqual(value.cell_type.__module__, importer.CELL_MODULE)
        self.assertEqual(value.wrapper_type.__module__, importer.WRAPPER_NAMESPACE + ".gclstm")
        self.assertIs(value.parameter_type, value.recurrent.GCLSTMParam)
        self.assertIs(value.head_type, value.head.LinkPredictor)
        self.assertIs(value.recurrent.NodeEmbeddingModel, value.node.NodeEmbeddingModel)
        self.assertEqual(sys.modules[importer.TEMPORAL].__path__, [])
        self.assertIsNone(sys.modules[importer.TEMPORAL].__loader__)
        provenance = value.provenance_dict(); json.dumps(provenance)
        self.assertFalse(provenance["Temporal_package_initializers_executed"])
        self.assertFalse(provenance["source_release_hashes_match"])

    def test_preexisting_foreign_parent_is_preserved_and_refused(self):
        existing = ModuleType(importer.TEMPORAL); existing.foreign = object()
        sys.modules[importer.TEMPORAL] = existing
        with self.assertRaisesRegex(importer.PinnedImportError, "preexisting"):
            self.load()
        self.assertIs(sys.modules[importer.TEMPORAL], existing)
        self.assertEqual(set(name for name in sys.modules if name.startswith(importer.TEMPORAL)), {importer.TEMPORAL})

    def test_preexisting_descendant_or_private_namespace_refused(self):
        for name in (importer.CELL_MODULE, importer.WRAPPER_NAMESPACE + ".foreign"):
            foreign = ModuleType(name); sys.modules[name] = foreign
            with self.subTest(name=name), self.assertRaisesRegex(importer.PinnedImportError, "preexisting"):
                self.load()
            self.assertIs(sys.modules.pop(name), foreign)
        self.assert_no_owned_namespace()

    def test_second_load_refused_without_replacing_first_modules(self):
        value = self.load()
        with self.assertRaisesRegex(importer.PinnedImportError, "fresh worker"):
            self.load()
        self.assertIs(sys.modules[importer.CELL_MODULE], value.cell)

    def test_bad_source_hash_refused_before_namespace_changes(self):
        pins = self.pins(); pins["model/link_predictor.py"] = "0" * 64
        with self.assertRaisesRegex(importer.PinnedImportError, "source bytes"):
            self.load(wrapper_sha256=pins)
        self.assert_no_owned_namespace()
        with self.assertRaises(importer.PinnedImportError):
            self.load(cell_sha256="0" * 64)
        self.assert_no_owned_namespace()

    def test_missing_extra_or_unsafe_source_refused(self):
        pins = self.pins(); pins["model/foreign.py"] = "0" * 64
        with self.assertRaisesRegex(importer.PinnedImportError, "three public"):
            self.load(wrapper_sha256=pins)
        linked = self.root / "linked-public"; linked.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(importer.PinnedImportError, "canonical"):
            self.load(source_root=linked)
        with self.assertRaisesRegex(importer.PinnedImportError, "distribution source path"):
            self.load(installed_cell_path=self.root / "wrong.py")
        self.assert_no_owned_namespace()

    def test_torch_must_already_be_loaded(self):
        del sys.modules["torch"]
        with self.assertRaisesRegex(importer.PinnedImportError, "already be imported"):
            self.load()
        self.assert_no_owned_namespace()

    def test_import_failure_rolls_back_only_owned_modules(self):
        untouched = ModuleType("untouched_fixture"); sys.modules["untouched_fixture"] = untouched
        (self.source / "model/link_predictor.py").write_text("raise RuntimeError('deliberate source failure')\n")
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            self.load()
        self.assert_no_owned_namespace()
        self.assertIs(sys.modules["untouched_fixture"], untouched)

    def test_source_mutation_during_execution_fails_and_rolls_back(self):
        self.cell.write_text("from pathlib import Path\nclass GCLSTM:\n    def forward(self):\n        return None\nPath(__file__).write_text('changed')\n")
        with self.assertRaisesRegex(importer.PinnedImportError, "source bytes"):
            self.load()
        self.assert_no_owned_namespace()

    def test_forged_cell_class_module_and_method_provenance_refused(self):
        self.cell.write_text("class GCLSTM:\n    def forward(self):\n        return None\nGCLSTM.__module__='foreign'\n")
        with self.assertRaisesRegex(importer.PinnedImportError, "class source/module"):
            self.load()
        self.assert_no_owned_namespace()
        self.cell.write_text("class GCLSTM:\n    pass\nexec(compile('def foreign(self): return None', 'foreign.py', 'exec'))\nGCLSTM.forward=foreign\n")
        with self.assertRaisesRegex(importer.PinnedImportError, "class method"):
            self.load()
        self.assert_no_owned_namespace()

    def test_wrapper_foreign_cell_binding_refused(self):
        path = self.source / "model/gclstm.py"
        path.write_text(path.read_text() + "GCLSTM=object\n")
        with self.assertRaisesRegex(importer.PinnedImportError, "exact installed"):
            self.load()
        self.assert_no_owned_namespace()

    def test_unexpected_foreign_namespace_injected_during_import_is_preserved(self):
        path = self.source / "model/link_predictor.py"
        path.write_text(path.read_text() + "import sys,types\nsys.modules['torch_geometric_temporal.foreign']=types.ModuleType('foreign')\n")
        with self.assertRaisesRegex(importer.PinnedImportError, "foreign module appeared"):
            self.load()
        self.assertIn("torch_geometric_temporal.foreign", sys.modules)
        self.assertNotIn(importer.TEMPORAL, sys.modules)

    def test_stop_after_publication_rolls_back_owned_namespace(self):
        def guard():
            if importer.CELL_MODULE in sys.modules:
                raise InterruptedError("fixture STOP")
        with self.assertRaisesRegex(InterruptedError, "STOP"):
            self.load(guard)
        self.assert_no_owned_namespace()

    def test_poisoned_timestamp_bytecode_cannot_replace_checked_source(self):
        new = self.cell.read_text(); self.cell.write_text(new.replace("'new'", "'old'"))
        py_compile.compile(str(self.cell), doraise=True)
        timestamp = self.cell.stat().st_mtime_ns
        self.cell.write_text(new); os.utime(self.cell, ns=(timestamp, timestamp))
        value = self.load()
        self.assertIn("new", value.cell.GCLSTM.forward.__code__.co_consts)
        self.assertNotIn("old", value.cell.GCLSTM.forward.__code__.co_consts)


if __name__ == "__main__":
    unittest.main()
