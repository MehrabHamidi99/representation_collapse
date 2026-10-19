"""Synthetic gating/byte tests only; no native runtime parity evidence."""
import contextlib
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from representation_collapse import tgrab_figure10_evaluate as ev
from representation_collapse import graph_reconstruction as gr
from representation_collapse.tgrab_selection_transaction import SelectionContext
from tests.test_tgrab_figure10_candidate_encoder import TensorFixture, _inputs


class Tensor(TensorFixture):
    def detach(self):
        return self

    def clone(self):
        return Tensor(self.data.copy())
    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self.data.ctypes.data, nbytes=lambda: self.data.nbytes)

    def storage_offset(self):
        return 0

    def numel(self):
        return self.data.size


class Model:
    def __init__(self, values, corrupt=False):
        self.values = {k: Tensor(v.data.copy()) for k, v in values.items()}
        self.training = True
        self.corrupt = corrupt
        self.loads = 0

    def state_dict(self):
        return self.values

    def load_state_dict(self, values, strict):
        assert strict
        self.loads += 1
        self.values = values
        if self.corrupt:
            next(iter(values.values())).data.flat[0] = 1
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def eval(self):
        self.training = False

    def modules(self):
        return [self]


class Backend:
    def __init__(self):
        self.device = "cpu"
        self.calls = []

    def evaluation_mode(self):
        return contextlib.nullcontext()

    def reset(self):
        zero = lambda: Tensor(np.zeros((1, 100, 128), dtype=np.float32))
        return gr.CompleteState(zero(), zero(), 0, tuple(range(100)))

    def consume(self, state, snapshot):
        self.calls.append(snapshot.timestamp)
        return gr.CompleteState(state.hidden, state.cell, snapshot.timestamp + 1, state.node_ids)

    def detach(self, state):
        return state

    def score_pairs(self, state, pairs):
        assert self.calls[-1] == state.next_target_t - 1
        assert not state.hidden.data.any() and not state.cell.data.any()
        return np.zeros(len(pairs))


class Ledger:
    def __init__(self, root, context):
        self.root, self.context = root, context

    def verify_committed_state(self):
        return {"finalized": True, "selected_pointer": {}}

    def _read_reference(self, *args):
        return {"manifest_relative_path": "candidate/manifest.json", "selected": {"checkpoint_sha256": "a" * 64}}, b""

    def _candidate(self, *args):
        return {"checkpoint_sha256": "a" * 64}, None, None, None


class EvaluationTests(unittest.TestCase):
    def test_strict_both_components_exact_postload_bytes(self):
        evidence, _, _ = _inputs()
        _, catalog = ev.codec._schema_and_catalog()
        verified = {}
        for component, spec in catalog["components"].items():
            values = evidence.model_state_dicts[spec["backend_key"]]
            next(iter(values.values())).data.flat[0] = -0.0
            verified[component] = ev._component_bytes(component, values, catalog)[0]
        backend = SimpleNamespace(device="cpu", torch=SimpleNamespace(from_numpy=Tensor),
                                  models={k: Model(v) for k, v in evidence.model_state_dicts.items()})
        ev.load_native_weights(backend, verified)
        self.assertTrue(all(not m.training for m in backend.models.values()))
        backend.models["link_pred"].corrupt = True
        with self.assertRaisesRegex(ev.EvaluationError, "weights differ"):
            ev.load_native_weights(backend, verified)

    def test_bad_second_component_never_mutates_first(self):
        evidence, _, _ = _inputs()
        backend = SimpleNamespace(device="cpu", torch=SimpleNamespace(from_numpy=Tensor),
                                  models={k: Model(v) for k, v in evidence.model_state_dicts.items()})
        del backend.models["link_pred"].values["seq.1.weight"]
        with self.assertRaises(Exception):
            ev.load_native_weights(backend, {})
        self.assertEqual(sum(m.loads for m in backend.models.values()), 0)

    def test_exact_cell_signed_zero_and_identity(self):
        state = Backend().reset()
        state = gr.CompleteState(state.hidden, state.cell, 2816, state.node_ids)
        verified = {"terminal_state": bytes(102400)}
        ev.verify_terminal(state, verified)
        state.cell.data.flat[-1] = -0.0
        with self.assertRaisesRegex(ev.EvaluationError, "H/C"):
            ev.verify_terminal(state, verified)
        with self.assertRaisesRegex(ev.EvaluationError, "ordering"):
            ev.verify_terminal(gr.CompleteState(state.hidden, state.cell, 2816, tuple(reversed(state.node_ids))), verified)

    def _run(self, root, *, load_fail=False, parity_fail=False, observe=False, observe_fail=False):
        backend = Backend()
        context = SelectionContext("r", "a", "b" * 64, ev.codec.SCHEMA_SHA256, "c" * 64)
        ledger = Ledger(root, context)
        snapshots = [gr.Snapshot(t, 100, ((0, 1), (1, 0)) if (t // 32) % 2 else ((2, 3), (3, 2)), tuple(range(100))) for t in range(3072)]
        output = root / "evaluation"
        verified = {"files_sha256": {"manifest.json": "a" * 64},
                    "terminal_state": bytes(102400) if not parity_fail else b"\x01" + bytes(102399)}
        opened = []
        def opener():
            self.assertEqual(backend.calls, list(range(2816)))
            self.assertTrue((output / "003-test_read.json").exists())
            opened.append(True)
            return iter(snapshots[2816:])
        def loader(*args):
            self.assertTrue((output / "000-charged.json").exists())
            if load_fail:
                raise RuntimeError("load failed")
        reports = []
        observed = []
        def observer(state, t):
            self.assertEqual(len(opened), 1)
            self.assertEqual(backend.calls[-1], t - 1)
            self.assertEqual(state.next_target_t, t)
            self.assertEqual(state.node_ids, tuple(range(100)))
            observed.append(t)
            state.hidden.data.fill(4)
            state.cell.data.fill(5)
            if observe_fail:
                raise RuntimeError("observer failure")
        kwargs = dict(ledger=ledger, expected_context=context, expected_final_head_sha256="c" * 64,
                      backend=backend, train=snapshots[:2560], validation=snapshots[2560:2816],
                      test_opener=opener, output_dir=output, report=reports.append,
                      observe_state=observer if observe or observe_fail else None)
        with patch.object(ev, "SelectionLedger", Ledger), patch.object(ev.codec, "read_candidate", return_value=verified), patch.object(ev, "load_native_weights", side_effect=loader):
            if load_fail or parity_fail or observe_fail:
                with self.assertRaises((RuntimeError, ev.EvaluationError)):
                    ev.run_selected_evaluation(**kwargs)
                self.assertEqual(len(opened), 1 if observe_fail else 0)
                if observe_fail:
                    self.assertEqual(backend.calls, list(range(2816)))
                    self.assertEqual(observed, [2816])
                self.assertTrue(list(output.glob("*-failed.json")))
            else:
                result = ev.run_selected_evaluation(**kwargs)
                self.assertEqual(result["reduction"]["targets"], 256)
                self.assertEqual(backend.calls, list(range(3072)))
                self.assertEqual(len(opened), 1)
                if observe:
                    self.assertEqual(observed, list(range(2816, 3072)))
            with self.assertRaises(FileExistsError):
                ev.run_selected_evaluation(**kwargs)
        self.assertEqual(reports[0]["event"], "evaluation_reporting_initialized")

    def test_loader_failure_never_opens_test_and_no_retry(self):
        with tempfile.TemporaryDirectory() as td:
            self._run(Path(td), load_fail=True)

    def test_replay_failure_never_opens_test_and_no_retry(self):
        with tempfile.TemporaryDirectory() as td:
            self._run(Path(td), parity_fail=True)

    def test_replay_then_single_charged_open_then_score_before_consume(self):
        with tempfile.TemporaryDirectory() as td:
            self._run(Path(td))

    def test_observer_is_before_score_cloned_and_does_not_change_modelcalls(self):
        with tempfile.TemporaryDirectory() as td:
            self._run(Path(td), observe=True)

    def test_observer_failure_is_terminal_no_extra_score_consume_or_opener(self):
        with tempfile.TemporaryDirectory() as td:
            self._run(Path(td), observe_fail=True)

    def test_real_finalized_ledger_codec_pointer_integration(self):
        from tests.test_tgrab_figure10_selection_sink import _new_sink, _evidence, _context
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve() / "run"
            sink, ledger, selection = _new_sink(root)
            evidence, _ = _evidence(1, 0.0)
            evidence.record.validation_selected = True
            sink.on_epoch_complete(evidence)
            final = ledger.finalize(
                transaction_id="eval-final", run_stop_reason="complete_epoch_cap",
                apply_selection=lambda record: selection.finalize(),
                verify_selection=lambda result, record: selection.finalized and result.epoch == record["selected"]["epoch"],
                created_at_utc="2026-10-19T12:00:00+00:00")
            ledger.close()
            context = _context(final.head_sha256)
            snapshots = [gr.Snapshot(t, 100, ((0, 1), (1, 0)) if (t // 32) % 2 else ((2, 3), (3, 2)), tuple(range(100))) for t in range(3072)]
            with ev.SelectionLedger(root, context) as reopened, patch.object(ev, "load_native_weights"):
                result = ev.run_selected_evaluation(
                    ledger=reopened, expected_context=context,
                    expected_final_head_sha256=final.head_sha256,
                    backend=Backend(), train=snapshots[:2560], validation=snapshots[2560:2816],
                    test_opener=lambda: iter(snapshots[2816:]), output_dir=Path(td) / "eval",
                    report=lambda row: None)
            self.assertEqual(result["checkpoint_sha256"], final.selected["checkpoint_sha256"])


if __name__ == "__main__":
    unittest.main()
