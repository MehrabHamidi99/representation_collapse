"""Synthetic renderer/calculation tests; none are trained-model evidence."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("plot_figure10_reconstruction", ROOT / "code/plot_figure10_reconstruction.py")
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


def synthetic_evaluation():
    rows = []
    for t in range(2816, 3072):
        transition = t % 32 == 0
        tp, fp, fn, tn = (5, 5, 5, 9885) if transition else (10, 0, 0, 9890)
        rows.append({"target_timestamp": t, "source_timestamp": t - 1, "target_phase": t % 64,
            "source_phase": (t - 1) % 64, "change_point": transition, "phase_transition_expected": transition,
            "pairs": 9900, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp), "recall": tp / (tp + fn),
            "f1": 2 * tp / (2 * tp + fp + fn), "accuracy": (tp + tn) / 9900})
    return {"event": "complete", "run_id": "synthetic-test-only", "rows": rows,
        "checkpoint_sha256": hashlib.sha256(b"synthetic-test-checkpoint").hexdigest(),
        "final_head_sha256": hashlib.sha256(b"synthetic-test-head").hexdigest(),
        "reduction": {"targets": 256, "primary_time_mean_f1": (8 * .5 + 248) / 256,
            "transition_time_mean_f1": .5, "nontransition_time_mean_f1": 1.,
            "per_target_f1": [row["f1"] for row in rows]}}


def write_inputs(root):
    evaluation = synthetic_evaluation()
    path = root / "synthetic-evaluation.json"
    path.write_text(json.dumps(evaluation, sort_keys=True))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    acceptance = {"kind": plot.ACCEPTANCE_KIND, "status": "accepted", "evaluation_sha256": digest,
        "model_seed": 0, "selected_epoch": 1, "run_id": "synthetic-test-only", "training_stop_reason": "deadline",
        "validation_selection_finalized": True, "terminal_replay_verified": True, "test_rows_verified": True,
        "scientific_acceptance": False}
    accepted = root / "synthetic-acceptance.json"
    accepted.write_text(json.dumps(acceptance, sort_keys=True))
    return path, digest, accepted, hashlib.sha256(accepted.read_bytes()).hexdigest()


class Figure10PlotTests(unittest.TestCase):
    def test_primary_reduction_and_motif_indices_are_recomputed(self):
        rows, reduction = plot.validate_evaluation(synthetic_evaluation())
        self.assertEqual(reduction["primary_time_mean_f1"], .984375)
        self.assertEqual(reduction["transition_time_mean_f1"], .5)
        self.assertEqual([row["target_timestamp"] for row in rows if row["change_point"]], list(range(2816, 3072, 32)))
        self.assertEqual(rows[0]["motif"], "A")
        self.assertEqual(rows[32]["motif"], "B")
        self.assertEqual(rows[-1]["motif_presentation"], 8)
        self.assertEqual(rows[-1]["test_cycle"], 4)

    def test_missing_duplicate_nonfinite_and_nonchronological_rows_rejected(self):
        baseline = synthetic_evaluation()
        variants = []
        missing = copy.deepcopy(baseline); missing["rows"].pop(); variants.append(missing)
        duplicate = copy.deepcopy(baseline); duplicate["rows"][1] = duplicate["rows"][0]; variants.append(duplicate)
        reordered = copy.deepcopy(baseline); reordered["rows"][0:2] = list(reversed(reordered["rows"][0:2])); variants.append(reordered)
        nonfinite = copy.deepcopy(baseline); nonfinite["rows"][0]["f1"] = float("nan"); variants.append(nonfinite)
        for index, variant in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(ValueError):
                plot.validate_evaluation(variant)

    def test_metrics_must_match_complete_confusion_counts_and_true_transitions(self):
        for field, value in (("f1", .9), ("tp", True), ("tn", 0), ("source_timestamp", 2816),
                             ("phase_transition_expected", False), ("change_point", False)):
            evaluation = synthetic_evaluation(); evaluation["rows"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                plot.validate_evaluation(evaluation)
        evaluation = synthetic_evaluation(); evaluation["reduction"]["primary_time_mean_f1"] = .4
        with self.assertRaises(ValueError):
            plot.validate_evaluation(evaluation)

    def test_independent_acceptance_is_required_and_bound_to_evaluation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path, digest, accepted, accepted_hash = write_inputs(root)
            acceptance = json.loads(accepted.read_text())
            plot.validate_acceptance(acceptance, digest)
            for field, value in (("terminal_replay_verified", False), ("model_seed", True),
                                 ("scientific_acceptance", True), ("selected_epoch", 0),
                                 ("evaluation_sha256", hashlib.sha256(b"wrong").hexdigest())):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    plot.validate_acceptance({**acceptance, field: value}, digest)
            with self.assertRaises(ValueError):
                plot.read_pinned_json(path, hashlib.sha256(b"wrong").hexdigest())

    def test_bad_metadata_does_not_publish_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path, digest, accepted, accepted_hash = write_inputs(root)
            bad = json.loads(path.read_text()); bad["rows"].pop()
            path.write_text(json.dumps(bad))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            acceptance = json.loads(accepted.read_text()); acceptance["evaluation_sha256"] = digest
            accepted.write_text(json.dumps(acceptance)); accepted_hash = hashlib.sha256(accepted.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                plot.render(evaluation_path=path, evaluation_hash=digest, acceptance_path=accepted,
                            acceptance_hash=accepted_hash, output=root / "figure")
            self.assertFalse((root / "figure").exists())

    def test_synthetic_render_has_all_artifacts_and_scope_caption(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib unavailable")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path, digest, accepted, accepted_hash = write_inputs(root)
            output = root / "figure"
            manifest = plot.render(evaluation_path=path, evaluation_hash=digest, acceptance_path=accepted,
                acceptance_hash=accepted_hash, output=output)
            self.assertEqual({p.name for p in output.iterdir()}, {"figure10-reconstruction.pdf", "figure10-reconstruction.png",
                "figure10-reconstruction.csv", "figure-manifest.json"})
            self.assertEqual(len((output / "figure10-reconstruction.csv").read_text().splitlines()), 257)
            self.assertTrue((output / "figure10-reconstruction.pdf").read_bytes().startswith(b"%PDF"))
            self.assertFalse(manifest["scientific_collapse_claim"])
            self.assertFalse(manifest["independent_motif_replicates"])
            self.assertIn("one contiguous sequence", manifest["caption"])
            for name, expected in manifest["files"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected)
            with self.assertRaises(FileExistsError):
                plot.render(evaluation_path=path, evaluation_hash=digest, acceptance_path=accepted,
                            acceptance_hash=accepted_hash, output=output)


if __name__ == "__main__":
    unittest.main()
