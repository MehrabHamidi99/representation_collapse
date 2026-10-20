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

    def test_phase_alignment_counts_and_fixed_window_contrasts(self):
        evaluation = synthetic_evaluation()
        for row in evaluation["rows"]:
            t = row["target_timestamp"]
            phase, motif = t % 32, "A" if t % 64 < 32 else "B"
            low = (motif == "A" and 24 <= phase <= 31) or (motif == "B" and 1 <= phase <= 8)
            tp, fp, fn, tn = (5, 5, 5, 9885) if low else (10, 0, 0, 9890)
            row.update(tp=tp, fp=fp, fn=fn, tn=tn, precision=tp/(tp+fp), recall=tp/(tp+fn),
                       f1=2*tp/(2*tp+fp+fn), accuracy=(tp+tn)/9900)
        rows = evaluation["rows"]
        evaluation["reduction"] = {"targets": 256, "per_target_f1": [r["f1"] for r in rows],
            "primary_time_mean_f1": sum(r["f1"] for r in rows)/256,
            "transition_time_mean_f1": sum(r["f1"] for r in rows if r["change_point"])/8,
            "nontransition_time_mean_f1": sum(r["f1"] for r in rows if not r["change_point"])/248}
        validated, _ = plot.validate_evaluation(evaluation)
        aligned, contrast = plot.within_motif_summary(validated)
        self.assertEqual(len(aligned), 64)
        self.assertTrue(all(row["observations"] == 4 for row in aligned))
        self.assertEqual(aligned[0]["contributing_target_timestamps"], [2816, 2880, 2944, 3008])
        self.assertEqual(contrast["motifs"]["A"]["late_minus_early_f1"], -.5)
        self.assertEqual(contrast["motifs"]["B"]["late_minus_early_f1"], .5)
        self.assertEqual(contrast["overall"]["late_minus_early_f1"], 0)
        self.assertEqual(contrast["overall"]["early_targets"], 64)
        self.assertEqual(contrast["overall"]["late_targets"], 64)
        self.assertFalse(contrast["independent_replicates"])
        # Phase zero contributes to the aligned curve, but never either window.
        changed = copy.deepcopy(validated)
        for row in changed:
            if row["target_timestamp"] % 32 == 0:
                row["f1"] = 0.0
        _, changed_contrast = plot.within_motif_summary(changed)
        self.assertEqual(changed_contrast, contrast)
        with self.assertRaises(ValueError):
            plot.within_motif_summary(validated[:-1])

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
                "figure10-reconstruction.csv", "figure10-within-motif.csv", "figure-manifest.json"})
            self.assertEqual(len((output / "figure10-reconstruction.csv").read_text().splitlines()), 257)
            self.assertEqual(len((output / "figure10-within-motif.csv").read_text().splitlines()), 65)
            self.assertEqual(manifest["within_motif_descriptive"]["overall"]["early_targets"], 64)
            self.assertTrue((output / "figure10-reconstruction.pdf").read_bytes().startswith(b"%PDF"))
            self.assertFalse(manifest["scientific_collapse_claim"])
            self.assertFalse(manifest["independent_motif_replicates"])
            self.assertIn("one contiguous sequence", manifest["caption"])
            for name, expected in manifest["files"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected)
            with self.assertRaises(FileExistsError):
                plot.render(evaluation_path=path, evaluation_hash=digest, acceptance_path=accepted,
                            acceptance_hash=accepted_hash, output=output)


    def test_primitive_v2_pointer_acceptance_and_identity_rejections(self):
        evaluation=synthetic_evaluation()
        pointer=evaluation.pop("final_head_sha256")
        evaluation.update(checkpoint_format="primitive_v2", selected_pointer_sha256=pointer,
                          executed_long_contract_sha256=plot.LONG_CONTRACT_SHA256)
        rows,reduction=plot.validate_evaluation(evaluation)
        self.assertEqual(len(rows),256)
        self.assertEqual(reduction,plot.validate_evaluation(synthetic_evaluation())[1])
        for key,value in (("final_head_sha256",pointer),("checkpoint_format","v1"),
                          ("executed_long_contract_sha256","f"*64),("selected_pointer_sha256","bad")):
            with self.subTest(key=key),self.assertRaises(ValueError):
                plot.validate_evaluation({**evaluation,key:value})
        for key in ("checkpoint_format","selected_pointer_sha256","executed_long_contract_sha256"):
            bad=copy.deepcopy(evaluation);bad.pop(key)
            with self.subTest(missing=key),self.assertRaises(ValueError):plot.validate_evaluation(bad)

    def test_v2_render_has_genuine_manifest_and_unchanged_legacy_reductions(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            path,digest,accepted,accepted_hash=write_inputs(root)
            legacy=plot.render(evaluation_path=path,evaluation_hash=digest,acceptance_path=accepted,
                acceptance_hash=accepted_hash,output=root/"legacy")
            self.assertEqual(legacy["kind"],"figure10_behavioral_reconstruction_plot_v1")
            self.assertIn("final_head_sha256",legacy)
            self.assertTrue({"checkpoint_format","selected_pointer_sha256","executed_long_contract_sha256"}.isdisjoint(legacy))
            evaluation=json.loads(path.read_text());pointer=evaluation.pop("final_head_sha256")
            identity=dict(checkpoint_format="primitive_v2",selected_pointer_sha256=pointer,
                          executed_long_contract_sha256=plot.LONG_CONTRACT_SHA256)
            evaluation.update(identity);path.write_text(json.dumps(evaluation))
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            acceptance=json.loads(accepted.read_text())
            acceptance.update(kind=plot.ACCEPTANCE_KIND_V2,selected_epoch=160,training_stop_reason="loss_patience",
                              evaluation_sha256=digest,**identity)
            accepted.write_text(json.dumps(acceptance));accepted_hash=hashlib.sha256(accepted.read_bytes()).hexdigest()
            current=plot.render(evaluation_path=path,evaluation_hash=digest,acceptance_path=accepted,
                acceptance_hash=accepted_hash,output=root/"v2")
            self.assertEqual(current["kind"],"figure10_behavioral_reconstruction_plot_v2")
            self.assertEqual(current["schema_version"],2)
            for key,value in identity.items():self.assertEqual(current[key],value)
            self.assertNotIn("final_head_sha256",current)
            self.assertEqual(current["selected_epoch"],160)
            self.assertEqual(current["reduction"],legacy["reduction"])
            self.assertEqual(current["within_motif_descriptive"],legacy["within_motif_descriptive"])
            self.assertEqual((root/"legacy/figure10-reconstruction.csv").read_bytes(),(root/"v2/figure10-reconstruction.csv").read_bytes())
            self.assertEqual((root/"legacy/figure10-within-motif.csv").read_bytes(),(root/"v2/figure10-within-motif.csv").read_bytes())
            for field,value in (("kind",plot.ACCEPTANCE_KIND),("kind","wrong"),("final_head_sha256",pointer),
                                ("selected_pointer_sha256","bad"),("checkpoint_format","v1"),
                                ("executed_long_contract_sha256","f"*64)):
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                    plot.validate_acceptance({**acceptance,field:value},digest)
            for field in identity:
                bad=copy.deepcopy(acceptance);bad.pop(field)
                with self.subTest(missing=field),self.assertRaises(ValueError):plot.validate_acceptance(bad,digest)
            acceptance["selected_pointer_sha256"]="f"*64
            accepted.write_text(json.dumps(acceptance));accepted_hash=hashlib.sha256(accepted.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                plot.render(evaluation_path=path,evaluation_hash=digest,acceptance_path=accepted,
                    acceptance_hash=accepted_hash,output=root/"mismatch")
            self.assertFalse((root/"mismatch").exists())


if __name__ == "__main__":
    unittest.main()
