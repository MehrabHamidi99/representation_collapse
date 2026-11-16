#!/usr/bin/env python3
"""Independent, NumPy-only reducer for downloaded multibit-memory artifacts.

This verifies raw held-out arrays and recomputes descriptive summaries. Saved
validation logits and integer counts are unavailable, so competence remains
worker-reported and summary-checked with inferred count integrality. It is
not independently replayed. Nothing here accepts a
scientific collapse claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys

import numpy as np


CONDITIONS = ((0, 1), (1, 1), (2, 1), (4, 1), (8, 1),
              (16, 1), (32, 1), (64, 1), (128, 1), (32, 4))
ARMS = ("periodic", "control")
ATOL, RTOL = 1e-8, 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    def reject(token):
        raise ValueError(f"non-finite JSON token {token}")
    with path.open("r", encoding="ascii") as stream:
        return json.load(stream, parse_constant=reject)


def close(actual, claimed) -> tuple[bool, float]:
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(claimed, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return False, math.inf
    difference = float(np.max(np.abs(a - b))) if a.size else 0.0
    return bool(np.allclose(a, b, atol=ATOL, rtol=RTOL)), difference


class Reducer:
    def __init__(self, root: Path):
        self.root = root.resolve(strict=True)
        self.checks: list[dict] = []
        self.max_abs_difference = 0.0

    def check(self, fit: str, name: str, ok: bool, detail: str = ""):
        self.checks.append({"fit": fit, "check": name,
                            "status": "pass" if ok else "fail",
                            **({"detail": detail} if detail else {})})

    def compare(self, fit: str, name: str, actual, claimed):
        ok, diff = close(actual, claimed)
        if math.isfinite(diff):
            self.max_abs_difference = max(self.max_abs_difference, diff)
        self.check(fit, name, ok, "max_abs_difference=" +
                   ("inf" if math.isinf(diff) else f"{diff:.12g}"))

    def inventory(self, fitdir: Path, fit: str) -> tuple[dict, dict]:
        path = fitdir / "artifact-inventory.json"
        if not path.is_file():
            self.check(fit, "inventory_present", False)
            return {}, []
        try:
            inventory = read_json(path)
            if not isinstance(inventory, dict):
                raise ValueError("inventory is not an object")
        except Exception as exc:
            self.check(fit, "inventory_parse", False, str(exc))
            return {}, []
        scientific = sorted([*fitdir.glob("*.json"), *fitdir.glob("*.npz")])
        verified = []
        for file in scientific:
            if file.name == "artifact-inventory.json":
                continue
            item = inventory.get(file.name)
            ok = (isinstance(item, dict) and item.get("sha256") == sha256(file)
                  and item.get("bytes") == file.stat().st_size)
            self.check(fit, f"inventory:{file.name}", ok,
                       "downloaded scientific file matches producer inventory" if ok
                       else "missing/mismatched inventory digest or byte count")
            if ok:
                verified.append(file.name)
        present = {p.name for p in fitdir.iterdir() if p.is_file()}
        omitted = sorted(name for name in inventory if name not in present)
        # The inventory is a producer-side top-level snapshot. It does not
        # enumerate nested reporting files, so state that omission explicitly.
        reporting = "reporting/ (nested offline reporting files not inventoried/downloaded)"
        if not (fitdir / "reporting").exists():
            omitted.append(reporting)
        reporting = fitdir / "reporting"
        nested = sorted(str(p.relative_to(fitdir)) for p in reporting.rglob("*") if p.is_file()) \
            if reporting.is_dir() else []
        raw_reporting = [name for name in nested
                         if name.endswith((".wandb", ".jsonl", ".jsonl.gz"))]
        reporting_receipt = reporting / "reporting-receipt.json"
        reporting_receipt_ok = False
        if reporting_receipt.is_file():
            try:
                receipt = read_json(reporting_receipt)
                reporting_receipt_ok = True
                for artifact in receipt.get("metadata_artifacts", []):
                    artifact_path = fitdir / "reporting" / "metadata-artifacts" / f"{artifact['name']}.json"
                    digest_ok = artifact_path.is_file() and sha256(artifact_path) == artifact.get("sha256")
                    self.check(fit, f"reporting_metadata:{artifact.get('name')}", digest_ok,
                               "receipt SHA verified" if digest_ok else "missing or SHA mismatch")
                    reporting_receipt_ok = reporting_receipt_ok and digest_ok
                self.check(fit, "reporting_metadata_receipt_hashes", reporting_receipt_ok)
            except Exception as exc:
                self.check(fit, "reporting_receipt_parse", False, str(exc))
        else:
            self.check(fit, "reporting_receipt_present", False)
        return inventory, {"verified_scientific_files": verified,
                           "omitted_inventory_files": omitted,
                           "inventory_scope": "producer inventory covers top-level regular files only; nested reporting files are outside it",
                           "omitted_checkpoint_reporting": {
                               "checkpoint_pt_not_downloaded": "checkpoint.pt" in omitted,
                               "other_inventory_files_not_downloaded": omitted,
                               "reporting_receipt_present": reporting_receipt.is_file(),
                               "reporting_metadata_files_present": [x for x in nested if x.startswith("reporting/metadata-artifacts/")],
                               "reporting_metadata_receipt_hashes_verified": reporting_receipt_ok,
                               "offline_wandb_raw_files_present": raw_reporting,
                               "offline_wandb_raw_files_note": ("none present in this downloaded artifact subset; only offline-run directory structure was fetched"
                                                                if not raw_reporting else "listed raw files are present")}}

    def competence(self, fit: str, result: dict, dimension: int,
                   validation_design_ok: bool) -> dict:
        stages = result.get("stage_results", [])
        expected = ((2000, (0,)), (6000, (0, 1, 2)),
                    (10000, (0, 1, 2, 4, 8)))
        status = result.get("status")
        all_passed = True
        consistency = True
        details = []
        for idx, (updates, repeats) in enumerate(expected, start=1):
            if len(stages) < idx:
                if status == "competence_passed":
                    self.check(fit, f"competence_stage_{idx}_present", False,
                               "successful fit lacks a required stage")
                    consistency = False
                break
            stage = stages[idx - 1]
            reasons = []
            if (stage.get("stage") != idx or stage.get("updates") != updates
                    or stage.get("required_repeats") != list(repeats)):
                reasons.append("stage schedule/update boundary differs from frozen protocol")
            checks = stage.get("checks", [])
            if len(checks) != len(repeats):
                reasons.append("validation check count differs from required repeats")
            gate_passed = len(checks) == len(repeats)
            for repeat, item in zip(repeats, checks):
                if item.get("repeats") != repeat or item.get("cue_vectors") != 256:
                    reasons.append(f"R{repeat} validation identity/count mismatch")
                    gate_passed = False
                bit = item.get("per_bit_accuracy_by_arm", {})
                polarity = item.get("per_bit_polarity_accuracy_by_arm", {})
                for arm in ARMS:
                    values = bit.get(arm, [])
                    if len(values) != dimension:
                        reasons.append(f"R{repeat} {arm} per-bit width mismatch")
                        gate_passed = False
                    for value in values:
                        count = round(float(value) * 256)
                        if abs(value * 256 - count) > 1e-7:
                            reasons.append(f"R{repeat} {arm} bit accuracy is not an integer/256 rate")
                            gate_passed = False
                        elif count < math.ceil(.95 * 256):
                            gate_passed = False
                    bybit = polarity.get(arm, {})
                    if set(bybit) != {str(i) for i in range(dimension)}:
                        reasons.append(f"R{repeat} {arm} polarity bit set mismatch")
                        gate_passed = False
                    for bit_index in range(dimension):
                        pair_rates = []
                        for sign in ("negative", "positive"):
                            value = bybit.get(str(bit_index), {}).get(sign)
                            if not isinstance(value, (int, float)) or not math.isfinite(value):
                                reasons.append(f"R{repeat} {arm} bit{bit_index} {sign} score missing")
                                gate_passed = False
                            else:
                                pair_rates.append(value)
                                count = round(value * 128)
                                if abs(value * 128 - count) > 1e-7:
                                    reasons.append(f"R{repeat} {arm} bit{bit_index} {sign} is not an integer/128 rate")
                                    gate_passed = False
                                elif count < math.ceil(.95 * 128):
                                    gate_passed = False
                        if len(pair_rates) == 2 and bit_index < len(values):
                            if abs(values[bit_index] - sum(pair_rates) / 2) > 1e-7:
                                reasons.append(f"R{repeat} {arm} bit{bit_index} overall accuracy differs from balanced polarity mean")
                                gate_passed = False
            recomputed_pass = gate_passed and not reasons
            if type(stage.get("passed")) is not bool or stage.get("passed") != recomputed_pass:
                reasons.append("saved stage pass flag disagrees with count-checked summaries")
                consistency = False
            if reasons:
                consistency = False
            all_passed &= recomputed_pass
            details.append({"stage": idx, "updates": updates,
                            "worker_reported_gate_passed": recomputed_pass,
                            "float_rates_count_checked": not reasons,
                            "failures": reasons})
            self.check(fit, f"competence_stage_{idx}_summary_consistency", not reasons,
                       "; ".join(reasons) if reasons else
                       ("worker-reported gate passed" if recomputed_pass else
                        "worker-reported gate failed; valid negative competence result"))
            if not recomputed_pass:
                if idx != len(stages):
                    self.check(fit, "stages_stop_after_first_failed_gate", False,
                               "producer continued after a failed competence stage")
                    consistency = False
                break
        if status == "competence_passed" and (len(stages) != 3 or not all_passed):
            consistency = False
        if status == "competence_failed" and (not stages or all_passed):
            consistency = False
        consistency &= validation_design_ok
        self.check(fit, "competence_status_matches_saved_stage_gates", consistency,
                   f"status={status}; all_stage_gates_passed={all_passed}; validation logits not retained")
        return {"worker_reported_summary_threshold_checked": all_passed and consistency,
                "fit_stopped_on_competence_failure": status == "competence_failed" and consistency,
                "validation_logits_retained": False,
                "validation_numerators_retained": False,
                "validation_deviation": "producer saved summary float rates, not raw logits or integer correct counts; infer numerator integrality from rates and known 256/128 denominators",
                "stages": details}

    def design_valid(self, fit: str, path: Path, result: dict, dimension: int,
                     *, heldout: bool) -> tuple[dict | None, bool]:
        try:
            design = read_json(path)
            bits = np.asarray(design["cue_bits"])
            seed_ok = design.get("dimension") == dimension and design.get("seed") == result.get("seed")
            balanced = (bits.shape == (256, dimension) and np.isin(bits, (-1, 1)).all()
                        and np.all((bits == 1).sum(axis=0) == 128))
            expected_stream = "test" if heldout else "validation"
            fields_ok = (design.get("seed_stream") == expected_stream and seed_ok and balanced)
            if heldout:
                fields_ok &= design.get("forcing_unit_ids") == [0, 1, 2, 3]
                conditions = design.get("conditions", {})
                for repeats, dwell in CONDITIONS:
                    pairs = conditions.get(f"R{repeats}-q{dwell}")
                    fields_ok &= (isinstance(pairs, list) and len(pairs) == 4 and
                                  all(isinstance(pair, list) and len(pair) == 2 and
                                      all(isinstance(body, str) and len(body) == 12 * repeats * dwell
                                          for body in pair) for pair in pairs))
            else:
                pairs_by_repeat = design.get("body_pairs", {})
                for repeats in (0, 1, 2, 4, 8):
                    pairs = pairs_by_repeat.get(str(repeats))
                    fields_ok &= (isinstance(pairs, list) and len(pairs) == 256 and
                                  all(isinstance(pair, list) and len(pair) == 2 and
                                      all(isinstance(body, str) and len(body) == 12 * repeats
                                          for body in pair) for pair in pairs))
            digest_field = "evaluation_design_sha256" if heldout else "validation_design_sha256"
            digest_ok = sha256(path) == result.get(digest_field)
            ok = bool(fields_ok and digest_ok)
            self.check(fit, f"{'evaluation' if heldout else 'validation'}_design_identity_and_balance",
                       ok, "verified saved panel, seed, balance, schedule and result SHA")
            return design, ok
        except Exception as exc:
            self.check(fit, f"{'evaluation' if heldout else 'validation'}_design_identity_and_balance",
                       False, str(exc))
            return None, False

    def reduce_eval(self, fit: str, fitdir: Path, item: dict, dimension: int,
                    design: dict | None) -> dict | None:
        condition = item.get("condition")
        raw_name = item.get("raw_artifact")
        try:
            rel = PurePosixPath(raw_name)
            if rel.is_absolute() or len(rel.parts) != 1 or rel.name != raw_name or ".." in rel.parts:
                raise ValueError("raw_artifact must name a file in the fit directory")
            with np.load(fitdir / raw_name, allow_pickle=False) as archive:
                required = {"logits", "predictions", "labels", "bit_errors", "oracle_predictions",
                            "before_query", "after_write", "cue_bits", "forcing_unit_ids", "arm_names",
                            "completed_cues", "complete", "state_layout"}
                if set(archive.files) != required:
                    raise ValueError(f"NPZ member set differs: {sorted(set(archive.files) ^ required)}")
                a = {key: archive[key] for key in required}
        except Exception as exc:
            self.check(fit, f"{condition}:raw_load", False, str(exc))
            return None

        shape = (4, 256, 2, dimension)
        good = a["logits"].shape == shape
        self.check(fit, f"{condition}:logit_shape", good, str(a["logits"].shape))
        if not good:
            return None
        expected_dtypes = {"logits": np.dtype("float32"), "predictions": np.dtype("uint8"),
                           "labels": np.dtype("bool"), "bit_errors": np.dtype("uint8"),
                           "oracle_predictions": np.dtype("uint8"),
                           "before_query": np.dtype("float32"), "after_write": np.dtype("float32"),
                           "cue_bits": np.dtype("int8"), "forcing_unit_ids": np.dtype("int32"),
                           "completed_cues": np.dtype("int32"), "complete": np.dtype("bool")}
        for key, dtype in expected_dtypes.items():
            self.check(fit, f"{condition}:{key}_dtype", a[key].dtype == dtype,
                       f"expected={dtype}, observed={a[key].dtype}")
        state_shape = (4, 256, 2, 128)
        for key in ("before_query", "after_write"):
            self.check(fit, f"{condition}:{key}_shape", a[key].shape == state_shape,
                       str(a[key].shape))
        self.check(fit, f"{condition}:finite_raw", all(np.isfinite(a[k]).all()
                    for k in ("logits", "before_query", "after_write")))
        self.check(fit, f"{condition}:complete_marker",
                   a["complete"].shape == () and bool(a["complete"].item()))
        self.check(fit, f"{condition}:all_cues_completed",
                   a["completed_cues"].shape == (4,) and np.array_equal(a["completed_cues"], [256]*4))
        self.check(fit, f"{condition}:forcing_ids",
                   np.array_equal(a["forcing_unit_ids"], np.arange(4)))
        self.check(fit, f"{condition}:fixed_arm_order",
                   np.array_equal(a["arm_names"], np.asarray(ARMS)))
        self.check(fit, f"{condition}:state_layout",
                   a["state_layout"].shape == () and a["state_layout"].item() == "hidden_then_cell")
        self.check(fit, f"{condition}:state_dimension_128", all(a[k].shape == state_shape
                   for k in ("before_query", "after_write")))

        bits = a["cue_bits"]
        cue_ok = (bits.shape == (256, dimension) and np.isin(bits, (-1, 1)).all()
                  and np.all((bits == 1).sum(axis=0) == 128))
        self.check(fit, f"{condition}:balanced_cue_bits", cue_ok)
        if not cue_ok:
            return None
        pred = a["logits"] >= 0
        labels = np.broadcast_to((bits > 0)[None, :, None, :], shape)
        errors = pred != labels
        self.check(fit, f"{condition}:prediction_alignment", np.array_equal(a["predictions"], pred))
        self.check(fit, f"{condition}:label_alignment", np.array_equal(a["labels"], labels))
        self.check(fit, f"{condition}:error_alignment", np.array_equal(a["bit_errors"], errors.astype(np.uint8)))
        if design:
            self.check(fit, f"{condition}:design_cue_panel",
                       np.array_equal(np.asarray(design.get("cue_bits")), bits))
            key = condition
            pairs = design.get("conditions", {}).get(key)
            r, q = item.get("repeats"), item.get("dwell")
            design_ok = (isinstance(pairs, list) and len(pairs) == 4 and
                         all(isinstance(pair, list) and len(pair) == 2 and
                             all(isinstance(body, str) and len(body) == 12*r*q for body in pair)
                             for pair in pairs)) if isinstance(r, int) and isinstance(q, int) else False
            self.check(fit, f"{condition}:forcing_design_ids_and_arm_pairs", design_ok)

        correct = ~errors
        per_bit = correct.mean(axis=(0, 1))
        per_error = errors.mean(axis=(0, 1, 3))
        exact = correct.all(axis=-1).mean(axis=(0, 1))
        paired_bit = (errors[:, :, 0, :].astype(np.float64)
                      - errors[:, :, 1, :].astype(np.float64)).mean(axis=(0, 1))
        delta = float(per_error[0] - per_error[1])
        per_forcing = (errors[:, :, 0, :].mean(axis=(1, 2))
                       - errors[:, :, 1, :].mean(axis=(1, 2)))
        self.compare(fit, f"{condition}:per_bit_accuracy", per_bit,
                     np.asarray([item["per_bit_accuracy_by_arm"][arm] for arm in ARMS]))
        self.compare(fit, f"{condition}:mean_bit_error", per_error,
                     [item["mean_bit_error_by_arm"][arm] for arm in ARMS])
        self.compare(fit, f"{condition}:whole_vector_exact_accuracy", exact,
                     [item["whole_vector_exact_accuracy_by_arm"][arm] for arm in ARMS])
        self.compare(fit, f"{condition}:paired_bit_error_by_bit", paired_bit,
                     item["paired_bit_error_difference_periodic_minus_control"])
        self.compare(fit, f"{condition}:paired_error_difference", delta,
                     item["paired_error_difference_periodic_minus_control"])
        self.compare(fit, f"{condition}:paired_error_by_forcing", per_forcing,
                     item["paired_error_difference_by_forcing_unit"])
        self.compare(fit, f"{condition}:per_forcing_arm_error",
                     errors.mean(axis=(1, 3)),
                     np.asarray(item["per_forcing_unit_mean_bit_error_by_arm"]))
        counts = {arm: int(errors[:, :, i, :].sum()) for i, arm in enumerate(ARMS)}
        self.check(fit, f"{condition}:error_counts", counts == item["error_count_by_arm"],
                   f"recomputed={counts}")
        if "oracle_predictions" in a:
            oracle_ok = np.array_equal(a["oracle_predictions"], np.broadcast_to(
                (bits > 0)[None, :, :], (4, 256, dimension)))
            self.check(fit, f"{condition}:oracle_alignment", oracle_ok)

        geometry = item.get("conditional_complete_state_geometry", {}).get("conditional_spectra", [])
        states = a["before_query"]
        summaries = []
        geo_ok = len(geometry) == 8 and states.shape == state_shape
        for forcing in range(4):
            for arm_i, arm in enumerate(ARMS):
                sample = states[forcing, :, arm_i, :].astype(np.float64)
                sample -= sample.mean(axis=0, keepdims=True)
                eig = np.linalg.eigvalsh(sample.T @ sample / 255)
                scale = max(float(np.max(np.abs(eig))), np.finfo(np.float64).tiny)
                tol = 64 * np.finfo(np.float64).eps * max(sample.shape) * scale
                minimum = float(eig.min())
                clamped = np.maximum(eig, 0.0)
                trace = float(clamped.sum())
                pr = float(trace * trace / np.dot(clamped, clamped)) if trace else None
                rank = min(255, 128, (1 << dimension) - 1)
                row = geometry[forcing * 2 + arm_i] if len(geometry) == 8 else {}
                if row.get("forcing_unit") != forcing or row.get("arm") != arm:
                    geo_ok = False
                comparisons = (("eigenvalues", eig.clip(min=0), row.get("eigenvalues_ascending")),
                               ("trace", trace, row.get("trace")),
                               ("participation_ratio", pr, row.get("participation_ratio")),
                               ("minimum_unclamped_eigenvalue", minimum,
                                row.get("minimum_unclamped_eigenvalue")))
                for label, actual, claimed in comparisons:
                    if claimed is None and actual is None:
                        continue
                    ok, diff = close(actual, claimed)
                    if math.isfinite(diff):
                        self.max_abs_difference = max(self.max_abs_difference, diff)
                    if not ok:
                        geo_ok = False
                if (minimum < -tol or row.get("rank_ceiling") != rank
                        or int(np.count_nonzero(clamped > tol)) > rank
                        or row.get("numerical_rank") != int(np.count_nonzero(clamped > tol))):
                    geo_ok = False
                summaries.append({"forcing_unit": forcing, "arm": arm,
                                  "trace": trace, "participation_ratio": pr,
                                  "rank_ceiling": rank})
        self.check(fit, f"{condition}:independent_covariance_spectra", geo_ok,
                   "FP64 within-forcing/arm centering; 1e-8 abs + 1e-6 relative comparison")
        means = {}
        for arm in ARMS:
            rows = [r for r in summaries if r["arm"] == arm]
            means[arm] = {"mean_trace": float(np.mean([r["trace"] for r in rows])),
                          "mean_participation_ratio": (float(np.mean([r["participation_ratio"] for r in rows]))
                                                        if all(r["participation_ratio"] is not None for r in rows)
                                                        else None),
                          "mean_bit_accuracy": float(per_bit[0 if arm == "periodic" else 1].mean()),
                          "whole_vector_exact_accuracy": float(exact[0 if arm == "periodic" else 1])}
        return {"condition": condition, "paired_error_delta": delta,
                "per_arm": means, "per_forcing_unit_delta": per_forcing.tolist()}

    def run(self) -> dict:
        fitdirs = sorted(p for p in self.root.glob("D*-seed*") if p.is_dir())
        self.check("campaign", "all_four_seed0_fits_present",
                   {p.name for p in fitdirs} == {f"D{d}-seed0" for d in (1, 4, 16, 64)})
        fits = []
        status_counts = {"terminal": 0, "partial": 0}
        for fitdir in fitdirs:
            fit = fitdir.name
            inventory, file_info = self.inventory(fitdir, fit)
            try:
                result = read_json(fitdir / "result.json")
                dimension = int(result["dimension"])
            except Exception as exc:
                result, dimension = {}, 0
                self.check(fit, "result_json", False, str(exc))
            status = result.get("status") or "result_missing"
            terminal = status in {"competence_passed", "competence_failed"}
            status_counts["terminal" if terminal else "partial"] += 1
            validation_design, valid_validation_design = (self.design_valid(
                fit, fitdir / "validation-design.json", result, dimension, heldout=False)
                if dimension else (None, False))
            competence = (self.competence(fit, result, dimension, valid_validation_design)
                          if dimension else {})
            design = None
            if result.get("evaluation"):
                design, _ = self.design_valid(fit, fitdir / "evaluation-design.json",
                                              result, dimension, heldout=True)
            else:
                self.check(fit, "heldout_absent_after_failed_gate",
                           status != "competence_failed" or not result.get("heldout_opened"))
            eval_summaries = []
            for item in result.get("evaluation", []):
                summary = self.reduce_eval(fit, fitdir, item, dimension, design)
                if summary:
                    eval_summaries.append(summary)
            by_cond = {x["condition"]: x["paired_error_delta"] for x in eval_summaries}
            evaluation_labels = [item.get("condition") for item in result.get("evaluation", [])]
            if status == "competence_passed":
                expected_labels = [f"R{r}-q{q}" for r, q in CONDITIONS]
                self.check(fit, "complete_frozen_evaluation_grid",
                           evaluation_labels == expected_labels,
                           f"observed={evaluation_labels}")
            else:
                self.check(fit, "failed_fit_has_no_heldout_evaluation",
                           not result.get("evaluation"))
            primary, anchor = by_cond.get("R32-q1"), by_cond.get("R2-q1")
            claimed_primary = result.get("primary_R32q1_periodic_minus_control")
            claimed_anchor = result.get("anchor_R2q1_periodic_minus_control")
            if primary is not None and claimed_primary is not None:
                self.compare(fit, "primary_R32q1_delta", primary, claimed_primary)
            else:
                self.check(fit, "primary_R32q1_delta_absence_matches",
                           primary is None and claimed_primary is None)
            if anchor is not None and claimed_anchor is not None:
                self.compare(fit, "anchor_R2q1_delta", anchor, claimed_anchor)
            else:
                self.check(fit, "anchor_R2q1_delta_absence_matches",
                           anchor is None and claimed_anchor is None)
            did = primary - anchor if primary is not None and anchor is not None else None
            claimed_did = result.get("R32q1_minus_R2q1_difference_in_differences")
            if did is not None and claimed_did is not None:
                self.compare(fit, "R32q1_minus_R2q1_DID", did, claimed_did)
            else:
                self.check(fit, "R32q1_minus_R2q1_DID_absence_matches",
                           did is None and claimed_did is None)
            omissions = file_info["omitted_inventory_files"]
            fits.append({"fit": fit, "status": status, "terminal": terminal,
                         "competence": competence, "verified_files": file_info["verified_scientific_files"],
                         "omitted_inventory_files": omissions,
                         "omitted_checkpoint_reporting": file_info["omitted_checkpoint_reporting"],
                         "evaluation": eval_summaries,
                         "primary_R32q1_delta": primary, "anchor_R2q1_delta": anchor,
                         "R32q1_minus_R2q1_DID": did})
        return {"schema": "independent-multibit-memory-reduction-v1",
                "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                "verifier_sha256": sha256(Path(__file__)),
                "input_inventory_sha256": {p.name: sha256(p / "artifact-inventory.json")
                                           for p in fitdirs if (p / "artifact-inventory.json").is_file()},
                "run_path": str(self.root), "fits_found": len(fitdirs),
                "fit_status_counts": status_counts,
                "max_abs_difference": self.max_abs_difference,
                "tolerances": {"absolute": ATOL, "relative": RTOL},
                "scientific_acceptance": False,
                "interpretation": "Raw held-out numerical integrity only; no harmful-collapse claim.",
                "failed_checks": [c for c in self.checks if c["status"] == "fail"],
                "checks": self.checks, "fits": fits}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = Reducer(args.run).run()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, sort_keys=True, indent=2,
                                          allow_nan=False) + "\n", encoding="ascii")
        print(json.dumps({"fits_found": report["fits_found"],
                          "fit_status_counts": report["fit_status_counts"],
                          "failed_check_count": len(report["failed_checks"]),
                          "output": str(args.output)}, sort_keys=True))
        return 0 if not report["failed_checks"] else 2
    except Exception as exc:
        print(f"reducer error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
