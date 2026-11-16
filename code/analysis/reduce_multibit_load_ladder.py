#!/usr/bin/env python3
"""Independent raw-artifact reduction for multibit load-ladder v2.

Uses only Python's standard library and NumPy. It verifies hashes, raw
validation counts/gates, endpoint replay receipts, held-out arrays and
conditional covariance summaries. It does not load checkpoint objects or
promote any result to scientific acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys

import numpy as np


LOADS, SEEDS = (4, 8, 12, 16, 24, 32, 48, 64), (0, 1, 2)
STAGES = ((6000, (0,)), (18000, (0, 1, 2)),
          (34000, (0, 1, 2, 4, 8)))
CONDITIONS = tuple((r, 1) for r in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)) + ((128, 4), (256, 2))
ARMS, POLARITIES = ("periodic", "control"), ("positive", "negative")
MOTIFS = ("AXAXAXBXBXBX", "AXAXBXAXBXBX", "AXAXBXBXAXBX",
          "AXBXAXAXBXBX", "AXBXBXAXAXBX")
ATOL, RTOL = 1e-8, 1e-6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def np_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()


def read_json(path: Path):
    def reject(value):
        raise ValueError(f"non-finite JSON value {value}")
    with path.open(encoding="ascii") as stream:
        return json.load(stream, parse_constant=reject)


def close(a, b):
    if a is None or b is None:
        return a is None and b is None, 0.0 if a is None and b is None else math.inf
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        return False, math.inf
    return bool(np.allclose(x, y, atol=ATOL, rtol=RTOL)), (float(np.max(np.abs(x-y))) if x.size else 0.0)


def _run_lengths(text):
    runs=[]
    for char in text:
        if runs and runs[-1][0] == char:
            runs[-1]=(char,runs[-1][1]+1)
        else:
            runs.append((char,1))
    return runs


def _has_proper_period(text):
    return any(len(text)%period==0 and text==text[:period]*(len(text)//period)
               for period in range(1,len(text)//2+1))


def valid_body_pair(pair, repeats, dwell):
    """Recheck the frozen five-motif periodic/control construction independently."""
    if (not isinstance(pair,list) or len(pair)!=2 or type(repeats) is not int
            or type(dwell) is not int or repeats<0 or dwell<1
            or any(not isinstance(body,str) for body in pair)):
        return False
    periodic,control=pair
    if repeats==0:
        return periodic==control==""
    horizon=12*repeats*dwell
    if (len(periodic)!=horizon or len(control)!=horizon
            or any(char not in "ABX" for char in periodic+control)):
        return False
    def motif_sequence(body):
        ids=[]; width=12*dwell
        for start in range(0,len(body),width):
            block=body[start:start+width]
            motif="".join(block[i] for i in range(0,width,dwell))
            if block!="".join(char*dwell for char in motif) or motif not in MOTIFS:
                return None
            ids.append(MOTIFS.index(motif))
        return ids
    pids,cids=motif_sequence(periodic),motif_sequence(control)
    if pids is None or cids is None or pids != [pids[-1]]*repeats or cids[-1]!=pids[-1]:
        return False
    if repeats==1:
        return periodic==control
    if periodic==control or _has_proper_period(control):
        return False
    alphabet="ABX"
    symbol_counts=lambda text: tuple(text.count(c) for c in alphabet)
    transitions=lambda text: tuple(text.count(a+b) for a in alphabet for b in alphabet)
    return (symbol_counts(periodic)==symbol_counts(control)
            and transitions(periodic)==transitions(control)
            and sorted(_run_lengths(periodic))==sorted(_run_lengths(control))
            and periodic[-1:]==control[-1:])


class Audit:
    def __init__(self, root: Path):
        self.root = root.resolve(strict=True)
        self.checks, self.maxdiff = [], 0.0

    def check(self, fit, name, ok, detail=""):
        self.checks.append({"fit": fit, "check": name, "status": "pass" if ok else "fail",
                            **({"detail": str(detail)} if detail else {})})

    def compare(self, fit, name, actual, claimed):
        ok, diff = close(actual, claimed)
        if math.isfinite(diff):
            self.maxdiff = max(self.maxdiff, diff)
        self.check(fit, name, ok, f"max_abs_difference={diff:.12g}" if math.isfinite(diff) else "shape/nonfinite mismatch")

    def inventory(self, directory: Path, fit: str):
        path = directory / "artifact-inventory.json"
        try:
            inventory = read_json(path)
            self.check(fit, "inventory_object", isinstance(inventory, dict))
        except Exception as exc:
            self.check(fit, "inventory_read", False, exc)
            return {}
        # Producer inventory covers top-level regular files, before the inventory itself.
        for file in sorted(directory.iterdir()):
            if not file.is_file() or file.name == path.name:
                continue
            expected = inventory.get(file.name)
            ok = (isinstance(expected, dict) and expected.get("sha256") == sha256(file)
                  and expected.get("bytes") == file.stat().st_size)
            self.check(fit, f"inventory:{file.name}", ok,
                       "downloaded file matches listed SHA/size" if ok else "missing or mismatched inventory entry")
        omitted = sorted(name for name in inventory if not (directory / name).is_file())
        return {"top_level_inventory_entries": len(inventory),
                "files_not_downloaded": omitted,
                "scope": "only files actually present here were digest-verified; no whole-inventory claim"}

    def validation_design(self, directory: Path, fit: str, result: dict, d: int, seed: int):
        try:
            design = read_json(directory / "validation-design.json")
            bits = np.asarray(design["cue_bits"])
            ok = (design.get("dimension") == d and design.get("seed") == seed
                  and design.get("rng_namespace") == "multibit-load-ladder-v2"
                  and bits.shape == (256, d) and np.isin(bits, (-1, 1)).all()
                  and np.all((bits == 1).sum(axis=0) == 128))
            for r in (0, 1, 2, 4, 8):
                pairs = design.get("body_pairs", {}).get(str(r))
                ok &= (isinstance(pairs, list) and len(pairs) == 256 and
                       all(valid_body_pair(pair, r, 1) for pair in pairs))
            ok &= sha256(directory / "validation-design.json") == result.get("validation_design_sha256")
            self.check(fit, "balanced_validation_panel_and_design_sha", bool(ok))
            return design, bits, bool(ok)
        except Exception as exc:
            self.check(fit, "validation_design", False, exc)
            return None, None, False

    def validate_stage_raw(self, fit, directory, stage, d, bits, expected_repeats):
        no, update = stage.get("stage"), stage.get("update")
        expected_keys = {"complete", "cue_bits", "arm_names", "polarity_names"}
        for r in expected_repeats:
            expected_keys.update(f"{stem}_R{r}" for stem in (
                "logits", "labels", "predictions", "correct", "bit_correct_counts",
                "bit_denominators", "polarity_correct_counts", "polarity_denominators", "complete"))
        raw_name = stage.get("raw_file")
        ok_path = isinstance(raw_name, str) and PurePosixPath(raw_name).name == raw_name
        self.check(fit, f"stage{no}:raw_path", ok_path)
        if not ok_path:
            return False, False
        try:
            with np.load(directory / raw_name, allow_pickle=False) as z:
                if set(z.files) != expected_keys:
                    raise ValueError(f"NPZ keys differ: missing={sorted(expected_keys-set(z.files))}, extra={sorted(set(z.files)-expected_keys)}")
                a = {key: z[key] for key in expected_keys}
        except Exception as exc:
            self.check(fit, f"stage{no}:raw_load", False, exc)
            return False, False
        self.check(fit, f"stage{no}:complete", a["complete"].shape == () and bool(a["complete"].item()))
        self.check(fit, f"stage{no}:cue_bits", np.array_equal(a["cue_bits"], bits))
        self.check(fit, f"stage{no}:arm_names", np.array_equal(a["arm_names"], np.asarray(ARMS)))
        self.check(fit, f"stage{no}:polarity_names", np.array_equal(a["polarity_names"], np.asarray(POLARITIES)))
        check_summaries = stage.get("checks", [])
        raw_gates = []
        full = True
        for i, r in enumerate(expected_repeats):
            prefix = f"stage{no}:R{r}"
            shape, dshape = (256, 2, d), (2, d)
            logits = a[f"logits_R{r}"]
            labels, predictions, correct = (a[f"labels_R{r}"], a[f"predictions_R{r}"], a[f"correct_R{r}"])
            full &= logits.shape == shape
            for key, expected_shape, dtype in ((f"labels_R{r}", shape, np.dtype("uint8")),
                                               (f"predictions_R{r}", shape, np.dtype("uint8")),
                                               (f"correct_R{r}", shape, np.dtype("uint8")),
                                               (f"bit_correct_counts_R{r}", dshape, np.dtype("int32")),
                                               (f"bit_denominators_R{r}", dshape, np.dtype("int32")),
                                               (f"polarity_correct_counts_R{r}", (2,d,2), np.dtype("int32")),
                                               (f"polarity_denominators_R{r}", (d,2), np.dtype("int32"))):
                self.check(fit, f"{prefix}:{key}_shape_dtype",
                           a[key].shape == expected_shape and a[key].dtype == dtype,
                           f"shape={a[key].shape}, dtype={a[key].dtype}")
                full &= a[key].shape == expected_shape and a[key].dtype == dtype
            target = np.broadcast_to((bits > 0)[:, None, :], shape).astype(np.uint8)
            pred = (logits >= 0).astype(np.uint8)
            cor = (pred == target).astype(np.uint8)
            self.check(fit, f"{prefix}:finite_logits", np.isfinite(logits).all())
            self.check(fit, f"{prefix}:labels_alignment", np.array_equal(labels, target))
            self.check(fit, f"{prefix}:predictions_alignment", np.array_equal(predictions, pred))
            self.check(fit, f"{prefix}:correct_alignment", np.array_equal(correct, cor))
            bit_counts = cor.sum(axis=0).astype(np.int32)
            bit_denoms = np.full((2, d), 256, dtype=np.int32)
            pol_counts = np.zeros((2, d, 2), dtype=np.int32)
            pol_denoms = np.zeros((d, 2), dtype=np.int32)
            for b in range(d):
                for pi, mask in enumerate((bits[:, b] > 0, bits[:, b] < 0)):
                    pol_denoms[b, pi] = int(mask.sum())
                    pol_counts[:, b, pi] = cor[mask, :, b].sum(axis=0)
            count_ok = (np.array_equal(a[f"bit_correct_counts_R{r}"], bit_counts)
                        and np.array_equal(a[f"bit_denominators_R{r}"], bit_denoms)
                        and np.array_equal(a[f"polarity_correct_counts_R{r}"], pol_counts)
                        and np.array_equal(a[f"polarity_denominators_R{r}"], pol_denoms))
            self.check(fit, f"{prefix}:raw_integer_counts_and_denominators", count_ok)
            summary = check_summaries[i] if i < len(check_summaries) else {}
            self.check(fit, f"{prefix}:summary_identity",
                       summary.get("repeats") == r and summary.get("cue_vectors") == 256
                       and summary.get("active_bits") == d)
            self.check(fit, f"{prefix}:summary_bit_counts",
                       summary.get("bit_correct_counts_by_arm") == bit_counts.tolist()
                       and summary.get("bit_denominators_by_arm") == [256, 256])
            expected_pc = {arm: pol_counts[ai].tolist() for ai, arm in enumerate(ARMS)}
            expected_pd = pol_denoms.tolist()
            self.check(fit, f"{prefix}:summary_polarity_counts",
                       summary.get("polarity_order") == list(POLARITIES)
                       and summary.get("polarity_correct_counts_by_arm") == expected_pc
                       and summary.get("polarity_denominators_per_bit") == expected_pd)
            bit_acc = bit_counts / bit_denoms
            pol_acc = pol_counts / pol_denoms[None, :, :]
            gate = bool((bit_counts * 1.0 / 256 >= .95).all() and
                        (pol_counts / pol_denoms[None, :, :] >= .95).all())
            # Verify summary accuracy floats against the exact integer fractions.
            reported_bit = summary.get("per_bit_accuracy_by_arm", {})
            for ai, arm in enumerate(ARMS):
                self.compare(fit, f"{prefix}:{arm}_bit_rates", bit_acc[ai], reported_bit.get(arm, []))
            reported_pol = summary.get("per_bit_polarity_accuracy_by_arm", {})
            for ai, arm in enumerate(ARMS):
                actual = {str(b): {"positive": float(pol_acc[ai,b,0]),
                                   "negative": float(pol_acc[ai,b,1])} for b in range(d)}
                self.compare(fit, f"{prefix}:{arm}_polarity_rates",
                             [actual[str(b)][p] for b in range(d) for p in POLARITIES],
                             [reported_pol.get(arm, {}).get(str(b), {}).get(p)
                              for b in range(d) for p in POLARITIES])
            raw_gates.append(gate and count_ok)
        all_gates = len(check_summaries) == len(expected_repeats) and len(raw_gates) == len(expected_repeats) and all(raw_gates)
        saved_pass = stage.get("passed") is True
        self.check(fit, f"stage{no}:gate_decision_from_raw_counts", saved_pass == all_gates,
                   f"raw_gate_passed={all_gates}, saved={stage.get('passed')}")
        manifest = read_json(directory / f"stage-{no}-validation.json")
        self.check(fit, f"stage{no}:validation_manifest_matches_result", manifest == stage)
        self.check(fit, f"stage{no}:checkpoint_identity",
                   stage.get("checkpoint_sha256") is not None)
        return all_gates, full

    def checkpoints(self, fit, directory, result):
        records = result.get("stage_checkpoints", [])
        ok = True
        for record in records:
            stage = record.get("stage")
            name = record.get("path")
            safe = isinstance(name, str) and PurePosixPath(name).name == name
            if not safe:
                self.check(fit, f"checkpoint_stage_{stage}:safe_path", False)
                ok = False; continue
            manifest_path = directory / f"stage-{stage:02d}-checkpoint.json" if not record.get("partial") else None
            if manifest_path:
                try:
                    manifest = read_json(manifest_path)
                    manifest_ok = manifest == record
                    self.check(fit, f"checkpoint_stage_{stage}:manifest_match", manifest_ok)
                    ok &= manifest_ok
                except Exception as exc:
                    self.check(fit, f"checkpoint_stage_{stage}:manifest_read", False, exc)
                    ok = False
            ckpt = directory / name
            digest_ok = ckpt.is_file() and sha256(ckpt) == record.get("sha256")
            self.check(fit, f"checkpoint_stage_{stage}:file_sha", digest_ok,
                       "checkpoint is byte-hash verified; not deserialized")
            ok &= digest_ok
        if records:
            last_ok = result.get("checkpoint_sha256") == records[-1].get("sha256")
            self.check(fit, "result_final_checkpoint_hash", last_ok)
            ok &= last_ok
        by_stage = {item.get("stage"): item.get("sha256") for item in records
                    if not item.get("partial")}
        expected_stages = {item.get("stage") for item in result.get("stage_results", [])}
        coverage_ok = expected_stages.issubset(set(by_stage))
        if result.get("stage_results") and not records:
            coverage_ok = False
        self.check(fit, "stage_checkpoint_record_coverage", coverage_ok)
        ok &= coverage_ok
        for stage in result.get("stage_results", []):
            match = by_stage.get(stage.get("stage")) == stage.get("checkpoint_sha256")
            self.check(fit, f"stage{stage.get('stage')}:checkpoint_hash_crosslink", match)
            ok &= match
        return bool(ok)

    def replay(self, fit, directory, result, dimension, stage_raw):
        receipt = result.get("endpoint_replay")
        if receipt is None:
            should = result.get("heldout_opened") is False and result.get("status") != "competence_passed"
            self.check(fit, "endpoint_replay_absent_only_when_not_opened", should)
            return should
        try:
            saved = read_json(directory / "endpoint-replay.json")
            self.check(fit, "endpoint_replay_receipt_matches_result", saved == receipt)
            expected_repeats = STAGES[-1][1]
            ok = (receipt.get("passed") is True and receipt.get("rtol") == 0.0
                  and receipt.get("atol") == 0.0 and
                  receipt.get("checkpoint_sha256") == result.get("checkpoint_sha256")
                  and isinstance(receipt.get("model_sha256"), str)
                  and len(receipt.get("model_sha256")) == 64
                  and all(c in "0123456789abcdef" for c in receipt.get("model_sha256", ""))
                  and len(receipt.get("comparisons", [])) == len(expected_repeats))
            with np.load(directory / "stage-3-validation.npz", allow_pickle=False) as z:
                for repeat, comparison in zip(expected_repeats, receipt.get("comparisons", [])):
                    raw_hash = np_sha(z[f"logits_R{repeat}"])
                    ok &= (comparison.get("repeats") == repeat and comparison.get("exact") is True
                           and comparison.get("original_logits_sha256") == raw_hash
                           and comparison.get("reloaded_logits_sha256") == raw_hash)
            self.check(fit, "endpoint_replay_exact_logits_hashes", bool(ok),
                       "raw stage-3 logits hash equals both replay receipt hashes; checkpoint not loaded independently")
            return bool(ok)
        except Exception as exc:
            self.check(fit, "endpoint_replay_receipt", False, exc)
            return False

    def evaluation(self, fit, directory, item, d, design):
        condition, rawname = item.get("condition"), item.get("raw_artifact")
        try:
            if (not isinstance(rawname, str) or PurePosixPath(rawname).name != rawname
                    or ".." in PurePosixPath(rawname).parts):
                raise ValueError("unsafe raw artifact path")
            with np.load(directory / rawname, allow_pickle=False) as z:
                required = {"logits", "predictions", "labels", "oracle_predictions", "bit_errors",
                            "before_query", "after_write", "cue_bits", "forcing_unit_ids", "arm_names",
                            "completed_cues", "complete", "state_layout"}
                if set(z.files) != required:
                    raise ValueError("evaluation NPZ member set mismatch")
                a = {k: z[k] for k in required}
        except Exception as exc:
            self.check(fit, f"{condition}:raw_load", False, exc)
            return None
        shape, sshape = (4,256,2,d), (4,256,2,128)
        dtype_ok = (a["logits"].dtype == np.dtype("float32")
                    and all(a[k].dtype == np.dtype("uint8") for k in
                            ("predictions", "oracle_predictions", "bit_errors"))
                    and a["labels"].dtype == np.dtype("bool")
                    and all(a[k].dtype == np.dtype("float32") for k in ("before_query", "after_write"))
                    and a["cue_bits"].dtype == np.dtype("int8")
                    and a["forcing_unit_ids"].dtype == np.dtype("int32")
                    and a["completed_cues"].dtype == np.dtype("int32")
                    and a["complete"].dtype == np.dtype("bool"))
        self.check(fit, f"{condition}:shape_and_dtype",
                   a["logits"].shape == shape and dtype_ok
                   and all(a[k].shape == shape for k in ("predictions","labels","bit_errors"))
                   and all(a[k].shape == sshape for k in ("before_query","after_write"))
                   and a["cue_bits"].shape == (256,d))
        if a["logits"].shape != shape or a["before_query"].shape != sshape:
            return None
        self.check(fit, f"{condition}:finite_arrays", all(np.isfinite(a[k]).all() for k in ("logits","before_query","after_write")))
        bits = a["cue_bits"]
        cue_ok = bits.shape == (256,d) and np.isin(bits,(-1,1)).all() and np.all((bits == 1).sum(axis=0) == 128)
        self.check(fit, f"{condition}:cue_balance", cue_ok)
        if not cue_ok:
            return None
        target = np.broadcast_to((bits > 0)[None,:,None,:], shape)
        pred = a["logits"] >= 0
        err = pred != target
        self.check(fit, f"{condition}:labels_predictions_errors",
                   np.array_equal(a["labels"], target)
                   and np.array_equal(a["predictions"], pred.astype(np.uint8))
                   and np.array_equal(a["bit_errors"], err.astype(np.uint8)))
        self.check(fit, f"{condition}:forcing_arm_and_completion",
                   np.array_equal(a["forcing_unit_ids"], np.arange(4))
                   and np.array_equal(a["arm_names"], np.asarray(ARMS))
                   and np.array_equal(a["completed_cues"], np.full(4,256))
                   and a["complete"].shape == () and bool(a["complete"].item())
                   and a["state_layout"].item() == "hidden_then_cell")
        if design:
            r,q = item.get("repeats"),item.get("dwell")
            pairs=design.get("conditions",{}).get(condition)
            design_ok=(type(r) is int and type(q) is int and (r,q) in CONDITIONS
                       and condition==f"R{r}-q{q}" and item.get("horizon")==12*r*q
                       and np.array_equal(bits, np.asarray(design.get("cue_bits")))
                       and design.get("forcing_unit_ids")==[0,1,2,3]
                       and isinstance(pairs,list) and len(pairs)==4
                       and all(valid_body_pair(pair,r,q) for pair in pairs))
            self.check(fit,f"{condition}:design_alignment",design_ok)
        else:
            design_ok=False
        cor = ~err
        bit_acc=cor.mean(axis=(0,1)); mean_err=err.mean(axis=(0,1,3)); whole=cor.all(axis=-1).mean(axis=(0,1))
        paired_bit=(err[:,:,0,:].astype(float)-err[:,:,1,:].astype(float)).mean(axis=(0,1))
        delta=float(mean_err[0]-mean_err[1])
        delta_force=(err[:,:,0,:].mean(axis=(1,2))-err[:,:,1,:].mean(axis=(1,2)))
        for key,actual,claimed in (
            ("per_bit_accuracy",bit_acc,item.get("per_bit_accuracy_by_arm")),
            ("mean_bit_error",mean_err,[item["mean_bit_error_by_arm"][a] for a in ARMS]),
            ("whole_vector_accuracy",whole,[item["whole_vector_exact_accuracy_by_arm"][a] for a in ARMS]),
            ("paired_bit_error",paired_bit,item["paired_bit_error_difference_periodic_minus_control"]),
            ("paired_error_delta",delta,item["paired_error_difference_periodic_minus_control"]),
            ("per_forcing_delta",delta_force,item["paired_error_difference_by_forcing_unit"]),
            ("per_forcing_arm_error",err.mean(axis=(1,3)),item["per_forcing_unit_mean_bit_error_by_arm"])):
            if key == "per_bit_accuracy":
                claimed=[claimed[a] for a in ARMS]
            self.compare(fit,f"{condition}:{key}",actual,claimed)
        oracle=np.array_equal(a["oracle_predictions"],np.broadcast_to((bits>0)[None],(4,256,d)))
        self.check(fit,f"{condition}:oracle",oracle)
        self.compare(fit,f"{condition}:oracle_exact_accuracy",
                     float(np.mean(np.all(a["oracle_predictions"] == (bits[None] > 0),axis=-1))),
                     item.get("oracle_exact_accuracy"))
        geo=item.get("conditional_complete_state_geometry",{}).get("conditional_spectra",[])
        geometry=item.get("conditional_complete_state_geometry",{})
        spectrum=[]; geo_ok=(len(geo)==8 and geometry.get("independent_forcing_units")==4
                             and geometry.get("cue_vectors_per_forcing_unit_arm")==256
                             and geometry.get("theoretical_rank_ceiling")==min(255,128,(1<<d)-1))
        for f in range(4):
            for ai,arm in enumerate(ARMS):
                x=a["before_query"][f,:,ai,:].astype(np.float64); x-=x.mean(0,keepdims=True)
                eig=np.linalg.eigvalsh(x.T@x/255); scale=max(float(np.max(np.abs(eig))),np.finfo(float).tiny)
                tol=64*np.finfo(float).eps*max(x.shape)*scale; minimum=float(eig.min()); eig2=np.maximum(eig,0)
                geo_ok &= minimum >= -tol
                trace=float(eig2.sum()); pr=float(trace*trace/np.dot(eig2,eig2)) if trace else None
                rank=min(255,128,(1<<d)-1); row=geo[f*2+ai] if len(geo)==8 else {}
                checks=((eig2,row.get("eigenvalues_ascending")),(trace,row.get("trace")),
                        (pr,row.get("participation_ratio")),(minimum,row.get("minimum_unclamped_eigenvalue")))
                for actual,claimed in checks:
                    ok,diff=close(actual,claimed)
                    if math.isfinite(diff): self.maxdiff=max(self.maxdiff,diff)
                    geo_ok &= ok
                geo_ok &= row.get("rank_ceiling")==rank and row.get("numerical_rank")==int(np.count_nonzero(eig2>tol))
                geo_ok &= row.get("forcing_unit")==f and row.get("arm")==arm
                geo_ok &= row.get("coordinates")=="hidden_then_cell" and row.get("state_dimension")==128
                tolerance_ok,_=close(row.get("negative_eigenvalue_tolerance"),tol)
                geo_ok &= row.get("samples")==256 and tolerance_ok
                geo_ok &= row.get("centering")=="within forcing unit and arm across cue vectors"
                geo_ok &= row.get("normalization")=="sample covariance, n-1; unrescaled FP32 state values"
                spectrum.append({"forcing":f,"arm":arm,"trace":trace,"participation_ratio":pr})
        self.check(fit,f"{condition}:independent_FP64_covariance_spectra",bool(geo_ok),
                   "center within forcing and arm; FP64 sample covariance; atol=1e-8 rtol=1e-6")
        per_arm={}
        for ai,arm in enumerate(ARMS):
            rows=[x for x in spectrum if x["arm"]==arm]
            per_arm[arm]={"mean_bit_accuracy":float(bit_acc[ai].mean()),
                          "whole_vector_accuracy":float(whole[ai]),
                          "mean_trace":float(np.mean([x["trace"] for x in rows])),
                          "mean_participation_ratio":float(np.mean([x["participation_ratio"] for x in rows])) if all(x["participation_ratio"] is not None for x in rows) else None}
        return {"condition":condition,"paired_error_delta":delta,"per_arm":per_arm,
                "per_forcing_delta":delta_force.tolist(),
                "treatment_construction_independently_verified":bool(design_ok)}

    @staticmethod
    def status_consistent(status, heldout, evaluation, gate_results, replay_ok, checkpoint_ok):
        if status not in {"competence_passed", "competence_failed", "stopped_partial"}:
            return False
        if any(not value for value in gate_results[:-1]):
            return False  # A later stage cannot follow an earlier failed/unverifiable gate.
        all_gates = len(gate_results) == 3 and all(gate_results)
        if heldout:
            return (status in {"competence_passed", "stopped_partial"}
                    and all_gates and replay_ok and checkpoint_ok)
        if evaluation:
            return False
        if status == "competence_passed":
            return False
        if status == "competence_failed":
            return bool(gate_results and not gate_results[-1]
                        and all(gate_results[:-1]) and checkpoint_ok)
        return not any(not value for value in gate_results)

    @staticmethod
    def duplicate_load_seed_groups(fits):
        groups = {}
        for item in fits:
            key = (item.get("dimension"), item.get("seed"))
            groups.setdefault(key, []).append(item)
        return [{"dimension": d, "seed": seed,
                 "attempts": [{"fit": item.get("fit"), "path": item.get("path"),
                               "run_id": item.get("lineage", {}).get("run_id"),
                               "source_sha256": item.get("lineage", {}).get("source_sha256"),
                               "hostname": item.get("lineage", {}).get("hostname"),
                               "slurm_job_id": item.get("lineage", {}).get("slurm_job_id")}
                              for item in attempts]}
                for (d, seed), attempts in sorted(groups.items()) if len(attempts) > 1]

    @staticmethod
    def evaluation_order_valid(labels):
        expected=[f"R{r}-q{q}" for r,q in CONDITIONS]
        return len(labels)<=len(expected) and labels==expected[:len(labels)]

    def fit(self, directory: Path):
        name=directory.name
        try: result=read_json(directory/"result.json")
        except Exception as exc:
            try:
                identity=read_json(directory/"identity.json")
                d,seed=identity.get("dimension"),identity.get("seed")
                self.check(name,"result_json",False,exc)
                return {"fit":f"D{d}-seed{seed}","path":str(directory),"dimension":d,"seed":seed,
                        "status":"result_missing","lineage":{"run_id":None,
                        "source_sha256":identity.get("source_sha256"),"hostname":identity.get("hostname"),
                        "slurm_job_id":identity.get("slurm_job_id")}}
            except Exception:
                self.check(name,"result_json",False,exc); return {"fit":name,"path":str(directory),"status":"result_missing"}
        d,seed=result.get("dimension"),result.get("seed")
        label=f"D{d}-seed{seed}"
        valid_name=directory.name==f"D{d}"
        self.check(label,"fit_identity",valid_name and d in LOADS and seed in SEEDS)
        inventory=self.inventory(directory,label)
        _,bits,design_ok=self.validation_design(directory,label,result,d,seed)
        stages=result.get("stage_results",[])
        gate_results=[]
        for i,stage in enumerate(stages):
            if i>=len(STAGES):
                self.check(label,"no_extra_stages",False);break
            endpoint,repeats=STAGES[i]
            self.check(label,f"stage{i+1}:schedule",stage.get("stage")==i+1 and stage.get("update")==endpoint)
            if bits is not None:
                gate,rawok=self.validate_stage_raw(label,directory,stage,d,bits,repeats)
                gate_results.append(gate and rawok)
        checkpoint_ok=self.checkpoints(label,directory,result)
        passed_all=len(stages)==3 and len(gate_results)==3 and all(gate_results)
        failed_stage=next((i for i,g in enumerate(gate_results) if not g),None)
        status=result.get("status")
        last_update=stages[-1].get("update") if stages else 0
        self.check(label,"result_update_matches_last_validation",
                   result.get("updates")==last_update if status in {"competence_passed","competence_failed"} else True,
                   f"result_updates={result.get('updates')}, last_stage_update={last_update}")
        replay_ok=self.replay(label,directory,result,d,bits)
        status_ok=self.status_consistent(status, bool(result.get("heldout_opened")),
                                         result.get("evaluation", []), gate_results,
                                         replay_ok, checkpoint_ok)
        self.check(label,"status_requires_ordered_gates_checkpoint_and_replay",status_ok)
        eval_design=None
        eval_items=result.get("evaluation",[])
        if eval_items:
            try:
                eval_design=read_json(directory/"evaluation-design.json")
                self.check(label,"evaluation_design_sha",
                           sha256(directory/"evaluation-design.json")==result.get("evaluation_design_sha256"))
            except Exception as exc:self.check(label,"evaluation_design",False,exc)
        reduced=[]
        for item in eval_items:
            val=self.evaluation(label,directory,item,d,eval_design)
            if val: reduced.append(val)
        labels=[x.get("condition") for x in eval_items]
        self.check(label,"evaluation_conditions_are_unique_frozen_prefix",
                   self.evaluation_order_valid(labels),f"observed={labels}")
        if status=="competence_passed":
            expected=[f"R{r}-q{q}" for r,q in CONDITIONS]
            self.check(label,"complete_13_condition_grid",labels==expected)
        treatment_verified=bool(design_ok and all(x.get("treatment_construction_independently_verified")
                                                   for x in reduced))
        by={x["condition"]:x["paired_error_delta"] for x in reduced}
        primary,anchor=by.get("R32-q1"),by.get("R2-q1")
        did=primary-anchor if primary is not None and anchor is not None else None
        for key,actual,claimed in (("primary_R32q1",primary,result.get("primary_R32q1_periodic_minus_control")),
                                   ("anchor_R2q1",anchor,result.get("anchor_R2q1_periodic_minus_control")),
                                   ("DID",did,result.get("R32q1_minus_R2q1_difference_in_differences"))):
            if actual is None and claimed is None:self.check(label,key+"_both_absent",True)
            elif actual is not None and claimed is not None:self.compare(label,key,actual,claimed)
            else:self.check(label,key+"_presence_matches",False)
        lineage={}
        try:
            identity=read_json(directory/"identity.json")
            lineage={k:identity.get(k) for k in ("source_sha256","primitive_source_sha256","protocol_sha256",
                      "config_sha256","git_commit","hostname","slurm_job_id")}
            run_identity=read_json(directory.parent/"run-identity.json")
            lineage.update({"run_id":run_identity.get("run_id"),"run_seed":run_identity.get("seed"),
                           "run_source_sha256":run_identity.get("source_sha256")})
        except Exception as exc:
            self.check(label,"attempt_lineage",False,exc)
        return {"fit":label,"path":str(directory),"lineage":lineage,"status":status,"seed":seed,"dimension":d,
                "stage_gate_results_raw_verified":gate_results,"checkpoint_files_verified":checkpoint_ok,
                "endpoint_replay_receipt_verified":replay_ok,"inventory":inventory,
                "treatment_construction_independently_verified":treatment_verified,
                "heldout_opened":result.get("heldout_opened"),"evaluation":reduced,
                "primary_R32q1_delta":primary,"anchor_R2q1_delta":anchor,"DID":did}

    def run(self):
        dirs=set()
        for filename in ("result.json","status.json","identity.json"):
            dirs.update(p.parent for p in self.root.rglob(filename)
                        if p.parent.name.startswith("D") and p.parent.name[1:].isdigit())
        dirs=sorted(dirs)
        fits=[self.fit(path) for path in dirs]
        observed={(f.get("dimension"),f.get("seed")) for f in fits}
        duplicate_groups=self.duplicate_load_seed_groups(fits)
        self.check("run","no_duplicate_load_seed_attempts",not duplicate_groups,
                   f"duplicates retained for lineage review: {duplicate_groups}" if duplicate_groups else "")
        missing=[{"dimension":d,"seed":s} for d in LOADS for s in SEEDS if (d,s) not in observed]
        return {"schema":"independent-multibit-load-ladder-v2-reduction-v1",
                "run_path":str(self.root),"fits_found":len(fits),"expected_loads":list(LOADS),
                "expected_seeds":list(SEEDS),"missing_load_seed_fits":missing,
                "duplicate_load_seed_attempts":duplicate_groups,
                "max_abs_difference":self.maxdiff,"tolerances":{"absolute":ATOL,"relative":RTOL},
                "scientific_acceptance":False,
                "periodicity_interpretation_accepted":False,
                "limitations":["No causal collapse or capacity claim is accepted.",
                    "Checkpoint files are byte-hash checked but not deserialized or independently replayed.",
                    "Endpoint replay is checked from raw validation-logit hashes and worker receipt, not a second LSTM implementation.",
                    "Zero-state query scores require model reconstruction and remain worker-reported."],
                "failed_checks":[c for c in self.checks if c["status"]=="fail"],
                "checks":self.checks,"fits":fits}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(argv)
    report=Audit(args.run).run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="ascii")
    print(json.dumps({"fits_found":report["fits_found"],"missing_fit_count":len(report["missing_load_seed_fits"]),
                      "failed_check_count":len(report["failed_checks"]),"output":str(args.output)},sort_keys=True))
    return 0 if not report["failed_checks"] else 2


if __name__=="__main__":
    raise SystemExit(main())
