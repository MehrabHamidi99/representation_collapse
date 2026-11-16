from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "analysis"))

from reduce_multibit_load_ladder import Audit, ARMS, CONDITIONS, valid_body_pair  # noqa: E402


def _validation_fixture(directory: Path, *, corrupt_counts: bool = False):
    d, bits = 4, np.tile(np.asarray([[-1, -1, 1, 1], [1, 1, -1, -1]], dtype=np.int8), (128, 1))
    target = np.broadcast_to((bits > 0)[:, None, :], (256, 2, d)).astype(np.uint8)
    logits = np.where(target, 1.0, -1.0).astype(np.float32)
    pred = (logits >= 0).astype(np.uint8)
    correct = (pred == target).astype(np.uint8)
    bit_counts = correct.sum(axis=0).astype(np.int32)
    polarity_counts = np.full((2, d, 2), 128, dtype=np.int32)
    if corrupt_counts:
        bit_counts[0, 0] -= 1
    arrays = {
        "complete": np.asarray(True), "cue_bits": bits,
        "arm_names": np.asarray(("periodic", "control")),
        "polarity_names": np.asarray(("positive", "negative")),
        "logits_R0": logits, "labels_R0": target, "predictions_R0": pred,
        "correct_R0": correct, "bit_correct_counts_R0": bit_counts,
        "bit_denominators_R0": np.full((2, d), 256, dtype=np.int32),
        "polarity_correct_counts_R0": polarity_counts,
        "polarity_denominators_R0": np.full((d, 2), 128, dtype=np.int32),
        "complete_R0": np.asarray(True),
    }
    np.savez(directory / "stage-1-validation.npz", **arrays)
    rates = {arm: [1.0] * d for arm in ("periodic", "control")}
    pol = {arm: {str(i): {"positive": 1.0, "negative": 1.0} for i in range(d)}
           for arm in ("periodic", "control")}
    summary = {"repeats": 0, "cue_vectors": 256, "active_bits": d,
               "bit_correct_counts_by_arm": correct.sum(axis=0).tolist(),
               "bit_denominators_by_arm": [256, 256],
               "polarity_order": ["positive", "negative"],
               "polarity_correct_counts_by_arm": {
                   arm: polarity_counts[i].tolist() for i, arm in enumerate(("periodic", "control"))},
               "polarity_denominators_per_bit": np.full((d, 2), 128).tolist(),
               "per_bit_accuracy_by_arm": rates,
               "per_bit_polarity_accuracy_by_arm": pol}
    stage = {"stage": 1, "update": 6000, "passed": True, "repeats": [0],
             "checks": [summary], "checkpoint_sha256": "a" * 64,
             "raw_file": "stage-1-validation.npz"}
    (directory / "stage-1-validation.json").write_text(json.dumps(stage) + "\n")
    return stage, bits


class MultibitLoadReducerCorruptionTests(unittest.TestCase):
    def test_frozen_paired_body_treatment_and_controls_are_reconstructed(self):
        motif0,motif1="AXAXAXBXBXBX","AXAXBXAXBXBX"
        self.assertTrue(valid_body_pair(["",""],0,1))
        self.assertTrue(valid_body_pair([motif0,motif0],1,1))
        self.assertTrue(valid_body_pair([motif0*2,motif1+motif0],2,1))
        q=4
        expand=lambda motif:"".join(char*q for char in motif)
        self.assertTrue(valid_body_pair([expand(motif0)*2,expand(motif1)+expand(motif0)],2,q))
        self.assertFalse(valid_body_pair([motif0*2,motif0*2],2,1))
        self.assertFalse(valid_body_pair([motif0,""],1,1))

    def test_independently_recomputed_raw_validation_counts_accept_clean_stage(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            stage, bits = _validation_fixture(directory)
            audit = Audit(directory)
            gates, raw_ok = audit.validate_stage_raw("D4-seed0", directory, stage, 4, bits, (0,))
            self.assertTrue(gates)
            self.assertTrue(raw_ok)
            self.assertFalse([x for x in audit.checks if x["status"] == "fail"])

    def test_invented_raw_correct_count_is_detected_even_when_gate_still_passes(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            stage, bits = _validation_fixture(directory, corrupt_counts=True)
            audit = Audit(directory)
            gates, _ = audit.validate_stage_raw("D4-seed0", directory, stage, 4, bits, (0,))
            self.assertFalse(gates)  # A malformed count makes the gate unverifiable.
            failures = {x["check"] for x in audit.checks if x["status"] == "fail"}
            self.assertIn("stage1:R0:raw_integer_counts_and_denominators", failures)

    def test_checkpoint_hash_manifest_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            payload = directory / "stage-01-checkpoint.pt"
            payload.write_bytes(b"synthetic checkpoint bytes")
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            record = {"stage": 1, "path": payload.name, "sha256": digest, "update": 6000}
            (directory / "stage-01-checkpoint.json").write_text(json.dumps(record) + "\n")
            record["sha256"] = "0" * 64
            result = {"stage_checkpoints": [record], "checkpoint_sha256": record["sha256"],
                      "stage_results": []}
            audit = Audit(directory)
            audit.checkpoints("D4-seed0", directory, result)
            failures = {x["check"] for x in audit.checks if x["status"] == "fail"}
            self.assertIn("checkpoint_stage_1:manifest_match", failures)
            self.assertIn("checkpoint_stage_1:file_sha", failures)

    def test_heldout_evaluation_accepts_producer_bool_labels_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            directory=Path(td); d=4; rng=np.random.default_rng(91)
            bits=np.empty((256,d),dtype=np.int8)
            for b in range(d):
                values=np.r_[np.full(128,-1,dtype=np.int8),np.full(128,1,dtype=np.int8)]
                rng.shuffle(values);bits[:,b]=values
            motif0,motif1="AXAXAXBXBXBX","AXAXBXAXBXBX"
            body=lambda motif:"".join(c for c in motif)
            pair=[body(motif0)+body(motif0),body(motif1)+body(motif0)]
            self.assertTrue(valid_body_pair(pair,2,1))
            states=rng.normal(size=(4,256,2,128)).astype(np.float32)
            after=rng.normal(size=(4,256,2,128)).astype(np.float32)
            labels=np.broadcast_to((bits>0)[None,:,None,:],(4,256,2,d)).copy()  # Producer saves bool labels.
            logits=np.where(labels,3.0,-3.0).astype(np.float32)
            logits[0,0,0,0]*=-1
            logits[2,4,1,2]*=-1
            pred=(logits>=0).astype(np.uint8); errors=(pred!=labels).astype(np.uint8)
            oracle=np.broadcast_to((bits>0)[None],(4,256,d)).astype(np.uint8).copy()
            arrays={"logits":logits,"predictions":pred,"labels":labels,"oracle_predictions":oracle,
                    "bit_errors":errors,"before_query":states,"after_write":after,"cue_bits":bits,
                    "forcing_unit_ids":np.arange(4,dtype=np.int32),"arm_names":np.asarray(ARMS),
                    "completed_cues":np.full(4,256,dtype=np.int32),"complete":np.asarray(True),
                    "state_layout":np.asarray("hidden_then_cell")}
            raw=directory/"R2-q1.npz";np.savez(raw,**arrays)
            correct=pred.astype(bool)==labels
            bit_acc=correct.mean(axis=(0,1));mean_err=errors.mean(axis=(0,1,3))
            whole=correct.all(axis=-1).mean(axis=(0,1))
            pol={arm:{} for arm in ARMS}
            for ai,arm in enumerate(ARMS):
                for b in range(d):
                    pol[arm][str(b)]={}
                    for sign,mask in (("positive",bits[:,b]>0),("negative",bits[:,b]<0)):
                        pol[arm][str(b)][sign]=float(correct[:,mask,ai,b].mean())
            geo=[]
            for f in range(4):
                for ai,arm in enumerate(ARMS):
                    x=states[f,:,ai,:].astype(np.float64);x-=x.mean(0,keepdims=True)
                    eig=np.linalg.eigvalsh(x.T@x/255);scale=max(float(np.max(np.abs(eig))),np.finfo(float).tiny)
                    tol=64*np.finfo(float).eps*max(x.shape)*scale; minimum=float(eig.min());eig=np.maximum(eig,0)
                    trace=float(eig.sum());pr=float(trace*trace/np.dot(eig,eig)) if trace else None
                    geo.append({"forcing_unit":f,"arm":arm,"samples":256,"coordinates":"hidden_then_cell",
                                "state_dimension":128,"centering":"within forcing unit and arm across cue vectors",
                                "normalization":"sample covariance, n-1; unrescaled FP32 state values",
                                "rank_ceiling":min(255,128,(1<<d)-1),"negative_eigenvalue_tolerance":tol,
                                "minimum_unclamped_eigenvalue":minimum,"trace":trace,"participation_ratio":pr,
                                "numerical_rank":int(np.count_nonzero(eig>tol)),"eigenvalues_ascending":eig.tolist()})
            delta=float(mean_err[0]-mean_err[1]);force_delta=errors[:,:,0,:].mean((1,2))-errors[:,:,1,:].mean((1,2))
            item={"condition":"R2-q1","raw_artifact":raw.name,"repeats":2,"dwell":1,"horizon":24,
                  "per_bit_accuracy_by_arm":{arm:bit_acc[i].tolist() for i,arm in enumerate(ARMS)},
                  "mean_bit_error_by_arm":{arm:float(mean_err[i]) for i,arm in enumerate(ARMS)},
                  "whole_vector_exact_accuracy_by_arm":{arm:float(whole[i]) for i,arm in enumerate(ARMS)},
                  "per_bit_polarity_accuracy_by_arm":pol,
                  "paired_bit_error_difference_periodic_minus_control":(errors[:,:,0,:].astype(float)-errors[:,:,1,:]).mean((0,1)).tolist(),
                  "paired_error_difference_periodic_minus_control":delta,
                  "paired_error_difference_by_forcing_unit":force_delta.tolist(),
                  "per_forcing_unit_mean_bit_error_by_arm":errors.mean((1,3)).tolist(),
                  "oracle_exact_accuracy":float(np.all(oracle==(bits>0)[None],axis=-1).mean()),
                  "conditional_complete_state_geometry":{"conditional_spectra":geo,
                      "independent_forcing_units":4,"cue_vectors_per_forcing_unit_arm":256,
                      "theoretical_rank_ceiling":min(255,128,(1<<d)-1)}}
            design={"cue_bits":bits.tolist(),"forcing_unit_ids":[0,1,2,3],
                    "conditions":{"R2-q1":[pair.copy() for _ in range(4)]}}
            clean=Audit(directory);self.assertIsNotNone(clean.evaluation("D4-seed0",directory,item,d,design))
            self.assertFalse([x for x in clean.checks if x["status"]=="fail"])
            mislabeled=dict(item,condition="R32-q1")
            mislabeled_design={**design,"conditions":{"R32-q1":[pair.copy() for _ in range(4)]}}
            mislabeled_audit=Audit(directory)
            mislabeled_audit.evaluation("D4-seed0",directory,mislabeled,d,mislabeled_design)
            mislabeled_failures={x["check"] for x in mislabeled_audit.checks if x["status"]=="fail"}
            self.assertIn("R32-q1:design_alignment",mislabeled_failures)
            arrays["labels"]=labels.copy();arrays["labels"][0,0,0,0]=~arrays["labels"][0,0,0,0]
            np.savez(raw,**arrays)
            corrupted=Audit(directory);corrupted.evaluation("D4-seed0",directory,item,d,design)
            failures={x["check"] for x in corrupted.checks if x["status"]=="fail"}
            self.assertIn("R2-q1:labels_predictions_errors",failures)

    def test_failed_or_unverified_gate_cannot_precede_later_stage_or_holdout(self):
        self.assertFalse(Audit.status_consistent("stopped_partial",False,[],[False,True],True,True))
        self.assertFalse(Audit.status_consistent("stopped_partial",True,[{}],[True,False,True],True,True))
        self.assertFalse(Audit.status_consistent("competence_passed",True,[{}],[True,True,True],True,False))
        self.assertFalse(Audit.status_consistent("unknown",False,[],[],True,True))
        self.assertTrue(Audit.status_consistent("competence_passed",True,[{}],[True,True,True],True,True))

    def test_duplicate_load_seed_attempts_are_retained_as_lineage_group(self):
        duplicate=Audit.duplicate_load_seed_groups([
            {"dimension":4,"seed":0,"fit":"D4-seed0","path":"mila/D4","lineage":{"run_id":"old"}},
            {"dimension":4,"seed":0,"fit":"D4-seed0","path":"vulcan/D4","lineage":{"run_id":"repair"}},
        ])
        self.assertEqual(len(duplicate),1)
        self.assertEqual([a["run_id"] for a in duplicate[0]["attempts"]],["old","repair"])

    def test_partial_evaluation_must_be_ordered_unique_protocol_prefix(self):
        expected=[f"R{r}-q{q}" for r,q in CONDITIONS]
        self.assertTrue(Audit.evaluation_order_valid(expected[:4]))
        self.assertTrue(Audit.evaluation_order_valid(expected))
        self.assertFalse(Audit.evaluation_order_valid([expected[0],expected[2]]))
        self.assertFalse(Audit.evaluation_order_valid([expected[0],expected[0]]))


if __name__ == "__main__":
    unittest.main()
