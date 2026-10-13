"""Original-task query averaging: native gradients and guarded fixtures."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
FIXTURES=ROOT/"runs/query-acquisition-v5-fixtures"

class QueryAcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def config(self):return json.loads((ROOT/"configs/experiments/query-averaged-acquisition-v5.json").read_text())
    def temporary(self):
        FIXTURES.mkdir(parents=True,exist_ok=True)
        return tempfile.TemporaryDirectory(dir=FIXTURES)
    def base(self,task):
        from representation_collapse.diagnostics import paired_batch_with_zero
        return paired_batch_with_zero(np.random.default_rng(2026101301),16,0,task)

    def test_contract_rejects_task_source_units_budget_and_calibration_changes(self):
        from representation_collapse.query_acquisition import validate_config
        c=self.config();validate_config(c)
        for key,value in (("required_queries",list(range(16))),("seed",False),("learning_rate",.003),("device","cuda"),
                          ("validation_sampling_seed",20003),("updates_per_fit",5001),("maximum_fits",8),("population_variance",1),
                          ("maximum_total_optimizer_updates",20001),("pre_fit_calibration",{}),("v3_fresh_references",[]),("v3_base_training_streams",[]),("exposure",{})):
            bad=deepcopy(c);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_config(bad)

    def test_all_eight_targets_original_prefix_and_rng_draws_are_preserved(self):
        from representation_collapse.query_acquisition import derive_all_queries
        from representation_collapse.diagnostics import paired_batch_with_zero
        for task in ("categorical","continuous"):
            rng=np.random.default_rng(2026101301);base=paired_batch_with_zero(rng,16,0,task);original=deepcopy(base)
            arrays=derive_all_queries(base,task)
            np.testing.assert_array_equal(arrays["prefix"],base.inputs[:,:16])
            np.testing.assert_array_equal(arrays["targets"],np.concatenate((base.memory[:,:8],base.memory[:,:8])).T)
            np.testing.assert_array_equal(arrays["queries"][:,:,:5].argmax(2),4)
            np.testing.assert_array_equal(arrays["queries"][:,:,5:21].argmax(2),np.broadcast_to(np.arange(8)[:,None],(8,32)))
            self.assertFalse(arrays["queries"][:,:,21:].any());np.testing.assert_array_equal(base.inputs,original.inputs)
            following=paired_batch_with_zero(rng,16,0,task)
            other=np.random.default_rng(2026101301);paired_batch_with_zero(other,16,0,task);expected=paired_batch_with_zero(other,16,0,task)
            np.testing.assert_array_equal(following.inputs,expected.inputs)

    def test_full_original_oracle_rejects_payload_event_address_and_wrong_target(self):
        from representation_collapse.query_acquisition import derive_all_queries
        for task in ("categorical","continuous"):
            for mutation in ("payload","event","address","target"):
                b=self.base(task)
                if mutation=="payload":b.inputs[:,-1,37]=1
                elif mutation=="event":b.inputs[:,0,1]=1
                elif mutation=="address":b.inputs[:,0,6]=1
                else:b.targets[0]+=1
                with self.subTest(task=task,mutation=mutation),self.assertRaises(ValueError):derive_all_queries(b,task)

    def test_actual_native_fork_gradient_and_complete_state_match_fp32_fp64(self):
        from representation_collapse.query_acquisition import derive_all_queries,_tensors,average_query_loss
        from representation_collapse.pilot import VectorMemoryModel
        for model_kind in ("gru","lstm"):
            for task in ("categorical","continuous"):
                for dtype in (torch.float32,torch.float64):
                    torch.manual_seed(0);model=VectorMemoryModel(model_kind,task,128,64).to(dtype);reference=deepcopy(model)
                    prefix,queries,targets=_tensors(derive_all_queries(self.base(task),task),task,dtype)
                    mean,branch=average_query_loss(model,prefix,queries,targets);expected_logits=[];expected_states=[];losses=[]
                    for q in range(8):
                        captured=[];hook=reference.core.register_forward_hook(lambda _m,_i,o:captured.append(reference.pack_state(o[1])))
                        try:pred=reference(torch.cat((prefix,queries[q][:,None]),1))
                        finally:hook.remove()
                        expected_logits.append(pred);expected_states.append(captured[0]);losses.append(F.cross_entropy(pred,targets[q]) if task=="categorical" else F.mse_loss(pred[:,0],targets[q]))
                    native_loss=torch.stack(losses).mean();ga=torch.autograd.grad(mean,list(model.parameters()));gb=torch.autograd.grad(native_loss,list(reference.parameters()))
                    tol={"atol":1e-6,"rtol":1e-5} if dtype==torch.float32 else {"atol":1e-12,"rtol":1e-10}
                    torch.testing.assert_close(mean,native_loss,**tol);torch.testing.assert_close(branch["logits"],torch.stack(expected_logits),**tol)
                    torch.testing.assert_close(branch["states"],torch.stack(expected_states),**tol)
                    self.assertEqual(branch["states"].shape,(8,32,256 if model_kind=="lstm" else 128));self.assertTrue(branch["prefix_state"].requires_grad)
                    for a,b in zip(ga,gb):self.assertGreater(float(b.norm()),0);torch.testing.assert_close(a,b,**tol)

    def test_all_query_prehash_matches_exact_v3_stream_and_detects_mismatch(self):
        from representation_collapse.query_acquisition import prehash
        for pin in self.config()["v3_base_training_streams"]:
            base=json.loads((ROOT/pin["manifest_relative_to_project"]).read_text());s=prehash(pin["task"],2,base,time.monotonic()+30)
            self.assertEqual(s["base_batch_sha256"],base["batch_sha256"][:2]);self.assertEqual(s["base_prefix_sha256"],base["prefix_sha256"][:2])
            bad=deepcopy(base);bad["batch_sha256"][0]="0"*64
            with self.assertRaises(ValueError):prehash(pin["task"],2,bad,time.monotonic()+30)

    def test_production_requires_clean_committed_hash_bound_freeze_before_output(self):
        from representation_collapse.query_acquisition import run,verify_freeze
        with self.temporary() as temporary:
            folder=Path(temporary);out=folder/"forbidden";config=ROOT/"configs/experiments/query-averaged-acquisition-v5.json"
            with self.assertRaises(FileNotFoundError):run(config,out,folder/"absent.json")
            self.assertFalse(out.exists());review=folder/"review.json";review.write_text('{"accepted":true}')
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            r={"kind":"query_acquisition_v5_freeze","protocol_id":"query-averaged-acquisition-v5","local_execution_authorized":True,"scientific_acceptance":False,"source_commit":"a"*40,"config_relative_to_project":str(config.relative_to(ROOT)),"config_sha256":sha(config),"independent_review":{"accepted":True,"receipt_relative_to_project":str(review.relative_to(ROOT)),"receipt_sha256":sha(review),"implementation_sha256":sha(ROOT/"src/representation_collapse/query_acquisition.py"),"config_sha256":sha(config)}}
            receipt=folder/"freeze.json";receipt.write_text(json.dumps(r))
            with patch("subprocess.check_output",side_effect=["a"*40+"\n"," M source.py\n"]),self.assertRaises(ValueError):verify_freeze(config,receipt,ROOT)

    def test_fixture_requires_actual_update_fit_or_wall_reduction(self):
        from representation_collapse.query_acquisition import FixtureBudget
        for budget in (FixtureBudget(updates_per_fit=5001),FixtureBudget(fit_indices=(0,0)),FixtureBudget(wall_seconds=1201),FixtureBudget(updates_per_fit=5000,evaluate_every=500,fit_indices=tuple(range(4)),wall_seconds=1200)):
            with self.assertRaises(ValueError):budget.validate()

    def test_fixture_one_clip_and_step_raw_units_all_gates_paired_baseline(self):
        from representation_collapse import query_acquisition as module
        with self.temporary() as temporary:
            output=Path(temporary)/"run";step=torch.optim.AdamW.step;clip=torch.nn.utils.clip_grad_norm_;calls={"step":0,"clip":0}
            def counted_step(*a,**k):calls["step"]+=1;return step(*a,**k)
            def counted_clip(*a,**k):calls["clip"]+=1;return clip(*a,**k)
            with patch("torch.optim.AdamW.step",new=counted_step),patch("torch.nn.utils.clip_grad_norm_",new=counted_clip):
                s=module.run_fixture(self.config(),output,ROOT,module.FixtureBudget(updates_per_fit=1,fit_indices=(3,)))
            self.assertEqual(calls,{"step":1,"clip":1});self.assertEqual(s["total_optimizer_updates"],1);self.assertTrue(s["fixture_only"]);self.assertFalse(s["scientific_acceptance"])
            path=output/"lstm-continuous";m=json.loads((path/"metrics.json").read_text());h=json.loads((path/"history.json").read_text());self.assertIsNone(m["selected_checkpoint"])
            self.assertAlmostEqual(h[0]["train_mean8_raw_loss"],np.mean(h[0]["query_raw_losses"]),places=6)
            self.assertEqual(len(m["endpoint"]["arms"]["periodic"]["coordinates"]),8)
            self.assertTrue(m["endpoint"]["fixture_only"]);self.assertFalse(m["endpoint"]["validation_eligibility_met"]);self.assertEqual(m["endpoint"]["trained_seed_units"],0)
            with np.load(path/"validation-step-000001.npz") as a,np.load(output/"baseline/lstm-continuous.npz") as b:
                self.assertEqual(a["post_query"].shape,(2048,256));np.testing.assert_array_equal(a["inputs"],b["inputs"])
                np.testing.assert_array_equal(a["post_write"],a["pre_query"])
                for row in m["paired_baseline_coordinate_ledger"]:
                    mask=a["queries"][:1024]==row["coordinate"];y=a["targets"][:1024][mask].astype(np.float64)
                    expected=((a["predictions"][:1024][mask].astype(np.float64)-y)**2-(b["predictions"][:1024][mask].astype(np.float64)-y)**2)/(1/3)
                    np.testing.assert_allclose(row["error_difference_averaged_minus_pinned_sampled"],expected,rtol=0,atol=1e-14)
                    self.assertIsNone(row["confidence_interval"]);self.assertEqual(len(expected),128)
            with self.assertRaises(FileExistsError):module.run_fixture(self.config(),output,ROOT,module.FixtureBudget(updates_per_fit=1))

    def test_between_evaluation_deadline_preserves_actual_complete_current_theta(self):
        from representation_collapse import query_acquisition as module
        with self.temporary() as temporary:
            output=Path(temporary)/"run";clock={"t":0.};original=torch.optim.AdamW.step
            def step(*a,**k):
                result=original(*a,**k);clock["t"]=2.;return result
            with patch.object(module.time,"monotonic",side_effect=lambda:clock["t"]),patch("torch.optim.AdamW.step",new=step):
                s=module.run_fixture(self.config(),output,ROOT,module.FixtureBudget(updates_per_fit=2,evaluate_every=2,fit_indices=(2,),wall_seconds=1))
            self.assertEqual(s["total_optimizer_updates"],1);self.assertEqual(s["state"],"incomplete_censored_overall_deadline")
            fit=s["fits"][0];cp=torch.load(output/"lstm-categorical"/fit["censored_current_checkpoint_id"],weights_only=True)
            self.assertEqual(cp["step"],1);self.assertFalse(cp["usable_for_evaluation"]);self.assertFalse(cp["endpoint_competence_known"]);self.assertIsNone(fit["endpoint_coordinate_gate_passed"])

    def test_returned_nonfinite_update_count_and_artifact_remain_explicitly_unusable(self):
        from representation_collapse import query_acquisition as module
        with self.temporary() as temporary:
            output=Path(temporary)/"run";original=torch.optim.AdamW.step
            def step(optimizer,*a,**k):
                result=original(optimizer,*a,**k)
                with torch.no_grad():optimizer.param_groups[0]["params"][0].flatten()[0]=float("nan")
                return result
            with patch("torch.optim.AdamW.step",new=step),self.assertRaises(FloatingPointError):module.run_fixture(self.config(),output,ROOT,module.FixtureBudget(updates_per_fit=2,evaluate_every=2))
            s=json.loads((output/"status.json").read_text());self.assertEqual(s["total_optimizer_updates"],1);fit=s["fits"][0]
            cp=torch.load(output/"gru-categorical"/fit["censored_current_checkpoint_id"],weights_only=True);self.assertEqual(cp["step"],1);self.assertFalse(cp["parameters_finite"]);self.assertFalse(cp["usable_for_evaluation"])

    def test_first_passing_fixture_validation_never_shortens_fixed_endpoint(self):
        from representation_collapse import query_acquisition as module
        with self.temporary() as temporary:
            output=Path(temporary)/"run";evaluate=module.evaluate
            def passing(*a,**k):
                record,raw,chronology=evaluate(*a,**k);record["coordinate_gate_passed"]=True;return record,raw,chronology
            with patch.object(module,"evaluate",side_effect=passing):s=module.run_fixture(self.config(),output,ROOT,module.FixtureBudget(updates_per_fit=2))
            fit=s["fits"][0];self.assertEqual(fit["endpoint_step"],2);self.assertEqual(fit["first_passing_validation_step"],1);self.assertIsNone(fit["selected_checkpoint"])

if __name__=="__main__":unittest.main()
