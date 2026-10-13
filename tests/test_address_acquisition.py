"""v4 domain/matching/freeze controls; explicit infrastructure fixtures only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
try:
    import torch
except ImportError:
    torch=None
ROOT=Path(__file__).resolve().parents[1]
FIXTURES=ROOT/"runs/address-acquisition-v4-fixtures"

@unittest.skipIf(torch is None,"Torch experiment dependencies required")
class AddressAcquisitionTests(unittest.TestCase):
    def config(self):return json.loads((ROOT/"configs/experiments/address-acquisition-discovery-v4.json").read_text())
    def temporary(self):
        FIXTURES.mkdir(parents=True,exist_ok=True)
        return tempfile.TemporaryDirectory(dir=FIXTURES)

    def test_production_contract_cannot_change_budget_data_units_or_query_support(self):
        from representation_collapse.address_acquisition import validate_config
        c=self.config();validate_config(c)
        for key,value in (("conditions",[0,7,8]),("maximum_fits",13),("seed",False),("maximum_total_optimizer_updates",60001),
                          ("validation_sampling_seed",20001),("training_stream_seed",1),("baseline_comparison_addresses",[0,7,15]),("device","cuda")):
            bad=deepcopy(c);bad[key]=value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):validate_config(bad)

    def test_derivation_changes_only_terminal_selector_target_and_metadata(self):
        from representation_collapse.address_acquisition import derive_batch,target_oracle
        from representation_collapse.diagnostics import paired_batch_with_zero
        for task in ("categorical","continuous"):
            rng=np.random.default_rng(2026101301);base=paired_batch_with_zero(rng,16,0,task);original=base.inputs.copy()
            for address in (0,7,15):
                new=derive_batch(base,address,task)
                np.testing.assert_array_equal(new.inputs[:,:16],base.inputs[:,:16])
                np.testing.assert_array_equal(new.inputs[:,-1,:5],base.inputs[:,-1,:5])
                np.testing.assert_array_equal(new.inputs[:,-1,21:],base.inputs[:,-1,21:])
                self.assertTrue(np.all(new.queries==address));self.assertEqual(target_oracle(new.inputs,new.targets,new.queries,task),32)
            np.testing.assert_array_equal(base.inputs,original)
            following=paired_batch_with_zero(rng,16,0,task)
            oracle_rng=np.random.default_rng(2026101301);paired_batch_with_zero(oracle_rng,16,0,task)
            oracle_next=paired_batch_with_zero(oracle_rng,16,0,task)
            np.testing.assert_array_equal(following.inputs,oracle_next.inputs)

    def test_domain15_native_lstm_adapter_does_not_relax_original_guard(self):
        from representation_collapse.address_acquisition import derive_batch,native_capture
        from representation_collapse.diagnostics import paired_batch_with_zero
        from representation_collapse.native_state import native_phase_capture
        from representation_collapse.pilot import VectorMemoryModel
        model=VectorMemoryModel("lstm","categorical",128,64)
        batch=derive_batch(paired_batch_with_zero(np.random.default_rng(1),16,0,"categorical"),15,"categorical")
        x=torch.from_numpy(batch.inputs.copy());got=native_capture(model,x)
        self.assertTrue(got["chronology"]["passed"]);self.assertEqual(got["chronology"]["state_dimension"],256)
        self.assertEqual(got["chronology"]["packing_order"],"hidden_then_cell")
        self.assertEqual(got["states"]["post_query"].shape,(32,256))
        with self.assertRaises(ValueError):native_phase_capture(model,x,16)

    def test_oracle_rejects_query_payload_wrong_label_and_wrong_address_chronology(self):
        from representation_collapse.address_acquisition import derive_batch,target_oracle
        from representation_collapse.diagnostics import paired_batch_with_zero
        b=derive_batch(paired_batch_with_zero(np.random.default_rng(1),16,0,"continuous"),15,"continuous")
        wrong=b.targets.copy();wrong[0]+=1
        with self.assertRaises(ValueError):target_oracle(b.inputs,wrong,b.queries,"continuous")
        x=b.inputs.copy();x[:,-1,37]=b.targets
        with self.assertRaises(ValueError):target_oracle(x,b.targets,b.queries,"continuous")
        x=b.inputs.copy();x[:,0,5:21]=np.roll(x[:,0,5:21],1,axis=1)
        with self.assertRaises(ValueError):target_oracle(x,b.targets,b.queries,"continuous")

    def test_prefit_base_hashes_equal_pinned_v3_and_detect_original_rng_mismatch(self):
        from representation_collapse.address_acquisition import prehash_streams
        pin=next(p for p in self.config()["v3_base_training_streams"] if p["task"]=="categorical")
        base=json.loads((ROOT/pin["manifest_relative_to_project"]).read_text())
        streams=prehash_streams("categorical",2,(0,7,15),base,time.monotonic()+60)
        for q,s in streams.items():
            self.assertEqual(s["base_batch_sha256"],base["batch_sha256"][:2])
            self.assertEqual(s["base_prefix_sha256"],base["prefix_sha256"][:2])
        self.assertEqual(len({s["stream_sha256"] for s in streams.values()}),3)
        base["batch_sha256"][0]="0"*64
        with self.assertRaises(ValueError):prehash_streams("categorical",2,(15,),base,time.monotonic()+60)

    def test_no_production_without_clean_hash_bound_freeze(self):
        from representation_collapse.address_acquisition import run,verify_freeze
        with self.temporary() as temporary:
            folder=Path(temporary);out=folder/"forbidden"
            with self.assertRaises(FileNotFoundError):run(ROOT/"configs/experiments/address-acquisition-discovery-v4.json",out,folder/"missing.json")
            self.assertFalse(out.exists())
            review=folder/"review.json";review.write_text('{"accepted":true}')
            config=ROOT/"configs/experiments/address-acquisition-discovery-v4.json";module=ROOT/"src/representation_collapse/address_acquisition.py"
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            receipt={"kind":"address_acquisition_v4_freeze","protocol_id":"address-acquisition-discovery-v4","local_execution_authorized":True,"scientific_acceptance":False,"source_commit":"a"*40,"config_relative_to_project":str(config.relative_to(ROOT)),"config_sha256":sha(config),"independent_review":{"accepted":True,"receipt_relative_to_project":str(review.relative_to(ROOT)),"receipt_sha256":sha(review),"implementation_sha256":sha(module),"config_sha256":sha(config)}}
            freeze=folder/"freeze.json";freeze.write_text(json.dumps(receipt))
            with patch("subprocess.check_output",side_effect=["a"*40+"\n"," M source.py\n"]):
                with self.assertRaises(ValueError):verify_freeze(config,freeze,ROOT)

    def test_fixture_envelope_strictly_reduces_and_never_changes_architecture(self):
        from representation_collapse.address_acquisition import FixtureBudget
        for b in (FixtureBudget(updates_per_fit=5001),FixtureBudget(fit_indices=(0,0)),FixtureBudget(wall_seconds=1201),
                  FixtureBudget(updates_per_fit=5000,evaluate_every=500,fit_indices=tuple(range(12)),wall_seconds=1200)):
            with self.assertRaises(ValueError):b.validate()

    def test_actual_fixture_native_query15_and_paired_supported_baseline(self):
        from representation_collapse.address_acquisition import FixtureBudget,run_fixture
        with self.temporary() as temporary:
            output=Path(temporary)/"run"
            status=run_fixture(self.config(),output,ROOT,FixtureBudget(updates_per_fit=1,fit_indices=(6,8)))
            self.assertEqual(status["total_optimizer_updates"],2);self.assertTrue(status["fixture_only"]);self.assertFalse(status["scientific_acceptance"])
            for q in (0,15):
                path=output/f"lstm-categorical-q{q}";metrics=json.loads((path/"metrics.json").read_text())
                self.assertIsNone(metrics["selected_checkpoint"])
                with np.load(path/"validation-step-000001.npz") as raw:
                    self.assertEqual(raw["post_query"].shape,(256,256));self.assertTrue(np.all(raw["queries"]==q))
                    np.testing.assert_array_equal(raw["post_write"],raw["pre_query"])
                if q==15:
                    self.assertIsNone(metrics["paired_baseline_contrast"]);self.assertFalse((output/"baseline/lstm-categorical-q15.npz").exists())
                else:
                    contrast=metrics["paired_baseline_contrast"];self.assertEqual(len(contrast["paired_error_difference"]),128)
                    self.assertIsNone(contrast["p_value"]);self.assertIsNone(contrast["confidence_interval"])
                    with np.load(path/"validation-step-000001.npz") as raw,np.load(output/"baseline/lstm-categorical-q0.npz") as baseline:
                        np.testing.assert_array_equal(raw["inputs"],baseline["inputs"]);np.testing.assert_array_equal(raw["targets"],baseline["targets"])
            with self.assertRaises(FileExistsError):run_fixture(self.config(),output,ROOT,FixtureBudget(updates_per_fit=1,fit_indices=(8,)))

    def test_censor_between_evaluations_preserves_realized_step_and_theta(self):
        from representation_collapse import address_acquisition as module
        with self.temporary() as temporary:
            output=Path(temporary)/"run";clock={"t":0.};original=torch.optim.AdamW.step
            def step(optimizer,*args,**kwargs):
                result=original(optimizer,*args,**kwargs);clock["t"]=2.;return result
            with patch.object(module.time,"monotonic",side_effect=lambda:clock["t"]),patch("torch.optim.AdamW.step",new=step):
                status=module.run_fixture(self.config(),output,ROOT,module.FixtureBudget(updates_per_fit=2,evaluate_every=2,fit_indices=(2,),wall_seconds=1))
            self.assertEqual(status["state"],"incomplete_censored_overall_deadline");self.assertEqual(status["total_optimizer_updates"],1)
            fit=status["fits"][0];path=output/"gru-categorical-q15"
            checkpoint=torch.load(path/fit["censored_current_checkpoint_id"],map_location="cpu",weights_only=True)
            self.assertEqual(checkpoint["step"],1);self.assertFalse(checkpoint["usable_for_evaluation"]);self.assertFalse(checkpoint["endpoint_competence_known"])
            self.assertIsNone(fit["endpoint_coordinate_gate_passed"])

    def test_nonfinite_returned_update_is_counted_and_preserved_unusable(self):
        from representation_collapse import address_acquisition as module
        with self.temporary() as temporary:
            output=Path(temporary)/"run";original=torch.optim.AdamW.step
            def step(optimizer,*args,**kwargs):
                result=original(optimizer,*args,**kwargs)
                with torch.no_grad():optimizer.param_groups[0]["params"][0].flatten()[0]=float("nan")
                return result
            with patch("torch.optim.AdamW.step",new=step):
                with self.assertRaises(FloatingPointError):module.run_fixture(self.config(),output,ROOT,module.FixtureBudget(updates_per_fit=2,evaluate_every=2,fit_indices=(2,)))
            status=json.loads((output/"status.json").read_text());self.assertEqual(status["total_optimizer_updates"],1)
            fit=status["fits"][0];checkpoint=torch.load(output/"gru-categorical-q15"/fit["censored_current_checkpoint_id"],map_location="cpu",weights_only=True)
            self.assertEqual(checkpoint["step"],1);self.assertFalse(checkpoint["parameters_finite"]);self.assertFalse(checkpoint["usable_for_evaluation"])

    def test_continuous_paired_baseline_contrast_uses_population_units_and_current_endpoint(self):
        from representation_collapse.address_acquisition import FixtureBudget,run_fixture
        with self.temporary() as temporary:
            output=Path(temporary)/"run"
            status=run_fixture(self.config(),output,ROOT,FixtureBudget(updates_per_fit=1,fit_indices=(3,)))
            self.assertEqual(status["total_optimizer_updates"],1)
            path=output/"gru-continuous-q0";m=json.loads((path/"metrics.json").read_text())
            self.assertEqual(m["endpoint_step"],1);self.assertIsNone(m["selected_checkpoint"])
            with np.load(path/"validation-step-000001.npz") as raw,np.load(output/"baseline/gru-continuous-q0.npz") as baseline:
                y=raw["targets"][:128].astype(np.float64);a=raw["predictions"][:128].astype(np.float64);b=baseline["predictions"][:128].astype(np.float64)
                expected=((a-y)**2-(b-y)**2)/(1/3)
                np.testing.assert_allclose(m["paired_baseline_contrast"]["paired_error_difference"],expected,rtol=0,atol=1e-14)
                self.assertAlmostEqual(m["paired_baseline_contrast"]["mean_paired_error_difference"],float(expected.mean()),places=14)
                self.assertEqual(raw["post_query"].shape,(256,128))
            p=json.loads((path/"provenance.json").read_text());self.assertEqual(p["initial_optimizer_state_entries"],0)

    def test_mocked_first_success_still_runs_to_fixed_fixture_endpoint(self):
        from representation_collapse import address_acquisition as module
        with self.temporary() as temporary:
            output=Path(temporary)/"run";original=module.evaluate
            def passing(*args,**kwargs):
                score,raw,report=original(*args,**kwargs);score["coordinate_gate_passed"]=True
                return score,raw,report
            with patch.object(module,"evaluate",side_effect=passing):
                status=module.run_fixture(self.config(),output,ROOT,module.FixtureBudget(updates_per_fit=2,fit_indices=(2,)))
            fit=status["fits"][0];self.assertEqual(fit["updates_completed"],2);self.assertEqual(fit["endpoint_step"],2)
            self.assertEqual(fit["first_passing_validation_step"],1);self.assertIsNone(fit["selected_checkpoint"])

if __name__=="__main__":unittest.main()
