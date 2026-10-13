"""Meaningful label-free tiny native fixtures; <=500 calls in this campaign."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from representation_collapse import chronology_calibration as m

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"runs/native-chronology-calibration-v1-fixtures"


class ChronologyCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUTPUT.mkdir(exist_ok=True)
        cls.campaign_calls=0
        cls.started=time.monotonic()
        cls.before=torch.nn.modules.rnn.RNNBase.register_forward_pre_hook
        cls.original_gru=torch.nn.GRU.forward
        cls.original_lstm=torch.nn.LSTM.forward
        def counted(original):
            def forward(instance,*args,**kwargs):
                cls.campaign_calls+=1
                if cls.campaign_calls>500:raise AssertionError("test campaign exceeds500 native calls")
                if time.monotonic()-cls.started>120:raise AssertionError("test campaign exceeds120 seconds")
                return original(instance,*args,**kwargs)
            return forward
        cls.gru_patch=patch.object(torch.nn.GRU,"forward",new=counted(cls.original_gru))
        cls.lstm_patch=patch.object(torch.nn.LSTM,"forward",new=counted(cls.original_lstm))
        cls.gru_patch.start();cls.lstm_patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.gru_patch.stop();cls.lstm_patch.stop()
        # Count independently at native forward, not using the implementation counter.
        p=OUTPUT/("focused-test-proof-"+str(time.time_ns())+".json")
        proof={"fixture_only":True,"native_forward_calls":cls.campaign_calls,
               "maximum_campaign_calls":500,"trained_checkpoint_loads":0,"optimizer_updates":0,
               "targets_read":False,"scientific_acceptance":False,"elapsed_seconds":time.monotonic()-cls.started}
        p.write_text(json.dumps(proof,indent=2)+"\n")
        assert cls.campaign_calls<=500
        assert time.monotonic()-cls.started<=120

    def folder(self,name):
        # Unique per test invocation; preserve previous fixture artifacts.
        return Path(tempfile.mkdtemp(prefix=name+"-",dir=OUTPUT))/"case"

    def test_exact_registry_and_config_no_semantic_overrides(self):
        self.assertEqual(m.registry(),{"primary":116288,"known_regression":2944,"N1":368,"N2":320,"N3":80})
        planned=m.case_registry()
        self.assertEqual(len(planned),372);self.assertEqual(sum(c["core_calls"] for c in planned),120000)
        m.validate_config(json.loads((ROOT/"configs/experiments/native-chronology-calibration-v1.json").read_text()))
        for key,value in (("maximum_core_calls",120001),("canonical_only_admitted",True),("targets_read",True)):
            config=m.configuration_contract();config[key]=value
            with self.assertRaises(ValueError):m.validate_config(config)
        for spec in (m.FixtureSpec(histories=3),m.FixtureSpec(hidden=128),m.FixtureSpec(maximum_calls=501),
                     m.FixtureSpec(wall_seconds=float("nan")),m.FixtureSpec(histories=1,include_nulls=True)):
            with self.assertRaises(ValueError):spec.validate()

    def test_ordered_ulp_zero_crossing_adjacency_extremes_and_nonfinite(self):
        for dtype in (np.float32,np.float64):
            tiny=np.nextafter(dtype(0),dtype(1),dtype=dtype)
            left=np.array([-tiny,-tiny,-0.,0.,1.,-1.],dtype=dtype)
            right=np.array([0.,tiny,+0.,tiny,np.nextafter(dtype(1),dtype(2),dtype=dtype),np.nextafter(dtype(-1),dtype(-2),dtype=dtype)],dtype=dtype)
            np.testing.assert_array_equal(m.ordered_ulp_distances(left,right),[1,2,0,1,1,1])
            limits=np.array([-np.finfo(dtype).max],dtype=dtype)
            distance=m.ordered_ulp_distances(limits,-limits)
            self.assertGreater(int(distance[0]),0)
            self.assertEqual(int(distance[0]),int(m.ordered_ulp_distances(-limits,limits)[0]))
        with self.assertRaises(ValueError):m.ordered_ulp_distances(np.array([np.inf]),np.array([0.]))
        self.assertFalse(m.compare(np.array([-0.],dtype=np.float32),np.array([+0.],dtype=np.float32),"float32",bitwise=True)["passed"])

    def test_asymmetric_directed_bound_true_violations_not_bitwise_differences(self):
        # One-entry exact-bound example has a known reference-dependent allowance.
        a=np.array([1.],dtype=np.float64);b=np.array([1.0000110001],dtype=np.float64)
        self.assertTrue(m.compare(a,b,"float32")["passed"])
        self.assertFalse(m.compare(b,a,"float32")["passed"])
        result=m.compare(np.array([0.,1.,2.],dtype=np.float32),np.array([2e-6,1+1e-7,2.],dtype=np.float32),"float32")
        self.assertEqual(result["violation_count"],1);self.assertEqual(result["violation_indices"],[[0]])
        with self.assertRaises(FloatingPointError):m.compare(np.array([np.nan]),np.array([0.]),"float64")
        maximum=np.finfo(np.float64).max
        with self.assertRaises(FloatingPointError):m.compare(np.array([-maximum]),np.array([maximum]),"float64")
        with self.assertRaises(FloatingPointError):m.compare(np.array([]),np.array([]),"float64")
        overflow=m.compare(np.array([1.]),np.array([np.nextafter(0.,1.)]),"float64")
        self.assertIsNone(overflow["relative_max_on_nonzero"]);self.assertEqual(overflow["relative_ratio_overflow_count"],1)
        self.assertEqual(m.compare(np.array([1.,2.]),np.array([0.,0.]),"float64")["signed_difference_min"],1.)
        self.assertEqual(m.compare(np.array([-1.,-2.]),np.array([0.,0.]),"float64")["signed_difference_max"],-1.)

    def test_freeze_blocks_known_load_and_output_before_any_native_call(self):
        config=ROOT/"configs/experiments/native-chronology-calibration-v1.json"
        out=self.folder("admission")
        with patch.object(m,"_known",side_effect=AssertionError("trained data accessed")),patch.object(torch,"load",side_effect=AssertionError("trained checkpoint accessed")):
            with self.assertRaises(FileNotFoundError):m.run(config,out,out.parent/"absent-freeze.json")
        self.assertFalse(out.exists())
        review=out.parent/"review.json"
        review_data={"implementation_preparation_accepted":True,"scientific_acceptance":False,
            "implementation_sha256":m.sha(m.__file__),"config_sha256":m.sha(config)}
        review.write_text(json.dumps(review_data))
        receipt={"kind":"native_chronology_calibration_v1_freeze","protocol_id":m.PROTOCOL,
            "calibration_execution_authorized":True,"scientific_acceptance":False,"training_authorized":False,
            "canonical_only_admitted":False,"source_commit":"f"*40,"config_relative_to_project":str(config.relative_to(ROOT)),
            "config_sha256":m.sha(config),"independent_review":{"accepted":True,
            "receipt_relative_to_project":str(review.relative_to(ROOT)),"receipt_sha256":m.sha(review),**review_data}}
        freeze=out.parent/"fixture-freeze.json";freeze.write_text(json.dumps(receipt))
        with patch.object(m.subprocess,"check_output",side_effect=["f"*40+"\n"," M source.py\n"]):
            with self.assertRaisesRegex(ValueError,"clean immutable"):m.verify_freeze(config,freeze,ROOT)

    def test_production_clock_includes_config_and_source_admission(self):
        config=ROOT/"configs/experiments/native-chronology-calibration-v1.json"
        with patch.object(m.time,"monotonic",return_value=12.25),patch.object(m,"verify_freeze",return_value={"mocked":True}),patch.object(m,"_engine",return_value={"mocked":True}) as engine:
            self.assertEqual(m.run(config,self.folder("clock-propagation"),ROOT/"unavailable-freeze.json"),{"mocked":True})
        self.assertEqual(engine.call_args.kwargs["_began"],12.25)

    def test_gru_tiny_all_paths_fixed_nulls_exact_count_raw_outputs(self):
        out=self.folder("gru-null")
        with patch.object(m,"_known",side_effect=AssertionError("trained reference read")),patch.object(torch,"load",side_effect=AssertionError("trained checkpoint read")):
            status=m.run_fixture(out,m.FixtureSpec(include_nulls=True))
        self.assertEqual(status["native_core_calls_attempted"],284)
        self.assertEqual(status["native_core_calls_completed"],284)
        self.assertTrue(status["matrix_completed"])
        self.assertTrue(status["strict_interoperability_passed"])
        self.assertEqual(status["stage_actual_core_calls"],{"primary":92,"N1":92,"N2":80,"N3":20})
        results=json.loads((out/"results.json").read_text())
        first=results["records"][0]
        with np.load(out/first["arrays"],allow_pickle=False) as raw:
            # Independent native chronology and identity properties from saved arrays.
            np.testing.assert_allclose(raw["P3__prefix_hidden"],raw["P1__prefix_hidden"],rtol=0,atol=0)
            np.testing.assert_allclose(raw["P0__terminal_hidden"],raw["P4__sequence_complete"][:,-1],atol=1e-12,rtol=1e-10)
            self.assertEqual(raw["P0__logits"].shape,(2,16))
        self.assertFalse(any("target" in key for key in raw.files))
        manifest=json.loads((out/"manifest.json").read_text())
        for path,digest in manifest["files"].items():self.assertEqual(hashlib.sha256((out/path).read_bytes()).hexdigest(),digest)
        with self.assertRaises(FileExistsError):m.run_fixture(out)

    def test_lstm_observable_cells_and_missing_or_invented_cell_rejected(self):
        out=self.folder("lstm")
        status=m.run_fixture(out,m.FixtureSpec(cell=("lstm","continuous")))
        self.assertEqual(status["native_core_calls_attempted"],92)
        self.assertTrue(status["matrix_completed"])
        record=json.loads((out/"results.json").read_text())["records"][0]
        self.assertIn("P0/prefix_cell",record["unavailable"])
        arrays=m._read_arrays(out/record["arrays"])
        self.assertNotIn("prefix_cell",arrays["P0"])
        self.assertIn("terminal_cell",arrays["P0"])
        hidden=arrays["P4"]["terminal_hidden"];cell=arrays["P4"]["terminal_cell"]
        np.testing.assert_array_equal(arrays["P4"]["sequence_complete"][:,-1],np.concatenate((hidden,cell),axis=1))
        broken=deepcopy(arrays);del broken["P0"]["terminal_cell"]
        with self.assertRaises(ValueError):m.directed_checks(broken,"float64","lstm","continuous")
        broken=deepcopy(arrays);broken["P0"]["prefix_cell"]=arrays["P1"]["prefix_cell"]
        with self.assertRaises(ValueError):m.directed_checks(broken,"float64","lstm","continuous")

    def test_attempted_ceiling_and_late_finalization_are_censored(self):
        out=self.folder("call-ceiling")
        status=m.run_fixture(out,m.FixtureSpec(maximum_calls=1))
        self.assertEqual(status["native_core_calls_attempted"],1)
        self.assertEqual(status["native_core_calls_completed"],1)
        self.assertFalse(status["matrix_completed"]);self.assertFalse(status["strict_interoperability_passed"])
        self.assertEqual(status["state"],"incomplete_censored_budget")
        self.assertEqual(status["expected_core_calls"],92)
        self.assertEqual(status["stage_actual_core_calls"],{"primary":1})
        self.assertEqual(len(status["unstarted_conditions"]),3)
        self.assertIsNotNone(status["interrupted_condition_id"])
        # Deadline during finalization cannot promote an incomplete matrix.
        out=self.folder("wall")
        with patch.object(m.time,"monotonic",side_effect=[0.]+[200.]*20):
            status=m.run_fixture(out)
        self.assertFalse(status["matrix_completed"])
        self.assertTrue(status["finalization_exceeded_deadline"])
        self.assertEqual(status["native_core_calls_attempted"],0)


if __name__=="__main__":unittest.main()
