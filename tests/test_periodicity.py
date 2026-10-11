import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,"Torch requires experiments dependency group")
class PeriodicityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)

    def test_native_complete_state_trajectory_and_query(self):
        from representation_collapse.pilot import VectorMemoryModel
        from representation_collapse.periodicity import replay
        from representation_collapse.vector_task import render
        memory=np.linspace(-1,1,16,dtype=np.float32)[None]
        inputs=render(memory,np.zeros(1,dtype=np.int64),["AXAXAXBXBXBX"*2],"continuous")
        for kind in ["gru","lstm"]:
            for dtype in [torch.float32,torch.float64]:
                model=VectorMemoryModel(kind,"continuous",4,3).to(dtype=dtype).eval()
                states,checks,prediction,native=replay(model,torch.as_tensor(inputs[0],dtype=dtype),return_native=True)
                self.assertTrue(checks['passed'])
                self.assertEqual(states.shape,(41,4 if kind=='gru' else 8))
                self.assertEqual(prediction.shape,(1,))
                self.assertTrue(np.array_equal(states[0],np.zeros(states.shape[1])))
                self.assertEqual(native['step_states'].shape,states.shape)
                self.assertEqual(native['full_pre_query_state'].shape,(states.shape[1],))
                self.assertEqual(native['full_hidden_trajectory'].shape,(40,4))
                np.testing.assert_allclose(states[-1],native['full_pre_query_state'],atol=1e-6,rtol=1e-5)

    def test_full_lstm_derivative_jvp_and_finite_difference(self):
        from representation_collapse.pilot import VectorMemoryModel
        from representation_collapse.periodicity import recurrent_step,window_map,tangent_window
        for kind in ['gru','lstm']:
            model=VectorMemoryModel(kind,'continuous',3,3).double()
            state=torch.randn(3 if kind=='gru' else 6,dtype=torch.float64)*0.1
            inputs=torch.randn(3,38,dtype=torch.float64)*0.1
            tangent=tangent_window(model,state,inputs)
            np.testing.assert_allclose(tangent['product'],tangent['direct_jacobian'],rtol=1e-12,atol=1e-12)
            v=torch.randn_like(state);v=v/torch.linalg.vector_norm(v)
            _,jvp=torch.autograd.functional.jvp(lambda z:window_map(model,z,inputs),state,v)
            fd=(window_map(model,state+1e-5*v,inputs)-window_map(model,state-1e-5*v,inputs))/(2e-5)
            torch.testing.assert_close(jvp,fd,rtol=1e-7,atol=1e-9)
            np.testing.assert_allclose(jvp.detach().numpy(),tangent['product']@v.numpy(),rtol=1e-10,atol=1e-12)
            if kind=='lstm':
                derivative=torch.autograd.functional.jacobian(lambda z:recurrent_step(model,z,inputs[0]),state)
                self.assertGreater(float(torch.linalg.vector_norm(derivative[:3,3:])),0)

    def test_noncommuting_chronological_product(self):
        from representation_collapse.pilot import VectorMemoryModel
        from representation_collapse.periodicity import tangent_window
        model=VectorMemoryModel('gru','continuous',3,3).double()
        state=torch.tensor([.1,-.2,.3],dtype=torch.float64)
        inputs=torch.randn(2,38,dtype=torch.float64)
        tangent=tangent_window(model,state,inputs)
        j=tangent['step_jacobians']
        np.testing.assert_allclose(tangent['direct_jacobian'],j[1]@j[0],atol=1e-12)
        self.assertGreater(np.linalg.norm(j[0]@j[1]-j[1]@j[0]),1e-4)

    def test_driver_matching_and_closure_reject_broadcast(self):
        from representation_collapse.periodicity import fixed_panels,distance
        from representation_collapse.vector_task import body_from_ids,proper_period
        panels=fixed_panels(8,2)
        self.assertNotEqual(panels[0][0],panels[1][0])
        for final,control in panels:
            for r in [2,16,32]:
                ids=np.r_[control[:r-1],final]
                self.assertIsNone(proper_period(body_from_ids(ids)))
                self.assertEqual(ids[-1],final)
        with self.assertRaises(ValueError):distance(np.zeros((12,4)),np.zeros((1,4)),1e-7,1e-5)

    def test_checkpoint_only_artifacts_and_fail_closed_registry(self):
        from representation_collapse.pilot import VectorMemoryModel,atomic_json,_manifest
        from representation_collapse.periodicity import run
        root=Path(__file__).resolve().parents[1]
        config=json.loads((root/'configs/experiments/periodicity-analysis-v1.json').read_text())
        config.update(driver_panels=1,cue_histories=2,derivative_cues=1,hidden_size=4,input_bottleneck=3)
        config.pop('cells')
        with tempfile.TemporaryDirectory() as temporary:
            parent=Path(temporary)/'original';parent.mkdir()
            model=VectorMemoryModel('lstm','continuous',4,3)
            saved={'model':'lstm','task':'continuous','seed':0,'hidden_size':4,'input_bottleneck':3,'effective_training_steps':1500,'max_steps_override':False}
            checkpoint=parent/'checkpoint.pt'
            torch.save({'model_state_dict':model.state_dict(),'config':saved,'selected_step':100},checkpoint)
            atomic_json(parent/'config.json',saved)
            atomic_json(parent/'status.json',{'state':'completed'})
            atomic_json(parent/'provenance.json',{'git_revision':config['training_source_commit'],'dirty_checkout':False})
            _manifest(parent)
            digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            output=Path(temporary)/'diagnostic'
            metrics=run(config,checkpoint,output,'lstm','continuous',0,'test-fixture',verify_source=False,checkpoint_sha256=digest)
            self.assertTrue(metrics['numerical_checks_passed'])
            self.assertEqual(metrics['dense_windows'],8)
            self.assertEqual(metrics['step_jacobian_evaluations'],192)
            self.assertFalse(metrics['scientific_acceptance'])
            with np.load(output/'inputs.npz') as raw:
                np.testing.assert_array_equal(raw['panel0_memory'][0,8:],raw['panel0_memory'][1,8:])
                for precision in ['float32','float64']:
                    np.testing.assert_array_equal(raw[precision+'_panel0_R1_periodic_inputs'],raw[precision+'_panel0_R1_control_inputs'])
            with np.load(output/'trajectories.npz') as raw:
                self.assertEqual(raw['float32_panel0_R1_periodic_cue0_cycle_state_closure_absolute'].shape,(0,))
                self.assertEqual(raw['float64_panel0_R16_periodic_cue0_cycle_state_closure_absolute'].shape,(15,))
                self.assertEqual(raw['float64_panel0_R32_control_cue0_states'].shape,(401,8))
                self.assertEqual(raw['float64_panel0_R32_control_cue0_native_full_pre_query_state'].shape,(8,))
                self.assertEqual(raw['float64_panel0_R32_control_cue0_native_full_hidden_trajectory'].shape,(400,4))
            for name,digest_ in json.loads((output/'manifest.json').read_text())['sha256'].items():
                self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(),digest_)
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),digest)
            with self.assertRaises(FileExistsError):run(config,checkpoint,output,'lstm','continuous',0,'fixture',verify_source=False,checkpoint_sha256=digest)
            failed=Path(temporary)/'bad'
            with self.assertRaises(ValueError):run(config,checkpoint,failed,'lstm','continuous',0,'fixture',verify_source=False,checkpoint_sha256='0'*64)
            self.assertEqual(json.loads((failed/'status.json').read_text())['state'],'failed')
            strict=json.loads(json.dumps(config))
            strict['tolerances']['float32']['adapter_absolute']=1e-30
            strict['tolerances']['float32']['adapter_relative']=1e-30
            quarantined=Path(temporary)/'quarantined'
            with self.assertRaisesRegex(ValueError,'raw diagnostic quarantined'):
                run(strict,checkpoint,quarantined,'lstm','continuous',0,'fixture',verify_source=False,checkpoint_sha256=digest)
            self.assertFalse(json.loads((quarantined/'metrics.json').read_text())['numerical_checks_passed'])
            self.assertEqual(json.loads((quarantined/'status.json').read_text())['state'],'failed')
            self.assertTrue((quarantined/'trajectories.npz').exists())
            self.assertTrue((quarantined/'tangents.npz').exists())


if __name__=='__main__':unittest.main()
