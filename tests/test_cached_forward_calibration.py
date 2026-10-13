"""Pure/fake guard tests; opt-in ONE native campaign <=200 calls /120 seconds.

Native campaign: PYTHONPATH=src .venv/bin/python tests/test_cached_forward_calibration.py
--native-campaign runs/review/cached-forward-v2-implementation/native-fixtures
Ordinary unittest discovery executes no native recurrent forwards.
"""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from representation_collapse import cached_forward_calibration as c


class PureContractTests(unittest.TestCase):
    def test_exact_production_plan_without_input_generation(self):
        config=json.loads(Path('configs/experiments/cached-state-forward-calibration-v2.json').read_text())
        registry=c.condition_registry(config)
        self.assertEqual(len(registry),32)
        self.assertEqual(sum(len(x['operations']) for x in registry),736)
        self.assertEqual(sum(x['operations'].count('B0') for x in registry),32)
        self.assertEqual(sum(x['operations'].count('cell_minus') for x in registry),16)

    def test_margin_flip_passes_forward_bound_but_no_decision_certificate(self):
        v=np.array([[0.,5e-7]],dtype=np.float32);u=v[:,::-1].copy()
        self.assertTrue(c.compare(u,v,'float32')['passed'])
        result=c.behavior_bounds(u,v,'categorical','float32')
        self.assertEqual(result['exact_prediction_agreement'],[False])
        self.assertEqual(result['margin_certified'],[False])
        self.assertFalse(c.behavior_bounds(np.array([[1.,1.]]),np.array([[1.,1.]]),'categorical','float64')['margin_certified'][0])

    def test_continuous_label_free_bound_covers_endpoint_errors(self):
        u=np.array([[2.],[-.8]]);v=np.array([[1.5],[-.9]])
        result=c.behavior_bounds(u,v,'continuous','float64')
        bound=np.array(result['normalized_error_change_bound_per_row'])
        for y in [-1.,1.]:self.assertTrue(np.all(np.abs(3*((u[:,0]-y)**2-(v[:,0]-y)**2))<=bound+1e-14))
        self.assertIsNone(result['actual_MSE'])

    def test_bitwise_signed_zero_and_complete_state(self):
        self.assertFalse(c.compare(np.array([-0.]),np.array([0.]),'float64',bitwise=True)['passed'])
        import torch
        with self.assertRaises(ValueError):c.state_components(torch.zeros(1,4,3),'lstm')
        with self.assertRaises(ValueError):c.state_components((torch.zeros(1,4,3),),'lstm')

    def test_zero_effect_is_unresolved_and_zero_delta_gain_undefined(self):
        result=c.perturbation_effect(np.zeros((2,1)),np.zeros((2,1)),np.array([[1.,0.],[0.,0.]]))
        self.assertEqual(result['status_per_row'],['unresolved_at_this_perturbation','represented_zero'])
        self.assertEqual(result['gain_per_row'],[0.,None])

    def test_subnormal_norm_and_gain_keep_explicit_undefined_reasons(self):
        result=c.perturbation_effect(np.array([[1e-300],[0.],[1e-300]]),np.zeros((3,1)),
            np.array([[1e-300,0.],[1e-300,0.],[1e300,0.]]))
        self.assertEqual(result['status_per_row'],['resolved_C_effect','unresolved_at_this_perturbation','gain_underflow'])
        self.assertEqual(result['gain_per_row'],[1.,0.,None])
        self.assertEqual(result['undefined_gain_count'],1)

    def test_deadline_cap_and_bookkeeping_crossing(self):
        budget=c.CallBudget(1,0,600,10)
        with patch.object(c.time,'monotonic',return_value=590):
            with self.assertRaises(c.StopCalibration):budget.check(native=True)
        budget.attempted=1
        with patch.object(c.time,'monotonic',return_value=1):
            with self.assertRaises(c.StopCalibration):budget.check(native=True)
        import torch
        budget=c.CallBudget(1,0,600,10)
        # Simulate a slow durable journal flush that crosses the native-start cutoff.
        clock=[589.]
        with patch.object(c.time,'monotonic',side_effect=lambda:clock[0]),patch.object(budget,'persist',side_effect=lambda:clock.__setitem__(0,590.)):
            with self.assertRaises(c.StopCalibration):budget.begin({'context':'fixture','dtype':'float32','repeat':1},'B0',torch.zeros(4,17,4),None,'gru')
        self.assertEqual(budget.attempted,1);self.assertEqual(budget.completed,0)
        self.assertIsNone(budget.entries[0]['native_start_elapsed'])

    def test_admission_cannot_be_inferred_from_config_or_fake_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/'config.json';config.write_text('{}')
            for freeze in [{},{'execution_authorized':True},{'owner':'root coordinator','execution_authorized':True,'scientific_acceptance':True}]:
                with self.assertRaises(ValueError):c.validate_admission_fields(freeze,config,Path(directory)/'out')
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaises(ValueError):c.production('missing','missing','.', 'unused',time.monotonic())

    def test_source_pin_rejects_changed_bytes_and_parent_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);file=root/'source.py';file.write_text('original source\n')
            pin={'path':'source.py','bytes':file.stat().st_size,'sha256':c.sha(file)}
            self.assertEqual(c.verify_pin(root,pin),file)
            file.write_text('modified source\n')
            with self.assertRaisesRegex(ValueError,'mismatch'):c.verify_pin(root,pin)
            with self.assertRaises(ValueError):c.verify_pin(root,{**pin,'path':'../source.py'})

    def test_fake_core_state_and_cache_mutation_are_detected_without_native_calls(self):
        import torch
        class Core(torch.nn.Module):
            num_layers=1
            def __init__(self, mutate):super().__init__();self.mutate=mutate
            def forward(self,x,state=None):
                if state is None:state=torch.zeros(1,x.shape[0],3,dtype=x.dtype)
                else:
                    if self.mutate=='state':state.add_(1)
                    else:x.add_(1)
                return torch.zeros(x.shape[0],x.shape[1],3,dtype=x.dtype),state
        class Encoder(torch.nn.Module):
            def forward(self,x):return torch.zeros(x.shape[0],17,4,dtype=x.dtype)
        class Head(torch.nn.Module):
            def forward(self,x):return torch.zeros(x.shape[0],16,dtype=x.dtype)
        class FakeModel(torch.nn.Module):
            model_kind='gru';task='categorical';hidden_size=3
            def __init__(self,mutation):super().__init__();self.core=Core(mutation);self.encoder=Encoder();self.head=Head()
            def forward(self,x):return self.head(self.core(self.encoder(x))[0][:,-1])
        for mutation in ['state','cache']:
            arrays={};budget=c.CallBudget(30,time.monotonic(),20,1)
            with torch.no_grad(),self.assertRaisesRegex(c.StopCalibration,'mutation'):
                c.capture_condition(FakeModel(mutation).eval(),torch.zeros(4,17,38),
                    {'hidden':np.tile(np.array([1.,0.,0.]),(4,1))},
                    {'context':'fake','dtype':'float32','repeat':1},budget,arrays)
            self.assertIn('C_step_0/cache_before',arrays)
            self.assertIn('C_step_0/cache_after',arrays)


def run_native_campaign(output):
    """Bounded authorized unit fixture campaign, never real checkpoint loading."""
    start=time.monotonic()
    import torch
    from representation_collapse.pilot import VectorMemoryModel
    from representation_collapse.vector_task import render
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    budget=c.CallBudget(200,start,120,5,registry_path=output/'engine-native-registry.json')
    independent=[];independent_completed=0;records=[];status='incomplete'
    source_files={p:c.sha(p) for p in [c.ENGINE_PATH,c.LAUNCHER_PATH,'tests/test_cached_forward_calibration.py']}
    rng=np.random.default_rng(4103)
    try:
        for kind,task in [('gru','categorical'),('lstm','continuous')]:
            torch.manual_seed(4101)
            base=VectorMemoryModel(kind,task,3,4)
            weights={k:v.detach().clone() for k,v in base.state_dict().items()};del base
            histories=(rng.integers(0,16,size=(2,16),dtype=np.int64) if task=='categorical' else rng.uniform(-1,1,size=(2,16)).astype(np.float32))
            assert not np.array_equal(histories[0],histories[1])
            raw=render(np.concatenate([histories,histories]),np.array([0,1,0,1]),['']*4,task)
            directions={}
            for component in ['hidden']+(['cell'] if kind=='lstm' else []):
                draw=rng.standard_normal((2,3));unit=draw/np.linalg.norm(draw,axis=1,keepdims=True)
                directions[component]=np.concatenate([unit,unit])
            c.atomic_npz(output/(kind+'-fixture-inputs.npz'),{'inputs':raw,**{'direction_'+k:v for k,v in directions.items()}})
            previous={}
            for dtype in ['float64','float32']:
                for repeat in [1,2]:
                    condition={'id':f'{kind}-{dtype}-r{repeat}','context':kind+'-tiny','model':kind,'task':task,'dtype':dtype,'repeat':repeat}
                    model=VectorMemoryModel(kind,task,3,4);model.load_state_dict(weights)
                    model=model.to(dtype=getattr(torch,dtype)).eval();tensor=torch.tensor(raw,dtype=getattr(torch,dtype))
                    def pre(module,args):
                        if len(independent)>=200 or time.monotonic()-start>=120:raise RuntimeError('independent native fixture cap')
                        independent.append({'index':len(independent)+1,'condition':condition['id'],'input_shape':list(args[0].shape),'elapsed_seconds':time.monotonic()-start})
                        c.atomic_json(output/'independent-native-registry.json',independent)
                    def post(module,args,result):
                        nonlocal independent_completed
                        independent_completed+=1
                    h1=model.core.register_forward_pre_hook(pre);h2=model.core.register_forward_hook(post)
                    arrays={}
                    try:
                        with torch.no_grad():record=c.capture_condition(model,tensor,directions,condition,budget,arrays)
                    finally:
                        h1.remove();h2.remove();c.atomic_npz(output/(condition['id']+'.npz'),arrays)
                    assert record['core_calls']==(22 if kind=='gru' else 24)
                    assert all(x['passed'] for x in record['checks'])
                    assert arrays['C/complete_sequence'].shape==(4,18,3*(2 if kind=='lstm' else 1))
                    assert np.array_equal(arrays['C/complete_sequence'][:2],arrays['C/complete_sequence'][2:])
                    assert record['B0_intermediate_cell'] is None
                    if kind=='lstm':assert 'C/origin_before/cell' in arrays and 'B0/terminal/cell' in arrays
                    for arm in c.operations(kind)[20:]:
                        assert arm+'/requested_displacement' in arrays and arm+'/displacement' in arrays
                        assert np.all(np.linalg.norm(arrays[arm+'/displacement'].astype(np.float64),axis=1)>0)
                    if repeat==2:
                        for key,value in arrays.items():assert c.compare(value,previous[dtype][key],dtype,bitwise=True)['passed']
                    else:previous[dtype]={k:v.copy() for k,v in arrays.items()}
                    if dtype=='float32':
                        record['cross_dtype_checks']=[{'field':key,**c.compare(value,previous['float64'][key],'float32')} for key,value in arrays.items()]
                    record['arrays_sha256']=c.sha(output/(condition['id']+'.npz'));c.atomic_json(output/(condition['id']+'.json'),record)
                    records.append(condition['id'])
        assert budget.attempted==budget.completed==len(independent)==independent_completed==184
        assert budget.encoder_calls==16 and budget.head_calls==56
        assert time.monotonic()-start<120
        status='completed'
    finally:
        budget.persist()
        receipt={'kind':'cached_forward_v2_author_tiny_fixtures','fixtures_passed':status=='completed',
            'native_core_attempts':len(independent),'native_core_completed':independent_completed,
            'engine_native_attempts':budget.attempted,'engine_native_completed':budget.completed,
            'wall_seconds':time.monotonic()-start,'production_calls':0,'checkpoint_deserializations':0,
            'state':'completed' if status=='completed' else 'incomplete','conditions':records,
            'source_files':source_files,'hidden':3,'encoder':4,'batch':4,'independent_histories':2,
            'cpu_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
            'torch_version':str(torch.__version__),'numpy_version':np.__version__,
            'scientific_acceptance':False,'wandb':'unit fixtures exempt; no experiment run',
            'files':{p.name:c.sha(p) for p in output.iterdir() if p.is_file()}}
        c.atomic_json(output/'fixture-receipt.json',receipt)
    print(json.dumps({'fixtures_passed':True,'native_calls':184,'wall_seconds':receipt['wall_seconds'],'receipt':str(output/'fixture-receipt.json')}))


if __name__=='__main__':
    import sys
    if len(sys.argv)==3 and sys.argv[1]=='--native-campaign':run_native_campaign(sys.argv[2])
    else:unittest.main()
