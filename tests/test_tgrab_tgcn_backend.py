"""Source-structure and synthetic tensor spies, never native TGCN execution."""
from contextlib import contextmanager
from types import SimpleNamespace
import subprocess
import sys
import unittest
import numpy as np
from representation_collapse import tgrab_tgcn_backend as tg
from representation_collapse.graph_reconstruction import Snapshot


class Tensor:
    def __init__(self,a,dtype='float32'):self.a=np.array(a,dtype=dtype);self.dtype=dtype;self.device='cpu'
    @property
    def shape(self):return self.a.shape
    @property
    def T(self):return Tensor(self.a.T,self.dtype)
    def reshape(self,*shape):return Tensor(self.a.reshape(*shape),self.dtype)
    def contiguous(self):return Tensor(self.a.copy(),self.dtype)
    def detach(self):return self
    def clone(self):return Tensor(self.a.copy(),self.dtype)
    def numel(self):return self.a.size
    def is_contiguous(self):return self.a.flags.c_contiguous
    def storage_offset(self):return 0
    def untyped_storage(self):return SimpleNamespace(data_ptr=lambda:self.a.__array_interface__['data'][0],nbytes=lambda:self.a.nbytes)
    def __getitem__(self,item):
        if isinstance(item,tuple):item=tuple(i.a if isinstance(i,Tensor) else i for i in item)
        return Tensor(self.a[item],self.dtype)
    def __ge__(self,other):return self.a>=other
    def __le__(self,other):return self.a<=other
    def cpu(self):return self
    def numpy(self):return self.a.copy()


class Torch:
    Tensor=Tensor;float32='float32';long='int64'
    def __init__(self):self.grad=True
    @staticmethod
    def zeros(shape,**kw):return Tensor(np.zeros(shape),kw.get('dtype','float32'))
    @staticmethod
    def ones(n,**kw):return Tensor(np.ones(n),kw.get('dtype','float32'))
    @staticmethod
    def eye(n,**kw):return Tensor(np.eye(n),kw.get('dtype','float32'))
    @staticmethod
    def tensor(a,**kw):return Tensor(a,kw.get('dtype','float32'))
    @staticmethod
    def isfinite(v):return np.isfinite(v.a)
    @staticmethod
    def equal(a,b):return np.array_equal(a.a,b.a)
    @contextmanager
    def set_grad_enabled(self,flag):
        old=self.grad;self.grad=flag
        try:yield
        finally:self.grad=old


class Model:
    def __init__(self,spec):
        self.values={name:Tensor(np.zeros(next(t['shape'] for t in spec['tensors'] if t['name']==name))) for name in spec['native_state_dict_order']}
        self.training=True;self.loads=0
    def state_dict(self):return self.values
    def named_parameters(self,remove_duplicate=False):return list(self.values.items())
    def modules(self):return [self]
    def train(self,mode):self.training=mode
    def eval(self):self.train(False)
    def load_state_dict(self,values,strict):
        assert strict;self.loads+=1;self.values={k:v.clone() for k,v in values.items()}
        return SimpleNamespace(missing_keys=[],unexpected_keys=[])


class TGCN:
    def __init__(self,n,h):
        self.in_channels=n;self.out_channels=h;self.improved=False;self.cached=False;self.add_self_loops=False
        for gate in 'zrh':setattr(self,'conv_'+gate,SimpleNamespace(cached=False,improved=False,add_self_loops=False,normalize=True,flow='source_to_target',_cached_edge_index=None,_cached_adj_t=None))


class Recurrent(Model):
    def __init__(self,spec,n,h):super().__init__(spec);self.num_units=1;self.units=[TGCN(n,h)];self.calls=[]
    def __call__(self,features,edges,hidden,*,edge_weight):
        self.calls.append((features,edges,hidden,edge_weight));out=Tensor(hidden.a+1)
        return out,out[-1]


class Head(Model):
    def __call__(self,a,b):return Tensor(np.full((a.shape[0],1),.5))


def fixture():
    layout=tg.TGCNLayout(4,3);catalog=tg.source_derived_catalog(nodes=4,hidden=3)
    models={'node_emb':Recurrent(catalog['components']['recurrent'],4,3),'link_pred':Head(catalog['components']['link_head'])}
    torch=Torch();backend=tg.NativeTGCNBackend(torch,models,layout=layout)
    return backend,models,torch


def graph(t=0,pairs=((0,1),(1,0))):return Snapshot(t,4,pairs,tuple(range(4)))


class TGCNBackendTests(unittest.TestCase):
    def test_module_import_does_not_import_native_torch(self):
        subprocess.run([sys.executable,'-c','import sys; import representation_collapse.tgrab_tgcn_backend; assert "torch" not in sys.modules'],check=True,capture_output=True)

    def test_source_catalog_production_counts_names_and_no_cell(self):
        c=tg.source_derived_catalog();self.assertEqual(c['total_parameters'],187009)
        self.assertEqual(c['components']['recurrent']['numel'],137472)
        self.assertEqual(c['components']['link_head']['numel'],49537)
        self.assertEqual(c['state_components'],['hidden']);self.assertFalse(c['native_inventory_verified'])
        self.assertEqual(tg.TGCNLayout().joint_dimension,12800)
        for spec in c['components'].values():
            offset=0
            for t in spec['tensors']:
                self.assertEqual(t['offset'],offset);offset+=t['nbytes']
            self.assertEqual(offset,spec['nbytes'])
        self.assertIn('units.0.conv_z.lin.weight',c['components']['recurrent']['serialization_order'])

    def test_constructor_argument_wiring_is_exact(self):
        calls=[]
        def params(*args,**kwargs):calls.append(('params',args,kwargs));return 'params'
        def wrapper(*args,**kwargs):calls.append(('wrapper',args,kwargs));return 'wrapper'
        def head(*args,**kwargs):calls.append(('head',args,kwargs));return 'head'
        self.assertEqual(tg.construct_models(wrapper,params,head),{'node_emb':'wrapper','link_pred':'head'})
        self.assertEqual(calls,[('params',(100,128),{'improved':False,'cached':False}),('wrapper',(1,'params'),{}),('head',(128,),{'num_layers':2,'p_dropout':.2})])

    def test_hidden_state_chronology_edge_weight_wiring_and_no_input_mutation(self):
        b,models,_=fixture();s=b.reset();self.assertFalse(hasattr(s,'cell'))
        out=b.consume(s,graph());self.assertEqual(out.next_target_t,1);self.assertEqual(s.next_target_t,0)
        self.assertTrue(np.array_equal(s.hidden.a,np.zeros((1,4,3))))
        features,edges,hidden,weights=models['node_emb'].calls[0]
        self.assertTrue(np.array_equal(features.a,np.eye(4)));self.assertEqual(edges.a.tolist(),[[0,1],[1,0]])
        self.assertEqual(weights.a.tolist(),[1.,1.]);self.assertIs(hidden,s.hidden)
        self.assertEqual(b.score_pairs(out,[(0,1),(1,0)]).tolist(),[.5,.5])
        with self.assertRaises(tg.TGCNError):b.consume(out,graph())
        with self.assertRaises(tg.TGCNError):b.consume(s,graph(pairs=((0,1),)))
        with self.assertRaises(tg.TGCNError):b.consume(s,graph(pairs=((1,0),(0,1))))

    def test_explicit_nonuniform_weight_wiring_and_shape_symmetry_guards(self):
        b,models,_=fixture();pairs=((0,1),(0,3),(1,0),(1,2),(2,1),(2,3),(3,0),(3,2))
        values=(.5,.75,.5,1.5,1.5,2.,.75,2.)
        b.consume(b.reset(),graph(pairs=pairs),edge_weight=values)
        self.assertEqual(models['node_emb'].calls[-1][3].a.tolist(),list(values))
        for bad in ((1.,),(.5,.75,1.,1.5,1.5,2.,.75,2.),(float('nan'),)*8,(True,)*8):
            with self.subTest(bad=bad),self.assertRaises(tg.TGCNError):b.consume(b.reset(),graph(pairs=pairs),edge_weight=bad)

    def test_state_shape_node_order_and_nonfinite_rejected(self):
        b,_,_=fixture()
        for s in (tg.HiddenOnlyState(Tensor(np.zeros((4,3))),0,(0,1,2,3)),
                  tg.HiddenOnlyState(Tensor(np.zeros((1,4,3))),0,(0,1,3,2)),
                  tg.HiddenOnlyState(Tensor(np.full((1,4,3),np.nan)),0,(0,1,2,3)),
                  tg.HiddenOnlyState(Tensor(np.zeros((1,4,3))),True,(0,1,2,3))):
            with self.subTest(s=s),self.assertRaises(tg.TGCNError):b.detach(s)
        with self.assertRaises(tg.TGCNError):b.detach(SimpleNamespace(hidden=Tensor(np.zeros((1,4,3))),cell=None))

    def test_detach_and_capture_are_independent(self):
        b,models,_=fixture();s=b.reset();d=b.detach(s);d.hidden.a[0,0,0]=5
        self.assertEqual(s.hidden.a[0,0,0],0)
        captured=b.capture_components();name=next(iter(captured['recurrent']));captured['recurrent'][name].a.flat[0]=4
        self.assertEqual(models['node_emb'].values[name].a.flat[0],0)

    def test_mode_restores_all_flags_and_gradient_on_exception(self):
        b,models,torch=fixture();models['link_pred'].training=False
        with self.assertRaisesRegex(RuntimeError,'sentinel'):
            with b.evaluation_mode():
                self.assertFalse(torch.grad);self.assertFalse(models['node_emb'].training)
                raise RuntimeError('sentinel')
        self.assertTrue(torch.grad);self.assertTrue(models['node_emb'].training);self.assertFalse(models['link_pred'].training)

    def test_both_load_components_validated_before_either_mutates(self):
        b,models,_=fixture();values=b.capture_components();key=next(iter(values['link_head']))
        values['link_head'][key]=Tensor([1])
        with self.assertRaises(tg.TGCNError):b.load_components(values)
        self.assertEqual(models['node_emb'].loads,0);self.assertEqual(models['link_pred'].loads,0)
        values=b.capture_components();first=next(iter(values['recurrent']));values['recurrent'][first].a.fill(2)
        b.load_components(values);self.assertEqual(models['node_emb'].loads,1)
        values['recurrent'][first].a.fill(9);self.assertEqual(models['node_emb'].values[first].a.flat[0],2)
        self.assertFalse(models['node_emb'].training);self.assertFalse(models['link_pred'].training)
        with self.assertRaises(tg.TGCNError):b.load_selected(None)

    def test_cached_or_selfloop_cell_and_wrong_tensor_inventory_rejected(self):
        for key in ('cached','improved','add_self_loops'):
            b,models,torch=fixture();setattr(models['node_emb'].units[0],key,True)
            with self.subTest(key=key),self.assertRaises(tg.TGCNError):tg.NativeTGCNBackend(torch,models,layout=b.layout)
        b,models,torch=fixture();models['node_emb'].units[0].conv_z._cached_edge_index='stale'
        with self.assertRaises(tg.TGCNError):tg.NativeTGCNBackend(torch,models,layout=b.layout)
        b,models,torch=fixture();models['link_pred'].values['extra']=Tensor([0])
        with self.assertRaises(tg.TGCNError):tg.NativeTGCNBackend(torch,models,layout=b.layout)

    def test_alias_payload_and_target_storage_rejected_before_any_load(self):
        b,models,_=fixture();values=b.capture_components()
        # Alias same-shaped bias tensors across recurrent gates.
        values['recurrent']['units.0.conv_r.bias']=values['recurrent']['units.0.conv_z.bias']
        with self.assertRaises(tg.TGCNError):b.load_components(values)
        self.assertEqual(models['node_emb'].loads,0)
        b,models,_=fixture();values=b.capture_components()
        models['node_emb'].values['units.0.conv_r.bias']=models['node_emb'].values['units.0.conv_z.bias']
        with self.assertRaises(tg.TGCNError):b.load_components(values)
        self.assertEqual(models['node_emb'].loads,0);self.assertEqual(models['link_pred'].loads,0)

    def test_signed_zero_readback_mismatch_is_not_hidden_by_numeric_equality(self):
        b,models,_=fixture();values=b.capture_components();name='seq.1.bias'
        values['link_head'][name].a.fill(-0.)
        load=models['link_pred'].load_state_dict
        def wrong(values,strict):
            report=load(values,strict);models['link_pred'].values[name].a.fill(0.);return report
        models['link_pred'].load_state_dict=wrong
        with self.assertRaisesRegex(tg.TGCNError,'byte readback'):b.load_components(values)

    def test_normalization_and_flow_are_frozen(self):
        for key,value in (('normalize',False),('flow','target_to_source')):
            b,models,torch=fixture();setattr(models['node_emb'].units[0].conv_z,key,value)
            with self.subTest(key=key),self.assertRaises(tg.TGCNError):tg.NativeTGCNBackend(torch,models,layout=b.layout)

    def test_update_rejects_inference_backend_before_optimizer_activity(self):
        b,_,_=fixture();state=b.consume(b.reset(),graph())
        with self.assertRaises(tg.TGCNError):b.loss_and_update(state,graph(1),[(0,2),(1,2)])

if __name__=='__main__':unittest.main()
