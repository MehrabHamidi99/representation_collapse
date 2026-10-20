#!/usr/bin/env python3
"""Future supervised T-GCN engineering calibration; imports are stdlib-only.

Synthetic inputs only. Passing this script is neither empirical evidence nor
permission to fit/evaluate a dataset. The outer supervisor owns resource limits.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import random
import stat
import sys
import time
from types import ModuleType

PROTOCOL_ID = 'tgrab_tgcn_native_calibration_v1'
WORKER = 'code/analysis/tgrab_tgcn_native_calibration.py'
WRAPPER_SHA256 = '028a85f0527833a7e62099e8bb826ac144df29738ee120795d205d96a8c6f78c'
HEAD_SHA256 = 'cca12cfc73b8660c92236f530504da08ba0b606ec13a1a681ffabcae43b7ca86'
NODE_SHA256 = '4be4450730f33949e0354348b3bd8f69ba0434e4aa911abec0fc5f798e8c2d29'
CELL_SHA256 = '1f417ad10b1fb3ed6af24731efb0e80b614495a3e4df3fc16254e631c3c8819e'
GCN_SHA256 = '036b464769ad1d98501c9fc18ff3132b507890b2c5cb41b3e6f9e1f22d4f57f6'
A = ((0,1),(0,3),(1,0),(1,2),(2,1),(2,3),(3,0),(3,2))
B = ((0,1),(0,2),(0,3),(1,0),(2,0),(3,0))
C_WEIGHTS = (.5,.75,.5,1.5,1.5,2.,.75,2.)
D = ((0,1),(1,0),(2,3),(3,2))
NEGATIVES = ((0,2),(1,3),(2,0),(3,1))
VERSIONS = {'torch':'2.0.0+cpu','torch-geometric':'2.6.1','torch-geometric-temporal':'0.54.0',
            'numpy':'1.24.4','torch-scatter':'2.1.2+pt20cpu','torch-sparse':'0.6.18+pt20cpu',
            'torch-cluster':'1.6.3+pt20cpu','torch-spline-conv':'1.2.2+pt20cpu','wandb':'0.30.0'}


def require(value, message):
    if not value: raise RuntimeError(message)


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def pinned(path, expected):
    path=Path(path);st=path.lstat()
    require(stat.S_ISREG(st.st_mode) and 0<st.st_size<=2_000_000,'bounded regular source/receipt required')
    raw=path.read_bytes();after=path.lstat()
    fields=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
    require(fields(st)==fields(after) and len(raw)==st.st_size and hashlib.sha256(raw).hexdigest()==expected,'source/receipt pin differs')
    return raw


def write_new(path,value):
    raw=canonical(value);path=Path(path);partial=path.with_suffix(path.suffix+'.partial')
    require(not path.exists() and not partial.exists(),'fresh result required')
    with partial.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    os.link(partial,path);partial.unlink()
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return hashlib.sha256(raw).hexdigest()


def make_guard(deadline,stop_file,clock=time.time):
    require(type(deadline) in (int,float) and math.isfinite(deadline),'finite deadline required')
    def guard():
        require(clock()<deadline,'calibration deadline reached')
        require(not os.path.lexists(stop_file),'calibration STOP present')
    guard();return guard


def exec_module(name,raw,path):
    require(name not in sys.modules,'fresh isolated source namespace required')
    module=ModuleType(name);module.__file__=str(path);module.__package__=name.rpartition('.')[0]
    sys.modules[name]=module
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module


def native_calibration(args,guard,reporter):
    """Actual native schedule, invoked only inside mandatory entered reporter."""
    guard()
    versions={name:importlib.metadata.version(name) for name in VERSIONS}
    require(versions==VERSIONS and sys.version_info[:2]==(3,10) and platform.system()=='Linux' and platform.machine()=='x86_64','pinned Linux CPU native environment required')
    import numpy as np
    import torch
    from torch_geometric.nn import GCNConv
    from representation_collapse import graph_reconstruction as gr
    from representation_collapse import tgrab_tgcn_backend as adapter
    backend_sha=hashlib.sha256(Path(inspect.getfile(adapter)).read_bytes()).hexdigest()
    require(args.source_closure['src/representation_collapse/tgrab_tgcn_backend.py']==backend_sha,'loaded backend source differs from supervisor closure')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    wrapper_raw=pinned(args.tgcn_wrapper,WRAPPER_SHA256);head_raw=pinned(args.link_head,HEAD_SHA256)
    node_path=Path(args.link_head).with_name('node_emb.py');node_raw=pinned(node_path,NODE_SHA256)
    dist=importlib.metadata.distribution('torch-geometric-temporal')
    files=[p for p in dist.files if str(p)=='torch_geometric_temporal/nn/recurrent/temporalgcn.py']
    require(len(files)==1,'exact installed temporal cell source required')
    cell_path=Path(dist.locate_file(files[0]));cell_raw=pinned(cell_path,CELL_SHA256)
    cell=exec_module('_rc_tgcn_calibration_cell',cell_raw,cell_path)
    package=ModuleType('_rc_tgcn_calibration_wrapper');package.__path__=[];sys.modules[package.__name__]=package
    exec_module(package.__name__+'.node_emb',node_raw,node_path)
    previous=sys.modules.get('torch_geometric_temporal');facade=ModuleType('torch_geometric_temporal');facade.TGCN=cell.TGCN
    sys.modules['torch_geometric_temporal']=facade
    try:wrapper=exec_module(package.__name__+'.tgcn',wrapper_raw,args.tgcn_wrapper)
    finally:
        if previous is None:sys.modules.pop('torch_geometric_temporal',None)
        else:sys.modules['torch_geometric_temporal']=previous
    head=exec_module(package.__name__+'.head',head_raw,args.link_head)
    gcn_path=Path(inspect.getfile(GCNConv));gcn_sha=hashlib.sha256(pinned(gcn_path,GCN_SHA256)).hexdigest()
    checks={};metrics={};exact_comparisons={}
    def check(name,predicate):
        guard();require(bool(predicate),'calibration gate failed: '+name);checks[name]=True
    def exact(name,left,right):
        check(name,tuple(left.shape)==tuple(right.shape) and left.dtype==right.dtype==torch.float32 and str(left.device)==str(right.device)=='cpu' and torch.isfinite(left).all() and torch.isfinite(right).all() and left.detach().contiguous().numpy().tobytes()==right.detach().contiguous().numpy().tobytes())
        exact_comparisons[name]={'max_abs_error':float((left.detach()-right.detach()).abs().max()),'bytes_equal':True}
    layout=adapter.TGCNLayout(nodes=4,hidden=3)
    def models(layout):
        return {key:model.to(device='cpu',dtype=torch.float32) for key,model in adapter.construct_models(wrapper.MultiLayerTGCN,wrapper.TGCNParams,head.LinkPredictor,layout=layout).items()}
    def optimizer(ms):
        return torch.optim.Adam([p for c in ('node_emb','link_pred') for _,p in sorted(ms[c].named_parameters())],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False)
    def seed_zero():
        random.seed(0);np.random.seed(0);torch.manual_seed(0)
    def component_hashes(values):
        hashes={}
        for component,model in values.items():
            digest=hashlib.sha256()
            for name,value in model.state_dict().items():
                digest.update(canonical({'name':name,'shape':list(value.shape),'dtype':str(value.dtype)}))
                digest.update(value.detach().contiguous().numpy().tobytes())
            hashes[component]=digest.hexdigest()
        return hashes
    seed_zero();first_seeded=models(layout);first_initial=component_hashes(first_seeded)
    first_rng=torch.get_rng_state().clone()
    seed_zero();second_seeded=models(layout);second_initial=component_hashes(second_seeded)
    for component in first_seeded:
        for name,value in first_seeded[component].state_dict().items():
            exact('repeated_seed0_'+component+'_'+name,value,second_seeded[component].state_dict()[name])
    check('repeated_seed0_initial_hashes',first_initial==second_initial)
    check('repeated_seed0_postconstruction_rng',torch.equal(first_rng,torch.get_rng_state()))
    initial_rng_sha=hashlib.sha256(first_rng.numpy().tobytes()).hexdigest()
    del first_seeded,second_seeded
    seed_zero()
    entry_grad_enabled=torch.is_grad_enabled()
    ms=models(layout);backend=adapter.NativeTGCNBackend(torch,ms,optimizer(ms),layout=layout,device='cpu')
    original_modes=[(c,n,m.training) for c,model in ms.items() for n,m in model.named_modules()]
    reference=models(layout)
    for component in ms:reference[component].load_state_dict(ms[component].state_dict(),strict=True)
    refopt=optimizer(reference)
    zero=backend.reset();check('hidden_only_zero_state',not hasattr(zero,'cell') and zero.next_target_t==0 and zero.node_ids==tuple(range(4)) and tuple(zero.hidden.shape)==(1,4,3) and not torch.count_nonzero(zero.hidden) and zero.hidden.numpy().tobytes()==bytes(48))
    x=torch.eye(4,dtype=torch.float32);pairs=gr.ordered_offdiagonal_pairs(4)
    snapshots=[gr.Snapshot(t,4,edges,tuple(range(4))) for t,edges in enumerate((A,B,A))]
    def direct(previous,snapshot,weights=None):
        edges=torch.tensor(snapshot.positive_pairs,dtype=torch.long).T.contiguous()
        weights=torch.ones(len(snapshot.positive_pairs)) if weights is None else weights
        return reference['node_emb'].units[0](x,edges,edge_weight=weights,H=previous[0]).unsqueeze(0)
    # Evaluation: independent direct dependency call bypasses wrapper.forward.
    with backend.evaluation_mode():
        for module in reference.values():module.eval()
        state=zero;rh=zero.hidden.detach().clone()
        for snapshot in snapshots:
            before=state.hidden.detach().clone();state_next=backend.consume(state,snapshot);rh=direct(rh,snapshot)
            exact('forward_'+str(snapshot.timestamp),state_next.hidden,rh);exact('input_unchanged_'+str(snapshot.timestamp),state.hidden,before)
            check('new_state_storage_'+str(snapshot.timestamp),state_next.hidden.data_ptr()!=state.hidden.data_ptr())
            state=backend.detach(state_next);check('detach_'+str(snapshot.timestamp),not state.hidden.requires_grad and state.hidden.grad_fn is None and state.hidden.data_ptr()!=state_next.hidden.data_ptr())
        check('nonzero_carried_hidden',bool(torch.count_nonzero(state.hidden)))
        for unit in ms['node_emb'].units:
            for conv in (unit.conv_z,unit.conv_r,unit.conv_h):
                check('flags_'+str(len(checks)),conv.cached is False and conv.add_self_loops is False and conv._cached_edge_index is None and conv._cached_adj_t is None)
        # Same zero H and A->B->A must reproduce A while graph B differs.
        outputs=[]
        for snapshot in snapshots:
            h,_=ms['node_emb'](x,torch.tensor(snapshot.positive_pairs,dtype=torch.long).T.contiguous(),zero.hidden,edge_weight=torch.ones(len(snapshot.positive_pairs)))
            outputs.append(h)
        exact('dynamic_A_B_A_no_cache',outputs[0],outputs[2]);check('dynamic_graph_used',not torch.equal(outputs[0],outputs[1]))
        weights=torch.tensor(C_WEIGHTS,dtype=torch.float32)
        hw,_=ms['node_emb'](x,torch.tensor(A,dtype=torch.long).T.contiguous(),zero.hidden,edge_weight=weights)
        exact('nonuniform_weight_wrapper_direct',hw,direct(zero.hidden,snapshots[0],weights))
        exact('nonuniform_weight_backend_direct',backend.consume(zero,snapshots[0],edge_weight=C_WEIGHTS).hidden,hw);check('nonuniform_weights_used',not torch.equal(hw,outputs[0]))
        empty_edges=torch.empty((2,0),dtype=torch.long)
        empty_h,_=ms['node_emb'](x,empty_edges,zero.hidden,edge_weight=torch.empty(0,dtype=torch.float32))
        permuted_h,_=ms['node_emb'](x.flip(0),empty_edges,zero.hidden,edge_weight=torch.empty(0,dtype=torch.float32))
        exact('empty_edge_features_no_selfloops',empty_h,permuted_h)
        empty_snapshot=gr.Snapshot(0,4,(),tuple(range(4)))
        exact('empty_edge_backend',backend.consume(zero,empty_snapshot).hidden,empty_h)
        probs=backend.score_pairs(state,pairs)
        idx=torch.tensor(pairs,dtype=torch.long);refprobs=reference['link_pred'](rh[-1,idx[:,0]],rh[-1,idx[:,1]]).reshape(-1)
        exact('score_all_pairs',torch.as_tensor(probs),refprobs)
    check('evaluation_restores_modes_and_grad',original_modes==[(c,n,m.training) for c,model in ms.items() for n,m in model.named_modules()] and torch.is_grad_enabled()==entry_grad_enabled)
    # One balanced score-before-consume update. Both branches replay identical RNG
    # for the real dropout calls and use independently constructed Adam objects.
    # Positive fixed head coefficients keep the tiny synthetic ReLU path active.
    # This is declared before execution and applies only to the calibration step.
    with torch.no_grad():
        for component in (ms['link_pred'],reference['link_pred']):
            for parameter in component.parameters():parameter.fill_(.1)
    torch.manual_seed(73)
    with backend.training_mode():
        for module in reference.values():module.train()
        with torch.no_grad():
            warm=backend.detach(backend.consume(backend.reset(),snapshots[0]));refwarm=direct(zero.hidden,snapshots[0]).detach().clone()
        exact('training_warm_hidden',warm.hidden,refwarm)
        check('training_warm_hidden_nonzero',bool(torch.count_nonzero(warm.hidden)))
        state=backend.consume(warm,snapshots[1]);rh=direct(refwarm,snapshots[1]);rng=torch.get_rng_state().clone()
        check('training_target_chronology',state.next_target_t==2 and warm.next_target_t==1)
        target=gr.Snapshot(2,4,D,tuple(range(4)))
        loss=backend.loss_and_update(state,target,NEGATIVES)
        torch.set_rng_state(rng);refopt.zero_grad(set_to_none=True)
        def rp(ps):
            ix=torch.tensor(ps,dtype=torch.long);return reference['link_pred'](rh[-1,ix[:,0]],rh[-1,ix[:,1]]).reshape(-1)
        pos,neg=rp(D),rp(NEGATIVES)
        ref_loss=torch.nn.functional.binary_cross_entropy(pos,torch.ones_like(pos))+torch.nn.functional.binary_cross_entropy(neg,torch.zeros_like(neg))
        ref_loss.backward();check('loss_exact',loss==float(ref_loss.detach()))
        for component in ms:
            for (name,p),(rn,q) in zip(ms[component].named_parameters(),reference[component].named_parameters()):
                check('gradient_presence_'+component+'_'+name,name==rn and (p.grad is None)==(q.grad is None))
                if p.grad is not None:exact('gradient_'+component+'_'+name,p.grad,q.grad)
        reset_gradient_tensors=sum(int(p.grad is not None and bool(torch.count_nonzero(p.grad))) for n,p in ms['node_emb'].named_parameters() if '.conv_r.' in n or '.linear_r.' in n)
        check('nonzero_reset_gate_gradient',reset_gradient_tensors>0)
        refopt.step()
        for component in ms:
            for name,p in ms[component].state_dict().items():exact('update_'+component+'_'+name,p,reference[component].state_dict()[name])
    check('training_restores_modes_and_grad',original_modes==[(c,n,m.training) for c,model in ms.items() for n,m in model.named_modules()] and torch.is_grad_enabled()==entry_grad_enabled)
    native_opt,reference_opt=backend.optimizer.state_dict(),refopt.state_dict()
    check('optimizer_groups_and_state_keys',native_opt['param_groups']==reference_opt['param_groups'] and set(native_opt['state'])==set(reference_opt['state']))
    for index,values in native_opt['state'].items():
        check('optimizer_state_keys_'+str(index),set(values)==set(reference_opt['state'][index]))
        for name,value in values.items():
            other=reference_opt['state'][index][name]
            if isinstance(value,torch.Tensor):exact('optimizer_'+str(index)+'_'+name,value,other)
            else:check('optimizer_'+str(index)+'_'+name,type(value) is type(other) and value==other)
    saved=backend.capture_components();fresh_ms=models(layout);fresh=adapter.NativeTGCNBackend(torch,fresh_ms,None,layout=layout,device='cpu')
    before_invalid=fresh.capture_components();bad={k:dict(v) for k,v in saved.items()};bad['link_head'].pop(next(iter(bad['link_head'])))
    rejected=False
    try:fresh.load_components(bad)
    except adapter.TGCNError:rejected=True
    check('strict_missing_head_rejected',rejected)
    after_invalid=fresh.capture_components()
    for component in before_invalid:
        for name,value in before_invalid[component].items():exact('invalid_load_no_mutation_'+component+'_'+name,value,after_invalid[component][name])
    fresh.load_components(saved)
    for component in ms:
        for name,p in ms[component].state_dict().items():
            exact('strict_roundtrip_'+component+'_'+name,p,fresh_ms[component].state_dict()[name]);check('capture_nonalias_'+component+'_'+name,saved['recurrent' if component=='node_emb' else 'link_head'][name].data_ptr()!=p.data_ptr())
    with backend.evaluation_mode(),fresh.evaluation_mode():
        left,right=backend.reset(),fresh.reset()
        for snapshot in snapshots:left=backend.detach(backend.consume(left,snapshot));right=fresh.detach(fresh.consume(right,snapshot))
        exact('fresh_zero_full_prefix_replay',left.hidden,right.hidden)
        check('terminal_identity',left.next_target_t==right.next_target_t==3 and left.node_ids==right.node_ids==tuple(range(4)))
    prod=models(adapter.TGCNLayout(nodes=100,hidden=128))
    counts={c:sum(p.numel() for p in m.parameters()) for c,m in prod.items()}
    check('production_native_catalog_counts',counts=={'node_emb':137472,'link_pred':49537})
    catalog={c:[{'name':n,'shape':list(p.shape),'dtype':str(p.dtype),'count':p.numel(),'nbytes':p.numel()*p.element_size()} for n,p in m.state_dict().items()] for c,m in prod.items()}
    prod_catalog=adapter.source_derived_catalog(nodes=100,hidden=128)
    for component,spec in prod_catalog['components'].items():
        actual=prod[spec['backend_key']].state_dict()
        check('production_catalog_order_'+component,list(actual)==spec['native_state_dict_order'])
        check('production_catalog_shapes_'+component,all(tuple(actual[r['name']].shape)==tuple(r['shape']) for r in spec['tensors']))
    production_versions={c:{name:record['version'] for name,record in m.state_dict()._metadata.items()} for c,m in prod.items()}
    check('production_module_versions',all(type(v) is int for versions in production_versions.values() for v in versions.values()))
    check('production_cpu_fp32_finite',all(p.dtype==torch.float32 and str(p.device)=='cpu' and torch.isfinite(p).all() for m in prod.values() for p in m.state_dict().values()))
    captured=backend.capture_state(left)
    check('hidden_capture_nonalias',captured.hidden.data_ptr()!=left.hidden.data_ptr() and captured.next_target_t==left.next_target_t and captured.node_ids==left.node_ids and not hasattr(captured,'cell'))
    exact('hidden_capture_exact',captured.hidden,left.hidden)
    terminal_bytes=captured.hidden.numpy().astype('<f4',copy=False).tobytes(order='C')
    live_before=left.hidden.detach().clone()
    with backend.evaluation_mode():
        scores_before=backend.score_pairs(left,pairs)
        captured.hidden.fill_(123.)
        scores_after=backend.score_pairs(left,pairs)
    exact('mutated_capture_live_hidden_unchanged',left.hidden,live_before)
    exact('mutated_capture_live_scores_unchanged',torch.as_tensor(scores_before),torch.as_tensor(scores_after))
    check('complete_hidden_capture_size',len(terminal_bytes)==1*4*3*4)
    metrics.update(maximum_exact_tensor_absolute_error=max(v['max_abs_error'] for v in exact_comparisons.values()),exact_tensor_comparisons=len(exact_comparisons),reset_gate_nonzero_gradient_tensors=reset_gradient_tensors,tiny_loss=loss,checks_passed=len(checks),recurrent_parameters=counts['node_emb'],head_parameters=counts['link_pred'])
    reporter.log(metrics,step=0)
    return {'kind':PROTOCOL_ID,'status':'synthetic_native_calibration_passed','checks':checks,'exact_tensor_comparisons':exact_comparisons,'metrics':metrics,'tensor_catalog':catalog,'production_state_dict_versions':production_versions,'model_sets_constructed':6,'initialization':{'repeated_seed0_constructions':2,'initial_component_sha256':first_initial,'postconstruction_torch_rng_sha256':initial_rng_sha,'python_seed':0,'numpy_seed':0,'torch_seed':0,'dropout_branch_seed':73,'training_head_parameter_fill':.1,'training_warm_snapshot_t':0,'training_tracked_predecessor_t':1,'training_target_t':2,'deterministic_algorithms':True,'cudnn_benchmark':False,'cudnn_deterministic':True},'runtime':{'python':platform.python_version(),'platform':platform.platform(),'versions':versions,'intraop_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'entry_grad_enabled':entry_grad_enabled},'sources':{'wrapper_sha256':WRAPPER_SHA256,'head_sha256':HEAD_SHA256,'node_base_sha256':NODE_SHA256,'temporalgcn_sha256':CELL_SHA256,'gcn_conv_sha256':gcn_sha,'backend_sha256':backend_sha},'fixtures':{'nodes':4,'hidden':3,'A':A,'B':B,'C_weights':C_WEIGHTS,'training_target_D':D,'negatives':NEGATIVES},'tolerance':{'absolute':0,'relative':0},'complete_state':'hidden_only','tiny_terminal_hidden_sha256':hashlib.sha256(terminal_bytes).hexdigest(),'tiny_terminal_hidden_shape':[1,4,3],'fixture_sha256':hashlib.sha256(canonical({'A':A,'B':B,'C_weights':C_WEIGHTS,'training_target_D':D,'negatives':NEGATIVES})).hexdigest(),'datasets_read':False,'held_out_opened':False,'scientific_acceptance':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('output','run-id','deadline-unix','stop-file','tgcn-wrapper','link-head'):p.add_argument('--'+flag,required=True,type=float if flag=='deadline-unix' else str)
    return p


def main(argv=None,*,reporter_factory=None,native_runner=native_calibration):
    args=parser().parse_args(argv);out=Path(args.output)
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,95}',args.run_id),'safe fresh run identity required')
    require(not os.path.lexists(out),'fresh calibration output required');out.mkdir(parents=True)
    guard=make_guard(args.deadline_unix,args.stop_file)
    if reporter_factory is None:
        from representation_collapse.reporting import ExperimentReporter
        reporter_factory=ExperimentReporter
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        with reporter_factory(out/'wandb',project='representation-collapse',run_id=args.run_id,mode='offline',config={'protocol':PROTOCOL_ID,'synthetic_only':True,'device':'cpu','precision':'float32'},identities={'worker_source_sha256':source_sha}) as reporter:
            receipt_path=os.environ.get('RC_SUPERVISOR_RECEIPT');receipt_sha=os.environ.get('RC_SUPERVISOR_RECEIPT_SHA256')
            require(receipt_path is not None and receipt_sha is not None,'actual supervisor receipt required')
            receipt=json.loads(pinned(receipt_path,receipt_sha))
            require(receipt['protocol_id']==PROTOCOL_ID and receipt['run_id']==args.run_id and receipt['closure_sha256'][WORKER]==source_sha and receipt['deadline_unix']==args.deadline_unix,'supervisor source/run/deadline binding differs')
            require(all(re.fullmatch(pattern,str(receipt.get(key,''))) for key,pattern in (('job_id','[1-9][0-9]*'),('source_commit','[0-9a-f]{40}'),('admission_sha256','[0-9a-f]{64}'))),'actual job/source/admission identity required')
            args.source_closure=receipt['closure_sha256']
            guard();result=native_runner(args,guard,reporter);guard()
            result['run_id']=args.run_id;result['supervisor_receipt_sha256']=receipt_sha
            for key in ('job_id','source_commit','admission_sha256'):result[key]=receipt[key]
            digest=write_new(out/'calibration-result.json',result)
            reporter.log_manifest('tgcn-native-calibration-result',metadata={'synthetic_only':True,'scientific_acceptance':False},identities={'result_sha256':digest,'worker_source_sha256':source_sha,'supervisor_receipt_sha256':receipt_sha,'source_commit':receipt['source_commit'],'admission_sha256':receipt['admission_sha256']})
        write_new(out/'manifest.json',{'kind':PROTOCOL_ID,'status':'completed','result_sha256':digest,'scientific_acceptance':False,'reporting_receipt_sha256':hashlib.sha256((out/'wandb/reporting-receipt.json').read_bytes()).hexdigest()})
        return 0
    except BaseException as exc:
        write_new(out/'failure.json',{'kind':PROTOCOL_ID,'status':'failed','error_type':type(exc).__name__,'error':str(exc)[:500],'scientific_acceptance':False})
        raise

if __name__=='__main__':raise SystemExit(main())
