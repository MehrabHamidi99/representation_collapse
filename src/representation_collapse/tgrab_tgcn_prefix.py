"""Admitted seed-0 T-GCN 256-update training-prefix diagnostic, never a fit.

Imports are stdlib-only. Mandatory offline reporting precedes native imports and
train-file reads. This worker has no validation/test opener or checkpoint path.
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
import random
import stat
import re
import sys
import time
from types import ModuleType

PROTOCOL_ID='tgrab_tgcn_training_prefix_v1'
WORKER='src/representation_collapse/tgrab_tgcn_prefix.py'
DATA_MANIFEST_SHA256='a6ff50183de570ff17cfbe92b3f1976bac9d2d85557ecc11379e24b2c737d1ce'
CALIBRATION_RESULT_SHA256='23b68f8c78e0f4b1fd00959c812463318cb687d8b9b95404c709c6ade5001619'
BACKEND_SHA256='0b03219543c0cce16e77c69cf7350a403e2d492134742bb1ec23aaf03dea512f'
CALIBRATION_HELPER='code/analysis/tgrab_tgcn_native_calibration.py'
CALIBRATION_HELPER_SHA256='049ec6e0514f0681500fba49c2e10b00524c051ffe9a019d1440aa121bf18ef9'


def require(value,message):
    if not value:raise RuntimeError(message)


def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('ascii')
def sha(raw):return hashlib.sha256(raw).hexdigest()


def pinned(path,expected):
    from .tgrab_figure10_canary import read_pinned_bytes
    return read_pinned_bytes(Path(path),expected,max_bytes=2_000_000)


def write_new(path,value):
    raw=canonical(value)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        with os.fdopen(fd,'wb',closefd=False) as stream:stream.write(raw);stream.flush();os.fsync(fd)
    finally:os.close(fd)
    directory=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(directory)
    finally:os.close(directory)
    return {'relative_path':Path(path).name,'bytes':len(raw),'sha256':sha(raw)}


def make_guard(deadline,stop_file,clock=time.time):
    require(type(deadline) in (float,int) and math.isfinite(deadline),'finite operational deadline required')
    def guard():
        require(clock()<deadline,'prefix operational deadline reached')
        require(not os.path.lexists(stop_file),'prefix owner STOP present')
    guard();return guard


class PrefixJournal:
    """Fresh hash-chained fsynced intents/outcomes, with honest poisoned state."""
    def __init__(self,root,context):
        self.root=Path(root);self.context=dict(context);self.previous=None;self.sequence=0
        self.attempted=self.completed=self.confirmed=0;self.pending=None;self.poisoned=False;self.entered=None
        self.path=self.root/'optimizer-journal.jsonl'
        self.stream=self.path.open('x+b');self.stream.flush();os.fsync(self.stream.fileno())
        self.identity=os.fstat(self.stream.fileno());self.byte_count=0;self.prefix_sha=sha(b'')
        directory=os.open(self.root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:os.fsync(directory)
        except BaseException:self.stream.close();raise
        finally:os.close(directory)

    def _append(self,kind,record):
        require(not self.poisoned,'journal failure is terminal')
        try:
            self._verify_file()
            payload={'kind':kind,**self.context,'event':self.sequence,'previous_record_sha256':self.previous,'record':record}
            raw=canonical(payload)
            require(self.stream.write(raw)==len(raw),'short journal write')
            self.stream.flush();os.fsync(self.stream.fileno())
            self.byte_count+=len(raw);self.prefix_sha=sha(os.pread(self.stream.fileno(),self.byte_count,0))
            self.previous=sha(raw);self.sequence+=1
            return self.previous
        except BaseException:
            self.poisoned=True;raise

    def _verify_file(self):
        st=self.path.lstat();opened=os.fstat(self.stream.fileno())
        identity=(self.identity.st_dev,self.identity.st_ino,1)
        require(stat.S_ISREG(st.st_mode) and stat.S_ISREG(opened.st_mode)
                and (st.st_dev,st.st_ino,st.st_nlink)==identity
                and (opened.st_dev,opened.st_ino,opened.st_nlink)==identity
                and st.st_size==opened.st_size==self.byte_count and self.stream.tell()==self.byte_count,'journal ownership or size changed')
        raw=os.pread(self.stream.fileno(),self.byte_count+1,0)
        require(len(raw)==self.byte_count and sha(raw)==self.prefix_sha,'prior journal bytes changed')
        return raw

    def on_attempt(self,record):
        try:return self._on_attempt(record)
        except BaseException:self.poisoned=True;raise

    def on_update(self,record):
        try:return self._on_update(record)
        except BaseException:self.poisoned=True;raise

    def _on_attempt(self,record):
        require(self.pending is None and not self.poisoned,'one pending attempt and no retry allowed')
        seq=record.get('sequence');require(type(seq) is int and seq==self.attempted+1 and seq<=256,'prefix intent sequence mismatch')
        require(record.get('epoch')==1 and record.get('target_timestamp')==seq and record.get('consumed_timestamp')==seq-1
                and record.get('training_order_index')==seq-1,'prefix intent chronology mismatch')
        self.entered={'sequence':seq,'stage':'intent_fsync_unconfirmed'}
        self._append('tgcn_prefix_optimizer_intent_v1',{**record,'kind':'tgcn_prefix_attempt_v1'})
        self.attempted=seq;self.pending={'sequence':seq,'intent_record_sha256':self.previous};self.entered=None

    def _on_update(self,record):
        require(not self.poisoned and self.pending is not None and record.get('attempt')==self.pending['sequence'],'outcome requires current intent')
        seq=self.pending['sequence'];status=record.get('optimizer_step_status')
        require(type(record.get('completed')) is bool and status in {'completed','unknown','not_started'},'explicit optimizer outcome required')
        completed=self.completed+int(record['completed']);confirmed=self.confirmed+int(status=='completed')
        require(record.get('attempted_updates')==self.attempted and record.get('completed_updates')==completed
                and record.get('optimizer_steps_confirmed')==confirmed and record.get('target_timestamp')==seq
                and record.get('consumed_timestamp')==seq-1,'outcome counters/chronology mismatch')
        self.entered={'sequence':seq,'stage':'outcome_fsync_unconfirmed','observed_step_status':status}
        self._append('tgcn_prefix_optimizer_outcome_v1',{**record,'intent_record_sha256':self.pending['intent_record_sha256']})
        self.completed=completed;self.confirmed=confirmed;self.pending=None;self.entered=None
        if record.get('error_type') is not None or record['completed'] is not True or status!='completed':self.poisoned=True

    def accounting(self):
        return {'durable_attempted_updates':self.attempted,'durable_completed_updates':self.completed,
            'durable_optimizer_steps_confirmed':self.confirmed,'pending':self.pending,'ambiguous_journal_boundary':self.entered,
            'journal_failed_or_terminal':self.poisoned,'journal_records':self.sequence,'last_record_sha256':self.previous}

    def audit(self):
        """Success-only full readback; ambiguous fsync boundaries never upgraded."""
        require(not self.poisoned and self.entered is None,'poisoned or ambiguous journal cannot certify success')
        try:
            raw=self._verify_file();lines=raw.splitlines(keepends=True)
            require(len(lines)==self.sequence and self.sequence==2*self.attempted,'complete intent/outcome pairs required')
            previous=None;completed=confirmed=0
            for index,line in enumerate(lines):
                event=json.loads(line)
                require(canonical(event)==line and event['event']==index and event['previous_record_sha256']==previous
                        and all(event[k]==v for k,v in self.context.items()),'journal hashchain/context differs')
                record=event['record'];seq=index//2+1
                if index%2==0:
                    require(event['kind']=='tgcn_prefix_optimizer_intent_v1' and record=={
                        'kind':'tgcn_prefix_attempt_v1','sequence':seq,'epoch':1,'training_order_index':seq-1,
                        'target_timestamp':seq,'consumed_timestamp':seq-1},'journal intent chronology differs')
                else:
                    require(event['kind']=='tgcn_prefix_optimizer_outcome_v1' and record['intent_record_sha256']==previous
                            and record['attempt']==record['attempted_updates']==record['epoch_attempted_updates']==seq
                            and record['epoch']==1 and record['target_timestamp']==seq and record['consumed_timestamp']==seq-1
                            and record['completed'] is True and record['optimizer_step_status']=='completed'
                            and record['optimizer_mutation_status']=='completed' and 'error_type' not in record,
                            'successful journal outcome differs')
                    completed+=1;confirmed+=1
                    require(record['completed_updates']==record['epoch_completed_updates']==completed
                            and record['optimizer_steps_confirmed']==confirmed and type(record['loss']) in (float,int)
                            and math.isfinite(record['loss']) and record['loss']>=0,'journal outcome counts/loss differ')
                previous=sha(line)
            require(previous==self.previous and completed==self.completed and confirmed==self.confirmed
                    and self.pending is None,'journal final accounting mismatch')
            return {**self.accounting(),'full_readback_verified':True,'journal_sha256':sha(raw),'journal_bytes':len(raw)}
        except BaseException:self.poisoned=True;raise

    def close(self):self.stream.close()


def run_prefix(*,backend,graphs,sampler,guard,journal,reporter,output):
    """Pure orchestration seam; actual runtime injects the witnessed native step."""
    from .tgrab_figure10_canary import run_prefix_updates,witnessed_native_update
    terminal={'state':None}
    class TrackingBackend:
        def __getattr__(self,name):return getattr(backend,name)
        def consume(self,state,graph):
            result=backend.consume(state,graph);terminal['state']=result;return result
    path=Path(output)/'updates.jsonl'
    stream=path.open('x+b');stream.flush();os.fsync(stream.fileno());raw_identity=os.fstat(stream.fileno());raw_bytes=[0]
    directory=os.open(Path(output),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(directory)
    finally:os.close(directory)
    def raw_check():
        st=path.lstat();opened=os.fstat(stream.fileno())
        require(stat.S_ISREG(st.st_mode) and (st.st_dev,st.st_ino,st.st_nlink)==(raw_identity.st_dev,raw_identity.st_ino,1)
                and (opened.st_dev,opened.st_ino,opened.st_nlink)==(raw_identity.st_dev,raw_identity.st_ino,1)
                and st.st_size==opened.st_size==raw_bytes[0], 'raw update log identity/size changed')
    def outcome(record):
        journal.on_update(record)
        raw_check();line=canonical(record);require(stream.write(line)==len(line),'short raw update write')
        stream.flush();os.fsync(stream.fileno());raw_bytes[0]+=len(line)
        guard()
        if record.get('completed') is True and 'loss' in record:
            reporter.log({'train/loss':record['loss'],'train/target_timestamp':record['target_timestamp'],
                'train/completed_updates':record['completed_updates']},step=record['attempt']-1)
    try:
        result=run_prefix_updates(backend=TrackingBackend(),graphs=graphs,sampler=sampler,guard=guard,
            on_attempt=journal.on_attempt,on_update=outcome,
            update=lambda state,target,negatives:witnessed_native_update(backend,state,target,negatives))
        require(journal.completed==journal.confirmed==journal.attempted==256 and journal.pending is None and not journal.poisoned,'complete durable prefix required')
        raw_check();updates_sha=sha(os.pread(stream.fileno(),raw_bytes[0],0))
        return result,terminal['state'],updates_sha
    finally:stream.close()


def validate_calibration(record):
    require(record.get('kind')=='tgcn_native_calibration_compact_acceptance_v1'
        and record.get('status')=='accepted_synthetic_execution' and record.get('source_closure_git_verified') is True
        and record.get('training_admitted') is False and record.get('scientific_acceptance') is False
        and record.get('result_sha256')==CALIBRATION_RESULT_SHA256 and record.get('checks_verified')==248
        and str(record.get('job_id'))=='11332092','independently accepted actual T-GCN calibration required')


def validate_supervisor(receipt,args,argv,source_sha):
    require(receipt.get('schema')=='tgrab-tgcn-prefix-supervisor-receipt-v1' and receipt.get('protocol_id')==PROTOCOL_ID
        and receipt.get('run_id')==args.run_id and receipt.get('deadline_unix')==args.deadline_unix
        and receipt.get('closure_sha256',{}).get(WORKER)==source_sha and receipt.get('canary_args')==list(argv),
        'actual prefix supervisor source/run/deadline/arguments mismatch')
    for key,pattern in [('job_id','[1-9][0-9]*'),('source_commit','[0-9a-f]{40}'),('admission_sha256','[0-9a-f]{64}')]:
        require(re.fullmatch(pattern,str(receipt.get(key,''))) is not None,'actual supervisor '+key+' required')
    if 'SLURM_JOB_ID' in os.environ:require(str(receipt['job_id'])==os.environ['SLURM_JOB_ID'],'actual Slurm job differs')
    require(receipt.get('calibration_acceptance')=={'path':args.calibration_acceptance,'sha256':args.calibration_acceptance_sha256}
            and receipt.get('data_root')==args.data_root and receipt.get('data_manifest_sha256')==args.data_manifest_sha256,
            'actual admitted calibration/data input pins differ')
    require(receipt['closure_sha256'].get('src/representation_collapse/tgrab_tgcn_backend.py')==BACKEND_SHA256
        and receipt['closure_sha256'].get(CALIBRATION_HELPER)==CALIBRATION_HELPER_SHA256,'calibrated backend/helper source changed')


def runtime(args,guard,reporter):
    """All numerical/native imports and graph payload reads stay in this call."""
    guard();repo=Path(__file__).resolve().parents[2];source=repo/CALIBRATION_HELPER
    module=ModuleType('_rc_tgcn_prefix_calibration_helpers');module.__file__=str(source)
    exec(compile(pinned(source,CALIBRATION_HELPER_SHA256),str(source),'exec'),module.__dict__)
    cal=module
    versions={name:importlib.metadata.version(name) for name in cal.VERSIONS}
    require(versions==cal.VERSIONS and sys.version_info[:2]==(3,10) and platform.system()=='Linux'
            and platform.machine()=='x86_64','pinned Linux CPU native runtime required')
    import numpy as np
    import torch
    from torch_geometric.nn import GCNConv
    from . import graph_data as gd
    from . import graph_reconstruction as gr
    from . import tgrab_tgcn_backend as adapter
    from .tgrab_figure10_bootstrap import actual_rng_snapshot
    require(sha(Path(inspect.getfile(adapter)).read_bytes())==BACKEND_SHA256,'executed calibrated backend changed')
    require(sha(pinned(Path(inspect.getfile(GCNConv)),cal.GCN_SHA256))==cal.GCN_SHA256,'GCN source mismatch')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    cell_dist=importlib.metadata.distribution('torch-geometric-temporal')
    matches=[p for p in cell_dist.files if str(p)=='torch_geometric_temporal/nn/recurrent/temporalgcn.py']
    require(len(matches)==1,'installed TGCN cell required')
    cell_path=Path(cell_dist.locate_file(matches[0]));cell=cal.exec_module('_rc_tgcn_prefix_cell',pinned(cell_path,cal.CELL_SHA256),cell_path)
    package=ModuleType('_rc_tgcn_prefix_wrapper');package.__path__=[];sys.modules[package.__name__]=package
    node_path=Path(args.link_head).with_name('node_emb.py')
    cal.exec_module(package.__name__+'.node_emb',pinned(node_path,cal.NODE_SHA256),node_path)
    previous=sys.modules.get('torch_geometric_temporal');facade=ModuleType('torch_geometric_temporal');facade.TGCN=cell.TGCN
    sys.modules['torch_geometric_temporal']=facade
    try:wrapper=cal.exec_module(package.__name__+'.tgcn',pinned(args.tgcn_wrapper,cal.WRAPPER_SHA256),args.tgcn_wrapper)
    finally:
        if previous is None:sys.modules.pop('torch_geometric_temporal',None)
        else:sys.modules['torch_geometric_temporal']=previous
    head=cal.exec_module(package.__name__+'.head',pinned(args.link_head,cal.HEAD_SHA256),args.link_head)
    guard();manifest=gd.read_manifest(Path(args.data_root)/'manifest.json',args.data_manifest_sha256,guard)
    require(manifest.metadata['kind']=='tgrab_primary_converted_v1','actual accepted primary conversion required')
    # Integrity-check the full training split only. No validation/test loader exists here.
    graphs=gd._load_split(manifest,'train',guard,gr.Snapshot)
    require(len(graphs)==2560,'complete verified primary training file required')
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    initial_rng=write_new(Path(args.output)/'seeding.json',actual_rng_snapshot(torch))
    models={key:value.to(device='cpu',dtype=torch.float32) for key,value in adapter.construct_models(wrapper.MultiLayerTGCN,wrapper.TGCNParams,head.LinkPredictor).items()}
    post_model_rng=write_new(Path(args.output)/'post-model-rng.json',actual_rng_snapshot(torch))
    named=[(f'{key}.{name}',p) for key in ('node_emb','link_pred') for name,p in sorted(models[key].named_parameters())]
    require(len({id(p) for _,p in named})==len(named),'aliased optimizer parameter identities')
    optimizer=torch.optim.Adam([p for _,p in named],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False)
    backend=adapter.NativeTGCNBackend(torch,models,optimizer,device='cpu');rng=np.random.default_rng(1)
    negative=write_new(Path(args.output)/'negative-generator.json',{'seed':1,'state':rng.bit_generator.state})
    def component_hashes():
        result={}
        for component,key in [('recurrent','node_emb'),('link_head','link_pred')]:
            h=hashlib.sha256();h.update(b'tgcn-component-v1\0')
            for name,tensor in models[key].state_dict().items():
                h.update(canonical({'name':name,'shape':list(tensor.shape),'dtype':'float32_le','order':'C'}));h.update(backend._f32le_bytes(tensor))
            result[component]=h.hexdigest()
        return result
    native_inventory={}
    for component,key in [('recurrent','node_emb'),('link_head','link_pred')]:
        values=models[key].state_dict()
        native_inventory[component]={'tensors':[{'name':name,'shape':list(value.shape),'dtype':str(value.dtype),
            'device':str(value.device),'nbytes':value.numel()*4} for name,value in values.items()],
            'numel':sum(value.numel() for value in values.values()),'tensor_count':len(values),
            'state_dict_versions':{name:meta['version'] for name,meta in values._metadata.items()}}
    require(native_inventory['recurrent']['numel']==137472 and native_inventory['link_head']['numel']==49537,
            'actual production parameter inventory differs')
    module_modes={}
    for label,context in [('train',backend.training_mode),('eval',backend.evaluation_mode)]:
        with context():
            module_modes[label]={'grad_enabled':torch.is_grad_enabled(),'flags':{key:{name:bool(module.training)
                for name,module in model.named_modules()} for key,model in models.items()}}
    initial=component_hashes();journal=PrefixJournal(args.output,{'run_id':args.run_id,'supervisor_receipt_sha256':args.supervisor_sha,
        'calibration_acceptance_sha256':args.calibration_acceptance_sha256,'data_manifest_sha256':args.data_manifest_sha256})
    try:
        result,state,updates_sha=run_prefix(backend=backend,graphs=graphs,sampler=lambda target:gr.sample_negatives(target,rng),
            guard=guard,journal=journal,reporter=reporter,output=args.output)
        accounting=journal.audit()
    except BaseException:
        write_new(Path(args.output)/'attempt-accounting.json',journal.accounting());raise
    finally:journal.close()
    terminal_bytes=backend._f32le_bytes(state.hidden)
    return {'kind':PROTOCOL_ID,'status':'prefix_completed','model_seed':0,'negative_seed':1,**result,'journal_accounting':accounting,
        'updates_sha256':updates_sha,'optimizer_journal_sha256':sha((Path(args.output)/'optimizer-journal.jsonl').read_bytes()),
        'terminal_hidden':{'shape':[1,100,128],'dtype':'float32_le','node_ids':list(range(100)),'next_target_t':256,
            'last_consumed_t':255,'sha256':sha(terminal_bytes),'semantic_sha256':sha(b'tgcn-hidden-v1\0'+canonical({'shape':[1,100,128],'node_ids':list(range(100)),'next_target_t':256})+terminal_bytes),
            'construction':'training_carry_after_predecessor255_before_target256_update_not_fixed_weight_replay','payload_written':False},
        'initialization':{'seeding_record':initial_rng,'post_model_rng_record':post_model_rng,'negative_generator_record':negative,
            'initial_component_semantic_sha256':initial,'terminal_component_semantic_sha256':component_hashes(),
            'optimizer_parameter_order':[n for n,_ in named]},
        'runtime':{'python':platform.python_version(),'machine':platform.machine(),'system':platform.system(),'versions':versions,
            'cpu_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'precision':'float32','device':'cpu',
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()},
        'data':{'manifest_sha256':manifest.sha256,'raw_sha256':manifest.metadata['raw']['sha256'],
            'train_split_sha256':manifest.metadata['splits']['train']['sha256'],'train_split_semantic_sha256':manifest.metadata['splits']['train']['semantic_sha256'],
            'train_interval':[0,2560],'computed_prefix_interval':[0,257]},
        'sources':{'wrapper_sha256':cal.WRAPPER_SHA256,'head_sha256':cal.HEAD_SHA256,'node_base_sha256':cal.NODE_SHA256,
            'cell_sha256':cal.CELL_SHA256,'gcn_sha256':cal.GCN_SHA256,'backend_sha256':BACKEND_SHA256},
        'calibration_result_sha256':CALIBRATION_RESULT_SHA256,'native_inventory':native_inventory,'module_modes':module_modes,'complete_epochs':0,
        'scientific_acceptance':False,'full_fit':False,'competence_claim':False}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','run-id','deadline-unix','stop-file','tgcn-wrapper','link-head','data-root','data-manifest-sha256',
                 'calibration-acceptance','calibration-acceptance-sha256'):
        p.add_argument('--'+name,required=True,type=float if name=='deadline-unix' else str)
    return p


def main(argv=None,*,reporter_factory=None,native_runner=runtime):
    argv=list(sys.argv[1:] if argv is None else argv);args=parser().parse_args(argv);out=Path(args.output)
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,95}',args.run_id),'safe run identity required')
    require(args.data_manifest_sha256==DATA_MANIFEST_SHA256,'same independently verified primary conversion required')
    require(not os.path.lexists(out),'fresh no-resume output required');out.mkdir(parents=True)
    if reporter_factory is None:
        from .reporting import ExperimentReporter
        reporter_factory=ExperimentReporter
    source_sha=sha(Path(__file__).read_bytes())
    try:
        guard=make_guard(args.deadline_unix,args.stop_file)
        with reporter_factory(out/'wandb',project='representation-collapse',mode='offline',run_id=args.run_id,
            config={'protocol':PROTOCOL_ID,'device':'cpu','precision':'float32','model_seed':0,'negative_seed':1,'max_updates':256,
                'validation_access':False,'test_access':False},identities={'worker_source_sha256':source_sha,
                'data_manifest_sha256':args.data_manifest_sha256,'calibration_acceptance_sha256':args.calibration_acceptance_sha256}) as reporter:
            receipt_path=os.environ.get('RC_SUPERVISOR_RECEIPT');args.supervisor_sha=os.environ.get('RC_SUPERVISOR_RECEIPT_SHA256')
            require(receipt_path and args.supervisor_sha,'actual admitted prefix supervisor required')
            receipt=json.loads(pinned(receipt_path,args.supervisor_sha));validate_supervisor(receipt,args,argv,source_sha)
            calibration=json.loads(pinned(args.calibration_acceptance,args.calibration_acceptance_sha256));validate_calibration(calibration)
            guard();result=native_runner(args,guard,reporter);guard()
            result.update(run_id=args.run_id,supervisor_receipt_sha256=args.supervisor_sha,
                calibration_acceptance_sha256=args.calibration_acceptance_sha256,source_commit=receipt['source_commit'],job_id=str(receipt['job_id']),
                admission_sha256=receipt['admission_sha256'],source_closure=receipt['closure_sha256'])
            ref=write_new(out/'prefix-result.json',result)
            reporter.log_manifest('tgcn-training-prefix-result',metadata={'completed_updates':result['completed_updates'],
                'validation_access':False,'test_access':False,'scientific_acceptance':False},identities={'result_sha256':ref['sha256'],
                'worker_source_sha256':source_sha,'supervisor_receipt_sha256':args.supervisor_sha,
                'calibration_acceptance_sha256':args.calibration_acceptance_sha256,'source_commit':receipt['source_commit']})
        write_new(out/'manifest.json',{'kind':PROTOCOL_ID,'status':'completed','result_sha256':ref['sha256'],
            'scientific_acceptance':False,'reporting_receipt_sha256':sha((out/'wandb/reporting-receipt.json').read_bytes())})
        return 0
    except BaseException as exc:
        write_new(out/'failure.json',{'kind':PROTOCOL_ID,'status':'failed','error_type':type(exc).__name__,'error':str(exc)[:500],
            'scientific_acceptance':False,'retry_allowed':False});raise

if __name__=='__main__':raise SystemExit(main())
