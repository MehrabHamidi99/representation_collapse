"""Train-motif dwell intervention on fixed, independently accepted GC-LSTM weights.

Explicit new diagnostic access; never opens validation/test or charges the prior
held-out evaluator. State observations precede target consumption, including the
separately flagged fresh-zero startup. No optimizer or checkpoint selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import platform
import subprocess
import time

PROTOCOL = 'gclstm_repetition_dwell_v1'
DWELLS = (8, 16, 32, 64, 128, 256)
HORIZON = 2048
PINS = {
    'manifest.json': '4abeba2b64261468c8844ee302570d6c5d48cec873a51f9eafd1e6010f75db61',
    'recurrent.f32le': '8e40c6b93fcfa567a1cdb8daef2ea67b444952f8b5aed693b30e0f0387ce502b',
    'link-head.f32le': 'c72a5f7919126cc07d2a23d8f00e93ab7d554a03bc133371042083473f28669f',
}
DATA_SHA = 'a6ff50183de570ff17cfbe92b3f1976bac9d2d85557ecc11379e24b2c737d1ce'


def schedule(dwell, start, horizon=HORIZON):
    if type(dwell) is not int or dwell not in DWELLS or start not in ('A', 'B'):
        raise ValueError('frozen dwell and start required')
    if type(horizon) is not int or horizon < 8*dwell or horizon % (2*dwell):
        raise ValueError('at least four whole cycles required')
    for t in range(horizon):
        yield t, ('A' if ((t // dwell) % 2 == 0) == (start == 'A') else 'B'), t % dwell, t // (2*dwell)


def reduce_rows(rows, dwell):
    def mean(items):
        return sum(r['f1'] for r in items) / len(items) if items else None
    cycles = HORIZON // (2*dwell)
    result = {'targets': len(rows), 'mean_f1': mean(rows), 'cycles': cycles, 'windows': {},
              'transition_mean_f1':mean([r for r in rows if r.get('transition')]),
              'nontransition_mean_f1':mean([r for r in rows if not r.get('transition') and not r.get('startup')]),
              'startup_f1':rows[0]['f1'],
              'per_cycle_mean_f1':[mean([r for r in rows if r['cycle']==c]) for c in range(cycles)]}
    for scope, keep in [('initial4', lambda r: r['cycle'] < 4), ('final4', lambda r: r['cycle'] >= cycles-4), ('last1024', lambda r: r['t'] >= HORIZON-1024)]:
        for motif in ('A', 'B'):
            subset = [r for r in rows if keep(r) and r['motif'] == motif]
            for window, phases in [('fixed2', ((1,2),(dwell-2,dwell-1))),
                                   ('relative_quarter', (tuple(range(1,1+(dwell-1)//4)),tuple(range(dwell-(dwell-1)//4,dwell))))]:
                early = [r for r in subset if r['phase'] in phases[0]]
                late = [r for r in subset if r['phase'] in phases[1]]
                a,b = mean(early),mean(late)
                result['windows'][f'{scope}/{motif}/{window}'] = {'early_f1':a,'late_f1':b,'late_minus_early':b-a,'early_minus_late_drop':a-b,'samples_per_window':len(early)}
    return result


def covariance_summary(samples):
    from .tgrab_figure10_trajectory_analysis import covariance_cloud
    import numpy as np
    x=np.asarray(samples,dtype=np.float64).reshape(len(samples),-1)
    return covariance_cloud(x)


def write_json(path, value):
    raw = (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    with Path(path).open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    return hashlib.sha256(raw).hexdigest()


def run(args, reporter, guard):
    import numpy as np
    import torch
    from . import graph_data as gd, graph_reconstruction as gr
    from .tgrab_figure10_canary import read_pinned_bytes
    from .tgrab_figure10_evaluate import load_native_weights
    from .pinned_graph_import import load_pinned_graph_modules, PINNED_CELL_SHA256, PINNED_WRAPPER_SHA256
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    payload = {name:read_pinned_bytes(args.candidate_dir/name, sha, max_bytes=12_582_912) for name,sha in PINS.items()}
    manifest_metadata=json.loads(payload['manifest.json'])
    if manifest_metadata['kind']!='tgrab_complete_epoch_candidate_v2' or manifest_metadata['epoch']!=160:
        raise RuntimeError('accepted candidate kind/epoch mismatch')
    for key,filename in [('recurrent','recurrent.f32le'),('link_head','link-head.f32le')]:
        descriptor=manifest_metadata['components'][key]
        if descriptor['filename']!=filename or descriptor['bytes']!=len(payload[filename]) or descriptor['sha256']!=PINS[filename]:
            raise RuntimeError('candidate component descriptor mismatch')
    modules = load_pinned_graph_modules(source_root=args.source_root,installed_cell_path=args.cell_path,
        cell_sha256=PINNED_CELL_SHA256,wrapper_sha256=PINNED_WRAPPER_SHA256,guard=guard)
    models = {'node_emb':modules.wrapper_type(1,modules.parameter_type(100,128,2)).to(device='cpu',dtype=torch.float32),
              'link_pred':modules.head_type(128,num_layers=2,p_dropout=0.2).to(device='cpu',dtype=torch.float32)}
    backend = gr.NativeGCLSTMBackend(torch,models,None,device='cpu')
    load_native_weights(backend,{'recurrent':payload['recurrent.f32le'],'link_head':payload['link-head.f32le']})
    manifest = gd.read_manifest(args.data_manifest,DATA_SHA,guard)
    train = gd._load_split(manifest,'train',guard,gr.Snapshot)
    motifs = {'A':train[0].positive_pairs,'B':train[32].positive_pairs}
    if motifs['A'] == motifs['B'] or any(s.positive_pairs != motifs['A' if s.timestamp%64<32 else 'B'] for s in train):
        raise RuntimeError('train-only two-motif identity check failed')
    del train
    motif_hashes = {k:hashlib.sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest() for k,v in motifs.items()}
    pairs = gr.ordered_offdiagonal_pairs(100)
    truths = {}
    for k,v in motifs.items():
        positive = set(v)
        truths[k] = np.asarray([p in positive for p in pairs])
    identities = {'checkpoint_pins':PINS,'data_manifest_sha256':DATA_SHA,'motif_sha256':motif_hashes,
                  'worker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'native_source_provenance':modules.provenance_dict(),
                  'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip(),
                  'protocol_sha256':hashlib.sha256((Path(__file__).resolve().parents[2]/'doc/auto/repetition-dwell-protocol-2026-10-20.md').read_bytes()).hexdigest(),
                  'python_version':platform.python_version(),'host':platform.node(),'cpu':platform.processor(),
                  'slurm_metadata':{k:os.getenv(k) for k in ('SLURM_JOB_ID','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE','SLURM_JOB_NODELIST')},
                  'motif_edges':{k:len(v) for k,v in motifs.items()},
                  'torch_version':torch.__version__,'numpy_version':np.__version__,'slurm_job_id':os.getenv('SLURM_JOB_ID'),
                  'test_payload_opened':False,'validation_payload_opened':False,'optimizer_created':False,
                  'state_coordinates':'H then C; layer,node,feature; unrescaled CPU FP32',
                  'checkpoint_load_scope':'new train-derived diagnostic, no held-out evaluation',
                  'relative_closure_zero_reference':'undefined/null when reference norm is zero',
                  'protocol':PROTOCOL,'horizon':HORIZON,'dwells':DWELLS}
    write_json(args.output/'inputs.json',identities)
    summaries=[];step=0
    with backend.evaluation_mode():
        for dwell in DWELLS:
            for start in ('A','B'):
                guard();state=backend.reset();rows=[];last=None;ring={};samples={}
                cycles=HORIZON//(2*dwell)
                previous_motif=None
                row_path=args.output/f'dwell-{dwell:03d}-{start}-rows.jsonl'
                with row_path.open('x') as stream:
                    for t,motif,phase,cycle in schedule(dwell,start):
                        guard()
                        snapshot=gr.Snapshot(t,100,motifs[motif],tuple(range(100)))
                        probability=np.asarray(backend.score_pairs(state,pairs))
                        if probability.shape != (9900,) or not np.isfinite(probability).all() or ((probability<0)|(probability>1)).any():raise RuntimeError('invalid prediction')
                        truth=truths[motif]
                        predicted=probability>=.5
                        tp=int((predicted&truth).sum());fp=int((predicted&~truth).sum());fn=int((~predicted&truth).sum())
                        baseline=None
                        if previous_motif is not None:
                            pred=truths[previous_motif];btp=int((pred&truth).sum());bfp=int((pred&~truth).sum());bfn=int((~pred&truth).sum())
                            baseline={'tp':btp,'fp':bfp,'fn':bfn,'tn':9900-btp-bfp-bfn,'f1':2*btp/(2*btp+bfp+bfn) if 2*btp+bfp+bfn else 0.}
                        previous_motif=motif
                        h=state.hidden.detach().numpy().reshape(-1).astype(np.float64)
                        c=state.cell.detach().numpy().reshape(-1).astype(np.float64)
                        joint=np.concatenate((h,c));previous=ring.get(t%(2*dwell))
                        row={'t':t,'motif':motif,'phase':phase,'cycle':cycle,'startup':t==0,'transition':phase==0 and t>0,
                             'tp':tp,'fp':fp,'fn':fn,'tn':len(pairs)-tp-fp-fn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                             'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,'persistence':baseline,
                             'period_absolute_l2':float(np.linalg.norm(joint-previous)) if previous is not None else None,
                             'period_reference_l2':float(np.linalg.norm(previous)) if previous is not None else None,
                             'hidden_l2':float(np.linalg.norm(h)),'cell_l2':float(np.linalg.norm(c)),
                             'complete_step_l2':float(np.linalg.norm(joint-last)) if last is not None else None,
                             'period_relative_l2':float(np.linalg.norm(joint-previous)/np.linalg.norm(previous)) if previous is not None and np.linalg.norm(previous)>0 else None}
                        ring[t%(2*dwell)]=joint;last=joint
                        if cycle>=cycles-4 and phase in (1,2,3,dwell-3,dwell-2,dwell-1):
                            window='early' if phase in (1,2,3) else 'late'
                            for kind,array in [('H',h),('C',c),('HC',joint)]:samples.setdefault(f'{motif}/{window}/{kind}',[]).append(array)
                        rows.append(row);stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
                        state=backend.detach(backend.consume(state,snapshot))
                        if (t+1)%64==0:
                            stream.flush();os.fsync(stream.fileno())
                            reporter.log({'diagnostic/dwell':dwell,'diagnostic/targets':t+1,'diagnostic/f1':row['f1']},step=step);step+=1
                cloud_path=args.output/f'dwell-{dwell:03d}-{start}-clouds.npz'
                with cloud_path.open('xb') as cloud_file:
                    np.savez(cloud_file,**{key:np.asarray(value,dtype=np.float64) for key,value in samples.items()})
                    cloud_file.flush();os.fsync(cloud_file.fileno())
                summary={'clouds_file':cloud_path.name,'clouds_sha256':hashlib.sha256(cloud_path.read_bytes()).hexdigest(),'dwell':dwell,'start':start,**reduce_rows(rows,dwell),
                         'covariance':{key:covariance_summary(value) for key,value in samples.items()},
                         'rows_file':row_path.name,'rows_sha256':hashlib.sha256(row_path.read_bytes()).hexdigest(),
                         'last_cycle_max_period_relative_l2':max((r['period_relative_l2'] for r in rows if r['cycle']==cycles-1 and r['period_relative_l2'] is not None),default=None)}
                write_json(args.output/f'dwell-{dwell:03d}-{start}-summary.json',summary);summaries.append(summary)
    result={**identities,'status':'complete','conditions':summaries,'scientific_acceptance':False,
            'memory_capacity_measured':False,'floquet_spectrum_measured':False}
    digest=write_json(args.output/'result.json',result)
    reporter.log_manifest('dwell-diagnostic-result',metadata={'conditions':len(summaries),'targets':len(summaries)*HORIZON,'payload_upload':False},identities={'result_sha256':digest})
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('candidate-dir','data-manifest','source-root','cell-path','output','stop-file'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--run-id',required=True);parser.add_argument('--deadline-unix',type=float,required=True)
    args=parser.parse_args(argv)
    def guard():
        if time.time()>=args.deadline_unix or args.stop_file.exists():raise RuntimeError('deadline or STOP')
    guard();args.output.mkdir(parents=True,exist_ok=False)
    from .reporting import ExperimentReporter
    try:
        with ExperimentReporter(args.output/'wandb',project='representation-collapse',mode='offline',run_id=args.run_id,
            config={'protocol':PROTOCOL,'device':'cpu','precision':'float32','horizon':HORIZON},
            identities={'candidate_manifest_sha256':PINS['manifest.json'],'data_manifest_sha256':DATA_SHA}) as reporter:
            run(args,reporter,guard)
    except BaseException as exc:
        write_json(args.output/'failure.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc)[:1000]})
        raise
    return 0

if __name__=='__main__':raise SystemExit(main())
