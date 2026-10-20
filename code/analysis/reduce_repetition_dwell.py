"""Independently audit compact dwell rows and render descriptive scientific plots."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

DWELLS=(8,16,32,64,128,256)
HORIZON=2048

def require(ok,message):
    if not ok: raise ValueError(message)

def close(a,b):
    return isinstance(a,(int,float)) and not isinstance(a,bool) and math.isfinite(a) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)

def confusion(row):
    values=[row[k] for k in ('tp','fp','fn','tn')]
    require(all(type(v) is int and v>=0 for v in values) and sum(values)==9900,'invalid confusion counts')
    tp,fp,fn,_=values
    require(close(row['f1'],2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.),'incorrect F1')
    if 'precision' in row: require(close(row['precision'],tp/(tp+fp) if tp+fp else 0.),'incorrect precision')
    if 'recall' in row: require(close(row['recall'],tp/(tp+fn) if tp+fn else 0.),'incorrect recall')

def verify_rows(rows,dwell,start):
    require(dwell in DWELLS and start in ('A','B'),'condition invalid')
    require(len(rows)==HORIZON,'incomplete horizon')
    for t,r in enumerate(rows):
        motif=start if (t//dwell)%2==0 else ('B' if start=='A' else 'A')
        require(type(r['t']) is int and r['t']==t and r['phase']==t%dwell and r['cycle']==t//(2*dwell) and r['motif']==motif,'chronology mismatch')
        require(r['startup'] is (t==0) and r['transition'] is (t>0 and t%dwell==0),'phase flags mismatch')
        confusion(r)
        if t==0: require(r['persistence'] is None,'startup baseline must be undefined')
        else:
            confusion(r['persistence'])
            require(r['persistence']['tp']+r['persistence']['fn']==r['tp']+r['fn'],'baseline truth mismatch')
            if t%dwell: require(r['persistence']['fp']==r['persistence']['fn']==0,'nontransition persistence mismatch')
        for key in ('hidden_l2','cell_l2'):
            require(isinstance(r[key],(int,float)) and math.isfinite(r[key]) and r[key]>=0,'invalid norm')
        if t<2*dwell:
            require(all(r[k] is None for k in ('period_absolute_l2','period_reference_l2','period_relative_l2')),'premature period comparison')
        else:
            a,b=r['period_absolute_l2'],r['period_reference_l2']
            require(math.isfinite(a) and math.isfinite(b) and min(a,b)>=0,'invalid closure')
            require(r['period_relative_l2'] is None if b==0 else close(r['period_relative_l2'],a/b),'closure ratio mismatch')
    return True

def average(rows,key='f1'):
    return sum(r[key] for r in rows)/len(rows) if rows else None

def reduce_rows(rows,dwell,start):
    verify_rows(rows,dwell,start)
    cycles=HORIZON//(2*dwell)
    result={'dwell':dwell,'start':start,'targets':len(rows),'mean_f1':average(rows),
            'last1024_mean_f1':average(rows[1024:]),'persistence_mean_f1_excluding_startup':average([r['persistence'] for r in rows[1:]]),
            'transition_mean_f1':average([r for r in rows if r['transition']]),'nontransition_mean_f1':average([r for r in rows if not r['transition'] and not r['startup']]),'windows':{}}
    for scope,selected in [('initial4',rows[:8*dwell]),('final4',rows[-8*dwell:]),('last1024',rows[-1024:])]:
        for motif in ('A','B'):
            for name,width in [('fixed2',2),('relative_quarter',(dwell-1)//4)]:
                a=[r for r in selected if r['motif']==motif and 1<=r['phase']<=width]
                b=[r for r in selected if r['motif']==motif and dwell-width<=r['phase']<dwell]
                require(len(a)==len(b)>0,'window counts mismatch')
                result['windows'][f'{scope}/{motif}/{name}']={'early_f1':average(a),'late_f1':average(b),'drop':average(a)-average(b),'samples_per_window':len(a)}
    result['phase']={m:[average([r for r in rows[-8*dwell:] if r['motif']==m and r['phase']==q]) for q in range(dwell)] for m in ('A','B')}
    result['phase_confusion']={m:{k:[average([r for r in rows[-8*dwell:] if r['motif']==m and r['phase']==q],k) for q in range(dwell)] for k in ('fp','fn')} for m in ('A','B')}
    return result

def pinned(path,digest):
    data=path.read_bytes()
    require(hashlib.sha256(data).hexdigest()==digest,'SHA mismatch: '+str(path))
    return data

def audit(directory,result_sha):
    original=json.loads(pinned(directory/'result.json',result_sha))
    require(original['status']=='complete' and original['protocol']=='gclstm_repetition_dwell_v1','incomplete/wrong protocol')
    require(original['horizon']==HORIZON and tuple(original['dwells'])==DWELLS,'wrong sweep')
    require(not original['test_payload_opened'] and not original['validation_payload_opened'] and not original['optimizer_created'],'access scope mismatch')
    require(original['checkpoint_pins']['manifest.json']=='4abeba2b64261468c8844ee302570d6c5d48cec873a51f9eafd1e6010f75db61','wrong selected checkpoint')
    require(original['data_manifest_sha256']=='a6ff50183de570ff17cfbe92b3f1976bac9d2d85557ecc11379e24b2c737d1ce','wrong data manifest')
    require(set(original['motif_sha256'])=={'A','B'} and original['motif_sha256']['A']!=original['motif_sha256']['B'],'motif identity mismatch')
    require(original['source_commit']=='3541ae02441357664a9328fcfebd515a472d9668','wrong admitted source revision')
    require(original['worker_sha256']=='62fe9a75bd0785fe430735a0514068a1f9f4db21286811a52a1125ceef8fd127','wrong admitted worker bytes')
    require(original['protocol_sha256']=='483f14d3a88558928d3c0b074789e3f31a253244c5b5d8dbfbbb7558ffeb966b','wrong frozen protocol bytes')
    require(original['checkpoint_pins']['recurrent.f32le']=='8e40c6b93fcfa567a1cdb8daef2ea67b444952f8b5aed693b30e0f0387ce502b','wrong recurrent weights')
    require(original['checkpoint_pins']['link-head.f32le']=='c72a5f7919126cc07d2a23d8f00e93ab7d554a03bc133371042083473f28669f','wrong head weights')
    seen=set();results=[]
    for summary in original['conditions']:
        dwell,start=summary['dwell'],summary['start'];key=(dwell,start)
        require(key not in seen,'duplicate condition');seen.add(key)
        require(summary['rows_file']==f'dwell-{dwell:03d}-{start}-rows.jsonl','unexpected row path')
        rows=[json.loads(line) for line in pinned(directory/summary['rows_file'],summary['rows_sha256']).splitlines()]
        item=reduce_rows(rows,dwell,start)
        require(close(item['mean_f1'],summary['mean_f1']),'worker mean mismatch')
        for name,w in item['windows'].items():
            old=summary['windows'][name]
            require(old['samples_per_window']==w['samples_per_window'] and close(old['early_f1'],w['early_f1']) and close(old['late_f1'],w['late_f1']) and close(old['early_minus_late_drop'],w['drop']),'worker window mismatch')
        covariance=summary['covariance']
        require(set(covariance)=={f'{m}/{w}/{v}' for m in ('A','B') for w in ('early','late') for v in ('H','C','HC')},'geometry clouds mismatch')
        for name,c in covariance.items():
            require(c['samples']==12 and c['time_rank_ceiling']==11,'geometry sample count mismatch')
            eigen=c['cleaned_eigenvalues_descending'];require(len(eigen)==12 and all(math.isfinite(v) and v>=0 for v in eigen),'invalid spectrum')
            if sum(eigen)>0:
                pr=sum(eigen)**2/sum(v*v for v in eigen)
                require(close(c['participation_ratio'],pr),'geometry PR mismatch')
            else: require(c['participation_ratio'] is None,'zero geometry must be undefined')
        item['covariance']=covariance;results.append(item)
    require(seen=={(d,s) for d in DWELLS for s in ('A','B')},'missing sweep condition')
    return {'kind':'independent_dwell_compact_reduction_v1','input_result_sha256':result_sha,'conditions':results,'targets_verified':HORIZON*12,'geometry_verification':'compact spectral arithmetic only; no binary state reanalysis','memory_capacity_measured':False,'floquet_measured':False}

def geometry_ratio(early,late):
    return late/early if early is not None and late is not None and early>0 else float('nan')

def plots(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    conditions=result['conditions'];fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for start,style in [('A','-'),('B','--')]:
        items=sorted((c for c in conditions if c['start']==start),key=lambda c:c['dwell'])
        for motif,color in [('A','tab:blue'),('B','tab:orange')]:
            ax[0].plot(DWELLS,[100*c['windows'][f'final4/{motif}/fixed2']['drop'] for c in items],style+'o',color=color,label=f'Motif {motif}, start {start}')
        ax[1].plot(DWELLS,[c['mean_f1'] for c in items],style+'o',label=f'Model, start {start}')
        ax[1].plot(DWELLS,[c['persistence_mean_f1_excluding_startup'] for c in items],style,color='gray',label=f'Persistence, start {start}')
    ax[0].axhline(0,color='gray',lw=.7);ax[0].set_ylabel('Early − late F1 (percentage points)');ax[1].set_ylabel('Mean target F1')
    for a in ax:a.set_xscale('log',base=2);a.set_xlabel('Motif dwell L');a.legend(fontsize=7);a.grid(alpha=.2)
    fig.suptitle('Fixed epoch-160 GC-LSTM · 2048 targets per condition · one model/graph\nLeft: final four cycles, two-step windows. Right: all 2048 targets; persistence excludes startup.')
    files=[]
    def save(fig,name):
        for ext in ('png','pdf'):
            p=out/f'{name}.{ext}';fig.savefig(p,dpi=180);files.append(p)
        plt.close(fig)
    save(fig,'dwell-behavior')
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for start,row in [('A',0),('B',1)]:
        for motif,col in [('A',0),('B',1)]:
            a=axes[row,col]
            for c in conditions:
                if c['start']==start:a.plot([q/c['dwell'] for q in range(c['dwell'])],c['phase'][motif],label=str(c['dwell']))
            a.set(title=f'Motif {motif}, start {start}',xlabel='Within-block phase / L',ylabel='F1');a.legend(title='Dwell',fontsize=7);a.grid(alpha=.2)
    fig.suptitle('Final-four-cycle phase averages · transition phase zero retained');save(fig,'dwell-phase')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for start,style in [('A','-'),('B','--')]:
        items=sorted((c for c in conditions if c['start']==start),key=lambda c:c['dwell'])
        for motif,color in [('A','tab:blue'),('B','tab:orange')]:
            for axis,field in zip(axes,('participation_ratio','sample_covariance_trace')):
                vals=[]
                for c in items:
                    early=c['covariance'][f'{motif}/early/HC'][field];late=c['covariance'][f'{motif}/late/HC'][field]
                    vals.append(geometry_ratio(early,late))
                axis.plot(DWELLS,vals,style+'o',color=color,label=f'{motif}, start {start}')
    for a,label in zip(axes,('H+C PR: late / early','H+C covariance trace: late / early')):
        a.set_xscale('log',base=2);a.set(xlabel='Motif dwell L',ylabel=label);a.axhline(1,color='gray',lw=.7);a.legend(fontsize=7);a.grid(alpha=.2)
    fig.suptitle('Temporal state geometry · three-phase windows × four cycles\n12 correlated rows/cloud; unscaled H+C. These are not memory-capacity measurements.');save(fig,'dwell-geometry')
    fig,axes=plt.subplots(3,2,figsize=(12,9),layout='constrained')
    colors=plt.get_cmap('viridis')([i/5 for i in range(6)])
    for col,motif in enumerate(('A','B')):
        for c in conditions:
            color=colors[DWELLS.index(c['dwell'])];style='-' if c['start']=='A' else '--'
            phase=list(range(c['dwell']))
            axes[0,col].plot(phase,c['phase'][motif],style,color=color,lw=1.1,label=f"L={c['dwell']}, start {c['start']}")
            for row,key in ((1,'fp'),(2,'fn')):
                axes[row,col].plot(phase,c['phase_confusion'][motif][key],style,color=color,lw=1.1)
        axes[0,col].set_title(f'Motif {motif}')
        for row,label in enumerate(('F1','False-positive pairs','False-negative pairs')):
            axes[row,col].set(xlabel='Absolute within-block phase (steps)',ylabel=label)
            axes[row,col].axvline(32,color='gray',lw=.8,alpha=.7)
            axes[row,col].grid(alpha=.2)
    axes[0,1].legend(fontsize=7,ncol=2)
    fig.suptitle('Final-four-cycle phase means at absolute repetition age\nSolid: start A; dashed: start B. Gray line marks training dwell 32; no mechanism inferred.')
    save(fig,'dwell-absolute-phase-errors')
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--result-sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args(argv)
    a.output.mkdir(parents=True,exist_ok=False)
    from representation_collapse.reporting import ExperimentReporter
    with ExperimentReporter(a.output/'reporting',project='representation-collapse',mode='offline',run_id=a.run_id,config={'protocol':'dwell-independent-reduction-v1'},identities={'result_sha256':a.result_sha256}) as reporter:
        result=audit(a.input,a.result_sha256)
        manifest=plots(result,a.output)
        raw=json.dumps(result,indent=2,allow_nan=False)+'\n';(a.output/'results.json').write_text(raw)
        lines=['# Repetition dwell descriptive results','', 'One selected checkpoint and graph; no independent-seed uncertainty or memory-capacity inference.','']
        for c in result['conditions']:
            drops=[100*c['windows'][f'final4/{m}/fixed2']['drop'] for m in ('A','B')]
            lines.append(f"- L={c['dwell']}, start {c['start']}: mean F1 {c['mean_f1']:.6f}; A/B early-minus-late drops {drops[0]:.4f}/{drops[1]:.4f} percentage points.")
        (a.output/'numbers.md').write_text('\n'.join(lines)+'\n')
        manifest['results.json']=hashlib.sha256(raw.encode()).hexdigest();(a.output/'plot-manifest.json').write_text(json.dumps({'input_result_sha256':a.result_sha256,'files':manifest},indent=2)+'\n')
        reporter.log({'verification/targets':result['targets_verified'],'verification/conditions':12},step=0)
        reporter.log_manifest('dwell-independent-results',metadata={'payload_upload':False},identities={'results_sha256':manifest['results.json']})
    return 0

if __name__=='__main__':raise SystemExit(main())
