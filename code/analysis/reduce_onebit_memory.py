"""Independent compact audit of frozen one-bit discovery; no model/state loading."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

MOTIFS = ('AXAXAXBXBXBX', 'AXAXBXAXBXBX', 'AXAXBXBXAXBX', 'AXBXAXAXBXBX', 'AXBXBXAXAXBX')
GRID = ((0,1),(1,1),(2,1),(4,1),(8,1),(16,1),(32,1),(16,2),(8,4))

def require(value, message):
    if not value: raise ValueError(message)

def number(x):
    return type(x) in (int, float) and math.isfinite(x)

def close(a, b):
    return number(a) and number(b) and math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-12)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def rows(path): return [json.loads(line) for line in path.read_text().splitlines()]

def gate(counts, n=512):
    require(type(n) is int and n == 512, 'unexpected gate denominator')
    require(len(counts)==4 and all(type(v) is int and 0<=v<=n for v in counts), 'invalid gate counts')
    return all(v>=487 for v in counts)

def designs(pairs, n, r, q):
    require(len(pairs)==n, 'schedule count')
    library = {''.join(c*q for c in s) for s in MOTIFS}
    for pair in pairs:
        require(type(pair) is list and len(pair)==2 and all(type(x) is str for x in pair), 'schedule pair')
        p,c=pair
        require(len(p)==len(c)==12*r*q, 'body duration')
        if r==0: continue
        width=12*q
        pb=[p[t:t+width] for t in range(0,len(p),width)]
        cb=[c[t:t+width] for t in range(0,len(c),width)]
        require(set(pb)<=library and set(cb)<=library and len(set(pb))==1, 'motif membership/repetition')
        require(pb[-1]==cb[-1], 'final motif mismatch')
        require(Counter(p)==Counter(c) and Counter(zip(p,p[1:]))==Counter(zip(c,c[1:])), 'body sufficient statistics')
        if r==1: require(p==c, 'R1 arms')
        else: require(not any(len(c)%k==0 and c==c[:k]*(len(c)//k) for k in range(1,len(c)//2+1)), 'control is periodic')

def reduce_rows(data,n,condition,interventions):
    require(len(data)==4*n, 'incomplete or duplicate condition rows')
    variants={'native','swap','self_copy','same_cue_copy','zero'} if interventions else {'native'}
    counts={(bit,arm):0 for bit in (0,1) for arm in (0,1)}
    ce={0:[],1:[]}; errors={0:[],1:[]}; accuracy={v:0 for v in variants}; donor=0
    for i,row in enumerate(data):
        bit=(i%4)//2; arm=i%2
        require(row['condition']==condition and type(row['forcing_unit']) is int and row['forcing_unit']==i//4, 'condition/unit ordering')
        require(type(row['bit']) is int and row['bit']==bit and type(row['arm']) is int and row['arm']==arm, 'cue/arm ordering')
        require(set(row['logits'])==set(row['prediction'])==set(row['error'])==variants, 'variant schema')
        for v in variants:
            logits=row['logits'][v]
            require(type(logits) is list and len(logits)==2 and all(number(x) for x in logits), 'nonfinite/invalid logits')
            prediction=int(logits[1]>logits[0])
            require(type(row['prediction'][v]) is int and row['prediction'][v]==prediction, 'argmax mismatch')
            require(type(row['error'][v]) is int and row['error'][v]==int(prediction!=bit), 'error mismatch')
            accuracy[v]+=int(prediction==bit)
        require(type(row['oracle_prediction']) is int and row['oracle_prediction']==bit, 'WRITE oracle label mismatch')
        good=int(row['prediction']['native']==bit); counts[bit,arm]+=good; errors[arm].append(1-good)
        a,b=row['logits']['native']; m=max(a,b)
        ce[arm].append(m+math.log(math.exp(a-m)+math.exp(b-m))-row['logits']['native'][bit])
        if interventions:
            opposite=i+2 if bit==0 else i-2
            chunk_start=(i//128)*128; chunk_units=min(32,n-chunk_start//4)
            previous=chunk_start+(((i-chunk_start)//4-1)%chunk_units)*4+i%4
            require(row['logits']['swap']==data[opposite]['logits']['native'], 'donor replay mismatch')
            require(row['logits']['self_copy']==row['logits']['native'], 'self-copy mismatch')
            require(row['logits']['same_cue_copy']==data[previous]['logits']['native'], 'same-cue replay mismatch')
            correct=int(row['prediction']['swap']==1-bit)
            require(type(row['swap_donor_correct']) is int and row['swap_donor_correct']==correct, 'donor score')
            donor+=correct
        else: require(row['swap_donor_correct']==0, 'validation donor sentinel')
    average=lambda a:sum(a)/len(a)
    result={'condition':condition,'forcing_units':n,'counts':{f'bit{b}_arm{a}':v for (b,a),v in counts.items()},
            'accuracy':{f'bit{b}_arm{a}':v/n for (b,a),v in counts.items()},
            'arm_accuracy':{str(a):1-average(errors[a]) for a in (0,1)},
            'mean_accuracy':accuracy['native']/(4*n), 'cross_entropy':{str(a):average(ce[a]) for a in (0,1)}}
    if interventions:
        # Average cue errors within a forcing draw before the arm contrast.
        diffs=[sum(data[4*j+k]['error']['native'] for k in (0,2))/2-sum(data[4*j+k]['error']['native'] for k in (1,3))/2 for j in range(n)]
        result.update(paired_error_difference=average(diffs),swap_donor_accuracy=donor/(4*n),
                      intervention_accuracy_against_original={k:v/(4*n) for k,v in accuracy.items()})
    return result

def compare_summary(actual,expected,interventions):
    require(actual['condition']==expected['condition'] and actual['forcing_units']==expected['forcing_units'], 'summary identity')
    require(actual['cue_body_units']==2*expected['forcing_units'], 'summary units')
    for key in ('accuracy','cross_entropy'):
        require(set(actual[key])==set(expected[key]) and all(close(actual[key][k],v) for k,v in expected[key].items()), 'summary '+key)
    require(close(actual['mean_accuracy'],expected['mean_accuracy']), 'summary mean')
    require(close(actual['constant_zero_readout_accuracy'],.5) and close(actual['register_oracle_accuracy'],1), 'baseline summary')
    if interventions:
        for key in ('paired_error_difference','swap_donor_accuracy'): require(close(actual[key],expected[key]), 'summary '+key)
        require(set(actual['intervention_accuracy_against_original'])==set(expected['intervention_accuracy_against_original']), 'intervention summary variants')
        for k,v in expected['intervention_accuracy_against_original'].items(): require(close(actual['intervention_accuracy_against_original'][k],v), 'intervention summary')

def audit(root,result_sha):
    require(sha(root/'result.json')==result_sha,'result digest')
    result=read(root/'result.json'); identity=read(root/'identity.json'); inventory=read(root/'artifact-inventory.json')
    require(result['schema']=='onebit-memory-discovery-v1','schema')
    required={'result.json','identity.json','validation-design.json','training.jsonl','checkpoint.pt','pre-evaluation-checkpoint.json'}
    require(required<=set(inventory),'missing mandatory inventory binding')
    for key in ('source_sha256','protocol_sha256','validation_sha256','initial_model_sha256'): require(result[key]==identity[key], 'identity '+key)
    for key,val in [('seed',0),('training_rng_seed',1000),('validation_rng_seed',2000),('test_rng_seed',3000)]: require(type(identity[key]) is int and identity[key]==val, 'RNG identity')
    checked=0; omitted=[]
    for name,entry in inventory.items():
        require(Path(name).name==name,'inventory path')
        path=root/name
        if not path.exists():
            require(name=='checkpoint.pt' or name.endswith('-states.npz'),'missing compact artifact '+name)
            omitted.append(name); continue
        require(path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],'inventory mismatch '+name); checked+=1
    require(inventory['result.json']['sha256']==result_sha,'inventory result')
    require(inventory['checkpoint.pt']['sha256']==result['checkpoint_sha256'],'checkpoint inventory')
    pre=read(root/'pre-evaluation-checkpoint.json')
    require(pre['checkpoint_sha256']==result['checkpoint_sha256'] and pre['model_sha256']==result['final_model_sha256'],'frozen checkpoint identity')
    require(sha(root/'validation-design.json')==result['validation_sha256'],'validation digest')
    validation=read(root/'validation-design.json'); require(set(validation)=={'0','1','2'},'validation horizons')
    for r in (0,1,2): designs(validation[str(r)],512,r,1)
    stages=result['stage_results']; require(type(stages) is list and 1<=len(stages)<=3,'stage count')
    reduced_stages=[]
    for index,stage in enumerate(stages,1):
        require({f'stage-{index}.json',f'stage-{index}-validation.jsonl'}<=set(inventory),'stage inventory binding')
        require(stage==read(root/f'stage-{index}.json') and stage['stage']==index and stage['update']==(1000,3000,5000)[index-1],'stage identity')
        data=rows(root/f'stage-{index}-validation.jsonl'); horizons=[0] if index==1 else [0,1,2]
        require(len(data)==len(horizons)*2048 and len(stage['checks'])==len(horizons),'stage rows/count')
        reduced=[]
        for j,r in enumerate(horizons):
            item=reduce_rows(data[j*2048:(j+1)*2048],512,f'validation-R{r}',False)
            compare_summary(stage['checks'][j],item,False); reduced.append(item)
            if r in (0,1):
                for row in data[j*2048:(j+1)*2048:2]:
                    k=j*2048+row['forcing_unit']*4+row['bit']*2
                    require(data[k]['logits']==data[k+1]['logits'],'identical-arm logits')
        passed=all(gate(list(x['counts'].values())) for x in reduced)
        require(type(stage['passed']) is bool and stage['passed']==passed,'stage competence decision')
        require(passed or index==len(stages),'continued failed stage')
        reduced_stages.append({'stage':index,'passed':passed,'checks':reduced})
    completed=stages[-1]['update']; require(type(result['updates']) is int and result['updates']==completed and pre['update']==completed,'update count')
    history=rows(root/'training.jsonl'); require(len(history)==completed,'incomplete training history')
    for update,row in enumerate(history,1):
        stage=1 if update<=1000 else 2 if update<=3000 else 3
        horizon_sets=([0],[0,1,2],[0,1,2,4,8]); offset=update-(0,1000,3000)[stage-1]-1
        require(all(type(row[k]) is int for k in ('update','stage','R')) and row['update']==update and row['stage']==stage and row['R']==horizon_sets[stage-1][offset%len(horizon_sets[stage-1])],'training chronology')
        require(number(row['loss']) and row['loss']>=0 and number(row['gradient_norm']) and row['gradient_norm']>=0,'training finite metrics')
    passed=len(stages)==3 and stages[-1]['passed']
    require(result['status']==('competence_passed' if passed else 'competence_failed') and result['heldout_opened'] is passed and pre['eligible'] is passed,'terminal competence status')
    output={'status':result['status'],'updates_verified':completed,'stages':reduced_stages,'conditions':[],
            'compact_files_verified':checked,'unopened_payloads':omitted,'input_result_sha256':result_sha,
            'limitations':['one discovery seed','no independent raw-state or model replay','scheduler and W&B stream audit separate','no capacity, erasure, or Floquet conclusion']}
    if not passed:
        require(not stages[-1]['passed'] and result['evaluation']==[] and not (root/'evaluation.jsonl').exists(),'invalid failed outcome/heldout contamination')
        return output
    require('evaluation.jsonl' in inventory,'evaluation inventory binding')
    data=rows(root/'evaluation.jsonl'); require(len(data)==len(GRID)*4096 and len(result['evaluation'])==len(GRID),'incomplete heldout evaluation')
    for j,(r,q) in enumerate(GRID):
        name=f'R{r}-q{q}'
        require({f'{name}-design.json',f'{name}-summary.json',f'{name}-states.npz'}<=set(inventory),'heldout inventory binding')
        designs(read(root/f'{name}-design.json'),1024,r,q)
        item=reduce_rows(data[j*4096:(j+1)*4096],1024,name,True)
        require(result['evaluation'][j]==read(root/f'{name}-summary.json'),'condition summary file')
        compare_summary(result['evaluation'][j],item,True)
        item.update(R=r,q=q,body_steps=12*r*q); output['conditions'].append(item)
    by_name={x['condition']:x for x in output['conditions']}
    d16=by_name['R16-q1']['paired_error_difference']; d2=by_name['R2-q1']['paired_error_difference']
    output['contrasts']={'delta16':d16,'delta2':d2,'difference_in_differences':d16-d2}
    return output

def plots(result,out):
    if not result['conditions']: return {}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    primary=[x for x in result['conditions'] if x['q']==1]
    dwell=sorted([x for x in result['conditions'] if x['body_steps']==384],key=lambda x:x['q'])
    for arm,label in [('0','Periodic'),('1','Matched order control')]:
        axes[0].plot([x['R'] for x in primary],[x['arm_accuracy'][arm] for x in primary],'o-',label=label)
        axes[1].plot([x['q'] for x in dwell],[x['arm_accuracy'][arm] for x in dwell],'o-',label=label)
    axes[0].set(xlabel='Motif repetitions R (q=1)',ylabel='Native bit accuracy',title='Increasing retention horizon')
    axes[1].set(xlabel='Symbol dwell q',ylabel='Native bit accuracy',title='Equal body length: 384 updates')
    for ax in axes: ax.set_ylim(-.02,1.02); ax.axhline(.5,color='gray',ls=':'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('One-bit retention · discovery seed 0 · paired cues and arms\nDescriptive results; no capacity or Floquet inference',fontsize=10)
    path=out/'onebit-retention.png'; fig.savefig(path,dpi=180); plt.close(fig)
    return {path.name:sha(path)}

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--result-sha256',required=True);p.add_argument('--run-id',required=True);a=p.parse_args(argv)
    a.output.mkdir(parents=True,exist_ok=False)
    from representation_collapse.reporting import ExperimentReporter
    with ExperimentReporter(a.output/'reporting',project='representation-collapse',mode='offline',run_id=a.run_id,
                            config={'protocol':'onebit-independent-reduction-v1'},identities={'result_sha256':a.result_sha256}) as reporter:
        result=audit(a.input,a.result_sha256); figures=plots(result,a.output)
        (a.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        (a.output/'plot-manifest.json').write_text(json.dumps(figures,indent=2)+'\n')
        reporter.log({'verification/updates':result['updates_verified'],'verification/conditions':len(result['conditions'])},step=0)
        reporter.log_manifest('onebit-independent-results',metadata={'payload_upload':False},identities={'results_sha256':sha(a.output/'results.json')})
    return 0

if __name__=='__main__': raise SystemExit(main())
