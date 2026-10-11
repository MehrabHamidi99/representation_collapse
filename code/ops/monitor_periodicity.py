#!/usr/bin/env python3
"""Read exact journaled checkpoint-periodicity jobs through cluv; never mutate."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex

from cluv.remote import Remote
from cluv_controller import configure_ssh


async def probe(cluster, cells):
    job_ids = [str(cell['job_id']) for cell in cells]
    code = '''import json,subprocess,os
from pathlib import Path
from datetime import datetime,timezone
cluster=CLUSTER
ids=IDS
d={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'cluster':cluster,'scheduler_timezone':'UTC'}
query_env=dict(os.environ,TZ='UTC')
commands={'sacct':['sacct','-j',','.join(ids),'-n','-P','--format=JobID,State,ExitCode,Elapsed,AllocTRES,ReqTRES,MaxRSS,CPUTimeRAW,Start,End,NodeList,Timelimit,Account,QOS'],
          'squeue':['squeue','-j',','.join(ids),'-h','-o','%i|%T|%R|%S']}
for name,argv in commands.items():
 r=subprocess.run(argv,capture_output=True,text=True,timeout=10,env=query_env)
 d[name]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
d['artifacts']={}
for job in ids:
 folder=Path('/scratch/mehrab/representation-collapse/logs/cluv')/(cluster+'_'+job)
 record={'path':str(folder)}
 for kind in ['periodicity']:
  p=folder/kind
  if p.is_dir():
   record[kind]={}
   for name in ['status','provenance','metrics']:
    f=p/(name+'.json')
    if f.exists():
     try: record[kind][name]=json.loads(f.read_text())
     except (OSError,ValueError) as e:record[kind][name+'_read_error']=type(e).__name__
   f=p/'history.json'
   if f.exists():
    try:record[kind]['optimization_steps_saved']=len(json.loads(f.read_text()))
    except (OSError,ValueError):pass
  f=folder/('slurm-'+job+'.out')
  if f.exists():record['slurm_tail']=f.read_text(errors='replace')[-1500:]
 d['artifacts'][job]=record
print(json.dumps(d,allow_nan=False))
'''.replace('CLUSTER', repr(cluster)).replace('IDS', repr(job_ids))
    result = await asyncio.wait_for(Remote(cluster).run(
        shlex.join(['bash', '-lc', shlex.join(['python3', '-c', code])]),
        display=False, warn=True, hide=True), 40)
    if result.returncode:
        return {'cluster': cluster, 'probe_exit_code': result.returncode,
                'probe_error': result.stderr[-2000:]}
    return json.loads(result.stdout)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    journal = json.loads(args.journal.read_text())
    if args.output.resolve() == args.journal.resolve():
        parser.error('monitor output must not overwrite submission journal')
    if journal['phase_id'] != 'periodicity-analysis-v1':
        parser.error('require a periodicity-analysis-v1 journal')
    groups = {}
    for cell in journal['cells'].values():
        if cell.get('job_id'):
            groups.setdefault(cell['cluster'], []).append(cell)
    if not groups or set(groups) - {'narval', 'rorqual'}:
        parser.error('only actual checkpoint-periodicity Narval/Rorqual jobs are supported')
    configure_ssh(batch=True)
    results = await asyncio.gather(*(probe(host, cells) for host, cells in groups.items()), return_exceptions=True)
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_commit': journal['source_commit'],
              'clusters': [r if isinstance(r, dict) else {'probe_error': str(r)} for r in results]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    asyncio.run(main())
