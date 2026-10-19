#!/usr/bin/env python3
"""Read-only Mila long-fit monitor; no scheduler mutations or provider calls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import time
import uuid

REPO = Path(__file__).resolve().parents[2]
SCRATCH = '/network/scratch/m/mehrab.hamidi/representation-collapse/runs/figure10-long'
POLL_SECONDS = 300
TERMINAL = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','PREEMPTED','NODE_FAIL','BOOT_FAIL','DEADLINE'}


def unique(pairs):
    result = {}
    for key,value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def decode(raw):
    def reject(value):
        raise ValueError('nonfinite JSON')
    return json.loads(raw,object_pairs_hook=unique,parse_constant=reject)


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def last_complete_line(raw, *, truncated_start=False):
    """A concurrent incomplete final append is never treated as an event."""
    lines=raw.splitlines(keepends=True)
    if truncated_start and lines:
        lines=lines[1:]
    complete=[line for line in lines if line.endswith(b'\n') and line.strip()]
    return decode(complete[-1]) if complete else None


REMOTE = r'''
import getpass,json,os,stat,subprocess,sys
from pathlib import Path
payload=json.loads(sys.argv[1]);root=Path(payload['root']);job=payload['job_id']
def read_json(path,limit=2000000):
 try:
  st=path.lstat()
  if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1 or st.st_size>limit: return {'read_status':'unsafe_or_oversized'}
  return json.loads(path.read_bytes())
 except FileNotFoundError:return None
 except (OSError,ValueError):return {'read_status':'incomplete_or_unavailable'}
def query(argv):
 r=subprocess.run(argv,capture_output=True,text=True,timeout=10,stdin=subprocess.DEVNULL)
 return {'returncode':r.returncode,'stdout':r.stdout[:8192]}
sacct=query(['sacct','-j',job,'-n','-P','--format=JobIDRaw,User%64,State,ExitCode,Elapsed,AllocCPUS,ReqMem'])
squeue=query(['squeue','-j',job,'-h','-o','%i|%u|%T|%R'])
sched={'state':'UNKNOWN','job_id':job,'accounting_available':sacct['returncode']==0}
for line in sacct['stdout'].splitlines():
 fields=line.split('|')
 if fields[0]==job:
  if fields[1]!=getpass.getuser():raise ValueError('job is not owned by SSH user')
  sched.update(state=fields[2].split()[0].rstrip('+'),exit_code=fields[3],elapsed=fields[4],cpus=fields[5],requested_memory=fields[6])
for line in squeue['stdout'].splitlines():
 fields=line.split('|')
 if fields[0]==job:
  if fields[1]!=getpass.getuser():raise ValueError('queued job is not owned by SSH user')
  sched.update(state=fields[2],reason=fields[3])
last=None
try:
 p=root/'fit-events.jsonl';st=p.lstat()
 if stat.S_ISREG(st.st_mode) and st.st_nlink==1:
  with p.open('rb') as stream:
   start=max(0,st.st_size-65536);stream.seek(start);raw=stream.read(65536)
  lines=raw.splitlines(keepends=True)
  if start and lines:lines=lines[1:]
  lines=[line for line in lines if line.endswith(b'\n') and line.strip()]
  if lines:last=json.loads(lines[-1])
except (OSError,ValueError):pass
validation=None
paths=sorted(root.glob('validation-[0-9][0-9][0-9].json'))
if paths:
 value=read_json(paths[-1])
 if isinstance(value,dict):validation={'filename':paths[-1].name,'record':value.get('record'),'rows_count':len(value.get('rows',[])),'read_status':value.get('read_status')}
terminal=read_json(root/'manifest.json')
if isinstance(terminal,dict):
 terminal={key:terminal[key] for key in ('status','stop_reason','error_type','attempted_updates','completed_updates','optimizer_steps_confirmed','validation_epochs_completed','selected','selected_validation_f1','finalized','test_payload_opened','scientific_acceptance','prefix_exact_loss_matches') if key in terminal}
print(json.dumps({'job_id':job,'scheduler':sched,'last_event':last,'validation':validation,
 'selected':read_json(root/'selection'/'selected.json',65536),
 'prefix_parity':read_json(root/'prefix-parity.json',65536),
 'prefix_failure':read_json(root/'prefix-parity-failure.json',65536),'terminal':terminal},allow_nan=False))
'''


def validate_launch(launch,job_id):
    if not re.fullmatch(r'[0-9]{1,12}',str(job_id)) or int(job_id)<=0:
        raise ValueError('one exact positive scheduler job ID required')
    run=launch.get('run_id','')
    match=re.fullmatch(r'long-(\d{8}T\d{6}Z)-[0-9a-f]{8}',run)
    if not match or launch.get('output_remote')!=SCRATCH+'/'+run:
        raise ValueError('long-run output must be exactly project scratch/run ID')
    if launch.get('cluster','mila')!='mila' or ('job_id' in launch and str(launch['job_id'])!=str(job_id)):
        raise ValueError('fixed Mila cluster and matching launch job required')
    if not re.fullmatch(r'[0-9a-f]{40}',str(launch.get('source_commit',''))):
        raise ValueError('immutable launch source revision required')
    if not re.fullmatch(r'[0-9a-f]{64}',str(launch.get('admission_sha256',''))):
        raise ValueError('launch admission pin required')
    launched=datetime.strptime(match[1],'%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc).timestamp()
    return launched+49*3600


def validate_optional_paths(evidence,inbox):
    if (evidence is None)!=(inbox is None):
        raise ValueError('evidence root and approved task inbox must be supplied together')
    if evidence is None:
        return None,None
    evidence,inbox=Path(evidence),Path(inbox)
    base=REPO/'runs/evidence/qwen-local-20261014'
    approved=REPO/'runs/qwen/local-continuous-20261014T1650Y/tasks'
    if (not evidence.is_dir() or evidence.is_symlink() or not evidence.resolve().is_relative_to(base.resolve())
            or not inbox.is_dir() or inbox.is_symlink() or inbox.resolve()!=approved.resolve()):
        raise ValueError('existing approved project evidence/inbox directories required')
    return evidence,inbox


def collect(launch,job_id):
    remote=shlex.join(['python3','-c',REMOTE,json.dumps({'root':launch['output_remote'],'job_id':str(job_id)},separators=(',',':'))])
    command=['ssh','-T','-o','BatchMode=yes','-o','NumberOfPasswordPrompts=0','-o','ConnectionAttempts=1',
             '-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=1','mila',remote]
    completed=subprocess.run(command,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=40)
    if completed.returncode:
        raise ConnectionError('BatchMode read-only Mila probe failed: exit '+str(completed.returncode))
    if len(completed.stdout)>200000:
        raise ValueError('bounded monitor response exceeded')
    result=decode(completed.stdout)
    if result.get('job_id')!=str(job_id) or result.get('scheduler',{}).get('job_id')!=str(job_id):
        raise ValueError('probe returned another job')
    return result


def milestone_keys(snapshot,seen):
    keys=[]
    parity=snapshot.get('prefix_parity') or {}
    if parity.get('status')=='pass' and parity.get('exact_loss_matches')==256 and 'parity-pass' not in seen:
        keys.append('parity-pass')
    selected=snapshot.get('selected') or {}
    epoch=selected.get('last_epoch',0)
    if type(epoch) is int and 0<=epoch<=200:
        keys += [f'epoch-{n:03d}' for n in range(10,epoch+1,10) if f'epoch-{n:03d}' not in seen]
    terminal=snapshot.get('terminal') or {}
    state=snapshot.get('scheduler',{}).get('state')
    if ('terminal' not in seen and ((snapshot.get('prefix_failure') or {}).get('status')=='failed' or terminal.get('status') in {'failed','fit_complete','fit_stopped'} or state in TERMINAL)):
        keys.append('terminal')
    return keys


def compact_summary(snapshot):
    selected=snapshot.get('selected') or {}
    validation=snapshot.get('validation') or {}
    record=validation.get('record') or {}
    event=(snapshot.get('last_event') or {}).get('record') or {}
    bounded=lambda value:value[:120] if isinstance(value,str) else None
    terminal=snapshot.get('terminal') or {}
    return {'job_id':snapshot['job_id'],'scheduler_state':snapshot.get('scheduler',{}).get('state'),
            'scheduler_reason':bounded(snapshot.get('scheduler',{}).get('reason')),
            'completed_candidate_epochs':selected.get('last_epoch',0),'selected_epoch':(selected.get('selected') or {}).get('epoch'),
            'selected_validation_f1':selected.get('best_f1'),'latest_validation_f1':record.get('validation_mean_f1'),
            'attempted_updates':event.get('attempted_updates'),'completed_updates':event.get('completed_updates'),
            'prefix_status':(snapshot.get('prefix_parity') or {}).get('status'),
            'prefix_failure_status':bounded((snapshot.get('prefix_failure') or {}).get('status')),
            'terminal_status':terminal.get('status'),'terminal_stop_reason':bounded(terminal.get('stop_reason')),
            'terminal_error_type':bounded(terminal.get('error_type')),
            'evidence_status':'provisional/unaccepted scientific evidence','proposal_only':True}


def atomic_json(path,value):
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)


def write_new(path,value):
    with path.open('xb') as stream:
        stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)


class Monitor:
    def __init__(self,launch,job_id,output,*,launch_sha256,deadline,evidence=None,inbox=None,clock=time.time):
        self.launch,self.job_id,self.output=launch,str(job_id),Path(output)
        self.deadline,self.evidence,self.inbox,self.clock=deadline,evidence,inbox,clock
        self.output.mkdir(mode=0o700,parents=False,exist_ok=False)
        self.directory_identity=self.output.stat().st_dev,self.output.stat().st_ino
        self.token=uuid.uuid4().hex;self.pid=os.getpid();self.sequence=0;self.errors=0;self.seen=set()
        write_new(self.output/'owner.json',{'pid':self.pid,'token':self.token,'job_id':self.job_id,
            'run_id':launch['run_id'],'launch_sha256':launch_sha256,'deadline_unix':deadline,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'created_unix':clock()})
        self.journal=(self.output/'journal.jsonl').open('xb')
        self.end_reason=None;self.last_snapshot=None

    def check(self):
        current=self.output.lstat()
        if os.getpid()!=self.pid or (current.st_dev,current.st_ino)!=self.directory_identity:
            raise RuntimeError('monitor ownership changed')
        if decode((self.output/'owner.json').read_bytes()).get('token')!=self.token:
            raise RuntimeError('monitor owner token changed')

    def record(self,event,**fields):
        self.check()
        self.journal.write(canonical({'sequence':self.sequence,'time_unix':self.clock(),'event':event,**fields}))
        self.journal.flush();os.fsync(self.journal.fileno());self.sequence+=1

    def publish(self,key,snapshot):
        if self.evidence is None:
            return
        task_id=f'fig10long-{self.job_id}-{key}'
        task_path=self.inbox/(task_id+'.json');evidence_path=self.evidence/(task_id+'.json')
        if os.path.lexists(task_path) or os.path.lexists(evidence_path):
            self.record('milestone_already_published',milestone=key);return
        summary=compact_summary(snapshot)
        text=('Proposal-only scientific review; this is provisional/unaccepted scientific evidence. '
              'Review this new long-fit milestone for interpretation risks and one scientifically justified next analysis. '
              'Do not perform actions or accept claims. Milestone: '+key+'. Snapshot: '+json.dumps(summary,sort_keys=True,separators=(',',':')))
        if len(text)>1500:
            raise ValueError('bounded task exceeded')
        write_new(evidence_path,{'kind':'figure10_long_monitor_provisional_milestone_v1','milestone':key,'summary':summary})
        write_new(task_path,{'task_id':task_id,'task':text})
        self.record('proposal_task_published',task_id=task_id,milestone=key)

    def run(self,*,fetch=collect,sleep=time.sleep):
        try:
            while True:
                self.check()
                if os.path.lexists(self.output/'STOP'):
                    self.record('monitor_stopped_by_owner');self.end_reason='owner_stop';return self.end_reason
                if self.clock()>=self.deadline:
                    self.record('monitor_deadline');self.end_reason='deadline';return self.end_reason
                try:
                    snapshot=fetch(self.launch,self.job_id)
                except (ConnectionError,subprocess.TimeoutExpired,OSError,ValueError) as exc:
                    self.errors+=1;delay=min(1800,POLL_SECONDS*2**min(self.errors-1,4))
                    self.record('probe_error',error_type=type(exc).__name__,consecutive_errors=self.errors,next_poll_seconds=delay)
                    atomic_json(self.output/'status.json',{'time_unix':self.clock(),'monitor_pid':self.pid,'state':'connection_or_probe_error',
                        'error_type':type(exc).__name__,'consecutive_errors':self.errors,'provisional':True})
                else:
                    self.errors=0
                    self.last_snapshot=snapshot
                    for key in milestone_keys(snapshot,self.seen):
                        self.publish(key,snapshot);self.seen.add(key)
                    self.record('poll',summary=compact_summary(snapshot))
                    atomic_json(self.output/'status.json',{'time_unix':self.clock(),'monitor_pid':self.pid,'state':'monitoring',
                        'snapshot':snapshot,'provisional':True,'scientific_acceptance':False})
                    if snapshot.get('scheduler',{}).get('state') in TERMINAL:
                        self.record('scheduler_terminal',state=snapshot['scheduler']['state']);self.end_reason='scheduler_terminal';return self.end_reason
                    delay=POLL_SECONDS
                atomic_json(self.output/'heartbeat.json',{'time_unix':self.clock(),'pid':self.pid,'next_poll_seconds':delay,
                    'deadline_unix':self.deadline,'health_poll_is_model_call':False})
                # Short interruptible slices keep monitor STOP responsive.
                until=min(self.clock()+delay,self.deadline)
                while self.clock()<until and not os.path.lexists(self.output/'STOP'):
                    sleep(min(5,until-self.clock()))
        finally:
            try:
                try:
                    self.check()
                except (RuntimeError,OSError,ValueError):
                    # An ownership failure never grants cleanup authority over
                    # a replacement namespace. Only close our existing fd.
                    pass
                else:
                    atomic_json(self.output/'status.json',{'time_unix':self.clock(),'monitor_pid':self.pid,'state':'stopped',
                        'reason':self.end_reason or 'failed','snapshot':self.last_snapshot,'consecutive_errors':self.errors,
                        'provisional':True,'scientific_acceptance':False})
                    atomic_json(self.output/'heartbeat.json',{'time_unix':self.clock(),'pid':self.pid,'state':'stopped',
                        'deadline_unix':self.deadline,'health_poll_is_model_call':False})
            finally:
                self.journal.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-json',type=Path,required=True)
    parser.add_argument('--job-id',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--deadline-unix',type=float)
    parser.add_argument('--evidence-root',type=Path)
    parser.add_argument('--task-inbox',type=Path)
    args=parser.parse_args(argv)
    info=args.launch_json.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_size>2_000_000:
        raise ValueError('bounded regular launch JSON required')
    raw=args.launch_json.read_bytes()
    launch=decode(raw);maximum=validate_launch(launch,args.job_id)
    deadline=maximum if args.deadline_unix is None else args.deadline_unix
    if not math.isfinite(deadline) or deadline>maximum or deadline<=time.time():
        raise ValueError('live monitor deadline must be at most launch plus49hours')
    evidence,inbox=validate_optional_paths(args.evidence_root,args.task_inbox)
    Monitor(launch,args.job_id,args.output,launch_sha256=hashlib.sha256(raw).hexdigest(),deadline=deadline,
            evidence=evidence,inbox=inbox).run()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
