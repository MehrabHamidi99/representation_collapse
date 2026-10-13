"""External 600-second watchdog for the admitted cached-forward-v2 engine.

No production run is authorized by this file. Freeze schema:
kind=cached_forward_v2_root_admission, owner=root coordinator,
execution_authorized=true, training_authorized=false, scientific_acceptance=false,
native_core_calls=736, wall_seconds=600, retries=0, absolute output/stop_path,
source_commit, config_sha256, source_files (relative path->SHA256),
implementation_review/fixture_receipt/offline_smoke (path/SHA256/bytes),
launcher_argv (exact argv including absolute executable/script paths).
The root supervisor starts the monotonic clock before spawning this launcher,
binds started_monotonic in its admission and exact argv, and independently
enforces the same 600-second deadline. Child termination begins at 598 seconds
to reserve two seconds for the parent receipt, without a termination grace.
The independent implementation receipt binds identical source_files/config;
see verify_admission in the engine. No receipt is generated automatically.
"""
import time

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','config','freeze','output'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--started-monotonic',required=True,type=float)
    args=parser.parse_args()
    started=args.started_monotonic
    if not 0<=time.monotonic()-started<598:raise ValueError('root shared clock invalid or exhausted')
    freeze=json.loads(args.freeze.read_text())
    actual=[str(Path(sys.executable).absolute()),str(Path(__file__).resolve()),*sys.argv[1:]]
    if freeze.get('launcher_argv')!=actual or freeze.get('output')!=str(args.output.resolve()):
        raise ValueError('external freeze must bind exact launcher argv and unique output')
    if freeze.get('execution_authorized') is not True or freeze.get('wall_seconds')!=600:
        raise ValueError('external numerical admission required')
    if freeze.get('started_monotonic') != started:raise ValueError('root admission must bind shared clock')
    if args.output.exists():raise ValueError('existing output refuses retry/resume')
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):env[key]='1'
    env.update(PYTHONPATH=str(args.root.resolve()/'src'),PYTHONDONTWRITEBYTECODE='1',RC_CACHED_FORWARD_LAUNCH='1')
    # Durable offline artifact bytes live in output; SDK operational cache/data stay alongside.
    sibling=args.output.resolve().parent/(args.output.name+'-reporting-support')
    if sibling.exists():raise ValueError('reporting support path must also be unique')
    sibling.mkdir(parents=True)
    env.update(WANDB_CACHE_DIR=str(sibling/'cache'),WANDB_CONFIG_DIR=str(sibling/'config'),WANDB_DATA_DIR=str(sibling/'data'))
    command=[sys.executable,'-m','representation_collapse.cached_forward_calibration',
        '--root',str(args.root.resolve()),'--config',str(args.config.resolve()),'--freeze',str(args.freeze.resolve()),
        '--output',str(args.output.resolve()),'--started-monotonic',repr(started)]
    if time.monotonic()-started>=598:raise ValueError('root clock exhausted before child spawn')
    process=subprocess.Popen(command,cwd=args.root.resolve(),env=env,start_new_session=True)
    timed_out=False
    try:
        remaining=598-(time.monotonic()-started)
        if remaining<=0:raise subprocess.TimeoutExpired(command,598)
        code=process.wait(timeout=remaining)
    except (subprocess.TimeoutExpired,KeyboardInterrupt):
        timed_out=True
        # The scientific process cannot continue beyond its external deadline.
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        try:process.wait(timeout=max(.001,600-(time.monotonic()-started)))
        except subprocess.TimeoutExpired:pass  # outer root supervisor owns the hard cap
        code=124
    late=time.monotonic()-started>=600
    if late:code=124
    receipt={'kind':'cached_forward_v2_launcher_receipt','started_monotonic':started,
        'elapsed_seconds':time.monotonic()-started,'pid':process.pid,'exit_code':code,
        'state':'completed' if code==0 and not late else 'incomplete',
        'parent_receipt_deadline_censored':late,'child_deadline_seconds':598,
        'external_deadline_reached':timed_out,'output':str(args.output.resolve()),'scientific_acceptance':False}
    path=sibling/'launcher-receipt.json';tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(receipt,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)
    if time.monotonic()-started>=600:return 124
    return code


if __name__=='__main__':sys.exit(main())
