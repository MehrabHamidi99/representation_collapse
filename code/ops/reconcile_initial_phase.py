#!/usr/bin/env python3
"""Accept a complete initial-phase roster only after offline and Slurm checks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('journal', 'scheduler', 'verification', 'amendments', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    if any(args.output.resolve() == getattr(args, name).resolve()
           for name in ('journal', 'scheduler', 'verification', 'amendments')):
        p.error('output must not overwrite an input evidence record')
    inputs = {name: json.loads(getattr(args, name).read_text())
              for name in ('journal', 'scheduler', 'verification', 'amendments')}
    checks = []

    def require(condition, name):
        checks.append({'check': name, 'passed': bool(condition)})
        if not condition:
            raise ValueError(name)

    accepted, error, jobs = False, None, []
    try:
        journal, scheduler, verification, amendments = (inputs[k] for k in ('journal', 'scheduler', 'verification', 'amendments'))
        sha = journal['source_commit']
        require(sha == scheduler['source_commit'] == verification['expected_source_sha'], 'matching full source revisions')
        require(set(journal['cells']) == {'0', '1', '2', '3', '4'}, 'exact five-cell roster')
        require(len({v['job_id'] for v in journal['cells'].values()}) == 5, 'unique actual scheduler IDs')
        roster = {'1': ('gru', 'categorical', 'narval'), '2': ('lstm', 'categorical', 'rorqual'),
                  '3': ('gru', 'continuous', 'narval'), '4': ('lstm', 'continuous', 'rorqual')}
        require(journal['cells']['0']['kind'] == 'calibration' and journal['cells']['0']['cluster'] == 'narval', 'registered calibration cell')
        require(all((journal['cells'][k]['model'], journal['cells'][k]['task'], journal['cells'][k]['cluster']) == v and
                    journal['cells'][k]['seed'] == 0 and journal['cells'][k]['kind'] == 'pilot' for k, v in roster.items()),
                'registered four model task cluster seed0 cells')
        records = verification['records']
        require(len(records) == 6, 'exact six artifact records')
        require(all(r.get('artifact_verified') is True and r.get('blocked_reason') is None and
                    isinstance(r.get('checks'), list) and bool(r['checks']) and
                    all(c.get('passed') is True for c in r['checks']) for r in records),
                'every artifact record independently verified without failed checks')
        require(verification['summary']['artifact_directories'] == len(records) and
                verification['summary']['verified'] == sum(r['artifact_verified'] is True for r in records) == 6 and
                verification['summary']['blocked'] == 0, 'six independently verified artifact directories')
        require(verification['summary']['scientific_acceptance'] is False, 'scientific acceptance remains false')
        calibrations = [r for r in verification['records'] if r['requested_kind'] == 'calibration']
        require(len(calibrations) == 2 and {r['reduction']['kind']: r['reduction']['checks'] for r in calibrations} ==
                {'synthetic_measurement_calibration': 10, 'synthetic_mechanism_calibration': 36}, 'exact 10 and36 calibration checks')
        require(all(Path(r['directory']).parent.name == 'narval_' + str(journal['cells']['0']['job_id']) for r in calibrations),
                'calibration directories match registered CPU job')
        expected = {(v['model'], v['task'], v['seed']) for v in journal['cells'].values() if v['kind'] == 'pilot'}
        pilots = [r for r in verification['records'] if r['requested_kind'] == 'pilot']
        require(len(pilots) == 4 and {(r['reduction']['model'], r['reduction']['task'], r['reduction']['seed']) for r in pilots} == expected,
                'all four distinct model/task/seed cells verified')
        for change in ('walltime', 'gres'):
            updates = amendments[change]['result']['updates']
            require(len(updates) == 2 and {str(r['job_id']) for r in updates} ==
                    {str(journal['cells'][k]['job_id']) for k in ('2', '4')}, change + ' exact amended job IDs')
            require(amendments[change]['probe_exit_code'] == 0 and
                    all(r.get('update_exit_code') == 0 for r in updates),
                    change + ' amendments succeeded without resubmission')
        clusters = {c['cluster']: c for c in scheduler['clusters']}
        intervals = []
        for key, cell in journal['cells'].items():
            host, job = cell['cluster'], str(cell['job_id'])
            cluster = clusters[host]
            require(cluster['scheduler_timezone'] == 'UTC' and cluster['sacct']['returncode'] == 0, job + ' UTC accounting available')
            rows = [line.split('|') for line in cluster['sacct']['stdout'].splitlines()]
            row = next(r for r in rows if r[0] == job)
            require(row[1:3] == ['COMPLETED', '0:0'], job + ' scheduler completed with exit0')
            require(row[11] == ('00:10:00' if key == '0' else '00:15:00' if host == 'rorqual' else '01:00:00'), job + ' actual bounded walltime request')
            require(row[12] == ('def-ioannism_cpu' if key == '0' else 'def-ioannism_gpu') and row[13] == 'normal', job + ' actual account and QOS')
            tres = dict(item.split('=', 1) for item in row[4].split(','))
            require(tres.get('cpu') == ('1' if key == '0' else '4') and tres.get('mem') == ('2G' if key == '0' else '16G') and tres.get('node') == '1',
                    job + ' approved CPU memory node allocation')
            start, end = (datetime.fromisoformat(row[i]).replace(tzinfo=timezone.utc) for i in (8, 9))
            elapsed = int((end - start).total_seconds())
            require(0 < elapsed <= (600 if key == '0' else 900 if host == 'rorqual' else 3600), job + ' bounded elapsed allocation')
            if key != '0':
                typed = 'gres/gpu:a100' if host == 'narval' else 'gres/gpu:nvidia_h100_80gb_hbm3_3g.40gb'
                require(tres.get('gres/gpu') == tres.get(typed) == '1', job + ' actual approved one-device allocation')
                artifact = cluster['artifacts'][job]['pilot']
                provenance = artifact['provenance']
                require(str(provenance['slurm_job_id']) == job and provenance['git_revision'] == sha and not provenance['dirty_checkout'], job + ' source and scheduler provenance')
                name = provenance['accelerator']['name']
                require(('A100' in name and 'MIG' not in name) if host == 'narval' else 'MIG 3g.40gb' in name, job + ' actual hardware matches amended request')
                record = next(r for r in pilots if (r['reduction']['model'], r['reduction']['task']) == (cell['model'], cell['task']))
                require(Path(record['directory']).parent.name == host + '_' + job, job + ' verifier directory matches roster')
                intervals.extend([(start, 1), (end, -1)])
                if key in ('3', '4'):
                    predecessor = journal['cells'][str(int(key) - 2)]
                    require(cell['resources']['dependency'] == 'afterany:' + str(predecessor['job_id']), job + ' recorded dependency')
                jobs.append({'cell': key, 'cluster': host, 'job_id': int(job), 'model': cell['model'], 'task': cell['task'],
                             'start_utc': start.isoformat(), 'end_utc': end.isoformat(), 'elapsed_seconds': elapsed,
                             'alloc_tres': tres, 'accelerator': provenance['accelerator'],
                             'max_step_rss': [r[6] for r in rows if r[0].startswith(job + '.') and r[6]]})
        running = peak = 0
        for _, delta in sorted(intervals):
            running += delta
            peak = max(peak, running)
        require(peak <= 2 and running == 0, 'actual maximum concurrent GPU devices at most two')
        for after, before in (('3', '1'), ('4', '2')):
            a, b = (next(j for j in jobs if j['cell'] == k) for k in (after, before))
            require(a['start_utc'] >= b['end_utc'], after + ' successor started after predecessor terminated')
        accepted = True
    except Exception as e:
        error = str(e)
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'engineering_phase_complete': accepted,
              'scientific_acceptance': False, 'checks': checks, 'blocked_reason': error, 'jobs': jobs,
              'gpu_device_hours': sum(j['elapsed_seconds'] for j in jobs) / 3600,
              'gpu_device_hour_definition': 'one allocated device, including a MIG slice, times elapsed allocation; not full-GPU equivalents',
              'input_sha256': {k: hashlib.sha256(getattr(args, k).read_bytes()).hexdigest() for k in inputs}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps({'engineering_phase_complete': accepted, 'blocked_reason': error, 'gpu_device_hours': report['gpu_device_hours']}))
    if not accepted:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
