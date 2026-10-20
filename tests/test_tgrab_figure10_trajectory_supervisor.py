import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('trajectory_supervisor', ROOT / 'code/ops/tgrab_figure10_trajectory_supervisor.py')
SUPERVISOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUPERVISOR)


class TrajectorySupervisorTests(unittest.TestCase):
    def probe(self, alter=None, *, v3=False, result_alter=None):
        job = dict(account='mila', partition='main-cpu', qos='normal', nodes=1,
                   tasks=1, cpus_on_node='1', cpus_per_task='1', time_limit='00:10:00',
                   gpu_allocation='0', memory_max_bytes=2147483648, job_id=None, node_list=None)
        acceptance = dict(kind='figure10_selected_evaluation_execution_acceptance_v2',status='accepted',
                          terminal_replay_verified=True,test_rows_verified=True,
                          evaluation_result_path='/evaluation/evaluation-result.json',evaluation_result_sha256='a'*64,
                          trajectory_path='/evaluation/test-state-trajectory.f32le',trajectory_sha256='b'*64)
        if v3:
            acceptance.update(kind=SUPERVISOR.PRIMITIVE_V2_ACCEPTANCE,checkpoint_format='primitive_v2',
                executed_long_contract_sha256=SUPERVISOR.LONG_CONTRACT_SHA256,
                candidate_manifest_sha256='f'*64,selected_pointer_sha256='1'*64)
        else:
            acceptance.update(checkpoint_sha256='f'*64,final_head_sha256='1'*64)
        result = {k:v for k,v in acceptance.items() if k in ('checkpoint_sha256','candidate_manifest_sha256',
            'checkpoint_format','executed_long_contract_sha256','selected_pointer_sha256')}
        result['state_trajectory'] = dict(filename='test-state-trajectory.f32le',sha256='b'*64)
        if result_alter: result_alter(result)
        args = ['--run-id','trajectory01']
        for key in ('evaluation_result_path','evaluation_result_sha256','trajectory_path','trajectory_sha256'):
            flag = {'evaluation_result_path':'--evaluation-result','trajectory_path':'--trajectory'}.get(key,'--'+key.replace('_','-'))
            args += [flag,acceptance[key]]
        d = dict(schema='tgrab-figure10-trajectory-admission-v1',protocol_id=SUPERVISOR.PROTOCOL_ID,
                 source_commit='c'*40,run_id='trajectory01',job=job,work_seconds=540,
                 outer_wall_seconds=570,canary_args=args,
                 environment_receipt={'path':'/environment','sha256':'d'*64},
                 evaluation_acceptance={'path':'/acceptance','sha256':'e'*64})
        if alter: alter(d,acceptance)
        reads = []
        def read(path, sha):
            reads.append(path)
            return json.dumps(acceptance if path=='/acceptance' else result if path=='/evaluation/evaluation-result.json' else {}).encode()
        def fail(message): raise RuntimeError(message)
        h = SimpleNamespace(fail=fail,regular_bytes=read,_require_arg_pair=SUPERVISOR.helper()._require_arg_pair)
        actual = {**job,'job_id':'123','node_list':'cpu01'}
        with patch.dict(os.environ,{'RC_JOB_WALL_SECONDS':'570'}):
            SUPERVISOR.validate_admission(d,h,actual=actual,memory={'effective_limit_bytes':2147483648})
        return reads

    def test_admitted_analysis_never_opens_numerical_state_before_reporter(self):
        reads = self.probe()
        self.assertIn('/evaluation/evaluation-result.json',reads)
        self.assertNotIn('/evaluation/test-state-trajectory.f32le',reads)

    def test_missing_replay_acceptance_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'acceptance required'):
            self.probe(lambda d,a:a.update(terminal_replay_verified=False))

    def test_unaccepted_rows_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'acceptance required'):
            self.probe(lambda d,a:a.update(test_rows_verified=False))

    def test_source_closure_contains_analysis_protocol(self):
        self.assertIn('doc/auto/figure10-trajectory-diagnostics-2026-10-19.md',SUPERVISOR.source_closure(ROOT))

    def test_primitive_v2_receipt_never_opens_state_before_reporter(self):
        reads = self.probe(v3=True)
        self.assertNotIn('/evaluation/test-state-trajectory.f32le',reads)

    def test_distinct_versions_reject_mixed_unknown_missing_or_tampered_identity(self):
        cases = [
            (False, lambda d,a:a.update(kind='unknown')),
            (False, lambda d,a:a.update(selected_pointer_sha256='1'*64)),
            (False, lambda d,a:a.pop('final_head_sha256')),
            (True, lambda d,a:a.update(final_head_sha256='1'*64)),
            (True, lambda d,a:a.update(executed_long_contract_sha256='0'*64)),
            (True, lambda d,a:a.pop('checkpoint_format')),
            (True, lambda d,a:a.update(selected_pointer_sha256='invalid')),
            (True, lambda d,a:a.pop('candidate_manifest_sha256')),
            (True, lambda d,a:a.pop('terminal_replay_verified')),
            (True, lambda d,a:a.update(test_rows_verified=1)),
            (True, lambda d,a:a.update(trajectory_sha256='invalid')),
            (True, lambda d,a:a.update(evaluation_result_sha256='invalid')),
            (True, lambda d,a:d['canary_args'].__setitem__(d['canary_args'].index('--trajectory-sha256')+1,'0'*64)),
            (True, lambda d,a:d['canary_args'].__setitem__(d['canary_args'].index('--evaluation-result')+1,'/other/result.json')),
        ]
        for i,(v3,alter) in enumerate(cases):
            with self.subTest(case=i),self.assertRaises((RuntimeError,SystemExit)):
                self.probe(alter,v3=v3)

    def test_pinned_result_must_bind_accepted_checkpoint_and_trajectory(self):
        for v3 in (False,True):
            for key in ('sha256','filename'):
                with self.subTest(v3=v3,key=key),self.assertRaises(RuntimeError):
                    self.probe(v3=v3,result_alter=lambda r:r['state_trajectory'].update({key:'foreign'}))
            key = 'candidate_manifest_sha256' if v3 else 'checkpoint_sha256'
            with self.subTest(v3=v3,key=key),self.assertRaises(RuntimeError):
                self.probe(v3=v3,result_alter=lambda r:r.update({key:'0'*64}))


if __name__=='__main__': unittest.main()
