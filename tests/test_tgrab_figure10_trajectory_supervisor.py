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
    def probe(self, alter=None):
        job = dict(account='mila', partition='main-cpu', qos='normal', nodes=1,
                   tasks=1, cpus_on_node='1', cpus_per_task='1', time_limit='00:10:00',
                   gpu_allocation='0', memory_max_bytes=2147483648, job_id=None, node_list=None)
        acceptance = dict(kind='figure10_selected_evaluation_execution_acceptance_v2',status='accepted',
                          terminal_replay_verified=True,test_rows_verified=True,
                          evaluation_result_path='/evaluation/evaluation-result.json',evaluation_result_sha256='a'*64,
                          trajectory_path='/evaluation/test-state-trajectory.f32le',trajectory_sha256='b'*64)
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
            return json.dumps(acceptance).encode() if path=='/acceptance' else b'{}'
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


if __name__=='__main__': unittest.main()
