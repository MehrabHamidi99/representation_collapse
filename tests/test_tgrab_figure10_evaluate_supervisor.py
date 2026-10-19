import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('eval_supervisor', ROOT / 'code/ops/tgrab_figure10_evaluate_supervisor.py')
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


class EvaluationSupervisorTests(unittest.TestCase):
    def probe(self, change=None):
        job = dict(account='mila', partition='main-cpu', qos='normal', nodes=1,
                   tasks=1, cpus_on_node='1', cpus_per_task='1', time_limit='00:10:00',
                   gpu_allocation='0', memory_max_bytes=2147483648, job_id=None, node_list=None)
        d = dict(schema='tgrab-figure10-evaluation-admission-v2', protocol_id=EVAL.PROTOCOL_ID,
                 source_commit='a'*40, run_id='eval01', job=job, work_seconds=540,
                 outer_wall_seconds=570, canary_args=['--run-id', 'eval01'])
        acceptance = dict(kind='figure10_fit_execution_acceptance_v1', status='accepted', test_accessed=False,
                          fit_run_id='fit01', fit_attempt_id='fit01', fit_admission_sha256='b'*64,
                          fit_summary_sha256='c'*64, final_head_sha256='d'*64, fit_run_dir='/fit')
        acceptance['artifact_sha256'] = {'fit-summary.json':'c'*64, 'manifest.json':'e'*64,
                                        'wandb/reporting-receipt.json':'f'*64,
                                        'checkpoint-ledger/selection-head.json':'d'*64}
        for key in ('environment_receipt', 'node2a_receipt', 'conversion_receipt'):
            d[key] = {'path':'/'+key, 'sha256':'1'*64}
            flag = '--'+key.replace('_','-')
            d['canary_args'] += [flag, '/'+key, flag+'-sha256', '1'*64]
        for key in ('fit_run_id','fit_attempt_id','fit_admission_sha256','fit_summary_sha256','final_head_sha256','fit_run_dir'):
            d['canary_args'] += ['--'+key.replace('_','-'), acceptance[key]]
        d['fit_acceptance'] = {'path':'/acceptance', 'sha256':'2'*64}
        if change: change(d, acceptance)
        def fail(message): raise RuntimeError(message)
        h = SimpleNamespace(fail=fail, _require_arg_pair=EVAL.helper()._require_arg_pair,
                            regular_bytes=lambda path, sha: json.dumps(acceptance).encode() if path=='/acceptance' else b'{}')
        actual = {**job, 'job_id':'123', 'node_list':'cpu01'}
        with patch.dict(os.environ, {'RC_JOB_WALL_SECONDS':'570'}):
            return EVAL.validate_admission(d,h,actual=actual,memory={'effective_limit_bytes':2147483648})

    def test_complete_bound_fit_acceptance_passes(self):
        actual,_ = self.probe()
        self.assertEqual(actual['job_id'],'123')

    def test_missing_artifacts_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'complete and consistent'):
            self.probe(lambda d,a:a.update(artifact_sha256={}))

    def test_different_final_head_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'complete and consistent'):
            self.probe(lambda d,a:a['artifact_sha256'].update({'checkpoint-ledger/selection-head.json':'e'*64}))

    def test_different_summary_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'complete and consistent'):
            self.probe(lambda d,a:a['artifact_sha256'].update({'fit-summary.json':'e'*64}))


if __name__=='__main__': unittest.main()
