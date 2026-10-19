import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('long_supervisor',ROOT/'code/ops/tgrab_figure10_long_supervisor.py')
LONG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LONG)


class LongSupervisorTests(unittest.TestCase):
    def probe(self, alter=None):
        job = dict(account='mila',partition='main-cpu',qos='normal',nodes=1,tasks=1,
                   cpus_on_node='1',cpus_per_task='1',time_limit='2-00:00:00',gpu_allocation='0',
                   memory_max_bytes=2147483648,job_id=None,node_list=None)
        d = dict(schema='tgrab-figure10-long-admission-v2',protocol_id=LONG.PROTOCOL_ID,
                 source_commit='a'*40,run_id='long01',job=job,work_seconds=172680,
                 outer_wall_seconds=172770,canary_args=['--run-id','long01','--source-commit','a'*40,'--runtime-seconds','172680'])
        for key in ('environment_receipt','node2a_receipt','conversion_receipt','long_contract','prefix_updates'):
            d[key] = {'path':'/'+key,'sha256':'b'*64}
            flag='--'+key.replace('_','-')
            d['canary_args'] += [flag,'/'+key,flag+'-sha256','b'*64]
        actual={**job,'job_id':'123','node_list':'cpu01'}
        if alter: alter(d)
        def fail(message): raise RuntimeError(message)
        def pair(args,flag,value):
            if args[args.index(flag)+1] != value: fail('argument pin mismatch')
        h=SimpleNamespace(fail=fail,_require_arg_pair=pair,regular_bytes=lambda p,s:b'{}')
        with patch.dict(os.environ,{'RC_JOB_WALL_SECONDS':'172770'}):
            return LONG.validate_admission(d,h,actual=actual,memory={'effective_limit_bytes':2147483648})

    def test_registered_two_day_cpu_envelope_passes(self):
        actual,_=self.probe()
        self.assertEqual(actual['time_limit'],'2-00:00:00')

    def test_one_hour_runtime_cannot_masquerade_as_long_contract(self):
        with self.assertRaises(RuntimeError):
            self.probe(lambda d:d['job'].update(time_limit='01:00:00'))

    def test_prefix_reference_hash_is_bound_to_arguments(self):
        with self.assertRaisesRegex(RuntimeError,'argument pin mismatch'):
            self.probe(lambda d:d['prefix_updates'].update(sha256='c'*64))

    def test_source_closure_includes_long_executed_contract(self):
        closure=LONG.source_closure(ROOT)
        self.assertIn('configs/graph/tgrab-long-fit-v2.json',closure)
        self.assertIn('doc/auto/figure10-long-fit-v2-2026-10-19.md',closure)


if __name__=='__main__': unittest.main()
