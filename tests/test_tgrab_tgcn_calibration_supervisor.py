"""Admission boundaries: no numerical runtime or cluster calls."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('tgcn_supervisor', ROOT/'code/ops/tgrab_tgcn_calibration_supervisor.py')
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)

class AdmissionTests(unittest.TestCase):
    def probe(self, alter=None, actual_alter=None, memory=2147483648):
        job = dict(account='mila',partition='main-cpu',qos='normal',nodes=1,tasks=1,
                   cpus_on_node='1',cpus_per_task='1',time_limit='00:10:00',gpu_allocation='0',
                   memory_max_bytes=2147483648,node_list=None)
        argv = ['--output','/output','--run-id','tgcn01']
        source='/network/scratch/m/mehrab.hamidi/representation-collapse/sources/tgcn-native-20261020/'
        for flag,(name,_) in M.SOURCE_INPUTS.items():argv += [flag,source+name]
        d=dict(schema='tgrab-tgcn-calibration-admission-v1',protocol_id=M.PROTOCOL_ID,
               source_commit='a'*40,run_id='tgcn01',job=job,work_seconds=540,outer_wall_seconds=570,
               canary_args=argv,environment_receipt=dict(path='/environment',sha256='b'*64),
               new_dataset_or_weight_reads=0,new_test_opener_calls=0)
        actual={**job,'job_id':'123','node_list':'cpu01'}
        if alter:alter(d)
        if actual_alter:actual_alter(actual)
        reads=[]
        def fail(msg):raise RuntimeError(msg)
        def read(path,digest):reads.append((str(path),digest));return b'{}'
        h=SimpleNamespace(fail=fail,regular_bytes=read,_require_arg_pair=M.helper()._require_arg_pair)
        with patch.dict(os.environ,{'RC_JOB_WALL_SECONDS':'570'}):
            M.validate_admission(d,h,actual=actual,memory={'effective_limit_bytes':memory})
        return reads

    def test_only_known_code_and_environment_reads(self):
        reads=self.probe()
        self.assertEqual(len(reads),4)
        self.assertTrue(reads[-1][0].endswith('/node_emb.py'))

    def test_arbitrary_and_duplicate_flags_rejected(self):
        for flags in [['--test','/test'],['--run-id','other'],['--deadline-unix','9999999999']]:
            with self.subTest(flags=flags),self.assertRaises((RuntimeError,SystemExit)):
                self.probe(lambda d:d['canary_args'].extend(flags))

    def test_foreign_wrapper_rejected(self):
        with self.assertRaises((RuntimeError,SystemExit)):
            self.probe(lambda d:d['canary_args'].__setitem__(d['canary_args'].index('--tgcn-wrapper')+1,'/foreign.py'))

    def test_dataset_or_test_admission_rejected(self):
        for key in ['new_dataset_or_weight_reads','new_test_opener_calls']:
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.probe(lambda d:d.update({key:1}))

    def test_wrong_actual_resources_rejected(self):
        for key,val in [('cpus_on_node','2'),('gpu_allocation','1'),('time_limit','01:00:00')]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.probe(actual_alter=lambda d:d.update({key:val}))
        with self.assertRaises(RuntimeError):self.probe(memory=4*1024**3)

    def test_deadline_is_derived_from_job_start(self):
        with self.assertRaises(RuntimeError):self.probe(lambda d:d.update(deadline_unix=123))
        with self.assertRaises(RuntimeError):self.probe(lambda d:d.update(work_seconds=10000))

if __name__=='__main__':unittest.main()
