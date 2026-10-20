"""Admission boundaries: no numerical runtime or cluster calls."""
import importlib.util
import os
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('tgcn_supervisor', ROOT/'code/ops/tgrab_tgcn_prefix_supervisor.py')
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)

class AdmissionTests(unittest.TestCase):
    def probe(self, alter=None, actual_alter=None, memory=2147483648):
        job = dict(account='mila',partition='main-cpu',qos='normal',nodes=1,tasks=1,
                   cpus_on_node='1',cpus_per_task='1',time_limit='00:30:00',gpu_allocation='0',
                   memory_max_bytes=2147483648,node_list=None)
        argv = ['--output','/output','--run-id','tgcn01']
        source='/network/scratch/m/mehrab.hamidi/representation-collapse/sources/tgcn-native-20261020/'
        for flag,(name,_) in M.SOURCE_INPUTS.items():argv += [flag,source+name]
        argv += ['--data-root','/network/scratch/m/mehrab.hamidi/representation-collapse/runs/primary-conversion-verified-20261016/attempt001/worker/converted', '--data-manifest-sha256','a6ff50183de570ff17cfbe92b3f1976bac9d2d85557ecc11379e24b2c737d1ce','--calibration-acceptance','/accepted-calibration.json','--calibration-acceptance-sha256','c'*64]
        d=dict(schema='tgrab-tgcn-prefix-admission-v1',protocol_id=M.PROTOCOL_ID,
               source_commit='a'*40,run_id='tgcn01',job=job,work_seconds=1740,outer_wall_seconds=1770,
               canary_args=argv,environment_receipt=dict(path='/environment',sha256='b'*64),
               training_prefix_only=True,new_test_opener_calls=0,calibration_acceptance=dict(path="/accepted-calibration.json",sha256="c"*64))
        actual={**job,'job_id':'123','node_list':'cpu01'}
        if alter:alter(d)
        if actual_alter:actual_alter(actual)
        reads=[]
        def fail(msg):raise RuntimeError(msg)
        def read(path,digest):
            reads.append((str(path),digest))
            accepted=dict(kind='tgcn_native_calibration_compact_acceptance_v1',status='accepted_synthetic_execution',checks_verified=248,source_commit='e0e6c20a72a98134aa1f2a2450f9eeb011718ab2',result_sha256='23b68f8c78e0f4b1fd00959c812463318cb687d8b9b95404c709c6ade5001619',actual_offline_history_rows=1)
            return json.dumps(accepted).encode() if str(path)=='/accepted-calibration.json' else b'{}' 
        h=SimpleNamespace(fail=fail,regular_bytes=read,_require_arg_pair=M.helper()._require_arg_pair)
        with patch.dict(os.environ,{'RC_JOB_WALL_SECONDS':'1770'}):
            M.validate_admission(d,h,actual=actual,memory={'effective_limit_bytes':memory})
        return reads

    def test_only_known_code_and_environment_reads(self):
        reads=self.probe()
        self.assertEqual(len(reads),6)
        self.assertTrue(any(p.endswith('/node_emb.py') for p,_ in reads))

    def test_arbitrary_and_duplicate_flags_rejected(self):
        for flags in [['--test','/test'],['--run-id','other'],['--deadline-unix','9999999999']]:
            with self.subTest(flags=flags),self.assertRaises((RuntimeError,SystemExit)):
                self.probe(lambda d:d['canary_args'].extend(flags))

    def test_foreign_wrapper_rejected(self):
        with self.assertRaises((RuntimeError,SystemExit)):
            self.probe(lambda d:d['canary_args'].__setitem__(d['canary_args'].index('--tgcn-wrapper')+1,'/foreign.py'))

    def test_nonprefix_and_test_admission_rejected(self):
        with self.assertRaises(RuntimeError):self.probe(lambda d:d.update(training_prefix_only=False))
        with self.assertRaises(RuntimeError):self.probe(lambda d:d.update(new_test_opener_calls=1))

    def test_wrong_actual_resources_rejected(self):
        for key,val in [('cpus_on_node','2'),('gpu_allocation','1'),('time_limit','01:00:00')]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.probe(actual_alter=lambda d:d.update({key:val}))
        with self.assertRaises(RuntimeError):self.probe(memory=4*1024**3)

    def test_deadline_is_derived_from_job_start(self):
        with self.assertRaises(RuntimeError):self.probe(lambda d:d.update(deadline_unix=123))
        with self.assertRaises(RuntimeError):self.probe(lambda d:d.update(work_seconds=10000))

if __name__=='__main__':unittest.main()
