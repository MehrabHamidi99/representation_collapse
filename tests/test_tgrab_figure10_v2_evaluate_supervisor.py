"""Pure admission/launch-envelope checks; no SSH, Slurm or native models."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('v2_eval_supervisor', ROOT / 'code/ops/tgrab_figure10_v2_evaluate_supervisor.py')
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)
CONTRACT_SHA = 'aec185fe544bf94a1dc0ea5e952d27b0d335c3653d815f7c0796d97b747ea381'


def sha(raw): return hashlib.sha256(raw).hexdigest()


class V2EvaluationSupervisorTests(unittest.TestCase):
    def probe(self, change=None, actual_change=None, memory=2147483648):
        helper = EVAL.helper()
        with tempfile.TemporaryDirectory() as td:
            base=Path(td).resolve(); fit=base/'fit';fit.mkdir()
            job=dict(account='mila',partition='main-cpu',qos='normal',nodes=1,tasks=1,
                cpus_on_node='1',cpus_per_task='1',time_limit='00:10:00',gpu_allocation='0',
                memory_max_bytes=2147483648,job_id=None,node_list=None)
            actual={**job,'job_id':'123','node_list':'cpu01'}
            d=dict(schema='tgrab-figure10-v2-evaluation-admission-v1',protocol_id=EVAL.PROTOCOL_ID,
                source_commit='a'*40,run_id='v2-eval01',job=job,work_seconds=540,outer_wall_seconds=570,
                canary_args=['--run-id','v2-eval01'])
            artifacts={}
            for filename in ('fit-summary.json','manifest.json','wandb/reporting-receipt.json','selection/selected.json'):
                p=fit/filename;p.parent.mkdir(parents=True,exist_ok=True)
                raw=json.dumps({'synthetic_artifact':filename}).encode();p.write_bytes(raw);artifacts[filename]=sha(raw)
            fit_receipt=base/'fit-supervisor.json';fit_receipt.write_text('{"synthetic_fit_receipt":true}')
            acceptance=dict(kind='figure10_long_fit_execution_acceptance_v2',status='accepted',test_accessed=False,
                validation_selection_finalized=True,source_closure_git_verified=True,
                offline_history_payload_verified=True,optimizer_history_verified=True,
                fit_run_id='long01',fit_attempt_id='longfit01',fit_admission_sha256=sha(fit_receipt.read_bytes()),
                fit_summary_sha256=artifacts['fit-summary.json'],selected_pointer_sha256=artifacts['selection/selected.json'],
                fit_run_dir=str(fit),artifact_sha256=artifacts,executed_long_contract_sha256=CONTRACT_SHA)
            def add_pin(key,path,raw=None):
                if raw is not None:path.write_bytes(raw)
                d[key]={'path':str(path),'sha256':sha(path.read_bytes())}
                flag='--'+key.replace('_','-');d['canary_args'] += [flag,str(path),flag+'-sha256',d[key]['sha256']]
            for key in ('environment_receipt','node2a_receipt','conversion_receipt'):
                add_pin(key,base/(key+'.json'),b'{}')
            for key in ('fit_run_id','fit_attempt_id','fit_admission_sha256','fit_summary_sha256','selected_pointer_sha256','fit_run_dir'):
                d['canary_args'] += ['--'+key.replace('_','-'),acceptance[key]]
            add_pin('long_contract',ROOT/'configs/graph/tgrab-long-fit-v2.json')
            d['fit_supervisor_receipt']={'path':str(fit_receipt),'sha256':sha(fit_receipt.read_bytes())}
            d['canary_args'] += ['--fit-supervisor-receipt',str(fit_receipt)]
            acceptance_path=base/'acceptance.json'
            # Pin these arguments after acceptance mutation too, representing an
            # externally authenticated but semantically invalid acceptance file.
            if change:change(d,acceptance,fit)
            raw=json.dumps(acceptance,sort_keys=True).encode();acceptance_path.write_bytes(raw)
            d['fit_acceptance']={'path':str(acceptance_path),'sha256':sha(raw)}
            d['canary_args'] += ['--fit-acceptance',str(acceptance_path),'--fit-acceptance-sha256',sha(raw)]
            if actual_change:actual_change(actual)
            with patch.dict(os.environ,{'RC_JOB_WALL_SECONDS':'570'}):
                return EVAL.validate_admission(d,helper,actual=actual,memory={'effective_limit_bytes':memory})

    def test_verified_primitive_v2_fit_passes_exact_cpu_envelope(self):
        actual,memory=self.probe();self.assertEqual(actual['job_id'],'123');self.assertEqual(memory['effective_limit_bytes'],2*1024**3)
        self.assertEqual(EVAL.PROTOCOL_ID,'tgrab_figure10_primitive_v2_selected_evaluation_v3')
        self.assertEqual(EVAL.WORKER,'src/representation_collapse/tgrab_figure10_v2_evaluate_job.py')

    def test_v1_acceptance_or_admission_cannot_be_coerced(self):
        for kind in ('figure10_fit_execution_acceptance_v1','figure10_selected_evaluation_execution_acceptance_v2'):
            with self.subTest(kind=kind),self.assertRaises((RuntimeError,SystemExit)):
                self.probe(lambda d,a,f:a.update(kind=kind))
        with self.assertRaises((RuntimeError,SystemExit)):
            self.probe(lambda d,a,f:d.update(schema='tgrab-figure10-evaluation-admission-v2'))

    def test_missing_acceptance_gates_rejected(self):
        for key in ('kind','status','test_accessed','validation_selection_finalized','source_closure_git_verified','offline_history_payload_verified','optimizer_history_verified'):
            with self.subTest(key=key),self.assertRaises((RuntimeError,SystemExit)):
                self.probe(lambda d,a,f:a.pop(key))
        with self.assertRaises((RuntimeError,SystemExit)):
            self.probe(lambda d,a,f:a.update(test_accessed=True))

    def test_four_artifact_inventory_and_cross_hashes_required(self):
        for key in ('fit-summary.json','manifest.json','wandb/reporting-receipt.json','selection/selected.json'):
            with self.subTest(key=key),self.assertRaises((RuntimeError,SystemExit)):
                self.probe(lambda d,a,f:a['artifact_sha256'].pop(key))
        for key in ('fit-summary.json','selection/selected.json'):
            with self.subTest(key=key),self.assertRaises((RuntimeError,SystemExit)):
                self.probe(lambda d,a,f:a['artifact_sha256'].update({key:'f'*64}))

    def test_changed_pointer_or_any_actual_artifact_bytes_rejected(self):
        for name in ('selection/selected.json','fit-summary.json','manifest.json','wandb/reporting-receipt.json'):
            with self.subTest(name=name),self.assertRaises((RuntimeError,SystemExit)):
                self.probe(lambda d,a,f:(f/name).write_text('{"tampered":true}'))

    def test_contract_and_actual_fit_supervisor_hashes_bound(self):
        with self.assertRaises((RuntimeError,SystemExit)):
            self.probe(lambda d,a,f:a.update(executed_long_contract_sha256='f'*64))
        # Rewrite acceptance and its matching CLI hash, leaving actual supervisor
        # receipt pinned to the original execution; the cross-check must fail.
        def wrong_supervisor(d,a,f):
            a['fit_admission_sha256']='e'*64
            i=d['canary_args'].index('--fit-admission-sha256');d['canary_args'][i+1]='e'*64
        with self.assertRaises((RuntimeError,SystemExit)):self.probe(wrong_supervisor)

    def test_missing_or_changed_contract_supervisor_cli_arguments_rejected(self):
        for flag in ('--long-contract','--long-contract-sha256','--fit-supervisor-receipt'):
            def mutate(d,a,f):
                i=d['canary_args'].index(flag);d['canary_args'][i+1]='wrong'
            with self.subTest(flag=flag),self.assertRaises((RuntimeError,SystemExit)):self.probe(mutate)

    def test_changed_actual_resources_rejected(self):
        for key,value in (('cpus_on_node','2'),('gpu_allocation','1'),('time_limit','00:20:00'),('account','other'),('partition','gpu')):
            with self.subTest(key=key),self.assertRaises((RuntimeError,SystemExit)):
                self.probe(actual_change=lambda a:a.update({key:value}))
        with self.assertRaises((RuntimeError,SystemExit)):self.probe(memory=4*1024**3)

    def test_timing_deadline_and_duplicate_flags_rejected(self):
        for key,value in (('work_seconds',541),('work_seconds',540.0),('outer_wall_seconds',571),('deadline_unix',123)):
            with self.subTest(key=key),self.assertRaises((RuntimeError,SystemExit)):
                self.probe(lambda d,a,f:d.update({key:value}))
        with self.assertRaises((RuntimeError,SystemExit)):
            self.probe(lambda d,a,f:d['canary_args'].extend(['--run-id','v2-eval01']))

    def test_source_closure_contains_v3_protocol_and_every_package_module(self):
        with patch.object(EVAL,'helper',return_value=SimpleNamespace(sha256_path=lambda p:'a'*64)):
            closure=EVAL.source_closure(ROOT)
        for required in ('doc/auto/figure10-selected-evaluation-v3.md','configs/graph/tgrab-long-fit-v2.json',EVAL.WORKER,
                         'src/representation_collapse/tgrab_figure10_v2_codec.py'):
            self.assertIn(required,closure)
        self.assertEqual({k for k in closure if k.startswith('src/representation_collapse/')},
            {str(p.relative_to(ROOT)) for p in (ROOT/'src/representation_collapse').glob('*.py')})
        self.assertIn('configs/graph/tgrab-selected-checkpoint-schema-v1.json',closure) # unchanged numeric encoding helpers only

    def test_job_wrapper_has_finite_shell_ceiling_and_cpu_threads(self):
        path=ROOT/'code/ops/tgrab_figure10_v2_evaluate_job.sh'
        subprocess.run(['/bin/bash','-n',str(path)],check=True,capture_output=True)
        text=path.read_text()
        self.assertIn('timeout --signal=TERM --kill-after=10 570s',text)
        self.assertIn('OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',text)
        self.assertIn('tgrab_figure10_v2_evaluate_supervisor.py',text)
        self.assertNotIn('sbatch',text)

if __name__=='__main__': unittest.main()
