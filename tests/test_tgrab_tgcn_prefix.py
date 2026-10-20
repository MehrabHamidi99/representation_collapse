"""Pure prefix orchestration/admission spies; no native model or graph file."""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from representation_collapse import tgrab_tgcn_prefix as prefix


class Optimizer:
    def __init__(self):self.steps=0
    def step(self):self.steps+=1


class Backend:
    def __init__(self,fail=None):self.optimizer=Optimizer();self.events=[];self.fail=fail
    def training_mode(self):return nullcontext()
    def reset(self):return SimpleNamespace(next_target_t=0)
    def detach(self,s):self.events.append(('detach',s.next_target_t));return s
    def consume(self,s,g):
        assert s.next_target_t==g.timestamp
        self.events.append(('consume',g.timestamp));return SimpleNamespace(next_target_t=g.timestamp+1)
    def loss_and_update(self,state,target,negatives):
        assert state.next_target_t==target.timestamp
        self.events.append(('loss',target.timestamp))
        if self.fail=='before' and target.timestamp==2:raise ArithmeticError('before')
        self.optimizer.step()
        if self.fail=='after' and target.timestamp==2:raise ArithmeticError('after')
        return 1/target.timestamp


class Reporter:
    def __init__(self):self.logs=[]
    def log(self,values,step):self.logs.append((step,values))


def graphs():return [SimpleNamespace(timestamp=t,nodes=100,node_ids=tuple(range(100))) for t in range(257)]
def context():return {'run_id':'synthetic-prefix','supervisor_receipt_sha256':'a'*64,
    'calibration_acceptance_sha256':'b'*64,'data_manifest_sha256':prefix.DATA_MANIFEST_SHA256}


class PrefixTests(unittest.TestCase):
    def test_exact256_causal_targets_durable_full_journal_and_no_target256_consume(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);journal=prefix.PrefixJournal(root,context());backend=Backend();reporter=Reporter()
            result,state,updates_sha=prefix.run_prefix(backend=backend,graphs=graphs(),sampler=lambda g:(),guard=lambda:None,
                journal=journal,reporter=reporter,output=root)
            self.assertTrue(journal.audit()['full_readback_verified'])
            journal.close();self.assertEqual(backend.optimizer.steps,256);self.assertEqual(state.next_target_t,256)
            self.assertEqual([t for kind,t in backend.events if kind=='consume'],list(range(256)))
            self.assertEqual([t for kind,t in backend.events if kind=='loss'],list(range(1,257)))
            self.assertEqual([step for step,_ in reporter.logs],list(range(256)))
            events=[json.loads(line) for line in (root/'optimizer-journal.jsonl').read_bytes().splitlines()]
            self.assertEqual(len(events),512);previous=None
            for n,(raw,event) in enumerate(zip((root/'optimizer-journal.jsonl').read_bytes().splitlines(keepends=True),events)):
                self.assertEqual(event['event'],n);self.assertEqual(event['previous_record_sha256'],previous)
                self.assertEqual(event['run_id'],context()['run_id']);previous=prefix.sha(raw)
            self.assertEqual(updates_sha,prefix.sha((root/'updates.jsonl').read_bytes()))
            self.assertEqual(len((root/'updates.jsonl').read_bytes().splitlines()),256)
            self.assertFalse(result['validation_replayed']);self.assertFalse(result['test_payload_opened'])

    def test_failure_before_or_after_step_keeps_distinct_confirmed_counts(self):
        for failure,steps,status in [('before',1,'not_started'),('after',2,'completed')]:
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as td:
                journal=prefix.PrefixJournal(td,context());backend=Backend(failure)
                with self.assertRaises(Exception):prefix.run_prefix(backend=backend,graphs=graphs(),sampler=lambda g:(),
                    guard=lambda:None,journal=journal,reporter=Reporter(),output=td)
                a=journal.accounting();journal.close()
                self.assertEqual(backend.optimizer.steps,steps);self.assertEqual(a['durable_attempted_updates'],2)
                self.assertEqual(a['durable_completed_updates'],1);self.assertEqual(a['durable_optimizer_steps_confirmed'],steps)
                self.assertTrue(a['journal_failed_or_terminal'])
                last=json.loads((Path(td)/'updates.jsonl').read_bytes().splitlines()[-1]);self.assertEqual(last['optimizer_step_status'],status)
                with self.assertRaises(Exception):journal.on_attempt({'sequence':3})

    def test_intent_fsync_failure_is_charged_ambiguous_not_falsely_zero_attempt(self):
        with tempfile.TemporaryDirectory() as td:
            journal=prefix.PrefixJournal(td,context())
            record={'sequence':1,'epoch':1,'target_timestamp':1,'consumed_timestamp':0,'training_order_index':0}
            with patch.object(prefix.os,'fsync',side_effect=OSError('fsync')):
                with self.assertRaises(OSError):journal.on_attempt(record)
            a=journal.accounting();journal.close()
            self.assertEqual(a['durable_attempted_updates'],0)
            self.assertEqual(a['ambiguous_journal_boundary'],{'sequence':1,'stage':'intent_fsync_unconfirmed'})
            self.assertTrue(a['journal_failed_or_terminal'])

    def test_directory_fsync_failure_and_invalid_callbacks_are_terminal(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(prefix.os,'fsync',side_effect=[None,OSError('directory fsync')]):
                with self.assertRaises(OSError):prefix.PrefixJournal(td,context())
        with tempfile.TemporaryDirectory() as td:
            journal=prefix.PrefixJournal(td,context())
            with self.assertRaises(RuntimeError):journal.on_attempt({'sequence':2})
            self.assertTrue(journal.poisoned)
            with self.assertRaises(RuntimeError):journal.on_attempt({'sequence':1})
            with self.assertRaises(RuntimeError):journal.audit()
            journal.close()

    def test_prior_bytes_torn_tail_and_wrong_path_identity_rejected(self):
        record={'sequence':1,'epoch':1,'target_timestamp':1,'consumed_timestamp':0,'training_order_index':0}
        for kind in ('prior','tail','replace'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                journal=prefix.PrefixJournal(td,context());journal.on_attempt(record)
                if kind=='prior':
                    raw=journal.path.read_bytes();journal.path.write_bytes(raw.replace(b'target_timestamp',b'target_timestamQ'))
                elif kind=='tail':
                    with journal.path.open('ab') as stream:stream.write(b'{')
                else:
                    journal.path.unlink();journal.path.write_bytes(b'')
                with self.assertRaises(RuntimeError):journal.audit()
                self.assertTrue(journal.poisoned);journal.close()

    def test_outcome_fsync_failure_never_promotes_visible_bytes_to_confirmed(self):
        with tempfile.TemporaryDirectory() as td:
            journal=prefix.PrefixJournal(td,context())
            journal.on_attempt({'sequence':1,'epoch':1,'target_timestamp':1,'consumed_timestamp':0,'training_order_index':0})
            record={'attempt':1,'attempted_updates':1,'completed':True,'completed_updates':1,'optimizer_steps_confirmed':1,
                'optimizer_step_status':'completed','target_timestamp':1,'consumed_timestamp':0}
            with patch.object(prefix.os,'fsync',side_effect=OSError('outcome fsync')):
                with self.assertRaises(OSError):journal.on_update(record)
            a=journal.accounting();self.assertEqual(a['durable_completed_updates'],0)
            self.assertEqual(a['durable_optimizer_steps_confirmed'],0)
            self.assertEqual(a['ambiguous_journal_boundary']['observed_step_status'],'completed')
            with self.assertRaises(RuntimeError):journal.audit()
            journal.close()

    def test_short_intent_write_is_terminal_and_keeps_boundary_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            journal=prefix.PrefixJournal(td,context());original=journal.stream
            class Short:
                def __getattr__(self,name):return getattr(original,name)
                def write(self,data):return original.write(data[:-1])
            journal.stream=Short()
            with self.assertRaises(RuntimeError):journal.on_attempt({'sequence':1,'epoch':1,'target_timestamp':1,
                'consumed_timestamp':0,'training_order_index':0})
            self.assertTrue(journal.poisoned);self.assertEqual(journal.attempted,0)
            self.assertEqual(journal.entered['sequence'],1);journal.close()

    def test_guard_stop_and_malformed_chronology_before_optimizer(self):
        with tempfile.TemporaryDirectory() as td:
            stop=Path(td)/'STOP';stop.touch()
            with self.assertRaises(RuntimeError):prefix.make_guard(time.time()+10,stop)
            with self.assertRaises(RuntimeError):prefix.make_guard(time.time()-1,Path(td)/'other')
            gs=graphs();gs[1].timestamp=2;journal=prefix.PrefixJournal(td,context());backend=Backend()
            with self.assertRaises(Exception):prefix.run_prefix(backend=backend,graphs=gs,sampler=lambda g:(),guard=lambda:None,
                journal=journal,reporter=Reporter(),output=td)
            journal.close();self.assertEqual(backend.optimizer.steps,0)

    def test_real_calibration_receipt_scope_required(self):
        record={'kind':'tgcn_native_calibration_compact_acceptance_v1','status':'accepted_synthetic_execution',
            'source_closure_git_verified':True,'training_admitted':False,'scientific_acceptance':False,
            'result_sha256':prefix.CALIBRATION_RESULT_SHA256,'checks_verified':248,'job_id':'11332092'}
        prefix.validate_calibration(record)
        for key,value in [('training_admitted',True),('status','accepted'),('result_sha256','f'*64),('job_id','1'),('checks_verified',0)]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):prefix.validate_calibration({**record,key:value})

    def test_main_reporter_failure_never_calls_native_or_reads_graph(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'out';called=[]
            args=['--output',str(out),'--run-id','synthetic','--deadline-unix',str(time.time()+100),
                '--stop-file',str(Path(td)/'STOP'),'--tgcn-wrapper','unused','--link-head','unused','--data-root','unused',
                '--data-manifest-sha256',prefix.DATA_MANIFEST_SHA256,'--calibration-acceptance','unused',
                '--calibration-acceptance-sha256','a'*64]
            class Broken:
                def __init__(self,*a,**kw):pass
                def __enter__(self):raise RuntimeError('offline reporting failed')
                def __exit__(self,*a):pass
            with self.assertRaises(RuntimeError):prefix.main(args,reporter_factory=Broken,native_runner=lambda *a:called.append(True))
            self.assertEqual(called,[]);self.assertTrue((out/'failure.json').exists())
            self.assertFalse((out/'updates.jsonl').exists())

    def test_supervisor_requires_exact_effective_args_and_inputs(self):
        args=SimpleNamespace(run_id='prefix01',deadline_unix=100.,calibration_acceptance='/acceptance',
            calibration_acceptance_sha256='b'*64,data_root='/converted',data_manifest_sha256=prefix.DATA_MANIFEST_SHA256)
        argv=['--run-id','prefix01'];source='c'*64
        receipt={'schema':'tgrab-tgcn-prefix-supervisor-receipt-v1','protocol_id':prefix.PROTOCOL_ID,
            'run_id':args.run_id,'deadline_unix':args.deadline_unix,'closure_sha256':{prefix.WORKER:source,
                'src/representation_collapse/tgrab_tgcn_backend.py':prefix.BACKEND_SHA256,prefix.CALIBRATION_HELPER:prefix.CALIBRATION_HELPER_SHA256},
            'canary_args':argv,'job_id':'123','source_commit':'d'*40,'admission_sha256':'e'*64,
            'calibration_acceptance':{'path':'/acceptance','sha256':'b'*64},'data_root':'/converted','data_manifest_sha256':prefix.DATA_MANIFEST_SHA256}
        prefix.validate_supervisor(receipt,args,argv,source)
        for key,value in [('canary_args',[]),('data_root','/other'),('schema','old'),('deadline_unix',101.)]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):prefix.validate_supervisor({**receipt,key:value},args,argv,source)

    def test_native_imports_and_train_loader_only_inside_runtime(self):
        import ast
        tree=ast.parse(Path(prefix.__file__).read_text())
        imports=[(n.module if isinstance(n,ast.ImportFrom) else a.name) for n in tree.body
            if isinstance(n,(ast.Import,ast.ImportFrom)) for a in (n.names if isinstance(n,ast.Import) else [None])]
        self.assertNotIn('torch',imports);self.assertNotIn('numpy',imports)
        source=Path(prefix.__file__).read_text();self.assertNotIn('open_train_validation(',source)
        self.assertIn("gd._load_split(manifest,'train'",source)

if __name__=='__main__':unittest.main()
