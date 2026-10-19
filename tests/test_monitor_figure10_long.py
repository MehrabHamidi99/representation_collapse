"""No real SSH, scheduler/model/provider or scientific data is accessed."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH=Path(__file__).resolve().parents[1]/'code/ops/monitor_figure10_long.py'
spec=importlib.util.spec_from_file_location('figure10_long_monitor',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def launch():
    run='long-20261019T215117Z-1891bdd5'
    return {'run_id':run,'output_remote':m.SCRATCH+'/'+run,'source_commit':'a'*40,'admission_sha256':'b'*64}


def snapshot(epoch=0,state='RUNNING'):
    return {'job_id':'123','scheduler':{'job_id':'123','state':state},
        'selected':{'last_epoch':epoch,'best_f1':0.5,'selected':{'epoch':max(1,epoch-1)}},
        'prefix_parity':{'status':'pass','exact_loss_matches':256},'prefix_failure':None,
        'validation':None,'terminal':None,'last_event':None}


class MonitorTests(unittest.TestCase):
    def test_fixed_cluster_scratch_job_and_deadline_scope(self):
        self.assertGreater(m.validate_launch(launch(),'123'),0)
        for field,value in (('output_remote','/other/project/run'),('cluster','tamia'),('job_id','456'),('run_id','../x')):
            with self.subTest(field=field),self.assertRaises(ValueError):
                m.validate_launch({**launch(),field:value},'123')
        for job in ('0','1;other','-1',''):
            with self.assertRaises(ValueError):
                m.validate_launch(launch(),job)

    def test_complete_tail_excludes_incomplete_append_and_truncated_first(self):
        self.assertEqual(m.last_complete_line(b'{"a":1}\n{"a":'),{'a':1})
        self.assertIsNone(m.last_complete_line(b'{"a":'))
        self.assertEqual(m.last_complete_line(b'broken}\n{"a":2}\n{"a":',truncated_start=True),{'a':2})
        with self.assertRaises(ValueError):
            m.last_complete_line(b'{"a":1,"a":2}\n')

    def test_milestones_deduplicate_thresholds_not_idle_polls(self):
        seen=set()
        keys=m.milestone_keys(snapshot(21),seen)
        self.assertEqual(keys,['parity-pass','epoch-010','epoch-020'])
        seen.update(keys)
        self.assertEqual(m.milestone_keys(snapshot(29),seen),[])
        self.assertEqual(m.milestone_keys(snapshot(30),seen),['epoch-030'])
        bad=snapshot();bad['prefix_failure']={'read_status':'incomplete_or_unavailable'}
        self.assertNotIn('terminal',m.milestone_keys(bad,seen))
        self.assertEqual(m.milestone_keys(snapshot(29,'FAILED'),seen),['terminal'])

    def test_user_held_pending_job_is_not_failed_released_or_tasked(self):
        held=snapshot(0,'PENDING')
        held['prefix_parity']=None
        held['selected']=None
        held['scheduler']['reason']='JobHeldUser'
        self.assertEqual(m.milestone_keys(held,set()),[])
        self.assertEqual(m.compact_summary(held)['scheduler_reason'],'JobHeldUser')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'monitor';now=[0.0]
            monitor=m.Monitor(launch(),'123',root,launch_sha256='c'*64,deadline=5000,clock=lambda:now[0])
            def sleep(seconds):
                now[0]+=seconds
                (root/'STOP').touch()
            with patch.object(monitor,'publish') as publish:
                self.assertEqual(monitor.run(fetch=lambda *args:held,sleep=sleep),'owner_stop')
                publish.assert_not_called()
            self.assertEqual(m.decode((root/'status.json').read_bytes())['snapshot']['scheduler']['state'],'PENDING')

    def test_cleanup_does_not_write_after_owner_or_directory_replacement(self):
        for replace_directory in (False,True):
            with self.subTest(directory=replace_directory),tempfile.TemporaryDirectory() as td:
                root=Path(td)/'monitor'
                monitor=m.Monitor(launch(),'123',root,launch_sha256='c'*64,deadline=5000,clock=lambda:0)
                if replace_directory:
                    root.rename(root.with_name('original'))
                    root.mkdir()
                (root/'owner.json').write_text('{"token":"another-owner"}')
                (root/'status.json').write_text('foreign status')
                (root/'heartbeat.json').write_text('foreign heartbeat')
                with self.assertRaises(RuntimeError):
                    monitor.run(fetch=lambda *args:self.fail('ownership failure must precede SSH'))
                self.assertEqual((root/'status.json').read_text(),'foreign status')
                self.assertEqual((root/'heartbeat.json').read_text(),'foreign heartbeat')
                self.assertTrue(monitor.journal.closed)

    def test_ssh_is_batch_only_one_exact_job_no_job_actions(self):
        with patch.object(m.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps(snapshot()),stderr='')) as run:
            self.assertEqual(m.collect(launch(),'123')['job_id'],'123')
            argv=run.call_args.args[0]
            self.assertIn('BatchMode=yes',argv)
            self.assertIn('NumberOfPasswordPrompts=0',argv)
            self.assertEqual(argv[-2],'mila')
            self.assertNotIn('sbatch',argv[-1]);self.assertNotIn('scancel',argv[-1]);self.assertNotIn('srun',argv[-1])
            self.assertEqual(run.call_args.kwargs['timeout'],40)
            self.assertEqual(run.call_args.kwargs['stdin'],subprocess.DEVNULL)
        with patch.object(m.subprocess,'run',return_value=SimpleNamespace(returncode=255,stdout='',stderr='interactive login needed')):
            with self.assertRaises(ConnectionError):
                m.collect(launch(),'123')

    def test_network_backoff_and_stop_are_monitor_only(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'monitor';now=[0.0];calls=[]
            monitor=m.Monitor(launch(),'123',root,launch_sha256='c'*64,deadline=5000,clock=lambda:now[0])
            def fetch(*args):
                calls.append(now[0])
                if len(calls)<3:
                    raise ConnectionError('unavailable')
                (root/'STOP').touch()
                return snapshot()
            def sleep(seconds):now[0]+=seconds
            self.assertEqual(monitor.run(fetch=fetch,sleep=sleep),'owner_stop')
            self.assertEqual(calls,[0.0,300.0,900.0])
            events=[m.decode(line) for line in (root/'journal.jsonl').read_bytes().splitlines()]
            self.assertEqual([r['next_poll_seconds'] for r in events if r['event']=='probe_error'],[300,600])
            self.assertEqual(m.decode((root/'status.json').read_bytes())['reason'],'owner_stop')
            self.assertEqual(m.decode((root/'heartbeat.json').read_bytes())['state'],'stopped')

    def test_tasks_are_bounded_provisional_deduplicated_and_no_provider_call(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);evidence=root/'evidence';inbox=root/'tasks';evidence.mkdir();inbox.mkdir()
            monitor=m.Monitor(launch(),'123',root/'monitor',launch_sha256='c'*64,deadline=5000,evidence=evidence,inbox=inbox,clock=lambda:0)
            try:
                monitor.publish('parity-pass',snapshot())
                monitor.publish('parity-pass',snapshot(2))
                tasks=list(inbox.glob('*.json'))
                self.assertEqual(len(tasks),1)
                task=m.decode(tasks[0].read_bytes())
                self.assertEqual(set(task),{'task_id','task'})
                self.assertLessEqual(len(task['task']),1500)
                self.assertIn('provisional/unaccepted scientific evidence',task['task'])
                self.assertIn('Proposal-only',task['task'])
            finally:
                monitor.journal.close()

    def test_optional_paths_require_existing_approved_locations(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):m.validate_optional_paths(Path(td),None)
            with self.assertRaises(ValueError):m.validate_optional_paths(Path(td),Path(td))


if __name__=='__main__':
    unittest.main()
