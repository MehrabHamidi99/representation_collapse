"""Real child cleanup and independent scheduler/controller admission checks."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'code/ops/tgrab_figure10_canary_supervisor.py'
spec = importlib.util.spec_from_file_location('canary_supervisor_tested', PATH)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

RECORD = ('JobId=123 JobState=RUNNING Account=mila Partition=main-cpu QOS=normal '
          'NumNodes=1 NumTasks=1 NumCPUs=1 CPUs/Task=1 NodeList=cpu1 TimeLimit=00:10:00 '
          'ReqTRES=cpu=1,mem=2G,node=1 AllocTRES=cpu=1,mem=2G,node=1 Requeue=0 Restarts=0')

class SchedulerTests(unittest.TestCase):
    def identity(self, record=RECORD, env=None):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', **(env or {})}, clear=True), patch.object(
                s.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, record, '')) as run:
            result = s.scheduler_identity()
            self.assertEqual(run.call_args.kwargs['timeout'], 5)
            return result

    def test_omitted_restart_and_gpu_environment_use_live_record(self):
        actual = self.identity()
        self.assertEqual(actual['tasks'], 1)
        self.assertEqual(actual['gpu_allocation'], '0')
        self.assertEqual(actual['scheduler_record']['Restarts'], '0')

    def test_restarted_requeued_incomplete_or_nonrunning_records_rejected(self):
        for before, after in [('Restarts=0', 'Restarts=1'), ('Requeue=0', 'Requeue=1'),
                              ('JobState=RUNNING', 'JobState=PENDING'), ('JobId=123', 'JobId=124'),
                              ('NumTasks=1 ', ''), ('AllocTRES=cpu=1,mem=2G,node=1', 'AllocTRES=cpu=1')]:
            with self.subTest(after=after), self.assertRaises(s.AdmissionError):
                self.identity(RECORD.replace(before, after))

    def test_gpu_tres_and_env_conflicts_rejected(self):
        for record, env in [(RECORD + ',gres/gpu=1', {}),
                            (RECORD.replace('AllocTRES=cpu=1,mem=2G,node=1', 'AllocTRES=cpu=1,mem=2G,node=1,gres/gpu=1'), {}),
                            (RECORD, {'SLURM_NTASKS': '2'}), (RECORD, {'SLURM_RESTART_COUNT': '1'}),
                            (RECORD, {'CUDA_VISIBLE_DEVICES': '1'})]:
            with self.subTest(env=env, record=record), self.assertRaises(s.AdmissionError):
                self.identity(record, env)

class AdmissionTests(unittest.TestCase):
    def fixture(self):
        job = {"account": "mila", "partition": "main-cpu", "qos": "normal", "nodes": 1,
               "tasks": 1, "cpus_on_node": "1", "cpus_per_task": "1", "time_limit": "00:10:00",
               "gpu_allocation": "0", "memory_max_bytes": 2 * 1024**3}
        admission = {"schema": "tgrab-figure10-prefix-admission-v2", "protocol_id": s.PROTOCOL_ID,
                     "source_commit": "a"*40, "closure_sha256": {}, "run_id": "prefix01",
                     "canary_args": ["--run-id", "prefix01", "--runtime-seconds", "540"],
                     "job": job, "work_seconds": 540, "outer_wall_seconds": 570,
                     "interpreter": {}, "environment_receipt": {}, "node2a_receipt": {}}
        return admission, {**job, "job_id": "123", "node_list": "cpu1"}, {"effective_limit_bytes": 2*1024**3}

    def test_relative_runtime_admission_has_no_prequeue_expiry(self):
        admission, actual, memory = self.fixture()
        with patch.dict(os.environ, {"RC_JOB_WALL_SECONDS": "570"}, clear=True):
            s.validate_admission(admission, actual=actual, memory=memory)

    def test_runtime_and_resource_mutations_rejected(self):
        import copy
        original, actual, memory = self.fixture()
        mutations = [lambda d: d.update(deadline_unix=time.time()+999),
                     lambda d: d.update(work_seconds=1000),
                     lambda d: d.update(outer_wall_seconds=600),
                     lambda d: d['canary_args'].extend(['--deadline-unix',str(time.time()+999)]),
                     lambda d: d['canary_args'].extend(['--runtime-seconds','540']),
                     lambda d: d['job'].update(cpus_on_node='2'),
                     lambda d: d['job'].update(memory_max_bytes=4*1024**3)]
        with patch.dict(os.environ, {"RC_JOB_WALL_SECONDS": "570"}, clear=True):
            for mutate in mutations:
                admission = copy.deepcopy(original); mutate(admission)
                with self.subTest(admission=admission), self.assertRaises(s.AdmissionError):
                    s.validate_admission(admission, actual=actual, memory=memory)

class CgroupTests(unittest.TestCase):
    def test_controllerless_ancestors_allowed_but_leaf_required(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root); leaf = base / 'job'; leaf.mkdir()
            (leaf / 'memory.max').write_text(str(2 * 1024**3))
            (leaf / 'memory.current').write_text('100')
            proc = base / 'proc-cgroup'; proc.write_text('0::/job\n')
            mounts = base / 'mountinfo'; mounts.write_text(f'20 10 0:9 / {base} rw - cgroup2 cgroup rw\n')
            memory = s.cgroup_memory(proc, mounts)
            self.assertEqual(memory['effective_limit_bytes'], 2 * 1024**3)
            (base / 'memory.max').write_text(str(1024**3))
            self.assertEqual(s.cgroup_memory(proc, mounts)['effective_limit_bytes'], 1024**3)
            (leaf / 'memory.max').unlink()
            with self.assertRaises(s.AdmissionError): s.cgroup_memory(proc, mounts)

    def test_nonpositive_and_malformed_limits_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            (base / 'memory.current').write_text('0')
            proc = base / 'proc'; proc.write_text('0::/\n')
            mounts = base / 'mount'; mounts.write_text(f'20 10 0:9 / {base} rw - cgroup2 cgroup rw\n')
            for value in ('0', '-1', 'garbage', 'max'):
                (base / 'memory.max').write_text(value)
                with self.subTest(value=value), self.assertRaises(s.AdmissionError): s.cgroup_memory(proc, mounts)

@unittest.skipUnless(sys.platform.startswith('linux') or sys.platform == 'darwin', 'POSIX sessions required')
class CleanupTests(unittest.TestCase):
    def test_stop_and_deadline_kill_stubborn_child(self):
        for reason in ('STOP', 'deadline', 'outer_signal_15'):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as root:
                ready = Path(root) / 'ready'; stop = Path(root) / 'STOP'
                child = subprocess.Popen([sys.executable, '-c',
                    'import signal,time,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);'
                    f'pathlib.Path({str(ready)!r}).touch();time.sleep(30)'], start_new_session=True)
                try:
                    until = time.monotonic() + 5
                    while not ready.exists() and time.monotonic() < until: time.sleep(.01)
                    self.assertTrue(ready.exists())
                    if reason == 'STOP': stop.touch()
                    status, actual = s.supervise_child(child, stop_path=stop,
                        deadline_unix=time.time() - 1 if reason == 'deadline' else time.time() + 5,
                        memory=None, memory_limit=None,
                        signal_state={'number': 15 if reason == 'outer_signal_15' else None}, cleanup_grace=.05)
                    self.assertEqual(actual, reason)
                    self.assertNotEqual(status, 0)
                    self.assertEqual(s.group_members(child.pid), [])
                finally:
                    if child.poll() is None: s.terminate_and_reap_group(child, grace_seconds=.05)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux subreaper required')
    def test_successful_parent_exit_still_reaps_orphan_grandchild(self):
        self.assertTrue(s.enable_child_subreaper())
        with tempfile.TemporaryDirectory() as root:
            pidfile = Path(root) / 'grandchild'
            code = ('import subprocess,sys,pathlib;'
                    'c=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]);'
                    f'pathlib.Path({str(pidfile)!r}).write_text(str(c.pid))')
            child = subprocess.Popen([sys.executable, '-c', code], start_new_session=True)
            try:
                self.assertEqual(child.wait(timeout=5), 0)
                self.assertTrue(pidfile.exists())
                status, reason = s.supervise_child(child, stop_path=Path(root)/'STOP',
                    deadline_unix=time.time()+5, memory=None, memory_limit=None, cleanup_grace=.05)
                self.assertEqual(status, 0)
                self.assertIsNone(reason)
                self.assertEqual(s.group_members(child.pid), [])
            finally:
                if s.group_members(child.pid): s.terminate_and_reap_group(child, grace_seconds=.05)

    def test_journal_failure_still_reaps(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], start_new_session=True)
        def broken(*a, **k): raise OSError('journal full')
        with self.assertRaises(OSError):
            s.supervise_child(child, stop_path='/nonexistent', deadline_unix=time.time()+3,
                              memory=None, memory_limit=None, journal=broken)
        self.assertIsNotNone(child.poll())
        self.assertEqual(s.group_members(child.pid), [])

class WrapperTests(unittest.TestCase):
    def test_explicit_arguments_pass_parsing_before_pinned_interpreter_check(self):
        wrapper = PATH.with_name('tgrab_figure10_canary_job.sh')
        result = subprocess.run(['bash', str(wrapper), '--admission', '/tmp/reviewed.json',
            '--sha256', 'a'*64, '--wall-seconds', '570'], capture_output=True, text=True, env={'PATH': os.environ['PATH']})
        # This test machine lacks the cluster-only interpreter. Parsing must succeed.
        self.assertEqual(result.returncode, 2)
        self.assertIn('pinned environment interpreter missing', result.stderr)
        self.assertNotIn('set admission', result.stderr)

    def test_unknown_duplicate_or_malformed_arguments_rejected(self):
        wrapper = PATH.with_name('tgrab_figure10_canary_job.sh')
        for args in [['--unknown'], ['--admission','/tmp/a','--admission','/tmp/b'],
                     ['--admission','relative','--sha256','a'*64,'--wall-seconds','570']]:
            with self.subTest(args=args):
                result = subprocess.run(['bash',str(wrapper),*args],capture_output=True,text=True,env={'PATH':os.environ['PATH']})
                self.assertEqual(result.returncode,2)
                self.assertNotIn('pinned environment interpreter missing',result.stderr)

if __name__ == '__main__': unittest.main()
