"""Pure preparation/flow stubs only: never import Torch/PyG or execute native math."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'code/analysis/tgrab_tgcn_native_calibration.py'
spec=importlib.util.spec_from_file_location('tgcn_calibration',SCRIPT)
job=importlib.util.module_from_spec(spec);spec.loader.exec_module(job)

class CalibrationPreparationTests(unittest.TestCase):
    def test_import_is_native_dependency_free(self):
        run=subprocess.run([sys.executable,'-B','-c',f"import runpy,sys;runpy.run_path({str(SCRIPT)!r});assert not any(x in sys.modules for x in ('torch','numpy','wandb','torch_geometric'))"],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)

    def test_canonical_dynamic_fixtures_weight_symmetry_and_source_preserving_negatives(self):
        for edges in (job.A,job.B,job.D):
            self.assertEqual(edges,tuple(sorted(set(edges))))
            self.assertTrue(all(u!=v and (v,u) in edges for u,v in edges))
        weights=dict(zip(job.A,job.C_WEIGHTS))
        self.assertTrue(all(weights[(u,v)]==weights[(v,u)] for u,v in job.A))
        self.assertEqual(weights[(0,1)],.5);self.assertEqual(weights[(1,2)],1.5)
        self.assertEqual(weights[(2,3)],2.);self.assertEqual(weights[(3,0)],.75)
        self.assertEqual(len(job.NEGATIVES),len(job.D))
        for (u,_),(a,b) in zip(job.D,job.NEGATIVES):
            self.assertEqual(u,a);self.assertNotEqual(a,b);self.assertNotIn((a,b),job.D)

    def test_deadline_stop_and_nonfinite_reject(self):
        with tempfile.TemporaryDirectory() as td:
            stop=Path(td)/'STOP';guard=job.make_guard(10,stop,clock=lambda:9);guard()
            stop.write_text('stop')
            with self.assertRaisesRegex(RuntimeError,'STOP'):guard()
            with self.assertRaises(RuntimeError):job.make_guard(10,Path(td)/'other',clock=lambda:10)
            with self.assertRaises(RuntimeError):job.make_guard(float('nan'),stop)

    def test_pin_and_create_only_durable_result(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'source';p.write_bytes(b'content');self.assertEqual(job.pinned(p,hashlib.sha256(b'content').hexdigest()),b'content')
            with self.assertRaises(RuntimeError):job.pinned(p,'0'*64)
            link=Path(td)/'link';link.symlink_to(p)
            with self.assertRaises(RuntimeError):job.pinned(link,hashlib.sha256(b'content').hexdigest())
            out=Path(td)/'result.json';digest=job.write_new(out,{'valid':True})
            self.assertEqual(digest,hashlib.sha256(out.read_bytes()).hexdigest())
            with self.assertRaises(RuntimeError):job.write_new(out,{'valid':False})
            self.assertEqual(json.loads(out.read_bytes()),{'valid':True})

    def flow(self,*,bad_receipt=False,native_error=False):
        actions=[]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'run';receipt=root/'supervisor.json'
            receipt.write_bytes(job.canonical({'job_id':'123','source_commit':'c'*40,'admission_sha256':'a'*64,'protocol_id':job.PROTOCOL_ID,'run_id':'calibration01','deadline_unix':999999999999.,'closure_sha256':{job.WORKER:hashlib.sha256(SCRIPT.read_bytes()).hexdigest()}}))
            receipt_sha=hashlib.sha256(receipt.read_bytes()).hexdigest()
            class Reporter:
                def __init__(self,path,**kwargs):self.path=path;self.kwargs=kwargs
                def __enter__(self):
                    actions.append('entered');self.path.mkdir();return self
                def log_manifest(self,*a,**kwargs):
                    actions.append('artifact');self_test.assertEqual(kwargs['identities']['result_sha256'],hashlib.sha256((out/'calibration-result.json').read_bytes()).hexdigest())
                def __exit__(self,*args):
                    actions.append('finished');(self.path/'reporting-receipt.json').write_text('{}')
            self_test=self
            def native(args,guard,reporter):
                self.assertEqual(actions,['entered']);guard();actions.append('native')
                if native_error:raise RuntimeError('native failure')
                return {'status':'synthetic_native_calibration_passed','scientific_acceptance':False}
            argv=['--output',str(out),'--run-id','calibration01','--deadline-unix','999999999999','--stop-file',str(root/'STOP'),'--tgcn-wrapper','/unused/tgcn.py','--link-head','/unused/head.py']
            with patch.dict(os.environ,{'RC_SUPERVISOR_RECEIPT':str(receipt),'RC_SUPERVISOR_RECEIPT_SHA256':'0'*64 if bad_receipt else receipt_sha}):
                if bad_receipt or native_error:
                    with self.assertRaises(RuntimeError):job.main(argv,reporter_factory=Reporter,native_runner=native)
                    self.assertFalse((out/'manifest.json').exists());self.assertEqual(json.loads((out/'failure.json').read_text())['status'],'failed')
                    self.assertNotIn('artifact',actions)
                else:
                    self.assertEqual(job.main(argv,reporter_factory=Reporter,native_runner=native),0)
                    self.assertEqual(actions,['entered','native','artifact','finished'])
                    self.assertEqual(json.loads((out/'manifest.json').read_text())['status'],'completed')
                    with self.assertRaises(RuntimeError):job.main(argv,reporter_factory=Reporter,native_runner=native)
            return actions

    def test_reporter_precedes_native_and_artifact_terminal(self):self.flow()
    def test_receipt_failure_never_calls_native(self):self.assertNotIn('native',self.flow(bad_receipt=True))
    def test_native_failure_cannot_publish_success(self):self.flow(native_error=True)

if __name__=='__main__':unittest.main()
