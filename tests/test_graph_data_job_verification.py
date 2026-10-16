"""Actual worker control flow with fake parsers/reporting; no primary or SDK IO."""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('conversion_verification_fixture', ROOT / 'code/ops/graph_data_conversion_job.py')
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)


@contextmanager
def worker_case(*, mutate=None, fail=False, wrong_source=False):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary).resolve(); output = base / 'attempt001'; output.mkdir()
        config = job.load_config(ROOT / job.CONFIG_PATH)
        for source in config['sources'].values():
            source['sha256'] = job.sha256(ROOT / source['path'])
        raw = base / 'fake.csv'; raw.write_bytes(b'{}')
        config['raw'] = {'path': str(raw), 'bytes': 2, 'sha256': hashlib.sha256(b'{}').hexdigest()}
        plan = {'source_commit': 'c' * 40, 'config_sha256': 'd' * 64,
                'environment': {'sha256': 'e' * 64}, 'stop_path': str(base / 'STOP')}
        args = SimpleNamespace(worker_admission=output / 'worker-admission.json', plan_sha256='f' * 64)
        scheduler = base / 'scheduler.json'
        scheduler.write_text(json.dumps({'kind': 'primary_conversion_scheduler_v1', 'status': 'verified', 'fields': {
            'JobId': '999', 'Account': 'mila', 'QOS': 'normal', 'Partition': 'main-cpu',
            'NumCPUs': '1', 'NumNodes': '1', 'NumTasks': '1', 'TimeLimit': '00:10:00', 'Requeue': '0',
            'ReqTRES': {'cpu': '1', 'mem': '2147483648', 'node': '1'},
            'AllocTRES': {'cpu': '1', 'mem': '2147483648', 'node': '1'}}}))
        prefix = [str(Path(sys.executable).absolute()), '-B', str(ROOT / job.MODULE_PATH), '--worker']
        args.worker_admission.write_text(json.dumps({'started_monotonic': job.time.monotonic(),
            'plan_sha256': args.plan_sha256, 'slurm_job_id': '999', 'worker_argv': prefix,
            'supervisor_identity': {'pid': 42}, 'scheduler': {'path': str(scheduler), 'sha256': job.sha256(scheduler)}}))
        args.worker_admission_sha256 = job.sha256(args.worker_admission)
        events = []; report = mock.Mock(); context = mock.MagicMock()
        def enter():
            events.append('report_enter'); return report
        def exit_report(*unused):
            events.append('report_exit'); return False
        context.__enter__.side_effect = enter; context.__exit__.side_effect = exit_report
        manifest = output / 'worker/converted/manifest.json'
        def convert(raw_path, directory, **kwargs):
            events.append('convert'); assert events[0] == 'report_enter'
            directory.mkdir()
            for name in ('manifest', 'train', 'validation', 'test'):
                (directory / (name + '.json')).write_bytes(b'{}')
            return manifest
        converter = SimpleNamespace(__file__=str(ROOT / config['sources']['converter']['path']),
            convert_primary_csv=mock.Mock(side_effect=convert),
            read_manifest=mock.Mock(return_value=SimpleNamespace(metadata={
                'kind': 'tgrab_primary_converted_v1',
                'converter': {'revision': plan['source_commit'], 'sha256': config['sources']['converter']['sha256']},
                'counts': {'raw_rows': 6144, 'identical_duplicate_rows': 0, 'source_marker_rows': 0, 'unique_offdiagonal_edges': 6144}, 'snapshot_semantic_sha256': ['a' * 64] * 3072})))
        def verify(raw_path, manifest_path, **kwargs):
            events.append('verify')
            charged = job.read_json(output / 'worker/status.json')
            assert charged['verification_attempts'] == 1 and charged['state'] == 'verifying'
            assert events == ['report_enter', 'convert', 'verify']
            if fail:
                raise ValueError('independent synthetic mismatch')
            receipt = {'kind': 'tgrab_primary_conversion_independent_verification_v1', 'status': 'verified',
                'primary_eligible': True, 'independent_of_producer': True, 'scientific_acceptance': False,
                'raw_sha256': config['raw']['sha256'], 'raw_bytes': 2, 'manifest_sha256': job.sha256(manifest),
                'converter_sha256': config['sources']['converter']['sha256'], 'converter_revision': plan['source_commit'],
                'config_sha256': config['reconstruction_config']['sha256'],
                'verifier_source_sha256': config['sources']['verifier']['sha256'], 'snapshot_count': 3072, 'split_count': 3,
                'converted_snapshot_comparisons': 3072, 'semantic_snapshot_comparisons': 3072,
                'split_byte_comparisons': 3, 'complete_manifest_byte_comparisons': 1, 'stable_input_files_checked': 7,
                'raw_rows': 6144, 'identical_duplicate_rows': 0, 'source_marker_rows': 0, 'unique_offdiagonal_edges': 6144,
                'spec': {'nodes': 100, 'block': 32, 'train_periods': 40, 'validation_periods': 4, 'test_periods': 4},
                'split_hashes': {name: {'filename': name + '.json', 'bytes': 2,
                    'interval': {'train':[0,2560], 'validation':[2560,2816], 'test':[2816,3072]}[name],
                    'semantic_sha256': 'a' * 64,
                    'sha256': job.sha256(manifest.parent / (name + '.json'))} for name in ('train', 'validation', 'test')}}
            if mutate:
                mutate(receipt)
            return receipt
        verifier = SimpleNamespace(__file__=str(ROOT / config['sources']['verifier']['path']) + ('.wrong' if wrong_source else ''),
                                   verify_primary_conversion=mock.Mock(side_effect=verify))
        helpers = {'same_process': lambda a,b: True, 'process_identity': lambda pid: {'pid':pid}, 'install_spawn_journal':mock.Mock()}
        with ExitStack() as patches:
            patches.enter_context(mock.patch.object(job.sys, 'argv', [str(ROOT / job.MODULE_PATH), '--worker', '--worker-admission-sha256', args.worker_admission_sha256]))
            patches.enter_context(mock.patch.dict(job.os.environ, {'SLURM_JOB_ID': '999'}))
            patches.enter_context(mock.patch.object(job, 'runtime', return_value={}))
            patches.enter_context(mock.patch.object(job.runpy, 'run_path', return_value=helpers))
            patches.enter_context(mock.patch.object(job.signal, 'signal'))
            patches.enter_context(mock.patch.object(job.signal, 'setitimer'))
            patches.enter_context(mock.patch.object(resource, 'setrlimit'))
            patches.enter_context(mock.patch('representation_collapse.reporting.ExperimentReporter', return_value=context))
            modules = {'representation_collapse.graph_data': converter, 'representation_collapse.graph_data_verifier': verifier}
            patches.enter_context(mock.patch.object(job.importlib, 'import_module', side_effect=lambda name: modules[name]))
            result = job.worker(args, plan, config, ROOT, output)
        yield SimpleNamespace(result=result, output=output, plan=plan, config=config, converter=converter,
                              verifier=verifier, report=report, events=events,
                              status=job.read_json(output / 'worker/status.json'))


class ConversionVerificationTests(unittest.TestCase):
    def test_success_charges_both_once_and_finishes_reporting_after_verifier(self):
        with worker_case() as case:
            self.assertEqual(case.result, 0)
            self.assertEqual(case.events, ['report_enter', 'convert', 'verify', 'report_exit'])
            self.assertEqual([c.kwargs['step'] for c in case.report.log.call_args_list], [0, 1, 2])
            self.assertEqual(case.status['conversion_attempts'], 1)
            self.assertEqual(case.status['verification_attempts'], 1)
            self.assertEqual(case.status['verification_sha256'], job.sha256(case.output / 'worker/independent-verification.json'))
            case.converter.convert_primary_csv.assert_called_once(); case.verifier.verify_primary_conversion.assert_called_once()
            kwargs = case.verifier.verify_primary_conversion.call_args.kwargs
            self.assertNotIn('fixture_only', kwargs); self.assertNotIn('spec', kwargs)
            self.assertEqual(kwargs['expected_converter_revision'], case.plan['source_commit'])
            self.assertEqual(kwargs['config_path'], ROOT / case.config['reconstruction_config']['path'])

    def test_verifier_failure_is_terminal_with_retained_conversion_and_no_success_step(self):
        with worker_case(fail=True) as case:
            self.assertEqual(case.result, 1)
            self.assertEqual(case.status['state'], 'failed')
            self.assertEqual(case.status['verification_attempts'], 1)
            case.verifier.verify_primary_conversion.assert_called_once()
            self.assertTrue((case.output / 'worker/conversion-receipt.json').is_file())
            self.assertFalse((case.output / 'worker/independent-verification.json').exists())
            self.assertEqual([c.kwargs['step'] for c in case.report.log.call_args_list], [0, 1])

    def test_fixture_wrong_hash_partial_count_and_bool_counter_cannot_succeed(self):
        for mutation in (lambda r:r.update(kind='tgrab_fixture_conversion_independent_verification_v1'),
                         lambda r:r.update(verifier_source_sha256='0'*64),
                         lambda r:r.update(snapshot_count=3071),
                         lambda r:r.update(split_byte_comparisons=True),
                         lambda r:r.update(independent_of_producer=1)):
            with self.subTest(mutation=mutation), worker_case(mutate=mutation) as case:
                self.assertEqual(case.result, 1)
                self.assertEqual(case.status['verification_attempts'], 1)
                self.assertFalse((case.output / 'worker/independent-verification.json').exists())
                case.verifier.verify_primary_conversion.assert_called_once()

    def test_wrong_import_source_never_enters_verifier(self):
        with worker_case(wrong_source=True) as case:
            self.assertEqual(case.result, 1)
            self.assertEqual(case.status['verification_attempts'], 0)
            case.verifier.verify_primary_conversion.assert_not_called()

    def test_final_inventory_accepts_verified_bytes_and_rejects_later_split_mutation(self):
        with worker_case() as case:
            def census():
                return job.inventory(case.output, max_files=260, max_bytes=469762048, hashes=True)
            job.validate_terminal_artifacts(census(), case.output, case.plan, case.config, lambda:None)
            (case.output / 'worker/converted/train.json').write_bytes(b'{"changed":true}')
            with self.assertRaisesRegex(job.ConversionJobError, 'split changed'):
                job.validate_terminal_artifacts(census(), case.output, case.plan, case.config, lambda:None)

    def test_final_inventory_rejects_later_manifest_receipt_and_missing_verification(self):
        for target in ('worker/converted/manifest.json', 'worker/independent-verification.json'):
            with self.subTest(target=target), worker_case() as case:
                path = case.output / target; data = json.loads(path.read_text()); data['mutation'] = True
                path.write_text(json.dumps(data))
                records = job.inventory(case.output, max_files=260, max_bytes=469762048, hashes=True)
                with self.assertRaises(job.ConversionJobError):
                    job.validate_terminal_artifacts(records, case.output, case.plan, case.config, lambda:None)
        with worker_case() as case:
            (case.output / 'worker/independent-verification.json').unlink()
            with self.assertRaises(FileNotFoundError):
                job.validate_terminal_artifacts({}, case.output, case.plan, case.config, lambda:None)


if __name__ == '__main__':
    unittest.main()
