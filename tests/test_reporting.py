"""Lifecycle/privacy failures tested without online services or scientific runs."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from representation_collapse.reporting import ExperimentReporter, PRIVACY_SETTINGS, ReportingError


class FakeRun:
    def __init__(self, args):
        self.id = args['id']
        self.settings = args['settings']
        self.config = args['config']
        self.entity = args['entity']
        self.url = 'https://wandb.ai/example/project/runs/example'
        self.dir = str(Path(args['dir']) / 'wandb' / 'offline-run-test' / 'files')
        self.summary = {}
        self.records = []
        self.finish_codes = []
        self.log_error = False
        self.finish_error = False
        self.artifacts = []

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def log(self, record, *, step, commit):
        if self.log_error:
            raise RuntimeError('PRIVATE_EXCEPTION_CONTENT')
        self.records.append((step, record, commit))

    def finish(self, *, exit_code):
        self.finish_codes.append(exit_code)
        if self.finish_error:
            raise TimeoutError('PRIVATE_EXCEPTION_CONTENT')


class FakeSDK:
    __version__ = '0.30.0'
    Settings = SimpleNamespace

    class Artifact:
        def __init__(self, name, **kwargs):
            self.name = name
            self.files = []

        def add_file(self, path, *, name, policy, skip_cache):
            if policy != 'immutable' or skip_cache is not True:
                raise AssertionError('metadata must remain in its durable local evidence tree')
            self.files.append((path, name))

    def __init__(self):
        self.run = None
        self.args = None
        self.init_error = False
        self.mode_override = None

    def init(self, **args):
        self.args = args
        if self.init_error:
            raise ConnectionError('PRIVATE_EXCEPTION_CONTENT')
        self.run = FakeRun(args)
        if self.mode_override:
            self.run.settings.mode = self.mode_override
        return self.run


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sdk = FakeSDK()
        self.patch = patch('representation_collapse.reporting.importlib.import_module', return_value=self.sdk)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def reporter(self, **overrides):
        options = dict(project='representation-collapse', config={'protocol_id': 'fixture'},
                       identities={'source_sha256': 'a'*64})
        options.update(overrides)
        return ExperimentReporter(Path(self.tmp.name) / 'new-run', **options)

    def receipt(self, reporter):
        return json.loads(reporter.receipt_path.read_text())

    def test_ready_before_work_and_success_with_monotonic_steps(self):
        reporter = self.reporter()
        with self.assertRaises(ReportingError):
            reporter.log({'loss': 1.0}, step=0)
        with reporter:
            self.assertTrue(self.receipt(reporter)['work_may_start'])
            reporter.log({'loss': 1.0}, step=0)
            for bad in [0, -1, True, 0.5]:
                with self.assertRaises(ValueError):
                    reporter.log({'loss': 2.0}, step=bad)
            reporter.log({'loss': 0.5}, step=3)
        receipt = self.receipt(reporter)
        self.assertEqual(receipt['state'], 'completed')
        self.assertEqual(receipt['last_completed_step'], 3)
        self.assertFalse(receipt['work_may_start'])
        self.assertFalse(receipt['remote_delivery_verified'])
        self.assertEqual(self.sdk.run.finish_codes, [0])
        self.assertEqual([r[0] for r in self.sdk.run.records], [0, 3])
        with self.assertRaises(ReportingError):
            reporter.log({'loss': 0.1}, step=4)

    def test_offline_default_and_online_requires_entity(self):
        with self.assertRaises(ValueError):
            self.reporter(mode='online')
        with self.assertRaises(ValueError):
            self.reporter(mode='disabled')
        with self.reporter() as reporter:
            self.assertEqual(self.sdk.args['mode'], 'offline')
            self.assertIsNone(self.sdk.args['entity'])
            self.assertEqual(self.sdk.args['resume'], 'never')
            self.assertEqual(self.sdk.args['reinit'], 'create_new')
        self.assertIsNone(self.receipt(reporter)['run_url'])

    def test_explicit_online_api_route_without_authentication_or_network(self):
        with self.reporter(mode='online', entity='actual-owner') as reporter:
            self.assertEqual(self.sdk.args['entity'], 'actual-owner')
        self.assertEqual(self.receipt(reporter)['run_url'], self.sdk.run.url)
        self.assertFalse(self.receipt(reporter)['remote_history_verified'])

    def test_privacy_settings_and_no_implicit_config(self):
        with self.reporter():
            for key, value in PRIVACY_SETTINGS.items():
                self.assertEqual(getattr(self.sdk.args['settings'], key), value)
            self.assertFalse(self.sdk.args['save_code'])
            self.assertFalse(self.sdk.args['sync_tensorboard'])
            self.assertEqual(set(self.sdk.args['config']), {'experiment', 'identities'})

    def test_init_failure_does_not_enter_work_and_keeps_sanitized_receipt(self):
        self.sdk.init_error = True
        reporter = self.reporter()
        entered = False
        with self.assertRaises(ReportingError):
            with reporter:
                entered = True
        self.assertFalse(entered)
        self.assertEqual(self.receipt(reporter)['state'], 'initialization_failed')
        self.assertNotIn('PRIVATE_EXCEPTION_CONTENT', reporter.receipt_path.read_text())

    def test_effective_mode_mismatch_closes_run_and_blocks_work(self):
        self.sdk.mode_override = 'online'
        reporter = self.reporter()
        with self.assertRaises(ReportingError):
            with reporter:
                self.fail('work must not start')
        self.assertEqual(self.sdk.run.finish_codes, [1])
        self.assertEqual(self.receipt(reporter)['state'], 'initialization_failed')

    def test_user_failure_preserved_and_marked_failed(self):
        reporter = self.reporter()
        with self.assertRaisesRegex(ValueError, 'original failure'):
            with reporter:
                reporter.log({'count': 1}, step=0)
                raise ValueError('original failure')
        self.assertEqual(self.receipt(reporter)['state'], 'failed')
        self.assertEqual(self.sdk.run.finish_codes, [1])
        self.assertEqual(self.sdk.run.summary['reporting/error_type'], 'ValueError')
        self.assertNotIn('original failure', reporter.receipt_path.read_text())

    def test_logging_failure_stops_run_and_does_not_advance_completed_step(self):
        reporter = self.reporter()
        with self.assertRaises(ReportingError):
            with reporter:
                self.sdk.run.log_error = True
                reporter.log({'count': 1}, step=0)
        self.assertEqual(self.receipt(reporter)['last_completed_step'], -1)
        self.assertEqual(self.receipt(reporter)['state'], 'failed')
        self.assertEqual(self.sdk.run.finish_codes, [1])

    def test_finish_failure_never_becomes_completion(self):
        reporter = self.reporter()
        with self.assertRaises(ReportingError):
            with reporter:
                self.sdk.run.finish_error = True
        receipt = self.receipt(reporter)
        self.assertEqual(receipt['state'], 'finish_failed')
        self.assertFalse(receipt['finish_returned'])
        self.assertNotIn('PRIVATE_EXCEPTION_CONTENT', reporter.receipt_path.read_text())

    def test_finish_failure_keeps_original_experiment_exception(self):
        reporter = self.reporter()
        with self.assertRaisesRegex(ValueError, 'original'):
            with reporter:
                self.sdk.run.finish_error = True
                raise ValueError('original')
        self.assertEqual(self.receipt(reporter)['state'], 'finish_failed')

    def test_raw_arrays_secrets_nonfinite_and_oversized_metadata_rejected(self):
        for config in [{'api_key': 'private'}, {'weights': [1, 2]}, {'loss': float('nan')},
                       {'text': 'x'*257}, {'x': {'nested': 1}}, {'x': 10**1000}]:
            with self.subTest(config_keys=list(config)):
                with self.assertRaises(ValueError):
                    self.reporter(config=config)
        with self.assertRaises(ValueError):
            self.reporter(identities={'source': '/private/source'})
        with self.reporter() as reporter:
            with self.assertRaises(ValueError):
                reporter.log({'examples': [1, 2]}, step=0)

    def test_no_overwrite_reentry_wrong_version_or_sweep_injection(self):
        reporter = self.reporter()
        with reporter:
            pass
        with self.assertRaises(ReportingError):
            reporter.__enter__()
        with self.assertRaises(FileExistsError):
            with self.reporter():
                pass
        other = ExperimentReporter(Path(self.tmp.name) / 'wrong-version', project='p', config={}, identities={'s': 'a'*64})
        self.sdk.__version__ = '0.29.0'
        with self.assertRaises(ReportingError):
            with other:
                pass
        third = ExperimentReporter(Path(self.tmp.name) / 'injection', project='p', config={}, identities={'s': 'a'*64})
        with patch.dict(os.environ, {'WANDB_CONFIG_PATHS': 'must-not-read'}):
            with self.assertRaises(ReportingError):
                with third:
                    pass

    def test_manifest_constructs_own_json_and_rejects_raw_file_or_array(self):
        with self.reporter() as reporter:
            for name, metadata in [('../private', {}), ('raw', {'data': [1, 2]}),
                                   ('file', Path('/private/data'))]:
                with self.assertRaises(ValueError):
                    reporter.log_manifest(name, metadata=metadata, identities={'s': 'b'*64})
            reporter.log_manifest('manifest', metadata={'count': 2}, identities={'source': 'b'*64})
            artifact = self.sdk.run.artifacts[0]
            path, name = artifact.files[0]
            self.assertEqual(name, 'manifest.json')
            self.assertEqual(Path(path).parent, reporter.output / 'metadata-artifacts')
            self.assertEqual(json.loads(Path(path).read_text())['metadata'], {'count': 2})
            with self.assertRaises(ValueError):
                reporter.log_manifest('manifest', metadata={}, identities={})


if __name__ == '__main__':
    unittest.main()
