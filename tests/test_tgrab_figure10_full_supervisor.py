import importlib.util
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('full_supervisor', ROOT / 'code/ops/tgrab_figure10_full_supervisor.py')
FULL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FULL)


class FullSupervisorTests(unittest.TestCase):
    def test_source_closure_authenticates_all_package_sources_and_protocol(self):
        closure = FULL.source_closure(ROOT)
        for path in (ROOT / 'src/representation_collapse').glob('*.py'):
            self.assertIn(str(path.relative_to(ROOT)), closure)
        self.assertIn('doc/auto/tgrab-primary-data-conversion-v1.md', closure)
        self.assertEqual(len(closure['code/ops/tgrab_figure10_full_supervisor.py']), 64)

    def test_wrapper_has_absolute_source_and_rejects_bad_arguments(self):
        path = ROOT / 'code/ops/tgrab_figure10_full_job.sh'
        result = subprocess.run(['bash', str(path)], capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn('dirname', path.read_text())
        self.assertIn('$PINNED_ROOT/source/code/ops/', path.read_text())

    def test_admission_rejects_different_allocation(self):
        h = FULL.helper()
        d = {'schema': 'tgrab-figure10-full-admission-v1', 'protocol_id': FULL.PROTOCOL_ID,
             'source_commit': 'a'*40, 'run_id': 'run01', 'job': {}}
        with self.assertRaises(SystemExit):
            FULL.validate_admission(d, h, actual={}, memory={'effective_limit_bytes': 2147483648})


if __name__ == '__main__':
    unittest.main()
