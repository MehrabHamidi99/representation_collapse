import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from representation_collapse.metrics import covariance_dimension


class ArtifactContractTest(unittest.TestCase):
    def test_saved_arrays_reproduce_metrics_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "attempt"
            subprocess.run([sys.executable, "-m", "representation_collapse.calibrate",
                            "--output", str(output)], check=True, capture_output=True)
            metrics = json.loads((output / "metrics.json").read_text())
            verification = json.loads((output / "verification.json").read_text())
            self.assertTrue(verification["passed"])
            self.assertEqual(json.loads((output / "status.json").read_text())["state"], "verified")
            with np.load(output / "arrays.npz") as arrays:
                for name in ["isotropic", "rank_one"]:
                    recomputed = covariance_dimension(arrays[name])
                    self.assertEqual(recomputed, metrics[name])
            manifest = json.loads((output / "manifest.json").read_text())
            for name, digest in manifest["sha256"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            original_manifest = (output / "manifest.json").read_bytes()
            duplicate = subprocess.run([sys.executable, "-m", "representation_collapse.calibrate",
                                        "--output", str(output)], capture_output=True)
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertEqual((output / "manifest.json").read_bytes(), original_manifest)


if __name__ == "__main__":
    unittest.main()
