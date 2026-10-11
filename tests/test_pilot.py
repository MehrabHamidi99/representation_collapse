import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from representation_collapse.vector_task import paired_batch


@unittest.skipIf(torch is None, "Torch requires the experiments dependency group")
class PilotTests(unittest.TestCase):
    def test_complete_lstm_state_and_query_timing(self):
        from representation_collapse.pilot import VectorMemoryModel
        torch.set_num_threads(1)
        batch = paired_batch(np.random.default_rng(2), 3, 2, "continuous")
        inputs = torch.as_tensor(batch.inputs)
        for kind, expected in [("gru", 8), ("lstm", 16)]:
            model = VectorMemoryModel(kind, "continuous", 8, 8)
            state = model.pre_query_state(inputs)
            packed = model.pack_state(state)
            self.assertEqual(tuple(packed.shape), (6, expected))
            if kind == "lstm":
                torch.testing.assert_close(packed[:, :8], state[0][0])
                torch.testing.assert_close(packed[:, 8:], state[1][0])
            torch.testing.assert_close(model(inputs), model.read_query(inputs[:, -1], state), rtol=1e-5, atol=1e-6)
            changed = inputs.clone()
            changed[:, -1, 5:21] = 0
            changed[:, -1, 12] = 1
            torch.testing.assert_close(packed, model.pack_state(model.pre_query_state(changed)))

    def test_probe_recovers_known_linear_memory_and_label_control(self):
        from representation_collapse.pilot import ridge_probe
        rng = np.random.default_rng(0)
        train = rng.uniform(-1, 1, (512, 8))
        test = rng.uniform(-1, 1, (256, 8))
        results, _ = ridge_probe(train, test, train, test, "continuous", 0.001, rng)
        self.assertLess(results["real"]["mean_mse"], 0.0001)
        self.assertGreater(results["label_control"]["mean_mse"], 0.2)

    def test_actual_optimization_reduces_fixed_batch_loss(self):
        from representation_collapse.pilot import VectorMemoryModel
        torch.set_num_threads(1)
        torch.manual_seed(0)
        batch = paired_batch(np.random.default_rng(0), 2, 1, "categorical")
        x, y = torch.as_tensor(batch.inputs), torch.as_tensor(batch.targets)
        for kind in ["gru", "lstm"]:
            model = VectorMemoryModel(kind, "categorical", 8, 8)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
            initial = float(torch.nn.functional.cross_entropy(model(x), y).detach())
            for _ in range(20):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()
            final = float(torch.nn.functional.cross_entropy(model(x), y).detach())
            self.assertLess(final, initial * 0.8)

    def test_tiny_run_complete_artifacts_and_paired_contrast(self):
        from representation_collapse.pilot import run
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs/experiments/vector-pilot-v1.json").read_text())
        config.update(hidden_size=8, input_bottleneck=8, training_steps=2, batch_size=4,
                      validation_pairs_per_horizon=2, evaluation_pairs_per_horizon=4,
                      cloud_driver_pairs_per_horizon=1, cloud_samples=8, probe_test_samples=8)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            metrics = run(config, output, "lstm", "continuous", 0, max_steps=1)
            self.assertFalse(metrics["scientific_acceptance"])
            self.assertEqual(metrics["state"]["dimension"], 16)
            self.assertEqual(metrics["state"]["persistent_fp32_bytes_per_sequence"], 64)
            self.assertAlmostEqual(metrics["difference_in_differences"], metrics["delta16"] - metrics["delta2"])
            with np.load(output / "paired_errors.npz") as raw:
                np.testing.assert_array_equal(raw["R1_periodic_errors"], raw["R1_control_errors"])
                for r in [2, 16]:
                    delta = (raw[f"R{r}_periodic_errors"] - raw[f"R{r}_control_errors"]).mean()
                    self.assertAlmostEqual(float(delta), metrics["behavior"][str(r)]["paired_error_difference_mean"])
            with np.load(output / "cue_clouds.npz") as clouds:
                for r in [2, 16]:
                    self.assertEqual(clouds[f"R{r}_panel0_periodic_train_states"].shape, (8, 16))
                    train = clouds[f"R{r}_panel0_periodic_train_memory"]
                    test = clouds[f"R{r}_panel0_periodic_test_memory"]
                    self.assertFalse(np.array_equal(train[:, :8], test[:, :8]))
                    np.testing.assert_array_equal(train[:, 8:], np.broadcast_to(train[0, 8:], train[:, 8:].shape))
            self.assertEqual(json.loads((output / "status.json").read_text())["state"], "completed")
            self.assertTrue(json.loads((output / "config.json").read_text())["max_steps_override"])
            manifest = json.loads((output / "manifest.json").read_text())["sha256"]
            for name, digest in manifest.items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            self.assertTrue({"checkpoint.pt", "history.json", "config.json", "provenance.json", "metrics.json", "paired_errors.npz", "cue_clouds.npz", "status.json"}.issubset(manifest))
            with self.assertRaises(FileExistsError):
                run(config, output, "lstm", "continuous", 0, max_steps=1)


if __name__ == "__main__":
    unittest.main()
