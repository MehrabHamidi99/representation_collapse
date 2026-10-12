import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qwen_evidence", ROOT / "code/ops/qwen_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class QwenEvidenceTests(unittest.TestCase):
    def test_actual_protocol_location_and_private_file_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs/experiments").mkdir(parents=True)
            (root / "doc/auto").mkdir(parents=True)
            public = {
                "configs/experiments/initial-phase-v1.json": '{"phase":"initial"}',
                "configs/experiments/competence-recovery-v2.json": '{"protocol":"competence"}',
                "doc/auto/competence-training-v2.md": "Native addressed recall; normalized MSE uses population variance1/3.",
            }
            for name, value in public.items(): (root / name).write_text(value)
            for name in ("approval.local.json", "provider.json", "phase1-private.json"):
                (root / "configs" / name).write_text("DO_NOT_SEND_PRIVATE_CONFIG")
            items, ids = worker.evidence(root, [], root / "worker", [], "", "tamia")
            for name, value in public.items(): self.assertEqual(items[name], value)
            self.assertNotIn("DO_NOT_SEND_PRIVATE_CONFIG", json.dumps(items))
            self.assertEqual(ids, [])

    def test_protocol_symlink_is_never_followed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs/experiments").mkdir(parents=True)
            private = root / "private.json"
            private.write_text("DO_NOT_SEND_SYMLINK_TARGET")
            (root / "configs/experiments/competence-recovery-v2.json").symlink_to(private)
            items, _ = worker.evidence(root, [], root / "worker", [], "", "tamia")
            self.assertNotIn("configs/experiments/competence-recovery-v2.json", items)
            self.assertNotIn("DO_NOT_SEND_SYMLINK_TARGET", json.dumps(items))

    def test_larger_public_context_remains_inside_approved_request_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("doc/TRACKER.md", "configs/research-plan.json", "doc/literature/novelty-review.md",
                         "doc/auto/competence-training-v2.md", "configs/experiments/initial-phase-v1.json",
                         "configs/experiments/competence-recovery-v2.json"):
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("public-protocol " * 1000)
            artifacts = root / "runs"
            for index in range(12):
                path = artifacts / str(index) / "metrics.json"
                path.parent.mkdir(parents=True); path.write_text("numeric-summary " * 100)
            items, _ = worker.evidence(root, [artifacts], root / "worker", [], "", "tamia")
            body = worker.request_body("Qwen/Qwen3.8-Flash-Next-FP8", items, "evidence_change")
            self.assertLessEqual(sum(len(m["content"]) for m in body["messages"]), 18000)
            self.assertEqual(body["max_tokens"], 2048)
            self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])

    def test_missing_public_protocols_leave_bounded_old_checkout_usable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            items, _ = worker.evidence(root, [], root / "worker", [], "", "tamia")
            self.assertEqual(set(items), {"scheduler:known-project-jobs"})
            self.assertIn("addressed memory recall", worker.SYSTEM)


if __name__ == "__main__":
    unittest.main()
