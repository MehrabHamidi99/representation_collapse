"""Offline behavioral fixtures: no provider, cluster, credential or network use."""

from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qwen_continuous", ROOT / "code/ops/qwen_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
MODEL = "Qwen/offline-test-fixture"
KEY = "OFFLINE_FIXTURE_ONLY_NOT_A_CREDENTIAL"


class Clock:
    def __init__(self):
        self.elapsed = 0.0
        self.on_sleep = lambda: None

    def monotonic(self):
        return self.elapsed

    def sleep(self, duration):
        self.elapsed += duration
        self.on_sleep()


def valid_reply(body, *, model=None, omit_usage=False):
    sources = json.loads(body["messages"][1]["content"])["evidence"]
    review = {"assessment": "Offline test proposal.",
              "evidence": [{"source": next(iter(sources)), "claim": "Fixture only."}],
              "next_experiment": {"question": "Fixture?", "design": "Validate arithmetic.",
                                  "resources": "Local fixture only.", "stopping_rule": "End fixture."},
              "acceptance_checks": ["Independent check."], "limitations": ["No real measurements."]}
    envelope = {"model": body["model"] if model is None else model,
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(review)}}]}
    if not omit_usage:
        envelope["usage"] = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    return json.dumps(envelope)


class ContinuousWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.output = self.root / "new-run"
        self.clock = Clock()
        self.requests, self.heartbeats = [], []

    def tearDown(self):
        self.temporary.cleanup()

    def run_worker(self, provider, *, evidence=None, flags=None):
        evidence = evidence or (lambda *args: ({"fixture": '{"step":1}'}, []))
        real_atomic = worker.atomic_json

        def capture_json(path, data):
            if path.name == "heartbeat.json":
                self.heartbeats.append(deepcopy(data))
            real_atomic(path, data)

        def fake_provider(base_url, key, body, timeout):
            self.requests.append({"started": self.clock.elapsed, "body": deepcopy(body), "timeout": timeout})
            return provider(body, len(self.requests))

        argv = ["--project", str(self.project), "--output", str(self.output), "--cluster", "tamia"]
        argv += ["--continuous"] if flags is None else flags
        with patch.dict(worker.os.environ, {"QWEN_BASE_URL": "http://offline.invalid/v1",
                        "QWEN_MODEL": MODEL, "CAIRN_API_KEY": KEY}), \
             patch.object(worker, "provider_call", fake_provider), \
             patch.object(worker, "evidence", evidence), \
             patch.object(worker, "atomic_json", capture_json), \
             patch.object(worker.signal, "signal"), \
             patch.object(worker.time, "monotonic", self.clock.monotonic), \
             patch.object(worker.time, "sleep", self.clock.sleep):
            worker.main(argv)

    def final(self):
        return json.loads((self.output / "status.json").read_text())

    def stop(self):
        (self.output / "STOP").write_text("Owner stop fixture.\n")

    def test_continuous_exceeds_24_calls_and_has_null_total_limits(self):
        def evidence(*args):
            return {"fixture": json.dumps({"step": int(self.clock.elapsed // 60)})}, []

        def provider(body, count):
            if count == 28:
                self.stop()
            return valid_reply(body)

        self.run_worker(provider, evidence=evidence)
        self.assertEqual(len(self.requests), 28)
        self.assertTrue((self.output / "call-0028/review.json").is_file())
        self.assertTrue(all(b["started"] - a["started"] >= 60
                            for a, b in zip(self.requests, self.requests[1:])))
        for packet in (json.loads((self.output / "config.json").read_text()), self.heartbeats[-1], self.final()):
            self.assertTrue(packet["continuous"])
            self.assertEqual(packet["execution_mode"], "continuous")
            self.assertIsNone(packet["max_calls"])
            self.assertIsNone(packet["max_hours"])
        self.assertEqual(self.final()["usage_totals"]["total_tokens"], 28 * 15)

    def test_unchanged_evidence_does_not_reason_hourly_or_stop_after_24_hours(self):
        def sleep_hook():
            self.clock.elapsed += 3600
            if self.clock.elapsed >= 26 * 3600:
                self.stop()
        self.clock.on_sleep = sleep_hook

        def evidence(*args):
            return {"fixture/status.json": json.dumps({"state": "running", "step": 1,
                "updated_at": str(self.clock.elapsed), "elapsed_seconds": self.clock.elapsed})}, []

        self.run_worker(lambda body, count: valid_reply(body), evidence=evidence)
        self.assertEqual(len(self.requests), 1)
        self.assertGreaterEqual(self.heartbeats[-1]["elapsed_seconds"], 26 * 3600)
        self.assertEqual(self.final()["stop_reason"], "stop_file_requested")

    def test_errors_back_off_remain_observable_and_recover_without_error_count_stop(self):
        def provider(body, count):
            if count <= 4:
                raise TimeoutError("offline failure " + KEY)
            self.stop()
            return valid_reply(body)

        self.run_worker(provider, flags=["--continuous", "--retry-base", "60", "--retry-maximum", "240"])
        self.assertEqual([request["started"] for request in self.requests], [0, 60, 180, 420, 660])
        self.assertTrue(any(p["state"] == "degraded" and p["consecutive_provider_errors"] == 1
                            for p in self.heartbeats))
        self.assertTrue(any(p["state"] == "stalled" and p["consecutive_provider_errors"] == 4
                            and p["retry_backoff_seconds"] == 240 for p in self.heartbeats))
        self.assertEqual(self.final()["schema_verified_reviews"], 1)
        self.assertEqual(self.heartbeats[-1]["consecutive_provider_errors"], 0)
        self.assertEqual(self.final()["usage_totals"], dict.fromkeys(("prompt_tokens", "completion_tokens", "total_tokens")))
        self.assertFalse(json.loads((self.output / "config.json").read_text())["http_provider_fallback"])
        self.assertNotIn(KEY, (self.output / "call-0001/status.json").read_text())

    def test_STOP_interrupts_retry_wait_and_signal_records_distinct_reason(self):
        self.clock.on_sleep = lambda: self.stop() if self.clock.elapsed >= 2 else None
        self.run_worker(lambda body, count: (_ for _ in ()).throw(TimeoutError("fixture")))
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.final()["stop_reason"], "stop_file_requested")
        self.assertLess(self.clock.elapsed, 60)

        self.output = self.root / "signal-run"
        self.clock = Clock()
        self.requests = []
        def provider(body, count):
            worker._STOP = True
            return valid_reply(body)
        self.run_worker(provider)
        self.assertEqual(self.final()["stop_reason"], "termination_signal")

    def test_new_task_triggers_once_on_unchanged_evidence_and_checks_exact_payload_hash(self):
        def provider(body, count):
            if count == 1:
                worker.atomic_json(self.output / "tasks/check-controls.json",
                                   {"task_id": "check-controls", "task": "Review the stated sham controls."})
            if count == 2:
                # Keep polling after successful consumption: the same durable
                # task must not trigger another request when still in the inbox.
                self.clock.on_sleep = lambda: self.stop() if self.clock.elapsed >= 3600 else None
            return valid_reply(body)

        self.run_worker(provider)
        self.assertEqual(len(self.requests), 2)
        task_request = json.loads(self.requests[1]["body"]["messages"][1]["content"])
        self.assertEqual(task_request["trigger"], "coordinator_task")
        self.assertIn("coordinator:explicit-research-task", task_request["evidence"])
        packet = json.loads((self.output / "call-0002/evidence.json").read_text())
        self.assertEqual(packet["sha256"], worker.digest(json.dumps(packet["items"], sort_keys=True)))
        self.assertEqual(self.heartbeats[-1]["completed_coordinator_tasks"], 1)
        self.assertEqual(self.heartbeats[-1]["last_completed_task_id"], "check-controls")
        self.assertEqual(json.loads((self.output / "call-0002/review.json").read_text())["evidence_sha256"], packet["sha256"])

    def test_rapid_changing_evidence_obeys_minimum_call_spacing(self):
        def evidence(*args):
            return {"fixture": json.dumps({"step": self.clock.elapsed})}, []
        def provider(body, count):
            if count == 3:
                self.stop()
            return valid_reply(body)
        self.run_worker(provider, evidence=evidence, flags=["--continuous", "--interval", "1"])
        self.assertEqual([request["started"] for request in self.requests], [0, 60, 120])

    def test_bounded_default_limits_and_three_error_stop_are_preserved(self):
        with self.assertRaises(SystemExit) as stopped:
            self.run_worker(lambda body, count: (_ for _ in ()).throw(TimeoutError("fixture")), flags=[])
        self.assertEqual(stopped.exception.code, 1)
        config = json.loads((self.output / "config.json").read_text())
        self.assertEqual(config["max_calls"], 24)
        self.assertEqual(config["max_hours"], 24)
        self.assertEqual(config["research_interval_seconds"], 3600)
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(self.final()["stop_reason"], "three_consecutive_provider_errors")

    def test_bounded_hourly_trigger_and_custom_call_cap_are_preserved(self):
        self.run_worker(lambda body, count: valid_reply(body), flags=["--max-calls", "2", "--research-interval", "3600"])
        self.assertEqual([request["started"] for request in self.requests], [0, 3600])
        self.assertEqual(self.final()["stop_reason"], "call_cap")

    def test_existing_output_is_not_reused_and_continuous_rejects_cap_flags(self):
        for flags in (["--continuous", "--max-calls", "24"], ["--continuous", "--max-hours", "24"],
                      ["--continuous", "--research-interval", "3600"],
                      ["--continuous", "--minimum-call-spacing", ".1"],
                      ["--continuous", "--retry-base", "120", "--retry-maximum", "60"]):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as invalid:
                self.run_worker(lambda body, count: valid_reply(body), flags=flags)
            self.assertEqual(invalid.exception.code, 2)
            self.assertFalse(self.output.exists())
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_worker(lambda body, count: valid_reply(body))
        self.assertEqual(self.requests, [])

    def test_model_identity_and_truncated_answer_fail_without_fallback(self):
        def provider(body, count):
            if count == 1:
                return valid_reply(body, model="OpenAI/not-a-fallback")
            self.stop()
            envelope = json.loads(valid_reply(body))
            envelope["choices"][0]["finish_reason"] = "length"
            return json.dumps(envelope)
        self.run_worker(provider)
        self.assertEqual(len(self.requests), 2)
        self.assertTrue(all(request["body"]["model"] == MODEL for request in self.requests))
        self.assertEqual(self.final()["schema_verified_reviews"], 0)
        self.assertEqual(self.final()["usage_totals"]["total_tokens"], 30)
        self.assertFalse((self.output / "call-0001/review.json").exists())
        self.assertFalse((self.output / "call-0002/review.json").exists())

    def test_broken_HTTP_error_body_does_not_lose_error_metadata_or_abort_retry(self):
        def provider(body, count):
            if count == 1:
                error = worker.urllib.error.HTTPError("http://offline.invalid/v1", 503, "fixture", {}, None)
                error.read = lambda limit: (_ for _ in ()).throw(OSError("broken offline stream"))
                raise error
            self.stop()
            return valid_reply(body)
        self.run_worker(provider)
        failed = json.loads((self.output / "call-0001/status.json").read_text())
        self.assertEqual(failed["state"], "failed")
        self.assertEqual(failed["error_body_state"], "unavailable:OSError")
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.final()["schema_verified_reviews"], 1)
        self.assertIsNone(self.final()["usage_totals"]["total_tokens"])

    def test_malformed_list_or_integer_usage_is_unknown_without_losing_valid_review(self):
        def evidence(*args):
            return {"fixture": json.dumps({"step": self.clock.elapsed})}, []
        def provider(body, count):
            envelope = json.loads(valid_reply(body))
            envelope["usage"] = [] if count == 1 else 123
            if count == 2:
                self.stop()
            return json.dumps(envelope)
        self.run_worker(provider, evidence=evidence)
        self.assertEqual(self.final()["schema_verified_reviews"], 2)
        self.assertEqual(self.final()["usage_totals"], dict.fromkeys(("prompt_tokens", "completion_tokens", "total_tokens")))
        for count in (1, 2):
            path = self.output / f"call-{count:04d}"
            usage = json.loads((path / "usage.json").read_text())
            self.assertTrue(usage["missing_usage_is_unknown"])
            self.assertIsNone(usage["reported"])
            self.assertTrue((path / "review.json").is_file())


class ContinuousTriggerTests(unittest.TestCase):
    def test_semantic_hash_ignores_clocks_but_detects_new_scientific_state_and_metrics(self):
        first = {"artifact": json.dumps({"state": "running", "metrics": {"loss": .2}, "updated_at": "old",
                                        "elapsed_seconds": 1})}
        updated = {"artifact": json.dumps({"state": "running", "metrics": {"loss": .2}, "updated_at": "new",
                                          "elapsed_seconds": 200})}
        self.assertEqual(worker.meaningful_evidence_hash(first), worker.meaningful_evidence_hash(updated))
        for changed in ({"state": "completed", "metrics": {"loss": .2}},
                        {"state": "running", "metrics": {"loss": .1}}):
            self.assertNotEqual(worker.meaningful_evidence_hash(first), worker.meaningful_evidence_hash({"artifact": json.dumps(changed)}))

    def test_truncated_JSON_clock_fields_are_normalized_but_loss_values_are_retained(self):
        first = {"artifact": '{"updated_at":"old", "elapsed_seconds":1, "loss":0.2, "unfinished":'}
        second = {"artifact": '{"updated_at":"new", "elapsed_seconds":999, "loss":0.2, "unfinished":'}
        self.assertEqual(worker.meaningful_evidence_hash(first), worker.meaningful_evidence_hash(second))
        self.assertNotEqual(worker.meaningful_evidence_hash(first),
                            worker.meaningful_evidence_hash({"artifact": second["artifact"].replace('"loss":0.2', '"loss":0.1')}))

    def test_task_schema_scrubbing_uniqueness_and_symlink_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            inbox = Path(temporary) / "tasks"
            inbox.mkdir()
            private = Path(temporary) / "private.json"
            private.write_text(json.dumps({"task_id": "a", "task": "DO_NOT_SEND"}))
            (inbox / "a.json").symlink_to(private)
            worker.atomic_json(inbox / "b.json", {"task_id": "b", "task": "public question " + KEY})
            identities = {}
            task, issues = worker.coordinator_tasks(inbox, set(), identities, KEY)
            self.assertEqual(task["task_id"], "b")
            self.assertNotIn(KEY, json.dumps(task))
            self.assertNotIn("DO_NOT_SEND", json.dumps(task))
            self.assertTrue(issues)
            worker.atomic_json(inbox / "b.json", {"task_id": "b", "task": "changed under old identity"})
            task, issues = worker.coordinator_tasks(inbox, set(), identities, KEY)
            self.assertIsNone(task)
            self.assertTrue(issues)
            worker.atomic_json(inbox / "c.json", {"task_id": "wrong", "task": "invalid filename mismatch"})
            task, issues = worker.coordinator_tasks(inbox, {"b"}, identities, KEY)
            self.assertIsNone(task)
            self.assertTrue(issues)

    def test_exponential_retry_delay_saturates_without_limited_attempt_count(self):
        self.assertEqual([worker.retry_delay(n, 60, 3600) for n in range(1, 9)],
                         [60, 120, 240, 480, 960, 1920, 3600, 3600])
        self.assertEqual(worker.retry_delay(1000000, 60, 3600), 3600)


if __name__ == "__main__":
    unittest.main()
