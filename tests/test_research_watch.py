import importlib.util
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("research_watch", ROOT / "code/ops/research_watch.py")
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


class ResearchWatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name).resolve()
        self.worker = self.parent / "existing-worker"
        self.worker.mkdir()
        self.now = datetime(2026, 10, 12, 22, tzinfo=timezone.utc)
        self.config = {"kind": "qwen_research_leaf_worker", "schema_version": 1,
                       "requested_model": watch.EXPECTED_MODEL, "max_calls": 24,
                       "max_hours": 24, "autonomous_actions": "structured_proposals_only",
                       "worker_source_sha256": "a" * 64}
        self.heartbeat = {"schema_version": 1, "state": "running", "updated_at": self.now.isoformat(),
                          "elapsed_seconds": 60, "provider_call_attempts": 0,
                          "schema_verified_reviews": 0, "consecutive_provider_errors": 0,
                          "usage_totals": {name: 0 for name in watch.TOKEN_FIELDS}}
        self.write("config.json", self.config)
        self.write("heartbeat.json", self.heartbeat)
        watch._STOP = False

    def write(self, name, packet):
        path = self.worker / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(packet))

    def observe(self, **kwargs):
        return watch.observe_worker(self.worker, now=self.now, worker_host="tamia3.tamia.ecpia.ca",
                                    observer_host="tamia1", worker_pid=524659, **kwargs)

    def add_call(self, number=1, state="proposal_ready", reported=None):
        self.write(f"call-{number:04d}/status.json", {
            "state": state, "requested_model": watch.EXPECTED_MODEL,
            "returned_model": watch.EXPECTED_MODEL, "model_identity_verified": True,
            "actions_executed": 0, "scientific_decision": "pending"})
        if state != "calling":
            self.write(f"call-{number:04d}/usage.json", {"reported": reported})

    def terminal(self, reason="termination_signal"):
        self.heartbeat.update(state="stopped", stop_reason=reason)
        self.write("heartbeat.json", self.heartbeat)
        self.write("status.json", {
            "state": "stopped", "stop_reason": reason,
            "exit_code": 1 if reason == "three_consecutive_provider_errors" else 0,
            "finished_at": self.now.isoformat(),
            "provider_call_attempts": self.heartbeat["provider_call_attempts"],
            "schema_verified_reviews": self.heartbeat["schema_verified_reviews"],
            "usage_totals": self.heartbeat["usage_totals"]})

    def test_fresh_heartbeat_wrong_host_never_probes_pid(self):
        with patch.object(watch.os, "kill", side_effect=AssertionError("wrong-host PID check")):
            result = self.observe()
        self.assertEqual(result["state"], "worker_heartbeat_fresh_process_unverified")
        self.assertIsNone(result["process_probe"]["process_alive"])
        self.assertFalse(result["worker_liveness_certified"])

    def test_stale_or_future_heartbeat_never_certifies_activity(self):
        self.heartbeat["updated_at"] = (self.now - timedelta(seconds=601)).isoformat()
        self.write("heartbeat.json", self.heartbeat)
        self.assertEqual(self.observe()["state"], "worker_heartbeat_stale")
        self.heartbeat["updated_at"] = (self.now + timedelta(seconds=1)).isoformat()
        self.write("heartbeat.json", self.heartbeat)
        self.assertIn("future_heartbeat_timestamp", self.observe()["issues"])

    def test_terminal_is_verified_even_when_old_but_never_proves_process_absence(self):
        self.heartbeat["updated_at"] = (self.now - timedelta(hours=2)).isoformat()
        self.terminal()
        result = self.observe()
        self.assertEqual(result["state"], "worker_stopped")
        self.assertTrue(result["terminal"]["verified"])
        self.assertFalse(result["heartbeat_fresh"])
        self.assertIsNone(result["process_probe"]["process_alive"])

    def test_stopped_heartbeat_requires_terminal_and_disagreements_are_rejected(self):
        self.heartbeat.update(state="stopped")
        self.write("heartbeat.json", self.heartbeat)
        self.assertEqual(self.observe()["state"], "stopped_heartbeat_terminal_pending")
        self.terminal()
        packet = json.loads((self.worker / "status.json").read_text())
        packet["provider_call_attempts"] = 1
        self.write("status.json", packet)
        self.assertEqual(self.observe()["state"], "metadata_inconsistent")
        self.assertIn("terminal_heartbeat_disagreement", self.observe()["issues"])

    def test_model_source_and_envelope_mismatch_fail_closed_without_echo(self):
        self.config["requested_model"] = "PRIVATE_ENDPOINT_SECRET"
        self.config["max_calls"] = 25
        self.write("config.json", self.config)
        result = self.observe(expected_source_sha256="b" * 64)
        self.assertEqual(result["state"], "metadata_inconsistent")
        self.assertIn("worker_source_mismatch", result["issues"])
        self.assertNotIn("PRIVATE_ENDPOINT_SECRET", json.dumps(result))

    def test_attempt_success_and_usage_must_reconcile(self):
        self.add_call(reported={name: 5 for name in watch.TOKEN_FIELDS})
        self.heartbeat.update(provider_call_attempts=1, schema_verified_reviews=1,
                              usage_totals={name: 5 for name in watch.TOKEN_FIELDS})
        self.write("heartbeat.json", self.heartbeat)
        self.assertEqual(self.observe()["issues"], [])
        self.heartbeat["usage_totals"]["total_tokens"] = 6
        self.write("heartbeat.json", self.heartbeat)
        self.assertIn("heartbeat_call_usage_disagree", self.observe()["issues"])
        self.heartbeat["provider_call_attempts"] = True
        self.write("heartbeat.json", self.heartbeat)
        self.assertIsNone(self.observe()["provider_call_attempts"])
        self.assertIn("invalid_heartbeat_counters", self.observe()["issues"])

    def test_failed_call_unknown_usage_and_terminal_failure_are_not_zero(self):
        for number in range(1, 4):
            self.add_call(number, state="failed")
        self.heartbeat.update(provider_call_attempts=3, schema_verified_reviews=0,
                              consecutive_provider_errors=3,
                              usage_totals={name: None for name in watch.TOKEN_FIELDS})
        self.terminal("three_consecutive_provider_errors")
        result = self.observe()
        self.assertEqual(result["state"], "worker_stopped_failure")
        self.assertIsNone(result["usage_totals"]["total_tokens"])

    def test_stop_reason_requires_its_actual_bound_or_failure_streak(self):
        self.terminal("call_cap")
        self.assertFalse(self.observe()["terminal"]["verified"])
        self.terminal("runtime_cap")
        self.assertFalse(self.observe()["terminal"]["verified"])
        self.terminal("three_consecutive_provider_errors")
        self.assertFalse(self.observe()["terminal"]["verified"])

    def test_inflight_call_has_no_completed_usage_requirement(self):
        self.add_call(state="calling")
        self.heartbeat["provider_call_attempts"] = 1
        self.write("heartbeat.json", self.heartbeat)
        self.assertEqual(self.observe()["issues"], [])
        self.assertIsNone(self.observe()["calls"][0]["reported_usage"]["total_tokens"])

    def test_terminal_with_inflight_attempt_never_verifies_stopped_worker(self):
        self.add_call(state="calling")
        self.heartbeat["provider_call_attempts"] = 1
        self.terminal()
        result = self.observe()
        self.assertEqual(result["state"], "metadata_inconsistent")
        self.assertFalse(result["terminal"]["verified"])
        self.assertIn("terminal_heartbeat_disagreement", result["issues"])

    def test_failed_tail_requires_matching_heartbeat_streak(self):
        self.add_call(state="failed", reported={name: 0 for name in watch.TOKEN_FIELDS})
        self.heartbeat["provider_call_attempts"] = 1
        self.write("heartbeat.json", self.heartbeat)
        result = self.observe()
        self.assertEqual(result["state"], "metadata_inconsistent")
        self.assertIn("heartbeat_completed_error_streak_disagrees", result["issues"])
        self.heartbeat["consecutive_provider_errors"] = 1
        self.write("heartbeat.json", self.heartbeat)
        self.assertEqual(self.observe()["issues"], [])

    def test_pending_retry_preserves_previous_failure_streak_and_success_resets_it(self):
        self.add_call(1, state="failed", reported={name: 0 for name in watch.TOKEN_FIELDS})
        self.add_call(2, state="calling")
        self.heartbeat.update(provider_call_attempts=2, consecutive_provider_errors=1)
        self.write("heartbeat.json", self.heartbeat)
        self.assertEqual(self.observe()["issues"], [])
        self.add_call(2, state="proposal_ready", reported={name: 0 for name in watch.TOKEN_FIELDS})
        self.heartbeat.update(schema_verified_reviews=1, consecutive_provider_errors=0)
        self.write("heartbeat.json", self.heartbeat)
        self.assertEqual(self.observe()["issues"], [])
        self.heartbeat["consecutive_provider_errors"] = 1
        self.write("heartbeat.json", self.heartbeat)
        self.assertIn("heartbeat_completed_error_streak_disagrees", self.observe()["issues"])

    def test_finished_attempt_status_before_new_heartbeat_is_conservatively_inconsistent(self):
        self.add_call(1, state="failed", reported={name: 0 for name in watch.TOKEN_FIELDS})
        self.add_call(2, state="calling")
        self.heartbeat.update(provider_call_attempts=2, consecutive_provider_errors=1)
        self.write("heartbeat.json", self.heartbeat)
        self.add_call(2, state="failed", reported={name: 0 for name in watch.TOKEN_FIELDS})
        self.assertIn("heartbeat_completed_error_streak_disagrees", self.observe()["issues"])
        self.heartbeat["consecutive_provider_errors"] = 2
        self.write("heartbeat.json", self.heartbeat)
        self.assertEqual(self.observe()["issues"], [])

    def test_inflight_attempt_must_be_the_final_journal_entry(self):
        self.add_call(1, state="calling")
        self.add_call(2, reported={name: 0 for name in watch.TOKEN_FIELDS})
        self.heartbeat.update(provider_call_attempts=2, schema_verified_reviews=1)
        self.write("heartbeat.json", self.heartbeat)
        self.assertIn("inflight_call_is_not_unique_final_attempt", self.observe()["issues"])

    def test_privacy_allowlist_never_reads_request_reply_review_or_error(self):
        secret = "SENTINEL_DO_NOT_READ_OR_PERSIST"
        self.add_call(state="failed")
        for name in ("request.json", "evidence.json", "raw-reply.json", "review.json", "raw-error.txt"):
            (self.worker / "call-0001" / name).write_text(secret)
        self.write("call-0001/status.json", {
            "state": "failed", "requested_model": watch.EXPECTED_MODEL,
            "returned_model": secret, "error": secret, "provider_endpoint": secret})
        self.config.update(provider_endpoint=secret, api_key=secret)
        self.write("config.json", self.config)
        self.heartbeat.update(provider_call_attempts=1, usage_totals={name: None for name in watch.TOKEN_FIELDS},
                              consecutive_provider_errors=1, arbitrary_secret=secret)
        self.write("heartbeat.json", self.heartbeat)
        opened = []
        actual = watch.read_metadata
        def record(path):
            opened.append(path.name)
            return actual(path)
        with patch.object(watch, "read_metadata", side_effect=record):
            result = self.observe()
        self.assertEqual(set(opened), {"config.json", "heartbeat.json", "status.json", "usage.json"})
        self.assertNotIn(secret, json.dumps(result))

    def test_metadata_symlinks_oversize_and_call_expansion_are_rejected(self):
        config = self.worker / "config.json"
        config.unlink()
        config.symlink_to(self.worker / "heartbeat.json")
        self.assertEqual(self.observe()["metadata_reads"]["config"], "unreadable_or_invalid")
        config.unlink()
        config.write_text("x" * (watch.MAX_METADATA_BYTES + 1))
        self.assertEqual(self.observe()["metadata_reads"]["config"], "unsafe_or_oversized")
        (self.worker / "call-0025").mkdir()
        self.assertIn("out_of_envelope_worker_journal", self.observe()["issues"])

    def fake_run(self, *, rollback=False):
        state = {"mono": 0.0}
        def now():
            offset = -state["mono"] if rollback else state["mono"]
            return self.now + timedelta(seconds=offset)
        def sleep(seconds):
            state["mono"] += seconds
        result = watch.run_watch(worker=self.worker, output=self.parent / "observer",
            deadline=self.now + timedelta(seconds=120), worker_host="tamia3", worker_pid=524659,
            interval=60, now_fn=now, monotonic_fn=lambda: state["mono"], sleep_fn=sleep,
            observer_host="tamia1")
        return result, state

    def test_absolute_deadline_unique_output_and_no_polling_drift(self):
        with patch.object(watch.subprocess, "run", side_effect=AssertionError("no optional scheduler requested")):
            result, state = self.fake_run()
        self.assertEqual(result["observations"], 2)
        self.assertEqual(state["mono"], 120)
        self.assertEqual(result["stop_reason"], "deadline")
        self.assertFalse(result["worker_terminated_by_observer"])
        records = [json.loads(line) for line in (self.parent / "observer/observations.jsonl").read_text().splitlines()]
        self.assertEqual([record["observation_number"] for record in records], [1, 2])
        self.assertEqual(watch.utc(records[1]["observed_at_utc"]) - watch.utc(records[0]["observed_at_utc"]), timedelta(seconds=60))
        with self.assertRaises(FileExistsError):
            self.fake_run()

    def test_monotonic_deadline_stops_even_if_wall_clock_moves_backwards(self):
        result, state = self.fake_run(rollback=True)
        self.assertEqual(result["stop_reason"], "deadline")
        self.assertEqual(state["mono"], 120)

    def test_collection_failure_writes_sanitized_terminal_without_worker_action(self):
        with patch.object(watch, "observe_worker", side_effect=RuntimeError("PRIVATE_SECRET")):
            result, _ = self.fake_run()
        self.assertEqual(result["stop_reason"], "metadata_observation_error")
        self.assertEqual(result["observations"], 0)
        self.assertEqual(result["provider_calls_by_observer"], 0)
        self.assertNotIn("PRIVATE_SECRET", (self.parent / "observer/status.json").read_text())

    def test_expired_unbounded_or_fast_polling_rejected_before_output_creation(self):
        for seconds, interval in ((0, 300), (-1, 300), (86401, 300), (120, 59), (120, float("nan"))):
            with self.assertRaises(ValueError):
                watch.run_watch(worker=self.worker, output=self.parent / "bad",
                    deadline=self.now + timedelta(seconds=seconds), worker_host="tamia3",
                    interval=interval, now_fn=lambda: self.now)
        self.assertFalse((self.parent / "bad").exists())

    def test_duplicate_observer_lock_and_later_reuse(self):
        with watch.observer_lock(self.parent, self.worker):
            with self.assertRaises(ValueError):
                with watch.observer_lock(self.parent, self.worker):
                    self.fail("duplicate lock succeeded")
        with watch.observer_lock(self.parent, self.worker):
            pass

    def test_local_scheduler_discards_other_projects_and_does_not_open_ssh(self):
        reply = SimpleNamespace(returncode=0, stdout="612745|RUNNING|rc-audit\n99|RUNNING|rc-unrelated\n612745|RUNNING|jepa-other\n", stderr="PRIVATE_SENTINEL")
        with patch.object(watch.subprocess, "run", return_value=reply) as invoke:
            result = watch.local_scheduler(["612745"])
        self.assertEqual(result["jobs"], [{"job_id": "612745", "state": "RUNNING", "name": "rc-audit"}])
        command = invoke.call_args.args[0]
        self.assertEqual(command[0], "squeue")
        self.assertIn("--me", command)
        self.assertIn("--jobs=612745", command)
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(result))
        self.assertFalse(result["absence_means_completed"])
        with self.assertRaises(ValueError):
            watch.local_scheduler(["1;sbatch"])


if __name__ == "__main__":
    unittest.main()
