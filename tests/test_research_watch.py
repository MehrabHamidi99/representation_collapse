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


class ContinuousResearchWatchTests(unittest.TestCase):
    write = ResearchWatchTests.write

    def setUp(self):
        ResearchWatchTests.setUp(self)
        self.config.update(schema_version=2, execution_mode="continuous", continuous=True,
            max_calls=None, max_hours=None, research_interval_seconds=None,
            trigger_policy="semantic_evidence_change_or_unique_coordinator_task_plus_backoff_retry",
            minimum_provider_call_spacing_seconds=60,
            retry_policy={"base_seconds": 60, "maximum_delay_seconds": 3600,
                          "consecutive_error_stop_count": None, "stalled_after_errors": 3},
            context_character_budget=18000, max_tokens=2048, scientific_acceptance="coordinator_only",
            http_provider_fallback=False)
        self.heartbeat.update(schema_version=2, execution_mode="continuous", continuous=True,
            max_calls=None, max_hours=None, activity="waiting_for_evidence", provider_call_in_flight=False,
            retry_backoff_seconds=0, next_provider_call_not_before=None,
            completed_coordinator_tasks=0, task_inbox_issues=[], http_provider_fallback=False,
            scientific_decision="pending")
        self.write("config.json", self.config)
        self.write("heartbeat.json", self.heartbeat)

    def observe(self, **kwargs):
        return watch.observe_worker(self.worker, now=self.now, worker_host="tamia3.tamia.ecpia.ca",
            observer_host="tamia1", worker_pid=524659, allow_continuous=True,
            expected_source_sha256="a" * 64, **kwargs)

    def add_call(self, number=1, state="proposal_ready", reported=None):
        ResearchWatchTests.add_call(self, number=number, state=state, reported=reported)
        packet = json.loads((self.worker / f"call-{number:04d}/status.json").read_text())
        packet["execution_mode"] = "continuous"
        self.write(f"call-{number:04d}/status.json", packet)

    def successful_journal(self, size):
        for number in range(1, size + 1):
            self.add_call(number, reported={name: 1 for name in watch.TOKEN_FIELDS})
        self.heartbeat.update(provider_call_attempts=size, schema_verified_reviews=size,
                              usage_totals={name: size for name in watch.TOKEN_FIELDS})
        self.write("heartbeat.json", self.heartbeat)

    def terminal(self, reason="termination_signal"):
        self.heartbeat.update(state="stopped", activity="stopped", stop_reason=reason)
        self.write("heartbeat.json", self.heartbeat)
        self.write("status.json", {"state": "stopped", "schema_version": 2,
            "execution_mode": "continuous", "continuous": True, "max_calls": None, "max_hours": None,
            "stop_reason": reason, "exit_code": 0, "finished_at": self.now.isoformat(),
            "provider_call_attempts": self.heartbeat["provider_call_attempts"],
            "schema_verified_reviews": self.heartbeat["schema_verified_reviews"],
            "usage_totals": self.heartbeat["usage_totals"], "scientific_decision": "pending"})

    def test_explicit_optin_source_and_continuous_config_required(self):
        with self.assertRaises(ValueError):
            watch.observe_worker(self.worker, now=self.now, worker_host="tamia3", observer_host="tamia1", allow_continuous=True)
        bounded = watch.observe_worker(self.worker, now=self.now, worker_host="tamia3", observer_host="tamia1")
        self.assertEqual(bounded["state"], "metadata_inconsistent")
        for source in (None, "b" * 64):
            with self.assertRaises(ValueError):
                watch.run_watch(worker=self.worker, output=self.parent / "invalid",
                    deadline=self.now + timedelta(seconds=120), worker_host="tamia3", interval=60,
                    expected_source_sha256=source, allow_continuous=True, now_fn=lambda: self.now)
        self.assertFalse((self.parent / "invalid").exists())
        self.config["max_calls"] = 24
        self.write("config.json", self.config)
        self.assertIn("continuous_worker_config_mismatch", self.observe()["issues"])

    def test_more_than_24_calls_have_explicit_partial_scope_and_incremental_cursor(self):
        self.successful_journal(50)
        first = self.observe()
        self.assertEqual(first["issues"], [])
        self.assertFalse(first["journal_scope"]["full_completed_history_observed"])
        self.assertFalse(first["journal_scope"]["global_counts_and_usage_reconciled"])
        self.assertEqual(first["journal_cursor"]["next_attempt"], 25)
        self.assertEqual(first["journal_scope"]["call_records_read_this_pass"], 25)
        self.assertEqual(first["journal_cursor"]["observed_usage_totals"]["total_tokens"], 24)
        second = self.observe(journal_cursor=first["journal_cursor"])
        self.assertEqual(second["issues"], [])
        self.assertEqual(second["journal_cursor"]["next_attempt"], 49)
        third = self.observe(journal_cursor=second["journal_cursor"])
        self.assertEqual(third["issues"], [])
        self.assertTrue(third["journal_scope"]["full_completed_history_observed"])
        self.assertTrue(third["journal_scope"]["global_counts_and_usage_reconciled"])
        self.assertFalse(third["journal_scope"]["previously_observed_files_immutability_verified"])
        self.assertEqual(third["journal_cursor"]["ready_reviews_observed"], 50)
        self.assertEqual(third["journal_cursor"]["observed_usage_totals"]["total_tokens"], 50)
        # The cursor accumulates bounded records; it does not retain all calls.
        self.assertEqual(len(third["calls"]), 2)
        self.assertFalse(third["worker_liveness_certified"])

    def test_large_attempt_number_uses_no_directory_scan_and_bounded_metadata_reads(self):
        for number in list(range(1, 25)) + [10000]:
            self.add_call(number, reported={name: 0 for name in watch.TOKEN_FIELDS})
        self.heartbeat.update(provider_call_attempts=10000, schema_verified_reviews=10000)
        self.write("heartbeat.json", self.heartbeat)
        actual = watch.read_metadata
        opened = []
        def record(path, **kwargs):
            opened.append(path)
            return actual(path, **kwargs)
        with patch.object(Path, "iterdir", side_effect=AssertionError("unbounded directory scan")), \
             patch.object(watch, "read_metadata", side_effect=record):
            result = self.observe()
        self.assertEqual(len(opened), 53)
        self.assertEqual(result["journal_scope"]["latest_attempt_inspected"], 10000)
        self.assertEqual(result["issues"], [])
        self.assertFalse(result["journal_scope"]["global_counts_and_usage_reconciled"])

    def test_degraded_and_stalled_are_retrying_never_stopped_and_unknown_usage_is_not_zero(self):
        for size, health in ((1, "degraded"), (4, "stalled")):
            for number in range(1, size + 1):
                self.add_call(number, state="failed")
            self.heartbeat.update(provider_call_attempts=size, schema_verified_reviews=0,
                consecutive_provider_errors=size, state=health, activity="waiting_for_retry",
                retry_backoff_seconds=60 * 2 ** (size - 1),
                next_provider_call_not_before=(self.now + timedelta(seconds=480)).isoformat(),
                usage_totals={name: None for name in watch.TOKEN_FIELDS})
            self.write("heartbeat.json", self.heartbeat)
            with patch.object(watch.os, "kill", side_effect=AssertionError("wrong-host PID probe")):
                result = self.observe()
            self.assertEqual(result["issues"], [])
            self.assertEqual(result["state"], "worker_" + health + "_retrying_process_unverified")
            self.assertIsNone(result["usage_totals"]["total_tokens"])
            self.assertFalse(result["worker_liveness_certified"])

    def test_pending_retry_and_success_reconcile_completed_history_and_error_streak(self):
        self.add_call(1, state="failed")
        self.add_call(2, state="calling")
        self.heartbeat.update(provider_call_attempts=2, state="degraded", activity="provider_call",
            provider_call_in_flight=True, consecutive_provider_errors=1, retry_backoff_seconds=60,
            usage_totals={name: None for name in watch.TOKEN_FIELDS})
        self.write("heartbeat.json", self.heartbeat)
        first = self.observe()
        self.assertEqual(first["issues"], [])
        self.assertEqual(first["journal_cursor"]["next_attempt"], 2)
        self.assertTrue(first["journal_scope"]["global_counts_and_usage_reconciled"])
        self.add_call(2, reported={name: 2 for name in watch.TOKEN_FIELDS})
        self.heartbeat.update(schema_verified_reviews=1, state="running", activity="waiting_for_evidence",
            provider_call_in_flight=False, consecutive_provider_errors=0, retry_backoff_seconds=0)
        self.write("heartbeat.json", self.heartbeat)
        result = self.observe(journal_cursor=first["journal_cursor"])
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["journal_cursor"]["next_attempt"], 3)
        self.assertIsNone(result["usage_totals"]["total_tokens"])
        self.heartbeat["consecutive_provider_errors"] = 1
        self.write("heartbeat.json", self.heartbeat)
        self.assertIn("continuous_state_error_streak_disagrees", self.observe()["issues"])

    def test_terminal_signal_cannot_certify_full_history_from_a_tail(self):
        self.successful_journal(40)
        self.terminal("stop_file_requested")
        first = self.observe()
        self.assertEqual(first["issues"], [])
        self.assertEqual(first["state"], "stopped_metadata_history_verification_pending")
        self.assertFalse(first["terminal"]["verified"])
        second = self.observe(journal_cursor=first["journal_cursor"])
        self.assertEqual(second["issues"], [])
        self.assertEqual(second["state"], "worker_stopped")
        self.assertTrue(second["terminal"]["verified"])
        self.assertIsNone(second["process_probe"]["process_alive"])
        self.assertFalse(second["worker_liveness_certified"])

    def test_continuous_terminal_disallows_caps_three_error_stop_and_inflight(self):
        self.terminal("three_consecutive_provider_errors")
        self.assertIn("terminal_heartbeat_disagreement", self.observe()["issues"])
        self.terminal("termination_signal")
        self.heartbeat.update(activity="provider_call", provider_call_in_flight=True)
        self.write("heartbeat.json", self.heartbeat)
        self.assertFalse(self.observe()["terminal"]["verified"])

    def test_cursor_monotonicity_and_sticky_usage_unknowns_fail_closed(self):
        self.add_call(1, state="failed")
        self.heartbeat.update(provider_call_attempts=1, state="degraded", activity="waiting_for_retry",
            consecutive_provider_errors=1, retry_backoff_seconds=60,
            usage_totals={name: None for name in watch.TOKEN_FIELDS})
        self.write("heartbeat.json", self.heartbeat)
        first = self.observe()
        self.assertEqual(first["issues"], [])
        self.heartbeat["usage_totals"] = {name: 0 for name in watch.TOKEN_FIELDS}
        self.write("heartbeat.json", self.heartbeat)
        self.assertIn("unknown_usage_became_known_without_reconciliation", self.observe(journal_cursor=first["journal_cursor"])["issues"])
        self.heartbeat["provider_call_attempts"] = 0
        self.heartbeat["consecutive_provider_errors"] = 0
        self.write("heartbeat.json", self.heartbeat)
        self.assertIn("continuous_heartbeat_regressed", self.observe(journal_cursor=first["journal_cursor"])["issues"])
        bad_cursor = dict(first["journal_cursor"], next_attempt=2)
        bad_cursor["completed_attempts_observed"] = 0
        with self.assertRaises(ValueError):
            self.observe(journal_cursor=bad_cursor)

    def test_model_actions_source_privacy_and_malformed_records_are_not_accepted(self):
        self.add_call(1, reported={name: 0 for name in watch.TOKEN_FIELDS})
        self.heartbeat.update(provider_call_attempts=1, schema_verified_reviews=1)
        self.write("heartbeat.json", self.heartbeat)
        packet = json.loads((self.worker / "call-0001/status.json").read_text())
        packet.update(returned_model="PRIVATE_SENTINEL", actions_executed=1)
        self.write("call-0001/status.json", packet)
        self.config["worker_source_sha256"] = "b" * 64
        self.config["provider_endpoint"] = "PRIVATE_SENTINEL"
        self.write("config.json", self.config)
        self.heartbeat.update(task_inbox_issues=["PRIVATE_SENTINEL"], active_task_id="PRIVATE_SENTINEL")
        self.write("heartbeat.json", self.heartbeat)
        for name in ("request.json", "evidence.json", "raw-reply.json", "review.json", "raw-error.txt"):
            (self.worker / "call-0001" / name).write_text("PRIVATE_SENTINEL")
        actual = watch.read_metadata
        opened = []
        def record(path, **kwargs):
            opened.append(path.name)
            return actual(path, **kwargs)
        with patch.object(watch, "read_metadata", side_effect=record):
            result = self.observe()
        self.assertIn("worker_source_mismatch", result["issues"])
        self.assertIn("unverified_proposal_model_or_actions", result["issues"])
        self.assertEqual(result["journal_cursor"]["completed_attempts_observed"], 0)
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(result))
        self.assertEqual(set(opened), {"config.json", "heartbeat.json", "status.json", "usage.json"})

    def test_continuous_monitor_keeps_own_deadline_and_never_calls_provider_or_worker_signal(self):
        self.successful_journal(50)
        state = {"mono": 0.0}
        def sleep(seconds):
            state["mono"] += seconds
        with patch.object(watch.subprocess, "run", side_effect=AssertionError("no scheduler requested")), \
             patch.object(watch.os, "kill", side_effect=AssertionError("wrong host; no signal")):
            result = watch.run_watch(worker=self.worker, output=self.parent / "observer",
                deadline=self.now + timedelta(seconds=180), worker_host="tamia3", worker_pid=524659,
                expected_source_sha256="a" * 64, allow_continuous=True, interval=60,
                now_fn=lambda: self.now + timedelta(seconds=state["mono"]), monotonic_fn=lambda: state["mono"],
                sleep_fn=sleep, observer_host="tamia1")
        self.assertEqual(result["observations"], 3)
        self.assertEqual(result["stop_reason"], "deadline")
        self.assertFalse(result["worker_terminated_by_observer"])
        self.assertEqual(result["provider_calls_by_observer"], 0)
        self.assertEqual(result["job_submissions_by_observer"], 0)
        observations = [json.loads(line) for line in (self.parent / "observer/observations.jsonl").read_text().splitlines()]
        self.assertEqual([record["worker"]["journal_cursor"]["next_attempt"] for record in observations], [25, 49, 51])
        self.assertTrue(observations[-1]["worker"]["journal_scope"]["global_counts_and_usage_reconciled"])

    def test_bad_state_retry_timestamp_and_partial_error_tail_remain_explicit(self):
        self.successful_journal(50)
        self.add_call(50, state="failed")
        self.heartbeat.update(schema_verified_reviews=49, consecutive_provider_errors=1,
            state="degraded", activity="waiting_for_retry", retry_backoff_seconds=60,
            usage_totals={name: None for name in watch.TOKEN_FIELDS})
        self.write("heartbeat.json", self.heartbeat)
        result = self.observe()
        self.assertEqual(result["issues"], [])
        self.assertFalse(result["journal_scope"]["global_counts_and_usage_reconciled"])
        self.heartbeat.update(next_provider_call_not_before="PRIVATE_SENTINEL", retry_backoff_seconds=3601)
        self.write("heartbeat.json", self.heartbeat)
        result = self.observe()
        self.assertIn("invalid_next_provider_call_timestamp", result["issues"])
        self.assertIn("retry_backoff_outside_configured_policy", result["issues"])
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(result))

    def test_continuous_call_directory_and_metadata_symlinks_and_oversize_are_rejected(self):
        self.successful_journal(1)
        original = self.worker / "call-0001"
        moved = self.parent / "other-directory"
        original.rename(moved)
        original.symlink_to(moved, target_is_directory=True)
        result = self.observe()
        self.assertIn("symlinked_call_directory", result["issues"])
        self.assertEqual(result["journal_cursor"]["completed_attempts_observed"], 0)
        original.unlink()
        moved.rename(original)
        status = original / "status.json"
        status.unlink()
        status.symlink_to(original / "usage.json")
        result = self.observe()
        self.assertEqual(result["calls"][0]["status_read"], "unreadable_or_invalid")
        status.unlink()
        status.write_text("x" * (watch.MAX_METADATA_BYTES + 1))
        result = self.observe()
        self.assertEqual(result["calls"][0]["status_read"], "unsafe_or_oversized")

    def test_open_call_directory_descriptor_prevents_symlink_swap_during_file_reads(self):
        self.successful_journal(1)
        original = self.worker / "call-0001"
        alien = self.parent / "alien"
        alien.mkdir()
        (alien / "status.json").write_text(json.dumps({"state": "PRIVATE_SENTINEL"}))
        moved = self.parent / "original-open-directory"
        actual = watch.read_metadata
        swapped = False
        def swap(path, **kwargs):
            nonlocal swapped
            if path == Path("status.json") and kwargs.get("directory_fd") is not None and not swapped:
                original.rename(moved)
                original.symlink_to(alien, target_is_directory=True)
                swapped = True
            return actual(path, **kwargs)
        with patch.object(watch, "read_metadata", side_effect=swap):
            result = self.observe()
        self.assertTrue(swapped)
        self.assertEqual(result["issues"], [])
        self.assertTrue(result["journal_scope"]["global_counts_and_usage_reconciled"])
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(result))

    def test_partial_latest_failure_cannot_have_zero_error_streak(self):
        self.successful_journal(50)
        self.add_call(50, state="failed")
        self.heartbeat.update(schema_verified_reviews=49, usage_totals={name: None for name in watch.TOKEN_FIELDS})
        self.write("heartbeat.json", self.heartbeat)
        result = self.observe()
        self.assertIn("latest_failure_missing_error_streak", result["issues"])
        self.assertFalse(result["journal_scope"]["global_counts_and_usage_reconciled"])

    def test_same_host_pid_absence_with_fresh_heartbeat_does_not_certify_stop(self):
        with patch.object(watch.os, "kill", side_effect=ProcessLookupError):
            result = watch.observe_worker(self.worker, now=self.now, worker_host="tamia1",
                observer_host="tamia1", worker_pid=524659, allow_continuous=True,
                expected_source_sha256="a" * 64)
        self.assertEqual(result["state"], "fresh_heartbeat_pid_absent")
        self.assertFalse(result["worker_liveness_certified"])
        self.assertIsNone(result["terminal"])
        with self.assertRaises(ValueError):
            watch.local_scheduler(["1;sbatch"])


if __name__ == "__main__":
    unittest.main()
