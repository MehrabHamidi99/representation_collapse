import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qwen_resume_budget", ROOT / "code/ops/qwen_resume_budget.py")
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


class QwenResumeBudgetTests(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2026, 10, 12, 9, 10, 30, tzinfo=timezone.utc)
        self.now = self.start + timedelta(hours=4)
        self.lease = {"lease_id": "approved-continuation", "explicit_owner_approval": True,
                      "started_at_utc": self.start.isoformat(), "max_hours": 24,
                      "max_total_attempts": 24}
        self.attempt = {"run_id": "attempt-1", "lease_id": self.lease["lease_id"],
                        "provider_call_attempts": 7, "state": "stopped", "process_alive": False,
                        "terminal_status_verified": True, "verified_at_utc": self.now.isoformat()}

    def plan(self, attempts=None, now=None):
        return budget.plan_resume(self.lease, attempts or [self.attempt], now=now or self.now)

    def test_recovery_subtracts_calls_and_elapsed_time(self):
        result = self.plan()
        self.assertTrue(result["bounded_resume_candidate"])
        self.assertEqual(result["remaining_call_attempts"], 17)
        self.assertEqual(result["maximum_restart_hours"], 20)
        self.assertFalse(result["launch_performed"])
        self.assertFalse(result["approval_granted_by_planner"])

    def test_multiple_recoveries_share_one_cap(self):
        second = {**self.attempt, "run_id": "attempt-2", "provider_call_attempts": 5}
        self.assertEqual(self.plan([self.attempt, second])["remaining_call_attempts"], 12)

    def test_live_or_unknown_prior_process_blocks_duplicates(self):
        for value in (True, None):
            self.attempt["process_alive"] = value
            self.assertFalse(self.plan()["bounded_resume_candidate"])

    def test_terminal_and_fresh_probe_required(self):
        self.attempt["terminal_status_verified"] = False
        self.assertFalse(self.plan()["bounded_resume_candidate"])
        self.attempt["terminal_status_verified"] = True
        for offset in (-301, 1):
            self.attempt["verified_at_utc"] = (self.now + timedelta(seconds=offset)).isoformat()
            self.assertFalse(self.plan()["bounded_resume_candidate"])

    def test_exhausted_calls_and_time_are_not_renewed(self):
        for count in (24, 25):
            self.attempt["provider_call_attempts"] = count
            self.assertFalse(self.plan()["bounded_resume_candidate"])
        self.attempt["provider_call_attempts"] = 7
        expired = self.start + timedelta(hours=24)
        self.attempt["verified_at_utc"] = expired.isoformat()
        self.assertEqual(self.plan(now=expired)["maximum_restart_hours"], 0)
        self.assertFalse(self.plan(now=expired)["bounded_resume_candidate"])

    def test_unknown_boolean_or_negative_usage_rejected(self):
        for count in (None, True, -1, 1.5):
            self.attempt["provider_call_attempts"] = count
            with self.assertRaises(ValueError): self.plan()

    def test_duplicate_attempts_and_foreign_leases_rejected(self):
        with self.assertRaises(ValueError): self.plan([self.attempt, self.attempt])
        self.attempt["lease_id"] = "another-project"
        with self.assertRaises(ValueError): self.plan()

    def test_approval_timezone_and_clock_cannot_expand_lease(self):
        self.lease["explicit_owner_approval"] = False
        with self.assertRaises(ValueError): self.plan()
        self.lease["explicit_owner_approval"] = True
        with self.assertRaises(ValueError): self.plan(now=self.start - timedelta(seconds=1))
        self.lease["started_at_utc"] = "2026-10-12T09:10:30"
        with self.assertRaises(ValueError): self.plan()


if __name__ == "__main__":
    unittest.main()
