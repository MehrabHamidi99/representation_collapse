#!/usr/bin/env python3
"""Plan recovery within one existing Qwen lease; never launch or send data."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path


def utc(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be an explicit timezone-aware string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("timestamp must explicitly use UTC")
    return parsed


def counter(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def plan_resume(lease: dict, attempts: list[dict], *, now: datetime) -> dict:
    """Require fresh stopped-process evidence and subtract every prior attempt.

    A caller supplies reconciled journals from the exact approved cohort. This
    planner cannot authenticate those journals or turn a proposal into approval.
    Unknown counts and live/unverified processes prohibit a recovery plan.
    """
    if now.tzinfo is None or now.utcoffset() != timedelta(0):
        raise ValueError("planning time must explicitly use UTC")
    if lease.get("explicit_owner_approval") is not True:
        raise ValueError("existing explicit owner approval required")
    start = utc(lease["started_at_utc"])
    hours = lease["max_hours"]
    if isinstance(hours, bool) or not isinstance(hours, (int, float)) or not math.isfinite(hours) or not 0 < hours <= 24:
        raise ValueError("lease must remain within the approved 24-hour ceiling")
    limit = counter(lease["max_total_attempts"], "max_total_attempts")
    if not 0 < limit <= 24:
        raise ValueError("lease must remain within the approved 24-attempt ceiling")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("reconciled prior-attempt journals required")
    if now < start:
        raise ValueError("planning clock precedes the approved cohort start")
    used = 0
    seen = set()
    blockers = []
    for attempt in attempts:
        run = attempt.get("run_id")
        if not isinstance(run, str) or not run.strip() or run in seen:
            raise ValueError("distinct immutable attempt identifiers required")
        seen.add(run)
        if attempt.get("lease_id") != lease.get("lease_id") or not lease.get("lease_id"):
            raise ValueError("attempt journal is outside the explicit lease")
        used += counter(attempt.get("provider_call_attempts"), "provider_call_attempts")
        if attempt.get("state") != "stopped" or attempt.get("process_alive") is not False:
            blockers.append("live_or_unverified_prior_process")
        if attempt.get("terminal_status_verified") is not True:
            blockers.append("unverified_terminal_status")
        observed = utc(attempt["verified_at_utc"])
        age = (now - observed).total_seconds()
        if age < 0 or age > 300:
            blockers.append("stale_or_future_process_probe")
    remaining_calls = max(0, limit - used)
    deadline = start + timedelta(hours=hours)
    remaining_seconds = max(0.0, (deadline - now).total_seconds())
    if used > limit:
        blockers.append("prior_attempts_exceeded_approved_cap")
    if not remaining_calls:
        blockers.append("attempt_cap_exhausted")
    if not remaining_seconds:
        blockers.append("runtime_lease_expired")
    return {
        "schema_version": 1, "kind": "qwen_existing_lease_recovery_plan",
        "lease_id": lease["lease_id"], "checked_at_utc": now.isoformat(),
        "deadline_utc": deadline.isoformat(), "prior_provider_attempts": used,
        "max_total_attempts": limit, "remaining_call_attempts": remaining_calls,
        "remaining_seconds": remaining_seconds,
        "maximum_restart_hours": remaining_seconds / 3600,
        "bounded_resume_candidate": not blockers,
        "blockers": sorted(set(blockers)), "launch_performed": False,
        "approval_granted_by_planner": False,
        "scope": "proposal-only recovery; provider identity and payload authorization unchanged",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True,
                        help="ignored sanitized lease/attempt metadata; no credentials")
    args = parser.parse_args()
    packet = json.loads(args.journal.read_text())
    result = plan_resume(packet["lease"], packet["attempts"], now=datetime.now(timezone.utc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
