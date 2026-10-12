#!/usr/bin/env python3
"""Observe one existing Qwen run until an explicit UTC deadline; never act on it.

Only public metadata files are opened. No SSH, provider request, job submission,
worker restart or scientific acceptance is implemented. The optional scheduler
probe is local, read-only and restricted to caller-supplied project job IDs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

EXPECTED_MODEL = "Qwen/Qwen3.8-Flash-Next-FP8"
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
STOP_REASONS = {"runtime_cap", "call_cap", "three_consecutive_provider_errors", "termination_signal"}
MAX_METADATA_BYTES = 65536
_STOP = False


def utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("explicit UTC timestamp required")
    return parsed


def count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def finite_nonnegative(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def atomic_json(path: Path, packet: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as handle:
        json.dump(packet, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_metadata(path: Path) -> tuple[dict | None, str]:
    """No symlink following, unlimited reads or raw exception text persistence."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as handle:
            import stat
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_METADATA_BYTES:
                return None, "unsafe_or_oversized"
            raw = handle.read(MAX_METADATA_BYTES + 1)
        if len(raw) > MAX_METADATA_BYTES:
            return None, "unsafe_or_oversized"
        packet = json.loads(raw)
        return (packet, "read") if isinstance(packet, dict) else (None, "invalid_json_object")
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError, UnicodeError):
        return None, "unreadable_or_invalid"


def usage(value) -> tuple[dict, bool]:
    if not isinstance(value, dict):
        return {key: None for key in TOKEN_FIELDS}, False
    valid = all(value.get(key) is None or count(value.get(key)) for key in TOKEN_FIELDS)
    return {key: value.get(key) if count(value.get(key)) else None for key in TOKEN_FIELDS}, valid


def normalized_host(value: str) -> str:
    return value.lower().split(".", 1)[0]


def process_probe(worker_host: str, observer_host: str, worker_pid: int | None) -> dict:
    same = normalized_host(worker_host) == normalized_host(observer_host)
    result = {"same_host": same, "worker_pid": worker_pid, "process_alive": None}
    if not same:
        result["state"] = "not_probed_different_host"
    elif worker_pid is None:
        result["state"] = "not_probed_no_pinned_pid"
    else:
        # Signal zero checks existence only. It cannot terminate the worker.
        try:
            os.kill(worker_pid, 0)
            result.update(state="pid_exists_identity_unverified", process_alive=True)
        except ProcessLookupError:
            result.update(state="pid_absent_on_original_host", process_alive=False)
        except (PermissionError, OSError):
            result["state"] = "pid_probe_unavailable"
    return result


def observe_worker(worker: Path, *, now: datetime, worker_host: str,
                   observer_host: str, worker_pid: int | None = None,
                   stale_seconds: float = 600, expected_source_sha256: str | None = None) -> dict:
    """Whitelist numeric/public identity metadata and conservatively reconcile it."""
    errors = []
    config, config_read = read_metadata(worker / "config.json")
    heartbeat, heartbeat_read = read_metadata(worker / "heartbeat.json")
    terminal, terminal_read = read_metadata(worker / "status.json")
    config, heartbeat = config or {}, heartbeat or {}
    config_valid = (config.get("kind") == "qwen_research_leaf_worker"
                    and config.get("schema_version") == 1
                    and config.get("requested_model") == EXPECTED_MODEL
                    and count(config.get("max_calls")) and 0 < config["max_calls"] <= 24
                    and finite_nonnegative(config.get("max_hours")) and 0 < config["max_hours"] <= 24
                    and config.get("autonomous_actions") == "structured_proposals_only")
    source_matches = (expected_source_sha256 is None or
                      config.get("worker_source_sha256") == expected_source_sha256)
    if not config_valid:
        errors.append("worker_config_or_requested_model_mismatch")
    if not source_matches:
        errors.append("worker_source_mismatch")
    limits = {"max_calls": config["max_calls"], "max_hours": config["max_hours"]} if config_valid else None
    attempts = heartbeat.get("provider_call_attempts")
    successes = heartbeat.get("schema_verified_reviews")
    consecutive_errors = heartbeat.get("consecutive_provider_errors")
    counters_valid = all(count(v) for v in (attempts, successes, consecutive_errors))
    if counters_valid:
        counters_valid = successes <= attempts <= 24 and consecutive_errors <= attempts - successes
        if limits:
            counters_valid = counters_valid and attempts <= limits["max_calls"]
    if not counters_valid:
        errors.append("invalid_heartbeat_counters")
    tokens, tokens_valid = usage(heartbeat.get("usage_totals"))
    if not tokens_valid:
        errors.append("invalid_heartbeat_usage")
    heartbeat_age = None
    try:
        updated = utc(heartbeat["updated_at"])
        heartbeat_age = (now - updated).total_seconds()
        fresh = 0 <= heartbeat_age <= stale_seconds
    except (KeyError, TypeError, ValueError, AttributeError):
        updated, fresh = None, False
        errors.append("invalid_heartbeat_timestamp")
    if heartbeat_age is not None and heartbeat_age < 0:
        errors.append("future_heartbeat_timestamp")
    if not finite_nonnegative(heartbeat.get("elapsed_seconds")):
        errors.append("invalid_heartbeat_elapsed")
    if heartbeat.get("state") not in {"running", "stopped"}:
        errors.append("invalid_heartbeat_state")
    if heartbeat.get("schema_version") != 1:
        errors.append("invalid_heartbeat_schema")

    calls = []
    ready = failed = calling = 0
    # The fixed 24-slot allowlist never opens request, evidence, review or replies.
    for number in range(1, 25):
        directory = worker / f"call-{number:04d}"
        if directory.is_symlink():
            errors.append("symlinked_call_directory")
            continue
        if not directory.exists():
            continue
        packet, status_read = read_metadata(directory / "status.json")
        reported, usage_read = read_metadata(directory / "usage.json")
        packet = packet or {}
        state = packet.get("state")
        identity = (packet.get("requested_model") == EXPECTED_MODEL and
                    (state == "calling" or packet.get("returned_model") == EXPECTED_MODEL))
        if state == "proposal_ready":
            ready += 1
            if packet.get("model_identity_verified") is not True or not identity or not count(packet.get("actions_executed")) or packet.get("actions_executed") != 0:
                errors.append("unverified_proposal_model_or_actions")
        elif state == "failed":
            failed += 1
        elif state == "calling":
            calling += 1
        else:
            errors.append("invalid_call_status")
        if packet.get("requested_model") != EXPECTED_MODEL:
            errors.append("call_requested_model_mismatch")
        per_call, per_call_valid = usage(reported.get("reported") if reported else None)
        if reported and reported.get("reported") is not None and not per_call_valid:
            errors.append("invalid_call_usage")
        if state != "calling" and usage_read != "read":
            errors.append("finished_call_usage_missing")
        calls.append({"attempt_number": number, "state": state if state in {"calling", "failed", "proposal_ready"} else "invalid",
                      "status_read": status_read, "usage_read": usage_read,
                      "requested_model_matches": packet.get("requested_model") == EXPECTED_MODEL,
                      "returned_model_matches": packet.get("returned_model") == EXPECTED_MODEL,
                      "schema_verified_proposal": state == "proposal_ready" and identity and packet.get("model_identity_verified") is True,
                      "scientific_acceptance": False, "reported_usage": per_call})
    # Detect an expanded worker journal without opening its content.
    try:
        names = [p.name for p in islice(worker.iterdir(), 129)]
        if len(names) > 128 or any(re.fullmatch(r"call-[0-9]+", name) and name not in
                                  {f"call-{n:04d}" for n in range(1, 25)} for name in names):
            errors.append("out_of_envelope_worker_journal")
    except OSError:
        errors.append("worker_directory_unavailable")
    if counters_valid and (len(calls) != attempts or ready != successes or calling > 1 or
                           [c["attempt_number"] for c in calls] != list(range(1, attempts + 1))):
        errors.append("heartbeat_call_journal_counts_disagree")
    if calling and (calling != 1 or calls[-1]["state"] != "calling"):
        errors.append("inflight_call_is_not_unique_final_attempt")
    # The worker carries the previous error streak into the next calling
    # heartbeat, then updates it only after that attempt finishes. Ignore only
    # a final in-flight attempt when deriving the completed-call failure tail.
    completed_calls = calls[:-1] if calls and calls[-1]["state"] == "calling" else calls
    trailing_failures = 0
    for attempt in reversed(completed_calls):
        if attempt["state"] != "failed":
            break
        trailing_failures += 1
    if counters_valid and consecutive_errors != trailing_failures:
        errors.append("heartbeat_completed_error_streak_disagrees")
    # Completed per-call usage must reconcile exactly; in-flight calls have not
    # yet contributed to the heartbeat. Null remains unknown, never zero.
    for field in TOKEN_FIELDS:
        values = [c["reported_usage"][field] for c in calls if c["state"] != "calling"]
        aggregate = None if any(v is None for v in values) else sum(values)
        if counters_valid and tokens_valid and tokens[field] != aggregate:
            errors.append("heartbeat_call_usage_disagree")

    terminal_valid = False
    terminal_summary = None
    if terminal_read == "read":
        terminal = terminal or {}
        term_tokens, term_usage_valid = usage(terminal.get("usage_totals"))
        reason = terminal.get("stop_reason")
        reason_consistent = (
            reason == "termination_signal" or
            reason == "call_cap" and limits is not None and attempts == limits["max_calls"] or
            reason == "runtime_cap" and limits is not None and finite_nonnegative(heartbeat.get("elapsed_seconds")) and heartbeat["elapsed_seconds"] >= limits["max_hours"] * 3600 or
            reason == "three_consecutive_provider_errors" and consecutive_errors == 3 and len(calls) >= 3 and all(c["state"] == "failed" for c in calls[-3:]))
        try:
            finished = utc(terminal["finished_at"])
            finish_valid = finished <= now and (updated is None or updated <= finished)
        except (KeyError, TypeError, ValueError, AttributeError):
            finished, finish_valid = None, False
        terminal_valid = (not errors and calling == 0 and len(calls) == ready + failed and
                          terminal.get("state") == "stopped" and reason in STOP_REASONS and reason_consistent and heartbeat.get("stop_reason") == reason and
                          count(terminal.get("exit_code")) and terminal.get("exit_code") == (1 if reason == "three_consecutive_provider_errors" else 0) and
                          count(terminal.get("provider_call_attempts")) and count(terminal.get("schema_verified_reviews")) and
                          terminal.get("provider_call_attempts") == attempts and terminal.get("schema_verified_reviews") == successes and
                          term_usage_valid and term_tokens == tokens and finish_valid and heartbeat.get("state") == "stopped")
        terminal_summary = {"verified": terminal_valid,
                            "stop_reason": reason if reason in STOP_REASONS else "unrecognized",
                            "finished_at_utc": finished.isoformat() if finished else None}
        if not terminal_valid:
            errors.append("terminal_heartbeat_disagreement")
    elif terminal_read != "missing":
        errors.append("terminal_metadata_unreadable")
    probe = process_probe(worker_host, observer_host, worker_pid)
    if terminal_valid and probe["process_alive"] is True:
        errors.append("terminal_with_existing_pid_identity_unverified")
    if errors:
        state = "metadata_inconsistent"
    elif terminal_valid:
        state = "worker_stopped_failure" if terminal_summary["stop_reason"] == "three_consecutive_provider_errors" else "worker_stopped"
    elif heartbeat.get("state") == "stopped":
        state = "stopped_heartbeat_terminal_pending"
    elif not fresh:
        state = "worker_heartbeat_stale"
    elif probe["process_alive"] is False:
        state = "fresh_heartbeat_pid_absent"
    else:
        state = "worker_heartbeat_fresh_process_unverified" if probe["process_alive"] is None else "worker_heartbeat_fresh_pid_exists"
    return {"state": state, "expected_model": EXPECTED_MODEL, "expected_source_matches": source_matches,
            "metadata_reads": {"config": config_read, "heartbeat": heartbeat_read, "terminal": terminal_read},
            "worker_host": worker_host, "observer_host": observer_host, "process_probe": probe,
            "heartbeat_at_utc": updated.isoformat() if updated else None,
            "heartbeat_age_seconds": heartbeat_age, "heartbeat_fresh": fresh,
            "worker_limits": limits, "provider_call_attempts": attempts if counters_valid else None,
            "schema_verified_reviews": successes if counters_valid else None,
            "consecutive_provider_errors": consecutive_errors if counters_valid else None,
            "usage_totals": tokens, "terminal": terminal_summary, "calls": calls,
            "issues": sorted(set(errors)), "scientific_acceptance": False,
            "worker_liveness_certified": False}


def local_scheduler(job_ids: list[str], *, timeout: float = 10) -> dict:
    if not job_ids:
        return {"state": "not_requested", "jobs": []}
    if len(job_ids) > 32 or not all(re.fullmatch(r"[0-9]+(?:_[0-9]+)?", item) for item in job_ids):
        raise ValueError("at most 32 explicit numeric project job IDs required")
    try:
        result = subprocess.run(["squeue", "--me", "--noheader", "--jobs=" + ",".join(job_ids),
                                 "--format=%i|%T|%j"], capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return {"state": "unavailable", "jobs": []}
    if result.returncode:
        return {"state": "unavailable", "returncode": result.returncode, "jobs": []}
    jobs = []
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) != 3:
            continue
        identifier, state, name = (p.strip() for p in parts)
        if identifier in job_ids and name.startswith("rc-") and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", name) and re.fullmatch(r"[A-Z_]{1,40}", state):
            jobs.append({"job_id": identifier, "state": state, "name": name})
    return {"state": "queried_local_cluster_only", "jobs": jobs,
            "absence_means_completed": False}


@contextmanager
def observer_lock(parent: Path, worker: Path):
    """Only one observer for this canonical worker path in this output parent."""
    key = hashlib.sha256(str(worker).encode()).hexdigest()[:24]
    lock_path = parent / f".research-watch-{key}.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("observer already active for this worker/output parent") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_watch(*, worker: Path, output: Path, deadline: datetime, worker_host: str,
              worker_pid: int | None = None, interval: float = 300, stale_seconds: float = 600,
              expected_source_sha256: str | None = None, job_ids: list[str] | None = None,
              now_fn=None, monotonic_fn=None, sleep_fn=None, observer_host: str | None = None) -> dict:
    now_fn = now_fn or (lambda: datetime.now(timezone.utc))
    monotonic_fn, sleep_fn = monotonic_fn or time.monotonic, sleep_fn or time.sleep
    start = now_fn()
    if start.tzinfo is None or start.utcoffset() != timedelta(0) or deadline.tzinfo is None or deadline.utcoffset() != timedelta(0):
        raise ValueError("UTC start and deadline required")
    duration = (deadline - start).total_seconds()
    if not 0 < duration <= 86400:
        raise ValueError("deadline must be future and at most 24 hours away")
    if not math.isfinite(interval) or not 60 <= interval <= 3600 or not math.isfinite(stale_seconds) or not 60 <= stale_seconds <= 3600:
        raise ValueError("polling 60–3600 seconds and finite staleness 60–3600 seconds required")
    if worker_pid is not None and (not count(worker_pid) or worker_pid == 0):
        raise ValueError("positive pinned worker PID required")
    if not re.fullmatch(r"[A-Za-z0-9.-]{1,100}", worker_host):
        raise ValueError("explicit worker host required")
    if expected_source_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", expected_source_sha256):
        raise ValueError("expected source SHA-256 must be lowercase 64-hex")
    # Reject symlinked run paths, including parents, rather than silently follow.
    for path in (worker.absolute(), output.absolute()):
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("symlinked worker/output path rejected")
    worker, output = worker.resolve(), output.resolve()
    if not worker.is_dir() or not output.parent.is_dir() or output == worker or output.is_relative_to(worker) or worker.is_relative_to(output):
        raise ValueError("existing worker and distinct existing output parent required")
    job_ids = job_ids or []
    # Validate before creating output; this never performs a scheduler probe.
    if len(job_ids) > 32 or not all(re.fullmatch(r"[0-9]+(?:_[0-9]+)?", item) for item in job_ids):
        raise ValueError("invalid explicit project job IDs")
    observer_host = observer_host or socket.gethostname()
    mono_start = monotonic_fn()
    mono_deadline = mono_start + duration
    stop_reason = "deadline"
    observations = 0
    last = None
    with observer_lock(output.parent, worker):
        output.mkdir(exist_ok=False)
        atomic_json(output / "config.json", {
            "kind": "bounded_research_metadata_observer", "schema_version": 1,
            "started_at_utc": start.isoformat(), "deadline_utc": deadline.isoformat(),
            "observer_pid": os.getpid(), "observer_host": observer_host,
            "worker_host": worker_host, "worker_pid": worker_pid,
            "worker_run": str(worker), "expected_model": EXPECTED_MODEL,
            "expected_worker_source_sha256": expected_source_sha256,
            "interval_seconds": interval, "stale_seconds": stale_seconds,
            "project_job_ids": job_ids, "scheduler_scope": "local_cluster_only",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "actions": "metadata_read_only; observer stops without stopping worker"})
        while True:
            now, monotonic = now_fn(), monotonic_fn()
            if _STOP:
                stop_reason = "termination_signal"
                break
            if now >= deadline or monotonic >= mono_deadline:
                break
            try:
                worker_status = observe_worker(worker, now=now, worker_host=worker_host,
                    observer_host=observer_host, worker_pid=worker_pid,
                    stale_seconds=stale_seconds, expected_source_sha256=expected_source_sha256)
                remaining = min(mono_deadline - monotonic_fn(), (deadline - now_fn()).total_seconds())
                if remaining <= 0:
                    break
                scheduler_status = local_scheduler(job_ids, timeout=min(10, remaining))
            except Exception:
                # No input content or exception text enters the terminal journal.
                stop_reason = "metadata_observation_error"
                break
            observation = {"schema_version": 1, "observed_at_utc": now.isoformat(),
                           "observation_number": observations + 1,
                           "worker": worker_status,
                           "scheduler": scheduler_status, "observer_state": "running"}
            with (output / "observations.jsonl").open("a") as handle:
                handle.write(json.dumps(observation, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            atomic_json(output / "status.json", observation)
            observations += 1
            last = observation
            # Keep the next tick relative to this tick, not observation end, so
            # metadata/scheduler cost cannot accumulate polling drift.
            next_tick = min(monotonic + interval, mono_deadline)
            while not _STOP:
                remaining = min(next_tick - monotonic_fn(), (deadline - now_fn()).total_seconds())
                if remaining <= 0:
                    break
                sleep_fn(min(1, remaining))
        terminal = {"schema_version": 1, "observer_state": "stopped", "stop_reason": stop_reason,
                    "finished_at_utc": now_fn().isoformat(), "deadline_utc": deadline.isoformat(),
                    "observations": observations, "last_observation": last,
                    "worker_terminated_by_observer": False, "provider_calls_by_observer": 0,
                    "job_submissions_by_observer": 0, "scientific_acceptance": False}
        atomic_json(output / "status.json", terminal)
        return terminal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--worker-host", required=True)
    parser.add_argument("--worker-pid", type=int)
    parser.add_argument("--expected-worker-source-sha256")
    parser.add_argument("--interval", type=float, default=300)
    parser.add_argument("--stale-seconds", type=float, default=600)
    parser.add_argument("--job-id", action="append", default=[])
    args = parser.parse_args()
    def stop(_signum, _frame):
        global _STOP
        _STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        result = run_watch(worker=args.worker_run, output=args.output,
            deadline=utc(args.deadline_utc), worker_host=args.worker_host, worker_pid=args.worker_pid,
            expected_source_sha256=args.expected_worker_source_sha256,
            interval=args.interval, stale_seconds=args.stale_seconds, job_ids=args.job_id)
    except (ValueError, OSError):
        parser.exit(2, "research_watch: invalid contract or unavailable output; no worker action performed\n")
    print(json.dumps({"observer_state": result["observer_state"], "stop_reason": result["stop_reason"],
                      "observations": result["observations"]}))


if __name__ == "__main__":
    main()
