# Continuous worker metadata observation

2026-10-13. Local implementation and regression checks pass; remote deployment
and current worker participation require separate live evidence. This observer
implements no provider requests, research proposals, restarts, worker stop
actions, job submissions or scientific acceptance.

The owner explicitly removed total Qwen call/runtime caps for a **new**
continuous leaf on October 13. Existing bounded runs and their journals remain
unchanged. The observer's default continues to verify the original schema-1
bounded worker with its 24-slot journal and approved limits. Observation of the
new schema requires explicit `--allow-continuous`, a schema-2 continuous config
with both limits present and null, and
`--expected-worker-source-sha256` matching the actual leaf source. An absent
source pin or bounded config is rejected before creating the monitor output.

Source: [research_watch.py](../../code/ops/research_watch.py).
Tests: [test_research_watch.py](../../tests/test_research_watch.py).

## Bounded reads with an explicit cursor

An unlimited leaf journal cannot be reduced from its last few calls as though
they were the whole run. Each continuous observation reads config, heartbeat
and terminal metadata, then at most **24 successive attempt records plus the
latest attempt**. Only each call's `status.json` and `usage.json` are opened.
There is no call-directory enumeration. Attempt numbers greater than 24 and
9999 are supported with the leaf's `call-{number:04d}` convention.

The in-process cursor starts at attempt 1 for every new observer. It advances
through a contiguous prefix of valid completed records and retains completed,
successful and failed counts, completed error streak, sticky-null usage totals
and a rolling digest of allowlisted public metadata. It does not accumulate an
unlimited list of call records. No persisted cursor is automatically imported
on startup; a new observer must build its own observation history.

`journal_scope` records the inspected range, latest attempt, records read this
pass, next cursor, and whether the full completed history has been observed.
Until then, global count, token-usage and completed-error-tail reconciliation
remain explicitly incomplete. A heartbeat's cumulative counters stay labeled
as reported values. A terminal signal with only a partial journal yields
`stopped_metadata_history_verification_pending`, rather than a verified stopped
worker. With a pending final call, the completed prefix excludes that in-flight
attempt and the cursor retains it for the next pass.

Once all completed attempts have been observed, the accumulated counts, error
tail and usage must agree with the heartbeat. This verifies the metadata seen
across those passes; it does **not** verify that previously observed files remain
immutable. That limitation has its own false certification flag. Independent
archival/hash audits remain necessary for immutable cohort evidence.

The per-pass file-read ceiling is 53 metadata files, each capped at 64 KiB.
Continuous call directories are opened with `O_DIRECTORY|O_NOFOLLOW` and both
metadata files are read relative to that descriptor, preventing a directory
symlink swap between those reads. File symlinks, oversize metadata, malformed
records, unsafe model/action metadata and source mismatches block acceptance.
Requests, raw replies, reviews, evidence, credentials, provider endpoints and
error text are never opened or echoed. Task health is recorded as counts;
untrusted task strings/IDs are not copied into observations.

## Health and stopping semantics

Continuous schema-2 heartbeats retain the same attempt/review/error/usage
counters. The observer checks mode and null limits, proposal-only authority,
requested/returned model identity for completed proposals, zero executed
actions, nonnegative counters, mode/activity/in-flight agreement, retry metadata
and timestamps. Incremental heartbeat regression and an unknown usage field
becoming a purported known total are flagged. Missing provider usage remains
unknown, never zero.

`degraded` and `stalled` mean the continuous leaf reports retry/backoff state.
They are **not** stopped states. A stale heartbeat, a missing SSH connection or
an absent PID on a different login node cannot certify that the leaf stopped.
The PID existence probe uses signal zero only on the pinned original host;
existence alone does not establish process identity. The observer always leaves
`worker_liveness_certified=false`. A stop is verified from consistent terminal
and heartbeat metadata, completed journal coverage and the appropriate signal
or regular STOP-file reason. A still-existing same-host PID remains a separate
identity anomaly requiring coordinator inspection.

The metadata observer retains its own explicit UTC deadline, at most 24 hours
away, plus monotonic deadline protection. This deadline limits observation;
it does not impose a call/runtime cap on the continuous leaf. A signal to the
observer stops observation without signaling the leaf. Output paths are unique,
append-only observations retain scope/cursor evidence, and the existing advisory
lock prevents duplicate observers for the same canonical worker/output parent.
Polling journals must remain outside the leaf's evidence roots.

Optional local `squeue` inspection remains read-only and restricted to at most
32 explicitly supplied project job IDs. Absence in that queue does not imply
completed scheduler work. The observer does not connect to clusters or change
other projects' connections, jobs or listeners. Native OpenAI fallback routing
belongs to the coordinator and available agents; this monitor performs no
external API fallback.

## Regression acceptance

```bash
.venv/bin/python -m unittest discover -s tests -p test_research_watch.py -v
```

All **37 tests pass**, no skips, in **0.205 seconds** of unittest time. This
includes the unchanged 22 bounded-worker regressions and 15 new continuous
checks. The latter cover a 50-call journal over three bounded passes, attempt
10000 without directory scans, exact IO ceiling, partial terminal coverage,
pending retry/completed error-tail consistency, degraded/stalled semantics,
source/model/action checks, sticky unknown usage, cursor regression, private
payload nonreads, file/directory symlinks and a simulated directory-swap race.
A fake-clock 180-second monitor completes three observations then stops itself,
with zero provider calls, zero job submissions and no worker termination.

Observer source SHA-256:
`3f260d12e5ea400a50280d35cf739b4547b3dd1cc3125c7ded16de7ed748e034`.
Test SHA-256:
`6205ba74fbf117872d5d355fd71e02277022104ef446b869f7aa2ef00c58648c`.

These local results demonstrate metadata-observation behavior. They do not
establish a deployed monitor, current leaf/provider availability, uninterrupted
autoresearch, cluster allocation authority or an accepted scientific result.
