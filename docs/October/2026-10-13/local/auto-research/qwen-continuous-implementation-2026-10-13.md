# Continuous Qwen proposal leaf — implementation, 2026-10-13

The owner's new instruction removes the Qwen call/runtime bounds and permits
OpenAI research when Qwen has problems. That explicit instruction supersedes
the earlier bounded-run policy for a new opted-in run; it does not alter the
historical worker, its immutable source or already recorded outcomes.
This document describes locally tested code. It does **not** establish a
deployed process, successful provider call, uninterrupted service or scientific
acceptance. Coordinator integration and independent review govern deployment.

## Opt-in and preserved authority

[qwen_worker.py](../../../../../code/ops/qwen_worker.py) accepts `--continuous`.
This mode has no total call count or runtime cap. Config, heartbeat and terminal
status have `schema_version=2`, `execution_mode="continuous"`,
`continuous=true`, `max_calls=null` and `max_hours=null`. The old 24-hour,
24-attempt defaults, hourly research trigger and three-error stop remain for
runs without the flag. Mixing continuous mode with cap or research-interval
flags is rejected, so an ignored cap cannot be mistaken for an enforced one.
A new output directory is mandatory; no existing run is reopened or rewritten.

The leaf continues to request only the configured `Qwen/` model and rejects
missing/different returned model identity, malformed reviews, unknown evidence
citations, absent final content and token-truncated answers. Every answer is a
proposal with `scientific_decision="pending"`; it cannot submit/cancel jobs,
change science, run code or approve its own conclusions. There is no HTTP
provider fallback in the leaf. The coordinator and exposed native OpenAI
subagents handle the owner's authorized alternative research route, with their
own evidence and operational status. Provider failure never manufactures a
successful Qwen review or a fallback model identity.

Evidence remains bounded to approved public project excerpts and numerical
summaries; private provider/approval configuration, datasets, checkpoints,
weights and directory/file symlinks stay outside discovery. Requests retain
the 18,000-character message budget and 2,048 requested output tokens. The
new public-task content shares that same budget. Credentials remain
environment-only, used for the authorization header; no endpoint/key is added
to config or argv. Existing key/credential-echo scrubbing remains active.

Removing total call/runtime limits does not remove the per-call response
policy. This version still requests compact proposals with
`enable_thinking=false`, `reasoning_effort="low"`, a 900-token prompt target and
the 2,048-token requested output limit. It has no unrestricted context or
reasoning mode. Native OpenAI coordinator/subagents carry out deeper theory,
implementation and independent review. A later optional extended-response mode
could expose a larger output-token request and a revised answer prompt while
preserving the approved input scope and schema. A thinking-enabled option would
need distinct final/reasoning accounting and actual backend/model/schema
validation. Neither option is implemented or deployed by this change, and the
legacy response defaults are unchanged.

## Work triggers and error recovery

Continuous mode requests an initial evidence assessment, then requests only
for a semantic evidence change, a unique explicit coordinator task, or a paced
retry of a failed attempt. It has no hourly duplicate-reasoning trigger.
For deduplication, the semantic fingerprint ignores explicitly listed polling
timestamps, elapsed/age fields and observer counts. State, source/seed identity,
scientific measurements and outcomes remain in the fingerprint. Incomplete JSON
excerpts receive the same targeted scalar metadata normalization. Sent excerpts
are not edited by this fingerprint normalization.

Every call keeps the exact SHA-256 of its serialized evidence items separately
from the semantic deduplication fingerprint. Review evidence hashes point to
the exact sent items. The heartbeat distinguishes the last sent evidence hash
from its semantic hash. Polling journals remain outside the evidence roots;
the worker excludes its entire output tree, including its task inbox.

Default minimum spacing is 60 seconds between provider-attempt starts, including
rapidly changing evidence and explicit tasks. Failed attempts retry after
60, 120, 240, 480, 960, 1,920 and then 3,600 seconds, remaining at that maximum
delay while errors continue. These are request-pacing controls, not total-call
or total-runtime caps. Pacing options require positive finite values; continuous
minimum call spacing and retry base cannot be below one second. New evidence
or a task cannot bypass an existing retry delay.

One or two consecutive errors give a `degraded` heartbeat; three or more give
`stalled`. Those states remain alive and retrying, with the pending-call flag,
activity, retry delay and next permitted attempt time observable. Success resets
the error streak/backoff and returns health to `running`. Repeated failure does
not stop continuous mode. All failures remain in unique call directories.
Missing usage on any attempt leaves the affected cumulative token total
unknown (`null`), even after a later successful response reports usage.
Call names use a minimum four-digit width, allowing `call-0025` and larger IDs.

## Explicit coordinator tasks and stop control

The continuous run creates an initially empty `OUTPUT/tasks/` inbox. A
coordinator supplies an atomic, immutable `TASK_ID.json` with exactly:

```json
{"task_id":"review-controls-001","task":"Review the stated intervention and sham controls."}
```

The filename stem must equal the public ID, which is at most 64 characters
using letters/digits/underscore/dot/hyphen. The question is nonempty and at most
1,500 characters; the file is at most 8 KiB. Symlinked or malformed tasks are
ignored and reported without provider calls. A first-observed ID cannot change
its sanitized question. A successfully reviewed ID is consumed once per run,
while its file and corresponding call artifacts remain a journal. Failed tasks
remain latched for retry rather than being silently acknowledged. Only public
research questions within this project's approved content scope belong here;
task text cannot grant compute or claim-acceptance authority.

SIGTERM, SIGINT, or a regular `OUTPUT/STOP` file stops this worker. Waiting loops
check stop control at least once per second; an in-flight synchronous HTTP call
can take up to its configured timeout before clean shutdown. Terminal reason
distinguishes `termination_signal` and `stop_file_requested`. The coordinator
must reconcile the old worker, actual process identity, source hash and existing
calls before starting a replacement. This implementation starts nothing.

## Offline validation and remaining limits

The command
`PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_qwen_*.py' -v`
passes **28 tests**, zero skips: 16 new continuous tests plus the existing four
evidence and eight bounded-lease tests. The local run used mocked provider,
clock, signal registration and evidence collection; it performed no HTTP,
cluster, credential or Git operations. Checks include:

- 28 schema-valid mock calls exceeding the former cap, null limits in all
  three metadata classes, correct usage and at least 60-second call spacing.
- More than 26 simulated hours with unchanged scientific evidence and changing
  metadata clocks: only the initial request, with no hourly duplicate or expiry.
- Four failures followed by success at deterministic starts
  `0,60,180,420,660` seconds with retry maximum 240 seconds, visible degraded/
  stalled states and sticky unknown usage. A million-error delay calculation
  stays at the delay ceiling without an overflowing exponent or attempt cap.
- A broken HTTP error-body stream still preserves the failed-attempt journal
  and allows paced recovery. Independent review found QWEN001: list/integer
  provider usage crashed the old usage writer after a valid proposal. The
  repair preserves valid reviews with unknown usage and a clean terminal
  state; both malformed-usage forms have behavior regressions.
- Explicit task execution exactly once, correct exact evidence hashes,
  immutable task identity, malformed/symlink task exclusion and key scrubbing.
- STOP during retry, signal shutdown, preserved bounded defaults/hourly trigger/
  three-error stop, mixed-cap rejection, output-directory immutability, model
  mismatch rejection and token-truncated-answer rejection without fallback.

Transport/proxy/provider availability still determines real uptime. A heartbeat
and PID alone do not establish model participation: actual returned model
identity, completed call metadata, review schema, payload bounds and source
provenance must be verified after deployment. Continuous task/event deduplication
is scoped to one immutable run; reconciliation before replacement remains the
coordinator's responsibility. Tests establish engineering behavior only.
No scientific result or new cluster allocation follows from this change.

Implementation SHA-256:
`8d9e6149f83a91cefdf8cd51c7e38777cdfade4c73cfb42136f104d7ea45a606`.
New continuous-test SHA-256:
`158049f6fe37839d1756a374ef6ab741f61634d43d04c3490269e9e1e56d43d3`.
The existing evidence-test file is unchanged, SHA-256:
`9ada457d1b2211d039dbc679cd2f3d3d6a27180b7daa4c6d48c133f68784c199`.
