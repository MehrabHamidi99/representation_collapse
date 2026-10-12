# Overnight research continuation

The owner requested continued autoresearch until morning. The assumed checkpoint
is **2026-10-13 08:00 Asia/Yerevan = 04:00 UTC**. This continuation does not
increase the existing model-call or cluster-allocation envelopes.

## Verified processes and scope

At 17:00 UTC the existing Tamia Qwen worker had a fresh shared-scratch heartbeat,
nine completed model-identity-verified proposal calls, nine valid reviews,
zero recorded provider errors and 49,965 reported tokens. Its source bytes match
the original immutable `ec1c489` worker. Calls during the earlier inspection gap
are present in its journal; absence of a reusable SSH socket did not establish
worker stoppage. No replacement or restart occurred.

A later 600-character public project update triggered call 10. At 17:26 UTC,
Qwen had ten valid completed calls, zero recorded errors and 55,482 reported
tokens. Its latest evidence contains the exact update bytes, independently
verified against the immutable feed event. At 17:29 UTC the observer's own
process owner/command identity, three observations and zero metadata issues
were verified. Fourteen provider attempts remained at that Qwen checkpoint.

The original run remains at
`/scratch/m/mehrab/representation-collapse/runs/qwen-continuation-20261012T091030Z`.
The recorded worker host is Tamia3. The present controller is Tamia1; a missing
PID on Tamia1 cannot establish that the Tamia3 worker stopped. Fresh heartbeat
and completed-call evidence are distinguished from a fresh original-host PID
identity check. Qwen writes proposals only; schema validity does not accept a
scientific claim or authorize code execution/submission.

The original 24-hour lease started at 09:10:30 UTC on October 12 and ends at
09:10:30 UTC on October 13, subject to the earlier 24-attempt cap or three-error
stop. At the 17:00 check, 15 attempts remained. Failures count; a restart cannot
reset either limit. The morning observer deadline is earlier than that lease
deadline. The observer does not terminate the worker or alter its approval.

## Bounded overnight observation

The reviewed [metadata observer](../../code/ops/research_watch.py), committed in
`60276a3741503a401ff7824c8c92479d9ed8582e`, is running on Tamia1 as PID **1299159**.
It polls every 300 seconds until 04:00 UTC, writes an append-only observation
journal and atomic current/final status, and records stale or inconsistent
metadata without automatically restarting, submitting or cancelling anything.
It reads only public config/heartbeat/status/call-status/usage metadata; it does
not open prompts, provider replies, credentials or endpoint settings.

Output:
`/scratch/m/mehrab/representation-collapse/observers/overnight-20261012T171633Z`.
Source SHA-256:
`4543a30e0fbd17fda94f54013f46171e6a3e814d861e8a39283973b52bbfad14`.
The worker-source pin is
`0f1684fafae8335cacb61ab3198dd08628e1e155fbd5137ccb7da3fae39661c7`.
The first status verified nine attempts, no metadata issues and fresh heartbeat;
the original-host process remains explicitly unprobed from Tamia1.

Observer outputs are outside every configured Qwen evidence root. Routine
polling therefore cannot change Qwen's evidence fingerprint or consume model
calls. The duplicate lock is scoped to the same canonical worker and observer
output parent; reuse that parent for future observation rather than creating
parallel observers elsewhere. The observer's optional scheduler mode is local
and explicit-ID-only; it is not an all-cluster scheduler service.

A temporary local `caffeinate -i` assertion, PID **74629**, ends at the same
morning deadline. It prevents idle sleep without changing persistent power
settings or other projects' processes. Model availability still depends on the
existing local proxy, SSH forwarding and network; the observer records failure
and does not provide a paid or alternate-provider fallback.

## Research advanced in this iteration

Independent review accepts the scoped observer after repairing two reproduced
metadata errors: a terminal state could otherwise coexist with an unfinished
call, and a completed error streak could disagree with the heartbeat. The
observer has 22 focused tests; the complete source suite passes **150 tests,
no skips**, in 12.861 seconds. This is engineering evidence, not learned-model
evidence.

The [statistics proposal](statistics-freeze-proposal-v2.md) separates three
estimands, preserves failed seeds, defines paired seed-level delta16/DID and
the eight-test family, and states exact inference/freeze blockers. An independent
exact counterexample confirms that a mean-null condition alone does not justify
shifted sign flipping. The proposal remains unfrozen; new seed IDs, measurement
manifests and confirmation allocations are not accepted by this document.

The [GPU-efficiency guide](gpu-efficiency-adoption-2026-10-12.md) is permanently
adopted, with all six local/Mila/project package files matching. Native cluv
0.1.5 verified the nine Alliance targets at 17:01–17:02 UTC, and their project
queues were empty. Mila's matching project queue was empty at 17:11 UTC through
the existing Mac-origin connection. Fir/Nibi expose account/partition/typed-GPU
availability, but updated training source and offline environments are missing.
The four-fit, two-requested-GPU-hour canary remains unapproved and unexecuted.
No live owned GPU exists to profile; missing telemetry is retained as unknown.

## Morning handoff

Read the observer's final `status.json` and `observations.jsonl`, then reconcile
the Qwen run's terminal status, heartbeat and every call before any recovery.
Review proposals against verified evidence; do not execute them directly or
accept harmful collapse/Floquet certification from a proxy. Retain all failures
and the original lease counters. Source/cache readiness and exact allocation
approval still precede new GPU work. The existing periodicity results remain
descriptive because all original competence gates failed.

Small receipts and test/review logs are in ignored `runs/ops/` and `runs/review/`.
Their hashes and final publication revision are recorded in the dated notebook,
[tracker](../TRACKER.md) and structured status.
