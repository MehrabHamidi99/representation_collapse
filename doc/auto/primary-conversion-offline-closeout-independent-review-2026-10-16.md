# Independent offline-reporting closeout — Mila 11273685

2026-10-16. **Accepted for the narrow completed-job engineering and durable
offline-reporting closeout. No blocker remains after the supplemental inventory
pin check.** This does not establish W&B server delivery, trained-model results,
scientific acceptance or authority to repeat the job.

The independent local checker passes **107 assertions** against saved captures,
returned metadata and the 3,472-byte transaction. It performs no remote action,
SDK run initialization, dataset parsing or model work. The existing W&B 0.30.0
protobuf descriptors are loaded as data by the checker; its framing follows the
retained pinned decoder contract and the focused
[smoke reader](../../code/ops/wandb_smoke.py#L14). The older general decoder's
SHA256 remains
`582470d0ff6dd0fcd0ef31f3ce17e36d65ffa66b089b2fec8cb7b26af173d000`.

## Verified evidence

All eight returned files match their captured base64 byte lengths/hashes and
local retained copies. The plan matches the externally expected SHA256
`b17364f2d746dab6087599142e850e3b3bebc06379179e04151700fc241f9dcb`;
all 14 frozen local source files match its published `0fe14aa` source manifest.
All 11 JSON bodies in the prior inspection reproduce their recorded canonical
bytes/hashes. The collector records a clean source checkout, six admission
reference rehashes, 20 regular output-file rehashes and one SDK log symlink
inspected without following it. Those are captured remote observations, not
fresh measurements by this reviewer.

The transaction independently matches the prior fixed inventory:
SHA256 `2fb8476387762d191b57809c372056ccdefb9fb3c884045a2581a15968d5573f`.
The version-0 header, all 19 FULL-record frame lengths, IEEE CRC32 values and
protobuf records decode consistently. There is one header, one run, four
telemetry records, nine summary records, three history records and one exit.
Corruption and truncation are rejected. This small single-block check does not
claim support for arbitrary fragmented or larger production streams.

The **actual protobuf** run ID is
`f177579e09d943c5917bd68522c478a7`, matching the filename and reporting receipt;
the actual sole exit code is **0**. Config/identity mappings agree exactly with
the receipt, project is `representation-collapse`, and hostname is the neutral
`not-collected`. Committed steps are exactly 0, 1 and 2: conversion start,
conversion completion (337,920 rows, 3,072 snapshots, 2,997,959 output bytes), and
independent verification completion (3,072 snapshots, three splits, one attempt).
These counts agree with the retained conversion/verifier receipts; this review
does not rerun their numerical/data work.

Final reporting state is completed, last attempted/completed step is 2,
`finish_returned=true`, and all explicit terminal summary values agree. No
environment, file-upload, system-statistic, console/output or artifact record is
present. Normal version/protocol telemetry is distinguished from those forbidden
records. Remote delivery/history flags remain false and sync is still required.

## Inventory binding and chronology correction

The initial collector checked outputs against the current remote inventory
without independently pinning that inventory to the earlier inspection. The
supplemental `closeout-retry002/evidence-inventory-pin-check.json` resolves this:
the captured current inventory is 3,855 bytes with SHA256
`2b83834d9e739bf5930cf67b4f4882ff1509763d03b09fe4ec7315b63482e61a`,
exactly the prior pin. Together with the recorded 20 rehashes, this closes the
identified reconciliation gap. It is cooperative retained-file evidence, not a
general concurrent-writer or same-user adversarial-security guarantee.

The new collector explicitly sets **TZ=UTC**. Its accounting establishes
**October 16, 02:09:36–02:11:42 UTC**, elapsed 126 seconds, COMPLETED/0:0,
one CPU/2 GiB and no GPU allocation. Timezone-aware reporting timestamps and all
three history timestamps fit that interval. Earlier unqualified scheduler and
SDK-directory times around October 15 22:09 must not be labeled UTC. A new
notebook correction should preserve the earlier observation and state this
resolved timezone; no execution was repeated to obtain it.

The reviewed collector uses reads, hashes and accounting queries only. Returned
files are seven compact metadata JSONs and this transaction; no dataset or
checkpoint body was transferred. No parser rerun, SDK initialization, submission
or dataset/checkpoint/job mutation is performed by the reviewed closeout path.
The original one-submission allowance remains consumed. Authenticated later sync
and independent server history/summary readback remain separate work.

Immutable independent receipt:
`runs/review/primary-conversion-offline-closeout-20261016/receipt.json`, SHA256
`7dd15a2c49c1288d58699c2b7504e40cbba73333f507466fcba6fe802870448a`.
It binds the collector, verifier, result, original inspection, supplemental pin,
decoder/protobuf identities and independent checker. Earlier failed transport
attempts and their receipts are preserved; none authorizes a retry of conversion.
