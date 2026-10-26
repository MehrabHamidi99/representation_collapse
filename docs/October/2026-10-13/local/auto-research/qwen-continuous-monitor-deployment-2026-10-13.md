# Continuous Qwen monitor deployment

2026-10-13. Both first five-minute observations and process identity verified.
The monitor remains active through its independent deadline. This is execution
evidence for deterministic monitoring and model participation, not scientific
acceptance or a new compute allocation.

The reviewed metadata observer from immutable source
`999d88d8679c0f353c6bb7fb4f5ab8433a7e730c` runs inside existing Mila job
**11228391** on **cn-f003.server.mila.quebec**. The original shared allocation
expires October 14 **08:55:56 UTC**. Observation has its own earlier deadline,
October 13 **20:00 UTC** (midnight Asia/Yerevan); this does not cap the continuous
Qwen leaf. The job provides two shared CPUs, 4 GiB and zero GPUs for compact
coordination. No additional allocation, training, model weights or heavy build
was requested.

The approved project-owned session `representation-collapse-qwen-20261013`
already contained the Qwen supervisor. A fresh process/window reconciliation
found no reviewed `research_watch.py` observing this run. Exactly one window,
**@7**, named `metadata-monitor-20261013`, was added inside that existing session.
The original supervisor/window, Radius infrastructure, shared Slurm-owned
Tamia socket and other projects' claims were preserved.

The monitor's exclusive claim
`representation-collapse-monitor-20261013T1001Z` was registered at
**10:05:37 UTC** before runtime mutation. A small project-owned wrapper updates
that claim every 60 seconds and owns only this metadata observer. Wrapper PID
**3248325** and observer PID **3248354** were independently checked at
**10:07:41 UTC**: both match the owner UID and job 11228391 cgroup on the actual
compute host. Their command identities match their separate wrapper/observer
roles. They remain attached to the allocation-owned tmux server after the
temporary inspection shell closes.

Source SHA-256:
`3f260d12e5ea400a50280d35cf739b4547b3dd1cc3125c7ded16de7ed748e034`.
The claim-wrapper source is separately retained in ignored local receipts;
SHA-256 `d75f7131ddd338104cbd259f2166a4209e2a2d27d257f6b571ea10c06e2bdfec`.
The observer did not require a second source copy: its pinned file already
existed in the immutable staged checkout. Only the small claim wrapper was
transferred, checksum-checked and compiled before execution.

## Observation scope

The exact worker run is
`/network/scratch/m/mehrab.hamidi/representation-collapse/runs/qwen-continuous-20261013T093936Z`.
The expected leaf source SHA is
`8d9e6149f83a91cefdf8cd51c7e38777cdfade4c73cfb42136f104d7ea45a606`.
The observer uses `--allow-continuous`, that explicit source pin, original
compute-host PID **3244371**, interval **300 seconds**, staleness **600 seconds**
and deadline `2026-10-13T20:00:00Z`. It requests no scheduler inspection or model
provider calls.

Its durable journal is
`/network/scratch/m/mehrab.hamidi/representation-collapse/monitoring/qwen-continuous-20261013T093936Z/monitor-20261013T1001Z/observer`.
This lies outside every leaf artifact root (`source/runs`, `source/logs` and the
project's separate evidence directory), and outside the entire leaf output,
tasks and proposals. Monitor/claim heartbeats therefore do not become model
evidence and cannot create provider-call feedback. The private wrapper runtime
uses its own `/tmp/radius-control/representation-collapse-monitor-20261013T1001Z`
directory. No credentials or endpoint configuration enter these receipts or Git.

At the first observation, **10:07:05.575407 UTC**, Qwen reports **5 attempts,
5 schema/model-verified proposals, zero errors, four completed coordinator
tasks and 24,774 reported tokens**. The incremental cursor is **6**, with five
completed records observed and no metadata issues. All observed cumulative
counts and usage reconcile at this checkpoint. The run's whole completed
history fits in this bounded pass; previously observed-file immutability is
explicitly uncertified. The observer's PID-existence probe does not certify
leaf command identity; the separate launch/identity audit provides that evidence.

The second observation at **10:12:05.575592 UTC** records **6 attempts, six
verified proposals, zero errors and 29,862 reported tokens**. The cursor advances
to **7**, with one newly observed completed call and all observed totals
reconciled. Both observations have no metadata issues; their gap is
**300.000185 seconds**. At the **10:13:00 UTC** identity check, both monitor
processes still match their UID/job/role, and the independently updated claim is
55.1 seconds old. The journal SHA at this two-observation cutoff is
`b58e24a5634845cc7c7504ee63375c292aaba66d8e3e3986bd8ffc4ea141d40e`.

Partial future journals remain explicitly partial until the cursor catches up.
Degraded/stalled means retrying; stale heartbeat or unavailable transport does
not mean stopped. Missing usage remains unknown. A stop requires the consistent
terminal/heartbeat/journal contract described in
[continuous monitor semantics](continuous-monitor-2026-10-13.md). Every proposal
and observation retains `scientific_acceptance=false`.

## Provenance and handoff

Ignored local receipts live under `runs/ops/qwen-continuous-monitor-20261013/`:
registered claim, window launch, first observation/process audit and wrapper
source. An initial SSH entry failed when inherited local forwarding was requested
again; client-only `ClearAllForwardings=yes` reused the existing master without
requesting Duo or changing shared channels. Tool approval review also rejected
one combined terminal input as too large to review safely. No remote mutation
occurred in that rejected call. Separate smaller claim, checked source transfer
and window launch actions then succeeded. Both failed attempts remain recorded.

Shared Radius heartbeat was separately observed as `observer_exited` during
preflight; that concerns its own service and is left to the shared owner. This
project's Qwen supervisor and leaf remained live. No provider dispatch migration,
other project's job/claim takeover, model-service fallback or new scientific
decision is implemented by this monitoring task.

Handoff remains with this project's coordinator. The monitor's own STOP file
is `/tmp/radius-control/representation-collapse-monitor-20261013T1001Z/STOP`;
it affects only the metadata observer. The leaf's distinct STOP controls remain
with its owner. The monitor claim continues its automatic 60-second heartbeat
until observation ends. A durable `handoff.json` in the monitor journal records
this ownership and deadline. Inspect the existing observer/claim before any
replacement; do not start a duplicate while it is active. Observation ends at
20:00 UTC without changing the continuous leaf's model-call authorization.
