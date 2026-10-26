# Mandatory experiment reporting

2026-10-14. `src/representation_collapse/reporting.py` is additive infrastructure
for newly admitted runs. It does not launch an experiment or retrofit W&B into
historical runs. The `reporting` dependency group pins **wandb==0.30.0**; existing
experiment, ops and paper groups are retained, as are all previous package
versions in the lockfile. Use `uv sync --frozen --group reporting --inexact` when
adding this group to an existing experiment environment without removing its
other installed groups. Cluster installation remains separately authorized.

## Usage and lifecycle

Enter `ExperimentReporter` before input generation, any numerical sweep or
training step. Constructor validation does not initialize tracking; successful
context entry does. The initial durable receipt has `work_may_start=true` only
after the exact SDK version, requested mode and effective privacy settings have
been checked and initial summary logging has succeeded. Missing SDK, failed
initialization or mismatched settings prevents context entry.

```python
from representation_collapse.reporting import ExperimentReporter

with ExperimentReporter(
    output="runs/example/attempt-001/reporting",  # must be a new directory
    project="representation-collapse",
    entity=None,  # permitted offline; online requires the actual owning entity
    mode="offline",
    config={"protocol_id": "reviewed-example", "seed": 0, "cpu_threads": 1},
    identities={"config_sha256": "<actual 64 lowercase hex characters>"},
) as report:
    # Authorized work begins only here.
    report.log({"calibration/conditions_completed": 1}, step=0)
```

The example hash is a placeholder and deliberately fails validation until
replaced. Supply actual reviewed source/config, input/split, seed and environment
identities appropriate to the experiment. Metadata is explicitly selected by
the caller; no namespace, model config, environment dictionary or directory is
automatically serialized. The reporter is not an experiment authorization gate.
Each integration must still enforce its source/config/resource freeze and
scientific admission before entering this lifecycle.

Steps are nonnegative integers and strictly increase. There is one explicit
committed history record per call. Repeated/decreasing/bool/noninteger steps are
rejected. The local receipt distinguishes the last attempted and completed SDK
log calls; SDK return alone does not establish remote acceptance. Context exit
writes completion or failure summary and calls `Run.finish`. Experiment
exceptions retain their original type/propagation; the receipt records only the
exception class, never its message/traceback. Logging or finalization failures
raise `ReportingError` and leave a failure receipt. No failure disables logging
or silently changes modes. Initialization timeout is 20 seconds and finish
timeout is 8 seconds with timeout raising enabled. These SDK limits do not
replace the experiment launcher's external wall-clock deadline.

Run IDs default to fresh UUIDs and existing output directories are refused.
Online calls use `resume="never"`; W&B ignores that option offline, so callers
must retain the default fresh ID or ensure any explicitly supplied ID is globally
unique before a later sync. No training resumption or implicit run merging is
implemented. A future resume needs a separate lineage/step contract.

## Explicit offline mode and delivery evidence

The owner-authorized project was created and server-verified on October 13 at
21:05 UTC: [representation-collapse](https://wandb.ai/mehrab-hamidi-milaquebec5431/representation-collapse),
entity `mehrab-hamidi-milaquebec5431`. Safe destination metadata is stored in
`configs/wandb.json`; credentials remain in their existing protected locations.
An initial SDK lookup mistakenly used the full entity/project path as a name
and returned a lazy object. Its malformed URL was rejected before any run
initialization. A corrected authenticated server listing established absence,
then exactly one project creation and one logging-only online run succeeded.

The [transport smoke](https://wandb.ai/mehrab-hamidi-milaquebec5431/representation-collapse/runs/rc-reporting-smoke-20261013t2058z)
used Tamia's existing SDK 0.21.0 (distribution `0.21.0+computecanada`). Independent
API readback verified the finished state, both history records at steps 0/1,
and final summary with zero native calls/updates and no scientific acceptance.
The receipt SHA-256 is
`71bde8c6b550262d3aef1dd9a9296c5d5fa92f2f02589ad246270e0322952a9b`,
under `runs/ops/status-check-20261014T0044Y/wandb-online-transport-smoke-attempt-002.json`.
This certifies the authenticated project and that transport route. The new
SDK 0.30.0 helper has separately verified offline evidence below; it has not
yet delivered a run remotely. Historical scientific runs have not been synced
or retroactively assigned live tracking. The transport smoke is infrastructure
evidence, with no numerical experiment or cluster job submission.

Offline is the default, including when an entity is not yet authenticated. The
project name remains explicit. Missing authentication does not block a durable
offline run. Root confirmed the owning project separately; safe destination
configuration belongs to `configs/wandb.json` when written by the coordinator.
This module never reads credentials or creates the remote project. An online
call requires an explicit entity and relies on the SDK's preexisting authorized
authentication. It never calls login or retries offline after an online failure.

The output contains atomic `reporting-receipt.json`, the durable W&B run directory
and transaction file. The receipt includes the run ID, requested/actual entity,
mode, online URL when available, scalar config/hash identities, terminal summary
and timing. `remote_delivery_verified` and `remote_history_verified` remain false
in both modes: an independent authenticated API read of history and summary is
required to certify server receipt. Offline runs additionally require an
explicit later authenticated sync with its own receipt; no sync occurs here.
The offline output must be retained until that step is verified.

## Metadata boundary

Supported data is compact JSON scalars: finite numbers, bools, nulls and short
strings, with at most 64 entries and 16 KiB per mapping. Hash identities accept
only lowercase SHA-1/SHA-256 hexadecimal strings. Arrays, tensors, nested user
objects, file paths as identities, common secret/raw-data keys, nonfinite values,
and oversized values are rejected. Caller-supplied strings must be public
metadata; this validation is not a general secret-detection mechanism.

Explicit SDK settings disable code saving, Git collection, console/logger
capture, SageMaker/Launch auto-config, system/machine metadata, automatic system
metrics, requirements capture and incidental file upload. Config-path injection,
ambient sweep/launch/config variables and a local `config-defaults.yaml` cause
a refusal before SDK initialization. The effective settings and final user
config are checked after initialization. SDK internal transaction records still
contain its normal version/protocol telemetry; the promise is that raw
datasets, full weights, environment snapshots, code and arbitrary files are not
submitted by this interface. No `watch`, `save`, model artifact or arbitrary
file-upload API is exposed. `log_manifest(name, metadata=..., identities=...)`
validates scalars/hashes, constructs its own immutable JSON of at most 16 KiB,
and submits only that file as a `calibration-metadata` artifact with
`policy="immutable", skip_cache=True`. The original JSON remains in the
reporter output tree; retain it and recheck its receipt hash before sync. It accepts no
source path or array. Its hash and pending remote-delivery status enter the
local receipt. Incidental file upload remains disabled; this intentional
metadata-artifact API is the only file-upload exception.

Automatic host statistics are disabled to avoid broad collection. Authorized
experiment code should explicitly log its approved CPU time, RSS, useful work
counts and elapsed time as scalars. This reporter does not make resource
utilization or scientific acceptance claims.

## Validation and SDK sources

Thirteen isolated lifecycle tests use a fake SDK: initialization/finish/log errors,
original exception preservation, mode mismatch, monotonic steps, privacy
settings, payload rejection, version mismatch, fresh output directories an explicit online route without authentication/network, and a metadata
artifact fixture that rejects raw files/arrays and verifies its generated JSON. Run:

```sh
PYTHONPATH=src .venv/bin/python -m unittest tests.test_reporting -v
```

A two-record real SDK smoke uses `code/ops/wandb_smoke.py`, defaults offline,
and performs zero model forwards/fits/label accesses. It supports an explicitly
authorized later online transport smoke with `--mode online --entity ACTUAL`.
It checks that `finish` returned, then independently parses its small local
`.wandb` transaction stream using the SDK protobuf schema, verifying header and
IEEE CRC32. It requires exactly two committed history records, the explicit
config, completed summary, exit and metadata artifact records, and absence of
automatic environment/files/stats/console records. The run hostname must be the
explicit neutral `not-collected` value. This focused reader supports single-block
FULL-record version-0 smoke streams; it deliberately rejects larger/fragmented
production streams. It hashes the stream and lifecycle receipt. It does not
establish server delivery.

```sh
PYTHONPATH=src .venv/bin/python code/ops/wandb_smoke.py \
  --output runs/ops/wandb-setup-20261014/smoke/new-offline-attempt
```

The initial `smoke/offline-01` attempt stopped before work when sandbox policy
blocked the local W&B service socket. That failure receipt is preserved.
`smoke/offline-02` then completed with narrowly escalated local-socket permission,
explicit offline mode and cache/config/data directories inside the ignored
smoke subtree. Run ID `8e89d5ed6d544a3283010ab47e610174` verified the initial lifecycle; deeper
inspection found a hostname in its run header despite metadata capture being
disabled. That immutable receipt remains historical evidence. The reporter now
sets `host="not-collected"` explicitly. A focused `smoke/offline-03` run
`a4ae3eefef334f3b82ad5c58d2732791` verified the final metadata-artifact API,
exactly two saved history records at steps 0/1, explicit config, completed
summary, neutral host and absence of automatic environment/code/files records.
It used the actual entity supplied by root in explicit offline mode, without
authentication or sync. Inspection also showed the SDK default artifact policy
depended on a separate staging cache. The final explicit immutable policy now
keeps metadata JSON in its own durable output tree. `smoke/offline-04`, run
`dfb0e42e08ee4564819df5eb40e0abb1`, verified the final source, all stream
checks above and matching artifact bytes/hash inside that tree. Two saved
parser fixtures confirm rejection of CRC corruption and a missing history
record. Earlier smoke evidence is retained. The smoke does not alter scientific
counts or authorize any numerical calibration.

APIs used are `wandb.Settings`, `wandb.init`, `Run.log(...,step=...,commit=True)`,
`Run.summary.update`, `Run.finish(exit_code=...)`, `wandb.Artifact`,
`Artifact.add_file` on self-generated JSON, `Run.log_artifact`, and run identity/settings/dir
properties. Settings include version-specific internal `x_disable_meta`,
`x_disable_stats`, `x_disable_machine_info`, `x_save_requirements`; upgrading the
SDK requires review and a repeated offline smoke. References are the official
[v0.30.0 settings source](https://github.com/wandb/wandb/blob/v0.30.0/wandb/sdk/wandb_settings.py),
[init source](https://github.com/wandb/wandb/blob/v0.30.0/wandb/sdk/wandb_init.py),
and [run source](https://github.com/wandb/wandb/blob/v0.30.0/wandb/sdk/wandb_run.py).
The installed PyPI 0.30.0 runtime was also checked directly and exercised; package
artifacts are hash-bound by `uv.lock`. No existing experiment implementation,
measurement contract or execution config is changed by this infrastructure.

Artifact APIs follow the official [artifact source](https://github.com/wandb/wandb/blob/v0.30.0/wandb/sdk/artifacts/artifact.py).

The smoke reader follows the official [transaction-log reader](https://github.com/wandb/wandb/blob/v0.30.0/core/internal/transactionlog/reader.go) and version-0 LevelDB framing. No protocol-buffer contents from experiments are read by this smoke.

## First numerical runs recorded offline

The complete-state forward calibration ran once from published `edc47ee` on
October 13 at 21:31 UTC. Offline run `94871a2d43e6406f8022e763a905f14e` has 32
verified committed history records at steps 0–31, completed summary and two
compact metadata manifests. Transaction SHA-256 is
`934de7bd6d21f778e499304b68b71e1cf9dbd3febb7683ca4d70c0f1f120dc1c`.
A separate NumPy reduction from `70b3a05` ran at 21:40 UTC; offline run
`6055233d62b4469796c50122e4cfa8ce` has one verified history record, completed
summary and one metadata manifest. Its transaction SHA-256 is
`d96706e1436caab1cac7149d501a3c7dc9058f71f69066f0cd5a835d13ea4429`.
Both SDK 0.30.0 reporters initialized before their numerical work. These are
actual contemporaneous offline experiment logs, not retrospective annotations.
Neither numerical run has a verified cloud delivery receipt; the dashboard's
verified online run remains the earlier SDK 0.21 transport smoke. Raw arrays
and weights are retained locally outside the W&B artifact boundary.
