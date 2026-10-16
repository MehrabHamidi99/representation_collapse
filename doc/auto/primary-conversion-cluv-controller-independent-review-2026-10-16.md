# Independent review of primary-conversion cluv controller

2026-10-16. **Accepted as preparation-only controller implementation.** No
blocking controller finding remains in the frozen source after the repairs
below. This is not a fresh owner grant, a submitted job or proof of worker
integration/runtime success.

Reviewed `code/ops/submit_primary_conversion.py`, SHA256
`eda63e23a6c62d21cc008f77af42dd9cbba51f0964cf07fd8dedb014895918fb`;
author tests SHA256
`9411e0b619492d1260c8f9a3f12c0fb3c65bc2943b22c81bf20299332a7079d2`;
and [controller protocol](primary-conversion-cluv-controller-2026-10-16.md), SHA256
`c915d5bdd62920296d4d1d6235fba569f59092724a0bc1196fc86bb9aa763b20`.
No author source/config/protocol was edited by this reviewer.

Review criteria were frozen before author handoff in
`runs/review/primary-conversion-submit-20261016/independent/acceptance-criteria.json`,
SHA256 `cbc16798dc7bc43c7778d555bffff6b7aeef92193cef3bf1d81fbd4d4afe8241`.
Sources include the current conversion worker/flat-plan contract, previous
reviewed `submit_program_v2.py` and actual local cluv0.1.5 source. The six native
distribution member hashes match the previously reviewed pins. Local installed
distribution metadata/source verification was executed without importing cluv
or invoking any native cluv API.

## Admission, source and immutable transaction

Preflight requires normal Python, the expected Mila login-host suffix and
compatible ambient cluster identity. It rejects controller SBATCH_* variables
and present SLURM_CLUSTERS, including empty values. The exact external plan
hash, one-submission conversion-only flags, allocation/resources and finite
limits are checked with type-sensitive JSON equality. The config and complete
source closure include the controller/protocol, worker shell/Python, helper
sources, package init, reconstruction config and launch request.

Every source/helper hash must match. Git HEAD and the locally recorded
origin/main ref must equal the admitted commit, and status must be clean. The
origin ref is local metadata: root's actual fresh publication/staging verification
remains necessary and is not replaced by this offline check. A technical enabled
config alone cannot authorize allocation.

The pinned owner receipt requires the actual owner-approved kind/status/actor
and true decision, with exact source/config/output/STOP/count/resource bindings.
Readiness/environment/storage receipts must match, with a conservative whole
project reservation. A fabricated JSON receipt is not evidence of the owner's
decision; root owns that provenance. Review fixtures intentionally use fake
approval records and never confer real authority.

Before author freeze, review prompted stronger bounded no-follow metadata reads,
hashing/decoding from one stable byte snapshot, late plan/owner/STOP/output/claim
rechecks, strict isolated log paths and imported-library origin checks. These
repairs are present in the reviewed source. Symlinks, hardlinks, FIFOs, growth,
duplicate JSON keys and nonfinite constants fail closed in the exercised paths.
This is cooperative operation/integrity protection, not an OS sandbox against
malicious same-UID Python or filesystem manipulation.

The claim payload includes the exact source/run/caps/count binding and has a
precomputed byte hash in the plan. It deliberately excludes the plan hash,
avoiding a plan→claim→plan hash cycle. Plan SHA is retained in intent, preview
and outcome. Exclusive no-follow creation and fsync establish intent followed
by the single claim before **native submit**. Earlier source import and local
configuration/cache preparation do not dispatch a job. Cache, evidence and
log paths are restricted to new owned project run locations; no global config
is changed. A repeated call cannot overwrite prior claim/intent/outcome or retry.

## Native expansion and actual command

Distribution version0.1.5 and six source pins are checked before import; actual
imported file paths/hashes must match before native configuration API use.
Fresh cluv namespaces are required. The new XDG cache is established first;
in-memory global env/sbatch options are cleared and the frozen Mila partial
config is replaced with one allocation. Only Mila remains in that process's
cluster mapping. No sync/install or fallback is introduced.

Exactly one guarded `get_submissions` expansion is allowed. The row must be
local Mila, unchunked, with exact source commit, shell/argv and normalized
resources: one node/task/CPU, 2 GiB, ten minutes, zero GPUs and no requeue.
All native-added name/output/chdir/export fields are exact. Extra rows,
remote targets, chunks, GPU/array flags, bool/integer substitutions and changed
source/argv are refused before a scheduler command is returned.

The actual shell command is checked separately from its metadata: exact
`bash --login -c`, admitted environment/source prefix, unique approved sbatch
flags and exact shell/arguments. Independent fixtures keep metadata correct
while changing hidden command resources, duplicate flags or adding GPU/array
input; those are rejected. A guard is prepended to the actual login-shell body
before sbatch to reject profile-introduced SBATCH_* and SLURM_CLUSTERS or
optimization input. Static inspection plus fake-row fixtures verify guard
placement; no real login shell or sbatch was executed.

The single submit call explicitly fixes autocommit=False, chunking=None,
vram=None, _skip_sync=True, sync_datasets=False and parsable=True. There is no
automatic retry/race/resume branch. API exceptions, None or invalid job identity
retain ambiguous reconciliation status, claim and available evidence; error
text/credentials are not copied into outcomes. Function wrapping is restored.
An abrupt process/disk failure can leave only earlier intent/claim/native logs;
root must reconcile those before any further action.

Actual pinned cluv source can return a singleton **PENDING** job. The controller
correctly labels a returned ID as requiring scheduler reconciliation and never
infers RUNNING or successful conversion from that return value.

## Checks and remaining integration

All **14 new independent fixtures pass in 0.410 seconds**:

```text
PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_primary_conversion_submit_independent.py -v
```

All **9 author fixtures were independently rerun and pass in 0.115 seconds**.
Python3.10 AST checks pass. New fixtures cover independent claim bytes/order,
full fake approval/preflight, published/dirty Git refusals, receipt rebinding,
resource/path/source/library changes, late mutation/STOP, bounded stable I/O,
actual command mismatch, multiple expansions, success/None/exception/invalid-ID
outcomes, prior-attempt preservation, and fake imported-module origin/cache
configuration. No native cluv configuration/submission function, SSH, scheduler,
SDK, worker, dataset or model was invoked. Native API behavior is exercised
entirely with fakes; the real local metadata/source hash check is distinct.

Root must still integrate controller/protocol source entries into the enabled
config and freeze the corresponding worker policy/source pins. Controller
fixtures add those entries; they do **not** prove that the current worker's
frozen config-policy hash accepts the new integrated config. Exercise the actual
worker `load_config` and exact source closure as a separate local integration
check before publication/staging/approval. Then bind final immutable source,
owner decision, readiness and new claim/plan bytes, and verify the actual cluv
and scheduler environment before the one authorized call.

After any launch, actual job/source/admission, scheduler outcome, cgroup limits,
conversion plus independent primary reduction, offline reporting and owned
cleanup require separate verification. The main research goal and trained
representation-collapse questions remain unfinished.
