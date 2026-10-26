# Prepared primary-data conversion allocation — October 16

**Prospective conversion plus independent verification. No allocation is authorized by this file.**
This revision adds the separately implemented, reviewed raw-to-artifact oracle
to the previously unexecuted preparation packet. Earlier source/fixture receipts
refer to their exact older hashes; no earlier allocation is reused.
All earlier CPU/GPU slots are consumed. Root owns scientific acceptance, source
freeze, actual scheduler/cluv inspection, staging, owner approval and submission.
The primary CSV body has not been inspected in this implementation task.
The accepted verifier protocol is [tgrab-primary-data-verifier-v1.md](../../../2026-10-26/local/auto-research/tgrab-primary-data-verifier-v1.md).

The requested future allocation is exactly one Mila/native-cluv0.1.5 CPU job:
account `mila`, QOS `normal`, partition `main-cpu`, one node/task/CPU, physical
2 GiB, ten minutes, zero GPUs, no requeue, race, retry, resume or replacement.
Maximum request is 1/6 CPU-hour. This is conversion/integrity work only, with
zero model, optimizer, evaluation or generation calls. A lexical/schema failure
is a valid terminal result; do not edit accepted policy to make the file pass.

Reuse the existing Python3.10.19/W&B0.30.0 environment at
`/network/scratch/m/mehrab.hamidi/representation-collapse/environments/graph-native-cpu-20261015-calibration01/venv`.
No environment installation, download, numerical model import or scientific
array computation belongs to this job. The reporting SDK may load its own
transitive dependencies; no claim of an empty numerical-package sys.modules
set is made.
Original bytes remain at
`/network/scratch/m/mehrab.hamidi/representation-collapse/datasets/tgrab-603388f332c7756f90bc1702b26053c2bc834654-primary/primary.csv`:
6,907,326 bytes, SHA256
`5a19f9e7949668473bc61cb59b6215f25f093ea0ee2d45b5e143338ded5523e3`.
Only its fixed header has been inspected previously. The conversion protocol is
[tgrab-primary-data-conversion-v1.md](../../../2026-10-26/local/auto-research/tgrab-primary-data-conversion-v1.md), with
the [trainer contract](../../../2026-10-15/local/auto-research/tgrab-training-runner-implementation-contract-2026-10-15.md)
and [payload receipt](../../../2026-10-15/local/auto-research/tgrab-primary-payload-staging-2026-10-15.md) supplying context.

## Freeze and admission before approval

The checked-in `configs/graph/primary-data-conversion-v1.json` is now a pinned,
technically enabled candidate (`execution_authorized=true`) so the exact runnable
source can be independently reviewed, published and staged **before** requesting
approval. This flag grants no allocation: the runner separately requires an
exact external owner-approval receipt and exclusive one-submission claim, which
root may create only after a fresh owner allocation decision. No such receipt
or submission exists at this freeze.

The four pins are converter, conversion protocol, independent verifier and
verification protocol. All four match independently accepted exact source bytes.
The policy hash masks only these four pins and the technical execution flag;
raw identity, resources, all attempt/output caps, API, reporting and other fields
remain fixed. Any new source must be independently reviewed and published before
updating an external plan. No commit ID is embedded in source that hashes itself.

Prepare a new external coordinator plan, not a tracked enabled approval file.
Its required fields are `kind=primary_data_conversion_single_submission_plan`,
`owner=root coordinator`, `protocol_id=primary_data_conversion_v1`,
`execution_authorized=true`, `training_authorized=false`,
`scientific_acceptance=false`, `max_submissions=1`, `retries=0`, exact
`source_root`, 40-hex `source_commit`, `config_sha256`, `output`, `stop_path`,
and exact config `allocation`, `resources`, `limits` mappings.

`source_files` is an exact relative-path/SHA256 mapping containing the Python
runner, shell wrapper, enabled config, this request, package `__init__.py`,
reconstruction config and every entry in config `sources` (converter, conversion
protocol, reporter, ownership helper). No missing, additional or nonexact source
pin is admitted. The exact accepted helper and reporter hashes remain
`8e1f388d4f5ce72e0bfd9a4b5a4b1f8c98f66f2512841d96875822bfd9bffe09` and
`443c9e89d2eba954ee60cc35e015fe61196fbec2f50331067540b4a8259e8184`.

Six external receipts each use `{path,sha256}` and must have `status=verified`:

- `implementation_review` and `reporting_readiness`: exact enabled
  `config_sha256`; the latter must exercise the actual compact metadata through
  the actual reporter constructor and review the retained offline lifecycle.
- `owner_approval`: exact `config_sha256`, `allocation`, `resources`,
  `source_commit`, `source_files`, `output`, `stop_path`, `limits`,
  `max_submissions`, `retries`; root must
  create this only after the owner's fresh allocation approval. CLI booleans
  and a locally manufactured receipt confer no authority.
- `environment`: `python=3.10.19`, `wandb=0.30.0`, exact `python_executable`.
- `storage_gate`: exact `source_root`, `output`, nonnegative `used_bytes` and
  `reserved_bytes`. Reservation is at least 469,762,048 bytes (192 MiB converter
  ceiling plus 256 MiB evidence ceiling); root must additionally include new
  source/controller/log/cache needs in its conservative reservation. Used plus
  reserved must remain <=21,474,836,480 bytes for the whole project scratch.
- `submission_claim`: `kind=root_exclusive_submission_claim`, `exclusive=true`,
  `claimed_submissions=1`, unique compact `claim_id`, and the exact source commit,
  source-files mapping, output/STOP paths, limits, max-submissions and retries
  bound above. Root's controller must claim this submission exclusively and
  reconcile any ambiguous scheduler outcome rather than make another attempt.

All runtime source/output/STOP paths must be canonical absolute project scratch
paths without symlink components. Output is fresh, with existing parent. STOP
is absent and outside output. The plan is hash-pinned externally and binds
`supervisor_argv` to the following exact list, with the literal final
`{plan_sha256}` placeholder preventing a circular hash. Runtime substitutes the
actual externally supplied plan SHA and records the full actual argv.

```text
[<venv>/bin/python, -B, <source>/code/ops/graph_data_conversion_job.py,
 --plan, <inputs>/plan.json, --plan-sha256, {plan_sha256}]
```

Concrete unique layout; `<scratch>` is exactly
`/network/scratch/m/mehrab.hamidi/representation-collapse`. No path is reused:

```text
<scratch>/sources/primary-conversion-verified-20261016-attempt001/
<scratch>/runs/primary-conversion-verified-20261016/inputs/plan.json
<scratch>/runs/primary-conversion-verified-20261016/attempt001/
<scratch>/runs/primary-conversion-verified-20261016/STOP
```

## Native cluv controller and future execution boundary

The source-bound controller is `code/ops/submit_primary_conversion.py`, with
[its protocol](primary-conversion-cluv-controller-2026-10-16.md). Both exact hashes
are fixed source entries in the technical config. It runs on the Mila login host
using the existing project controller Python, from the isolated published source.
It checks actual nativecluv0.1.5 library hashes, clean HEAD/origin main, owner
receipt and complete source/config identity before native submission. It creates
the hash-bound claim exclusively, records intent/actual native expansion/outcome,
and never retries an ambiguous result. A returned job ID is not evidence of
RUNNING; root must inspect the scheduler.

External plan fields additionally bind `claim_id`, `controller_cache`,
`preview_receipt`, `submission_receipt`, `logs` and `native_cluv_source_sha256`;
the last mapping is exactly the six audited library pins in the controller.
The controller cache and evidence paths are new under `inputs/`; logs are exactly
`<scratch>/logs/primary-conversion-verified-20261016` and must be new or empty.
Storage reservation is at least1GiB for new source/cache/logs plus conversion and
reporting output, within the existing20GiB whole-project cap.

Root's future invocation, from the exact source cwd, is:

```text
<scratch>/source/.venv/bin/python -B code/ops/submit_primary_conversion.py --plan <scratch>/runs/primary-conversion-verified-20261016/inputs/plan.json --plan-sha256 <owner-admitted-plan-SHA256>
```

The plan hash is finalized only after the owner's approval receipt is retained;
this changes admission metadata, not the published scientific/source settings.
No owner receipt or plan authorizing submission is created by this request.


After owner approval and native-cluv/source readiness, the controller submits only the reviewed
`code/ops/graph_data_conversion_job.sh --plan <absolute-plan>
--plan-sha256 <frozen-SHA256>`. Recheck native cluv0.1.5 and the actual scheduler
before root materializes the launch. The reviewed native scheduler expansion is:

```text
sbatch --parsable --account=mila --qos=normal --partition=main-cpu --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:10:00 --no-requeue --job-name=cluv-rc-primary-conversion-v1 --output=<new-log-root>/mila_%j/slurm-%j.out --chdir=<new-source> --export=ALL code/ops/graph_data_conversion_job.sh --plan <absolute-plan> --plan-sha256 <frozen-SHA256>
```

This command is prospective, with unresolved source/plan hashes and no current
submission authority. Automatic source environment sync/install is excluded.
An exclusive root submission claim and job-ID reconciliation must prevent
duplicates. No existing hosting allocation or other project's jobs are used.

The stdlib Python runner starts monotonic time before its other imports. Source,
config and receipt checks may precede reporting. The supervisor binds actual
Slurm job ID, one-CPU affinity and inherited Linux physical memory cgroup <=2 GiB.
It performs exactly one bounded read-only `scontrol show job -o <actualJobID>`;
only unique JobId/Account/QOS/Partition/NumCPUs/NumNodes/NumTasks/TimeLimit/Requeue
and requested/allocated TRES fields are retained, requiring this exact allocation.
TimeLimit must be00:10:00 and Requeue0; both TRES mappings must contain only
cpu1/mem2GiB/node1 and no GPU. Recognized memory units normalize2G/2048M/2097152K
or2147483648 bytes. Missing capability or fields fail
before SDK/data access. Optional Slurm account/QOS/partition environment fields
must agree; their absence is allowed. The scheduler receipt SHA, actual job,
parent PID/start identity, plan SHA, exact worker argv and original monotonic
start enter the supervisor-owned worker admission.
Root's controller also verifies the exact submitted resources and exclusive
single-submission claim against actual scheduler outcomes. The runner independently
enforces its physical600-second/one-CPU/2 GiB process bounds and imports no GPU framework.

The worker deadline is start+590 seconds, with outer physical supervision through
start+600. Both remain import-inclusive. Python-spawned SDK children are journaled
before SDK import. CPU/numerical/Go thread limits are one. A 64 MiB per-file
RLIMIT_FSIZE complements physical memory cgroups and aggregate output monitoring;
RLIMIT_AS is not used. Worker/SDK processes remain supervised and waited upon;
cleanup signals only this attempt's recorded PID/start identities, with finite
exit reconciliation for retained owned identities. The full spawn journal is
preserved separately, including SDK processes that exited before a refresh.
No claim that every historical journal identity was retained is implied.
No detached continuation remains.

Offline `ExperimentReporter` must successfully initialize W&B0.30.0 before the
first primary CSV stat/read/hash/parse. Metadata consists only of explicit compact
scalars, public relative code paths and hash-only identities. No dataset, labels,
weights, environment mapping or automatic file upload is supplied to W&B. After
entry the worker calls the actual `convert_primary_csv(raw_path, output_dir,
expected_raw_sha256=..., expected_raw_bytes=..., config_path=...,
converter_revision=..., guard=...)`, once. Fixture mode is never enabled.

The converter checks exactly bounded raw bytes and at most 500,000 rows, 3,072
timestamps and N=100, according to the frozen protocol. It produces three split
JSON files and the final manifest; each split is bounded to64 MiB and the
manifest to2 MiB. The runner accepts exactly those four final files, hashes each,
checks the published manifest with producer `read_manifest`, and records counts
and identities. That producer call is self-consistency, not independent reduction.

Within the same original590-second work/600-second physical budget, it then
charges exactly one independent verifier attempt before entry and calls the
actual `verify_primary_conversion(raw_path, manifest_path, *,
expected_raw_sha256, expected_raw_bytes, expected_manifest_sha256,
expected_converter_sha256, expected_converter_revision, config_path, guard)`.
Fixture mode/spec are never enabled. The second parser independently rebuilds
every raw snapshot, all split bytes, all semantic hashes and the entire canonical
manifest, without producer imports/helpers. It writes no data; the worker
atomically retains its primitive receipt as `worker/independent-verification.json`.
Any mismatch is terminal, with conversion artifacts/failure counts retained and
no retry. This is independent data verification outside model fitting; reading
the test split here grants no training/selection access.

The caller and supervisor require the production verification kind, exact
raw/config/converter/verifier/revision/manifest identities, primary-eligible and
independent flags, all3072 snapshots,3splits,1complete-manifest comparison and
7stable input files. The closed25-key receipt also requires the exact primary
spec, all four strictly typed raw counts, successful marker count0, at least6144
unique offdiagonal occurrences, rows=unique+duplicates and equality with producer
counts. Every split record has exactly filename/interval/size/fileSHA/semanticSHA
with fixed boundaries and valid bounded types. Booleans cannot substitute for counts. After owned cleanup, the terminal file inventory must still agree
with the accepted receipt SHA, manifest SHA and every split size/SHA. It never
constructs numerical snapshots/models or opens train/test store APIs.
Output monitoring terminates an attempt above 260 files or 448 MiB;
conversion files must separately fit192 MiB. Bounds are stopping rules, not a
promise that partial output cannot briefly exceed an aggregate ceiling between
monitor checks. The outer gate reserves this maximum plus source/log needs.
Converted files reject all links. SDK metadata census alone permits the known
W&B0.30 `worker/reporting/wandb/offline-run-<timestamp>-<this-run-id>/logs/debug-core.log`
symlink, tied to this exact offline reporter receipt. It counts only the link's
lstat size, records `skipped_sdk_symlink=true`, and never follows or reads its
target. Other symlinks and special files fail. Live census tolerates at most16
vanishing files per scan only under SDK reporting/support paths; terminal census
after owned cleanup requires stable entries. Hashing checks the shared deadline
between chunks. This exception changes neither converted data nor the accepted
reporter source/privacy settings.

## Evidence and acceptance

Retain exact plan/config bytes, actual argv/source/job/admission lineage,
scheduler and physical-resource receipts, worker status/RSS/attempt count,
per-file conversion hashes/counts, offline W&B receipt/history/summary, process
journal, supervisor cleanup identities/signals/exit reconciliation and final
evidence inventory. Inventory excludes its own bytes to avoid a hash cycle.
Status/lifecycle JSON updates are atomic; terminal outputs are retained rather
than overwritten by another attempt. An early refused launch may have only its
durable cluv/Slurm error log. CSV lexical failure preserves available partial
files and its failure record; no silent repair or retry follows.

Success requires a completed worker, exactly one conversion attempt, the exact
primary manifest/file closure and a completed offline reporting receipt with
finish returned and all three metric steps (conversion started0/completed1,
independent verification completed2). Exactly one conversion and one independent
verification attempt must complete; a missing, fixture-kind, mismatched or
changed verification receipt is incomplete. Exit0 alone is insufficient. Root must
reconcile actual independent-verifier results, final hashes, retained offline
transaction and scheduler outcome before accepting engineering evidence. Offline durability
does not establish dashboard delivery. Training, task competence, periodic orbit
closure, Floquet contraction and harmful-collapse claims remain unverified.

Preparation checks use synthetic/fake process fixtures and the actual reporter
constructor without entering its SDK lifecycle. Native operation, scheduler
capability and primary-body compatibility remain unverified.
