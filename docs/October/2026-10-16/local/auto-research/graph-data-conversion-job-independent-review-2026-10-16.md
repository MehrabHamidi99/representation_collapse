# Independent review: prospective primary conversion job

2026-10-16. **Accepted as a preparation packet only.** The corrected runner,
shell, disabled configuration, fixtures and allocation request below have no
remaining blocking implementation finding in this bounded review. This is not
compute approval, verified native execution, independent primary-data reduction
or scientific acceptance. The converter itself received a separate independent
review; this review covers another agent's physical/reporting/admission wrapper.

Reviewed immutable bytes:

- `code/ops/graph_data_conversion_job.py`: SHA256
  `40f8ea327e0a3fd7da9a9781f543a29fbe3d8f070ef3c51cbeb0410b12843022`.
- `code/ops/graph_data_conversion_job.sh`: SHA256
  `4698781a78186883c877e8879d23c723eb14d05387da3ee0f3db8e23d06089b4`.
- `configs/graph/primary-data-conversion-v1.json`: SHA256
  `1e021c779b00ea0346ef49d35f09d040c3b2c5c718f87239758e19effb77520d`.
- `tests/test_graph_data_job.py`: SHA256
  `aede211e575da5e0b13a825bdbae69e1af83e6937b560e6ddcbf53bc3e89a60b`.
- [Allocation request](graph-data-conversion-launch-request-2026-10-16.md): SHA256
  `235aed8b6217d2e813b9cce562a8ff6a00dba31fbd9adc1c5bbd115556921591`.

Converter source remains frozen at
`498e60f7be048411c253e4b8cb63e4b907973f969c397bfdfacde949389f057e`.
Accepted reporter and ownership-helper bytes remain respectively
`443c9e89d2eba954ee60cc35e015fe61196fbec2f50331067540b4a8259e8184`
and `8e1f388d4f5ce72e0bfd9a4b5a4b1f8c98f66f2512841d96875822bfd9bffe09`.
No reviewer source edits were made to these artifacts.

## Evidence and resolved findings

The initial inspected runner `d1bab6c7a47888a9251667d923c20a31f012762887eba7be731f06c7b609b6bf`
accepted synthetic scheduler output with 30 minutes, requeue enabled, 4 GiB and
one GPU because it checked only seven scheduler fields. The fake-query
reproduction is retained under
`runs/review/graph-data-job-20261016/independent/pre-freeze-scheduler-finding.json`.
This was a preparation defect, not an actual erroneous submission.

The author corrected actual `scontrol show job -o <jobID>` verification to
require the exact job/account/QOS/partition, one CPU/task/node, ten minutes,
Requeue=0 and both requested and allocated TRES maps containing only cpu=1,
node=1, memory=2 GiB. Recognized memory spellings normalize explicitly; missing,
duplicate, mismatched or GPU TRES fail. Independent fixtures exercise the actual
previously retained Mila11261627 scheduler stdout **without a scheduler call**,
plus adversarial resource/time/requeue/duplicate cases.

Admission receipts originally underbound source/run identity and compared
nested resources with Python equality, which can conflate booleans and integers.
The corrected code binds owner approval and the external exclusive submission
claim to source commit/files, output/STOP, limits, submission count and retries;
owner approval also binds config, allocation and resources. Canonical nested
comparison distinguishes bool from int. A verified root-exclusive claim must
carry exactly one typed claimed submission. The supervisor argv contains a
literal final `{plan_sha256}` placeholder; runtime checks its substitution by
the externally supplied plan hash against the **actual** argument vector.
Independent complete-plan fixtures validate that path and reject altered
source bytes, owner bindings, counts, limits, storage, existing output and STOP.

The initial full-output census would have refused an SDK-created diagnostic
symlink already observed in the project's real offline W&B lifecycle. The
corrected exception is limited to the known `debug-core.log` path inside this
exact offline run, checked against the reporter receipt. Census counts lstat
bytes and records a skipped symlink; it never follows/hashes its target.
Arbitrary links, conversion links, directories linked elsewhere and special
files remain refused. Independent fixtures use an inaccessible nonexistent
target to verify this behavior and reject a different reporter run identity.
The exception does not change reporter source or privacy settings. Live census
tolerates at most 16 disappearing SDK/support files per scan; final census after
cleanup requires stable entries. This policy is not a general filesystem sandbox.

## Verified preparation behavior

The module starts its monotonic clock before other imports and has a stdlib-only
top-level surface. The shared original start is carried in the hash-bound worker
admission. Worker work/reporting remains covered by start+590 seconds, including
the reporter context exit; start+600 seconds covers parent supervision and owned
cleanup. Guards run during converter records/read chunks and output hashing.
Worker SIGALRM remains armed until its final block. Parent ownership checks use
the supervisor PID/start identity, a fresh worker process session and the frozen
spawn journal installed before SDK import. Physical scheduler/cgroup behavior
and actual finish latency still require the future native run; local mocks do
not prove those runtime properties or make blocked filesystem I/O impossible.

Runtime requires Linux/x86_64/CPython3.10.19, installed W&B0.30.0, one-CPU
affinity and a real inherited memory cgroup cap at most 2 GiB. Public metadata
includes actual `sys.version`, implementation, affinity and cgroup path/cap.
One-thread CPU/Go environment limits and per-file RLIMIT_FSIZE complement the
cgroup; virtual-address limits do not stand in for physical memory enforcement.
The exact scheduled request remains one CPU, 2 GiB, zero GPUs and ten minutes.

The actual dictionaries returned by `reporter_metadata` were passed to the
actual `ExperimentReporter` constructor with SDK import forbidden. Constructor
validation passed, all identities are hashes, and no reporter directory or SDK
session was created. Fake worker fixtures verify reporter entry before the
first conversion/raw-file action, one converter attempt, the real converter API
argument names, original revision binding, steps 0/1 and reporter exit before
completed status. Author failure fixtures independently replayed here verify
SDK-entry failure blocks conversion and preserves the zero-attempt status.

The production call never enables fixture mode. Raw size/hash is the frozen
6,907,326-byte candidate; the converter's existing strict policy bounds rows to
500,000, timestamps to 3,072 and nodes to 100. Exactly four conversion files must
fit 192 MiB; total monitored output is at most 260 files/448 MiB, with a whole
project reservation below 20 GiB required before entry. These aggregate bounds
are monitored stopping rules, with the documented interval between checks.
No model, optimizer, generation, Snapshot/store opening or NumPy/Torch import
belongs to this job.

Fake supervisor fixtures require completed worker/reporting state, both metric
steps and `finish_returned=True`, then successful retained-owned cleanup before
overall success. Failed reporting finish or unverified owned cleanup yields an
incomplete attempt. Both worker and SDK retained identities are included in the
fixture, with the original 600-second cleanup deadline. The reused helper
reconciles retained known PID/start identities; already-exited historical journal
entries may not have been retained. The full spawn journal is separate evidence.
This review does not inflate that scope into independent exit proof for every
historical journal record or assert that an actual worker is running.

## Checks and outstanding execution gates

Sixteen new independent fixtures passed in 0.220 seconds:

```text
.venv/bin/python -B runs/review/graph-data-job-20261016/independent/test_job_review.py -v
```

All sixteen author fixtures were independently rerun and passed in 0.057 seconds:

```text
PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_graph_data_job.py -v
```

Shell syntax and Python3.10 AST compatibility checks passed. Earlier intermediate
independent suites passed 13 then 15 cases; they are development evidence, not
additional independent runs of the future conversion. This review executed no
SDK lifecycle, real subprocess worker, scheduler/network call, primary payload
read, model/numerical campaign or submission. All processes/metadata for runtime
fixtures were explicitly fake; only small temporary fixture files were written.

The checked-in config is deliberately disabled and has null converter/protocol
pins. Root must bind the final source pins and enabled config in a published,
verified immutable source, create config-bound review/reporting/environment/
storage/owner receipts and a real exclusive cluv submission claim, verify the
current scheduler/tooling, and obtain a fresh owner envelope. An ordinary JSON
claim is not independent proof of exclusive controller creation or user approval;
the coordinator owns that actual transaction and ambiguous-outcome reconciliation.
These preparations do not confer a recurring allocation.

After any future run, root must reconcile source/admission/job outcome, physical
resource receipts, cleanup, byte hashes and the native offline W&B transaction.
The job's call to the producer's `read_manifest` is a self-consistency check,
**not independent primary conversion reduction**. A separately implemented,
frozen reduction of original CSV and converted split semantics remains an
unresolved gate. No trained benchmark competence, periodic orbit, Floquet,
harmful-collapse or cross-modal scientific claim is accepted by this review.
