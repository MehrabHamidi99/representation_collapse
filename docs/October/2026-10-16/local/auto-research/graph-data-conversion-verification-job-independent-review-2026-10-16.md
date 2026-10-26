# Independent review of the conversion-plus-verification job

2026-10-16. **Accepted as preparation for a fresh, source-bound allocation
decision.** This review grants no allocation and supplies no actual primary-data
result. The technically enabled config still requires external owner approval,
an exclusive one-submission claim and all exact plan/readiness inputs. None is
created by this review. Earlier conversion-only reviews describe older bytes.

## Reviewed final release

- `code/ops/graph_data_conversion_job.py`: SHA256
  `a61e153044a81e8aa85435603e7f03a01fb05c37dae85d99b25757aaa514772a`.
- `configs/graph/primary-data-conversion-v1.json`: SHA256
  `8b45c356879cac2aa4c19ef1677c3412076eb6f5325f8b288032c00d65f5b2ed`.
- [Launch request](graph-data-conversion-launch-request-2026-10-16.md): SHA256
  `2fde65c182d2c67882a5cebeb3551d91823ae1be664b88014ec5e8baaae47ce0`.
- Unchanged shell: SHA256
  `4698781a78186883c877e8879d23c723eb14d05387da3ee0f3db8e23d06089b4`.
- Independent tests, `tests/test_graph_data_job_verification_independent.py`:
  SHA256 `a6ba4288b2d8816df9aa13e2b2c99f0066cba45093a6dab25f2750364aa95fcc`.

Acceptance criteria were frozen before the new integration handoff in
`runs/review/graph-data-job-verification-20261016/independent/acceptance-criteria.json`,
SHA256 `abf63791f881899db640de9a5f07a696d2f10725e9692470100f37e4a1d5c4ac`.
The reviewer authored the separately reviewed verifier, but did not author this
job integration. New integration fixtures were independently constructed rather
than importing the job author's test helper.

## Resolved finding and final checks

The initially handed-off runner `6d5370ef...` validated fixed comparison totals
but ignored the receipt's raw row/deduplication/marker/edge counts and full spec.
It also did not close each split descriptor schema. The author repaired this
before acceptance: the production receipt now has exactly 25 keys, an exact
primary spec, all four strictly typed raw counts, zero successful markers,
row conservation, feasible count bounds and equality with producer counts.
Each split requires its exact five fields, fixed basename/interval, bounded
integer size and lowercase file/semantic hashes. Worker, supervisor and final
reconciliation all pass the producer counts. Independent cases remove fields,
change types/counts, substitute a different interval/hash or add an unknown key;
each fails before a verification success receipt is retained.

The final command
`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p 'test_graph_data_job*.py' -v`
passed **28 fixtures in 0.360 seconds**: 22 existing/author integration fixtures
and six new independent fixtures. All final runner/config/request/test hashes
were checked immediately before and after execution. These are mocked local
control-flow tests, not additional experiments or scheduler submissions.
The retained final log and test-run record are in the independent review folder.

The independent worker fixture replaces both parsers and SDK lifecycle with
explicit fakes. Its synthetic input path does not exist; only the guarded
size-stat boundary is mocked. It verifies actual worker ordering, durably charged
attempts, the exact production verifier keyword set and caller identities.
No fixture/spec override reaches the production call. It independently exercises
verifier exceptions and SDK finish failure after both fake parsers return:
the status remains failed, no retry occurs, and retained intermediate receipts
do not become successful terminal evidence. The fixed finished stage is 2,
after conversion start 0 and conversion completion 1.

Post-worker tests alter the retained verification receipt, manifest or test split.
Final inventory reconciliation refuses each mutation. The supervisor requires
child exit zero, completed status with exactly one conversion/verification
attempt, the receipt's current hash and production identities/counters, and a
completed offline reporter with finish returned and last step 2. After owned
cleanup, final inventory hashes must still agree with the verified receipt,
manifest and all three split sizes/hashes. A partial or missing artifact cannot
be promoted by exit status alone.

## Source, reporting and lifetime boundaries

All eight configuration source pins match actual bytes, including the fixed
submission controller/protocol pins added after the functional repair. The
existing exact source-closure loop includes both new sources. The policy mask
still permits only the same four converter/verifier protocol pins and technical
execution flag; raw identity, resources, limits, fixed helper/reporter and
controller/protocol hashes remain covered by policy. The unchanged helper and
reporter hashes are checked; actual compact reporter metadata passes the real
constructor without entering an SDK lifecycle.

Both parser imports/calls and the first primary stat/read/hash follow successful
reporter entry. A verifier attempt is persisted before invocation and remains
consumed on failure. The worker uses the supervisor's original monotonic start
plus 590 seconds for both phases. Outer supervision remains import-inclusive
at start plus 600 seconds, one CPU, physical 2 GiB and zero GPUs. No stage resets
the deadline or acquires another allocation. Existing STOP, output ceilings,
single-attempt/no-requeue policy and owned-process cleanup remain in force.

The separate submission controller is source-pinned here; its native-cluv
implementation receives its own independent review. This review does not
replace that review or verify a live scheduler, SSH session, environment,
storage reservation, staged checkout or exclusive submission claim.

## Remaining acceptance

Publishing/staging this exact release, source-bound implementation/reporting
readiness receipts, actual cluv/scheduler/environment/storage checks and fresh
owner approval remain coordinator actions. No real raw file was read, no parser
was invoked on target data, no SDK was entered, and no research worker process,
model, provider call or remote job was launched by these fixtures. Mocked process/SDK calls
are explicitly distinguished from real execution.

Actual primary conversion and independent reduction remain unrun. A lexical or
semantic incompatibility must terminate and be retained without broadening the
frozen protocol. Successful future data integrity would still establish neither
trained competence nor a representation-collapse, causal-memory or Floquet result.
