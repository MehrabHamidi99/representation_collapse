# Single-submission primary-conversion controller — October 16

**Preparation only. No fresh owner allocation grant or submission accompanies
this implementation.** The controller is
`code/ops/submit_primary_conversion.py`; it neither runs conversion itself nor
changes the admitted science, resource envelope, worker, configuration or
preflight. Root owns publication, staging, independent acceptance, the owner's
allocation question and actual launch.

The prior source anchor is
`runs/ops/native-graph-reporting-fix-20261015/submit_program_v2.py` and its reviewed
native-cluv0.1.5 pattern. Audited library identities come from
`runs/review/native-graph-cluv-ops-20261015/receipt.json`: cache, native submit,
sync, config, remote and sbatch-argument source. The controller pins all six
exact SHA256 values in `LIBRARY_PINS`; they must match both distribution-located
files and actual imported modules before native API use. Native distribution
version must be exactly0.1.5. This task inspected source and used mocks; it made
no actual cluv/API, network, scheduler, SDK, worker or numerical call.

## Entrypoint and exact plan

Future command, only after the independently accepted source and fresh owner
approval exist:

```text
<controller-python> -B code/ops/submit_primary_conversion.py --plan <absolute-inputs/plan.json> --plan-sha256 <external-plan-SHA256>
```

Import is stdlib-only. Normal Python is mandatory: reject interpreter optimize
flags and nonempty PYTHONOPTIMIZE. The actual hostname must end with
`.server.mila.quebec`, matching inspected native current-cluster detection.
Ambient CLUV_CLUSTER/CC_CLUSTER may be absent or mila. The controller must run
from the isolated project-scratch source root, with its own exact file there.
Git HEAD and the locally recorded published `refs/remotes/origin/main` must
equal the admitted commit, and git status must be clean. This local published-ref
check is complemented by root's fresh GitHub clone/staging verification; no
fetch/push/sync/install is performed by the controller.
Metadata reads are bounded, no-follow, single-link regular-file reads checked
against descriptor/path identity, size and timestamps before/after. Pinned
receipt hash verification and JSON decoding use the same byte snapshot; no
hash-old/parse-new race is accepted.

The worker's flat external plan retains all its existing fields. Additional
controller fields are `controller_cache`, `preview_receipt`,
`submission_receipt`, `logs`, `claim_id`, and `native_cluv_source_sha256`.
The last is an exact mapping of the six `cluv/...py` relative members to the
audited hashes in the controller. Canonical paths must be beneath
`/network/scratch/m/mehrab.hamidi/representation-collapse`, without symlink
components, shell expansion/metacharacters or ownership mismatches.

With run root `<run>` and prepared parent directories, exact paths are:

- Plan: `<run>/inputs/plan.json`.
- Output: `<run>/attempt001`; STOP: `<run>/STOP`, both absent.
- Owner receipt: `<run>/inputs/owner-allocation-approval.json`.
- New cache: `<run>/inputs/controller-cache`.
- New preview: `<run>/inputs/native-submission-preview.json`.
- New intent: `<run>/inputs/native-submission-intent.json`.
- New outcome: `<run>/inputs/native-submission-receipt.json`.
- New exclusive claim: `<run>/inputs/native-submission-claim.json`.

Root's concrete source/run/log layout remains the launch request's unique
`primary-conversion-verified-20261016` paths. Source and run trees must be
separate. Logs are derived as `<project-scratch>/logs/<run-root-name>` and must
be absent or an empty owned directory, excluding source/input overlap. Source
closure is derived from config: worker Python/shell, config,
launch request, package init, reconstruction config and every config `sources`
entry. Root must add this controller and this protocol as exact config source
records, freeze the corresponding worker policy/config pins, and publish them
before admission. The plan's source-file mapping must equal that entire closure,
and every byte hash/config helper pin must match. No hardcoded source commit or
config self-hash creates a circular dependency.

The technical config flag alone grants no allocation. The owner receipt must
have `kind=primary_data_conversion_owner_allocation_approval`,
`status=verified`, `approved_by=owner`, `approved=true`. It must exactly bind
source commit, complete source-file mapping, config SHA, output/STOP paths,
limits, allocation, resources, maximum1 submission and retries0. Nested values
compare by JSON encoding, so True/False cannot stand in for1/0. Root must create
this receipt only after the owner's actual decision. The controller also checks
the pinned implementation/reporting/environment/storage receipts. The storage
gate reserves at least1 GiB for new source,448 MiB output/evidence and controller
cache/log overhead, within the whole-project20 GiB ceiling.

## Native API boundary and resource guard

The exact allocation is Mila/accountmila/QOSnormal/main-cpu, nodes1/tasks1/CPU1,
memory2G, walltime00:10:00, GPU0, no-requeue; no array, race, chunks or retry.
The controller clears only its in-memory effective native global env/sbatch
options and replaces the frozen Mila cluster configuration with one allocation
using actual `dataclasses.replace`. It keeps only Mila in that process's cluster
mapping. It writes no global/user config and performs no environment or dataset
synchronization. New XDG_CACHE_HOME is set to the owned input cache before
importing cluv; no HOME default cache is used.

The actual native call is `await cluv.cli.submit.submit("mila", Path(<shell>),
[], ["--plan", <path>, "--plan-sha256", <hash>], autocommit=False,
chunking=None, vram=None, _skip_sync=True, sync_datasets=False, parsable=True)`.
The reviewed source hash checks and published checkout precede `_skip_sync`.
This is one invocation, with no retry branch.

Wrap native `get_submissions` with a one-entry cap. Require exactly one row:
cluster mila, local remote=None, n_chunks=None, exact full normalized resource
mapping, admitted git commit, exact relative worker shell and exact program argv.
Only native additions job-name=`cluv-rc-primary-conversion-v1`,
output=`<logs>/mila_%j/slurm-%j.out`, chdir=`<source>`, export=ALL are accepted.
Unknown resources, GPU/array/requeue expansion, extra rows and bool-as-integer
resource substitutions fail before a scheduler command is returned.

Validate the actual native shell command too: exact `bash --login -c` shape;
only `GIT_COMMIT=<commit> CLUV_CLUSTER=mila; cd <source> && sbatch --parsable`
prefix; exact nonduplicate sbatch flags, shell path and argv. Matching row
metadata alone is insufficient. Prepend a bounded shell guard rejecting any
SBATCH_* variable, present SLURM_CLUSTERS or nonempty PYTHONOPTIMIZE after the
login profile runs. The controller rejects those scheduler variables in its
own environment too. A reviewed preview records the guarded actual command,
resource expansion, source and argv. No unsupported cluv dry-run flag is used.

## Exclusive claim, ambiguity and retained evidence

The runner's `submission_claim` remains `{path,sha256}`. Its path is absent
before controller entry. Root precomputes the SHA256 of `encoded(claim_payload(plan))`:
ASCII JSON, sorted keys, indent2, one trailing newline. The exact payload has
kind `root_exclusive_submission_claim`, status verified, exclusive true,
claim_id, claimed_submissions1, source_commit, source_files, output, stop_path,
limits, max_submissions1 and retries0. It intentionally omits plan SHA, avoiding
the cycle plan→claim→plan. The plan SHA stays in intent/preview/outcome.

After all gates, the controller writes/fsyncs exclusive intent, then creates
the claim with O_EXCL/O_NOFOLLOW and fsyncs file/parent immediately before its
single native submit invocation. An existing claim refuses the attempt. The
plan byte hash and exact context, current pinned owner approval/bindings,
absent STOP/output and precomputed claim byte hash are checked again immediately
before intent/claim publication. The
controller restores its temporary native function wrapper after return/error.
Claim, intent and preview are never removed to permit a retry.

Persist a fresh outcome on success, native None, malformed result or exception.
Only error class names are retained, not exception text or credentials. A valid
returned job ID is labelled `native_returned_job_requires_scheduler_reconciliation`.
The inspected native API may return a singleton PENDING job; that return proves
neither RUNNING nor successful conversion. Native None/exception/uncertain
response is `ambiguous_requires_reconciliation`, with no fabricated job ID and
no automatic resubmission. A crash may leave only intent/claim/preview/native
logs; root must reconcile exact job name/source/log path/owned queue state before
deciding what happened. Existing files or incomplete receipts are never silently
overwritten. No cancellation of other jobs is introduced.

The preparation suite exercises real controller admission logic against fake
files/Git/hostname, exact command/resource guard mutations, canonical exclusive
claim bytes, mocked singleton-PENDING return and an ambiguous exception. It uses
no actual native cluv imports/API, remote connection, SDK, Slurm or conversion.
Root still must complete independent review, freeze/publication/staging, receipt
creation after the owner decision, actual native source checks and scheduler
reconciliation. Worker reporting/conversion/independent-data-verification and
scientific acceptance remain their own gates.
