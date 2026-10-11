# 2026-10-11 — project bootstrap

## Scope and preserved evidence

Set up recurrent representation dimension-collapse research around exact discrete
Floquet and finite-time analysis, starting from temporal graphs. Adapted the
Skill Inference project's canonical skills, month/day notebook, tracker,
scratch-first storage and evidence DAG. Its spending approvals, model-provider
configuration and GPU reservations were not imported.

Initial local tree contained manuscript/figure snapshots under `floquet_analysis/`,
`ACM___TGRAB/`, and `ACM___TGRAB_KDD_2026/`, with no git metadata or experiment
implementation. These sources were preserved in place. The appendix identifies
public T-GRAB code/data; source revision, checkpoint provenance, raw trajectories
and dimension-analysis implementation still require recovery. Scientific
milestones and controls are in `doc/auto/`.

## Implementation and delegated work

The coordinator integrated root guidance, package/lockfile, local NumPy reference
measurements and calibration, artifact checks, discovery links and the notebook.
Three bounded agents owned disjoint scientific documentation, cluv compute,
and research workflow scopes. A separate forward review checked numerical
semantics and compute readiness; findings and corrections are appended below.

Canonical skills: `floquet-analysis`, `experiment-loop`, `project-ops`, and
`model-routing`. Discovery symlinks under `.agents/skills/` point back to these
files. `configs/research-plan.json` defines seven acyclic dependency nodes;
remote submissions are disabled and resource caps unset until a concrete
approval. This is an active-session workflow, not an unattended controller.

## Local evidence

Environment: CPython 3.11.13, NumPy 2.4.6, cluster-uv 0.1.5 in the `ops` group;
full dependency provenance is `uv.lock`. Commands:

```bash
uv sync --locked --group ops
uv run --locked python -m unittest discover -s tests -v
uv run --locked python -m representation_collapse.calibrate --output runs/calibration/setup-2026-10-11-v2
uv run --locked --group ops python code/ops/preflight.py
```

The calibration passed ten checks. Isotropic four-feature covariance has PR 4;
rank-one covariance has PR 1; a contracted rotation has four-step multiplier
modulus `0.8^4 = 0.4096`; a nonnormal matrix has spectral radius 0.8 with
singular gain greater than one. A stable periodically forced response closes
its four-step orbit while its multiplier modulus is `0.5^4 = 0.0625`.
These are synthetic numerical/analytic controls, not findings about trained
graph models. Zero-variance normalized dimensions are undefined scientifically;
the API encodes zero sentinels with `normalized_dimension_defined=false`.

Artifacts include source/lock hashes, configuration, raw arrays, metrics,
verification, atomic status and SHA-256 manifest. New attempts refuse to
overwrite existing outputs. The final verification count is appended below.

## Compute preparation and authorization

Read-only probes verified Mila and Tamia via Mila; direct laptop Tamia access
timed out. Tamia account/partition for the proposed CPU smoke:
`aip-ioannism`, QOS `normal`, `cpubase_bycore_b1`. Request: one node, one task,
one CPU, 2 GB, ten minutes, zero GPUs, maximum one submission, no retry.
Additional Alliance profiles are examples with unverified project entitlements.

cluv 0.1.5 supports `submit first` and multiple `sbatch_args` configurations;
both can create several candidates. Race scope must match the approved actual
connected set. No race is part of the CPU smoke. New environment variables in
this release's generated sbatch shell are not all exported; the wrapper receives
the immutable source SHA as an explicit argument and exports job-local cache
settings itself. Queued jobs use a detached source snapshot and unique outputs.

The native cluv config is in `pyproject.toml`; its default preflight validates
without SSH or remote side effects. Cluster staging, source access and exact
submission commands are specified in `doc/auto/compute-plan.md`. No remote
allocation, environment preparation, or project-file copy was performed during
bootstrap. Approval is requested only for the concrete bounded next step.

## Scientific decisions and next work

Use exact discrete updates rather than assuming the draft's `F-s` vector field
embeds the recurrence. Verify orbit/Jacobian periodicity before interpreting
products as Floquet maps. Treat covariance dimension, perturbation contraction,
task information and causal memory as separate estimands. Periodic output can
persist under contracting dynamics; repeated-period covariance has a sampling
rank ceiling. The inherited AB/AABB task notation requires resolution.

Next: pin T-GRAB source/data; recover checkpoints and producing analysis;
validate complete graph hidden/cell state and output chronology on a tiny graph;
only then prepare a reviewed GPU reproduction pilot. Cross-modal matched
tasks and independent natural-data pilots follow graph evidence review.

## Integration validation

Final local checks passed: nine unit/integration tests, ten synthetic calibration
checks, all four skill validators, all shell syntax checks, repository-relative
document links, seven-node acyclic plan, and source-audit artifact hashes.
Final pre-commit calibration is `runs/calibration/setup-2026-10-11-v4/`; earlier
attempts are retained. The saved arrays reproduce reported dimensions and all
manifest hashes verify; duplicate-output attempts are rejected.

Independent review found and corrected covariance overflow, ambiguous degenerate
rank reporting, nonexistent source SHA acceptance, alternative-config preview
mismatch, and incomplete offline Python staging. Preparation now explicitly
installs the pinned Python into scratch and checks a fresh offline environment.

Remote uv is 0.9.18; the Mac's global uv is 0.8.13. cluv can automatically
align remote uv to the client version. The ops group therefore pins uv 0.9.18,
confirmed through `uv run --locked --group ops uv --version`, avoiding an
unintended downgrade of shared cluster tooling. Remote GitHub SSH access to
this initially empty repository was verified on both Mila and Tamia. Local
GitHub SSH failed; HTTPS listing succeeded. Git retains the requested SSH
origin and uses a separate HTTPS push URL for local publication.

No cluster job was submitted. A committed-source Tamia preview and publication
verification follow this integration record.

## Publication and concrete next allocation

Bootstrap source commit `52025761004f04723daca3a543a2ff39715606ea` was pushed
to GitHub main through the configured HTTPS push URL; the requested SSH
origin remains available for cluster cloning. The working tree was clean.
Original snapshots are tracked unchanged; their inherited whitespace was
preserved, and new infrastructure passes the whitespace check.

Committed-source preflight passed for Tamia. Negative checks reject nonexistent
source commits, `first` racing, and alternative-config cluster previews.
Review artifact: `runs/calibration/tamia-cpu-smoke-preview.json` (ignored local
runtime artifact). Exact approved-next-step candidate:

```bash
uv run --group ops cluv submit tamia code/ops/calibration_job.sh --no-sync-datasets -- --source-commit 52025761004f04723daca3a543a2ff39715606ea --run-id cpu-smoke-001 --cluster tamia
```

Execute from the scratch controller on Mila after bounded source/environment
staging on Mila and Tamia, followed by explicit Tamia-only cluv sync. Request:
account `aip-ioannism`, QOS `normal`, partition `cpubase_bycore_b1`, one CPU,
2 GB, ten minutes, zero GPUs, one submission, no race/retry. The approval
question was presented to the owner; no approval has been received at this
handoff and no cluster job or staging command has been executed.

Successful submission is followed by scheduler reconciliation, explicit cluv
result retrieval, manifest verification and numerical check acceptance. A
failed infrastructure smoke returns evidence for a smaller repair; it does
not trigger an unapproved resubmission. The later GPU pilot needs its own
scientific inputs and approved envelope.

## 06:59 UTC — Periodicity research plan saved

Purpose: persist the theoretical and experimental plan developed with the owner
in the documentation directory.

Added `doc/auto/periodicity-research-plan.md` and linked it from `doc/README.md`
and `doc/TRACKER.md`. The plan records fixed-weight inference as the primary
setting, linked continuous/categorical memory studies, matched periodic and
nonperiodic driver controls, the recurrent architecture registry, separate
state-carried and persistent-access depth loops, numerical/scientific acceptance,
confirmation, native validation, and bounded cluv compute stages.

The plan proposes 351 controlled pilot checkpoints before tuning, confirmation,
native models, or targeted ablations. This is planned coverage, not an approved
allocation or a completed experiment. Legacy graph recovery and independent
synthetic work may proceed in parallel under the new research path; the current
bootstrap execution DAG and operational skills have not yet been revised.

Validation for this documentation change: whitespace checks and repository-local
Markdown link resolution. No training, cluster submission, or source-snapshot
modification was performed. Next: translate the documented plan into frozen
task/adapter configurations, acceptance fixtures, and bounded execution nodes.

## 07:28 UTC — Literature library and adversarial novelty audit completed

Purpose: assess the project's novelty against close existing work, assemble a
scoped bibliography, retrieve available public papers, and identify defensible
open questions before committing to the broad experimental screen.

Added `doc/literature/novelty-review.md` (about 3,800 words), a search log,
reading guide, annotated bibliography, importable BibTeX, canonical reference
JSON, four source registries/reviews, retrieval manifest, hash-bound manual
identity checks and a validation record. Three bounded subagents independently
reviewed theory, memory/experiments and architectures; the coordinator reviewed
graph/measurement antecedents and integrated a second scientific and metadata
check. The 98 source records merge to 90 distinct works. Evidence depth: 26
substantive selected-section readings, 60 abstract records, four metadata-only
records. These labels do not mean complete proof or implementation verification.

All 90 works have public PDF copies under ignored `literature/papers/`
(197,085,607 bytes; approximately 188 MiB; 2,267 pages including appendices).
Extracted text and PDF renders are also ignored. Several publisher or obsolete
links failed; legitimate author/institutional/publisher alternatives resolved
all entries. The ledger preserves attempted URLs and errors. Reviewed arXiv
versions were pinned where version drift was detected. Bengio and Dahmen copies
are manuscripts, explicitly distinguished from typeset published versions.
Jaeger and Bengio text extraction is imperfect; their identity/title pages were
rendered and visually checked against the recorded hashes.

The integrated verdict rejects generic novelty for periodic Floquet memory,
recurrent/graph/transformer collapse, noise-maintained geometry, and contracting
loops with input reinjection. Close antecedents include public T-GRAB, Pals,
Ashwin–Ceni, Arroyo, Ganguli, Sharma, Fokam, MeSH, Labovich and Viakhirev.
Recent preprint conclusions are treated as conditional claims, not reproduced
facts. The report specifies five candidate open questions around required
history loss, its mechanism, causal prediction, native transfer and bounded
mitigation, each with decisive tests and falsifiers.

Key corrections to carry into implementation: the equal-horizon affine and
commuting-product cue-channel nulls; complete-state conditioning; common joint
noise-law assumptions; continuous versus categorical memory; readout offsets;
finite cue-sample rank ceilings; Gaussian-channel assumptions; and individual
horizon arm gaps alongside the primary difference-in-differences. A positive
interaction alone need not imply periodic harm. Broad architecture coverage
should follow a small mechanism study and independent confirmation. The plan
now links this reassessment; executable DAG/configurations are still pending.

Commands used include:

```bash
uv run --no-project --with pypdf --with fonttools code/literature/build_library.py --download --workers 3
python3 code/literature/build_library.py
git diff --check
```

Targeted retrievals used `--only` and `--refresh`. An ephemeral validation script
used `pypdf` and `bibtexparser==1.4.3`; rendered exceptions used `pymupdf`.
The initially resolved BibTeX parser exposed a different API, so validation was
rerun with the explicit compatible version. No project dependency or lockfile
was changed. Validation passed: 90 parsed BibTeX entries with matching IDs;
90 unique PDF signatures/hashes/byte counts/page counts; local text presence;
zero unresolved identity flags; 155 local Markdown paths; reading-list anchors;
helper syntax; and whitespace checks. See `doc/literature/validation.json`.
Minor PDF object-offset warnings did not prevent page parsing; this is not an
all-pages rendering audit.

No training, cluster jobs, source-snapshot edits, commits or pushes were
performed for this literature task. Remaining work: expand citation chains
before submission, audit uninspected proofs and selected preprint code, recover
legacy graph assets, and translate the revised scientific gates into frozen
vector pilot configurations. No absence-of-prior-work guarantee is claimed.


## Authorized initial execution and node-local Alliance access

The owner explicitly instructed cluv synchronization, initial submissions,
prioritization of other Alliance clusters, and an actual continuously working
Qwen cluster researcher. This supersedes the earlier pending-allocation state.
The bounded initial envelope is in ignored `configs/approval.local.json`:
maximum four CPU smokes (1 CPU/2 GiB/10 minutes each), four seed-0 GRU/LSTM
categorical/continuous GPU pilots (1 GPU/4 CPU/16 GiB/1 hour each; at most two
running), scratch staging up to 20 GiB per live-verified host, no race or paid
fallback. This is an initial engineering phase, not the broad scientific sweep.

Mila and Tamia scratch source/Python/locked NumPy environments were staged.
Native cluv synchronized laptop->Mila and Mila->Tamia. Cluv API submissions
recorded exact CPU source `7ca2be4c7827d0644b0216cb86bec1dbb855f5a4`:

- Tamia **536757**: COMPLETED, exit **0:0**, elapsed three seconds. Calibration
  JSON/NPZ/provenance artifacts were retrieved from Tamia to Mila using cluv.
- Mila **11190579**: FAILED, exit **2:0**, elapsed one second. It stopped at
  the prepared-Python check. `uv python find --no-project` selected the inherited
  controller virtual environment. Adding `--system` selects the exact prepared
  scratch-managed interpreter; verified on Mila. No unrecorded retry occurred.

The new immutable source adds seven mechanism calibration families (36 checks),
paired vector-memory tasks, complete-state GRU/LSTM training/evaluation,
held-out cue probes, effect contrasts and provenance. Full local suite passed
29 tests in 1.587 seconds after scheduler identities were added to status
artifacts. Jobs export `CLUV_CLUSTER_NAME`; Qwen only imports status job IDs for
its own controller cluster, avoiding cross-cluster ID collisions. CUDA remains
unverified locally. Publication commits: `bb81ee4` experiment implementation,
`1fb4402` literature/research plan, `7996fc9` cluster-scoped monitoring; all pushed
and synchronized to Mila/Tamia. Source snapshots remain untouched.

Owner's supplied cluv transcript authenticated from **tamia1** to Narval,
Rorqual, Fir, Nibi, Trillium, Trillium-GPU, Killarney and Vulcan without reported
errors; Mila failed public-key authentication from that route. Coordinator's
existing SSH master lands on **tamia2**. Pinned cluv API checks there verified
zero of eight reusable Alliance remote masters. UNIX control sockets are
login-node-local even when filenames appear in shared home. Strict BatchMode
routes to tamia1 require Duo. Two additional interactive attempts timed out or
closed before approval; no host-key bypass or SSH configuration change occurred.
The passive `code/ops/verify_cluv_access.py` is prepared for the existing owner
shell, with no interactive auth, remote mutations or scheduling. Other-cluster
accounts/one-GPU profiles remain unverified; their pilot launches are pending.

An actual Qwen service was verified from Mila with exact model identity
`Qwen/Qwen3.8-Flash-Next-FP8`. The first persistent leaf worker (Mila PID2808708,
08:10:33 UTC) consumed its 2,048-token response cap entirely in reasoning and
returned no final answer; one request used 5,923 total tokens. The coordinator
stopped that exact PID before its retry. The direct-JSON repair uses official
Qwen chat-template controls, rejects truncated/empty final content, and retains
raw failures and usage. One actual local worker verification then passed exact
model identity and review schema: 5,509 prompt/464 completion/5,973 total tokens.
Two phase research attempts are accounted; restarted daemon gets at most 22
additional attempts over 24 hours, polls evidence each minute and proposes hourly.
Credentials travel only via secure stdin to process environment; no values are
persisted. Qwen drafts falsifiable next-step proposals; scientific acceptance,
code changes and submissions remain with the coordinator. Restart/liveness
verification and new-cluster authentication are the next operations.


### Persistent worker and passive verifier completion

Replacement Qwen worker is running on Mila as **PID2812601**, started
2026-10-11 08:15:21 UTC from source
`61e2598ae0a086ce908f72b9253d872bc6dd3d91`. Output:
`/network/scratch/m/mehrab.hamidi/representation-collapse/runs/qwen-worker-phase1-v2-20261011T081521Z`.
First actual request completed 08:15:24 UTC with proposal_ready, exact model
identity, schema verification, no reasoning output and no provider errors;
3,911 prompt +468 completion =4,379 total tokens. Process and heartbeat were
independently checked. Maximum 22 remaining attempts/24 hours, hourly proposals
and minute evidence polls; no paid fallback or autonomous scheduling.

Coordinator reviewed that first draft and requires revision before allocation:
10 CPU-hours exceed the envelope; train-with-fixed-weights is ambiguous;
p>0.05 is not an equivalence or absence-of-effect criterion. The existing frozen
engineering pilot remains the next executable path after account verification.
Review saved in ignored `runs/ops/qwen-first-proposal-review.json`; no scientific
claim or submission was accepted from the response.

The passive verifier was exercised against Mila from the laptop: active cluv
master, remote identity, scratch, partition and account/QOS queries succeeded.
Default target parsing initially exposed argparse's empty-list/choices behavior;
fixed and rerun successfully. Source fix `5b51a38` was published and synced to
Tamia through native cluv. A fresh Tamia2 passive check at 08:16:12 UTC verified
local Tamia resources; eight cross-node control checks hit their ten-second
bounds. They remain unverified from the owner's tamia1 controller. This probe
sent no Duo requests and changed no remote scheduler/SSH state. The two earlier
extra Tamia1 Duo attempts are closed, with no authentication pending.

The owner can run the prepared verifier directly in the existing tamia1 shell:
`cd "$SCRATCH/representation-collapse/source"` then
`uv run --locked --group ops python code/ops/verify_cluv_access.py`.
This reads controller identity and saved masters without requesting new Duo.
GPU jobs and other-cluster staging remain pending that usable connection and
scheduler-account evidence; the initial phase already includes the completed
Tamia calibration and active Qwen researcher.

### Fresh Alliance authentication and reusable access — 12:27 UTC

The owner requested another authentication round and then said "try again".
Fresh native cluv logins from Tamia2 completed Duo on all eight remote aliases.
The first round's sockets were absent after the controller command ended.
No surviving private SSH masters were found; effective logind
`KillUserProcesses=false` was verified, so that proposed explanation is not
supported. The actual disappearance cause remains unresolved. Independently,
cluv's synchronous `ssh -G` resolution cost roughly 1.2–1.4 seconds per host,
which can consume simultaneous ten-second verifier deadlines. The verifier now
checks hosts sequentially with a separate 15-second acquisition deadline and
25-second probe deadline; scheduler failures are recorded separately from
authenticated identity.

A project adapter retains native cluv 0.1.5 and uses UID/project-specific
owner-only sockets on `/tmp`, with strict known-host checks. It changes no shared
SSH settings. A second controller holds its native login event loop and terminal
for a bounded eight hours. Narval first passed an independent passive check at
12:23:02 UTC; then Rorqual, Fir, Nibi, Trillium, Trillium-GPU, Killarney and Vulcan
completed sequential Duo and identity checks. Tool session 83529 remains open;
no Duo request remains pending. A different passive process at 12:27:34 UTC
verified all nine Alliance identities, eight reusable remote masters and local
Tamia. Seven initial shells exposed both scratch and scheduler associations;
Killarney exposed neither, and Vulcan's scratch variable was unset despite
successful scheduler queries. Additional read-only shell probes investigate
those two environments. Reuse after closing the held controller remains
unproven; the current workaround requires it to remain open.

Raw proof: ignored `runs/ops/cluv-alliance-access-20261011.json` and
`runs/ops/cluv-access-probe-debug.json`. Current routes, accounts and commands:
[cluster access](../../auto/cluster-access.md). Code review corrected SSH option
precedence so passive BatchMode safety wins, captured base callables to avoid
recursive wrappers after reloading, retained local stderr, and bounded passive
post-login identity checks. Both files pass syntax checks and `git diff --check`;
actual native cluv probes provide the acceptance evidence. No scheduler jobs
were submitted during this audit, and no other project was modified.

Mila separately passed its native cluv identity/scratch/account/partition probe
from the Mac at 12:24:00 UTC. Exact Qwen PID2812601 remained alive on login-4;
heartbeat at 12:23:59 UTC reported five schema-verified reviews, zero consecutive
provider errors, and 21,920 cumulative tokens (19,551 prompt +2,369 completion).
Safe evidence is in `runs/ops/cluv-mila-access-20261011.json`; no credentials,
service endpoint or process arguments were printed. These remain proposals,
not accepted scientific findings or authorization for additional allocations.

Follow-up resolution: Killarney and Vulcan both expose writable
`/scratch/mehrab`, account `aip-ioannism`, QOS `interac,normal` under `bash -lc`.
Killarney initializes Slurm from `/cm/shared/apps/slurm/current/bin` and exposes
only GPU partitions, including H100/L40S; Vulcan exposes CPU and L40S GPU
partitions. No module loads or profile edits were required. The passive verifier
now invokes remote probes in a login shell. A complete rerun at **12:31:07 UTC**
passed **9/9** identity, scratch-directory, partition and account checks,
with exit code zero for every probe and **8/8** reusable remote masters.
Acceptance assertions were independently applied to the saved JSON:
`runs/ops/cluv-alliance-login-environment-20261011.json`. Detailed environment
proof is `runs/ops/cluv-killarney-vulcan-environment-20261011.json`.
Future cluv sync/submission still needs explicit scratch/Slurm environment
initialization; the socket adapter does not itself modify remote profiles.
