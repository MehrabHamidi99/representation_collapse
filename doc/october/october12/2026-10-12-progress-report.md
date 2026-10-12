# 2026-10-12 — submission framing: question, theory, design, novelty, structure

## Purpose

The owner asked for an investigation of the project (documents, literature,
inherited Floquet draft, tests) and for the project structure, theory,
experimental setup, novelty statement and research question to be made clearer
and suitable for submission. This entry records the resulting document
consolidation. It contains **no new empirical result**: no training, no cluster
submission, no raw-array reduction, no model-service call. Every statement
below about trained models repeats the verified state of 2026-10-11.

## Diagnosis before the work

The scientific content was strong but fragmented. The question appeared in
four forms (research-program hypotheses H1–H4, the periodicity plan's central
question and five preregistration questions, the novelty audit's narrowed
question with OQ1–OQ5, the competence-recovery framing). Theory was spread over
research-program.md, periodicity-research-plan.md §2 and novelty-review.md §3
without a single proposition list carrying assumptions, proofs, fixtures,
counterexamples and predictions, although the plan itself demanded that
format. The experimental plan still led with the 351-checkpoint screen that the
novelty audit had deprioritized, and its broad text did not reflect the
competence failure of all four v1 pilots. The novelty audit was a critique, not
a positive delta table usable in a paper. Operations and science documents
shared `doc/auto/` with no reading order, and the only manuscript was the
inherited May 2025 note whose inferences are known to be invalid.

## Frozen spine shared by all workers

Working framing: when does repetition erase what a recurrent network needs?
Useful-history loss under controlled periodic forcing in fixed-weight recurrent
models. Primary question: at fixed weights and equal retention horizon, does
higher-order repetitive forcing of a distractor body cause excess loss of
previously written, task-required information relative to a matched
non-periodic body; through which state-transition mechanism; and when does
stable attractor memory prevent that loss. Three estimands kept separate
(cue-conditioned covariance geometry; finite-time tangent propagation with
Floquet labels only under verified closure; behavior and information with
causal interventions). Hypotheses H1–H4 plus the readout-offset and stable
categorical attractor alternatives. Primary contrast `δR` at `R∈{2,16}` and
`DID=δ16−δ2` behind a coordinate-wise competence gate. Mandatory analytic
nulls. Gate order A–D from the novelty audit. Evidence state exactly as in the
tracker. Non-claims listed explicitly.

## Delegated work and deliverables

Four bounded workers with disjoint write scopes, inherited session model, no
cluster or model-service access, followed by two follow-up passes:

| Owner | Deliverable | Size | Verification performed by the worker |
| --- | --- | --- | --- |
| Theory | `doc/auto/theory-propositions.md` | 492 lines | 34/34 tests; calibration CLIs 10 and 36 checks; scratch NumPy script 29/29 checks for P1, P3, P5, P6, P7 (outside the repository) |
| Design | `doc/auto/experimental-design-v2.md`; `configs/research-plan.json` revision v2 | 692 lines after follow-up; +160/−2 | Task statistics re-derived from `vector_task.py`; 39/39 links resolve; plan JSON valid, 16 nodes, acyclic, nine pre-existing nodes byte-identical |
| Charter and manuscript | `doc/auto/research-charter.md`; `paper/` (main, 13 sections, snapshot bib, README) | 295 lines; about 1,670 lines | `latexmk` build 34 pages after follow-up (33 before), 0 errors, 0 undefined references, 0 overfull boxes; 86 cite keys all present; 12 hashes traced to committed artifacts; build directory deleted |
| Implementation audit | `doc/auto/implementation-audit.md` | 450 lines | `uv sync`, test suite, both calibration CLIs with manifest hash recomputation; gap claims verified against file:line |

Coordinator integration: science-first `README.md`, `doc/README.md` reading
map, `.gitignore` for manuscript builds, tracker, this report, final checks.

## Content decisions

- **Theory.** P1 affine ordering null (with offset `k(h)`, joint-noise-law
  condition, nonlinear encoder, readout-offset corollary); P2 commuting
  switched-affine null with the process-noise caveat and a derived remark that a
  single diagonal selective state-space layer with symbol-dependent transitions
  satisfies the null; P3 noncommutativity necessary but not sufficient, with two
  scratch-verified counterexamples; P4 continuous-memory rank bound (Seung
  antecedent); P5 forced contracting scalar recurrence; P6 rank ceilings with a
  general `m` distinct rows ⇒ centered rank ≤ `m−1` lemma and a finite-sample PR
  bias approximation; P7 PR invariances; P8 nonnormal gain; P9 Gaussian channel
  as declared surrogate; P10 depth-loop reinjection; P11 stable categorical
  basins; P12 open Gate C target stated as what must be proved, not claimed.
  Fourteen corrections to `floquet_analysis/main.tex` with line numbers,
  including its self-contradiction at line 716; the draft is untouched.
- **Design.** Gate B is competence-first with the six-stage ladder, frozen
  learning-rate grid, selection by validation loss only, coordinate-wise gates,
  three discovery seeds then ten confirmation seeds. Registered family:
  `δ16` and `DID` × four cells, Holm at 0.05; proposed smallest effect of
  interest 0.02 (categorical error difference and normalized MSE); paired
  bootstrap `B=10,000` cross-checked with the exact sign-flip test and a paired
  t-interval; TOST for equivalence; power check before confirmation; a hashed
  freeze record before any confirmation manifest is inspected. Sample sizes
  restored to 1,024 paired episodes per horizon and 8 driver pairs × 512 cues.
  Floquet labels require input periodicity, complete-state closure on every
  adjacent cycle pair and Jacobian closure; otherwise finite-time propagator.
- **Charter.** Safe, conditional and forbidden contribution language; an
  18-row antecedent delta table; T-GRAB relation (coauthorship, reused
  `Pdet(k,n)` tasks, GC-LSTM baseline, per-time F1 protocol); publication
  threshold distinguishing a mechanism paper from a null/counterexample paper,
  with the decision rule applied after Gate B confirmation.
- **Manuscript.** Results report only the four failed-competence pilots (native
  errors, zero-predictor baseline, probe ranges, variance and PR), the executed
  but unreduced periodicity batch (five named `\todo` items), and the legacy
  graph observations as unverified attributed observations. Seventeen named
  placeholders remain; the README's checklist converts the skeleton to a
  submission only after Gate B evidence exists.

## Findings that qualify existing evidence

1. **Probe feature floor.** `pilot.py:128` floors per-feature standardization
   at `1e-8`. Observed v1 cloud variances (~1e-16 over 128–256 coordinates)
   imply per-feature std ≈ 1e-9, so most features sit at the floor; with exact
   linear noise-free memory at that scale the probe returns MSE 2.7e-4 instead
   of 0, and at std 1e-10 MSE 0.175 against 0.33 control. The v1 R2-versus-R16
   probe contrast therefore carries a horizon-dependent effective ridge
   strength. The R2 counterevidence is retained but qualified; floor-hit counts
   are recoverable from the saved `probe_feature_scale` arrays.
2. **FP32 closure tolerance.** The periodicity protocol's FP32 closure
   `atol=1e-7` is below the float32 half-ULP L2 floor (≈2.4e-6 for a 13×128
   block at unit scale); an FP32 closure pass effectively means a bitwise
   repeated cycle. The pending independent reduction must express closure in
   ULP units at the measured state scale against the FP64 replay.
3. **Provenance.** `pilot.py` neither sets nor records TF32/cuDNN flags;
   `deterministic_algorithms: True` is a constant. CPU repeat runs are bitwise
   identical (audit); CUDA repeat-run determinism is unverified.
4. **Document-versus-code mismatches** (design v2 §1.5, audit §4): evaluation
   pairs 256 executed vs 1,024 planned; driver pairs 2 vs 8; the 38→64 encoder
   is a widening, not a bottleneck; single learning rate and seed 0 executed
   although `vector-pilot-v1.json` lists seeds 0–2; aggregate 0.25 engineering
   gates vs coordinate-wise 0.95/0.05 research gates; R0 unsupported by
   `paired_body_ids` and both validators; dwell fixed at 1; control bodies are
   nested (`control[:R−1]+final`) although the protocol text says they need not
   be; the competence-recovery plan's oracle check has no committed code;
   `pilot.py`/`periodicity.py` emit no `argv` or per-run `verification.json`
   required by the workflow contract; D90/D95 and stable rank are not computed.
5. **Environment.** `uv sync --locked` alone removes torch and the ops packages
   (they live in non-default groups), so the README's former command could not
   reproduce 34 tests; corrected to `--all-groups`. On this Mac `uv run` cannot
   import the package because every `.venv` file carries the macOS hidden flag
   and CPython 3.11.13 skips hidden `.pth` files; the flag is re-applied by an
   unidentified process. `PYTHONPATH="$PWD/src" .venv/bin/python` is the
   documented fallback. During the audit the local environment was temporarily
   torch-less and was restored with `uv sync --locked --all-groups`.

## Commands and evidence

```bash
PYTHONPATH="$PWD/src" .venv/bin/python -m unittest discover -s tests -v   # Ran 34 tests, OK (4.4 s and 3.3 s in two runs)
PYTHONPATH="$PWD/src" .venv/bin/python -m representation_collapse.calibrate --output /tmp/rc-audit/calib            # 10 checks passed
PYTHONPATH="$PWD/src" .venv/bin/python -m representation_collapse.mechanism_calibration --output /tmp/rc-audit/mech # 36 checks passed
cd paper && latexmk -pdf -interaction=nonstopmode -outdir=build main.tex   # 33 pages, 0 errors; build/ deleted afterwards
```

Versions: Python 3.11.13, NumPy 2.4.6, torch 2.6.0 (CPU, macOS arm64); TeX Live
2025 with latexmk 4.86a. `metrics.py` SHA-256 `41a4790b…` matches the hash cited
in the 2026-10-11 scientific review. Scratch verification scripts live only
under `/tmp/rc-theory/`, `/tmp/rc-design/`, `/tmp/rc-paper/`, `/tmp/rc-audit/`.

## Scientific decisions

- The charter governs; older plans are deferred coverage where they disagree.
- The 351-checkpoint screen is not the next compute request. The next
  experiment remains the competence-first discovery grid, which needs the
  Gate B prerequisites implemented and a new approval envelope.
- No claim is promoted. The harmful-collapse decision stays inconclusive; the
  periodicity batch stays executed-unverified; the legacy multipliers stay
  preliminary-attributed.

## Unresolved

- Independent reduction of the periodicity raw arrays (now with the ULP and
  probe-floor qualifications).
- Pinning the T-GRAB commit and dataset revision; the AB versus AABB identity
  of the inherited figures.
- Freezing the proposed statistics (SEI, family, interval method) in a hashed
  record before any confirmation seed.
- The cause of the re-applied hidden flag on `.venv` files.
- Whether the executed GPU pilots ran with TF32 enabled.

## Integration validation and publication

Follow-up passes threaded the audit caveats into the design (Gate A fixtures
for ULP-unit closure and probe scale equivariance; Gate B provenance
requirements for TF32/cuDNN flags and repeat-run determinism; probe floor
declared relative to cloud scale with floor-hit counts and an unstandardized
sensitivity analysis; an optional commuting-null selective-SSM architecture
control B.5; Holm for the eight-test decision family with Bonferroni-adjusted
paired intervals at `1−0.05/8` as the effect-size report) and into the paper
(results, limitations, reproducibility, protocol appendix, design section).
`configs/research-plan.json` revision v2 has sixteen nodes; the nine
pre-existing nodes are byte-identical to the previous commit; new nodes are
`periodicity_independent_reduction`, `competence_recovery_discovery`,
`mechanism_study_gate_b`, `theory_gate_c` (drafted),
`research_charter_and_paper_skeleton` (drafted), `implementation_audit`
(drafted) and `native_transfer_gate_d`; `current_handoff` points here.

Coordinator checks: design and paper agree on the Bonferroni/Holm wording, ULP
closure units, TF32 provenance and the SSM control (grep); the plan loads,
every `depends_on` target exists and a topological sort succeeds; 360 relative
Markdown links across 32 documentation files resolve; `git diff --check` and a
trailing-whitespace scan of all new files are clean; `paper/build/` is absent
and ignored. Inherited directories `floquet_analysis/`, `ACM___TGRAB/` and
`ACM___TGRAB_KDD_2026/`, `src/`, `tests/`, frozen experiment configurations and
the 2026-10-11 report are unchanged. The commit for this work uses the `doc:`
prefix. No cluster job, model-service call or compute approval is part of it.

## Later — implementation pass stopped

Five Gate B implementation/pin workers were launched after `eb12ea1` and then
stopped: the first batch failed on a model-usage limit; the second batch was
interrupted by the owner. No new Gate B code was accepted, no new compute was
submitted, and no empirical claim was accepted. Working tree after the stop:
committed documentation session remains `eb12ea1` (ahead of `origin/main`, not
pushed); one untracked leftover `src/representation_collapse/diagnostics.py`
(320 lines: R0 constructors, single-WRITE episodes, streaming-register oracle
and shuffle controls) is not imported by existing modules and has no matching
test. It is not an empty stub, so it was left untracked rather than deleted.
Expected siblings (`gates.py`, `stats.py`, `capture.py`, `interventions.py`,
`provenance.py`, `reduce_periodicity.py`, `probe_floor.py`,
`graph-assets-pin.md` and matching tests) were not created. Next actions remain
independent periodicity reduction, Gate B constructors, a hashed freeze record,
and the graph-asset pin.

## Follow-up review and runtime-status investigation — 01:26–01:40 UTC

The owner requested inspection of the new work and a current cluster research
worker check, then supplied the stopped-worker handoff. This review confirmed
`eb12ea1c86cbcfb56dbf8ccad95290cdd059b76c` and
`dc5e020a3b7489b8b1abf2191937476a33469f67` in the local repository. The latter
arrived during inspection; its stopped-pass entry above is preserved. No
implementation pass was resumed and no leftover file was deleted or accepted.

Two bounded independent reviews, with disjoint ignored receipt paths, audited
the original documentation commit against `8fce9b0`. They found substantive
corrections rather than a new empirical result. The current design, audit,
theory and paper now correct the following statements:

- A passing FP32 closure tolerance need not be bitwise equality. A sparse
  one-ULP difference at 0.1 passes the actual frozen global bounds; a dense
  half-ULP norm is not a universal nonzero residual floor.
- Commuting deterministic cue propagation does not require equal native
  error: offsets and process-noise propagation can still differ. The optional
  SSM control's prediction now concerns the cue channel.
- The linear readout-offset MSE gap includes its residual/offset cross term.
  P1 distinguishes shared realized noise from episode-noise covariance.
- Average approximate MSE competence does not imply noise-resolved local
  rank. Exact smooth-memory P4 is retained with its original assumptions.
- Sample-rank bounds include state dimension; motif trigram differences and
  hundredfold probe regularization changes are not universal across pairs/cells.
- Sign-flip exactness requires a justified null symmetry/randomization;
  shifted SEI hypotheses and equivalence multiplicity must be frozen. The
  proposed adjusted ten-seed t-interval half-width is 1.1215 seed SD, compared
  with 0.7154 for the ordinary 95% interval. Bootstrap coverage is approximate.
- Scalar forced-response uniqueness, nonzero coupling, harmonic scope and
  participation-ratio equality conditions are stated precisely.

The initial document session's claims above remain a historical record; this
appendix corrects the present interpretation rather than rewriting that record.
No frozen v1 tolerance, model source, existing test, experiment configuration,
lockfile or inherited manuscript was changed. Remaining design issues include
explicit vector versus graph readiness, statistics/prediction freeze
dependencies, confirmation competence/missing-seed policy and measured costing
of the expanded battery. They are recorded in the
[review](../../auto/new-work-review-2026-10-12.md); no gate is promoted.

The untracked 320-line `diagnostics.py` has valid syntax and passes eight small
R0/R1/R2 and single-WRITE constructor/oracle fixture cases across both target
types. Its hash is recorded; bytes remain unchanged. These bounded inspection
checks are not a matching test suite or Gate B integration. Other stopped-pass
modules and graph pin are still absent. The sixteen-node dependency graph is
acyclic; 220 relative links resolve; `git diff --check` passes. The corrected
manuscript builds to 35 pages with no errors, undefined references or overfull
boxes; build artifacts stay outside the repository under `/private/tmp/`.
The existing 34-test suite was not rerun for these documentation corrections.

The passive live SSH attempt at 01:26:44 UTC returned exit 255 before reaching
Mila: the saved control socket was inaccessible and hostname resolution failed
under this session's restricted network/filesystem policy. Local cluv remains
0.1.5, but no remote cluv/scheduler or Qwen process check was possible. **Current
worker state is unverified.** The latest verified Qwen record remains October
11 16:23:17 UTC, 14 schema-valid reviews, zero consecutive errors, 76,635
reported tokens; call0014 consumed the periodicity completion packet. The
24-hour/22-attempt contract may since have reached its call cap; no running or
stopped state is inferred. The stopped implementation workers are separate.

The four periodicity jobs retain their independently verified October 11
completion and hash evidence; raw-array numerical reduction is still pending.
New documents reaching Mila are unverified. The local branch has two commits
beyond cached `origin/main`; a fresh publication check was unavailable. Current
review changes are left locally: Git metadata is read-only in this session,
and remote network access is restricted. No submission, worker restart,
authentication request, paid fallback or manual model-service call occurred.

The tracker and [dated status](../../auto/research-status-2026-10-12.json) now
separate historical evidence from fresh observations. Detailed proofs and
validation receipts are in ignored `runs/ops/20261012-new-work-theory-review.json`,
`20261012-new-work-design-review.json` and
`20261012-investigation-verification.json`. Next steps remain independent raw
reduction, reviewed Gate B implementation, the hashed statistics freeze and
T-GRAB provenance recovery. A fresh cluster check requires a session with
working network access; no additional compute follows from this review.


## SSH retry and diagnostic-code repair — follow-up

The owner asked to retry SSH and fix the leftover diagnostic draft. The first
bounded passive SSH probe again failed inside the local sandbox. Retrying via
the approved execution mechanism succeeded; Mila returned
`login-4.server.mila.quebec`. The earlier inability to connect was a session
restriction, not evidence that the owner's SSH setup was broken.

At 2026-10-12 01:50:35 UTC, terminal status and heartbeat agree that the Qwen
worker stopped normally on October 11 at 23:28:39 UTC after its 22-call cap.
All 22 review files have `schema_verified=true`; all scientific decisions remain
pending. There were zero consecutive provider errors, 125,730 reported tokens,
and recorded exit code 0. PID 2812601 was absent. The last proposal recommends
independent raw-array reduction; its suggested thresholds and scientific
interpretation are not accepted automatically. The safe receipt is
`runs/ops/qwen-recheck-20261012.json`; private endpoint/config files were not read.

Mila's queue query succeeded and showed no project-name matching jobs. Remote
cluv remains 0.1.5. The existing source checkout is still `8fce9b0`, so the new
document commits were not delivered there. A passive attempt from Mila to the
old Tamia2 controller hostname failed DNS resolution; Alliance scheduler status
was not refreshed. The four periodicity completions retain their historical
verified status. No authentication prompt, worker restart, provider call or new
compute submission occurred.

“Untracked” meant the draft existed on disk but was absent from Git's index.
“Unaccepted” described missing code review/tests and integration, not rejection
of its research purpose. One bounded implementation worker repaired only
`src/representation_collapse/diagnostics.py` and added
`tests/test_diagnostics.py`; a separate reviewer inspected those files without
editing. The coordinator integrated documentation and validation.

The repaired additive module validates exact type/address flags, homogeneous
payloads, one terminal QUERY and no value leakage. It rejects invalid integer
arguments before consuming RNG, handles repeated writes with last-WRITE-wins
semantics, ignores nuisance writes and rejects unwritten query targets. R0 and
single-WRITE constructors are covered across both tasks and all eight required
coordinates. R≥1 outputs and subsequent RNG state match v1 exactly. Corruption
controls preserve dtype and caller arrays; memory permutations retain legitimate
identity draws. Chance equality is possible and is documented.

Focused tests: 14 pass. Full command
`PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v` passes **48/48**
with no skips (2.882 seconds). Independent review additionally checks 400
streaming cases and six exact v1/RNG comparisons, finding no blockers. Existing
v1 model/task/periodicity files, existing tests, frozen configs and lockfile are
unchanged. Validation receipts and the full test log stay in ignored `runs/ops/`.
The module, tests and usage guide were added to Git's index using approved Git
access after the sandbox rejected index.lock creation. No commit or push.

These utilities are accepted as local engineering infrastructure. Gate B still
needs its stage-aware training validator, coordinate-wise competence gates,
state capture, intervention/sham and statistics implementation. No learned-model
competence or collapse finding follows. See
[usage and scope](../../auto/diagnostics-implementation.md) and the updated
[dated status](../../auto/research-status-2026-10-12.json).


## Requested GitHub publication and Overleaf compatibility

The owner explicitly requested committing and pushing current changes and
reported Overleaf's rejection of symbolic links. Git's index identified exactly
four links: `.agents/skills/{experiment-loop,floquet-analysis,model-routing,
project-ops}` (mode 120000). These known project-owned links were replaced
with regular directories containing discovery `SKILL.md` entries. Each entry
retains its name/description, directs the agent to read the full canonical
`skills/<name>/SKILL.md`, and resolves supporting references from that canonical
file. Canonical skill content and inherited source snapshots were preserved.
AGENTS/README and the paper guide document the new layout and Overleaf main
document `paper/main.tex`.

Qwen's 22-attempt cap was a coordinator-selected bound, not a model/provider
limit. The initial worker contract had 24 attempts and 24 hours; one failed
response and one direct-JSON verification consumed two attempts before the
replacement launch, leaving 22. The ignored launch record explicitly supplies
`--max-calls 22`; the code default is 24. This explained the recorded stop
reason `call_cap`. No new worker call limit or restart was requested here.

GitHub SSH authentication failed public-key validation; the existing HTTPS
push route successfully read remote main at `8fce9b0`. Publication uses that
configured HTTPS route, with a normal fast-forward push. Ignored approvals,
credentials, runtime receipts and cluster artifacts stay outside the commits.

Publication completed: `4023674d64c302cc9da3bd8e3616edae7464a95e` was
committed and pushed normally to GitHub main, advancing `8fce9b0..4023674`.
This also published the previously local `eb12ea1` and `dc5e020` document
commits. A fresh HTTPS `git ls-remote` confirms the exact main SHA. The
committed tree contains zero mode-120000 entries; four regular skill discovery
entries retain matching frontmatter and valid canonical targets. Checks resolve
130 relative links in changed Markdown, confirm no inherited-snapshot edits or
ignored runtime/approval files in the index, and pass `git diff --check`.
The prior 48/48 test result remains applicable: no further source changes were
made during the import fix. This publication note is a follow-up documentation
commit. The actual Overleaf import has not been retried from this session.


## Conference-format manuscript revision

The owner requested a more precise, better structured paper with plots and
selected a venue-neutral ML format. Three bounded workers owned disjoint
narrative, theory/proof, and figure/results files. A subsequent read-only review
checked mathematical assumptions and claim scope; the coordinator integrated
corrections and reviewed the rendered manuscript. No cluster submission,
remote worker restart, or model-service call occurred.

The new title is **When Does Repetition Erase Memory? Identifiability Limits
for Recurrent-State Collapse**. The active source is `paper/main.tex`; all
inherited graph/Floquet source snapshots remain unchanged. The previous
35-page research-program skeleton is replaced by seven pages of main text,
one page of references and seven pages of appendices. The main argument
separates conditioned geometry, complete-state propagation and task information;
it presents affine/commuting ordering nulls, readout-offset effects, categorical
attractor memory, a matched-order experiment and the verified competence audit.
Internal operations and anticipated-result placeholders are removed. The
inherited-draft critique remains in the source inventory and theory documents.

Four visuals now support the argument: a WRITE/body/QUERY schematic, four
analytic dissociations, native pilot behavior, and an appendix counterexample
for the probe's fixed feature-scale floor. The generator
`code/analysis/make_paper_figures.py` creates vector PDFs, PNG previews, all 32
pilot-cloud table rows, and a manifest of source/output hashes, formulas,
parameters and plotted values. Two complete generations produced identical
PDF/PNG/table bytes. The generator passes all 36 mechanism checks, independently
checks the noncommuting risk formula through singular values, and compares the
probe-floor formula to the actual v1 ridge implementation at 101 amplitudes
(maximum absolute discrepancy 4.44e-16). These are constructed controls, not
new trained-model results. Single-seed pilot plots do not invent seed intervals.

The review tightened several assumptions: unrestricted decoding invariance
uses a rotationally invariant full noise law (independent isotropic Gaussian
noise in the example), not just isotropic covariance; linear-risk calculations
specify independent zero-mean noise; the offset-loss cross term remains;
exact smooth continuous-memory neutral modes retain their stronger hypotheses.
Failed native competence is not localized to acquisition, since R2 probes
already recover some cue information. A scalar invertible-code example shows
a possible probe confound without claiming to explain the actual R16 results.
The ABAB/AABB analytic construction is explicitly distinguished from the
stricter experimental motif match. Primary antecedents Pals et al. (2024),
Ashwin and Ceni (2023), and Ganguli et al. (2008) were spot-checked.

Validation commands:

```bash
uv sync --locked --all-groups
PYTHONPATH=src .venv/bin/python code/analysis/make_paper_figures.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

The optional `paper` dependency group adds locked Matplotlib and PyMuPDF.
The full source suite passes **48/48**, with no skips. The final PDF has
**15 pages**, four figures and 19 cited bibliography entries, with no build
warnings, overfull/underfull boxes or unresolved references. PyMuPDF-rendered
pages were visually checked, including figures, tables, references and the
reproduction command block. Final validation receipts and rendered previews
remain in ignored `runs/paper-revision-20261012/`; the full test log is
`runs/ops/paper-revision-tests-20261012.log`. An offline uv lock check initially
hit a macOS system-configuration panic inside the sandbox and passed when run
through the approved execution mechanism.

The [readiness review](../../auto/manuscript-review-2026-10-12.md) is explicit:
this is an editorially complete diagnostic manuscript. The analytic ingredients
substantially overlap prior work; all four trained cells fail native competence.
A strong empirical mechanism submission still needs competent replicated
baselines, independently reduced periodicity arrays, a frozen confirmation
procedure, and an original prediction or causal intervention with shams.
No harmful-collapse or empirical Floquet claim is accepted by this revision.
The broader research charter retains its publication gates. Source and figure
publication uses the owner's existing commit/push authorization; the exact
publication receipt is recorded after the push in ignored operations artifacts.


## Requested automatic-research status refresh — 07:52–07:54 UTC

The owner requested a status update and confirmation that automatic research
has actually run. Fresh read-only SSH checks reached
`login-4.server.mila.quebec`. The recorded Qwen heartbeat and terminal status
agree on a normal stop at the call cap on October 11 at 23:28:39 UTC: 22
provider attempts, 22 schema-valid reviews, zero consecutive errors, 125,730
reported tokens and exit code 0. Its recorded PID 2812601 is absent; a worker
process-name check also finds no process. The elapsed runtime is 54,797.7
seconds, about 15.2 hours. This establishes historical automatic proposal
generation, **not an ongoing unattended researcher**. All proposals remain
pending scientific acceptance.

Mila's live `squeue --me` query succeeded; project-name filtering finds no
matching jobs. The four training jobs and four checkpoint-periodicity jobs
retain the independently verified October 11 `COMPLETED/0:0` outcomes.
Their scheduler states were not freshly reconciled on the Alliance clusters.
Native cluv 0.1.5 passive checks from Mila find no reusable Narval or Rorqual
masters. The official Tamia alias now reaches authentication but denies
BatchMode access with `Permission denied (keyboard-interactive)`; no Duo
prompt was requested. The old controller route is not an active monitoring
session. A first combined collector was interrupted by a DNS lookup's OS error;
the subsequent isolated worker/queue collector and native cluv check succeeded.

Fresh safe evidence is in ignored
`runs/ops/autoresearch-live-status-20261012.json` and
`runs/ops/autoresearch-cluv-access-20261012.json`; their hashes are recorded
in the updated public `doc/auto/research-status-2026-10-12.json`. The probe reads
only project status/reviews and process/scheduler metadata, not credentials,
provider configuration or weights. No provider request, worker restart, job
submission or queue mutation occurred. Both GPU batch submission caps and the
initial worker attempt allowance remain consumed.

GitHub main was newer than the local manuscript commit: the owner's Overleaf
merge `4c0b56e96000bde758f48115254b8a0d1c45e718` includes
`01a9a0b` after `c7c5f44`. Inspection found only four wrapper executable-mode
changes, with no content changes. The clean local branch was fast-forwarded to
preserve that merge before writing this status update. No wrapper modes were
restored or other owner edits overwritten. Mila still runs source `8fce9b0`;
the current manuscript/diagnostic documents have not been delivered there.

Research progress is concrete: eight GPU experiments completed, automatic Qwen
proposals produced, diagnostic utilities reviewed with 48 passing tests, and a
complete 15-page manuscript with four reproducible figures published. The
scientific conclusion remains limited: all four native-competence gates failed,
and periodicity raw arrays still need independent reduction. Next work remains
that reduction, stage-aware/coordinate-wise competence recovery and the frozen
v2 statistics. A continued cluster worker needs a new bounded call envelope and
current source; renewed allocations need a new concrete compute envelope.
This status request did not restart either service. Updated tracker, compute
plan and structured status distinguish historical execution from present
inactivity. Publication uses the owner's existing commit/push authorization.


## Restored-access continuation — 08:38–09:51 UTC

The owner reported restored Tamia/native-cluv access and instructed continuation.
The coordinator applied the project-ops, experiment-loop, floquet-analysis and
model-routing skills. Bounded workers owned the independent reducer, additive
competence implementation, independent numerical review and result/figure
publication; the coordinator owns integration and scientific acceptance.
Original graph/Floquet source snapshots and prior reports remain preserved.

### Authentication, immutable source and cluv retrieval

The working route is Mac → Mila → **Tamia3** (`tamia3.tamia.ecpia.ca`), replacing
the unavailable historical Tamia2 route. Native **cluv 0.1.5** on Tamia3
verified identity, writable scratch and scheduler/account visibility for all
nine configured Alliance targets: Tamia, Narval, Rorqual, Fir, Nibi, Trillium,
Trillium-GPU, Killarney and Vulcan. Mila was verified separately. The restored
connections use ordinary cluv masters; the old project-private socket override
was deliberately not applied. No new Duo prompt was needed. A cold scratch
interpreter attempt timed out; system Python with the pinned ops site path
completed the same bounded passive audit:

```bash
cd /scratch/m/mehrab/representation-collapse/source
PYTHONPATH="$PWD/.venv/lib/python3.11/site-packages" \
  python3 -S code/ops/verify_cluv_access.py
```

The source implementation commit **ec1c489fc97c2e6cf94fd58e8a4e04dd3e8ba901**
was published to GitHub before execution. A 20 MB project Git bundle, SHA-256
`626ccd768bf180251e4641c8b99e742cf0c8ce553b298c2b9876e0df34cf5682`,
provided authenticated-source content where remote private GitHub access was
unavailable. Isolated source snapshots under
`representation-collapse/continuations/continuation-20261012-ec1c489/source`
were staged on all nine Alliance aliases and Mila through native
`cluv.cli.sync.clone_project` plus a process-local scratch-mirror Git URL
rewrite. Tamia uses `/scratch/m/mehrab`, the other Alliance targets use
`/scratch/mehrab`, and Mila uses `/network/scratch/m/mehrab.hamidi`.
Trillium-GPU shares Trillium's filesystem. All ten final source receipts match
`ec1c489`; previous checkouts, including Tamia's known untracked helpers, were
preserved. No global Git/SSH configuration or unrelated project jobs changed.

Two source-sync attempts stopped on controller config type errors before
completion: mutating the frozen partial config, then supplying a PurePosixPath
where its validator requires a string. A validated process-local
`dataclasses.replace(..., project_dir=str(path))` completed synchronization;
existing bundles/mirrors were retained. These failures were not compute jobs
or scientific observations. Source staging is not an assertion that every
cluster has a ready Torch environment.

Native `cluv.cli.sync.fetch_results` retrieved all four existing periodicity
directories to Tamia scratch. Original arrays and weights stayed on cluster
storage; only derived numeric audit outputs reached the ignored Mac runtime
area. The original four training jobs and four diagnostic jobs were freshly
reconciled as `COMPLETED/0:0`. A subsequent native-cluv read-only byte hash
check at 09:50:57 UTC verified all four original checkpoints against the frozen
registry without loading them. Receipts are in ignored
`runs/ops/tamia-restored-cluv-access-system-20261012.json`,
`continuation-cluv-source-sync-final-20261012.json`,
`continuation-cluv-source-sync-mila-20261012.json` and
`continuation-checkpoint-integrity-20261012.json`.

### One completed CPU audit; no new training

The frozen [independent reduction protocol](../../auto/periodicity-independent-reduction-v1.md)
was committed before real-cell reduction. The separately recorded continuation
allows exactly one Tamia submission: account `aip-ioannism`, partition
`cpubase_bycore_b1`, QOS `normal`, one CPU/2 GiB/ten minutes/zero GPUs,
no retry or race. Offline NumPy preflight passed with scratch-managed Python
3.11.13 and the locked environment. Submission used native
`cluv.cli.submit.submit` from the clean isolated source; its skip-sync option
followed explicit successful staging. The submitted wrapper arguments were:

```text
code/ops/periodicity_reduction_job.sh
  --source-commit ec1c489fc97c2e6cf94fd58e8a4e04dd3e8ba901
  --run-id independent-per-v1-20261012-ec1c489
  --cluster tamia
```

**Tamia job 612745** ran on `tc11101`, 09:00:41–09:00:55 UTC, and completed
`0:0` in **14 seconds**. Maximum recorded step RSS was **684,480 KiB**.
The reducer audited four cells, 64 tangent windows and 1,536 saved step-Jacobian
evaluations with no consistency errors. Independent result review passed
**31,264 assertions**, covering spectra, closure, denominators, phase alignment,
precision/adapter labels and derivative aggregates. All 28 original artifact
hashes and four original manifests matched prior independently verified bindings.
The reviewer checked the full derived audit and provenance; the CPU reducer
performed fresh raw-NPZ arithmetic. Their responsibilities are distinct.

Output is
`/scratch/m/mehrab/representation-collapse/logs/cluv/tamia_612745/reduction/`.
Reducer SHA-256 is
`8c8acfc68ae82baa7f746673ad56ea47be667a4d53cb015e5171c18ad2708bf4`;
raw report byte hash is
`dd27cf606a2d0a0598d5018983731225aaa7b341ac331ed56fcfac53264b0c1b`.
The ignored collected envelope is
`runs/ops/periodicity-independent-reduction-612745-20261012.json`, SHA-256
`7d1676cadeb0aa0db143aa9f6ee5566db64f2ae0f4ac0106e61839be5da79562`.
Independent review receipt is `runs/ops/periodicity-reduction-review-20261012.json`,
SHA-256 `f7373aa3e1e9bc3a7a503db864af8efcdf76c8fc2ad605f2cda22b9ee193915b`.

Slurm emitted a pre-step working-directory warning for a batch-created `/tmp`
snapshot. The correct executed reducer/config hashes, successful status and
consistent audit support numerical acceptance; the operational deviation stays
recorded. The future wrapper now puts its snapshot in shared scratch and passes
explicit `srun --chdir`. That fix was not used by job 612745 and was not rerun.
Bash syntax and seven argument-rejection checks passed. The one-submission
CPU continuation is consumed; this does not extend either prior GPU allowance.

### Accepted numerical scope and rejected overclaims

The [descriptive report](../../auto/periodicity-independent-results.md),
[exact scalar summary](../../auto/periodicity-independent-results.json) and
[vector plot](../../auto/figures/periodicity-independent-v1.pdf) are new verified
artifacts. Four-panel plots preserve all 128 native paired-cue measurements,
two driver panels individually, both arms and precisions, with no fabricated
confidence interval or positive replacement for zero. Regeneration from the
committed scalar JSON reproduces all derived outputs byte-for-byte. The PDF
and PNG were visually reviewed; figure provenance includes exact plotted values.

All 64 manual final-12-step tangent gains are below one, roughly 0.00417–0.01264,
in both forcing arms. Native R16/R32 cue separation is roughly 6.73e-9–2.77e-8
in FP32 and 1.39e-17–4.54e-17 in FP64; all terminal pairs remain bitwise
nonidentical and nonzero. Periodic last-cycle blocks pass tolerance at R16/R32,
with no exact bitwise closure. Controls compare different adjacent driver
blocks, so their larger closure residuals are not a retained-memory effect.
LSTM categorical/control panel 1 at R16 provides a manual/native classification
discrepancy under unchanged tolerances. Complete control bodies need not be
nested across horizons; R32 has state measurements only, no tangent assay.

**Decision: accept descriptive numerical audit only.** All four checkpoints
failed native competence, have one seed per cell and only two cue histories.
The derivative windows use cue zero and manual recurrence. These measurements
do not establish harmful collapse, population equivalence, a causal mechanism,
representation dimension, certified Floquet multipliers or cross-modal transfer.
Manual QUERY arrays and original probe clouds are missing from this schema;
query-adapter validation is partial and probe-floor robustness is still pending.
The current paper remains at its earlier evidence cutoff, with this audit
published separately; no new empirical paper claim was inserted automatically.

### Renewed approved Qwen participation and its limits

Automatic approval review first rejected the proposed renewed provider launch
because renewed transmission approval was absent. The coordinator presented
its exact project-excerpt payload, configured Qwen/Cairn destination and
SSH-forwarding route; the owner answered **“Approve the bounded Qwen run.”**
No renewed model call preceded that answer. Approval resolved the rejection
and is recorded only in ignored `configs/approval.local.json`.

The actual Tamia worker launched at **09:10:30 UTC**, PID **524659**, output
`/scratch/m/mehrab/representation-collapse/runs/qwen-continuation-20261012T091030Z`,
from source `ec1c489`. Bounds: 24 hours/24 attempts, evidence checks every 60
seconds, hourly proposals when unchanged, at most 18,000 content characters
and 2,048 requested output tokens, HTTP timeout 180 seconds, three consecutive
provider errors stop, no paid fallback. Credentials were delivered via secure
stdin to process environment; no key/provider endpoint configuration entered Git.
Encrypted loopback SSH forwarding reuses the existing Mila and Tamia masters;
other project connections remain intact. Uptime depends on the local proxy
and these hops remaining alive, so this is not a guarantee of uninterrupted
24-hour service or a permanent autonomous research dispatcher.

Two actual calls returned the exact requested model
`Qwen/Qwen3.8-Flash-Next-FP8`, with `finish_reason=stop`, no reasoning-text
fallback and two schema-valid proposals. At 09:47:47 UTC the PID was live,
heartbeat state running, zero consecutive errors, and cumulative usage was
10,683 prompt plus 988 completion tokens (**11,671 total**). Four compact
verified numerical/competence/coordinator-review packets were delivered to its
explicit evidence root; call-0002 cited that new feed. Worker observation is
deterministic; model calls generate proposals only. It cannot train, submit,
cancel, execute response text or accept scientific claims.

Both proposals remain scientifically unaccepted. Call-0001 proposed Floquet
exponents without certification and a rank-based stopping rule without support.
Call-0002 used synthetic analytic simulation to diagnose trained checkpoints,
blurred normalized MSE units and made a desired failure pattern an acceptance
condition. Those designs are not executable research protocols. Identity/schema
verification proves participation, not validity. Current structured status
preserves the initial stopped 22-call run as history and records this separate
explicitly approved continuation.

### Local competence work, verification and next node

Additive `competence.py` implements immediate single-WRITE/query, full-prefix
R0/R1/R2 assays, 128 episodes per required coordinate/arm, native QUERY-phase
score records, frozen 0.95 categorical and 0.05 population-normalized continuous
gates, paired raw-error contrasts and validation/test/fixture separation.
Score-stage eligibility excludes held-out selection and reports remaining
training obligations. Synthetic oracle fixtures count as zero trained-seed
units. Fifteen focused tests pass; the full source suite passes **81 tests,
zero skips, 7.441 seconds**. Receipt:
`runs/ops/restored-continuation-tests-20261012-final.log`, SHA-256
`a5a60adaeb85d8df1526f020a64f2b136d3fd6e091fd1e8725c1d212800d2d31`.

The next bounded engineering node is frozen training integration: update
boundaries, fixed-batch overfit/untouched IID generalization, gradients/parameter
changes, validation-only checkpoint/LR selection and native three-phase capture.
Then freeze effect-size, uncertainty, multiplicity and failed-seed rules before
new confirmation data. Recover probe/query evidence or define a separately
frozen assay. New GPU training needs an exact prepared allocation after those
prerequisites; no scientific claim or new allocation follows from Qwen proposals.
The living tracker, compute plan, cluster route, worker protocol, implementation
guide and DAG are updated; historical reports and original allocations remain
preserved. Publication uses the owner's existing commit/push authorization.


### Final independent competence repair and handoff

Independent code review identified COMP-001: finite continuous predictions
can have finite individual squared errors while their FP64 mean reduction or
population normalization overflows, producing Inf JSON metrics. No false
competence pass occurred. The fix explicitly rejects nonfinite coordinate,
aggregate, reference or paired means and normalized scores with `ValueError`;
it neither clamps scores nor changes thresholds. A regression covers coordinate
sum, aggregate sum and normalization overflow, with large representable failures
still reportable. Valid undersampled coordinates retain failed gates and None
for missing measurements; the existing nonempty-episode contract rejects empty
batches.

The independent reviewer passed sixteen focused tests and 25 supplemental
assertions. Sixteen ordinary fixture/finite-failure reports match ec1c489
exactly as sorted JSON, and overflow rejection emits no unhandled warnings.
The final full suite passes **82 tests, zero skips, 7.041 seconds**. Repair and
future shared-scratch wrapper correction are committed as **78ce9746b83ee4b1b6b6f48ff1bc641ce59bd8a0**; all ten
cluster source snapshots intentionally remain immutable ec1c489. This repair
must be restaged before new training, and was not the source of CPU job 612745.
No old outputs were rewritten or rerun.

Final review receipt `runs/ops/competence-independent-review-20261012.json`
SHA-256 `28e99e27b74d960ef136efa7125b6015d68a116426082172d7984519c4e28b6e`
preserves the original blocker and its resolution. Full repaired-suite log:
`runs/ops/restored-continuation-tests-20261012-repaired.log`, SHA-256
`e45fe4d1b3e64e71b41b7701073685643148e506b5ef7cf05656d85bbaf522f4`.
Coordinator regeneration of the scalar-derived report/PDF/PNG/manifest was
byte-identical; all 128 plotted native cue values and source hashes remain
verified. The plot's PDF SHA-256 is
`6226d8cac2066ec9ba13ff58c672ee56d82d424df8f85125b0440aa5b8ae2426`.

A fresh Qwen check at **2026-10-12T10:03:04.163200+00:00** found PID 524659 alive,
heartbeat running, 2 attempts and
2 schema-verified proposals,
0 consecutive provider errors.
The latest evidence contains all four curated continuation packets and the
latest response cites them. Numerical acceptance remains descriptive only;
Qwen proposals remain scientifically unaccepted. Runtime/provider configuration
and local approval records stay ignored. The final results/status publication
commit and remote main check are recorded separately after pushing.


## Requested continuous research/status iteration — 12:24–13:12 UTC

The owner requested updated status, concrete progress and continuous automatic
research. The coordinator applied project-ops, experiment-loop, model-routing
and floquet-analysis, assigning bounded disjoint implementation, native/selection,
operations and independent review tasks. A subsequent bounded graph-source audit
advanced asset recovery while remote authentication remained unavailable.

### Current remote status and continuity

At 12:24 UTC, Mac→Mila worked but the Mila→Tamia reusable master was absent.
BatchMode reached the official Tamia endpoint and was denied keyboard-interactive
login. One interactive Duo push attempt closed before successful authentication.
A fresh readiness/push question remains pending; no second unsolicited push,
worker restart or duplicate was launched. Native cluv **0.1.5** on Mila at
**12:48:38 UTC** independently reported no reusable Tamia master. This iteration
cannot establish current Qwen PID, call counter or scheduler state. Its last
successful liveness check remains **10:07:18UTC**, when PID 524659 was running
with two verified proposals. Current liveness is **unverified**, not reported as
running or stopped. Earlier completed scheduler outcomes remain historical;
no new compute or coordinator provider request occurred in this iteration.

Private route receipts: `runs/ops/autoresearch-route-recovery-20261012.json` and
`runs/ops/autoresearch-cluv-passive-recovery-20261012.json`. Failed probing output
is not accepted heartbeat evidence. The public current status preserves the
older successful snapshot and records the new authentication failure explicitly.

A new deterministic `code/ops/qwen_resume_budget.py` planner requires fresh
terminal status and absent-process evidence, sums all prior attempts in the
exact lease, and subtracts elapsed time from the original 24-hour/24-attempt
allowance. Live, stale, unknown or missing evidence prevents a duplicate. It
never launches, sends a provider request or grants approval. Renewal does not
reset the clock or multiply the call allowance. Existing transmission scope
and configured backend remain unchanged; no paid fallback is added.

Review also found that the worker searched the wrong location for public
protocols: it globbed `configs/`, whereas actual protocols are under
`configs/experiments/`. The repaired collector uses an explicit public
allowlist plus the competence integration document, preserves 18,000 content
characters/2,048 output tokens, and clarifies native addressed recall in its
research instruction. Four tests and seven independent mocked privacy/size
checks pass; private configurations, weights-folder metadata, wrong-cluster
job IDs and unrelated scheduler jobs remain excluded. No live model/scheduler
call was made in these checks. This repair is published but not deployed.

### Verified competence training integration

New additive modules provide the bounded trainer, validation-only selector and
native phase adapter; frozen v1 code/configs and all inherited snapshots remain
unchanged. The prospective protocol is
[competence-training-v2.md](../../auto/competence-training-v2.md), with
`configs/experiments/competence-recovery-v2.json`. The schedule remains at most
500/500/500/500/1,500 updates. Stage 1 explicitly assigns 100 fixed-batch updates
(four examples per required coordinate) then 400 fresh IID updates; its overfit
check is distinct from 128-episode/coordinate validation. A fixed-batch failure
stops at 100, while stage transitions use current-boundary gates, never a
historical best checkpoint. Stage 4 records 167/167/166 horizon updates, because
500 cannot divide equally among three horizons; paired arms are exactly balanced.

Seven validation batches are generated and saved before optimizer construction.
The six selection-horizon identities are caller-frozen hashes/sampling seeds,
checked again at evaluation and selection. Native three-phase capture preserves
all GRU hidden coordinates and both LSTM hidden/cell components. Behavior uses
uninterrupted full-forward QUERY logits; forked output is a numerical chronology
check, not a substitute for full native behavior. Pure FP32 training sets and
records actual TF32/cuDNN/matmul/determinism flags. Gradient, input-gradient and
per-module update norms, raw native logits, complete phase arrays, failed stages,
checkpoints and atomic manifests are retained.

Independent review reproduced and repaired INT-001 (validation batches could
change at selector boundary) and INT-002 (immediate QUERY could target an
unwritten address). It verifies full R0/R1/R2 coordinate gates, equal-arm/horizon
CE or raw-MSE selection, earliest-step/frozen-LR-order ties, and exclusions of
test, geometry and desired arm effects. Targeted tiny GRU/LSTM fits execute one
actual gradient update and retain early failures; stage 5 bookkeeping is tested
with explicitly mocked validation. Additional actual untrained R4/R8/R16
assays verify 2,048 episodes per assay,128 per coordinate and complete finite state.
These are engineering fixtures, not evidence of learned competence.

The full source suite passes **121 tests, no skips, 12.255 seconds**. Independent
integration review passes 35 targeted tests plus 75 additional assertions, followed
by the four worker-evidence tests/seven supplemental checks. Receipt hash:
`d82727a54af158176e87f5355984be5603637d81312ba68ac34a67c2ce41a7e8`.
The complete test log SHA-256 is
`d67f8ab16e7a5814f02c7a1e00ee9da489779438f24834c40ce49c89360bbbf3`.
Implementation commit **af20702cc3998d2d9793d1a674cdb4e52a787952** was pushed to GitHub before handoff.
All ten cluster source snapshots remain the earlier immutable ec1c489; this
new implementation has not yet been staged or executed on a cluster.

### Prepared four-fit canary, not an approved allocation

`configs/experiments/competence-canary-v2.json` prepares GRU/LSTM×categorical/
continuous, seed 0, LR 0.001, at most 3,500 updates per fit on Fir/Nibi. Requests
are one GPU/four CPUs/16 GiB/30 minutes per fit, four submissions, two dependency
chains, maximum two simultaneous jobs and two requested GPU-hours total; no retry,
requeue, arrays, races or fallback. Architecture is confounded with the chosen
cluster in this canary. It is four of twelve discovery fits, not a final grid-selected LR
or confirmation. Both target scheduler/typedGPU/account/cache fields are
explicitly unverified; no owner allocation approval was manufactured.

The new wrapper snapshots shared scratch with explicit `srun --chdir`, verifies
source/config/resource/approval bindings, uses locked offline Torch 2.6 and writes
atomic job provenance/status. The native-cluv submitter defaults to zero remote
actions and requires exact ignored owner approval plus fresh source/scheduler/
offline-cache reconciliation. Serial journaling and dependencies prohibit
unauthorized expansion, duplicate retry and more than two simultaneous jobs.
OPS-001, an embedded Python indentation error missed by outer compilation,
was independently found and repaired. All five generated Python snippets now
compile locally before preview or submission. Independent ops review passes
**91 checks**; physical Slurm/GPU/cache behavior remains untested. Its receipt
with a separately labelled coordinator metadata-only annotation has SHA-256
`db22a5198227f276765b972b3c00606586c86deda0b79921d98e5505f02780d9`.
Four committed-source cell previews confirm source-ready and zero actions;
target readiness and approval remain false. Preview command:

```bash
.venv/bin/python code/ops/submit_competence_recovery.py --cell 0
```

Exact allocation approval will follow fresh target/account/typedGPU/cache
verification. The old GPU budgets and completed CPU-audit cap are unchanged.
No new full-budget training has run, and complete Gate B still lacks competent
replicas, full-resource/determinism checks, probe/precision/causal measurements
and the statistics freeze.

### Graph upstream source/metadata progress

A bounded worker pinned T-GRAB source
`31293102b065feef7817ff86211427d7594edfdb`, dataset metadata revision
`603388f332c7756f90bc1702b26053c2bc834654`, and PyG-Temporal v0.54.0 tag commit
`46abd39abec0cd8de23c0e489a617c320c3fc5e8`. The coordinator rechecked all 23 small
source-file Git blob identities/SHA-256s, API receipts and dependency-cell hash.
Only 143,542 source bytes plus small public metadata/card files were fetched;
no dataset, weight, training, package installation or cluster operation.
[Audit](../../auto/tgrab-upstream-pin-2026-10-12.md) and
[pin record](../../../configs/graph/tgrab-upstream-pin.json) distinguish
provider-declared LFS hashes from unverified payload checksums.

Generator semantics are now verified: `(2,1)=AB`, `(2,2)=AABB`. The legacy
checkpoint's actual condition remains unknown. Complete core state is 2LNH,
not pooled 2H; pipeline cached z/previous-graph chronology also needs auditing.
Source inspection verifies a conditional eval-only branch that consumes the
current target before scoring and consumes it again when no cached z exists;
ordinary cached z evaluation is explicitly distinguished. Periodic stopping
checks cached test-F1==1, so a new scientific experiment must independently
freeze validation-only stopping. These are source findings, not empirically
reproduced behavior. The repository citation file declares a 2026 KDD DOI;
publisher verification and bibliography reconciliation remain separate.

### Scientific decision and next actions

Accept the local engineering integration and immutable public source/metadata
pins. No new harmful-collapse, competent-model equivalence, graph reproduction,
causal or certified-Floquet claim is accepted. Previously audited periodicity
measurements remain descriptive, and all original native-competence failures
remain preserved. Next: restore Duo access and reconcile Qwen before recovery;
stage the reviewed source and verify selected targets; obtain exact bounded
canary approval, then run/reduce every outcome; implement tiny graph complete-
state/chronology/derivative fixtures and recover legacy payload/checkpoint assets.
The main manuscript remains at its earlier evidence cutoff; no new empirical
result was inserted. Publication uses the owner's existing commit/push request.


## Allocation-envelope review and final integration — 13:33–13:46 UTC

The coordinator and independent reviewer found OPS-002: native cluv's sbatch
command had no array/federation override, so inherited `SBATCH_ARRAY_INX`,
`SBATCH_CLUSTERS` or `SLURM_CLUSTERS` could expand or redirect an allocation
before the batch wrapper examined it. This follows
[Slurm's input-environment precedence](https://slurm.schedmd.com/sbatch.html#SECTION_INPUT-ENVIRONMENT-VARIABLES).
No such option was observed on a real frontend; this was a locally demonstrated
submission-contract gap, not a reported remote allocation.

Repair commit **9e22c6cbb5821200bfafe4c6c41f168677d4691b** rejects configured
scheduler-input keys and rejects inherited options inside the actual native
`bash --login -c` body immediately before `sbatch`. Exact native cluv 0.1.5
command shape is checked, native `submit_job` remains the submission API, and
both native job records and the serial journal retain the exact guarded command.
Independent review passes 51 new checks, following 16 checks demonstrating the
finding; the earlier 91 checks and OPS-001 repair remain recorded. OPS-002 is
resolved with no implementation blocker. These checks use local stubs and
mocked remotes, never a real scheduler.

Seven persistent regressions exercise real local Bash with a harmless sbatch
stub: clean invocation exactly once; inherited options, including empty values,
prevent any invocation; configured inputs, malformed commands, unexpected
versions/wrappers and embedded Python failures reject; production execution
journals the exact guarded command. Twelve independent static checks accept
the test artifact. The amended ignored review receipt has SHA-256
`5ba0980039dd333c23c4a133a1347debb6e89049868594388f055f7eb962cdce`;
prior finding/repair receipts remain in its amendment chain. Local login-profile
portability is recorded separately from physical Slurm behavior.

The complete source suite now passes **128 tests, no skips**, in 12.649 seconds
(13.784 seconds measured process wall time). Log:
`runs/ops/autoresearch-integration-tests-20261012-guarded.log`, SHA-256
`6741a2e69ec050c8c0aafc7e74de4cd17ccde2044334bbcef6cf51e09250150a`.
All four guarded-source previews bind commit 9e22c6c and confirm source readiness,
zero remote actions, unverified targets and no allocation approval. Original
previews and source/review hashes are preserved rather than overwritten.

No new remote job, coordinator provider call, duplicate Qwen worker or
scientific training was launched. Tamia liveness remains unverified, last
successfully checked at 10:07 UTC; fresh Duo authentication is still pending.
The existing Qwen lease retains its original 24-hour/24-attempt bound. Cluster
source snapshots are still ec1c489 and need isolated staging before execution.
The scientific decision remains descriptive-only; failed competence does not
support harmful collapse. The tracker, dependency plan, prospective protocol,
source pins and status record are updated. The final publication receipt in
ignored `runs/ops/` records the verified GitHub revision after the push.

Final passive master check at 2026-10-12T13:49:43.084905+00:00: Mila remains reachable; the
configured Tamia socket is still absent (`ssh -O check`, exit 255). No Duo was
requested. The latest ignored receipt is
`runs/ops/autoresearch-final-passive-master-check-20261012.json`. This confirms
route unavailability, not a stopped or running Qwen process.

Evidence wording clarification: earlier entries described the absent SSH master
as expired. The directly observed fact is that its reusable control socket is
absent; the disconnection cause was not established. Living status documents
now distinguish that observation from the unverified Qwen process state.
