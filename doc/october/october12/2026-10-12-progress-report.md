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
