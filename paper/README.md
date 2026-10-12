# Manuscript skeleton

Submission-oriented skeleton for the paper described in
[`doc/auto/research-charter.md`](../doc/auto/research-charter.md). Created
2026-10-12. It contains no accepted scientific claim; every empirical sentence
is tagged with its evidence status and every placeholder is a visible `\todo{}`.

## Build

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -outdir=build main.tex
```

For Overleaf, import the GitHub repository and select `paper/main.tex` as the
main document. The repository's skill discovery entries are regular files
because Overleaf's GitHub importer rejects symbolic links.

Requires a TeX Live with `natbib`, `cleveref`, `booktabs`, `soul`, `xcolor`,
`hyperref`, `enumitem` and `plainnat.bst` (TeX Live 2025 was used). Delete
`paper/build/` before committing; build artifacts are not tracked.

Without a toolchain, run the structural check (balanced braces, matched
environments, existing `\input` targets, cite keys present in the snapshot
bibliography) with a script such as `/tmp/rc-paper/check_tex.py` from the
session that created this skeleton; the check is not part of the repository.

## Layout

| File | Content | Canonical source |
| --- | --- | --- |
| `main.tex` | class, packages, theorem environments, `\todo{}` and `\evidence{}` macros, section inputs | this README |
| `references.bib` | snapshot copy of the canonical bibliography | `doc/literature/references.bib` (regenerate by copying; never edit here) |
| `sections/00-abstract.tex` | conditional abstract with bracketed placeholders | charter §1, §6, §9 |
| `sections/01-introduction.tex` | motivation, RQ, why dimension is insufficient, contributions with status tags | charter §1–§3, §6, §10; novelty review §1, §4, §7 |
| `sections/02-related-work.tex` | grouped antecedents; each paragraph ends with the delta to demonstrate | novelty review §2–§3; bibliography; charter §7 |
| `sections/03-setting.tex` | notation, complete state, discrete framework, Floquet applicability, finite-time propagators, estimands | `research-program.md`; `periodicity-analysis-v1.md`; charter §4 |
| `sections/04-theory.tex` | statements P1–P11, fixture coverage, explicit OPEN Gate C target (P12) | `doc/auto/theory-propositions.md`; novelty review §3; `periodicity-research-plan.md` §2 |
| `sections/05-task-and-design.tex` | task, intervention, arms, horizons, gates, three-phase battery, interventions (a)–(d), noise assays, controls incl. optional commuting-null SSM control, statistics (proposed values from design v2 §2, §5), preregistration | charter §2, §5; `doc/auto/experimental-design-v2.md` §2, §5; `competence-recovery-plan.md` |
| `sections/06-results.tex` | strictly the verified evidence state; tables of v1 native errors, probes, geometry; probe-floor and FP32-closure caveats; pending periodicity reduction; unverified legacy observations | `initial-phase-results.json`; `initial-phase-scientific-review.md`; `periodicity-analysis-status.json`; `implementation-audit.md` §4–5 |
| `sections/07-discussion.tex` | interpretation matrix, alternatives, meaning of each outcome | charter §3, §11; `periodicity-research-plan.md` §4 |
| `sections/08-limitations.tex` | limitations, including the probe feature-scale floor | scientific review; novelty review §4; `implementation-audit.md` §4–5 |
| `sections/09-reproducibility.tex` | commits, jobs, hashes, artifact locations, environment, seeds, state shapes, numerical settings, determinism/TF32 status of v1 training | `initial-phase-report.md`; `periodicity-analysis-launch.json`; progress report 2026-10-11; `implementation-audit.md` §4–5, §8 |
| `sections/A-proofs.tex` | complete proofs of P1, P2, P5, P6, P7; sketches of P4, P8, P9, P10; P3 example; P11 computation | novelty review §3; `mechanism_calibration.py` constants |
| `sections/B-inherited-draft.tex` | one-page table: May 2025 draft assertions and their replacements | `floquet_analysis/main.tex`; `source-inventory.md`; `graph-reproduction.md` |
| `sections/C-protocol.tex` | motif library, matching, competence ladder, capture points, tolerances, fixtures, graph gate | `competence-recovery-plan.md`; `periodicity-analysis-v1.md`; `vector_task.py` |

## Placeholder and evidence policy

- `\todo{plain text}` renders highlighted. Arguments must be plain text (no
  `\cite`, no math). Every `\todo` names the artifact or review that will fill
  it. A manuscript with any remaining `\todo` is not submittable.
- `\evidence{status}` tags the status of a claim inline. Allowed values:
  `analytic`, `infrastructure`, `executed-verified`, `executed-unreduced`,
  `inconclusive`, `attributed-unverified`, `protocol`, `planned`,
  `conditional`. Remove tags only when converting to the submission version,
  and only after the status is `executed-verified` plus independent scientific
  acceptance.
- Every number traces to a committed `doc/auto/` artifact or to the
  independent scientific review (`doc/auto/initial-phase-scientific-review.md`).
  Table values in `06-results.tex` come from `doc/auto/initial-phase-results.json`.
  Hashes and job IDs come from `doc/auto/*.json` and the dated progress report.
  Nothing from the pending periodicity reduction, from Qwen proposals, or from
  the inherited draft's figures is reported as a result.
- Non-claims (charter §6) are never written as claims; "novel" and "first" are
  not used unless the novelty audit supports them.

## Inherited sources consulted

- `floquet_analysis/main.tex` (May 2025 draft): lines 66–140 (introduction,
  periodic task definitions), 145–220 (ODE surrogate and assumed periodic
  solution), 295–345 (discrete monodromy product), 437–565 (AABB prediction,
  six multipliers, envelopes, linearization errors), 675–750 (PSD and PCA
  interpretation). Treated as unverified attributed observations; corrections
  are in Appendix B.
- `ACM___TGRAB_KDD_2026/kdd/main.tex`: entrypoint lines 327–354; active
  includes `kdd/tex/000abstract`, `010intro`, `020related`, `030prelims`,
  `040method_ai`, `050periodic_ai`, `060cause_ai`, `070long_ai`,
  `080discussion_ai`, `085conclusion`, appendix `100appendix`. The periodic
  task definition `Pdet(k,n)`/`Psto(k,n)` was read from `050periodic_ai.tex`
  lines 1–40; other section contents were taken from `doc/auto/source-inventory.md`
  and `doc/auto/graph-reproduction.md` rather than reread in full.
- No inherited file was modified.

## Checklist: skeleton to submission (after Gate B evidence)

1. Independent raw-array reduction of the periodicity batch accepted; fill the
   five `\todo` items in §6.4 or delete the subsection if superseded.
2. Competence ladder passed in both arms for the Gate B cells; replace Table 1
   with the competent-model native errors and add the per-coordinate gate
   table; keep the v1 failures in an appendix or supplement.
3. Preregistered `δ2`, `δ16`, `DID` with paired seed-level intervals, Holm
   correction and the declared smallest effect; write the Discussion against
   the interpretation matrix without renaming outcomes.
4. Fill the conditional abstract and contribution sentences only from accepted
   evidence; otherwise convert the paper to the null/counterexample form per
   charter §11.
5. Replace the OPEN Gate C paragraph with a stated result (assumptions, proof
   or counterexample, added prediction) or move it to future work.
6. Remove every `\evidence{}` tag whose status is `executed-verified` and
   accepted; delete any sentence whose tag is still `planned` or
   `executed-unreduced`.
7. Update `references.bib` by copying the canonical file; add missing
   antecedents (for example ChaosBook cycle stability) to the canonical
   library first.
8. Regenerate Section 9 from the final `doc/auto/*.json` records; verify every
   hash and job ID against them.
9. Author list, affiliations, funding, compute acknowledgement and T-GRAB
   coauthorship statement.
10. Run the build, check for overfull boxes in hash strings, and remove
    `paper/build/`.
