# Literature and novelty audit

**2026-10-11:** 90 distinct references; all 90 have a public PDF copy saved locally. The main report is about 3,800 words and was checked by independent theory, architecture and experimental reviewers.

Start with the [critical novelty report](novelty-review.md). Its verdict is that the broad collapse narrative is heavily anticipated; the candidate contribution is **specified, useful-history loss under controlled repetitive forcing**, with explicit analytic nulls and successful categorical-memory counterexamples. Five open questions, their closest antecedents, decisive tests, falsifiers and staged research gates are included. These are research recommendations, not claims of completed experiments.

## Artifacts

- [Annotated bibliography](bibliography.md): all 90 works, relevance, novelty boundaries, evidence levels and local PDF links.
- [BibTeX bibliography](../../../../../doc/literature/references.bib): importable citation records.
- [Reference JSON](../../../../../doc/literature/references.json): canonical metadata, aliases, sources and version qualifications.
- [Search log](search-log.md): scope, queries, verification, access issues and known omissions.
- [Download manifest](../../../../../doc/literature/download-manifest.json): source URL, retrieval errors, bytes, pages, hashes and identity checks.
- [Local paper directory](../../../../../literature/papers): 90 PDFs, approximately 188 MiB. Extracted text is in `literature/text/`.
- Independent evidence reviews: [theory](../theory/evidence/theory-review.md), [memory and experiments](evidence/memory-review.md), [architectures](evidence/architectures-review.md), [graph/methods](evidence/graph-methods-review.md).
- [Validation record](../../../../../doc/literature/validation.json): final library consistency checks and their limits.

The corpus contains **26 records checked in substantive full-text sections, 60 at abstract level, and four at metadata level**. Those categories describe reading depth, not retrieval success. No paper's empirical results or entire proof set was reproduced. This is a comprehensive scoped bibliography with a dated search boundary; it does not prove that all relevant work has been found.

## Suggested first reading

1. [T-GRAB](bibliography.md#dizaji2025tgrab) and [Arroyo et al.](bibliography.md#arroyo2025bridging): the public graph predecessor and the existing GNN–recurrence–Jacobian bridge.
2. [Pals et al.](bibliography.md#pals2024-phase-locked-memory) and [Ashwin–Ceni](bibliography.md#ashwin2023-echo-repetitions): periodic Floquet memory, multistability, and repetition-sensitive driven responses.
3. [Ganguli et al.](bibliography.md#ganguli2008-memory-traces), [Sharma et al.](bibliography.md#sharma2026-temporal-superposition) and [Fokam et al.](bibliography.md#fokam2026-temporal-correlations): task/noise-sensitive memory and correlation alternatives.
4. [MeSH](bibliography.md#mesh), [Labovich](bibliography.md#looped_stability_recall) and [Think Shallow, Solve Deep](bibliography.md#think_shallow_solve_deep): closest depth-recurrence overlaps.
5. [All Routes Lead to Collapse](bibliography.md#all_routes_collapse) and [Chehreghani](bibliography.md#chehreghani2026noise): broad architectural scope and the distinction between anti-collapse geometry and useful memory.

## Rebuilding the library

Edit the source records in `evidence/*-references.json`, then regenerate the canonical outputs:

```bash
python3 code/literature/build_library.py
```

Retrieve available public PDFs and extract text without changing the project environment:

```bash
uv run --no-project --with pypdf --with fonttools code/literature/build_library.py --download --workers 3
```

To update an inspected version, first change the source record, then use `--download --refresh --only CITATION_ID`. Run one builder process at a time. The default reuses cached PDFs; merely changing a candidate URL does not replace an existing file. The helper is a retrieval/provenance utility, not an automated scientific reader or an unattended literature-monitoring service. Title screening is heuristic; hash-bound manual exceptions are recorded separately.

PDFs, text and render caches are ignored by git. The bibliography, annotations, retrieval manifest and reports are tracked-source artifacts intended for the repository. A fresh clone must retrieve the PDFs again. Some copies are author manuscripts or inspected older preprint versions; read the version notes before citing publication page numbers. Jaeger and Bengio have imperfect text extraction; inspect their PDF pages directly for detailed reading.
