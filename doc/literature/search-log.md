# Search method, coverage and limitations

Cutoff and retrieval date: **2026-10-11**. This is a comprehensive **scoped** bibliography for the repository's question, not a systematic-review claim that every relevant work has been found. No fabricated PRISMA counts or priority guarantee are implied.

## Scope and procedure

The starting objects were the repository's preserved graph/Floquet manuscripts, source inventory, and proposed periodicity research plan. The coordinator covered graph antecedents, dimensionality measurement and cross-domain context. Three bounded independent reviews covered forced-system theory, memory/experimental interpretation, and modern recurrent architectures. They returned reference records, evidence locators, novelty objections and search notes before integration; a subsequent independent check reviewed the main report's mathematics and interpretation.

Discovery used web search, backward references in the closest papers, and targeted recent-work queries. Verification used primary arXiv records and full text; PMLR, NeurIPS, ICLR/OpenReview and ACL proceedings; PLOS, Nature, PNAS/PMC, eLife and IEEE records; and author/university repositories. Crossref and PubMed resolved identifiers and publication metadata. Secondary indexes served only as discovery aids. There was no comprehensive forward-citation export or subscription-database systematic search.

Included records either directly threaten the proposed novelty, establish a necessary theoretical or measurement foundation, supply a meaningful alternative explanation/control, or define a planned architecture/native-task baseline. Broad application papers with no connection to the mechanism were not accumulated merely to increase the count. Conference and preprint copies of the same work are merged by DOI, arXiv identifier or normalized title; original IDs remain as aliases. The four evidence registries contain 98 records, yielding 90 distinct works after merging eight duplicate records.

## Query families and source trails

Actual representative query families included:

- “Floquet recurrent neural network representation collapse periodic input memory” and phase-locked working-memory cycles; Pals → echo states, continuous attractors and low-rank dynamics.
- “Transitions echo index dependence input repetitions Ashwin Ceni”; repetition, symbol dwell, multistability and input-specific ESP.
- “Temporal superposition feature geometry RNNs memory demands” and “How Temporal Correlations Shape Memory Linear Recurrent Neural Networks”; targeted 2026 checks and theorem-scope reading.
- “Memory traces dynamical systems Ganguli Huh Sompolinsky Fisher”, noise and linear recurrent memory, information-processing capacity, slow points and finite-precision memory.
- “GNN recurrent Jacobian oversmoothing rank collapse”, “Persistent Gaussian Perturbations Prevent Oversmoothing in Recurrent Graph Neural Networks”, T-GRAB and its temporal-graph task definitions.
- Recursive/looped transformer collapse, test-time recurrent depth, Lyapunov diagnostics, MeSH, Huginn, input recall, fixed-point reasoners, LoopUS and “All Routes Lead to Collapse”.
- S4, Mamba/Mamba-2, RWKV-7, xLSTM, Gated DeltaNet, Block-Recurrent Transformers; exact titles and official metadata checks.
- “The Effective Rank: a Measure of Effective Dimensionality”, “A theory of multineuronal dimensionality, dynamics and measurement”, recurrent spiking-network dimension and “Strong and localized recurrence controls the dimensionality”.
- Exact-title public-copy searches for classic papers, including “Learning long-term dependencies with gradient descent is difficult”.

The three independent review files preserve their more detailed query logs and access limitations. This list records search families rather than asserting a deterministic search-engine result set. Search engines and current arXiv landing pages can change after this date.

## Evidence and metadata checks

The canonical library has **26 full-text-section records, 60 abstract records and four metadata-only records**. Relevant section/page locators accompany substantive readings. A full-text label does not certify every proof or code path. Architecture QA compared title and ordered author metadata against **63 unique arXiv records**; five close modern works received additional provenance checks. Source initials were retained where expanded names were unverified. Missing DOI or venue information is explicit.

Important version qualifications:

- T-GRAB and Arroyo use inspected v2 PDFs; the noisy recurrent-graph paper uses v1.
- MeSH, Labovich and Huginn were initially read at v1 while current unversioned links resolve to newer versions. Their downloaded copies were refreshed to the inspected v1. MeSH acceptance and relevant router/spectrum details were additionally checked against v2. Huginn's downloaded v1 title page was checked against its canonical metadata.
- Ashwin–Ceni, Guan and Fokam use pinned inspected preprints. A proceedings record and a preprint need not have the same publication year.
- Jaeger's short-term-memory report is often indexed as 2001. The rendered title page gives a 2002 manuscript and publication note; the library retains 2002 and explains the discrepancy.
- The Bengio 1994 copy is an institutional author-manuscript copy. The Dahmen 2026 copy is an author-hosted manuscript dated August 13, preceding the August 19 published version; it is not the typeset publisher PDF.
- DEQ's NeurIPS 2019 venue was verified; the accepted selective-SSM paper and its arXiv metadata use different legitimate forms of one author's name, recorded as an alias.

## Retrieval and validation

All **90** works have a local PDF copy (approximately **188 MiB**, **2,267 PDF pages**). These counts include appendices and do not measure how much was read. Download candidates were restricted to available public publisher, proceedings, arXiv, institutional or author copies. PDF signatures and page parsing were checked; each file has a source URL, byte count and SHA-256 hash. First-two-page title-token matching is a screening heuristic, not proof of identity. Exceptions received rendered-page inspection with hash-bound notes in [manual-identity-checks.json](manual-identity-checks.json).

Several initial links returned HTML, an access error, or a certificate failure. Alternate legitimate copies resolved all entries: Groningen for Jaeger, Ghent for Dambre, eLife's publisher CDN for Koyluoglu, PLOS for Hu, Columbia for Rajan, NYU for Orhan–Ma, CMU for Bengio, and a coauthor's laboratory site for Dahmen. The manifest retains failed attempts. No authentication barrier or paywall was bypassed.

Text extraction is a convenience cache, not the authoritative paper. Jaeger's report has broken font encoding and Bengio's manuscript has fragmented extraction; both require rendered pages for close reading. Some figure-heavy PDFs can omit graphical text or hit an extractor's form limit. Formula-dependent arguments should be checked against rendered PDF/primary HTML, not trusted solely to extracted text. PDF bytes and the provenance ledger remain authoritative.

The library was built with the repository helper using ephemeral `pypdf` and `fonttools` dependencies; no project training environment or lockfile was changed. Only one build process should write its manifest at a time. Use `--only` for selected retrievals and `--refresh` to replace a cached copy after changing its version pin.

## Known gaps and next search obligations

This audit is strongest on the intersection of repeated forcing, recurrent memory, graph/transformer collapse and task-sensitive dynamics. It is not a complete survey of biological repetition suppression/habituation, all cognitive working-memory models, continual-learning collapse during repeated training, chaotic synchronization theory, control-theoretic observability, every intrinsic-dimension estimator, or every native audio/video application. Those settings have different causal objects and need dedicated extensions if they enter the paper's claims.

Before a novelty statement is frozen:

1. Extend backward and forward citation searches from Pals, Ashwin–Ceni, Arroyo, Sharma, Fokam, MeSH, Labovich and Viakhirev; check subsequent versions and venue status.
2. Read the remaining relevant Proca/Bordelon and low-rank/ESP proofs against the exact proposed assumptions; do not cite abstract-level records as proved support.
3. Inspect the implementation and precise null semantics of All Routes before escalating the conditional permutation critique into an implementation claim.
4. Audit unincorporated architecture leads flagged by the architecture reviewer (sphere-constrained Mamba, EvoEmbedding and DeepLoop) before making an exhaustive architecture-level claim.
5. Add native-task/data references when those specific experiments are selected; this bibliography is not a dataset license or training-data audit.
6. Recover graph artifacts and reproduce predecessor evidence. Literature proximity is independent of whether this repository's inherited graphs are reproducible.

A bounded negative search result licenses “not found in the inspected sources,” never “no prior work exists.”
