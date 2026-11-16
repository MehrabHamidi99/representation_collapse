# Project status and public research note — 2026-11-16

## Source and evidence refresh

An HTTPS fetch of GitHub `main` confirmed the local and remote project at
`b45b354d25731e773aea7dc88dc10f2c877fbe16`. There were no incoming commits or
newer committed experimental results. The October 26 changes reorganized
documentation and reporting policy. The latest accepted experiments remain
the October 23 one-bit extension and the preceding graph/one-bit results.
This is a repository/evidence refresh, not a live cluster or worker audit.

Inspected the canonical tracker, latest dated progress report, graph long-fit
and dwell-sweep reports, one-bit reports, compact numerical outputs and theory
propositions. Public claims are restricted to those artifacts. The graph sweep
reports a fixed-checkpoint mean-F1 decline from 0.889429 at dwell32 to 0.656541
at dwell256 under a changed temporal distribution. The one-bit extension
reports perfect native accuracy in both arms through6144 body steps. Neither
establishes capacity loss, a universal collapse mechanism or a trained-model
Floquet spectrum. Several-bit tasks, more seeds, timing controls and tangent
analysis remain future work.

Evidence integrity checks verified all19 dwell inventory entries, resolving
`numbers.md` through `documentation-index.json`: its old path is now a redirect,
while its canonical migrated file matches the original manifest hash. The
other18 files match at their original artifact paths. All6 long-horizon
manifest entries match. No original evidence or manifest was rewritten.

## Blog and validation

Title: **Does repetition make a recurrent model forget?**

Public URL: https://mehrabhamidi99.github.io/blog/repetition-and-memory/

Editable source:
https://github.com/MehrabHamidi99/MehrabHamidi99.github.io/blob/main/content/blog/2026-11-16-repetition-and-memory.md

Website main commit: `23f1707c921f9da54299f94527c1adfb3aee34e9`.
The post is published with `draft: false`, dated2026-11-16. It includes two
figures regenerated directly from verified compact results, accessible image
descriptions/full-size links, fixed-revision source links, theory assumptions,
limitations and future experiments. A news entry points to the post.

The authoring clone is `/private/tmp/representation-collapse-blog-20261116`;
the durable source is GitHub. No profile or publication list was changed.
The website's existing Python build and link checker passed:9pages,9publications,
5news entries,3posts, local links/assets/anchors/metadata/RSS/sitemap valid.
Desktop1280px and mobile390px screenshots were inspected; both images load,
and document scrollWidth equals viewport width. The connected Browser runtime
had no available browser, so isolated headless Chrome previews were used.

One bounded read-only scientific reviewer checked the draft against the
accepted reports and theory, accepted its claim boundaries, and returned no
scientific blocker. Its suggested removal of an unrelated operational phrase
was applied. It is completed; no persistent subagent service was started.

Reproducible figure script: `code/analysis/plot_blog_summary.py`.
Inputs/output hashes: [plot provenance](blog-assets/plot-provenance.json).
W&B eligibility: **not_applicable**. This task re-presents existing evidence
and publishes documentation; it is not a monitored scientific experiment.
No new job, training, test evaluation, Qwen call or cluster synchronization
was performed. Earlier offline W&B delivery remains pending as documented.

## Deployment

The source push triggered the existing Deploy site workflow:
https://github.com/MehrabHamidi99/MehrabHamidi99.github.io/actions/runs/43966292817

The Deploy site workflow succeeded. GitHub Pages subsequently published the
generated branch. At22:07UTC the article, both figure URLs and the news page
returned HTTP200 and their SHA-256 hashes exactly matched the checked local
build. See [live verification receipt](publication-verification.json).

The project README and canonical tracker were updated to include the latest
accepted memory results and this public note. The README's stale claim of a
live Qwen worker was replaced with its recorded shutdown and an explicit
statement that current cluster liveness was not checked in this publication task.

## Floquet revision and citation — November16 continuation

The owner requested stronger coverage of Floquet multipliers and both supportive
and conflicting experiments. Website commit `28e0a78d1f36c8cb8fd3b86c79455e21bfa17d27`
publishes **Does repetition erase memory? A Floquet view** at the same permanent URL.
It now develops the discrete monodromy product, per-period versus per-update
rates, forced rhythms with subunit multipliers, exact continuous-memory neutral
directions, categorical attractors, ordering nulls and the analytic folding
counterexample. These are established results and elementary deductions with
explicit assumptions, not a new universal theorem.

Added the recovered six subunit stored eigenvalues and repeated envelope-row
audit, together with the failed-competence finite-time tangent pilots. Neither
is promoted to certified trained-model Floquet evidence. The accepted graph
and one-bit findings and their limitations remain. The post includes a plain
citation and BibTeX with author, title, date and stable URL; no DOI or peer-review
status is invented. A bounded writing agent prepared the revision; the
coordinator checked its theory and quantitative claims against the linked
sources and corrected the mobile citation link's overflow.

The Python build and site checker passed. Desktop1280px and mobile390px render
with loaded images and no document-wide horizontal overflow. Deploy site run
43970086222 and Pages run43970165632 succeeded. The public article and news
page returned HTTP200 and matched the local build byte-for-byte; the citation
is present in the served HTML. See [revision verification](publication-verification-v2.json).

An authenticated GitHub repository lookup and a separate anonymous lookup both
confirmed `MehrabHamidi99/representation_collapse` already had public visibility.
No visibility change was necessary; the owner subsequently confirmed it.

The owner also requested higher-dimensional follow-up experiments in this
continuation. Those have a separate prospective protocol and submission record;
they do not change the completed October evidence reported in the blog. The
goal tool refused a new goal because the thread has an unfinished earlier goal;
no new automatic goal was armed or earlier scientific objective marked complete.

The follow-up request about **largest eigenvalues** is published in website
commit `ea3aefa`. A dedicated section distinguishes spectral radius from the
largest singular value and covariance eigenvalues; explains chronological JVPs,
Arnoldi, reverse VJPs, eigenpair residuals and partial-spectrum limitations; and
uses an explicitly analytic nonnormal example with radius0.8 and maximum
one-step singular gain4.154. These are measurement methods and a mathematical
example, not a newly measured GC-LSTM spectrum. The served page includes this
section and the citation and matches the local build exactly; see
[third revision verification](publication-verification-v3.json).
