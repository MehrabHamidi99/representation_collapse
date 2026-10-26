# Documentation, W&B reporting and Git synchronization

Effective 2026-10-26. This policy supersedes older path conventions and broader
requirements to create W&B runs for every task with numerical output. Scientific
protocols, publication restrictions, budgets and immutable attempt pins still apply.

## Documentation paths

Write substantive project documentation under
`docs/MONTH/DAY/CLUSTER/[SUBPROJECT]/TYPE/...md`.
Use the English month name and an ISO date, e.g.
`docs/October/2026-10-26/tamia/jepa/experiment/ablation.md`.
The ISO day includes the year to prevent collisions. Use the project's timezone
(default America/Toronto); retain original timestamps and timezone in the document.

CLUSTER names the subject cluster, even when written on a laptop. Use `local` for
local-only work and `cross-cluster` for shared guidance or work spanning clusters.
Omit SUBPROJECT when inapplicable; otherwise use the owning project's stable slug.
TYPE is `theory`, `experiment`, `report`, `auto-research`, `submitted-jobs`,
`guidelines`, `literature` or `planning`. Put submission ledgers and receipts in
`submitted-jobs`; document autonomous research decisions in `auto-research`.
Keep an existing document at its canonical dated path when updating it; a new
report gets a new date. Do not duplicate current guidance into every day folder.

README.md, AGENTS.md and SKILL.md remain discovery/agent entrypoints. Skill-bundled
references and scaffold inputs remain packaged with their consumers; generated
project notes use the dated layout. Raw evidence, imported sources and immutable
execution snapshots retain their provenance paths. They are not current guidance.

Existing doc/ and docs/ Markdown is migrated with references rebased, including
relative figure/asset links. `documentation-index.json` records every old/new path,
hashes and date attribution. Unknown dates use the migration date explicitly;
legacy month/day paths record any inferred year. Old locations contain small
regular-file redirects. Where executable provenance contracts require exact bytes,
retain a frozen legacy copy listed in the index and edit only the canonical copy.
No symlinks are introduced. Frozen copies must not be used as new mutable notes.
Run `python3 scripts/documentation.py --project .` to preview additional migration;
add `--apply` after inspecting the plan. New writers must select the canonical
path directly; do not overwrite a compatibility redirect with a fresh report.

## W&B scope

Create W&B runs only for jobs that **both require monitoring and produce scientific
benchmarks or research results**. Training, evaluation, baseline, comparison and
ablation jobs meeting those criteria report all interpretable result values to
the relevant W&B project/subproject, including valid negative and partial results.
Attach aggregate comparison tables to the relevant experiment/report; do not create
a separate run merely to upload or verify existing results.

Do not create W&B runs for preprocessing, dataset downloads, tokenization, cache
building, progress-only tasks, orchestration, status polls, infrastructure smoke
tests or reporting/sync verification. Counts, percentages, elapsed time and health
checks alone do not make such tasks research experiments. Keep their progress,
validation and receipts in local logs and dated Git-synchronized documentation.
A dedicated experiment studying preprocessing scientifically is classified by its
actual benchmark objective; routine data preparation remains excluded.

Before a new job, record `wandb_required` and the reason in its manifest or ledger.
For qualifying jobs, use the established entity/project/subproject and run identity,
initialize before measured work, log live metrics and final results, and retain
source/config/data/seed provenance, metric definitions, units and attempt status.
Verify remote history/summary and record run links. When unavailable, retain durable
offline records and mark delivery pending until verified. Do not rerun GPU work just
to repair reporting. For excluded tasks record `not_applicable` rather than a missing
W&B result. Preserve existing runs/history and active job configurations.

## Synchronization

Git is the primary synchronization mechanism for source, configuration, skills,
guidelines and all current documentation across local and cluster checkouts.
Commit and push intended changes on the owning branch; fetch and inspect the
receiving checkout, then fast-forward a clean documentation/development branch.
Return cluster-authored documentation through Git too. Never overwrite dirty or
divergent work or move queued/running source pins. Stage new runs from the exact
committed revision. Cluv staging and artifact transfers do not replace Git history.

Record the pushed commit and each receiving cluster's verified commit or pending
reason. A local edit or push alone is not verified cluster synchronization. When
access is unavailable, leave delivery explicitly pending; use existing authentication
or a verified Git bundle when an allowed route becomes available. No automatic
merge, force push, job submission or new compute authority follows from this policy.
