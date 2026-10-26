# Research results in W&B

W&B is required only for jobs that both need monitoring and produce scientific
benchmarks or research results. Qualifying training, evaluation, baseline,
comparison and ablation jobs report all interpretable results, including valid
negative and partial outcomes, under the relevant project/subproject.

Do not create W&B runs for preprocessing, downloads, tokenization, cache building,
progress-only tasks, orchestration/status checks, infrastructure smoke tests or
reporting/sync verification. Keep those logs and receipts in dated documentation.
Counts and timings alone do not qualify. Record `wandb_required` and the reason;
use `not_applicable` for excluded tasks. Preserve existing runs and pinned jobs.
See [documentation and reporting scope](documentation-layout.md).

## Destination and organization

Use the owning project's established W&B entity, project and subproject organization.
Keep baselines and comparisons discoverable alongside the methods they evaluate.
Record the destination in `project.json` and the experiment manifest; use the
existing grouping convention for studies, methods, seeds and attempts. Reuse a
known applicable destination without asking for approval again. Resolve a missing
or ambiguous destination before upload; do not use an unrelated project's namespace.

## What to report

Report the actual numbers as metrics, curves or structured tables, including
aggregate comparisons and evaluations performed after training. Identify metric
meaning, units, dataset/split, method/baseline, seed or sampling unit,
step/checkpoint and evaluation protocol. Include sample counts and uncertainty
where applicable; link aggregates to constituent runs. Preserve the project's
allowed metadata scope and retain full provenance locally when it cannot be published.

Distinguish preliminary, partial, failed and scientifically accepted results.
Report valid measurements from failed attempts with their failure status. Never
fabricate missing values, convert unknown values to zero, hide unfavorable
baselines, or overwrite earlier attempts with recoveries. Scientific acceptance
and W&B reporting status remain separate fields.

## Delivery and completion

Use stable run/attempt identities so retries and backfills do not duplicate results.
Verify that the expected values are visible in the intended W&B destination and
record run/report links in the dated notebook and tracker. A local write, offline
run or upload attempt is not confirmed reporting.

When W&B is unavailable, preserve durable local results and an upload queue, mark
reporting pending with the reason, and backfill when access returns. Continue
independent research while reporting is pending; do not mark the reporting
obligation complete until delivery is verified. Use the existing project reporter
when present, preserving its publication scope and active source pins. This
requirement concerns results and their interpretation; it does not require
uploading datasets, weights, source code or credentials.
