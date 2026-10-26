---
name: research-wandb
description: Set up and verify Weights & Biases for jobs requiring monitoring and producing scientific benchmarks/results, including training, evaluation, baselines and comparisons. Excludes routine preprocessing, progress-only tasks and infrastructure checks.
---

# Research experiment reporting

Create W&B runs only for jobs that both require monitoring and produce scientific
benchmarks or research results. Qualifying training, evaluation, baseline,
comparison and ablation jobs report interpretable values to the relevant
project/subproject from the beginning, including valid negative/partial results.
Record `wandb_required` and its reason in the attempt manifest or ledger.

Do not create W&B runs for routine preprocessing, downloads, tokenization, cache
building, progress-only tasks, orchestration/status polling, infrastructure smoke
tests or upload/verification tasks. Counts, timings and percentages alone do not
qualify. Keep those logs in dated documentation and mark W&B `not_applicable`.
Preserve existing W&B history and active job configurations. Attach aggregate
comparisons or repaired uploads to the original experiment's reporting lineage.

Initialize qualifying tracking before measured work; log live losses/validation,
final evaluation, relevant resource measurements, failure and completion status.

Use the owning project's W&B project/entity, existing authenticated account,
and source/config/data/split/seed/environment identities. Store the run URL/ID
in local evidence and the cluster ledger. Preserve resume lineage and monotonic
training steps; do not silently mix changed experiments into a previous run.
Keep true-group test metrics out of training and model selection.

If a compute node cannot reach W&B, start in explicit offline mode and retain
its durable W&B directory for authenticated sync. Verify remote history and
summary after sync; an offline log or successful process exit is not evidence
that the dashboard received results. Do not silently disable tracking. If
authentication is missing, keep local evidence and identify the remaining sync
step without exposing or copying credentials. Do not upload raw datasets,
credentials or full model weights unless required by the experiment's scope.

For implementation, use the official W&B SDK documentation and add a small
offline smoke test plus an authenticated live receipt when available. For qualifying runs, log
compact manifests/scientific calibration records as artifacts. Experiment reporting is
part of completing the run, not a later optional documentation task.

After SDK finish, allow a finite read-only reconciliation window for remote
state, summary, history and required artifacts, within the attempt deadline.
A completed local run can precede a finished remote state. Preserve terminal
failures and exact source/input/numeric checks; never turn missing reporting
into zero metrics or success. If numerical work is already complete and a
reporting check fails, first inspect retained outputs and backend delivery.
Prefer CPU verification/sync with local receipts and explicit original-run lineage
to repeating GPU work; do not create a new W&B run just for verification/sync. Keep the original failed attempt, receipts and costs.

Write documentation at `docs/MONTH/YYYY-MM-DD/CLUSTER/[SUBPROJECT]/TYPE/...md`
(English month name); use Git as the primary local/cluster synchronization mechanism.
