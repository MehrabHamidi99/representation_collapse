# Evidence workflow

## Design and freeze

Start from `docs/PROTOCOL.md` and a manifest copied from
`configs/plan.example.json`. Bind the full source commit and protocol SHA-256.
Declare node ownership, dependencies, output paths, cost reservation and stop rule.
Use independent seeds/sampling units, matched controls, predetermined estimands,
selection boundaries and failure categories. Keep pilot tuning separate from final
held-out evidence. Cite primary papers and distinguish their claims from your own.

Run `python3 scripts/research.py validate --plan configs/plan.json`.
The helper rejects cyclic graphs, oversubscribed declared envelopes, nonfinite
values and missing provenance. A valid plan is not itself permission to submit.
Actual allocation/model-cost accounting remains the executor's responsibility.

## Advance a task

```bash
python3 scripts/research.py ready --plan configs/plan.json --state runs/campaign/state.json
python3 scripts/research.py record --plan configs/plan.json --state runs/campaign/state.json \
  --node calibration --status running --reason 'Inputs checked; one bounded calibration'
```

Execute that node through the current session's tools or its existing admitted
executor. Record each attempt immediately; preserve logs and unconfirmed effects.
The helper does not execute commands. For cluster work append a submission ledger
at dispatch, not only at completion. Store actual source/config/data/environment
hashes and allocation metadata alongside the node outputs.

Each declared output is a JSON object with `project`, `node_id`, `source_commit`,
`protocol_sha256`, `state=succeeded`, and the task's metrics/evidence. Preserve
finite measured values with units; omit unavailable metrics or mark them explicitly
unknown. `source_commit` is the actual full Git commit. The standalone demo alone
uses a script SHA-256 and labels it as infrastructure calibration.

For monitored jobs producing scientific results, report interpretable values to W&B
under the [reporting guideline](../../../docs/October/2026-10-26/cross-cluster/guidelines/results-reporting.md). Include
baseline, comparison and evaluation numbers with their interpretation
and result status. Exclude preprocessing, progress-only and infrastructure tasks.
Verify delivery for qualifying jobs; retain a pending upload queue when unavailable.

```bash
python3 scripts/research.py record --plan configs/plan.json --state runs/campaign/state.json \
  --node calibration --status verified --attempt runs/campaign/calibration \
  --reason 'Artifacts complete, identities match, independent numerical checks passed'
python3 scripts/research.py record --plan configs/plan.json --state runs/campaign/state.json \
  --node calibration --status accepted --reviewer scientific-review \
  --reason 'Reviewed the frozen criterion and limiting-case checks; evidence supports calibration only'
```

Verification checks provenance and hashes; the review must actually inspect the
science. A reviewer identity is an audit field, not proof of an independent person
or a model's authority. Use `research-review` and preserve its written rationale.
Only accepted prerequisites release dependents. Rejection/failed attempts stay
terminal; follow-up work requires a new plan/journal preserving the earlier costs.

For `submission_unknown`, retain full reservation and reconcile the scheduler;
do not call the submission again. Recording a resolved state requires a named
reviewer and an evidence-bearing reason. The helper never cancels jobs.

## Hand off

Use `docs/notebook/TEMPLATE.md`: question, input identity, commands, actual result,
failed/negative evidence, cost, limitations, decision and next task. Update
`docs/TRACKER.md` and `paper/claims.json` only to the degree the evidence supports.
Include the W&B destination and verified result links or outstanding reporting work.
For changed reusable operations, inspect the linked template's sync queue. Keep
scientific permissions and project conclusions out of portable skills.
