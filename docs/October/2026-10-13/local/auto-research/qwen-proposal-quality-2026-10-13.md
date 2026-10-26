# Qwen proposal quality and task-binding audit

2026-10-13. Native OpenAI fallback review of the six actual Qwen replies through
10:07:33 UTC. **None of the six complete proposals is recommended for scientific
acceptance or execution.** Several factual statements and diagnostic components
are useful and are retained below. This audit does not withdraw the separately
verified native acquisition results or authorize any new experiment.

The immutable continuous worker is source
`999d88d8679c0f353c6bb7fb4f5ab8433a7e730c`, worker SHA-256
`8d9e6149f83a91cefdf8cd51c7e38777cdfade4c73cfb42136f104d7ea45a606`.
All six replies passed schema and exact requested/returned model identity checks,
reported zero executed actions and remained `scientific_decision=pending`.
They report **29,862 tokens**. Calls 2–5 consumed four coordinator task IDs;
calls 1 and 6 were evidence-change responses. Those machine counters establish
provider participation and request consumption, **not four answered or accepted
scientific tasks**. No Qwen-derived scientific claim is accepted here.

The audit reads the sanitized local status/evidence/review/usage files under
`runs/ops/qwen-continuous-deployment-20261013/reviews/`. For every call, the
recorded evidence-items hash recomputes exactly, matches the review's evidence
binding, and every cited identifier exists in the actual supplied evidence.
Explicit task IDs agree between their evidence record and call status. This
checks provenance; a valid identifier does not establish that its cited claim
is correct or that the answer addresses its task.

The machine-readable independent audit at
`runs/qwen-task-binding-fixtures/proposal-quality-audit-v2.json` retains all 24
status/evidence/review/usage file hashes, task texts, decisions, reasons and
prospective prompt sizes. SHA-256:
`021730bea2a1f9d127af93cb276108307fe15f2aa6b488cced0ee28e9cba0e05`.
The per-call hashes below identify the actual `review.json` bytes and supplied
evidence-items object, not a newly generated answer.

## Call 1 — unassigned evidence review

Finished **09:56:12 UTC**, trigger `evidence_changed`, task ID null.
Review SHA `d7947dcb72d8a5b77960dbfc60859bcfc9cbcfc297f6dfff88d5b9b6d3d17a2e`.
Evidence SHA `24f5316bf57681ff33c9677f7450e940fb291dbe38f4a41e31ba8312df725a5e`.

**Complete proposal rejected. Useful component retained:** the supplied v2
summary establishes scoped acquisition failure before periodic-body training;
it does not establish harmful collapse. The initial feed only prepared v3,
so this reply is not faulted for failing to know later v3 execution evidence.

The auxiliary-R0-loss proposal leaves its construction, weight and comparator
unspecified despite R0 training already optimizing recall. Reaching stage 5
in two of four fits is not the frozen all-coordinate checkpoint-eligibility
rule. Its categorical accuracy stopping rule is not the continuous normalized
MSE gate. Four failures after an unspecified modification would not exclude
loss weighting as a contributor. Calling the full-prefix failure simply
retention does not separate writing, interference, binding and native readout.

## Call 2 — acquisition ordering controls

Finished **10:03:16 UTC**, task ID
`acquisition-order-controls-20261013`.
Review SHA `cd12c9b26df4481f946e51597dceccccf6e23d13fd178afcf486ec423fd64668`.
Evidence SHA `cfb360c2f628202df90802556cade94d66fe1f204eb6d730658100f4d9df1bdf`.

**Assigned answer and complete proposal rejected. Useful component retained:**
a failed acquisition checkpoint cannot supply the intended competent-checkpoint
retention comparison. The task explicitly requested controls separating QUERY
support/memory demand, target WRITE position/lag and nuisance-WRITE interference,
and explicitly supplied the eight failed 5,000-update v3 fits.

The answer instead repeats immediate-500/full-prefix-500 training, already a
weaker exposure than the failed v3 comparison, without the requested matched
controls, counterbalancing or possible-outcome/confound analysis. It applies
accuracy gates generically to continuous tasks and mixes an update-100 stop
with update-1000 acceptance. Its dismissal of the current evidence as only
engineering calibration is also too broad: the verified learned-model failures
are empirical acquisition evidence conditional on the frozen schedule and
observed seed, although they are not periodicity-harm evidence.

## Call 3 — Floquet scope and falsification

Finished **10:04:20 UTC**, task ID
`floquet-scope-falsification-20261013`.
Review SHA `b721eedfac2e04f3bfded47d0157d953a90da8837142b23edd878fa805b4c5a1`.
Evidence SHA `edf5af0d3702b7639b5224733698abcdf077e2dd6fd5e48100079cfb9759a8e0`.

**Assigned answer and complete proposal rejected. Useful component retained:**
there is no eligible trained checkpoint for the planned retention claim.
That does not prevent the requested analytic scope/falsification work.

The reply omits orbit/Jacobian closure, phase controls, complete graph/h+c
domains, nonnormal counterexamples, a falsifiable prediction and an analytic
null. It instead relaxes accuracy eligibility from 0.95 to 0.80. Relaxing a
criterion changes acceptance and the selected population; it does not increase
model capacity or repair the original acquisition failure. This was not the
assigned question and cannot amend a frozen gate. Its negative stopping
interpretation also cannot establish a fundamental architecture/optimization
mismatch from four finite fits. See the unchanged
[Floquet measurement contract](../../../2026-10-26/local/auto-research/research-program.md).

## Call 4 — stored information versus native readout

Finished **10:05:23 UTC**, task ID
`memory-readout-identification-20261013`.
Review SHA `1ec4c1618b38f83f19721137db1efbfee8e20abde3530c66a8f94886a041150a`.
Evidence SHA `00a56935a96701e14900a22cd25a2f168bc4965688f1e3fac81cbb43b7b331b3`.

**Complete proposal and mechanistic conclusions rejected. Useful components
retained:** phase-matched frozen-state probes, shuffled-label controls and
precision/floor diagnostics are appropriate prospective ingredients. This reply
does address part of the assigned topic, unlike calls 3 and 5.

The requested fixed-weight diagnostic is replaced with new training and a GPU
request, despite available failed checkpoints being usable for descriptive
readout/acquisition diagnostics. Chance native performance plus chance linear
probe performance does not prove absence of information: nonlinear encoding,
finite samples, decoder optimization/regularization, feature floors and
precision remain alternatives. Improved held-out probe performance demonstrates
decodability for that probe/data/phase; it does not alone identify a causal
native-readout failure or the information used by the task. Its own limitation
acknowledges nonlinear encoding, contradicting the definitive loss stopping
rule. The design also leaves joint h+c analysis, train-fit/untouched probe
evaluation, equal-budget decoder controls and intervention/support caveats
incomplete. Casting a replay to FP64 alone does not certify absence of a native
readout or feature-floor problem.

## Call 5 — seed-level inference contract

Finished **10:06:29 UTC**, task ID
`seed-inference-contract-20261013`.
Review SHA `cf741aba9499db38ee0b502c5873c57afc761651476b50135efcb2909236b973`.
Evidence SHA `6763b7aabc2e86a34dbebddd450eea9e50d2321725a6fc2183c559e27d25aa55`.

**Assigned answer and complete proposal rejected. Useful component retained:**
the observed v2 failures do not establish periodicity harm.

The reply supplies no analysis of ten independent training seeds, paired
delta16/DID, mean-t assumptions, eight retained Holm slots, selected-checkpoint
population, missing/ineligible cells, unfrozen smallest effects or robust
sensitivity preserving the mean estimand. It substitutes a hidden-state-drift
regularizer and another 500-update seed-0 training plan. A finite observed
failure is not a general inability theorem, and passing an earlier stage is not
final checkpoint eligibility. The actual
[inference proposal](../../../2026-10-26/local/auto-research/statistics-freeze-proposal-v2.md) remains unfrozen.
Median/sign analyses would answer a different estimand; an alternative meant
to support the original mean needs its own stated range/tail assumptions and
coverage justification. None of this was supplied by the reply.

## Call 6 — updated v3 evidence

Finished **10:07:33 UTC**, trigger `evidence_changed`, task ID null.
Review SHA `f33757bf787f4f5a31b9a9ffcb5da669cff3d7e9ddaf2036afeda3d5c4b11a24`.
Evidence SHA `0980c8387c6e5e4d1e95682d6e99d2d0b607d22769ca2a701e22e7b787b8d7e8`.

**Complete proposal and full-gate assessment rejected. Useful component
retained:** the zero-body audit leaves periodicity harm untested and directs
attention toward prefix acquisition.

The fresh supplied feed explicitly says all eight full-coordinate gates fail.
The reply instead says most conditions fail and treats some GRU continuous/LSTM
categorical transfer conditions as passed. Those are **individual coordinate**
passes, not full-condition competence; GRU continuous coordinate passes also
occur in both fresh and transfer conditions. The
[verified endpoint ledger](prefix-acquisition-v3-results-2026-10-13.md) resolves
this distinction. More immediate-task updates are proposed without a matched
mechanism control even though ten v2 fits passed the immediate task and the
separate 5,000-update full-prefix comparison failed. A new discovery schedule
could be evaluated only as a separately specified prospective study; it cannot
silently edit or rescue the frozen v2/v3 cohorts.

## Local prompt repair and acceptance boundary

The old system prompt demanded a generic next experiment while placing the
explicit coordinator question inside untrusted evidence. The local
[worker repair](../../../../../code/ops/qwen_worker.py) passes the already validated inbox
task explicitly into request construction, places its full question first in
the user message, and requires `assessment` to answer that question. Analytic
checks, reviews and diagnostics are permitted in the unchanged
`next_experiment` schema. Unrequested job defaults, gate relaxation and repeated
failed schedules are prohibited. Probe failure and partial-coordinate success
receive explicit interpretation limits.

The question appears once. Its exact task ID/text must match the original
supplied evidence record; a compact pointer preserves the same citation
identifier in the evidence map. Other excerpts remain untrusted. If the new
headers exceed the original context budget, the longest serialized background
excerpt is shortened deterministically, with lexical tie breaking and an
explicit `background_excerpt_compacted` flag. All source identifiers, the full
question and its exact binding remain; only background text is compacted. The
caller evidence is never mutated. An arbitrary artifact or reserved-key excerpt does not become an
assigned task. The caller's full evidence map and its saved hash are unchanged;
the raw request records the deterministic foreground projection. This keeps
the approved data scope while making task priority explicit.

Reconstructed prompts for all six actual evidence packets remain within the
unchanged **18,000-character** limit: 17,390; 17,599; 17,968; 17,988; 17,573;
17,389 characters. Call 4 needs a small background-only compaction after the
clarified fallback instruction. All questions remain exact and appear once.
The prospective requests were saved locally, **not sent**. Response schema,
2,048-token cap, model identity, provider/endpoint handling, pacing/backoff,
bounded defaults, null continuous limits and stop semantics are unchanged.
The prompt now precisely distinguishes this leaf's absent external HTTP model
fallback from the available native OpenAI coordinator review; it introduces no
endpoint, credential or model-service call.
AST comparison against immutable 999d88d verifies every operational helper is
identical and the main loop differs only in supplying the validated task to
request construction. The running source 999d88d has not been edited, restarted
or replaced by this repair.

The [new offline tests](../../../../../tests/test_qwen_task_binding.py) check actual inbox
to prompt binding, exact question/ID integrity, evidence immutability, task
priority, untrusted excerpt handling, the unchanged analytic-response schema
and the 1,500-character task/18,000-character context boundary. A deliberately
irrelevant schema-valid mock reply still increments the provider-consumption
counters while remaining scientifically pending with zero executed actions.
This regression documents that prompt compliance/answer quality cannot be
certified by schema checks or task counters.

Command:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_qwen*.py' -v
```

All **37 Qwen-focused tests pass**, no skips: nine new binding checks and 28
existing operational/evidence/recovery checks. Coordinator integration and
independent review remain separate. No live model call tests this repair's
effectiveness; actual answer quality improvement remains **unverified**. The
native OpenAI work here addresses the observed scientific-relevance failure
without inferring access to an external OpenAI API or changing cluster budgets.

Local repaired worker SHA-256:
`0f78851940d7f3323fc5cf4408fd0f682a33d93d151c6be4ff6358e1185d8997`.
New test SHA-256:
`4dddef010cdb6e25305f30d957a81c8c8e193711a31b0078da537eba87968e4e`.

## Seventh live proposal — participation verified, recommendation rejected

At16:12:47UTC the existing999d88d worker answered a new compact v5 numerical-stop
event. The exact assigned question and619-character scientific note were
verified in its16,722-character request. Model/schema/usage checks pass:
4,472prompt+560completion=5,032tokens; cumulative seven replies/34,894tokens.
All original six reply files remain unchanged. The client returns to waiting
for evidence with no provider error or in-flight request. Final receiptSHA
e1eff3e6eeed25ab96bab14f37a0cbd813c18cd8649b71be36c4f702105a874e
is under ignored runs/ops/qwen-v5-review-event-20261013T1608Z/.

The coordinator rejects its scientific recommendation. Agreement between
execution paths does not establish retained information about earlier inputs.
The precision comparison and tiny encoder residual do not identify accumulation
order as the cause or prove behavioral insignificance. Without target scores,
the known checkpoint's competence remains unmeasured. Its new1e-5stopping
predicate and zero/nonzero state rule contradict the assigned frozen numerical
contract and cannot establish a causal distinction. Original provider output
and its pending scientific-decision field are immutable; rejection is recorded
in a separate coordinator receiptSHA
6bf81a29a090426ed286306a7eb364800c34e8b4121c29ff28dec2112501bd9d.
No suggested experiment or threshold change was executed. The independent
numerical calibration was separately frozen before this reply. The reviewed
prompt repair remains staged; this call did not test the repaired source.
