# Query-averaged acquisition v5: stopped production attempt

2026-10-13. **No fixed endpoint was obtained.** The exact reviewed v5 campaign
ran locally and stopped during its first GRU/categorical fit after 4,000 of the
planned 5,000 updates because the original native chronology adapter rejected a
numerical comparison. The remaining three fits were unstarted. This is an
execution/measurement failure with unknown endpoint competence, not a finding
that query averaging succeeds or fails at acquisition. No fit was resumed,
replaced, rescued or selected, and no tolerance was changed.

## Actual execution and authority

The coordinator admitted committed source
`f766df5e792c8a6dedaa3ea17f2c92fd630f067d` after independent implementation review
and 271 passing repository tests. The external freeze allowed exactly four
local CPU fits × 5,000 updates, at most 20,000 optimizer updates and 1,200 engine
seconds. Execution used the existing project Python environment and a clean
isolated source tree; all twenty copied reference/review artifacts were hashed
before launch. The configured native model/data authority remained
`999d88d8679c0f353c6bb7fb4f5ab8433a7e730c`. There were zero GPUs, cluster
submissions or provider calls.

The actual child PID was **43157**; its launcher PID was 43154. Launch was
2026-10-13 11:46:45.346 UTC; exit was 11:47:16.866 UTC, code 1. Actual launcher
walltime was 31.520 s, engine walltime 30.274 s. Realized execution was 4,000
optimizer updates, 1,024,000 QUERY-branch loss instances and 3,072,000 nominal
recurrent row updates. Batch 32 represented sixteen independent memory
histories with duplicated R0 arms. These are realized workload counts, not a
compute-efficiency or statistical sample-size claim.

Before fitting, all four original-source initializations passed actual
128-hidden/64-encoder/batch-32 helper calibration in FP32 and FP64, including
all parameter gradients, native query logits and complete states, plus the
applicable detached-prefix, chained-query, omitted-cell and missing-average
negative controls. Those checks performed zero optimizer updates. Both complete
5,000-batch prospective streams and the new stratified validation manifests
were frozen before the first update. This finite calibration did not guarantee
that the adapter tolerance would hold at every subsequently trained weight.

## Failure and retained evidence

At scheduled step 4,000 the unchanged `native_state.py` adapter rejected
`native_read_query_vs_forked_logits`. The first failing 64-episode chunk had
maximum absolute logit difference **1.4491379261e-6**. Its frozen rule was
elementwise `abs(candidate-reference) <= 1e-6 + 1e-5*abs(reference)`;
the maximum absolute difference alone does not determine which entries fail.
The exception describes numerical disagreement between equivalent query paths,
not nonfinite learned parameters. All preserved parameters remained finite.

The step-4,000 scheduled checkpoint was already written. Cleanup additionally
preserved a censored current-weight checkpoint at the same exact realized step,
hash `b57b645f80a8e85a1743385e01df8075706904e122a57923b6216310365f3788`,
with endpoint competence unknown, evaluation usability false, no selected
checkpoint and no resume authorization. History contains every returned
optimizer update 1–4,000. Complete validation arrays, native states, logits and
scalars were saved at steps 500, 1,000, 1,500, 2,000, 2,500, 3,000 and 3,500.
No step-4,000 validation record passing the original numerical guard or fixed
step-5,000 endpoint exists.

The outcome ledger is therefore:

- GRU/categorical: 4,000 realized updates; numerical chronology failure;
  fixed step-5,000 endpoint and its preregistered paired contrast unknown.
- GRU/continuous: unstarted; endpoint and contrast unknown.
- LSTM/categorical: unstarted; endpoint and contrast unknown.
- LSTM/continuous: unstarted; endpoint and contrast unknown.

All seven saved GRU/categorical partial evaluations that passed the original
numerical guard failed the all-eight coordinate gate. At the last such saved
evaluation, step 3,500, QUERY
0–7 accuracies were respectively **0.328125, 0.2890625, 0.2890625, 0.3125,
0.328125, 0.4453125, 0.875, 0.9921875**. Each coordinate used 128 independent
episode pairs; the identical R0 arms agree. QUERY 7 cleared its coordinate
threshold while the required conjunction failed. These partial observations
are neither selected endpoints nor a replacement for the missing fixed-budget
comparison.

## Offline raw reduction and numerical diagnosis

A separately written reducer imports no project scoring, phase-capture,
target-derivation or data-constructor helper. It reconstructed both complete
training streams directly, preserving every original sampled QUERY RNG draw,
all eight derived target/query views and every rolling hash; independently
reconstructed seed-20005 validation arrays byte for byte; verified fifty
manifested raw files and source/lock/config identities; and checked every
realized update identity, saved coordinate score and episode-paired difference.
All assertions passed. This is producer verification, distinct from the
coordinator's independent scientific result acceptance.

All four pinned v3 fresh step-5,000 baselines were reevaluated, with zero
training, on these exact new original-task validation histories before fitting.
Their saved native arrays were replayed from their pinned checkpoints. Baseline
LSTM verification included both hidden and cell state. The reducer also
replayed all seven saved GRU checkpoints, checking full native logits, complete
prefix and post-query states against the saved arrays. It retained the 128
episode-level error differences for each coordinate at each partial saved
step. These are explicitly **descriptive partial-history** comparisons to a
baseline trained for 5,000 updates; they do not estimate the preregistered
equal-update endpoint contrast. No CI, p-value or trained-checkpoint selection
was introduced.

For diagnosis only, the reducer replayed the preserved current step-4,000
weights on all 2,048 validation arm rows in both FP32 and FP64, with zero
optimizer updates and no target scoring. This full replay found:

- FP32 separately encoded QUERY versus its slice from the full encoded
  sequence differed by at most **5.9604644775e-8**. This encoding difference
  itself satisfied the declared FP32 comparison bound.
- FP32 native read-query logits versus forked logits had two entries exceeding
  the elementwise bound, maximum absolute difference **1.9073486328e-6** and
  maximum error/allowed-error ratio **1.42007**. This all-row maximum differs
  from the earlier first-failing-chunk maximum.
- FP32 forked versus uninterrupted logits and complete post-query states were
  exactly equal in this replay.
- The same saved FP32 tensors represented in FP64 had zero terminal-encoding
  difference and read-query/forked-logit difference at most
  **2.6645352591e-15**, within the original FP64 bound. FP64 forked versus
  uninterrupted complete states and logits were exactly equal.

These observations are consistent with floating-point path effects involving
separate versus full-sequence encoding; they do not certify a single kernel as
the cause. FP64 diagnosis does not repair, accept, resume or rescore the failed
FP32 campaign. Any prospective measurement change needs separate source,
label-independent calibration, review and admission; the v5 failed attempt
remains immutable.

The plot is **local-only**, under ignored runtime storage:
`runs/query-acquisition-v5-20261013/reduction/plots-final/v5-stopped-attempt.png`.
Its create-only generator is
`runs/query-acquisition-v5-20261013/plot.py`; exportable PDF and SVG accompany
the PNG. The coordinator can review these artifacts before publishing a copy.

The left panel shows the seven saved partial coordinate evaluations that passed
the original numerical guard; the fixed endpoint is unmeasured. The right panel
compares maximum
error with each dtype's already declared elementwise allowance at preserved
step 4,000. FP64 is a diagnostic replay of identical stored weights, not a new
training condition.

## Provenance and next decision

All runtime artifacts are under ignored
`runs/query-acquisition-v5-20261013/`. Stable receipt hashes are:

- Frozen configuration: `d6d89b431942c8e786f1ac169d5fee09ef51e8f85f2030cee1ec8bcdd9d4289e`.
- Reviewed implementation: `f2af2c73088731e3664b7ca59c54ed9c9f083dc863b83881b20956ffb429a19d`.
- External freeze: `ff73dbedc6215b71d581f966bff8268e1197a677e5c302245accb2bf00c210fd`.
- Actual launch receipt: `7bc71c018773c006d3464545975fd0538a11af7781431e3b436f31568826b48f`.
- Actual exit receipt: `0f296547b1f8698cde7d215803ba53da67c3081cb69adeafbc7e515628ae5d3a`.
- Offline reducer: `cd85a209ffd031bda95b05bd0867b01995ebe3231d3ceac9e2591f0928c253f2`.
- Raw results dossier: `b5d12914e290116dba86bdfea4cd8243108e46d4242e0f9a3b96574498ca5f3a`.
- Terminology-clarified publication dossier: `b3c161e35cdc1a98112339ec50bd3261ed219fad21ff95176cbf7459a4e74774`.
- Reduction verification: `82d5b6864a43b12ef39c1e21a22c8ab7c59fb667469b5facce8d565a07709d0b`.
- Local-only plot generator: `7d1dad49b3b1c22c6d7488fb6c7d1051b62bedd250c42b15c6cd64c131d60f70`.
- Local-only plot PNG: `0380a2ec0c19e00fef25de58206f6c0afd8c7fa94532320968fda8d3ce2c22e1`.

`results-publication.json` changes only "accepted trajectory" terminology to
"saved trajectory passing the original numerical guard," preserves all values,
and binds the original immutable `results.json`. Neither producer artifact
asserts independent scientific acceptance.

Independent result review is separate and was requested against these immutable
raw artifacts. The next decision is a prospective numerical chronology
calibration protocol. This attempt grants no R0 endpoint competence, Gate B
R1/R2 eligibility, periodicity-harm inference, geometry/collapse finding,
Floquet certification or cross-modal replication.

## Coordinator integrity acceptance

The [independent result review](query-averaged-acquisition-v5-independent-results-review-2026-10-13.md)
passes at receipt SHA-256
`46b5dc21549373cd11bedb556b780d3956b6f93d6507d324f6fc9777055a2331`.
The coordinator accepts the verified execution, numerical stop, retained
current weights and seven saved partial observations. All planned endpoint
outcomes remain unknown; no fixed-endpoint scientific claim or R0/GateB
eligibility is accepted. The [scalar dossier](../../../../../doc/auto/query-averaged-acquisition-v5-results.json)
keeps failed/unstarted cells, partial measurements and numerical diagnosis
separate from acceptance. The [numerical calibration proposal](native-chronology-calibration-proposal-2026-10-13.md)
is a new prospective design and authorizes no retry or tolerance change.
