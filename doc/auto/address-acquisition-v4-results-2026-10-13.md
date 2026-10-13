# Address acquisition v4 — local production results, 2026-10-13

Eight singleton-address fits passed their fixed endpoint gate, while four
failed. Every GRU/LSTM categorical/continuous fit at QUERY address 7 or 15
passed after exactly 5,000 updates; every address-0 fit failed. These are
separate predictably queried singleton tasks. They do not confer original
eight-coordinate eligibility, establish pure lag/capacity causality or test
periodicity-induced harm.

## Frozen execution and controls

The [v4 protocol](address-acquisition-discovery-v4.md) was reviewed and frozen
before execution. Actual local CPU PID 35825 ran from
2026-10-13 10:25:42.506410 to 10:29:23.389457 UTC, returning exit 0. All twelve
fits completed exactly 5,000 new R0 optimizer updates: 60,000 updates total,
220.881 process-wall seconds, within the 1,200-second engine ceiling. No timeout,
nonfinite failure, early-success stopping, selected checkpoint, resume,
replacement, extension, held-out confirmation, cluster submission or provider
call occurred. Baseline reevaluation consumed no optimizer updates or new fits.

Production source was the isolated clean Git snapshot
`ab96cad9dece7d881f04065431c0d102f7665882`, with unchanged model/data code pinned
to v3 source `999d88d8679c0f353c6bb7fb4f5ab8433a7e730c`. Configuration SHA-256:
`55c564c5c607ce3cf298e4daabaeafb2700d50419a4ee859200e0c9bba811c55`.
Implementation SHA-256:
`cb06f60d45cd26e27f1e8c4959de05d183544568eb19614f60d72d9636376958`.
The exact ignored freeze receipt, `runs/ops/address-acquisition-v4-freeze.json`,
has SHA-256 `9d744484564846e5339dff9323089ae996624b5c674ae8bc22704c857c6b3868`.
Its pre-execution independent-review receipt has SHA-256
`6a5718f0575a2e2d8c7b1e39e20e57209c77ae144516d14f3383fd4fd0405a46`.

All fits used fresh seed-0 parameters matching the pinned v3 fresh initial
tensors exactly. Architecture was hidden 128/encoder 64/batch 32; AdamW used
LR 0.001, weight decay 0.01 and clipping 1. Each optimizer began with empty
state. Actual execution used native Torch 2.6.0 CPU FP32, one thread,
deterministic algorithms, TF32 off and highest matmul precision. The original
sixteen ordered WRITEs and terminal QUERY stayed fixed, with no distractor body.
Only terminal selector/target changed across the derived training conditions;
all original v3 RNG draws and WRITE payloads were preserved and hashed.

Validation used new seed 20003 and 128 independent histories per address/arm,
persisted before optimizer step 0. Histories were shared across address
conditions within a task. Identical R0 arms were counted as 128 paired units,
not 256 independent histories. Scores below show their shared value once.

## Every fixed endpoint

Triplets list QUERY addresses 0, 7 and 15 in that order. Categorical accuracy
must be at least 0.95; continuous MSE/(1/3) must be at most 0.05, using the fixed
population variance and declared four-ULP numerical threshold allowance.
All scheduled results are retained; first passing step is descriptive only.

- GRU categorical accuracy: **0.09375 / 1.0 / 1.0**. Address 0 failed;
  address 7 first passed at update 1,000 and address 15 at update 500.
- LSTM categorical accuracy: **0.046875 / 1.0 / 1.0**. Address 0 failed;
  address 7 first passed at update 2,000 and address 15 at update 500.
- GRU continuous MSE/(1/3): **1.171140092 / 0.000048020 / 0.000031024**.
  Address 0 failed; addresses 7 and 15 first passed at update 500.
- LSTM continuous MSE/(1/3): **1.169030579 / 0.000281015 / 0.000015860**.
  Address 0 failed; addresses 7 and 15 first passed at update 500.

No address-0 scheduled checkpoint passed. No successful endpoint was selected
as a replacement for another condition or for the original task. An address-7
fit and an address-15 fit have different trained weights; their successes
cannot be combined into a claim that one model retains multiple required items.

## Supported paired baseline contrasts

The exact four v3 *fresh* current step-5,000 endpoints were reevaluated, without
retraining or selection, on the same new validation histories and terminal
selectors 0 or 7. These addresses are within their original trained QUERY
support. The contrast is singleton-trained minus pinned-random-query-trained
native error on 128 matched histories. Positive means worse. Categorical units
are error probability; continuous units are MSE/(1/3).

- GRU categorical: address 0 **+0.0703125**; address 7 **−0.765625**.
  Baseline accuracies were 0.1640625 and 0.234375.
- LSTM categorical: address 0 **+0.015625**; address 7 **−0.9375**.
  Baseline accuracies were 0.0625 at both addresses.
- GRU continuous: address 0 **+0.459997036**; address 7 **−0.018902516**.
- LSTM continuous: address 0 **+0.001601605**; address 7 **−0.957099609**.

Every per-episode paired difference is stored in the endpoint metrics and the
reduction receipt. There is no primary baseline evaluation/contrast at address
15, outside the original model's trained QUERY-selection support. Historical
v3 seed-20001 scores remain unchanged and are not pooled into these seed-20003
comparisons. No confidence interval, p-value, population-competence guarantee
or equivalence test was enabled. The independent training unit remains one
initialization/discovery seed; repeated validation, address conditions and
paired arm copies do not supply independent trained replications.

## Deterministic verification and acceptance boundary

The producer's separate reducer implements direct payload/target oracles,
canonical hashes, singleton metrics, paired-error reduction and native
checkpoint replay without calling production derivation/scoring/capture
helpers. It verified 333 run-file hashes, nine source digests and sixteen
pinned v3 reference files. It reconstructed both complete original v3 streams
(10,000 base batches), all six derived streams (30,000 batches), fresh
validation arrays and all 60,000 recorded update identities. It recomputed
240 scheduled arm metrics and 1,024 supported paired baseline differences.
Three hundred overlapping per-fit file-manifest entries were also reconciled.

All 120 trained scheduled checkpoints and eight pinned baseline evaluations
were replayed directly through the frozen native model class/core/head. All
278,528 logit entries and 18,874,368 complete-state entries matched saved CPU
arrays exactly. Captures include post-WRITE, pre-QUERY and post-QUERY phases:
128 hidden coordinates for GRU; 128 hidden then 128 cell coordinates for LSTM.
The new address-domain adapter leaves the original v2/v3 QUERY-domain guard
untouched. Input/weight/source identities and every preserved native check
remain available. Verification counts describe audit operations on shared
data, not independent scientific samples.

This computational reduction is distinct from independent reviewer/coordinator
result acceptance, which was pending when this report was created. It does not
establish repeat-training determinism, GPU parity, covariance dimension,
tangent contraction, causal memory or Floquet closure.

Ignored artifacts reside under `runs/address-acquisition-v4-20261013/`:
`launch-receipt.json`, `production.log`, isolated `source/`, complete
`experiment/` histories/checkpoints/arrays and `reduce.py`. The scalar/native
dossier `verification-receipt.json` has SHA-256
`f023febf294399660a0c17344d3ee0ad607d5ba7d4c4e7d8353b6b8a8b6c3aeb`.
It binds source, freeze, status, stream/file identities, every endpoint and
trajectory, supported baseline contrasts and all failures. Prior v2/v3
artifacts and reports were preserved.

## What remains open

Singleton acquisition succeeds at addresses 7 and 15 under this bounded
schedule, while address 0 remains unresolved. Addresses differ simultaneously
in identity and the number of later WRITE updates: 15/8/0 before the common
QUERY update. Fixed versus random QUERY training also changes required load
and selection uncertainty. The pattern cannot isolate lag, capacity,
overwriting, addressing or credit assignment. It is not architectural
impossibility, periodicity harm, cross-modal replication or a universal
collapse result.

A separate prospective address/position counterbalance and matched late-WRITE
replacement control could resolve those ambiguities. A periodicity experiment
would additionally need a predeclared competence/forcing protocol at fixed
weights, complete-state and precision controls, independent trained seeds,
causal interventions/shams and reviewed inference. No further execution or
checkpoint selection is granted by this report.

## Coordinator acceptance after independent reduction

The coordinator accepts the scoped singleton acquisition audit after the
[independent result review](address-acquisition-v4-independent-results-review-2026-10-13.md).
Receipt SHA256:
`6d0ef52b2c33215cf6fccc28adb19cb4c8d80936f018dd1098f83d82f9f0dda7`.
All eight q7/q15 empirical singleton gates pass and all four q0 gates fail
under this exact seed-0 schedule. The review independently reconstructs the
streams without production constructors, re-reduces every raw metric/paired
baseline error and replays 24 trained checkpoints plus all eight baselines.
It verifies 69,632 logits and 4,718,592 complete-state entries bitwise.

There are 128 shared validation histories per task, 256 task-specific draws
total, reused across addresses and architectures with duplicated arms. Neither
the twelve fits nor the eight passing singleton weights are independent
trained-seed replications or one model retaining all required coordinates.
The [committed scalar dossier](address-acquisition-v4-results.json) preserves
all endpoints, failed fits and supported per-history baseline contrasts.
No main eight-coordinate eligibility, periodicity harm, pure lag/capacity
attribution, population guarantee or Floquet claim is accepted.
