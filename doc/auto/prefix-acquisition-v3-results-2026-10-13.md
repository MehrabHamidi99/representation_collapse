# Prefix acquisition v3 — local production results, 2026-10-13

All eight fixed-endpoint fits completed, but none passed competence on every
required coordinate. Increasing R0 exposure to 5,000 updates and comparing
fresh initialization with immediate-task parameter transfer did not yield a
competent full-prefix checkpoint under this protocol. Some individual
coordinates improved or passed; that partial acquisition is preserved. No
periodic distractor body was used, so these results do not test periodicity harm.

## Frozen execution

The [prospective v3 protocol](prefix-acquisition-discovery-v3.md) was committed
and reviewed before production. Actual process PID 28903 ran locally from
09:32:30.813166 to 09:35:40.115822 UTC on October 13, returning exit 0. The engine
completed eight sequential fits, each with exactly 5,000 R0 optimizer updates:
40,000 updates total, 187.759 engine seconds and 189.303 process-wall seconds.
There was no timeout, nonfinite failure, replacement, rescue, stage extension,
early-success stop, held-out evaluation, GPU allocation, cluster submission or
provider call. This was the actual production entrypoint, not fixture mode.

Immutable source: `999d88d8679c0f353c6bb7fb4f5ab8433a7e730c`. The clean archived
source has its own Git identity under ignored
`runs/prefix-acquisition-v3-20261013/source/`. Pinned warm checkpoints and the
independent pre-execution source-review receipt were copied and verified there,
so subsequent coordinator edits cannot change this run's source identity.
Configuration SHA-256:
`c5d32b861cf22b063c91cb14cf6126617ee80cbca357c8720bb1fb140928fef9`.
Implementation SHA-256:
`cf3f443b9596ddc93cf675f33bb152fa1a3a31fd618aadda11de4263eb88dc52`.
The ignored exact freeze receipt is `runs/ops/prefix-acquisition-v3-freeze.json`,
SHA-256 `bb2ab77a6fa1f3eabc94ab6004d398f3af502b9026463f9f615dcea4c4d507fc`.

Hidden 128/encoder 64, batch 32, LR 0.001, AdamW with weight decay 0.01,
clipping 1, one CPU thread, Torch 2.6.0 and native FP32 were unchanged. Actual
deterministic, TF32-off and highest-matmul-precision flags were recorded.
Fresh initialization uses seed 0. Transfer starts from the exact v2 current
step-500 immediate-task weights, with a newly empty AdamW optimizer. Transfer
has 500 prior immediate-task updates; both conditions have equal subsequent
5,000-update R0 exposure. It is not equal total historical exposure.

Both complete training streams were pinned before fitting, regenerated exactly
per condition and checked at every update. Their final SHA-256 identities are
`71845dbba715abaa68a20037028eb91fa1f358c382056e16852721cab26c16c1`
(categorical) and
`c394311766112db25e69c791f8f2f2acd4a9da48e0a255049e0300b6948df7d3`
(continuous). Training stream seed was 2026101301. Fresh validation seed 20001
was distinct from v2 and training; its fixed inputs/targets were persisted
before optimizer step 0 and shared across the matched conditions.

## Fixed endpoint scores

Each list below gives required coordinates 0–7 in order, using the current
5,000-update checkpoint. Categorical values are native-query accuracy;
continuous values are MSE/(1/3), normalized by the frozen population variance.
Per-coordinate gates remain accuracy ≥0.95 or normalized MSE ≤0.05, with the
unchanged scorer's declared numerical threshold allowance. Each coordinate has
128 paired validation episodes. At R0 the two forcing arms are identical, so
the shared scores are displayed once. No aggregate substitutes for a coordinate
gate, and no earlier checkpoint substitutes for the fixed endpoint.

- GRU categorical, fresh: 0.265625, 0.234375, 0.203125, 0.3046875,
  0.21875, 0.2265625, 0.171875, 0.203125. All eight coordinates failed.
- GRU categorical, transfer: 0.2890625, 0.21875, 0.1953125, 0.265625,
  0.2109375, 0.1953125, 0.25, 0.265625. All eight failed.
- LSTM categorical, fresh: 0.09375, 0.0703125, 0.078125, 0.0546875,
  0.0625, 0.046875, 0.0625, 0.03125. All eight failed.
- LSTM categorical, transfer: 0.984375, 0.125, 0.1328125, 0.15625,
  0.1640625, 0.203125, 0.171875, 0.140625. Coordinate 0 passed; 1–7 failed.
- GRU continuous, fresh: 0.784858, 0.668228, 0.751565, 0.775240,
  0.598282, 0.082462, 0.038888, 0.019501. Coordinates 6 and 7 passed;
  0–5 failed.
- GRU continuous, transfer: 0.397080, 0.584573, 0.611114, 0.702404,
  0.596922, 0.409667, 0.048008, 0.026074. Coordinates 6 and 7 passed;
  0–5 failed.
- LSTM continuous, fresh: 0.977353, 0.937739, 1.008225, 1.027364,
  0.962570, 0.847167, 0.988840, 1.106339. All eight failed.
- LSTM continuous, transfer: 0.977231, 0.937472, 1.008197, 1.027479,
  0.962502, 0.846819, 0.988119, 1.106293. All eight failed.

No scheduled validation checkpoint passed all coordinates, and every
`first_passing_validation_step` is null. All eighty scheduled evaluations and
checkpoints are retained. There is no selected checkpoint or LR choice.

The explicit paired-coordinate ledger stores the same episode identities,
fresh/transfer errors and differences for every coordinate. Mean transfer-minus-
fresh native error, averaged equally across coordinates, was −0.0078125 for
GRU categorical, −0.197265625 for LSTM categorical, −0.042897724 for GRU
continuous and −0.000185522 for LSTM continuous. Categorical units are error
probability; continuous units are normalized MSE. Negative means lower observed
transfer error. These are descriptive conditional-seed contrasts, without
trained-seed confidence intervals, population inference or equivalence tests.
In particular, better aggregate error does not imply all-coordinate competence.

## Verification, plot and acceptance boundary

The producer's separate deterministic reducer independently implements hashes,
direct addressed-payload oracles, per-coordinate score reduction and native
checkpoint/state replay; it does not call the production scoring or capture
helpers. It verified 209 run-file hashes and nine source digests, reconstructed
10,000 task-specific training batches and reconciled all 40,000 update identities.
It checked 1,280 coordinate metrics across all eighty scheduled validations and
163,840 validation arm instances against their addressed targets. These repeated
evaluations use the same fixed validation draws; the counts are verification
operations, not additional independent scientific samples.

Every one of the eighty checkpoints was replayed through the immutable native
model class/core/head. All 1,392,640 logit entries and 94,371,840 complete-state
entries matched the saved CPU arrays exactly. GRU captures include 128 hidden
coordinates; LSTM captures include 128 hidden and 128 cell coordinates. All
three phases and the native terminal QUERY are preserved. The additional audit
reconciled 200 overlapping per-fit file-manifest entries and actual endpoint
weight identities, including fresh seed-0 initialization and exact warm θ500.
This is artifact/numerical verification, not repeat-training determinism, GPU
parity, independent human acceptance or a causal memory/Floquet result.
Independent reviewer/coordinator result acceptance remains separate and was
pending when this report was created.

Ignored evidence, under `runs/prefix-acquisition-v3-20261013/`:

- Main `verification-receipt.json`, SHA-256
  `d12297e6205f71c11fee12091aaa872925141273bcf02a5a70496d38fe6e1ecd`.
- `checkpoint-metadata-verification.json`, SHA-256
  `b224dfdb21dec9933c4f2094cfe0b8f5db45aa44a521c7ad1801a5aa13f18cd0`.
- `paired-coordinate-endpoint-ledger.json`, SHA-256
  `4c1d6abd3f7afa83cc8965f2067e068d451ff22dab6ad8f26037a0c741881d81`.
- Reproducible, visually inspected plot:
  `figures/prefix-acquisition-v3-endpoints.pdf` and `.png`, generated by
  `plot_results.py` from the verified receipt. Plot manifest SHA-256:
  `f31f0f42767863b6eb846e52439e454e0e8e1a2839875302b939f611f2ac6d1e`.

The plot shows every fixed-endpoint coordinate and its gate, with identical R0
arms shown once. It has no across-seed uncertainty bars because only seed 0 was
trained. Full native arrays, histories, initialization/endpoints, source, freeze,
stream identities and process receipts remain available for review.

## Next prospective controls — proposals only

Preserve the completed v2/v3 failures. A separate protocol could compare the
original sixteen-WRITE prefix under fixed QUERY address 0, 7 or 15 with the
original randomly chosen required address 0–7, using paired prefix histories,
equal training exposure, fresh validation and fixed endpoints. Fixed-address
conditions test single-value demand; the random required QUERY tests addressed
multi-item demand. The intervening WRITE counts are 15, 8 and 0 for addresses
0, 7 and 15, respectively, before the common terminal QUERY. Address 15 becomes
task-required only in that separate diagnostic. Existing v2 competence/capture
interfaces deliberately prohibit QUERY addresses ≥8; a separately validated
diagnostic adapter is needed, without relaxing those existing guards.

Those three addresses also differ in identity as well as lag. A retained-address
counterbalance that moves the same queried address/value among WRITE positions
0, 7 and 15 would be needed before attributing their difference specifically to
lag. That diagnostic changes WRITE ordering and therefore also requires its own
validated chronology contract. Fixed-address success alone cannot distinguish
retention from addressed multi-item binding in the original task.

A precise matched late-WRITE replacement control could retain all sixteen
update slots, target-bearing WRITEs and terminal QUERY, while replacing nuisance
positions 8–15 with valid payload-free X updates. Pair the original and replaced
sequences on required values/QUERY, keeping target chronology and total length
fixed. This tests the joint cost of nuisance WRITE type/address/value versus X
updates; it does not isolate payload interference or overwriting alone. An
additional payload-only sham would be needed for that attribution. Mixed
WRITE/X prefix handling must be implemented in a separate adapter, since v2
correctly requires sixteen initial WRITEs. These are prospective controls, with
no execution, code change, new budget or scientific acceptance granted here.

The acquisition bottleneck remains unresolved. Individual coordinate success is
useful diagnostic evidence, while no architecture-impossibility, cross-modal,
periodicity-causal-harm, covariance-collapse, equivalence or certified-Floquet
claim follows from this run.

## Coordinator decision after independent result review

The coordinator accepts the scoped acquisition audit after the independent
[review](prefix-acquisition-v3-independent-review-2026-10-13.md), receipt SHA
`c55a189c18fa97c4fa76016339a9780230835aa969219a632b08cdda568ad0fd`.
It independently regenerates the stream without the renderer, reconciles all
raw validations and replays sixteen endpoint/intermediate checkpoints exactly.
All eight full-coordinate gates fail, with the five passing coordinate scores
retained. The 4,096 displayed per-cell arm instances reuse 2,048 unique
**task-specific** validation histories; they are not 4,096 independent histories.
Conditions share these draws and one training seed. No harmful-collapse,
Floquet, population-equivalence or architectural-impossibility claim is accepted.

Committed scalar evidence is [prefix-acquisition-v3-results.json](prefix-acquisition-v3-results.json).
Reproduce the publication figure without weights or raw arrays:

```bash
.venv/bin/python code/analysis/summarize_prefix_acquisition.py
```

The [PDF](figures/prefix-acquisition-v3-endpoints.pdf), PNG and manifest record
all fixed-endpoint values, distinct metric units and the numerical/source pins.
