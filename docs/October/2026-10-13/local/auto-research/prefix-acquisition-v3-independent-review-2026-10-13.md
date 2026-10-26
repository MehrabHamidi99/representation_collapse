# Prefix acquisition v3 — independent result review

2026-10-13. Independent reduction verifies all eight fixed-endpoint local fits
and their failure to meet the full eight-coordinate competence criterion.
Some coordinates pass, so “all coordinates fail” would be incorrect. These are
observations about seed 0 under the frozen acquisition schedule. Coordinator
acceptance remains separate; no periodicity-harm or Floquet result follows.

The reviewed [production report](prefix-acquisition-v3-results-2026-10-13.md)
preserves all eight fits, their 40,000 R0 optimizer updates and every scheduled
validation. The process ran for 189.303 seconds, with 187.759 seconds inside
the engine, below the 1,200-second bound. All endpoints are current update-5,000
weights. No fit passes the complete gate at any of the ten scheduled evaluations;
no earlier checkpoint, selected learning rate, replacement or successful subset
substitutes for the endpoint.

## Source, matching and measurement checks

The immutable execution source is
`999d88d8679c0f353c6bb7fb4f5ab8433a7e730c`. The archive's Git commit comment,
archive SHA-256 and all **490 archived file contents** agree with the isolated
source tree. Nine recorded runtime source digests agree. The pre-execution
freeze binds the reviewed implementation, configuration and independent
engineering receipt. Configuration SHA-256 is
`c5d32b861cf22b063c91cb14cf6126617ee80cbca357c8720bb1fb140928fef9`;
implementation SHA-256 is
`cf3f443b9596ddc93cf675f33bb152fa1a3a31fd618aadda11de4263eb88dc52`.

The reviewer independently reconstructed both complete 5,000-batch training
streams directly from NumPy draws and channel assignments, without calling the
producer's renderer or training constructors. All **10,000 batch identities**
and rolling hashes match the prefit manifests. All **40,000 recorded update
rows** match those streams, have finite recorded loss/gradient diagnostics and
retain the declared evaluation schedule. Training batch 32 consists of
16 independent histories with duplicated R0 arms.

Fresh initial weights reproduce seed-0 initialization. Transfer initial weights
match the exact pinned immediate-task step-500 checkpoints. Both conditions
report empty initial AdamW state; the reviewed executing source constructs a
new optimizer and rejects nonempty initial state. Optimizer updates were not
retrained by this review. Equality concerns subsequent R0 exposure: transfer
also has 500 earlier immediate-task updates.

The reviewer verified **209 output-file hashes** and recomputed all **1,280
coordinate metrics and gate booleans** from the eighty raw validation files.
Inputs, targets, selectors and pair IDs match the prefit task-specific manifests.
Each coordinate has 128 episodes per arm; addresses and payloads match sixteen
ordered WRITEs and one payload-free terminal QUERY. The R0 arms are identical
and contain no distractor body. Continuous metrics use raw squared error divided
by the fixed population variance 1/3.

Independent native replay used steps 500 and 5,000 in every fit: **16
checkpoints**, **278,528 logit entries** and **18,874,368 complete-state entries**.
They match saved CPU arrays exactly. Replay called the immutable native
model/core/head directly, without the producer's capture or scoring helpers.
GRU state contains 128 hidden coordinates; LSTM state contains 128 hidden and
128 cell coordinates, packed hidden then cell. Post-WRITE, pre-QUERY and
post-QUERY are checked separately. The producer's larger eighty-checkpoint
replay is recorded in its own receipt; it is distinct from this reviewer's
sixteen-checkpoint subset.

## Verified endpoint observations

Both GRU continuous conditions pass coordinates 6 and 7 and fail 0–5.
Fresh coordinate MSE/(1/3) is 0.038888 and 0.019501 for 6 and 7; transfer gives
0.048008 and 0.026074. Transfer LSTM categorical passes coordinate 0 at
126/128 correct, accuracy 0.984375, while coordinates 1–7 fail. The remaining
five conditions pass no endpoint coordinate. Every fit therefore fails the
declared complete criterion.

The paired endpoint ledger preserves 128 episode identities per coordinate.
Its transfer-minus-fresh native-error means, equally weighting coordinates, are
−0.0078125 for GRU categorical, −0.197265625 for LSTM categorical,
−0.042897724 for GRU continuous and −0.000185522 for LSTM continuous.
Categorical units are error probability; continuous units are MSE/(1/3).
These are descriptive contrasts for this fixed validation set. Improved
aggregate error does not establish complete competence.

The ledger's 4,096 per-cell episode comparisons reuse **2,048 unique
task-specific validation histories** across architectures. Repeated checkpoint
evaluations reuse those same histories. Neither count supplies additional
independent training seeds. No p-value or trained-seed confidence interval is
reported. The plot's input/output hashes agree with its manifest, and the PNG
was visually checked against the endpoint values and gates.

## Scope and next gate

The measurements support a scoped negative acquisition result and heterogeneous
partial task performance. They do not establish architectural impossibility,
a memory mechanism, periodicity-induced harm, covariance collapse, cross-modal
replication, equivalence or certified Floquet dynamics. CPU replay equality
does not establish GPU parity or repeat-training determinism.

Prospective controls should separate addressed multi-item demand, semantic
coordinate identity, WRITE position and nuisance updates. An apparent
coordinate/lag pattern cannot identify a mechanism because the original
coordinate and WRITE position are confounded. Changed ordering or QUERY address
15 requires a separately validated protocol and adapter; existing v2 guards
remain intact. Preserve these v2/v3 outcomes before undertaking that new work.

Ignored review receipt:
`runs/review/prefix-acquisition-v3-independent-review-20261013.json`.
This reviewer performed no new training, provider call, cluster operation,
Git action or source change.
