# Progress report — 2026-10-17

## 12:33 UTC / 15:33 Qatar — enforce frozen CPU policy before fitting

An independent protocol audit found that the epoch-evidence adapter could
accept CUDA source tensors and clone them to CPU, even though selected-checkpoint
v1 pins CPU/FP32 training and rejects device migration. This was a real release
gate: the old guard ran only after a complete epoch, so an incompatible fit
could have performed up to 2,559 updates before failing.

The native adapter now validates the backend device and every initial
state-dictionary tensor against the pinned names, order, shapes, dtype and CPU
device during construction, then repeats the check at `fit()` entry before any
update. Focused tests reject a CUDA backend and a post-construction mixed-device
mutation with zero optimizer updates. Independent review accepted source SHA
`4c2a50c140d39f80c728b3c1e4bb277c511bd989d1956725cb604d08637d68d0` and test
SHA `a0e644b74a4a2a84b895e8793601eb274a67e8c458483ed82b351de7d4587b47`.
The native fit, selected-checkpoint codec and selection-transaction suites pass
39/39; `git diff --check` passes.

This is protocol enforcement and synthetic engineering evidence only. The
epoch-to-bundle assembler and production selection callback remain missing;
source/data/runtime admission and selected-checkpoint load/replay parity also
remain unreleased. No primary fit, test access, cluster operation, submission,
Figure 10 reproduction or representation-collapse finding occurred. Next local
gate: independently review the typed epoch-evidence-to-six-file candidate
encoder, then connect verified candidate publication to the durable selection
ledger while preserving the external trust boundary.

After the owner confirmed Mila access, read-only BatchMode SSH probes for Mila
and Tamia from this local environment failed at hostname resolution, before
authentication. No cluv version or live scheduler state was obtained; this
does not establish that the owner's restored interactive session is unavailable.

## 12:38 UTC / 15:38 Qatar — epoch-candidate encoder accepted synthetically

Implemented `encode_epoch_candidate` in
`src/representation_collapse/tgrab_figure10_candidate_encoder.py`, consuming a
complete epoch snapshot plus explicit caller-supplied provenance and manifest
identity pins. It emits the frozen six canonical file payloads after checking
exact epoch/update counters, catalog names/order/shapes, CPU FP32 tensors,
terminal H/C chronology, 256 ordered validation rows and recomputed time-mean
F1. It validates the metadata against the pinned schema and verifies
cross-file identities. A regression discovered in independent review—schema
identity was not tied to `provenance.source_sha256.schema`—is fixed and has a
mutation-rejection fixture.

Independent review accepted current exact hashes: encoder
`08381a63be5d8c271d7cae70088c098516cb2e7a734441e5ea24aa28c5986b32`; tests
`8ee97a15548e63994f67e13558f31a8021334a9f2f4f88e0d83acefaa4cb9081`. Reviewer
found no other internal cross-binding defect. The synthetic bundle is accepted
by the existing create-only writer and `read_candidate`; the test preserves
signed-zero and subnormal FP32 bits. Combined Figure 10, fit, adapter, encoder,
canary, checkpoint and transaction suites pass 82 tests (3 platform-specific
skips); diff checks pass.

This builder checks consistency but cannot authenticate its caller's provenance
or admission. Split-data hashes and the conversion receipt still require
authentication by an external trusted admission path. The builder is not yet
wired into a production fit callback or the S07/S08 ledger. No trained candidate
was produced, no primary data or test targets were opened, and no training or
cluster job ran. The local readiness test still correctly reports launch
blocked by the frozen disabled config, missing runtime/dependency receipts and
the absent native checkpoint/reporting runner. A fresh read-only SSH probe could
not resolve Mila or Tamia hostnames from this execution environment, so current
cluv/scheduler state remains unverified here.

Next local gate: integrate complete-epoch evidence encoding, candidate
publication and durable selection as one production callback. Keep transaction
failure terminal, commit a candidate only after read-back verification, and
require explicit trusted admission inputs; then independently review and
exercise failure/reopen behavior before considering a fit.

## 23:01 UTC / 02:01 Qatar — durable selection transaction gate

Continued the Figure 10 reproduction subgoal after Mila access was restored.
No new cluster job, data access, model construction, or training was performed.

Implemented the pure-data S07/S08 selected-checkpoint transaction ledger in
`src/representation_collapse/tgrab_selection_transaction.py`. The ledger
verifies a complete candidate bundle before recording decisions; applies the
strict validation-F1 improvement rule with earliest-tie retention; stores
immutable intent, receipt, and pointer objects; atomically replaces the
selection head as the sole commit point; and requires the caller to provide an
externally trusted run admission and expected prior-head SHA. It cannot create
that trust itself. Failures poison the live ledger, persist a terminal failure
stage, and prohibit automatic recovery or retry.

Independent review caught a no-head edge case where a stale `selected.json`
alias could have been accepted as a fresh ledger. The fix rejects any alias
when the head is absent and binds historical intent transaction IDs, sequence
numbers, and timestamps to their receipt. Independent review then accepted
exact hashes: module
`fb7efb7780afbdb4bb094f21a601a08533d72535d2046e56a94666f4e9b2e872` and tests
`30fd7986e5ccf1e2306de04972a2a754f67cc8e7aa7b6ee8212d040509013d3c`.
The focused synthetic ledger suite passes 11/11; compilation and scoped diff
checks pass. A minor non-blocking same-user scratch robustness caveat remains:
the lock inode is not rechecked against its pathname after acquiring `flock`;
normal project code never unlinks the lock.

Current accepted engineering gates are: Mila node2a synthetic native parity
(job11280629, zero updates and no data split opened); the observed graph-change
metric guard; the data-only checkpoint codec; exact fit-policy enforcement and
pre-update attempt callback; and this durable selection transaction logic.
They are engineering evidence only. Integration still must connect each
complete epoch to cloned recurrent/head weights, all 256 validation rows, and
terminal hidden/cell state at `s[2816]`; persist selected candidates; bind
inputs to verified data/source/environment admissions; and pass node2b reload
plus full hidden/cell replay before any test split is opened. Figure 10 remains
unreproduced, and it is a behavioral F1 figure rather than evidence of
representation-dimension collapse.

## 12:29 UTC / 15:29 Qatar — complete-epoch candidate evidence handoff

Added `on_epoch_complete(record, terminal_state)` to the fit lifecycle and a
native-adapter evidence snapshot. The hook fires once per complete epoch only
after all 256 validation rows have been scored and consumed and the final
epoch guard succeeds; it receives state after consuming validation timestamp
2815 (`next_target_t=2816`, the full hidden and cell state). It runs before the
selection callback/application, although the core has already calculated the
validation-selection Boolean and updated its in-memory best fields.

The adapter validates terminal chronology, canonical node order/shape, exact
state_dict names/order/shapes/dtypes against the pinned 20-recurrent/6-head
tensor catalog, consistent source device, and all 256 ordered validation
rows. It returns read-only containers and independent detached CPU FP32 clones
of H/C and both component state dictionaries. Tests show those snapshots stay
unchanged after later synthetic optimizer mutation. The synthetic CUDA-device
shim exercises transfer behavior; no actual GPU or native model was run.

Independent review accepted exact core/adapter/test hashes:
`2c771a13eada794aecb14d740b8cc0dccffd0501db6fffa533f3170039e8cae0`,
`63f707dbf9c1b2a0b38647dfb990fe638f1836b1174c53d312c600fbd1cbd184`,
`debf456c488236e1c3a9aec7b85fb4a4027776d9dc54d54f85c684779da5d4d8`, and
`ad40cc2f1582cbecbb8956e7bf1fd5312fcb3e0d4b28913bb90453a73e96b536`.
Focused synthetic suites pass 13 fit-core + 15 adapter tests. This supplies a
candidate-evidence handoff only; it does not encode/write the six-file bundle,
commit selection, establish external trust, or load/replay a selected model.
No data or test split was opened, and no training or cluster work occurred.
