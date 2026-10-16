# Progress report — 2026-10-17

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
