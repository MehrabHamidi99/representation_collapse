# Progress report — 2026-10-23

## One-bit memory discovery accepted

The previously submitted cluv Tamia job 633041 (source `790fb9443565941a3cff9cd01c5a034d24480c18`, run `onebit-20261020T204921Z-790fb944`) has completed `0:0` in one minute on `tc11103`, batch MaxRSS 476828K. No new submission or retry was needed. Collection brought back 34 compact JSON/JSONL/W&B files (25,771,226 bytes) with remote SHA-256 and byte-count checks. Checkpoint and `.npz` recurrent states remain on Tamia scratch.

The independently reviewed reducer, `code/analysis/reduce_onebit_memory.py`, passed its ten focused tests and verified all 5,000 training rows, three prespecified competence boundaries and 36,864 held-out episode rows. Every required validation cell was 512/512. All nine held-out conditions scored 1.0 in both periodic and matched-control arms; `δ16=δ2=DID=0`. Actual offline W&B protobuf verification reconciled 201 history rows with the training log, source/runtime identity, metadata artifact and terminal reporting receipt. The result digest is `ab36f689f337d57ebfe165a523c772c4a6dd7404155781dadf3b5e92270886ba`; stream digest `fcc3d3a2153fe71686d48be470d4e2dbc7775e564e698d026c09a455ce9c90e7`. Online W&B delivery remains unverified.

The result and limitations are in [the public report](../../auto/onebit-memory-results-2026-10-23.md), with a [plot and compact evidence](../../auto/figures/onebit-memory-2026-10-23/manifest.json). The pooled H+C covariance PR near one is a descriptive state geometry result, not harmful collapse: cue separation increases and recall remains perfect. This single-seed, one-bit task cannot establish general memorization capacity. Next scientific node: freeze a higher-load task with separately queried stored bits and matched body forcing before opening any held-out results.

## Agent and connection cleanup

The three one-bit implementation/protocol/review subagents had completed. The idle local Qwen worker was stopped by its project STOP file and verified to have exited. Other active Codex CLI sessions were located in other project directories, and Mila SSH sessions were owned by a terminal or other projects; none was terminated. No new subagents were started for this continuation.
