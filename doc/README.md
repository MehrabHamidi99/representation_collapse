# Lab notebook

Adapted from the Skill Inference project's month/day convention. The canonical
current state is [TRACKER.md](TRACKER.md); reusable protocols are under `auto/`.

Write one report per workday:
`doc/<lowercase-month>/<month><day>/YYYY-MM-DD-progress-report.md`.
For example, `doc/october/october11/2026-10-11-progress-report.md`.
Append timestamped sections when continuing the same day. Reports retain
history; the tracker is edited to reflect current state.

Every report records purpose, changes, commands, evidence/artifact paths,
results with their scope, unresolved issues, and the next bounded tasks.
Distinguish proposed, implemented, tested locally, tested on cluster,
reproduced, and scientifically accepted. A successful job alone is insufficient.
Link scratch artifacts rather than copying model weights or datasets into git.

Record protocol changes in a dated decision note and link the new frozen
protocol. Analysis references source hashes, environment lock, run ID,
cluster/job ID if applicable, seeds, controls, and uncertainty method.
Speculative ideas remain labeled hypotheses until measured.
