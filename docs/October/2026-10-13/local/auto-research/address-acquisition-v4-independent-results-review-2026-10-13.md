# Address acquisition v4 — independent result review, 2026-10-13

The fixed-endpoint measurements pass independent review within the local
seed-0 singleton-task scope. All twelve fits completed 5,000 updates; all eight
address-7/address-15 fits passed their empirical gate, and all four address-0
fits failed. No address-0 scheduled evaluation passed. Scientific acceptance
remains with the coordinator. This review does not confer original
eight-coordinate eligibility or establish periodicity-induced harm.

## Source, freeze and actual execution

The production source is the clean isolated commit
`ab96cad9dece7d881f04065431c0d102f7665882`.
A read-only Git-object parser independently checked the commit SHA1 and all
504 tracked source files against its tree, without executing Git commands.
Nine runtime source hashes, four unchanged v3 model/data hashes and sixteen
pinned v3 reference files matched. Configuration
`55c564c5c607ce3cf298e4daabaeafb2700d50419a4ee859200e0c9bba811c55`
and external freeze
`9d744484564846e5339dff9323089ae996624b5c674ae8bc22704c857c6b3868`
match the pre-execution independent implementation review. The freeze predates
launch; the preparation metadata in the committed configuration is preserved.

The launch returned exit 0 after 220.881 process-wall seconds. Complete status,
60,000 contiguous recorded update rows, every fixed endpoint and the exact
twelve-condition registry agree. There was no timeout, nonfinite termination,
checkpoint selection, replacement or extension. The engine used CPU FP32,
one thread, deterministic algorithms, TF32 off and highest matmul precision.
No cluster or provider action was involved.

Four native seed-0 model initializations were independently recreated. All
twelve saved initial parameter sets match their pinned v3 fresh tensors
exactly. The reviewed executed source constructs a new AdamW for each fit and
requires empty state; each provenance record reports zero initial entries.
This review did not reexecute the training trajectory.

## Independent numerical checks

Direct NumPy RNG reconstruction verified both original v3 streams, including
the consumed random QUERY draws, and all six derived address streams:
10,000 original batches, 30,000 derived batches and all 60,000 recorded update
identities. Only terminal selector, target and selector metadata change;
all sixteen ordered WRITE payloads remain matched. The seed-20003 validation
arrays and twelve prefit file hashes also match.

Independent reduction recomputed all 120 scheduled validations, 240 arm
scores and all 1,024 supported per-history baseline differences. All 333
run-file hashes, 300 overlapping per-fit manifest entries and 120 checkpoint
metadata records reconcile. Categorical scores are accuracy; paired contrasts
use error probability. Continuous scores and contrasts divide squared native
error by the fixed population variance 1/3.

Direct frozen encoder/core/head inference was performed at updates 500 and
5,000 for each of twelve trained conditions, plus all eight pinned baseline
evaluations. The independently chosen 32-evaluation subset matched 69,632
logit entries and 4,718,592 complete-state entries bitwise. Capture chronology
is sixteen WRITEs, no body and one QUERY. Post-WRITE and pre-QUERY states
coincide; GRU state has 128 hidden coordinates and LSTM state has 128 hidden
then 128 cell coordinates. Other scheduled checkpoints were checked through
raw reduction, metadata, hashes and saved chronology; they were not all
independently native-replayed.

At addresses 0/7/15, GRU categorical accuracy is 0.09375/1/1 and LSTM accuracy
is 0.046875/1/1. GRU continuous normalized MSE is
1.171140092/0.000048020/0.000031024; LSTM is
1.169030579/0.000281015/0.000015860. First passing scheduled evaluations agree
with the producer report and remain descriptive; no earlier weights replace
the fixed update-5,000 endpoints.

All supported baseline contrasts reproduce exactly. The baseline consists of
the four SHA-pinned v3 fresh update-5,000 models reevaluated, without retraining,
on the same new histories at QUERY 0 or 7. No primary baseline at QUERY 15
exists, because it lies outside the original trained selection support.
Historic v3 seed-20001 scores remain separate.

## Interpretation and limits

Each task has 128 distinct validation histories, shared across address
conditions, model classes, epochs and baseline reevaluations: 256
task-specific histories across the two tasks. The two R0 arms are identical
copies. There is one initialization/discovery seed per condition, not twelve
independent seed replications. No confidence interval, p-value, population
competence guarantee or equivalence inference is enabled.

An address-7 model and an address-15 model have different trained parameters.
Their successes cannot be combined into a claim that one model retains
multiple required items. The original all-eight-coordinate competence gate
has not been passed. Address identity is confounded with 15/8/0 later WRITE
updates; fixing the future query also changes required information load and
selection uncertainty. The observed pattern therefore does not isolate lag,
capacity, addressing, overwriting or credit assignment, and address-0 failure
does not establish architectural impossibility.

These results establish bounded singleton acquisition outcomes. No
periodicity forcing, covariance-dimension measurement, tangent contraction,
causal-memory intervention or Floquet closure was tested. Such claims require
a separately reviewed prospective protocol and independent trained seeds.

The independent receipt is
`runs/review/address-acquisition-v4-results-review-20261013.json`.
It binds executable replay checks, exact evidence hashes, counters, full
scheduled scalar trajectories and paired contrasts. The producer receipt
`f023febf294399660a0c17344d3ee0ad607d5ba7d4c4e7d8353b6b8a8b6c3aeb`
remains a separate artifact. No author source, prior report, training output
or scientific protocol was changed by this review.

