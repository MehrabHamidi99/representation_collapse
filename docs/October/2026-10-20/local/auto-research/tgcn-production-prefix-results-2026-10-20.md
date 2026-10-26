# T-GCN production training prefix — 2026-10-20

**The first empirical production T-GCN prefix completed 256 training updates
and passed independent compact execution verification.** Mila job **11332237**
completed `0:0` in **30 seconds** on `cn-f003`, using one CPU, 2 GiB and zero
GPUs through cluv. Its source was
`371aba996af02826df5fee03abb8a5223b9849b5`.

This is actual training on the accepted graph training sequence, distinct from
the earlier [synthetic calibration](tgcn-native-calibration-results-2026-10-20.md).
It completed **zero epochs**. It establishes execution of a short training
prefix; it establishes no competence, full fit, held-out score, F1 decline,
representation collapse or Figure 10 reproduction.

The [machine-readable result summary](../../../../../doc/auto/tgcn-production-prefix-results-2026-10-20.json)
and [compact evidence inventory](../../../../../doc/auto/evidence/tgcn-production-prefix-2026-10-20/inventory.json)
retain the original execution records and the reporting-repair lineage. The
[frozen prefix protocol](tgcn-production-prefix-protocol-2026-10-20.md) and
[prelaunch review](../../../../../doc/auto/evidence/tgcn-production-prefix-2026-10-20/prelaunch-review.json)
remain the prospective scope for this attempt.
The [independent engineering review](tgcn-production-prefix-independent-review-2026-10-20.md)
records the acceptance boundary and remaining full-fit work.

## What ran and what was verified

The calibrated native TGCN has one recurrent layer, 100 one-hot node inputs,
hidden width 128 and the public link head. Persistent state is hidden only,
with canonical node IDs 0–99; there is no LSTM cell component. Native inventory
retained **137,472 recurrent** and **49,537 head** parameters, CPU FP32 tensor
shapes and native module-state versions. Python, NumPy and Torch used seed 0,
with a separate seed-1 negative generator. The synthetic calibration's positive
head fill and dropout-seed override were not used in this empirical prefix.

Training targets were **1–256**. Each step detached carried hidden state,
consumed exactly its predecessor **0–255**, then scored its target and performed
one witnessed Adam update using positive BCE plus an equally sized,
source-preserving negative sample. The entire training file was checked under
the existing manifest and semantic integrity contract; numerical work used the
prefix only. No validation or test payload was opened, no selection occurred,
and no checkpoint was written.

Independent verification accepted:

- **256 attempted, completed and confirmed optimizer updates**.
- **256 chronological loss rows**, matched to their raw loss representations.
- **512 durable optimizer-journal records**, with paired pre-update intents,
  outcomes, context, canonical hashes and successful full readback.
- **256 actual offline W&B history rows**, matched to losses and counters,
  together with the result artifact and successful terminal framing.
- The actual scheduler outcome, source closure against Git, pinned calibration
  and data metadata, runtime settings, initialization references, native
  inventory, module modes and final accounting.

The reported mean training loss is **1.3214061078615487**. It is the mean of
these 256 update losses, not a validation measure or an estimate of learned
competence. A short prefix's elapsed time does not establish full-fit runtime,
convergence or the appropriate training budget.

The runtime was CPython **3.10.19**, Torch **2.0.0+cpu**, PyG **2.6.1** and
PyG Temporal **0.54.0**, with one intra-op thread, one inter-op thread and
deterministic algorithms. The complete exact versions and source identities
are retained in the [original result](../../../../../doc/auto/evidence/tgcn-production-prefix-2026-10-20/prefix-result.json).

The terminal hidden-state metadata describes a **training-carried**
`H[1,100,128]`, next target 256 and last consumed graph 255. This state was
computed during changing-parameter training, before the target-256 parameter
update. Its hash is not evidence of a fresh replay at fixed final weights.
No hidden-state payload was written or independently reanalysed by the
compact verifier, and native inference was not independently executed again.

## Reporting failure and repair: no training retry

The training job's own offline reporting completed successfully. A separate
independent verification attempt, **001**, completed its compact numerical and
execution checks and wrote a provisional verification record. Its final
reporting call then failed with a `ValueError`: the metric name
`raw_loss_rows_verified` triggered the reporter's prohibition on raw-data keys.
Its failed reporting receipt and provisional record remain unchanged in the
evidence directory. **That record was not promoted to canonical acceptance.**

The fix changed only the reporting metric to
`training/loss_rows_verified`. The canonical acceptance field
`raw_loss_rows_verified` and numerical verification logic were unchanged.
A pure regression exercised the actual final metric expression through the
real reporting privacy guard.

Verification attempt **002** rechecked the same collected artifacts under a
fresh offline reporting run ID. It completed, and its reporter returned
`finish_returned=true`; only then was its verification record promoted to
[canonical acceptance](../../../../../doc/auto/evidence/tgcn-production-prefix-2026-10-20/acceptance.json).
There was **no training retry, second native prefix or new model execution**.
The [repair lineage](../../../../../doc/auto/evidence/tgcn-production-prefix-2026-10-20/reporting-repair-lineage.json)
pins both reporting runs and preserves their different outcomes.

All reporting here is offline. The actual retained histories and artifacts
were verified locally; **cloud delivery and remote history remain unverified**.
The independent audit verifies compact arithmetic and provenance, not an
independently repeated native training run or raw hidden-state reanalysis.

## Remaining work

A T-GCN-specific primitive checkpoint and tensor catalog, finalized
validation-only selection protocol, strict dual-component loader and fresh
fixed-weight hidden-state replay gate remain to be implemented and reviewed.
The GC-LSTM H+C checkpoint format cannot stand in for hidden-only T-GCN state.

A separately frozen full-fit policy must specify stopping, operational caps
and the intended model-seed cohort before evaluation. After the fit is finalized and execution/replay checks pass, each selected checkpoint needs a
single controlled held-out pass and the existing fixed descriptive phases and
persistence controls. No such held-out T-GCN result is available from this
prefix. Matched ordering interventions and causal retention controls remain a
separate scientific requirement for the project's novelty claim.

The public evidence contains compact metadata, scalar loss rows and optimizer
chronology. It excludes graph payloads, weights, recurrent and RNG state
payloads, W&B streams, credentials and provider configuration. Initialization
and state hashes remain references in the original result rather than raw
state copies.

The [full-fit readiness audit](tgcn-full-fit-readiness-2026-10-20.md) maps the
reusable epoch engine and the required T-GCN-specific adapter, codec and selection
work. Competence assessment remains distinct from faithful benchmark reconstruction.
