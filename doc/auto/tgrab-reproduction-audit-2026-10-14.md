# T-GRAB graph reproduction: recovered assets and debugging priorities

2026-10-14. This is a source/provenance audit with seven small engineering
unit checks. No graph model was trained or evaluated, no array payload or
checkpoint was loaded, and no new job was submitted. Three native agents
independently audited manuscript claims, code parity and the Mila archive;
the coordinator owns this integration.

## What has and has not been tested

**There has been no negative replication of the original graph result.**
The project's recent GRU/LSTM vector fits and 736-call calibration use different
models, inputs and objectives. Their failed competence gates and numerical
checks say nothing directly about T-GRAB graph collapse. Original graph
reproduction is now the immediate priority before further vector expansion.

The original [T-GRAB paper, Sections 3.1 and Appendix H.1](https://arxiv.org/html/2507.10183v1)
reports predictive performance across periodic graph tasks, including declining
change-point performance with longer repetition and declining memorization
performance with more patterns. GC-LSTM is among the strongest periodicity
methods. This supplies a behavioral reproduction target, rather than a
reported covariance-rank or tangent-spectrum result. Its evaluation uses
per-time F1 over candidate node pairs. The published settings use 100-node
ER graphs with probability 0.01 and 40/4/4 train/validation/test periods.

The inherited sources identify three separate conditions:

- **AB:** `Pdet(2,1)`, period 2. The Floquet introduction cites a perfect-accuracy
  GC-LSTM on this condition (`floquet_analysis/main.tex:69`).
- **AABB:** `Pdet(2,2)`, period 4. The draft's period-map and six reported
  magnitudes 0.41–0.44 describe this construction
  (`floquet_analysis/main.tex:436–510`). No recovered checkpoint connects it
  to the introduction's AB model.
- **Long repetitions:** `Pdet(2,32)`, period 64. This is the behavioral forgetting
  curve in `ACM___TGRAB_KDD_2026/kdd/tex/100appendix.tex:319–338`, actively
  included by that snapshot's `main.tex:353`. It is a different condition.

The older `ACM___TGRAB/kdd/main.tex` excludes its appendix; the KDD 2026 snapshot
actively includes it. An inactive local section or similar plot name cannot
stand in for a source-matched result. A low-dimensional driven orbit, tangent
contraction and loss of task-required information remain separate findings.
In particular, a period-4 orbit has a time-centered covariance rank ceiling of
three regardless of whether its retained state is useful.

## Original assets located on Mila

The owner directed the search to the TSA archive. A bounded read-only search
located:

- Source: `/network/archive/m/mehrab.hamidi/projects/lab/TSA`.
- Numerical candidates: `/network/archive/m/mehrab.hamidi/floquet_analysis`.
- Archived TSA HEAD: `403bd6f47e42f5fd98c1cf366da6891b955fc208`, branch
  `appendix`. Working-tree equality to HEAD is unverified; individual inspected
  text files have independent SHA-256 identities.

Four header-only inspections read 512 bytes in total, with no numerical payload:
`eigvals.npy` is complex64 with shape `(6,)`; `eigvecs.npy` is complex64 with
shape `(25600,6)`; `envelope_0.npy` is complex64 `(20,25600,6)`; and
`norm_env_0.npy` is float32 `(20,6)`. These are metadata, not independently
reduced eigenvalues or trajectories. The state-vector size is consistent with
full graph-state analysis; it does not establish the exact node/layer layout
or prove an implementation omitted nodes. The inspected source concatenates
flattened hidden and cell tensors, with layers stacked before nodes/features.

Twenty Jacobian files total 52,428,815,150 bytes (52.43 GB). They were neither
hashed, downloaded nor deserialized. Matching envelope, Fourier, PCA and
linearization plot names and additional `.npy` files are present. All originals
remain untouched. Metadata pins are in
[legacy-asset-candidates-2026-10-14.json](../../configs/graph/legacy-asset-candidates-2026-10-14.json).

No recurrent/link-head checkpoint was found in the enumerated user's TSA
scopes. The expected `TSA/res` and `TSA/saved_models` paths are absent, and the
user's scratch TSA link is dangling into a collaborator's scratch. That other
user's storage was not searched. The inventory records depth limits, excluded
paths and an oversized log not inspected; this is not a global absence claim.
No exact dataset/seed/run/checkpoint identity has been connected to the arrays.

## Concrete code problems and their evidential limits

The archived analysis file is
`TSA/train/DTDG/link_pred/trainer_gclstm.py`, SHA-256
`6f5b539648162a5abf258514c639c365b238b199dd656f8752e7bb74744e77d0`.
Its modification time is June 1, 2025; the inspected numerical candidates are
April 25, 2025. Modification times neither prove nor exclude a producer link.
This source currently reloads existing eigenpairs, rather than recomputing them.

1. **Complex operator mismatch, reproduced in two local unit checks.** The
   archived matvec casts an input through `.double()` at line 726 while declaring
   a complex LinearOperator at line 744. With an identity Jacobian and complex
   input, the exact selected function discards the imaginary part and violates
   complex linearity. This uses local Torch 2.6.0, no saved Jacobian or model.
   The independent reviewer accepted the scoped reproducer. The archived `eigs`
   call is commented at line 748 and fixed eigenpairs are loaded at 761–762;
   this does not establish what generated the original six values.
2. **Envelope time indexing.** Line 796 stacks `prefix @ eigvecs` unchanged for
   every time index, inside a loop that updates the prefix only outside that
   stack. Each saved tensor therefore repeats one prefix's action across its
   time axis in this source. The raw archived arrays have not yet been checked
   against this behavior.
3. **Window and alignment.** Lines 459–480 choose `T=20`, state slice `140:160`,
   but graph/feature indices start at `k`. A twenty-step propagator cannot be
   called the draft's four-step map without an explicit period/closure check.
   The offset is not automatically a phase error: 140 is divisible by four,
   and identical periodic inputs could align; actual capture timing and data
   equality must be tested. Lines 409–438 currently capture one state without
   subsequent evaluation updates, another source-level consistency concern.
4. **Unbound caches.** Existing Jacobians are reused by filename (701–715);
   eigenvalue/eigenvector saving is commented and fixed arrays are loaded.
   Selected code has no checkpoint/input/state hash binding. Re-running this
   file could silently combine old numerical files with new settings.

The separate public T-GRAB source is pinned at
`31293102b065feef7817ff86211427d7594edfdb`; it must not be conflated with the
archived analysis file. Its negative sampler at
`train/DTDG/link_pred/trainer.py:194–218` uses tuples of scalar Tensor objects as
membership keys. Three local Torch 2.6.0 unit checks reproduce a forced positive
edge returned as a negative. Legacy Torch 2.0.0 and the actual contamination
frequency remain untested. Two additional symbolic tests confirm conditional
fresh evaluation `consume target → score target → consume target` versus warm
evaluation `score → consume`. Under the ordinary inference wrapper, the affected
first target is the first replayed training graph; this alone does not prove
held-out test leakage. Static review also found test-score-dependent stopping,
test-dependent continuation caches and loss-plateau patience resets.

Two missing public entrypoint files were recovered at the pinned commit and
verified against Git blob IDs. They construct the trainer before calling
`run()`, which applies the seeds. No explicit earlier seed is present in the
entrypoint; imported-module side effects remain outside that conclusion.
All these findings require historical behavior to be recorded faithfully and
prospective corrections to be labeled. No original source was patched in place.

## Immediate reproduction path

1. Preserve and hash a bounded numerical subset; trace dataset, checkpoint,
   source and capture chronology. Audit stored eigenpair residuals and envelope
   construction with a frozen reader and appropriate separately admitted CPU
   resources. Inspect headers/schema before any `.pt` deserialization. A
   metadata match or six magnitudes below one is not sufficient acceptance.
2. If original checkpoints remain unavailable, reconstruct the published
   GC-LSTM setting with a pinned recurrent model **and link head**, data
   conversion, graph-generation/model seed roles, reset/carry-over behavior,
   environment and selection rule. Do not promise byte-identical recovery:
   even a reproduced configuration need not recreate the original RNG history
   or nondeterministic training trajectory.
3. Reproduce per-step and change-point F1 on `Pdet(2,32)` first. Keep AB and AABB
   as separate registered conditions for resolving the Floquet note. Reproduce
   persistence and EdgeBank under the identical pair mask and chronology.
   Historical source parity and corrected prospective behavior need separate
   explicit configurations; test-controlled stopping must not enter new science.
4. Once task behavior matches, capture complete hidden/cell state over every
   layer and node, establish phase-aligned orbit closure, and calculate ordered
   discrete tangent products. Use matrix-free methods with a tiny dense
   reference, solver residuals and finite-perturbation checks. Compute covariance
   spectra, absolute variance and task retention separately.
5. Diagnose a mismatch by data/checkpoint hash, source/dependency identity,
   prediction chronology, resets/caches, period, full-state layout and numerical
   tolerances. Retain a valid discrepancy; do not train or relax thresholds
   until the expected collapse appears.

The [graph protocol](graph-reproduction.md) and dependency plan now prioritize
this path. Further graph training still needs the recovered/reconstructed
config, reviewed measurement code and exact compute envelope; this audit grants
no allocation. There is currently no evidence of either a reproduced graph
collapse or a failed graph replication. The archive discoveries and code
reproducers make that next test substantially more concrete.

## Evidence identity

- Manuscript audit: `7aae4e51e9b629dfea6466f33e2aa1ca65a25b5026d551a105d836cb39eaabc3`.
- Public source parity audit: `aada82ee7be2fb2c516c513c4572e01c332edc8343b002249fbe369587f94427`.
- Mila inventory: `c847ed429110a01707e37adaefac5b97db87c79163e3f30584260f91f8229577`.
- Archived analysis review: `64a3c51b247c0fd883470f5eaec8c8611cae9b357e3089bdfdc32ce484d28264`.

Small source excerpts, receipts for all seven unit tests, source/metadata probes and
independent reviews remain under ignored `runs/review/`. This report records
verified identities and does not replace missing raw-data/checkpoint provenance.
