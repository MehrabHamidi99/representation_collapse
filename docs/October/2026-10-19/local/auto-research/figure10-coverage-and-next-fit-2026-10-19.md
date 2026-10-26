# Figure 10 coverage audit and the next complete fits

2026-10-19. Read-only audit of retained manuscript, figure, source and protocol
files. No deployment, source patch, fit, evaluation or scheduler operation was
performed by this audit. The coordinator reported live fit job **11321105**, run
`full-20261019T211030Z-6d89c657`, from source `6d89c657`; this report does not
independently establish that job's latest state or outcome.

## What Figure 10 actually covers

The source target is the appendix figure labelled
`fig:(2,N)_per_timestep`, captioned per-timestep F1 on **Pdet(2,32)**, with DTDG
methods on the left and CTDG methods on the right. Figure numbering is
version-dependent: this audit identifies the label, caption and included PDF
bytes rather than inferring the number from another manuscript's build.

The active local include chain inspected here is
`ACM___TGRAB_KDD_2026/kdd/main.tex:353` ->
`kdd/tex/100appendix.tex:312–338`. The selected PDF assets are
`figs/periodicity/Fixed_ER/per_timestep/(2,32)DTDG_avg_f1.pdf`, the corresponding
`CTDG_avg_f1.pdf`, and `figs/legends/all.pdf`. Both plots and the legend were
rendered and visually inspected. The older `ACM___TGRAB/kdd/tex/100appendix.tex`
contains the same target at lines 235–261, but its inspected `main.tex` ends
without including that appendix. Its separate `content.tex` is also not the
active body. These distinctions matter when attributing inherited findings.
This audit did not establish byte-for-byte equivalence of the local KDD draft
and the published arXiv Figure 10.

The figure assets identify these series:

- Left: **GC-LSTM, T-GCN, EGCN, GCN, GAT, Persistence**. EGCN is the figure's
  label for the EvolveGCN family; which H/O variant and exact constructor
  produced this particular asset remains to be established.
- Right: **CTAN, DyGFormer, TGAT, TGN, EdgeBank, Persistence**. Persistence is
  drawn in both panels; it is one shared baseline, not two separate models.
- Thus the full target has **nine learned methods plus two baselines**,
  eleven distinct methods and twelve panel placements. A single GC-LSTM curve
  covers only one learned method in the left panel.

The archived horizontal axis is evaluation-relative timestep, with ticks
0,32,...,224, over 256 targets. In the corrected implementation these are
absolute targets **2816..3071**; `evaluation_index = target_timestamp - 2816`.
The two plots show alternating 32-step motifs, eight actual change points,
and four 64-step periods. Their visual uncertainty bands are present, but the
assets alone do not specify whether each band is a standard deviation,
standard error or confidence interval. Do not assign a band convention without
recovering the plotting implementation or underlying numerical records.

The local manuscript explicitly describes a fall in F1 between change points
and rapid recovery after observing a changed graph. This is an attributed
behavioral observation in the inherited source. Neither the caption nor the
curves measure state covariance rank, tangent dimension, a Floquet spectrum or
causal retention. Its recurrence within each motif should also be distinguished
from deterioration over successive 64-step cycles; the archived curves appear
roughly repeated rather than demonstrating cumulative decline across cycles.

## Confirmed training facts and unresolved historical details

`ACM___TGRAB_KDD_2026/kdd/tex/100appendix.tex:35–44` states: 40/4/4 complete
periods for train/validation/test; a single message-passing layer; Adam at
1e-4; one snapshot per batch; five random seeds; validation-based selection;
and training-loss patience 100 **without a fixed maximum epoch count**.
For this primary dataset the split intervals are [0,2560), [2560,2816),
[2816,3072). The paper also reports GPU experiments, with runtimes across the
whole benchmark ranging from minutes to days; that broad range does not give
the runtime of this particular GC-LSTM figure.

The retained public source pin is
`alirezadizaji/T-GRAB@31293102b065feef7817ff86211427d7594edfdb`, recorded in
`configs/graph/tgrab-upstream-pin.json`. Relevant files were inspected under
`runs/graph/upstream-pin-20261012/source/`; they remain read-only source
snapshots. The GC-LSTM shell supplies `--patience=100 --num-epoch=100000`,
backpropagation window 1 and `backward_only_last`. The generic parser supplies
tolerance 1e-6. The retained launcher currently loops over **SEED=3457** and
its active method list contains only DyGFormer, while its generic run loop
seeds with `args.seed + run_idx`. Therefore **five seeds is a paper fact;
seed values 0..4 are not a verified historical Figure 10 fact**. They are the
proposed corrected cohort. The current public launcher is not an executable
manifest of all historical figure runs.

Historical checkpoint bytes, selected epochs, per-seed curves, exact plotted
run roster, band reduction and figure-producing plot script have not been
recovered by this audit. The public pin postdates the original arXiv submission;
its exact relationship to the original figure-producing revision remains
unknown. GPU and CPU initialization/numerics also need not produce identical
trajectories. Reproducing a qualitative shape is consequently a corrected
reconstruction, not historical bitwise replay.

The existing [reconstruction specification](../../../2026-10-26/local/auto-research/tgrab-checkpoint-reconstruction-v1.md)
records why the public trainer is not used unchanged: model construction can
precede seeding; repeated runs reuse model/optimizer objects; cached predecessor
state can cross boundaries; a cold evaluation branch consumes a target twice;
test is inspected whenever validation improves; perfect cached test F1 can
stop training; and the decreasing-loss early-stop inequality can reset on a
plateau. The actual retained implementations confirm the last two issues in
`train/DTDG/link_pred/periodicity/trainer.py:74–78` and
`utils/early_stopping.py:50–70`. Paper prose, public implementation and corrected
protocol must be reported separately. Historical test-dependent stopping must
not be silently reintroduced to make a curve match.

## What the current job can establish

The [current fit protocol](figure10-full-fit-2026-10-19.md) is a fresh **seed-0
GC-LSTM** under frozen reconstruction v1: CPU FP32, one CPU thread, hidden
width 128, one recurrent layer, Chebyshev K=2, negative seed 1, 2,559 updates
per complete epoch, validation-only earliest-tie selection and corrected
chronology. Its limits are the first of 200 complete epochs, corrected loss
patience 100, or the one-hour entire-pilot cap. Its registered supervisor work
interval is 3,480 seconds, outer bound 3,570, Slurm allocation 3,600. These
limits remain facts of that attempt even though the owner subsequently
removed a compute-budget obstacle to further retraining.

Only independently audited complete candidates can be evaluated. The selected
recurrent and link-head weights must load strictly; fresh zero-state replay of
all 2,816 train plus validation graphs must match the selected complete hidden
and cell terminal state bit-for-bit before the sealed test opener is used.
A deadline-truncated fit remains capped even if it has a selected candidate
and evaluation succeeds. It supplies one seed, one contiguous graph realization
and no between-seed uncertainty. The plotting CLI's extra motif-mean panel is
an explanatory summary, not a second original Figure 10 panel.

## Smallest honest versioned extension

Do **not** lengthen the deployed v1 job, mutate its configuration or relabel its
output as converged. The user has authorized additional retraining; the
remaining work is to register a concrete, executable v2 rather than ask again
for already granted compute permission.

1. Finish and audit the present fit, then perform its separately admitted
   selected-checkpoint evaluation once. Preserve all partial updates, its
   selected epoch and actual stopping reason. Use the observed complete-epoch
   walltime, update throughput and memory to size the next allocation. A
   256-update prefix timing is not a complete-epoch timing estimate because
   replay, validation, fsync and candidate publication have material cost.
2. Freeze a **corrected cohort v2** with explicit fresh run identities for
   seeds **0,1,2,3,4**, each with at most 200 complete epochs, 511,800 updates,
   patience 100 and unchanged relative-loss tolerance 1e-6. Preserve all
   architecture, model-source, dataset, optimizer, batch, precision, negative
   sampling and chronology settings. Keep negative seed 1 for this first
   model-seed cohort so the changed factor is declared; this is conditional
   variability on one graph/negative-sampling policy, not graph variability.
3. Give v2 a newly frozen allocation/runtime envelope sized from the **upper
   observed complete-epoch cost** with setup/finalization margin, accommodating
   200 epochs when the scheduler permits. Record the actual numerical walltime
   before submission; no unmeasured hours/day estimate is certified here. A
   conservative ceiling must still produce an explicit capped exit if reached.
   If scheduler maxima prevent one uninterrupted run, implement and verify
   optimizer/RNG/recurrent-state resume before splitting a fit. Existing
   selected-inference checkpoint bundles deliberately cannot resume training.
4. Restart seed 0 from its original fresh initialization under v2; do not
   append to a finalized v1 fit or substitute its validation-selected weights
   for the final partial optimizer state. Require numerical parity of a small
   seed-0 v2 prefix against v1 under the same frozen native environment. Extend
   seed application and provenance checks for seeds 1..4, with fresh model and
   optimizer construction for every independent run.
5. Freeze the five-run aggregation before reading their held-out rows: align
   absolute targets, average the five per-target F1 values, retain the full
   five-by-256 matrix and seed-level summaries, and state the chosen band
   convention. Recommend mean plus sample standard deviation as a transparent
   corrected-cohort description, explicitly not a verified reconstruction of
   the original band's convention. Never treat the 256 correlated targets or
   eight motif blocks as independent training replicates.

**Two hundred epochs is still a prospective corrected cap.** If patience has
not fired at epoch 200, report that fact and do not claim the paper's
uncapped stopping rule was reproduced. A later convergence extension must be
another frozen version, selected using training/validation evidence and fixed
before any additional test access. Do not repeatedly inspect test curves to
choose duration, seeds, model changes or a favorable checkpoint.

## Rigid dependencies that v2 must address

Changing only a command-line deadline is both insufficient and scientifically
misleading. The actual v1 guards include:

- `configs/graph/tgrab-reconstruction-v1.json` SHA
  `22bef6bf781c5bfb5493dd5e03169ea0e658b2463ccaa9ba09189af454508f1f`;
  canonical `candidate` SHA
  `dea0e7684f52a0a53cba5072e77778d22995cb0b0e30f3352a6e349b89628092`.
  The candidate contains model seed 0 and the 3,600-second cap.
  `graph_reconstruction.validate_frozen_config` requires these exact values
  and hash, including the one-hour cap.
- `tgrab_figure10_native_fit.make_seeded_components` and construction-receipt
  verification require seed exactly 0. The fit lifecycle itself requires the
  exact frozen `FitPolicy` (200 epochs/100 patience/1e-6), which may remain
  numerically unchanged for the 200-epoch cohort. Higher epoch caps would
  require another policy/schema version.
- `graph_data.py` checks the immutable **conversion-time v1 config identity**
  when reading a verified dataset manifest. Do not regenerate the graph or
  rewrite its conversion provenance merely to change training seed/walltime.
  V2 must distinguish `conversion_config_sha256` from
  `executed_training_config_sha256`, explicitly permit the accepted v1 dataset
  as an input, and retain identical raw, manifest, split and semantic hashes.
- Selected-checkpoint schema v1 SHA
  `db2baaa64f891533f5c2c36229dd250733f675a0a2045d0436fc93d3145f2acc`
  contains constant v1 candidate/config identities and seed-0 initialization.
  The codec hardcodes its schema path/hash and candidate identity; the encoder,
  ledger context, source pins, admission and evaluator depend on that identity.
  V2 therefore needs its own versioned schema/codec dispatch and run roster,
  not a relaxed check that silently accepts arbitrary metadata. Preserve v1
  decoding and tests. The existing tensor catalog/weight layout may remain
  unchanged if its native shapes/order are unchanged; do not change its bytes
  just to renumber a training protocol.
- `tgrab_figure10_full_fit.py`, its supervisor/job, bootstrap bindings and
  evaluator job have v1 config/seed/runtime pins; the reporter configuration
  and plotting acceptance are also seed-0-specific. New source closure and
  admission receipts must bind the exact new implementations and arguments.
  Retain native numerical parity evidence for the unchanged wrapper/cell/
  environment, while explicitly reviewing changed orchestration; an old
  receipt does not automatically authenticate new source bytes.

A minimal implementation can add a v2 training contract/validator and
schema-aware adapters around the unchanged native numerical backend and v1
verified data reader. Store per-run seed and execution envelope separately
from a shared scientific-core identity, while retaining the immutable full
per-run config digest. This avoids editing inherited source snapshots and
avoids cascading training-policy edits into already verified dataset bytes.
It still requires targeted seed/parity/provenance/selection/evaluation tests
and an independent engineering review before new jobs.

## Next experiments, in priority order

**First: complete the GC-LSTM behavioral reproduction.** Audit and evaluate
v1, then run the newly registered seed-0 v2 fit long enough to reach its
200-epoch-or-patience stopping rule. Execute the other four preregistered model
seeds under the same contract; report every outcome, including valid negative
results. Overlay no inferred original numeric curve. Compare transition,
nontransition and within-motif early/late F1 using a frozen reduction and show
raw per-target curves. Failure to recover the archived decrease does not by
itself prove a bug; first test chronology/load/RNG/metric parity and account
for corrected stopping, seeding, device and convergence differences.

**Second: original DTDG comparators and baselines.** Persistence can be exactly
computed from the verified adjacent primary graphs with the same threshold/
ordered-pair contract, without fitting. Pin EdgeBank's exact public definition
before its later CTDG-panel inclusion. Prioritize T-GCN after GC-LSTM because
its archived purple within-motif decline is directly relevant; then the exact
EvolveGCN variant, GCN and GAT to complete the left panel. Each needs original
constructor/hyperparameter, state/reset, source and checkpoint contracts;
GC-LSTM's 26-tensor codec cannot be relabelled as another architecture.

**Third: complete the CTDG panel.** Reproduce TGAT, TGN, DyGFormer and CTAN,
plus EdgeBank and the same Persistence reference, with their original
snapshot-training/neighbor-sampling/event-ordering contracts. The manuscript
states most-recent neighborhood size 20 when applicable, but that broad
statement is not a sufficient model-specific executable specification. CTDG
memory and sampling are not interchangeable with the GC-LSTM state adapter.
Recover per-method scripts and hyperparameters before launching those fits.

**Fourth: test the collapse mechanism on competent, fixed models.** Freeze
complete-state H/C recording, phase-aligned covariance spectra with absolute
variance, finite-time tangent diagnostics and matched memory/ordering
interventions. Confirm orbit/Jacobian periodicity before a Floquet label.
Separate within-motif behavioral decline from useful-state information loss
and from the time-only rank ceiling of a periodic trajectory. Carry the
matched-retention-horizon question in the research charter into cross-modal
experiments only after the graph reproduction and measurement checks.

This ordered path advances the requested original benchmark reproduction
without treating one capped seed as the full figure, asserting historical
seed identities that were not recovered, or presenting F1 alone as a
representation-collapse result.
