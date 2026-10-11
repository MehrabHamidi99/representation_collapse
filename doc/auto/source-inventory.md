# Scientific source inventory

Inventory date: 2026-10-11. Paths and line locators refer to inherited files
observed at setup. Preserve those directories and record hashes before a
reproduction; bibliography entries and figure files are not executable evidence.

## Inherited local sources

- `floquet_analysis/main.tex`: project-specific draft. Introduction at line 66;
  periodic-task definitions at 102–140; ODE surrogate construction at 160–210;
  discrete cycle-map product at 299–340; AABB interpretation at 443–455;
  preliminary six-multiplier values at 502–510; local-linearization check at
  545–559; Fourier interpretation at 679–716; PCA/clock interpretation at
  730–747. Treat its empirical statements as preliminary and unverified.
- `floquet_analysis/ref.bib`: inherited references; verify each relevant
  primary source before extending theory or reusing a bibliographic claim.
- `floquet_analysis/figs/interpret/`: inherited PNG figures for PCA, linearization
  errors, envelopes, frequencies and PSD. No producing code, raw arrays,
  checkpoint identity or job provenance was discovered with those figures.
- `ACM___TGRAB/kdd/main.tex`: benchmark manuscript entrypoint. Active inputs
  at 307–324 select several `_ai` section variants. Keep both original and
  active variants; `content.tex` is legacy material, not the active body.
- `ACM___TGRAB_KDD_2026/kdd/main.tex`: another manuscript entrypoint. Active
  section selection at 327–344 and appendix include at 353. It shares much
  material with the earlier folder but is not an identical copy.
- `ACM___TGRAB/kdd/tex/050periodic.tex:18–32`: deterministic versus stochastic
  periodic tasks; `k` is pattern count and `n` is block length. Same original
  file contents observed in the KDD_2026 tree; active `_ai` variants differ.
- `ACM___TGRAB/kdd/tex/040method.tex:36–37`: exhaustive-node-pair per-time F1
  evaluation and target-node restrictions for other families.
- `ACM___TGRAB/kdd/tex/100appendix.tex:23,28–37,96,105–106,231–232`:
  upstream code/data links, chronology, training details, model-library leads,
  node/edge features and reported historical compute. The KDD_2026 appendix
  differs, including extra analyses. Historical runtime and GPU descriptions
  are not present-day resource estimates or authorization.
- `ACM___TGRAB/kdd/tex/060cause.tex` and `070long.tex`: delayed memory and
  graph-distance task families for graph extensions. Treat manuscript rankings
  as inherited statements until code/data/artifact reconstruction.

At initial inspection, searches of this repository for `.py`, `.ipynb`,
`.ckpt`, `.pt`, `.pth`, `.csv`, `.json`, `.sh`, `.yaml` and `.yml` did not
discover an experiment implementation, checkpoint, or raw numerical run.
This is a statement about the local inherited tree, not other storage or
upstream repositories. New setup infrastructure must be distinguished from
the missing legacy scientific pipeline.

## External recovery leads checked on 2026-10-11

- [T-GRAB upstream repository](https://github.com/alirezadizaji/T-GRAB):
  accessible public source, with dataset/model/train/scripts directories and
  GC-LSTM among supported methods. Pin and audit a commit before use;
  availability does not prove reproduction parity. Upstream batch Slurm
  scripts must be adapted to this project's cluv plan rather than executed
  wholesale.
- [T-GRAB dataset card](https://huggingface.co/datasets/Gilestel/T-GRAB):
  accessible dataset lead; visible columns include endpoint/time/features and
  split masks. Freeze revision, configurations, checksums and split semantics
  before training. The card's outer `train` container is not a substitute for
  auditing the benchmark's train/validation/test masks.
- [GC-LSTM primary paper](https://arxiv.org/abs/1812.04206): original
  architecture reference. Identify whether the exact benchmark implementation
  differs before asserting architectural equivalence.
- [PyTorch Geometric Temporal documentation](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html):
  library lead from the local appendix; use a pinned source version for adapter
  semantics rather than mutable latest documentation.

## Primary methodological sources

- [Cvitanović et al., Cycle stability](https://chaosbook.org/chapters/stability.pdf):
  cycle-map Jacobians and Floquet multiplier interpretation. A continuously
  updated textbook; freeze the used edition/file when citing in a manuscript.
- [Vogt et al., On Lyapunov Exponents for RNNs](https://arxiv.org/abs/2006.14123):
  input-driven recurrent stability and numerical Lyapunov methodology. Pin
  paper version for a final literature record.
- [Hu and Sompolinsky, covariance spectra of recurrent networks](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010327):
  covariance participation ratio and the distinction between covariance and
  recurrence spectra. Its particular random linear models are not a theorem
  about trained GC-LSTM.
- [Roy and Vetterli, Effective Rank](https://core.ac.uk/download/pdf/147929764.pdf):
  normalized singular-value entropy. In this project distinguish rank of the
  data matrix from rank of the covariance matrix explicitly.

## Evidence ledger

- **Observed:** manuscript and figure files exist; source entrypoints and
  upstream recovery links inspected.
- **Preliminary/unverified:** inherited graph performance claims, six reported
  multiplier magnitudes, spectral/PCA narratives and historical compute usage.
- **Derived/protocol:** mathematical controls and proposed measurement rules
  in [research program](research-program.md), including corrections to unsafe
  inferences from the draft. They are not new trained-model observations.
- **Not recovered:** legacy checkpoint, complete seeds/config/environment,
  analysis source and raw result arrays with job identity.
- **Not run at initial inspection:** graph reconstruction,
  dimension-measurement calibration, modality pilots, and any new empirical
  generalization test. The setup adds CPU calibration code/checks; their
  execution and results belong in the dated notebook. Synthetic calibration
  validates measurement fixtures and is not new graph-model evidence.

Update dated reports with newly recovered immutable references and validation;
do not silently promote available upstream source into verified local evidence.
