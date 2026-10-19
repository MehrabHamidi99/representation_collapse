# Preregistered deterministic Figure 10 controls

Frozen before any selected-model test access, 2026-10-19. These analytic
controls support interpretation of the seed-0 Pdet(2,32) behavioral
reconstruction; they do not measure representation dimension or causal memory.
No tuning, learned parameters, checkpoint access or native model calls occur.

`code/figure10_deterministic_baselines.py` reads only the hash-pinned converted
manifest via the frozen standard-library `graph_data.read_manifest` metadata
reader. It validates primary raw/source/config/encoding/semantic identities,
the full frozen dataset specification, and converter source hash. It never
calls a split opener or reads train, validation or test payload files. Its two
known A/B edge sets come from `pattern_pairs` metadata, whose generator phase
is fixed before model evaluation. This is explicit generator knowledge, not
an unseen-task generalization test.

All controls predict the 9,900 ordered off-diagonal node pairs for targets
2816..3071. The true pattern is A when floor(t/32) is even, B otherwise. The
fixed controls are:

1. Previous-graph persistence predicts exactly the edges present at t-1.
2. Known-generator phase counter predicts the known A/B set for t. This is a
   descriptive oracle upper bound with privileged pattern/phase knowledge;
   perfect F1 is not evidence of learned memory.
3. Always-negative predicts no positive edges. Its F1 is zero for the required
   nonempty patterns; accuracy can remain high because edges are sparse.

For true edge set T and predicted set P, TP=|T∩P|, FP=|P\T|,
FN=|T\P|, TN=9900-TP-FP-FN, and F1=2TP/(2TP+FP+FN), with zero for a zero
denominator. Primary statistic is the unweighted mean of all 256 target F1
values. Separately report 8 transition targets t mod64∈{0,32}, 248 other
targets, all per-target counts/F1 and eight consecutive 32-step block means.
Require actual pattern change to equal expected phase transition.

Persistence is perfect on all 248 nontransition targets. At transitions its
F1 is 2|A∩B|/(|A|+|B|), identical in both directions even when precision and
recall differ. Its temporal mean is (248+8F1_transition)/256. Therefore a
large average F1 can coexist with complete failure at each change point;
compare selected-model transition behavior explicitly. The oracle and
always-negative controls give temporal F1=1 and 0 respectively. These are
analytic predictions, not observed trained-model results.

The CLI requires `--manifest`, `--manifest-sha256`, and a fresh `--output` JSON
file, created exclusively and fsynced. It retains manifest/raw/config/converter
and baseline-source identities, all metric rows and reductions, and explicitly
records `split_payloads_opened=false` and `scientific_acceptance=false`.
Manifest reading checks references and semantic metadata; no claim is made
that this independently revalidates split payload bytes or replaces the
conversion audit. A model agreeing with or underperforming persistence does
not alone establish internal representation collapse.

CLI execution must enter `ExperimentReporter` in explicit offline mode under
project `representation-collapse` before manifest reading or numerical work.
Pure `calculate` unit tests create no runs. Each invocation has a fresh unique
reporting run ID and sibling `<output-stem>.wandb/` evidence directory; existing
output or reporting paths are rejected before side effects. The sibling must
remain in ignored runtime storage, as must the numerical output. No credentials,
online access or sync is used. Log all 256 rows for each of the three controls
and each final reduction with 771 strictly increasing event steps. Log a compact
metadata artifact binding the final numerical JSON hash, source and manifest
identities. Publish the numerical output exclusively only after reporting
finish succeeds. A reporting failure retains its failed offline receipt and
publishes no numerical result. Offline retention does not verify dashboard
delivery; authenticated sync and remote verification remain separate work.
