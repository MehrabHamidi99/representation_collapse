# Competence diagnostic constructors and oracle

The additive `src/representation_collapse/diagnostics.py` module supplies
the local fixtures needed to start the competence-recovery ladder. It does
not change the frozen v1 training or periodicity configurations.

- `paired_batch_with_zero` adds R0: sixteen WRITE observations and one QUERY,
  identical paired arms and no distractor body. At R≥1 it delegates to the
  existing constructor, preserving its outputs and random-number consumption.
- `single_write_episode` emits the addressed WRITE followed immediately by
  its QUERY, with either a specified required coordinate or a sampled one.
- `streaming_register_oracle` stores eight required-coordinate values, ignores
  nuisance writes and distractors, and applies last-WRITE-wins semantics.
  It rejects unwritten targets and malformed or leaking observations.
- `shuffle_memories` permutes WRITE payloads; `shuffle_queries` changes each
  terminal query to a different required address. Both preserve input dtype,
  leave the caller's array unchanged and return new oracle-consistent targets.
  Query shuffling requires all eight required registers to have been written.
- Baseline helpers report uniform categorical cross-entropy, sample
  zero-predictor MSE and MSE normalized by the fixed population variance 1/3.

Example, using the repository's existing environment:

```python
import numpy as np
from representation_collapse.diagnostics import (
    paired_batch_with_zero, streaming_register_oracle,
)

batch = paired_batch_with_zero(np.random.default_rng(17), 128, 0, "continuous")
predictions = streaming_register_oracle(batch.inputs, "continuous")
np.testing.assert_array_equal(predictions, batch.targets)
```

Run the focused regression tests with:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_diagnostics.py -v
```

These are explicit-memory reference fixtures, not learned-model competence.
For corruption controls compare the new predictions with the **original**
targets; compare with the returned new targets to check oracle consistency.
A random permutation can be the identity, repeated categorical values can
match, and different query addresses can hold equal values. Therefore no
per-episode disagreement guarantee is valid. Regression fixtures use known
distinct values and fixed seeds to test the intended changes deterministically.

The module validates exact type/address flags, payload encoding and a single
terminal QUERY. It supports repeated and interleaved WRITEs for streaming
checks, but is not a training-protocol validator. Stage-aware training config,
three-phase state capture, causal
interventions/shams and seed-level statistical analysis remain separate
Gate B implementation work. See the [design](experimental-design-v2.md) and
[implementation audit](implementation-audit.md).

## Additive coordinate-wise competence assays — source `ec1c489`

The tested [competence module](../../src/representation_collapse/competence.py)
adds four distinct native assays: one WRITE followed immediately by QUERY,
the full sixteen-WRITE prefix with no body (R0), and that full prefix with
R1/R2 bodies. It draws 128 episodes per required coordinate and arm, preserving
paired memory/QUERY identity. Immediate recall has one arm; R0/R1 paired
arms are identical. The eight required coordinates must each meet
accuracy ≥0.95 for categorical recall or population-normalized MSE ≤0.05 for
continuous recall, using the fixed Uniform[-1,1] variance 1/3. The inclusive
metric-boundary comparison has a frozen four-ULP FP64 allowance; this does not
relax a state/adapter tolerance. Aggregate accuracy cannot hide a failed,
missing or undersampled coordinate.

`make_competence_batch` freezes assay inputs; `evaluate_competence` scores
decoded native predictions after the terminal QUERY; `assess_stage_competence`
checks the score portion of ladder-stage eligibility. Test records cannot
select a checkpoint, and oracle fixtures contribute zero trained-seed evidence
units. The module records protocol/batch hashes and explicitly labels paired
continuous loss differences as **raw squared error**, periodic minus control.
These contrasts are distinct from the normalized competence metric. Sixteen
focused competence tests pass after a finite-score overflow repair; the full
suite passes 82 tests without skips. The repair rejects overflowing coordinate,
aggregate or normalized metrics before JSON serialization, without clamping
scores or changing gates. It is newer than the immutable ec1c489 cluster snapshot.

The caller's phase label is an assertion, not proof of uninterrupted native
execution. Training-update boundaries, gradients/parameter changes, fixed-batch
overfit and untouched IID generalization, checkpoint/LR selection and three-phase
state capture still require integration. No trained checkpoint has passed the
new gates, and no new training was launched. R16 is outside selection eligibility
and must remain reportable even if its competence fails. See the prospective
[competence recovery protocol](competence-recovery-plan.md).
