# Independent review of realized input-order statistics

2026-10-16. **Accepted as descriptive engineering infrastructure.** This
implements an input-statistics requirement in experimental-design v2, not a
trained-memory result or a new admitted experiment. The review used synthetic
unit arrays and literal arithmetic only; no datasets, models, SDK runs, provider
calls, remote work or cluster resources were used.

## Fixed reviewed inputs

- `src/representation_collapse/input_diagnostics.py`: SHA-256
  `45a17f0449a783364bce5b3eba1160acb3855f2c026c3751a9950d2b2a42c8d4`.
- `tests/test_input_diagnostics.py`: SHA-256
  `52a3cbaedf6219189a3acacd013db7cdd9ace283eeeecc4907b1950e8eee08fa`.
- `tests/test_input_diagnostics_independent.py`: SHA-256
  `2d0249beb0af3e666a11dbee237deb95e0bed2410a526c67569c6bd0e7985dbd`.
- [Diagnostic protocol](matched-input-statistics-v1.md): SHA-256
  `31f47574974d8d7dad0c9f3722e74717e044d2bc0b569815b550f5e0dead62ee`.
- Existing `vector_task.py`: SHA-256
  `df3843330ac9c59205823c8d55868e7878e903c0065c5af8bc89d1308243a297`;
  no motif, constructor or task changes were made.

The focused command
`PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p 'test_input_diagnostics*.py' -v`
passed **15 tests in 0.052 seconds**, consisting of eight author tests and
seven independent tests. The test driver checked all five hashes immediately
before and after execution. Its log, source identities and receipt live under
`runs/review/input-diagnostics-20261016/independent/`.

## Precision finding repaired

The inspected version first called `np.asarray(features)`, then checked the
resulting dtype and integer range. A mixed Python sequence could therefore
round a large integer inside the helper before the range check. A tiny review
reproducer passed `[[2**53+1,0.0],[2**53+1,1.0]]` and observed a reported first
mean of **9007199254740992.0**, rather than the requested integer
**9007199254740993**. This was an actual unit-level observation, not a model run.

The repaired API requires an explicitly typed NumPy array before coercion.
Independent list/tuple regression fixtures now require refusal. Existing typed
arrays and the body's internally constructed one-hot array remain supported.
This moves ownership of any earlier sequence-to-array conversion explicitly to
the caller; a float array cannot certify that the original data was never
rounded upstream. Signed/unsigned integer arrays outside the exact float64
range and extended floating precision are still refused before computation.

## Independent numerical checks

A literal complex DFT, written without `np.fft`, reconstructs every reported
one-sided bin for length-five and length-six two-feature arrays. It verifies
the T-squared normalization, doubled interior bins, undoubled even Nyquist bin,
frequency units and Parseval energy. A separate `Fraction` oracle verifies all
cyclic covariance entries for a nonsymmetric lag-one example, including the
forward `(t+k) mod T` direction, per-feature body centering and denominator T.
The labels correctly describe wrapped descriptive statistics; the wrap is not
an extra physical recurrent transition or an independent observation.

Empty, singleton and constant arrays retain undefined normalized power with
an explicit false flag and null value, rather than inventing a normalized
spectrum. Read-only float16 arrays retain their original values and dtype
identity while computation uses an independent float64 copy. Precision and
work-bound tests cover both exact integer endpoints, signed/unsigned violations,
extended floating precision where available, booleans/complex arrays, excessive
dimensions/scalars, reported covariance entries and nominal product work.
Author cases also preserve nonfinite/overflow failures and scale/offset checks.
The documented underflow/roundoff limitation remains: zero measured energy
does not prove mathematical constancy or lost memory.

R0/R1 identical anchors and all 20 distinct R2 motif pairs pass the author
tests. Independent R3 fixtures at dwell 1 and 3 include allowed local reuse of
the final motif in a globally non-tiling control, validate literal cyclic
bigram/trigram counts and retain unequal final-motif exposures (3 versus 2).
These dwell fixtures test the helper; they do not change Gate B's frozen q=1.
One realized pair can share cyclic trigrams while another differs. The module
does not assume that every matched pair differs at trigram order, and it
retains the actual uncontrolled higher-order statistics.

## Accepted scope and open work

The source preserves the registered motif library, equal horizon, low-order
matching, start/end context and final-motif controls. It does not certify cue,
query, nuisance or checkpoint pairing; these are outside a body-only API and
remain the orchestrator's responsibility. No new causal-memory threshold,
competence rule, statistical estimand or confirmation rule was introduced.

Caller-supplied encoder features are measured as provided. The helper neither
invokes nor authenticates an encoder, checkpoint or capture point; future
orchestration must verify and record those identities and initialize mandatory
reporting. The helper is not yet integrated into a newly admitted campaign.
Input periodicity, spectral lines, matched counts or realized higher-order
differences establish neither periodic hidden trajectories/Jacobians nor a
Floquet, recoverability, causal-use or representation-collapse result.
