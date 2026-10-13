# Cached forward v2: independent evidence closeout

2026-10-14. **The completed numerical audit has accepted integrity; strict
all-comparison acceptance is false.** Four cross-dtype checks fail their frozen
bounds. This review supplies no competence, causal-transfer, information-loss,
dimension-collapse or Floquet finding. No inherited manuscript or active TeX
include supplies a claim here.

The prior independent reviewer completed producer metadata checks, reducer
implementation review and extraction of the admitted numerical audit, then
lost its response stream before writing the final report. This fresh closeout
reconciles those durable artifacts. It rehashed the saved audit without parsing
its 271 MB JSON, checked compact JSON consistency and source identities, and
read the summary script and relevant reducer routines. It did **not** rerun
the reducer, tests or experiments, open NPZ/checkpoint/target data, or decode
W&B transactions again. Numerical results below are attributed to the existing
independent audit and review summary, not a new raw-data replication.

## Execution and evidence identity

The producer ran source `edc47ee8914125b20d9fee9816e3255f194bc462` under
root admission `a8f0e3e57d30792e8a726560e2c67ff217b9d2e0644e0aabd0806d6f43e69172`:
32 conditions, 736 attempted/completed native core calls, 64 encoder calls and
224 head calls, exit 0, 49.663 seconds externally. The separately admitted
NumPy reduction used source `70b3a05a49fcd8e08fe2af2d5409095d6b2d1f60`,
reducer SHA `3bf42cd1994c3bbc4763a1f99128659e838a2c55ba1f021eee72a1a2e0bc7b5c`,
and admission `027ea3ceba3cc4e3a4fc9760274d8b69fcb59085957a2e1ae2278b440b3cc7c7`.
Its external elapsed time was 41.057 seconds, exit 0, with no deadline censor
or retry. Both stages used the frozen one-CPU envelope. No training,
derivatives, task targets or new cluster allocation occurred.

The [frozen contract](post-calibration-measurement-contract-v2-2026-10-14.md)
and config remain unchanged. Their static `execution_authorized=false` field
is preserved; separate root admission records authorized these exact stages.
The prior metadata review verified all 72 raw-file hashes. This closeout
reverified its hash-bound evidence and current source identities, including
the reducer in its isolated source directory; it did not repeat the 72 raw-file
scan or inspect Git publication.

## Numerical outcome and independent units

The admitted audit contains **15,024 comparisons of 300,399,104 reused scalar
entries**. These entries are comparison accounting, not independent samples.
All 112 C-versus-B0 checks, 160 copy/zero controls, 6,464 duplicated-half
checks, 1,824 operand/origin byte-identity checks and 3,232 exact-repeat checks
pass. Of 3,232 cross-dtype checks, four fail: hidden-plus and hidden-minus
logits for the fixed-challenge GRU/categorical checkpoint, in both repeats,
with FP64 as reference and the frozen FP32 directed bound.

Each failed check has violated coordinates `[18,2]` and `[50,2]`. The plus
gap is `1.3658319945692929e-6`, exceeding its bound
`1.0153170288453145e-6`; the minus gap is `1.1428534464523812e-6`, exceeding
`1.0158837146168186e-6`. These are the **violating-coordinate gaps**. The
respective whole-edge maximum gaps are `4.983791732193765e-6` and
`2.995495561819439e-6`; maximum absolute gap need not identify the failed
coordinate because the bound depends on the reference magnitude.

There are eight scalar violations, representing one independent history
(ID 18) reused across duplicated rows, two signs and numerical repeats.
The design has 32 independent generated histories per task, duplicated to
64 rows; seed 30019 fixes inputs, seed 30029 fixes directions. Dtypes,
repeats, signs and checkpoint contexts add no independent trained seeds.
This deterministic calibration makes no population confidence-interval claim.

All 6,144 saved C intervention-effect rows are finite, nonzero
`resolved_C_effect` records. They describe finite signed perturbations of the
complete C state at the declared origin and do not measure retained task
information. The baseline categorical B0/C predictions agree on all 64 rows
in all 16 categorical conditions, with no ties and all frozen margin
certificates passing. Continuous conditions have zero saved baseline output
discrepancy and zero corresponding label-free loss-change bounds; no target
loss or competence score was calculated. Baseline agreement does not transport
C interventions to B0; B0's intermediate LSTM cell remains unobserved.

## Reporting and disposition

The producer offline run `94871a2d43e6406f8022e763a905f14e` has 32 verified
history records; the reduction offline run `6055233d62b4469796c50122e4cfa8ce`
has one. Root's transaction verifiers checked completed summaries and compact
metadata artifacts. This closeout rehashed their transactions, receipts and
artifacts. Both remain unsynced; authenticated remote delivery is unverified.

Numerical audit integrity is accepted with all four failures retained, no
tolerance relaxation and no FP64 ground-truth or kernel-cause assertion.
Original eight-coordinate competence eligibility remains **zero**; the prior
32 complete fits plus one incomplete fit and 114,200 updates are unchanged.
No scientific or causal bridge is accepted. A future measurement map and any
derivative/training stage still need their own prospective review and admission.

Reproducible evidence (SHA-256):

- Existing producer review: `4edc1342b287c6847363ccf77cee861b0c22e659937055c25b2e0bc7856628ec`.
- Existing reducer review: `039f50060a16fcca2f4ab992bcc40a995cb77edf283ce2c7892d214272edef37`.
- Existing numerical summary: `996839267b720fb6a9b441af7034bbb303e3fccf9ae5151ebe4b2668eb4e72d9`.
- Actual audit: `d36e9ca551f9a2b31602abe2afe7da84cef4b2b911b919e5501dd01faffab035`.
- Fresh [metadata reconciliation](../../runs/review/cached-forward-v2-closeout/metadata-reconciliation.json):
  `9a0708148a91b2b252aa6dd7db415fb3561fd6a67d2122f4f9c0b66532c8bfd5`.

The first three artifacts and their extraction source/console are under
`runs/review/cached-forward-v2-results-independent/`; the actual audit is
`runs/cached-forward-v2-reduction-20261014T0140Y/audit/audit.json`.
Ignored runtime evidence is retained locally; this report records its durable
identities. Root owns final integration into the tracker and dated notebook.
