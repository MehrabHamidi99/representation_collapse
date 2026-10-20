# Independent one-bit discovery prelaunch review

2026-10-20. **Scientific design and worker accepted for the bounded seed-0
discovery execution.** This is prelaunch review, not acceptance of numerical
results. The launcher must run inside its pinned source directory so the
worker's `git rev-parse HEAD` records that source. The coordinator owns final
immutable staging and scheduler admission.

Reviewed identities:

- Worker `src/representation_collapse/onebit_memory.py`:
  `92d855488b675f82d318e646f9811a0d65d36aa0786f555c5f8cc0c592d50535`.
- Protocol `doc/auto/onebit-memory-discovery-v1.md`:
  `1b97d6b019f835d9d27d533597adde75573deace21df29398be90df429a03baf`.
- Tests `tests/test_onebit_memory.py`:
  `fc6261ac3700655b9dfe2a2a162e677674a3f772538e29e1f347a83633682ea1`.
- Reused body construction `src/representation_collapse/vector_task.py`:
  `df3843330ac9c59205823c8d55868e7878e903c0065c5af8bc89d1308243a297`.

## Checked design and implementation

Only WRITE carries the bit; body and QUERY observations are independent of its
value. The same forcing is crossed with both cue values during evaluation.
Training uses 32 independent schedule draws with shuffled balanced cues,
duplicated across arms. Both arms have matched symbol counts, bigrams, runs,
endpoints and duration. The control rejects exact whole-body repetition;
higher-order order, predictability and motif diversity remain jointly changed.
The effect is consequently that specified ordering intervention.

Training, validation and evaluation use distinct RNG streams. Validation is
materialized before optimization. Held-out panels are generated only after
all three competence boundaries pass. Each gate requires at least 487/512
correct independently in every applicable cue/arm/horizon cell. Fixed boundary
weights, not favorable historical weights, determine continuation. Raw validation
logits permit independent eligibility reconstruction. Failed gates stop the fit.
Independent streams do not guarantee distinct body strings; R2 has only 20
allowed motif-ID pairs and sampling is with replacement.

Every model path uses the same native LSTMCell update, carries H and C, and
consumes QUERY before scoring. Complete pre-QUERY state excludes that query
update. Saved state layout is [schedule, cue, arm, 64], H followed by C.
Loss is QUERY-only cross-entropy; training has no geometry objective. The
checkpoint and in-memory parameter bytes are hashed before and after evaluation.

Interventions exchange both H and C. Opposite-cue swaps preserve schedule and
arm and are scored against donor as well as recipient labels. Exact donor
logits and self-copy logits are asserted. Same-cue copying uses the preceding
draw within each fixed chunk; a different draw can have an identical body.
Zero-state corruption is an additional control, not a matched-magnitude sham.
Since every QUERY input is identical, full-state replacement reproduces the
donor readout by construction. It demonstrates dependence on retained state,
not a localized memory mechanism or recovery of erased information.

Covariance uses raw complete-state coordinates, separately by arm, pooling
cue values and schedule draws; FP64 centering, N−1 normalization, explicit
negative-eigenvalue tolerance, and undefined zero-trace PR are recorded.
That covariance includes forcing variability and does not identify the useful
bit's information dimension. Its paired observations are dependent.

## Corrections made before launch

The review requested and checked persistence of per-example validation outputs,
finite inference checks, exact donor replay, explicit baseline semantics,
pre/post-evaluation checkpoint checks, and agreement of protocol and code on
training sampling, covariance output and donor direction. The constant baseline
is calculated from labels; the register reference extracts the rendered WRITE
value. This input-target sanity check is not a separately implemented streaming
register. The launcher was also found to lack the required protocol argument;
the argument was added before this review. The pinned-source working-directory
requirement above remains a final coordinator check.

## Verification and remaining acceptance work

`PYTHONPATH=src .venv/bin/python -m unittest tests.test_onebit_memory
tests.test_vector_task` passed all 10 focused tests on the final worker. An
independent tiny random-model evaluator fixture additionally checked the saved
complete-state shape, exact self-copy aggregate behavior, same-cue copying, and
equality of donor-swap accuracy with native donor accuracy. Some local sandbox
attempts failed before native execution because OpenMP could not open shared
memory; equivalent authorized runs outside that restriction passed. These are
unit checks, not empirical experiments or repeat training.

No cluster job, training outcome or actual W&B history was inspected by this
review. After execution, independently reconcile all update rows, stage logits,
schedule construction, checkpoint/source/environment hashes, raw held-out
reductions and interventions, scheduler outcome, and completed offline reporting.
The worker writes results before its reporter exits, so a result JSON alone does
not establish successful reporting or scientific acceptance. Source/runtime
identity and artifact hashes are emitted; the outer launcher must bind the
entire committed source, resources, deadline and unique output location.

## Limits on conclusions

This is one trained seed and one-bit retention, not a capacity measurement.
Paired schedule draws, cue flips, arms and time steps are not model replications.
No p-value or across-seed confidence interval is supported. The descriptive
delta16, delta2 and their difference-in-differences must all be reported; a
positive DID without excess periodic error at R16 is insufficient.

All-X controls, shuffled-label readouts, learned probes, precision-sensitivity
assays and partial-state interventions are explicitly deferred. Thus native
accuracy loss cannot yet distinguish inaccessible information from erasure.
No tangent Jacobian or Floquet spectrum is measured. Preserved memory, negative
ordering effects and acquisition failure are valid outcomes and must remain
visible. Confirmation seeds and higher memory loads need subsequent protocols.

## Prelaunch host amendment review

The launcher now changes into its pinned source directory before execution.
Mila authentication failed before submission; the protocol now specifies Tamia,
account aip-ioannism, partition cpubase_bycore_b1, one CPU/4 GiB/two hours/zero
GPUs. The worker scientific bytes above are unchanged. The dedicated CPU
environment and exact launch identity remain coordinator verification duties.
The updated protocol hash above includes this execution-only amendment.
