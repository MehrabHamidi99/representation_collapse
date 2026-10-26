# Legacy Floquet array audit v1

Frozen 2026-10-14 before numerical payload inspection. This stage audits the
recovered numerical outputs; it does not replay a trained graph model.

## Inputs and acquisition

The metadata-only archive search identified
`/network/archive/m/mehrab.hamidi/floquet_analysis`. Acquire exactly forty regular
`.npy` result files: `eigvals.npy`, `eigvecs.npy`, and `envelope_i.npy` plus
`norm_env_i.npy` for i=0..18. Exclude the twenty `.pt` Jacobians, all datasets and
checkpoints. The metadata-declared byte total is 468,187,088. Keep the originals
read-only. A bounded result-file transfer through the authenticated Mila
transport creates a new ignored local input replica; this is acquisition,
not a new cluster allocation or graph-training dataset download.

Before interpreting values, record each file's streamed SHA-256, exact bytes,
regular-file/symlink status, source stat identity before/after and local byte
identity. Freeze an immutable manifest with relative basenames, hashes,
expected dtype and shape. Incomplete acquisition remains a distinct attempt;
never reuse or overwrite a partial output. Sender and receiver enforce the
whitelist and byte ceiling. Acquisition computes no numerical statistic.

Expected schemas from headers/source metadata: complex64 eigenvalues `(6,)`,
complex64 eigenvectors `(25600,6)`, complex64 envelopes `(20,25600,6)`, float32
saved norms `(20,6)`. Only four headers were previously inspected; verifying
all forty headers is part of the new audit. A dtype/shape discrepancy fails
closed rather than triggering an alternative parser or schema.

## Prespecified audit

1. Verify manifest/source/config/lock identities, safe numeric-only headers,
   exact payload length and streamed hashes. Refuse object/structured arrays,
   nested/escaping filenames, links, extra bytes and any `.pt` deserialization.
2. Retain the six stored complex eigenvalues, their magnitudes and principal
   phases in radians per unspecified stored product. Do not normalize phases
   to a guessed fundamental period or call them certified Floquet multipliers.
3. Describe eigenvector norms and the six-column Gram matrix. Their dimension
   and orthogonality are neither complete-state covariance rank nor solver
   residuals. No whole-system spectral assertion is permitted.
4. For each envelope, compare all twenty time rows with row zero, retaining
   exact byte-repeat indicators and absolute differences. The primary
   source-linked diagnostic asks whether the time axis repeats one fixed
   prefix action, as the inspected June source would produce. April arrays
   need not have that same producer; agreement does not prove lineage.
5. Recompute each mode's norm over the 25,600 state coordinates, using bounded
   chunks and float64 accumulation. Directed comparison uses derived norms as
   candidate and saved float32 norms as reference:
   `abs(candidate-reference) <= 1e-6 + 1e-5*abs(reference)`.
   Keep every failed comparison; do not relax tolerances or rerun for a sign.
6. Describe differences between consecutive envelope prefix files using the
   same declared coordinate layout. All nineteen prefixes and twenty rows
   derive from one unlinked archived run; they are not independent seeds.

Reject nonfinite inputs with an explicit failure receipt. The implementation
must never load all envelopes at once. Memory-map numeric files and iterate
in bounded chunks. The twenty saved Jacobians are outside this stage, so
**eigenpair residuals, Jacobian chronology, orbit closure, task performance,
causal memory and representation-collapse claims remain unverified**.

## Execution and reporting

One local CPU attempt, one numerical thread, maximum 600 seconds total, no
GPU, optimizer, model forward, gradients or provider calls. Shared clock starts
before launcher spawn/reporting initialization. Numerical work stops at 590
seconds; an owning outer supervisor terminates descendants by 600 seconds and
marks late/incomplete finalization. Tests use only tiny synthetic arrays.

Use immutable published source in a clean isolated worktree, unique inputs and
output paths, an exact external admission record and independent source review.
Initialize pinned SDK 0.30.0 offline W&B in the already created
`mehrab-hamidi-milaquebec5431/representation-collapse` destination before audit
array/hash work. Record selected identities and live progress; no weights or
raw arrays are W&B artifacts. An offline receipt is not cloud delivery.

An independent reducer/reviewer must reconcile the resulting compact audit
against saved inputs before accepting numerical findings. Numerical integrity
and the primary repeat/norm diagnostics are separate from scientific acceptance.
This protocol neither starts a persistent model worker nor authorizes a new
Slurm/GPU allocation. The prior shared allocation ended after its hosting guard.

## Next dependency

If integrity passes, use the audit to determine which archived plots can be
regenerated and which analysis definitions are inconsistent. Residual/closure
verification and full graph checkpoint reconstruction require separate pinned
inputs, measurement code and compute admission. Missing checkpoints permit an
explicit reconstruction, never a claim of byte-identical recovery.
