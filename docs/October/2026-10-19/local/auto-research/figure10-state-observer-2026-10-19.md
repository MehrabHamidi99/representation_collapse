# Frozen pre-target complete-state observer

2026-10-19, before the first selected evaluation. This adds passive state
retention to the previously specified single held-out pass. It changes no
checkpoint selection, scores, candidate pairs, phases, thresholds or metrics.
It requires no second test read, replay, model call, optimizer or RNG draw.

`run_selected_evaluation(..., observe_state=None)` optionally invokes
`observe_state(state, t)` once at each t=2816..3071, after the strict prefix
replay gate and immediately before scoring target t. The provided complete
state is s[t], after consuming G[t-1] and before consuming G[t]. Hidden and
cell are detached independent clones of the native tensors. An observer may
read or mutate these clones without changing live recurrent storage. Native
evaluation/no-grad mode remains in force. Callback failure terminates the
already charged evaluation, with no score/consume of that target and no
reopen or retry.

The standalone CLI supplies a `TrajectoryWriter`. It streams each hidden then
cell payload through the existing strict finite CPU FP32 little-endian byte
reader. Required shape is `(256,2,1,100,128)` in C order: time, component
(hidden/cell), layer, ascending canonical node, feature. Every observation
must match its exact expected timestamp, complete H/C shape, dtype and node
order. Signed zeros and all FP32 bits are preserved. The stream is exactly
26,214,400 bytes (25 MiB). Its directory is cluster scratch under the unique
evaluation run. No trajectory or graph payload is sent to W&B or git.

`test-state-trajectory.f32le.partial` is exclusively created. Count/order/write
failures poison the writer, preserve the partial payload and fail the run.
Only after exactly 256 observations does it flush/fsync and publish
`test-state-trajectory.f32le` using a create-only hard link, then remove the
partial name and fsync the directory. Publication failures remain terminal;
partial data is retained where possible and no result descriptor is accepted.
The completed descriptor records SHA, byte count, shape/order/dtype, node IDs,
timestamp interval and pre-target timing. It is attached to
`evaluation-result.json` and the outer terminal manifest. The evaluator's
earlier diagnostic metric-completion event does not independently certify
trajectory publication; the outer success manifest and descriptor do.

Before publication an independent bounded streaming readback verifies that
the partial is the originally opened regular single-link inode, its exact
size and full SHA equal the writer's observation stream, and its identity and
metadata remain stable during reading. The final hard link must still point
to that original inode. Replaced or corrupted partial files fail and retain
no accepted descriptor. This is cooperative integrity checking, not protection
against a hostile owner able to rewrite the process or its files.

This is one seed's trajectory cloud, retained for a future separately frozen
analysis. Centering 256 states gives a time-cloud rank ceiling of 255; repeated
phase trajectories impose further trivial ceilings. Hidden-only, cell-only
and joint covariance dimensions remain different measurements. These samples
are not independent seeds, intervention responses or tangent directions.
Retention alone establishes no effective-dimension, harmful-collapse, causal
memory, Lyapunov or Floquet scientific claim. Later analysis must state
centering, normalization, phase alignment, numerical tolerance and comparison
controls before interpreting the cloud.
