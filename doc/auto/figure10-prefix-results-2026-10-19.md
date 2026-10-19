# Figure 10 native training-prefix diagnostic — verified results

The pinned native graph model completed 256 real CPU optimizer updates on the
Pdet(2,32) training prefix. Independent artifact verification accepted this
engineering result. No complete epoch, validation, test, checkpoint selection
or Figure 10 reproduction was performed.

## Executed scope and outcomes

Protocol: [training-prefix diagnostic](figure10-prefix-diagnostic-2026-10-19.md).
Model: pinned GC-LSTM, N=100, H=128, K=2, one recurrent layer and registered
link head; CPU FP32, model seed 0, negative seed 1. Chronology: detach incoming
H/C, consume G[t-1], then score/backpropagate/update against G[t], t=1..256.

- Mila **11320580**, source `7dadf7e`: failed `2:0` in one second, before the
  supervisor/model could start. Slurm's spool copy exposed the wrapper's
  incorrect relative supervisor path. The failed admission and records remain
  preserved. A shell regression verifies the corrected absolute scratch path.
- Mila **11320665**, source `c72323cd36a3aafd4d7f6cc22e4a1d81bf36bc3e`:
  **COMPLETED, exit `0:0`, 156 seconds**, node `cn-f002`. Actual allocation:
  one CPU, 2 GiB, zero GPUs, with no requeue/restart. The distinct admission
  and output paths do not reuse the failed run.
- **256 attempts, 256 paired durable outcomes, 256 confirmed optimizer steps,
  zero unresolved attempts.** All 256 recorded losses are finite/nonnegative.
  Independently recomputed arithmetic mean loss: **1.2817290551029146**.
  Update-loop elapsed time: 46.886 seconds; worker total: 148.597 seconds.
  These timings describe this prefix only and do not size a full fit.
- Largest sampled cgroup memory: **482,934,784 bytes**. A true cgroup peak was
  unavailable; scheduler batch MaxRSS was 471,468 KiB. The supervisor verified
  the admitted memory ceiling and recorded the owned process group empty.

Both jobs requested one CPU for ten minutes; observed allocated walltime was
157 CPU-seconds in total. No GPU allocation was used.

## Independent verification and reporting

The coordinator collected and rehashed 520 metadata/journal files (486,497
bytes), plus the 1,324-byte result-metadata artifact referenced by W&B. No raw
dataset or weights were transferred. The independent verifier checked source,
admission, installed runtime, conversion-manifest and node2a receipt bindings;
all intent/outcome SHA links, exact timestamps and counters; raw loss reduction;
scheduler exit and process-group cleanup.

The actual offline W&B transaction contains 784 CRC-checked physical fragments
and 781 decoded protobuf records. Its **256 history rows** match raw losses and
steps 0..255; terminal summary and exit record report completion. The offline
reporting receipt has `finish_returned=true`. **Dashboard delivery remains
unverified**; offline completion does not establish remote synchronization.

Exact identities:

- Launch admission: `262bfcffee21f2e7b2a2c4caedfedcec6408d769c8b3a7a2d8852f5f3c0d72b4`.
- Converted data manifest: `a6ff50183de570ff17cfbe92b3f1976bac9d2d85557ecc11379e24b2c737d1ce`.
- Node2a pass receipt: `550fa0aa5810819c8f154bbc5edb2ea8b4cd7345dcc8681394295279c7109a96`.
- Independent verification JSON: `a78c4c87756b302af07e45f9697897f3491f83802a23cd39a40ce01fd26c8948`.
- Independent verifier: `af624cd1efdf2a1917d66a1483d3265ac4fd0c0d5623346c557107bee77caf43`.

Local runtime evidence (ignored):
`runs/ops/figure10-prefix-20261019/prefix-20261019T204605Z-c72323cd/`.
Remote run:
`/network/scratch/m/mehrab.hamidi/representation-collapse/runs/figure10-prefix/prefix-20261019T204605Z-c72323cd/`.

## Next executable research step

Implement the thin admitted **fit-only runner**, using the existing native
adapter, attempt journal, selection sink/ledger, train/validation opener and
reporter. Attach loss/epoch reporting after durable commits; construct actual
seed/provenance receipts; finalize the existing ledger, mapping fit reason
`training_loss_patience` to ledger reason `loss_patience`; preserve selected
checkpoint references and all validation rows. No further synthetic training
gate is needed before this integration.

Native selected-checkpoint loading and a full H/C observer at s[2816] remain
absent. Implement those separately for node2b on the already-budgeted final
replay, before the first held-out test iterator advance. Only then can the
256-target Figure 10 behavioral curve be produced. Representation geometry,
tangent dynamics and causal-memory measurements remain separate research
outcomes; the present loss diagnostic establishes none of them.
