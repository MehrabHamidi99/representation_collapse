# Deterministic Figure 10 controls: verified results

2026-10-19. These are analytic controls on the known Pdet(2,32) generator,
computed after the control protocol and code were committed as `0139fbd`.
Only pinned converted-manifest metadata was read; no train/validation/test
split payload, learned model or checkpoint was opened.

The two patterns contain 100 and 120 ordered positive edges with zero overlap,
among 9,900 possible ordered off-diagonal pairs. Persistence therefore has
F1 **0.96875** averaged over 256 targets: **0** at all eight changes and **1**
at the other 248 targets. The known-generator phase oracle scores **1** and
the always-negative control **0**. The oracle has privileged generator/phase
knowledge and is not a learned memory result.

Independent verification recomputed all 768 control rows and reductions,
including confusion counts, phase alignment and eight block means. It decoded
771 actual offline W&B history rows, verified their exact metrics and steps,
2,344 physical-fragment CRCs, 2,324 protobuf records and terminal exit.
The compact metadata artifact and completed reporting receipt also match.
Remote W&B delivery remains unverified.

Numerical result SHA:
`ec1f8defe423188c106b6ee90181fab3db5a85239690117b4016d2c1682c60ff`.
Independent verification SHA:
`052a60e9825c54b9dd4794b1788bc18a8a29a78d03a8cb1a2f186b360f095172`.
Full public provenance is retained in the accompanying JSON. Local ignored
artifacts are under `runs/ops/figure10-controls-20261019/` and
`runs/review/figure10-controls-20261019/`.

The result shows why average F1 alone is inadequate to assess counting:
persistence can average nearly 0.97 while failing at every change. It supplies
a behavioral reference, not evidence about representation dimension, causal
memory loss or a trained model. The actual GC-LSTM fit remains a separate run.
