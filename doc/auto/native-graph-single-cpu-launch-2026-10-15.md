# One native GC-LSTM CPU calibration — October 15

Prepared under the one unused initial analytic CPU-calibration submission and
the owner's explicit instruction to continue and resubmit. This is an
engineering calibration of untrained pinned source, not graph training or a
reproduction claim. Source publication, independent review, verified installed
metadata, offline reporting readiness and the exact coordinator admission
must precede submission. No job has been submitted by this document alone.

## Allocation and stopping rule

Use native `cluster-uv==0.1.5`, explicitly targeting **Mila**. Account `mila`,
QOS `normal`, partition `main-cpu`; one node, one task, one CPU, `--mem=2G`,
`--time=00:10:00`, no GPU request, `--no-requeue`. Exactly one submission;
no race, retry, replacement allocation, training, dataset or checkpoint.
This consumes the fourth and final initial CPU slot even if the job fails.
Three prior distinct initial submissions were 11190579, 536757 and 5234514.
Tamia 612745 used its own separate envelope. No GPU allowance is renewed.

The calibration's import-inclusive supervisor has a 600-second deadline,
590-second worker limit and owned-descendant cleanup. It verifies the real
Slurm one-CPU identity and a kernel physical-memory cgroup cap <=2GiB, then
sets one-CPU affinity. STOP is owner-controlled in the unique run directory.
A missing guarantee or reporting failure stops before native numerical work.
Reconcile actual scheduler ID and outcomes after any ambiguous submission;
never resubmit to repair uncertainty.

## Isolated paths and exact invocation

All cluster paths are under
`/network/scratch/m/mehrab.hamidi/representation-collapse/`:

- published source: `sources/native-gclstm-calibration-20261015-attempt001/`
- environment: `environments/graph-native-cpu-20261015-calibration01/venv/`
- toolchain: `toolchains/python/cpython-3.10.19-linux-x86_64-gnu/`
- public wrapper fixtures: `runs/native-gclstm-calibration-20261015/inputs/upstream/`
- root plan and receipts: `runs/native-gclstm-calibration-20261015/inputs/`
- exact admission: `runs/native-gclstm-calibration-20261015/admission.json`
- output: `runs/native-gclstm-calibration-20261015/attempt001/`
- STOP: `runs/native-gclstm-calibration-20261015/STOP`
- cluv logs: `logs/native-gclstm-calibration-20261015/`

The exact supervisor argv is the prepared environment's `bin/python`, `-m`,
`representation_collapse.native_graph_calibration`, followed by absolute
`--root`, `--config`, `--source-root`, `--admission` and `--output` arguments.
`PYTHONPATH=<published-source>/src` imports source without installing the main
Python>=3.11 project. The separately pinned CPython3.10.19 environment retains
the exact Temporal cell and CPU-build versions in the frozen calibration.

Cluv submits `code/ops/native_graph_calibration_job.sh --plan <absolute-plan>
--plan-sha256 <immutable-SHA256>`. Its stdlib Python launcher verifies the
source and plan, fills only the actual Slurm job ID plus plan identity in the
frozen admission template, exclusively creates the admission, and execs the
exact supervisor. No installs or package downloads occur in the allocation.
The coordinator binds reviewed source/config/helper/reporter/wrapper hashes,
Git commit, actual environment and lock/artifact identities, review/reporting
receipts, argv, finite counts and resource limits in the plan before cluv
submission. The actual admission is retained and hashed after job start.

Cluv's source-only clone is used from a clean local `main` controller clone.
Native0.1.5's detached-HEAD sync mishandles an absent `GITHUB_HEAD_REF`; that
path is excluded. After native sync, the new owned remote checkout is pinned
to the exact published commit and its clean tree and hashes are verified.
Automatic root-project
environment sync is excluded after independently verifying staged source and
the separate CPU environment. Effective cluv configuration is changed only
in the submission process, scoped to these new paths. The submission login
shell rejects inherited `SBATCH_*` or `SLURM_CLUSTERS` inputs. Other project
jobs, shared hosting infrastructure, old source trees and root locks remain
outside this launch's ownership.

## Readiness and acceptance

CPython3.10.19 staging completed through native cluv in 8.85 seconds, with
`--no-bin` and no HOME shims. Binary SHA-256:
`b081289003128d701e48880da11800d7e4410a5777fec6afaa9e31d07419c52f`.
Existing scratch uv0.9.18 SHA-256:
`7f73d66b9767a281c2cae604b7cd0b6eaa1f0dda5ae28b9067e6de745d9593eb`.
No model or SDK import occurred during that step.

The engineering protocol fixes 50 cell forward hooks, five head forward
hooks, six head sequential forward hooks, four dense Jacobian, nine JVP and
five VJP API entries. All declared comparisons must pass fixed tolerances,
including every finite-difference epsilon and complete-state permutation.
Preserve failures without replacing fixtures. Independent review of saved
tiny arrays must verify chronological products, comparison residuals, source
identities, physical resource/scheduler outcomes, owned cleanup and durable
offline W&B records. Offline durability does not assert cloud synchronization.

Even a successful assay proves only this tiny native engineering calibration.
The original graph checkpoint is still unidentified; reconstruction training,
task competence, periodic-orbit certification and harmful collapse remain
separate unfinished scientific steps. No sign of collapse is presupposed.
