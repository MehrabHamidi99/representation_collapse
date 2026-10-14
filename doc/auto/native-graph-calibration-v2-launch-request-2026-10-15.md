# Prepared v2 calibration allocation — October 15

**Prospective; execution is not authorized.** Initial CPU submissions are
exhausted. The unchanged-cell import correction is prepared as a distinct
v2 assay, preserving failed v1 job11256293 and all its evidence. A new owner
allocation approval is required before a new cluv submission.

## Exact requested scope

One nativecluv0.1.5 submission to **Mila**, account `mila`, QOS `normal`,
partition `main-cpu`, one node/task/CPU, `--mem=2G`, `--time=00:10:00`,
zero GPUs and `--no-requeue`. Requested maximum: **1/6 CPU-hour**.
No race, retries, replacement allocation, training, dataset, checkpoint,
paid model-service call or scientific acceptance. One attempt even on failure.

The actual pinned GC-LSTM and public wrapper/head are loaded through the
reviewed isolated source importer. This avoids an unrelated Temporal attention
initializer while leaving cell and wrapper bytes unchanged. Full Temporal
package API compatibility is explicitly outside the claim. Numerical fixtures,
counts, tolerances, dependency versions and physical supervision are unchanged
from v1: full H+C D12, N3/H2/seed0,50cell/5head/6sequence forwards,
4dense/9JVP/5VJP API entries, all three fixed finite-difference epsilons,
node permutation, complex linear extension and noncommuting controls.

Offline W&B0.30 initializes before Torch/cell loading. One-CPU affinity and
kernel physical memory <=2GiB are mandatory. The same import-inclusive
600-second supervisor/590-second worker deadline and owned cleanup apply.
A preexisting output or STOP refuses execution. Preserve every failure;
no fixture, tolerance, version or seed replacement.

## Source and commands

All source must first be published, independently reviewed and hash-verified.
The exact launch descriptor is
`configs/experiments/native-graph-calibration-v2-launch.json`; its code and
configuration hashes are fixed before approval. Root records the final source
commit and immutable plan hash in the runtime continuation journal.

New scratch source:
`/network/scratch/m/mehrab.hamidi/representation-collapse/sources/native-gclstm-calibration-v2-20261015-attempt001`.
New run root:
`/network/scratch/m/mehrab.hamidi/representation-collapse/runs/native-gclstm-calibration-v2-20261015`.
Output is that root's `attempt001/`, owner STOP is `STOP`, actual job admission
is `admission.json`, and immutable plan/readiness records are under `inputs/`.
Reuse the already installed and hash-verified environment
`environments/graph-native-cpu-20261015-calibration01/venv/`; no new package
installation or download in the allocation. Whole-project scratch must remain
under20GiB. Controller cache, cluv logs and W&B support remain on project scratch.

Native cluv submits the reviewed
`code/ops/native_graph_calibration_v2_job.sh --plan <absolute-inputs/plan.json>
--plan-sha256 <frozen-SHA256>`. The exact native `sbatch` expansion must equal:

```text
sbatch --parsable --account=mila --qos=normal --partition=main-cpu --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:10:00 --no-requeue --job-name=cluv-rc-native-gclstm-v2 --output=<new-log-root>/mila_%j/slurm-%j.out --chdir=<new-source> --export=ALL code/ops/native_graph_calibration_v2_job.sh --plan <absolute-plan> --plan-sha256 <frozen-SHA256>
```

The materializer fills the actual Slurm job ID into the immutable v2 admission
template and execs the prepared environment's `bin/python -m
representation_collapse.native_graph_calibration_v2 --root <new-source>
--config <new-source>/configs/graph/native-graph-calibration-v2.json
--source-root <verified-public-wrapper-inputs> --admission <new-run>/admission.json
--output <new-run>/attempt001`, with `PYTHONPATH=<new-source>/src`.

Source-only nativecluv synchronization uses a clean published main controller
checkout, followed by exact remote commit pinning and checks. Automatic root
environment sync remains excluded. Per-process cluv configuration and exact
resource/script/source/argv guards prevent inherited defaults from changing
the request. Exclusive submission claim, native job-ID reconciliation and
no-retry rule apply. No other project's job or hosting allocation is used.

## Acceptance and limits

Accept only engineering results after independent saved-array reconciliation,
source/admission/environment hashes, exact counts, real scheduler/resource
records, owned cleanup and durable offline transaction verification. Native
v1 produced no arrays and its result reducer was not run. The v2 reducer must
bind new source/config/import identities before inspecting v2 outputs.
Full graph reconstruction, trained competence, periodic orbit closure,
Floquet certification and harmful-collapse claims remain separate unfinished
steps. This ten-minute allocation grants no follow-on training budget.
