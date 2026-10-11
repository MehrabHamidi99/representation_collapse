# Verified cluster access

Checked 2026-10-11 at 12:31:07 UTC using pinned `cluster-uv==0.1.5`.
All nine configured Alliance targets passed identity checks: Tamia, Narval,
Rorqual, Fir, Nibi, Trillium, Trillium-GPU, Killarney and Vulcan. Eight remote
targets had reusable SSH masters; Tamia was the local controller. Mila passed
an independent check from the Mac at 12:24:00 UTC. These checks establish
authentication and visibility, not successful allocation or training.

## Routes and session lifetime

The working route is Mac → Mila → `tamia2.tamia.ecpia.ca` → other Alliance
targets. Mila uses the existing Mac master. Cross-cluster masters belong to
Tamia2 and this scratch checkout; they cannot be assumed reusable on Tamia1.

The project adapter invokes native cluv and overrides SSH options only in its
own process. Its sockets use an owner-only node-local directory
`/tmp/rc-cluv-3115794-9014ca33ffb5`, derived from the UID and resolved project
path. Shared SSH configuration, keys and other projects' masters were untouched.
Strict known-host checking remains enabled. Ordinary cluv commands use a
different socket namespace; use the adapter consistently.

An initial round authenticated all eight targets, but sockets were absent after
the controller command ended. The cause remains unproven; effective systemd
`KillUserProcesses=false` rules out that specific proposed explanation. A
controller that keeps the same Python event loop and terminal open allowed
separate passive cluv processes to reuse all eight connections successfully.
The current controller began holding at approximately 12:26 UTC for at most
eight hours. Authentication can expire or be interrupted earlier; recheck
masters before staging or submitting. Persistence after closing this controller
has not been established.

From the same Tamia2 scratch checkout, after source synchronization:

```bash
cd /scratch/m/mehrab/representation-collapse/source
uv run --locked --group ops python code/ops/cluv_controller.py --verify-access
```

To establish a new controller explicitly, keep this command open and approve
its sequential Duo pushes:

```bash
uv run --locked --group ops python code/ops/cluv_controller.py --hold-login \
  narval rorqual fir nibi trillium trillium-gpu killarney vulcan
```

For later authorized native cluv operations in another terminal on the same
node and checkout, use `cluv_controller.py --batch-mode` followed by the usual
cluv arguments. Passive verification never requests Duo. Verification is
sequential because cluv's synchronous SSH configuration lookup consumed the
original concurrent checks' timeout budget. Identity, storage, scheduler
visibility and scheduler exit codes are reported separately.

## Storage and scheduler observations

- Tamia: `/scratch/m/mehrab`, account `aip-ioannism`, QOS `normal`.
- Narval and Rorqual: `/scratch/mehrab`; `def-ioannism_cpu` and
  `def-ioannism_gpu`, QOS `normal`. Both expose UP `gpubase_bygpu_b1` with
  a three-hour partition limit.
- Fir: `/scratch/mehrab`; `def-ioannism_cpu` and `def-ioannism_gpu`, QOS
  `interac,normal`. UP `gpubase_bygpu_b1` permits up to three hours.
- Nibi: `/scratch/mehrab`; `def-ioannism_cpu` and `def-ioannism_gpu`, QOS
  `interac,normal`. UP `gpubase_bygpu_b1` permits up to three hours.
- Trillium: `/scratch/mehrab`, `def-ioannism`, QOS `normal`; CPU scheduler
  reports `compute`, with a one-day limit.
- Trillium-GPU: `/scratch/mehrab`, `def-ioannism`, QOS `normal`; scheduler
  cluster name is `grillium`. UP `compute`, `compute_h200` and `compute_b200`
  report one-day limits. Cluv recognizes the shared filesystem with Trillium;
  do not concurrently modify their checkout.
- Killarney: `/scratch/mehrab`, account `aip-ioannism`, QOS `interac,normal`.
  Login-shell initialization supplies Slurm under
  `/cm/shared/apps/slurm/current/bin`. UP `gpubase_h100_b1` and
  `gpubase_l40s_b1` permit three hours; only GPU partitions were visible.
  Do not send the CPU-only default to this cluster.
- Vulcan: `/scratch/mehrab`, account `aip-ioannism`, QOS `interac,normal`;
  UP `gpubase_bygpu_b1` permits three hours and exposes L40S GPUs.
  UP `cpubase_bycore_b1` also permits three hours.
- Mila: `/network/scratch/m/mehrab.hamidi`, account `mila`, QOS `normal`.

Partition names and account visibility do not establish one-GPU eligibility,
current capacity or environment readiness. No allocation was requested during
this access audit. Narval and Rorqual are the first other-cluster pilot candidates
after locked cache staging and exact one-GPU request validation, within the
existing four-job/four-GPU-hour envelope. Race mode remains excluded.

Killarney's initial non-login shell lacked Slurm PATH and `$SCRATCH`; Vulcan's
lacked `$SCRATCH`. Follow-up native cluv probes under `bash -lc` verified both
schedulers and writable scratch directories without loading modules or changing
profiles. The verifier now uses that login shell for remote probes. Native cluv
sync/submission commands also need their scratch/Slurm environment supplied
explicitly before execution; the socket adapter alone does not initialize it.

## Evidence

Complete login-shell identity, storage, partition and association outputs are in
ignored local `runs/ops/cluv-alliance-login-environment-20261011.json`.
The initial non-login snapshot is `runs/ops/cluv-alliance-access-20261011.json`;
Mila and safe Qwen liveness evidence
are in `runs/ops/cluv-mila-access-20261011.json`. Earlier socket/debug evidence
is in `runs/ops/cluv-access-probe-debug.json`. No credentials or Duo responses
are stored in these artifacts. The audit covers the project's configured aliases;
it does not assert access to every institution or service in the Alliance.
The resolved Killarney/Vulcan environment proof is in
`runs/ops/cluv-killarney-vulcan-environment-20261011.json`.
