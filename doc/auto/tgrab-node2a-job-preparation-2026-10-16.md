# Figure 10 node2a runtime identity capture

This is a one-shot Mila CPU preparation record for capturing the pinned native
runtime identity used by the synthetic node2a fixture. The capture constructs
no model, opens no graph data, performs no training or fixture update, and has
no train/validation/test interfaces. It imports the pinned native packages,
including Torch, to measure package/module/RECORD identities. The bootstrap
bypasses `torch_geometric_temporal` package initializers and uses the standalone
pinned graph importer. This artifact is environment evidence only, not a
historical Figure 10 replay or scientific result.

## Readiness

The bootstrap and the existing node2a operations code passed final independent
review; the bootstrap review receipt SHA-256 is
`9244d6da8bf1ac08e140879fb426c01ac105fb1fdb8d861eb38fbca175f996be`. The
environment preparation receipt is staged locally at
`/private/tmp/figure10-node2a-environment-receipt.json` and its verified
SHA-256 is
`4e753d13680738a7b48374950ae41e33f0b7f4935c889f882b149678301eb6c2`.
The pinned bootstrap SHA-256 is
`37a0c0839e167d3a71cc4e1186f97a9d91c211cf006ad2df00dcc64652e62253`; the
standalone importer SHA-256 is
`2f14b3683f67d33f4a5f6325b5b204aee3a086e1fad1966acb9810a302d04344`.

The cluv route is connected to Mila. The remote source replica is clean at an
older commit; it must be synchronized to the reviewed committed source before
capture. Its current scratch footprint is about 221 MB; the pinned CPU
environment is about 2.4 GB. No identity capture or Figure 10 fixture job has
run or been submitted. The owner waived separate compute approval and limits
for the Figure 10 subgoal. The capture wrapper passed independent review after
its bootstrap time-of-check/time-of-use race was closed by hashing, transferring
and compiling the same in-memory bytes and checking source closure again at
exit. Its nine tests pass. After the reviewed source is committed, a job-specific
accepted JSON review receipt must bind that exact commit/tree and wrapper hashes
before the single capture job can be prepared and submitted through cluv.

## Fixed capture envelope

| Setting | Value |
|---|---|
| Cluster / account / QOS / partition | Mila / `mila` / `normal` / `main-cpu` |
| Nodes / tasks / CPUs | 1 / 1 / 1 |
| Memory / walltime / GPUs | 2 GiB / 10 minutes / 0 |
| Submissions / retry / requeue | 1 / none / none |
| Deadline | 570 seconds, inside the 600-second Slurm allocation |
| Supervisor Python | Pinned standalone CPython 3.10.19 at `/network/scratch/m/mehrab.hamidi/representation-collapse/toolchains/python/cpython-3.10.19-linux-x86_64-gnu/bin/python3.10` |
| Captured environment | CPython 3.10.19 venv at `/network/scratch/m/mehrab.hamidi/representation-collapse/environments/graph-native-cpu-20261015-calibration01/venv/bin/python` |
| Install receipt | `/network/scratch/m/mehrab.hamidi/representation-collapse/environments/graph-native-cpu-20261015-calibration01/preparation-receipt.json` |
| Identity output | `/network/scratch/m/mehrab.hamidi/representation-collapse/environments/graph-native-cpu-20261015-calibration01/node2a-environment-identity.json` |
| Source | `/network/scratch/m/mehrab.hamidi/representation-collapse/source` |
| Unique journal root | `/network/scratch/m/mehrab.hamidi/representation-collapse/runs/figure10-node2a-capture/<run-id>/` |
| STOP | `<journal>/<run-id>/STOP`, polled during execution |

The supervisor checks the exact live Slurm allocation (including `scontrol`,
zero GPUs and the effective cgroup-v2 memory limit), immutable clean source
commit/tree, the pinned bootstrap/importer/interpreter/environment receipt,
review receipt, unique absent output and run journal, fixed STOP path and
deadline. Before spawning and after capture, it checks that the venv Python
`lstat` type is a symlink resolving to the pinned standalone CPython binary and
that the target SHA-256 matches. The supervisor reads the bounded bootstrap
source bytes, verifies their pinned digest, and sends that exact byte string to
the child for compilation (without reopening a mutable source path). After the
child exits, it rechecks worktree cleanliness, commit/tree, and the bootstrap,
importer, job and supervisor hashes before accepting the inventory. It disables
bytecode caches and compiles the hash-verified bootstrap source directly. It owns and terminates the child
process group on STOP, signal or deadline. GNU `timeout` sends TERM at 575
seconds and allows 15 more seconds for supervisor cleanup, within the 600-second
Slurm allocation. Start and
terminal records include the live job ID and memory observations. Existing or
partial output is a hard stop; the operation is not retried or overwritten.

## Review and staging sequence

The project source must first be committed cleanly and synchronized to Mila.
The capture job script and supervisor have passed independent code review; a
job-specific accepted JSON receipt must now bind their exact source commit/tree
and hashes. The receipt must bind
those hashes, the bootstrap/importer pins, install receipt hash and allocation.
Stage that accepted JSON (for example, `scp <accepted-review.json>
mila:<fixed-path>`) at the fixed remote path
`/network/scratch/m/mehrab.hamidi/representation-collapse/runs/review/figure10-node2a-capture-review.json`
and compare local/remote `sha256sum` values. The capture plan must bind this same
receipt path and digest and the exact fixed envelope; the supervisor rejects a
different path, STOP location, output or deadline. The remote journal parent
is created owner-only by the supervisor if absent.

After independent review and dry plan validation, the intended single
submission shape is:

```sh
uv run --group ops cluv submit mila code/ops/tgrab_node2a_capture_job.sh --no-sync-datasets -- \
  --plan /absolute/path/to/reviewed-capture-plan.json \
  --plan-sha256 <reviewed-plan-sha256>
```

Do not use race mode, retries, or `cluv submit first`. This command is recorded
for planning and has not been run. Once capture completes, fetch the inventory
and independently compare its local SHA-256 with the remote digest and the
terminal journal. A second review must accept the resulting inventory before it
can be supplied to the later synthetic node2a fixture.

## Limits on interpretation

The resource request is a conservative engineering envelope, not a measured
efficiency claim. A successful capture establishes only the identity of the
installed runtime and the pinned source inputs. It does not establish native
fixture parity, training competence, reproduction of Figure 10, representation
collapse, or permission to access held-out data. Those require their own frozen
protocols and admissions.
