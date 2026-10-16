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
| Project source | `/network/scratch/m/mehrab.hamidi/representation-collapse/source` (clean Git worktree pinned by commit/tree) |
| Pinned upstream model wrappers | `/network/scratch/m/mehrab.hamidi/representation-collapse/runs/graph/upstream-pin-20261012/source` (revision label `31293102b065feef7817ff86211427d7594edfdb`; three executable wrapper files checked against SHA-256 pins in the capture plan/receipt) |
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

## Current state — 2026-10-16 18:26 UTC

The capture described above has now completed as Mila job `11280346` (exit
`0`, 12.14 seconds, `main-cpu`, 1 CPU, 2 GiB, zero GPUs, no requeue/restart).
The terminal record and locally fetched environment identity inventory agree
on the plan and inventory hashes. An independent reviewer initially rejected
the inventory for node2a because this document's former preparation validator
expected a stale module/schema layout. The validator was updated to accept only
the exact source-identity pins and runtime module-to-distribution map, with
mutation tests; independent review accepted that change, and the captured
inventory now passes validation.

The exact source is commit `58fb20259dfd48d29b3a89cd863e30c85f47fdde`, tree
`e9d8e11aa0d3450cd6e79d81fdf503c3ab8ff007`. The candidate synthetic node2a
plan is retained locally at
`runs/review/figure10-node2a-plan-20261016T182500Z-58fb2025.json`, SHA-256
`15175738b983e75da68373b158caff4ac6fec0f8705de5c54f0a8fef59fcaa6d`. It
binds the reviewed bootstrap/importer/supervisor/job wrapper, exact runtime
receipt and inventory, unique output/STOP paths, and the fixed 570-second
CPU-only allocation. It invokes synthetic forward/loss/backward parity with
zero optimizer updates and no primary/validation/test data; “zero-update” does
not mean zero numerical work. Scientific acceptance is explicitly false.

A fresh read-only Mila check at 18:26 UTC found no queued
`cluv-rc-calibration` or node2a job and confirmed the unique planned output
path did not exist. The listed jobs belong to other experiments and remain
untouched. The independent reviewer accepted the exact plan for a single
synthetic parity submission. Mila job `11280406` then failed after 1.34 seconds:
the supervisor created the run journal before starting the bootstrap, while the
bootstrap required its output path to be absent. No fixture arrays were
published and no model/data training path ran. Its owner-only
`supervisor-start.json` and `supervisor-terminal.json` are retained locally in
ignored `runs/figure10-node2a/node2a-20261016T182500Z-58fb2025/`. The failed
run ID will not be reused.

The corrective change routes bootstrap atomic publication to the child path
`<run>/fixture/`, leaving scheduler/supervisor records at `<run>/`. Both the
preparation helper and supervisor now construct/verify this child path. Focused
tests pass 30/30, plus compilation, shell syntax and diff checks. Independent
review accepted the exact path-fix hashes: preparation helper
`809c0ddfc2a3e4ba9d68ece3155d1761cdf3f13a1066e2635a28516181e74a09`, supervisor
`a79a96a6235cb9ada038bb19390240a48a29d5aab64778fe672f06f89b1f7ebd`, and
tests `60d98d1d837c1226758b08e321c3846b6692146e06d816864d980df17b9e0781` /
`4291b57faf0502009294d35998c38e276c1681366a7be6289b806d9a5302091f`. The next
attempt must use a new source commit, plan hash, fresh queue/output check and
unique run ID. The compute waiver applies; it does not waive source, scheduler,
STOP/deadline, provenance, or scientific interpretation gates. This fixture
remains engineering evidence only.

### Follow-up — second fixture attempt and source-root correction (2026-10-16)

Job `11280432` failed in 7.35 seconds before model construction because the
supervisor routed `--source-root` to the project checkout, which lacks
`model/node_emb.py`. No data or optimizer path ran. Its retained start and
terminal records are under ignored
`runs/figure10-node2a/node2a-20261016T183200Z-3168e941/`.

The reviewed fix uses the separate pinned scratch source
`runs/graph/upstream-pin-20261012/source`, label
`31293102b065feef7817ff86211427d7594edfdb`, and exact SHA-256 pins for the
three wrappers. Supervisor validation checks those files before and after the
fixture; byte hashes, rather than an unavailable Git HEAD in that ignored
snapshot, establish executable identity. Independent review accepted the
preparation, supervisor and test changes (hashes recorded in the dated report).

No fixture parity has passed. The next attempt requires a committed clean
source, newly prepared unique plan/run ID, independent job-specific review,
and fresh queue/output checks. This zero-update CPU fixture is the only job
admitted at this point; it cannot establish scientific collapse.

### Follow-up — cluv script selection and successful node2a fixture (2026-10-16)

A v3 plan attempt was rejected because its review bundle's STOP path omitted the
last character of the unique run ID; the bundle and plan were corrected and
rebound before submission. A v4-reviewed attempt then exposed a cluv 0.1.5
configuration detail: `get_cluster_job_script_path` uses configured
`job_script_path`, overriding an explicit local script argument. Job `11280586`
ran default `phase1_job.sh`, failed immediately with `unknown argument` (exit
2:0, zero elapsed), and did not create the node2a output directory. Its failure
is retained; that run ID was not reused.

For the fresh v4 run, an isolated controller copy at committed main
`40942e5642c02b7098263dfe1294f907a71d2d16` changed only its local
`job_script_path` to `code/ops/tgrab_node2a_job.sh`. Cluv 0.1.5's resolved
remote script path was checked and independently reviewed. The project source
and remote checkout stayed clean and exact; `SKIP_CLEAN_GIT_CHECK=1` applied
only to this local controller config override, while `GIT_COMMIT` remained the
reviewed committed SHA. The exact cluv sbatch record names the node2a wrapper.

Job `11280629`, unique run
`node2a-20261016T184800Z-40942e57`, completed exit 0 in 14 seconds with 1 CPU,
2 GiB and zero GPUs. Its start and terminal receipts bind plan SHA
`b0ca7527f5809394444953370488d8eb0628af84fa346e04d2c9dbaf58c231c4`, source
commit/tree, runtime inventory and three pinned upstream wrapper bytes. The
synthetic fixture reports component and complete hidden/cell-state
round-tripping, forward/loss/gradient and mode/dropout parity, transaction
crash/reopen, zero optimizer updates, and no primary/validation/test payloads.
All six local artifact digests match Mila and their receipt bindings. This is
engineering parity only, with `scientific_acceptance=false`; it does not
reproduce Figure 10 or establish collapse. Independent artifact review is
pending. Before any data-bearing fit or held-out access, integrate and review
the missing production fit/replay/metric/plot workflow noted in the readiness
audit.
