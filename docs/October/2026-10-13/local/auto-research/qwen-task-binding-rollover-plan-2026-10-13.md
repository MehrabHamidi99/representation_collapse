# Task-binding Qwen rollover preparation — 2026-10-13

Prepared at 11:23 UTC; phase1 executed under coordinator authorization at 11:27–11:32 UTC. **Claim, native-cluv source staging and unexecuted helper preparation completed; phase2 published only the new compact feed.** The subsequent old-cohort shutdown was rejected by automatic approval review before execution. No STOP/cancellation/window/client/monitor/guard launch occurred. A specific owner-confirmed graceful rollover is now pending; unlimited existing Qwen operation remains active. The reviewed, pushed replacement is `a36cae812490af89f57843771e7aabae3aab0781`; its worker SHA-256 is `0f78851940d7f3323fc5cf4408fd0f682a33d93d151c6be4ff6358e1185d8997`. The active source remains `999d88d` / worker `8d9e6149...`.

## Fresh original-cohort certificate

At 11:17:13 UTC, all five project-owned processes were present on `cn-f003.server.mila.quebec` with matching owner UID, source argument/hash, and job 11228391 cgroups: leaf 3244371; project supervisor 3244355; observer 3248354; monitor supervisor 3248325; physical hosting guard 3255471. The allocation is RUNNING, 2 CPUs / 4 GiB / zero GPUs, and expires October 14 at 08:55:56 UTC. Its armed guard triggers at 08:52:40 UTC.

The original worker has six completed provider attempts, six schema-valid reviews, zero pending requests, zero consecutive provider errors, and known usage of 26,117 prompt + 3,745 completion = 29,862 tokens. All six completed metadata records were reconciled; issues were empty. The observer had 15 observations and fresh independent heartbeats. These counters describe provider consumption; the six proposal answers have not earned complete scientific acceptance.

Receipt: `runs/ops/qwen-task-binding-rollover-20261013/fresh-certificate-20261013T111713Z.json`. Initial diagnostic import created an ignored Python bytecode cache at 11:15:44 UTC; the final diagnostic used `python3 -B`. No source, worker, runtime controls, or model journals were changed. This is not a certification of old-file immutability.

## Resources and authority

Use only existing allocation 11228391 and its authenticated Mila master. There are zero new allocations, Slurm submissions, GPUs, weights, training, or environment builds. Use the existing protected backend configuration and key in-node only. Model identity, 18,000-character evidence bound, 2,048 output-token bound, minimum 60-second provider spacing, retry policy, response schema, and null total-call/runtime caps are preserved.

Use precisely the existing project session `representation-collapse-qwen-20261013` on `/tmp/radius-control/tmux-11228391.sock`. Do not create a second shared session, modify Radius windows/settings, or close the Slurm-owned Tamia socket.

All new paths and identities are frozen in ignored `rollover-manifest.json`. The fresh cohort is `qwen-taskbinding-20261013T1123Z`; its source staging path ends in `-a36cae8/source`. Monitoring is outside the new source's runs/logs and evidence feed. Its daytime deadline remains October 13 at 20:00 UTC, independently of the leaf. The physical guard continues to protect the allocation deadline.

## Exact preparation and source staging commands

The registration and staging helpers executed successfully in phase1. Their exact commands below are a record and must not be rerun over existing paths. `claim.json`, `source-stage.json` and `helpers-prepared.json` preserve the receipts. Claim registration permits the three explicitly named active old-cohort claims while rejecting unexpected active claims for this project; it requires fresh, absent output paths. The staging helper uses native cluv 0.1.5 `cluv.cli.sync.clone_project`, a project-only scratch Git mirror, and an isolated local controller on tracking branch `main`. It avoids the earlier detached-HEAD failure and checks the committed worker/observer hashes. It never changes global cluv/Git settings or the canonical checkout.

Executed phase1 commands, from the repository root:

```bash
git clone --no-hardlinks --bare . runs/ops/qwen-task-binding-rollover-20261013/bundle-controller.git
git --git-dir=runs/ops/qwen-task-binding-rollover-20261013/bundle-controller.git update-ref refs/heads/rollover-a36cae8 a36cae812490af89f57843771e7aabae3aab0781
git --git-dir=runs/ops/qwen-task-binding-rollover-20261013/bundle-controller.git bundle create runs/ops/qwen-task-binding-rollover-20261013/source-a36cae8.bundle refs/heads/rollover-a36cae8
.venv/bin/python -B runs/ops/qwen-task-binding-rollover-20261013/register_claim.py --execute-after-parent-go
.venv/bin/python -B runs/ops/qwen-task-binding-rollover-20261013/stage_native_cluv.py --execute-after-parent-go
```

The staging receipt must show the exact commit, clean source, worker `0f788...`, observer `3f260...`, and existing namespace/host. The old client remains active throughout staging. If any command is ambiguous or interrupted, inspect its local intent/receipt, remote claim/path/commit and owned processes; preserve failed attempts and reconcile before retrying.

For further control actions enter the existing namespace:

```bash
ssh -t -oClearAllForwardings=yes -oBatchMode=yes -oConnectTimeout=10 -oConnectionAttempts=1 mila 'srun --jobid=11228391 --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 --pty bash -l'
```

Before STOP, refresh scheduler expiry, source hashes, process UID/argv/cgroups, metadata pending-call state, and ownership. Register any additional monitor/guard claim before its first mutation. Keep all secret handling inside the node; do not print environment variables or protected configuration.

## Ordered old-cohort handoff

Create one inactive project-owned anchor window **before** stopping old processes. Otherwise, closing the supervisor, observer and guard windows can destroy the approved session before replacement. This anchor keeps the same session alive and performs no model work:

```bash
tmux -S /tmp/radius-control/tmux-11228391.sock has-session -t representation-collapse-qwen-20261013
tmux -S /tmp/radius-control/tmux-11228391.sock new-window -d -t representation-collapse-qwen-20261013 -n rollover-handoff-20261013T1123Z 'bash --noprofile --norc'
```

Check the window name is absent before creation; do not duplicate on an uncertain return. Request graceful stop of the **old leaf only**, using a new exclusive STOP file:

```bash
python3 -B -c 'from pathlib import Path; p=Path("/network/scratch/m/mehrab.hamidi/representation-collapse/runs/qwen-continuous-20261013T093936Z/STOP"); p.open("x").close()'
```

Wait for any already-started HTTP call to finish (current request timeout is 180 seconds), a valid old terminal status, zero pending call, final accounting, and absence of **both** old leaf 3244371 and project supervisor 3244355. Stop controls are not proof of process exit. Do not send signals to an unidentified/reused PID. If pending work or inconsistent metadata persists, preserve it and stop the rollover rather than duplicate a client. The supervisor should naturally record worker exit; its runtime STOP is an escalation only after coordinator review.

Capture final old heartbeat/config/status and per-call status/evidence/review/usage hashes before disabling its observer. Preserve its six-call cohort and counts; do not copy old tasks into the new task inbox.

After old leaf/supervisor exit, stop only the old monitor wrapper:

```bash
python3 -B -c 'from pathlib import Path; p=Path("/tmp/radius-control/representation-collapse-monitor-20261013T1001Z/STOP"); p.open("x").close()'
```

Verify observer 3248354 and wrapper 3248325 are absent, and capture both terminal records. Then cancel only the old guard:

```bash
python3 -B -c 'from pathlib import Path; p=Path("/tmp/radius-control/representation-collapse-root-qwen-20261013T093936Z/hosting-expiry-guard/CANCEL"); p.open("x").close()'
```

Verify guard 3255471 is absent and its own terminal/claim record says cancelled. A cancellation requested before old leaf exit would remove its hosting protection and is prohibited. Existing claims are finalized by their owning controllers; do not take over foreign/stale claims.

## Replacement helpers and exact phase2 launch commands

Phase1 prepared unexecuted helpers inside the new project runtime. The supervisor is byte-identical to the existing reviewed controller (SHA-256 `7ca8c6b3dae1c1d90d8e5063513b9b33d5e0fd256141a5d933619567849e8eb9`). Its existing interface receives the config path as an argument; no controller source change is required. Ten public config keys change source/run/claim/runtime/deadline/feed/journal identities, and all nonpublic fields remain equal, in-node only. The preparation's first pre-write assertion assumed a literal config path; it failed before creating helpers and is preserved in `helpers-prepare-attempt-001-failed.json`. The reconciled retry used the actual argv interface successfully.

The new monitor wrapper SHA-256 is `b7ebf15733b6a3d3b299b20aa1cc5dbe1bb5c9122d6dc8af726d719cc79a5255`. It changes public paths/hash identities and replaces the old fixed PID with the actual new `launch.json` PID, checking source/host/UID/argv/cgroup before creating the reviewed observer. Interval 300 seconds, stale allowance 600 seconds and deadline 20:00 UTC are unchanged. The guard SHA-256 is `33e28acdad5b83ba298f4a5bb60d659961e4053ae41d6f53a7c1fe5df13ed45b`; its sole code change is the exact own run-path assertion. Guard trigger remains October 14 08:52:40 UTC before allocation expiry 08:55:56 UTC. No key or provider configuration was printed or moved outside the node.

After separate phase2 authorization and complete old-client/observer/guard handoff above, inside the allocation launch the prepared supervisor in the existing own session:

```bash
tmux -S /tmp/radius-control/tmux-11228391.sock new-window -d -t representation-collapse-qwen-20261013 -n taskbinding-client-20261013T1123Z '/network/scratch/m/mehrab.hamidi/representation-collapse/source/.venv/bin/python -B /tmp/radius-control/representation-collapse-root-qwen-taskbinding-20261013T1123Z/project_supervisor.py /tmp/radius-control/representation-collapse-root-qwen-taskbinding-20261013T1123Z/client-config.json'
```

Reconcile the new supervisor and leaf actual PID/source/UID/cgroups and `launch.json` before proceeding. There is no auto-restart. From the local repository, the prepared phase2 installer then registers distinct monitor/guard claims, records the root claim's explicit subpath handoffs, installs the prepared files, and binds guard configuration to the verified new PID:

```bash
.venv/bin/python -B runs/ops/qwen-task-binding-rollover-20261013/install_phase2_monitor_guard.py --execute-after-phase2-go
```

The installer is unexecuted. It requires all five old PIDs absent, valid old stopped status and cancelled guard, fresh new launch provenance, and absent new monitor/guard scopes. It launches no processes. On an ambiguous failure reconcile partial claims/files; do not rerun blindly.

Inside the allocation, start the physical guard first, then the metadata monitor, in two new windows of the same existing project session:

```bash
tmux -S /tmp/radius-control/tmux-11228391.sock new-window -d -t representation-collapse-qwen-20261013 -n taskbinding-hostguard-20261013T1123Z 'python3 -B /tmp/radius-control/representation-collapse-root-qwen-taskbinding-20261013T1123Z/hosting-expiry-guard/hosting_stop_guard.py /tmp/radius-control/representation-collapse-root-qwen-taskbinding-20261013T1123Z/hosting-expiry-guard/config.json'
tmux -S /tmp/radius-control/tmux-11228391.sock new-window -d -t representation-collapse-qwen-20261013 -n taskbinding-monitor-20261013T1123Z 'python3 -B /tmp/radius-control/representation-collapse-monitor-taskbinding-20261013T1123Z/claim_supervisor.py'
```

Check each unique window is absent before creation and preserve intent/return receipts. Independently certify new guard armed, monitor observations/cursor, all owned PIDs and source/host/job/UID identities, fresh root/monitor/guard claims, and STOP controls. The original cohort and counters remain immutable historical accounting. First startup may produce an unassigned evidence-triggered review; verify its model/schema identity separately and publish the affine task as the first assigned question, without replaying old task IDs. Only actual answer review can establish prompt relevance. Degraded/stalled status remains distinct from stopped.

## First task: affine commuting null

The phase1 coordinator authorization selects the concrete affine question first. The fresh, unpublished packet is `first-task-packet.json`; its source question is preserved separately from the prospective v5 packet. Compare A(z)=Jz+(1,0), B(z)=Jz+(0,1), J=diag(0.8,0.5), in chronological ABAB versus AABB from identical initial samples. The expected tangent products are both `diag(0.4096,0.0625)`; terminal means from z0=0 are ABAB `(1.312,1.25)` and AABB `(1.152,1.5)`. Paired perturbations propagate identically and centered covariance agrees for identical initial distributions. Different affine offsets do not alone establish excess information loss. Invertible finite-time products do not prove rank loss; noisy/readout-limited recoverability needs separate assumptions.

Review the actual answer for task relevance, arithmetic, full-state/readout scope, and absence of gate relaxation or unsolicited jobs. Schema-valid consumption is not scientific acceptance. Only after this review should the second v5 question be published; do not create a recurring idle-call schedule.

## Second task: prospective v5 review

The coordinator selected v5 conditional-equivalence review as the second assigned task. The exact fresh task packet is ignored `second-task-v5-packet.json`; no task has been published. Its source inventory and current full-file hashes are in `first-task-evidence-sources.json`. Stage bounded public excerpts describing the latest 32 actual fits / 110,200 updates and the current prospective v5 protocol. Preserve excerpt labels, hashes and total 18,000-character request bound. Monitoring/status metadata stays outside the evidence roots; no periodic idle calls are introduced.

The question asks for conditional expected-loss and unclipped-gradient equivalence at fixed weights, exact `1/(32*8)` normalization, complete differentiable prefix state including LSTM h/c, paired-arm dependence, the conditional-noise variance decomposition, and its limits for clipping/AdamW/compute exposure/competence. It requires distinguishing prior singleton passes, all-eight original R0 eligibility, numerical calibration, prospective production and scientific acceptance. It prohibits invented results, gate changes, repeated failed schedules and new jobs.

Coordinator reference: uniform QUERY conditional on the same sixteen independent prefix histories, deterministic native model, complete shared differentiable state and the original targets make the conditional mean loss and raw gradient equal to averaging the eight counterfactual QUERY branches. The two paired arms do not create 32 independent histories. Law of total covariance removes the nonnegative conditional-selector-noise term for the unclipped fixed-weight gradient under finite second moments. It does not establish covariance reduction after nonlinear clipping, equivalent AdamW updates, matching realized training paths, equal computational exposure, or competence. No v5 production fit or confirmation score exists in this preparation record. Singleton q7/q15 success does not meet the frozen all-eight original R0 gate.

First-call schema/model validity and prompt-task binding must be independently verified against the actual task ID/evidence hash. Accept/reject the reasoning separately; schema/count increments never self-certify scientific correctness. A generic scheduling answer fails this test. No accepted trained-model claim follows even from a correct analytic response.

## Approval-pending handoff after automatic review

At 11:38:45 UTC, phase2 published the prepared eight public feed artifacts into the new cohort's evidence directory (3,726 characters total, each at most 650, source hashes pinned to a36cae8). The old worker does not read this new feed. The next action, `stop_old_cohort.py --execute-after-phase2-go`, was rejected before execution by automatic approval review. Its stated reason was that stopping the active Qwen worker and observer/guard cohort could interrupt an in-flight provider call and service continuity, and that research authorization was insufficient for this exact rollover stop. The review explicitly prohibits indirect workaround execution.

The coordinator stopped the rollover and will request owner confirmation for the specific graceful replacement. This objection does not revoke or narrow the owner's authorized unlimited Qwen research operation; the original worker still runs with null call/hour caps. No anchor, old STOP, monitor STOP, guard CANCEL, new client, new guard or new observer was created/launched. No provider request or task was triggered by this operation.

The independent read-only certificate at 11:41:34 UTC verified all five original source/UID/cgroup identities, fresh heartbeat, waiting-for-evidence activity, zero pending requests, no active task, unchanged six completed/schema-valid calls, 29,862 tokens, and absent stop/cancel controls. It preserves hashes of all original call files. The two prepared task packets remain unpublished; v5 remains the second question after independent review of the affine answer.

Review records and exact file hashes are in ignored `APPROVAL-PENDING-HANDOFF.json`. Safe continued work consists of local scientific implementation/review and read-only monitoring of the existing cohort. Do not retry shutdown or launch the replacement until this exact review objection is resolved.
