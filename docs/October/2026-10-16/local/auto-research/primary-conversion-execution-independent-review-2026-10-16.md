# Independent review of observed primary conversion execution

2026-10-16. **The observed completed run and independent data-verification
receipt are internally consistent. Full engineering closeout remains pending.**
This is a local review of retained remote-inspection metadata, not another raw
CSV reduction or a fresh cluster inspection. No target CSV/split payload was
read locally, and this reviewer made no SDK, model, remote or submission call.

## Observed execution and data evidence

The owner approved one CPU conversion-plus-verification job from published
`0fe14aa7e8433da5a019e3e80952742cb0e551e7`. Native cluv returned **11273685**.
The retained October 16 02:13 UTC inspection reports Slurm **COMPLETED, 0:0,
126 seconds**, node `cn-h001`, with an empty queue query for that job. The
scheduler receipt specifies Mila/account mila/QOS normal/main-cpu, one CPU,
2 GiB, zero GPUs, ten minutes and no requeue; Slurm also reports zero restarts.
Recorded runtime verifies physical cgroup **2,147,483,648 bytes** and CPU affinity
`[60]`. Worker/supervisor elapsed times are 125.429/125.754 seconds, within the
original 590/600-second limits. Worker peak RSS is 205,946,880 bytes. These
resource observations concern this completed job only.

The production independent-verifier receipt records:

- exact pinned raw CSV: **6,907,326 bytes**, SHA256
  `5a19f9e7949668473bc61cb59b6215f25f093ea0ee2d45b5e143338ded5523e3`;
- **337,920 raw rows and unique directed edge occurrences**, zero identical
  duplicates and zero source-marker rows;
- **3,072 converted and semantic snapshot comparisons**, three split-byte
  comparisons, one complete canonical-manifest comparison and seven stable
  input-file checks;
- N100, 32 A/32 B, 40/4/4 periods; train [0,2560), validation [2560,2816),
  test [2816,3072), with primary eligibility and independent-of-producer flags.

Under the accepted verifier's actual source contract, successful completion
checks all raw lexical/mask/unit-weight, missing-step, reverse-edge, negative
feasibility, distinct-pattern and repeated-order gates, then compares every
canonical converted snapshot and full manifest against its independent raw
oracle. This review verifies the recorded receipt and source lineage; it does
**not** claim to have rerun those checks locally.

Converted manifest SHA256 is
`a6ff50183de570ff17cfbe92b3f1976bac9d2d85557ecc11379e24b2c737d1ce`.
The four converted files total **2,997,959 bytes**. Producer counts, independent
counts, worker manifest/verification hashes and all three split size/hash
records agree with the terminal inventory. The independent receipt SHA256 is
`9e279518ab997e8a7596a3a23f67fd3d9ea0362d761d814149fb5eec8e88e2c8`.

## What was independently checked here

All 14 staged/claimed source pins match the exact files in published commit
0fe14aa. Runner/config/verifier identities match their accepted reviews.
Owner-grant/admission summaries, exclusive claim, native expansion and actual
job receipt bind the same source and plan SHA
`b17364f2d746dab6087599142e850e3b3bebc06379179e04151700fc241f9dcb`.
The record shows one guarded expansion/submission, one conversion attempt and
one verifier attempt. **One allocation is consumed; zero remain; no retry is
authorized.** Earlier intent/approval snapshots retain their historical states.

The inspection embeds 11 metadata JSON objects. Reconstructing each with the
pinned writer's exact sorted/indented encoding reproduces its reported byte
length and SHA. Every inspected attempt artifact agrees with its terminal
inventory entry; all 21 inventory entries sum to the declared 3,031,607 bytes.
The one documented SDK debug symlink is counted without following its target;
the converted files have no such exception.

The supervisor reports no cleanup errors or signals and original-process exit
for the **two retained identities**, worker PID2057660 and SDK-core PID2057939.
This establishes the scope of the recorded reconciliation, not a fresh PID
inspection or reconciliation of every historical spawn-journal entry.

The offline reporting receipt records SDK0.30.0, project representation-collapse,
run `f177579e09d943c5917bd68522c478a7`, completed state, finish returned and last
attempted/completed step 2. Its identities agree with the conversion receipt,
and the recorded privacy settings disable automatic code/environment/statistics
capture. Dashboard delivery remains false/pending. The raw offline transaction
has a recorded inventory hash, but its CRC/records/history/exit have **not** yet
been decoded and independently reconciled for this run.

## Why full closeout is not accepted

The follow-up collection returned SSH255 without metadata. A fresh native-cluv
connection at **03:39:24 UTC** failed with **Network is unreachable** to Mila's
port2222. These are observation-route failures; they neither undo the earlier
terminal Slurm observation nor authorize rerunning a completed job.

The missing closeout evidence is explicit:

1. Actual plan, owner-approval and remaining admission/readiness JSON contents
   reconciled with their declared hashes; in particular exact worker admission,
   original clock, parent identity, scheduler pin and worker argv binding.
2. Fresh post-run source HEAD/clean status and source/artifact hash observations,
   distinguished from the successful pre-run staging checks already recorded.
3. Full spawn-journal contents and any fresh retained-identity observation;
   current cleanup acceptance is limited to the supervisor's two reported exits.
4. CRC/record decoding of the exact offline SDK transaction, actual history
   steps 0/1/2 and their values, run/identity/privacy records and terminal exit.
   A reporter receipt alone does not replace this stream check.

When the route returns, collect these **existing job11273685 metadata only**,
retain the original failed collection/login records, and reconcile exact hashes.
Do not resubmit conversion, renew the consumed allocation or broaden its frozen
lexical policy. The local consistency record is
`runs/review/primary-conversion-execution-20261016/independent/metadata-crosschecks.json`,
SHA256 `24dcce4cdbd3901f848539f9a9eaf8380767b01cfb85b366585a1c4529adb30e`.

No trained model or checkpoint was produced here. Competence, temporal-graph
collapse, causal memory, periodic hidden/Jacobian closure and Floquet claims
remain unverified. This review accepts the scoped recorded execution evidence
conditionally and leaves full closeout **unaccepted**.
