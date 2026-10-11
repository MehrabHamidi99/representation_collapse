# Persistent Qwen research worker

The [worker](../../code/ops/qwen_worker.py) is a real OpenAI-compatible Qwen
client, implemented using the Python standard library. The coordinator launches
it on Mila with the already verified backend supplied through environment
variables. This document and the [launch recipe](../../configs/qwen-worker.example.json)
contain no endpoint values or credentials. Creating these files does not launch
a process or make a model call.

Invoke it with `python3 code/ops/qwen_worker.py --project PROJECT --output NEWDIR
--cluster LOCAL_CONTROLLER_CLUSTER --max-hours 24 --max-calls 24 --interval 60
--research-interval 3600`.
Set `QWEN_BASE_URL`, `QWEN_MODEL`, and `CAIRN_API_KEY` in the worker environment.
The coordinator delivers credentials through secure stdin into the launching
process's environment; never put the key in argv, shell history, a job script,
this JSON recipe, a log, or a file. Do not use shell tracing when delivering it.
Keep its output under project scratch and retain its process identifier for
termination and liveness checks.

The worker checks for evidence changes every 60 seconds and produces another
research proposal hourly even when evidence is unchanged. Its initial snapshot
triggers a real Qwen request. A successful unchanged snapshot does not cause
polling-time model calls. Scheduler elapsed time is deliberately excluded from
the evidence fingerprint. Failed provider/schema calls retry after the polling
interval, and three consecutive failures stop the process. Every HTTP attempt
consumes the call cap. There is no paid fallback. Call cap, runtime cap and
SIGTERM/SIGINT also stop it; a new output directory is required for a restart.
The runtime includes HTTP requests, whose timeout is bounded by remaining time.

Evidence is bounded to tracker/research-plan excerpts, phase-1 JSON protocol
excerpts, the novelty review, and the newest project `status.json`, `metrics.json`
and `manifest.json` artifacts under `runs/` and `logs/`. Repeat `--artifact-root`
only for an explicit project artifact directory. No checkpoints, PDFs,
datasets, hidden files or approval/credential configuration are read. Traversal
is bounded, does not descend directory symlinks, and excludes the worker's own
output so its heartbeat cannot trigger self-generated requests. Excerpts can
omit information; every review must acknowledge this limitation.

`--cluster` is required and records the exact identity of the local Slurm
controller queried by `squeue`. Repeat `--job-id` only for coordinator-known
project Slurm IDs on that controller. Additional IDs can be obtained from the
selected project status artifacts' numeric `slurm_job_id`, `scheduler_job_id`,
or `job_id` fields only when the artifact's `cluster` exactly equals the worker's
controller cluster. Missing or different cluster identities contribute no IDs.
This prevents synced Alliance artifacts with coincident IDs from matching
unrelated Mila jobs. Artifact excerpts from other clusters remain research
evidence but never identify jobs for the local scheduler query.
Without known IDs there is no scheduler
query. Otherwise the only scheduler command is `squeue --me --noheader
--format=%A|%T|%j`; other jobs are discarded before persistence or Qwen input.
The worker neither submits nor cancels scheduler jobs.

Each request has at most 18,000 message-content characters and 2,048 requested
output tokens. The worker sends the actual model identifier from the environment,
requests JSON output, and uses the key only for the HTTP authorization header.
It does not execute response text or shell instructions. The five-field review
schema is:

- `assessment`: nonempty string.
- `evidence`: nonempty list of `{source, claim}` objects. Every source must be
  an exact supplied evidence identifier; claim text still needs human/scientific
  review.
- `next_experiment`: `{question, design, resources, stopping_rule}` with
  nonempty strings.
- `acceptance_checks`: nonempty list of nonempty strings.
- `limitations`: nonempty list of nonempty strings.

Extra fields, unknown evidence identifiers and malformed responses fail
verification. The provider's returned model identifier must exactly match the
requested model identifier; missing or mismatched identity fails the attempt.
The requested identifier must use the `Qwen/` namespace. This checks the
provider's declared identity; it does not attest the underlying weights.
Run configuration and call status record requested and returned identity.
Schema verification checks structure, source identity and provider model identity;
it does not certify factual claims, experiment merit, job success, approval,
or scientific acceptance. Every proposal remains `scientific_decision=pending`.
The worker can actively design the next experiment but cannot modify code,
submit work, acquire resources or adopt its conclusions in this initial phase.

Each immutable `call-NNNN/` contains evidence and its SHA-256 fingerprint,
the request body, raw reply (with credential-echo redaction), validated review
when available, reported usage and atomic status. Failed attempts remain on
disk, with HTTP error bodies preserved when available. A missing provider usage
field makes that cumulative token total unknown (`null`), rather than zero;
failed requests may have consumed server tokens. `heartbeat.json` records call
attempts, verified proposal count, elapsed runtime, consecutive errors and usage.
It is refreshed after polling and before/after calls; a pending call can delay
it up to the configured HTTP timeout. Final `status.json` records the stop reason.
Three consecutive errors also return a nonzero process exit status.

The coordinator must inspect a successful actual request/review and heartbeat
after launch before claiming the Qwen subagent is running. A PID alone does not
prove model participation. Preserve launch provenance, output location and
resource usage separately, and stop only this worker's recorded process when
requested or when its bounds are exhausted.
