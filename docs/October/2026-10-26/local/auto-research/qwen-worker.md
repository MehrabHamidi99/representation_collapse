# Persistent Qwen research worker

The [worker](../../../../../code/ops/qwen_worker.py) is a real OpenAI-compatible Qwen
client, implemented using the Python standard library. The coordinator launches
it on the recorded cluster with the verified backend supplied through environment
variables. This document and the [launch recipe](../../../../../configs/qwen-worker.example.json)
contain no endpoint values or credentials. Creating these files does not launch
a process or make a model call.

## Overnight continuation — 2026-10-12, current

The original worker continued across the earlier inspection gap. Fresh 17:26 UTC
heartbeat/call records verify ten valid model-identity-matched proposals, zero
errors and 55,482 reported tokens. Call 10 consumed the exact 600-character public
progress update; no restart or new lease occurred. The existing 24-hour/24-attempt
limits remain. A pinned metadata observer is running until 08:00 Asia/Yerevan on
October 13, with journals outside all worker evidence roots. It makes no provider
requests and cannot submit, restart or accept claims. See the
[overnight handoff](../../../2026-10-12/local/auto-research/overnight-research-2026-10-12.md). Public-protocol discovery
repairs below are not deployed into the immutable running worker; the bounded
progress feed carries current corrections instead.

## Earlier verification and continuity repair — 2026-10-12

The Mila-to-Tamia reusable master is absent at the latest probe. Current worker
liveness is unverified; its last successful check was 10:07 UTC. Authentication
recovery is pending, and no replacement worker has been launched. Before any
recovery, reconcile the recorded PID, final status and all attempt counters.
The [lease planner](../../../../../code/ops/qwen_resume_budget.py) subtracts prior calls
and elapsed time from the original approved 24-hour/24-attempt bounds; unknown
or live prior processes block a duplicate. This planner never launches or
grants approval.

A source repair corrects protocol discovery: public JSON protocols live under
`configs/experiments/`, not the old root-level glob. An explicit public allowlist
adds the bounded competence integration protocol and keeps approval/provider
settings excluded. Request limits remain 18,000 characters/2,048 tokens. Four tests
and independent privacy/budget checks pass. This source repair is published;
remote deployment remains pending. Existing scientific proposals stay unaccepted.

## Earlier renewed Tamia continuation — 2026-10-12

The owner explicitly approved a **separate 24-hour/24-attempt run**, including
the bounded project-excerpt payload, configured backend destination and
encrypted SSH forwarding. Automatic approval review initially rejected the
renewed launch because that transmission approval was missing; no call was
made before the explicit reply. The reply resolved the rejection.

The worker launched at **09:10:30 UTC** on Tamia3, PID **524659**, from clean
source `ec1c489fc97c2e6cf94fd58e8a4e04dd3e8ba901`. Output is
`/scratch/m/mehrab/representation-collapse/runs/qwen-continuation-20261012T091030Z`.
Its first two actual replies matched `Qwen/Qwen3.8-Flash-Next-FP8`, finished
with `stop` and passed the response schema; no provider errors were recorded.
The latest counts, usage and heartbeat timestamp are in
[structured status](../../../../../doc/auto/research-status-2026-10-12.json). The first run's 22-call
cap remains historical and consumed; it was a launch allocation, not a Qwen
provider restriction. This renewed allowance is neither recurring nor unlimited.

The evidence feed contains bounded verified numerical summaries and coordinator
corrections under project scratch, alongside the immutable source documents.
The second reply cited that feed, demonstrating consumption of new evidence.
Both proposals remain scientifically unaccepted: the first suggested Floquet
exponents without certification and an unsupported rank stopping rule; the
second proposed synthetic simulation to diagnose a trained checkpoint and
confused raw with population-normalized error gates. Schema validity and model
participation do not establish research quality. No proposal text was executed.

Availability requires the Mac's local proxy and both encrypted SSH hops to
remain live. The worker's runtime/call caps and three-error stop rule still
apply; these dependencies prevent a promise of uninterrupted 24-hour service.
The forwarding does not alter shared SSH configuration or close other projects'
masters. Credentials were delivered only into the worker environment; provider
configuration remains outside Git. This worker watches evidence and writes
proposals, with no training, submission, cancellation or claim-acceptance role.

## Worker contract

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
It requests direct answers with `chat_template_kwargs.enable_thinking=false`
and `reasoning_effort=low`, and asks for concise JSON under 900 answer tokens.
These settings follow the official [Qwen3.8 model card's API and non-thinking
examples](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#instruct-or-non-thinking-mode);
the answer target is a prompt instruction, while 2,048 is the request's hard
output-token cap. Backend compatibility must be checked through a successful
actual worker call before deployment. A missing final answer or
`finish_reason=length` fails the attempt with an explicit error; reasoning text
is never parsed as final JSON. Call status records finish reason and reasoning
character count, and usage includes consumed tokens even on such failures.
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

On 2026-10-11, the direct-answer configuration passed one actual local worker
call to the authorized backend. Requested and returned identity both matched
`Qwen/Qwen3.8-Flash-Next-FP8`; the reply finished with `stop`, contained zero
reasoning characters, and produced one schema-verified proposal. Usage was
5,509 prompt tokens and 464 completion tokens, totaling 5,973. Evidence is
retained locally under `runs/qwen/direct-json-verification-20261011T081305Z/`.
This verification process exited at its one-call cap; it did not start a daemon
or establish continuing deployment. The previously failed deployment attempt
and this verification consume two of the phase's 24 attempts, leaving 22 for
the replacement daemon. The proposal remains pending scientific review.
