---
name: model-routing
description: Route authorized project research tasks to deterministic tools and currently exposed native subagents, with bounded evidence packets and distinct engineering versus scientific review responsibilities.
---

# Route work using live capabilities

Use deterministic commands for file inventories, hashing, metric extraction,
schema/dependency checks, scheduler polling, and reproducible calibration.
Reserve model reasoning for experiment design, implementation choices, evidence
conflicts, and scientific interpretation. This skill does not start persistent
workers, schedule paid calls, or authorize new compute.

At task time inspect the available delegation tool and model overrides. The
setup session exposed `gpt-6.1-sol`, `gpt-6-astra`, `gpt-6-sol`,
`gpt-6-luna`, and `gpt-5.6-sol`; availability can differ later. Do not infer
provider access from local caches or this list. The owner separately authorized
a bounded Qwen cluster worker on 2026-10-11; its actual run and proposal-only
contract are documented in `doc/auto/qwen-worker.md` and the tracker. The
initial worker stopped at its consumed cap. On 2026-10-12 the owner explicitly
approved a separate 24-hour/24-attempt Tamia continuation; its actual first
response matched the requested Qwen identity. Read current heartbeat and
approval evidence before treating that route as active. The October 12 bounds
describe that historical cohort. On October 13 the owner removed fixed Qwen
call/runtime caps and authorized OpenAI fallback. Use the current continuous
project worker and currently exposed native OpenAI agents within the approved
project-excerpt scope; do not infer credentials or entitlement to an unavailable
external API. Qwen remains a proposal-only cluster process. Record actual
requests, answers and scientific acceptance separately from worker liveness.

Prefer the inherited parent model and effort for native delegation. Specify a
different model/effort only when the user or applicable instructions request
that override. Current native tool overrides require a self-contained task
with `fork_turns="none"`; full-history forks inherit the parent's settings.
When an override is authorized and available, route by the work:

- Small bounded clerical/triage tasks can use a lighter exposed model, with
  deterministic verification of their output.
- Implementation, environment repair, routine analysis, and research planning
  suit the exposed workhorse model (for example `gpt-6.1-sol`).
- Novel causal/methodological claims, disputed assumptions, and surprising
  scientific results warrant an independent strong review when available
  (for example `gpt-6-astra`). Model choice does not certify correctness.

The user requested subagents to manage research for this repository. Apply
[the agent protocol](../../doc/auto/agent-protocol.md): send one bounded task,
exclusive paths, source evidence, acceptance checks, a stopping condition,
and a durable output. Limit concurrency to the actual session's slots and avoid
two workers modifying the same file. Review delegated edits and evidence before
marking the node verified.

On a tool/provider error preserve the attempt and inspect actual local/remote
state before any side-effect retry. One compact read-only diagnostic can
distinguish a transport/serialization failure from an operation failure.
Repeated route errors stop that route; choose a healthy exposed route only
within existing authorization. A timeout hands back partial artifacts and a
smaller continuation, not a restarted oversized prompt. Record unavailable
usage as unknown, not zero; do not import budgets from the template project.
