# Adopt or transfer GPU efficiency

Copy the **whole `gpu-efficiency/` directory**, including its references and
agent metadata. The Markdown entrypoint works with any agent that can read
files; native skill discovery depends on the agent application and may require
a new session. Do not assume already-running agents reload changed files.

## Shared installation on a machine

The deployed user-level copy is `~/.agents/skills/gpu-efficiency/`. A
`~/.codex/skills/gpu-efficiency` compatibility link points to it where installed.
Other tools can use an absolute path in their instructions; no special model
or API is required to read the guideline.

For a first copy from Mila to another machine (Mila SSH access required):

```bash
mkdir -p "$HOME/.agents/skills"
rsync -a mila:/home/mila/m/mehrab.hamidi/.agents/skills/gpu-efficiency/ \
  "$HOME/.agents/skills/gpu-efficiency/"
```

Before updating an existing copy, inspect local changes and compare package
hashes; preserve custom instructions. A skill installation never conveys
cluster credentials, campaign ownership, allocation budgets or W&B permission.

## Adopt in a different repository

From that repository root, choose a project-scoped copy:

```bash
mkdir -p .agents/skills
test ! -e .agents/skills/gpu-efficiency || exit 1
cp -R "$HOME/.agents/skills/gpu-efficiency" .agents/skills/gpu-efficiency
```

Merge the following short instruction into the repository's existing
`AGENTS.md`, `CLAUDE.md`, or equivalent agent entrypoint. Preserve its current
instructions; do not replace the file. Some projects keep all agent instructions
in a parent control plane: add the pointer there instead.

```markdown
For GPU job admission, scaling or utilization diagnosis, read
`.agents/skills/gpu-efficiency/SKILL.md` and follow its guideline.
Require a measured real-data efficiency receipt or a documented justified
exception before new long GPU jobs/sweeps. Preserve this project's scientific
controls, ownership, budgets, immutable sources and stop/resume authority.
Missing telemetry is unknown; a low utilization number alone does not authorize
cancellation or scientific changes. Record useful throughput and actual cost.
```

Alternatively point that instruction to the shared directory's **absolute**
`SKILL.md` path. A project copy is portable and versioned with the project;
a shared copy is easier to update for one user. Neither automatically edits
other projects' instructions.

## Handoff prompt for another agent

```text
Read /absolute/path/to/gpu-efficiency/SKILL.md and its GUIDELINES.md.
Apply them to this project's GPU work under its existing authority. First
identify owned jobs, measure actual progress and per-GPU efficiency, and create
a compact evidence record. Preserve scientific controls and active source pins.
Use existing authorization for fixes; do not infer cancellation or upload rights
from the skill. Consult the project's own instructions before any mutation.
```

With native skill discovery, the short request is:
`Use $gpu-efficiency for this project's GPU jobs.`

The detailed guideline is `GUIDELINES.md`; practical commands and the evidence
template are under `references/`. Cluster-specific examples are examples,
not defaults for every project. Authoritative source links are embedded next
to the relevant claims; recheck site documentation when changing cluster setup.
