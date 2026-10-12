# Representation Collapse — research operations

Study when recurrent models lose useful state dimension across modalities,
starting with temporal graphs and discrete Floquet analysis. The owner is
Mehrab Hamidi. Remote: `git@github.com:MehrabHamidi99/representation_collapse.git`.

## Start a session

1. Read this file, `doc/TRACKER.md`, and the latest dated progress report.
2. For scientific work read `skills/floquet-analysis/SKILL.md` and the relevant
   protocol in `doc/auto/`.
3. For delegated research read `skills/model-routing/SKILL.md` and
   `skills/experiment-loop/SKILL.md`.
4. Before cluster operations read `skills/project-ops/SKILL.md` and
   `doc/auto/compute-plan.md`; recheck the actual scheduler and cluv version.

## Scientific invariants

- Preserve inherited `floquet_analysis/`, `ACM___TGRAB/`, and
  `ACM___TGRAB_KDD_2026/` as source snapshots. New analysis belongs in `src/`,
  protocols in `doc/auto/`. Draft assertions are preliminary evidence, not
  reproduced findings. Record which draft and which active TeX include you use.
- Work with discrete recurrent updates first. Periodic input alone does not
  establish a periodic hidden trajectory or periodic Jacobians. Label products
  on arbitrary trajectories as finite-time propagators.
- Separate covariance dimension, tangent contraction, task performance, and
  causal memory. No single proxy proves collapse. For LSTM dynamics include
  hidden and cell state; graph state includes nodes and their ordering.
- Report seeds, independent sampling units, phase alignment, centering,
  normalization, numerical tolerances, controls, and paired uncertainty. Accept
  valid negative findings. Freeze protocols before examining held-out results.
- Synthetic calibration is infrastructure evidence, not empirical evidence
  about trained models. Large Jacobians require matrix-free methods and checks
  against the dense reference on small cases.

## Autonomy and delegation

The user explicitly requests research managed through subagents. Assign bounded
research, implementation, and review tasks with disjoint write scopes, named
artifacts, acceptance checks, and stopping conditions. The coordinator owns
integration and scientific acceptance. Prefer deterministic tools for polling,
hashing, and schema checks; inherit the session model unless a routing policy
and actual tool availability justify an override.

The owner subsequently authorized initial cluv launches and an actual bounded
Qwen cluster research worker on 2026-10-11. Current execution evidence and
resource limits are in the tracker, dated notebook and ignored approval record.
The Qwen leaf worker writes proposals; it cannot submit work or accept scientific
claims. No paid model-service fallback is approved.
Do not import authorization, provider credentials, or GPU-hour limits from
Skill Inference or another project. Carry out local development, literature
research, read-only cluster inspection, and preparation autonomously.

For compute, the user says they can approve Compute Canada work. Prepare the
exact commands, cluster/account, resources, walltime, number of submissions,
outputs, and stopping rule before requesting approval. A recorded envelope
authorizes its subsequent jobs without repeated questions. Until then, no
cluster submissions. Race mode requires approval for every candidate and
configuration and uses the actual connected-cluster set, not a presumed list.

## Storage and provenance

Use cluster scratch for environments, caches, datasets, checkpoints, source
replicas, runs, and logs. Keep local calibration artifacts in ignored `runs/`.
Use `uv.lock`, immutable source revisions, unique run/attempt paths, and atomic
JSON artifacts. Verify hashes and scheduler outcomes before interpreting runs.
Inspect existing project jobs before retries; reconcile ambiguous submissions
instead of submitting duplicates. Do not change other projects' jobs.

Canonical skills live in `skills/`; regular `.agents/skills/*/SKILL.md`
discovery entries point to them. Keep tracked files free of symbolic links so
Overleaf can import the GitHub repository. Do not
write credentials, SSH keys, API tokens, or model endpoint configuration to git.
Keep local approval records in ignored `configs/approval.local.json`.

## Session handoff

Append evidence, commands, results, decisions, blockers, and next actions to
`doc/<month>/<month><day>/YYYY-MM-DD-progress-report.md`, and update the living
tracker. Never rewrite prior reports to match later results. Commit prefixes:
`infra:`, `code:`, `exp:`, `doc:`. Keep source, configs, and lockfiles in git;
data, model weights, caches, and runtime journals stay outside it.
