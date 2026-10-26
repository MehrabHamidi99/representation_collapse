# Adopt the template

## New project

1. Run `python3 scripts/template.py create ../study-name --name study-name` from
   the template. The destination must be empty. It copies portable files and
   registers a relative link to this template; it does not initialize or push Git.
2. In the new project, fill `project.json` and its canonical protocol/tracker paths.
   Set actual ownership, timezone, cluster scratch roots and the intended question.
   Set the relevant W&B project/subproject for qualifying monitored scientific jobs;
   follow [results reporting](results-reporting.md).
3. Run `python3 scripts/research.py demo`. This standard-library calibration works
   offline and produces ignored evidence under `runs/template-demo/`.
4. Put scientific code in `src/`, commands in `scripts/`, configs in `configs/`.
   Choose and lock project dependencies. Keep the cluster operations environment
   separate when inherited code requires an incompatible Python/library stack.
5. Initialize Git and commit source/configs/lockfile. Pin that full commit in a
   copied `configs/plan.example.json`; replace the protocol hash with the real hash.
   Run `python3 scripts/research.py validate --plan configs/plan.json`.
   Use the project remote as the primary [local/cluster sync route](source-sync.md).
6. Use the auto-research skill to calibrate, reproduce a baseline, then prepare a
   measured pilot. Record any compute authorization once, with exact scope and
   stop conditions; subsequent work within that scope does not need repeated approval.

For a standalone GitHub template checkout, use this repository as the control
plane or generate the actual project with the command above. The generated project
is the lightweight starter; `sources/` and the maintainer's registry are not copied.

## Existing project

Use `adopt --project PATH` first. It adds `.research-template.json` and
`RESEARCH_TEMPLATE.md`. Existing byte-identical shared files are registered for
contributions; different files are preserved. With `--install`, only missing
portable skills and guides are added. Read the returned preserved-file list before
deciding whether to merge a project-specific variant manually.

Keep independent nested repositories independent. Do not move legacy manuscripts,
datasets, environments, code forks or dated notes. A local folder without Git can
still link its documents; creating a repository is a separate choice.

## Portability and updates

The generated project can run its local helper without the template present.
The relative `.research-template.json` link is needed only for sync/update.
If repositories move, update that link and the template registry path together.
Review `pull --project PATH` before using `--apply`; tracked bases prevent silent
overwriting of local edits. Existing generated scientific configs and AGENTS.md
are project-owned and are never overwritten by the shared-content updater.

Run the helper with Python 3.11+ on macOS or Linux. macOS launchd installation is
optional outside the maintainer workspace; Linux can run `sync` from an existing
timer. Do not put a persistent service on a cluster login host.
