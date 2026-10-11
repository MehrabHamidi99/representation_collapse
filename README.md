# Representation Collapse

Floquet and finite-time analysis of dimension collapse in recurrent models,
across temporal graphs, time series, vision, audio, and text. We start by
reconstructing the existing temporal graph work, validate the measurements,
then test whether the same mechanisms extend across modalities.

Agents start at [AGENTS.md](AGENTS.md). Research state is in
[doc/TRACKER.md](doc/TRACKER.md); the dated notebook follows
[doc/README.md](doc/README.md). The [research program](doc/auto/research-program.md)
and [graph reproduction protocol](doc/auto/graph-reproduction.md) distinguish
inherited claims from new evidence. Original manuscript directories are
preserved; [source inventory](doc/auto/source-inventory.md) records their role.

## Local measurement checks

```bash
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run python -m representation_collapse.calibrate --output runs/calibration/local
```

The NumPy reference implementation checks covariance participation ratio and
entropy rank, ordered Jacobian products, Floquet multipliers on known periodic
systems, nonnormal transient gain, and stable periodically forced responses.
Output contains provenance, metrics, verification, arrays, and SHA-256 hashes.
Choose a new output directory for every attempt. This calibrates analysis code;
it does not reproduce a trained graph model.

## Research and compute

Canonical operating skills are in `skills/`, discoverable through project
`.agents/skills/` symlinks. The [agent protocol](doc/auto/agent-protocol.md)
defines bounded subagent handoffs; [research-plan.json](configs/research-plan.json)
is a declarative dependency plan. An unattended controller is not installed.

```bash
uv sync --locked --group ops
uv run --group ops cluv --help
uv run --group ops python code/ops/preflight.py
```

Follow the [compute plan](doc/auto/compute-plan.md) for Mila, Tamia, and Alliance
clusters. The installed cluv version is locked. Its native configuration is in
`pyproject.toml`; [example profiles](configs/cluv.example.toml) show expansion
to additional clusters. Cluster jobs require a concrete approved resource
envelope; this setup has submitted none. Graph code/data leads from the
manuscripts must be pinned and audited before training or reproduction.
