---
name: research-review
description: Review research claims, proofs, experimental controls and result provenance before accepting evidence or updating a manuscript.
---

# Research review

Inspect the frozen question/protocol, actual source and artifacts. Start with the
claim that matters and the strongest plausible alternative explanation. Distinguish
theoretical guarantees, calibration evidence, observational results and causal
conclusions. Test the assumptions needed for each inference.

For proofs, check definitions, quantifiers, boundary/degenerate cases and the
deterministic-to-stochastic interface. For experiments, check baseline equivalence,
independent sampling units, seed coverage, selection leakage, uncertainty, missing
outcomes, numerical error and all failed/censored attempts. Do not accept a metric
as a proxy for an unmeasured scientific quantity without an argument.

Trace each figure and numerical claim to immutable inputs, aggregation code and
outputs. Compare realized execution with the protocol, including batch/loss/data
semantics after performance changes. A reviewer who cannot inspect artifacts must
state that limitation and cannot approve them from a summary alone.

Check that interpretable results from monitored scientific jobs, including baselines,
comparisons and evaluations, are reported in the relevant W&B project/subproject with enough
context to interpret them. Preprocessing/progress-only tasks are not applicable.
Verify links and values; record pending reporting
separately from scientific acceptance. See
[results reporting](../../docs/October/2026-10-26/cross-cluster/guidelines/results-reporting.md).

Write a concise decision: accepted, rejected or unresolved; evidence inspected;
claim boundary; material issues; and a minimal discriminating next step. Negative
findings can be accepted. Never treat an eloquent model response as verification.
If delegation is authorized, review a pinned version in a separate read-only task;
otherwise use explicit checks and disclose the review's independence limits.

Follow [documentation, reporting scope and Git sync](../../docs/October/2026-10-26/cross-cluster/guidelines/documentation-layout.md)
for dated records and W&B eligibility.
