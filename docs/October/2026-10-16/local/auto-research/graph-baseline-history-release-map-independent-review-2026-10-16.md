# Independent review — graph baseline/history release map

**Disposition: revise one release dependency before accepting v1.** The map is scientifically well scoped and accurately distinguishes the original T-GRAB baseline, native parity, vector Gate B, and a later graph required-history study. Its reproduction-versus-reconstruction boundary and claim limits agree with the cited audit. One operational dependency needs to be made explicit before this serves as a release map: whether the prospective graph fit in node 1 is blocked on node 2's checkpoint/native parity gate.

## Concrete finding

**BLOCKER — node 1 does not state its dependency on native checkpoint parity.** Node 1 includes an independent next-graph F1 reproduction/reconstruction and its acceptance criteria include a checkpoint identity and selected-checkpoint evidence (lines 28–46). Node 2 defines the selected-checkpoint load/replay/loss-gradient parity gate. Node 4 explicitly requires node 2 before interpreting trained graph results, but node 1 does not say whether fitting, checkpoint selection, or held-out evaluation is blocked on node 2. The current release-order paragraph says the immediate work is local and not an experiment submission, which prevents an immediate launch, but it does not resolve the future dependency.

This matters because the Gate B readiness review explicitly says to require trusted loading, full-state chronology, and no test exposure **before releasing a graph fit** (review §“Three changes to freeze before a training request”). The graph reproduction protocol separately requires a validation-selected checkpoint, test data closed until selection, and correct chronology. The release map should therefore either (a) make node 2 a prerequisite for any new node-1 fit/selection/held-out evaluation, or (b) define the exact bounded node-1 work allowed before parity and state what checkpoint-selection/test-access protections it uses. Avoid leaving readers to infer that baseline F1 is exempt from the release gate.

A related sequencing detail should be resolved in the same sentence if the intended order is node 2 first: node 2's object is a “selected checkpoint,” while node 1 appears to be the source of a new selected checkpoint when original lineage is absent. Identify the fixture/checkpoint provenance that allows parity to be completed without silently launching the graph fit that parity is meant to gate, or state the permitted bootstrap path. This is a dependency clarification, not a request to weaken parity.

## Checks that passed

- The map correctly describes the TSA result as a bounded search and does not claim global absence. The audit says no recurrent/link-head checkpoint was found in the enumerated scope and no original run/data/checkpoint identity is linked to the arrays.
- It correctly treats new fitting without verified original lineage as reconstruction, not exact reproduction; similar metrics cannot establish exact reproduction.
- It keeps AB (`Pdet(2,1)`), AABB (`Pdet(2,2)`), and long-repetition `Pdet(2,32)` behavior separate, consistent with the reproduction audit.
- It does not treat next-graph F1 as arbitrary-prefix retention, causal memory, covariance collapse, tangent contraction, or Floquet evidence.
- It carries forward the Gate B assigned-order estimand and its motif-exposure limitation, with valid negative outcomes and separate discovery/confirmation inference.
- The graph-history node requires task dependence on earlier history, competence, complete hidden/cell state across nodes, causal controls, and periodic trajectory/Jacobian closure before Floquet terminology. These requirements are consistent with the active goal and graph protocol.
- The proposed next **local** task (typed codec/transaction semantics and adversarial fixtures) is not blocked by the missing cluster allocation. The map correctly gives no compute or execution authorization. No new numerical run or submission is needed to resolve this review finding.

## Suggestions, not blockers

1. In node 2, name the source and intended role of its first parity fixture (e.g. deterministic initialized fixture versus a selected trained checkpoint), and distinguish fixture-level codec correctness from the stronger selected-checkpoint/native parity result. This makes the bootstrap dependency auditable.
2. In node 1, specify whether “independent reproduction” means a prospective reconstruction's own replication across seeds or a comparison against inherited published summaries; the text currently permits a useful interpretation but could be more explicit.

## Verdict and limitations

**Revise** the node-1/node-2 dependency and bootstrap statement; the rest of the release map is acceptable within its stated scope. This is a document-only review. I did not execute code, inspect datasets/checkpoints, access a cluster, or verify any runtime behavior. No model competence, original graph reproduction, collapse, Floquet behavior, or scientific acceptance is established by this review.

## Reviewed identities

Snapshot: 2026-10-16 (workspace review).

- `docs/October/2026-10-26/local/auto-research/graph-baseline-history-release-map-v1.md`: `c6105e63009f88e2a4c9100f4baa7f51371dfb6ee97195ec98939fc77b700f61`
- `docs/October/2026-10-16/local/auto-research/gate-b-readiness-independent-review-2026-10-16.md`: `a612a029a71ce3db7811a39485c98cbfaff102b45c52a1a26ddb4df18fe540c7`
- `docs/October/2026-10-14/local/auto-research/tgrab-reproduction-audit-2026-10-14.md`: `96dc51d23d3345ab314649760e8a6337a6eb2d5c3e25427a01bb31b1ddc06741`
- `docs/October/2026-10-26/local/auto-research/graph-reproduction.md`: `2daed7958e8429ff4459f0fc9ea5e29f94997bb26a0b62124352fc24cbdff727`
- `docs/October/2026-10-26/local/auto-research/experimental-design-v2.md`: `65451640b7ecf38e81307c576a16bc1ad488b7a4bfec9fcb5de387a5ced453e5`
- `docs/October/2026-10-26/local/auto-research/active-research-goal.md`: `0629ac304d0f47e9eca8bad8b75e01188ec8ab445ff413019fa48a89846f60b5`
- `doc/auto/research-status-2026-10-16.json`: `299034eebd652fad16447fd7bcc64a3d243bb3726cc80995e759c278fa7ee087`

## Revision addendum — 2026-10-16

**Follow-up disposition: RELEASE-001 is resolved for fresh reconstructions; one exact-path wording clarification remains.** Reviewed the revised map at SHA-256 `691a68a8940c9fe4e56159126f18f978a4ce961d715c300d81b792b3301ecdff`.

The revision now makes the prospective ordering explicit: source/data/protocol audits and provenance searches may happen first; any fresh fit and validation-only selection wait for node2a; node2a is a deterministic synthetic initialized-state parity fixture, explicitly without training, primary-data access, or held-out targets; and the actual selected checkpoint must pass node2b before any test targets are opened. It also says that a fresh reconstruction needs a separately frozen multi-seed replication plan, so the prior replication suggestion is addressed. The node2a role is now explicit and does not substitute for trained-checkpoint validation.

The exact-reproduction exception is appropriately narrow: skipping a fresh fit requires recovery of the original checkpoint **and its run lineage**, followed by node2b. This satisfies the requested provenance condition. However, node2b is described in its own paragraph as applying “after a separately authorized fit and validation-only selection.” That sentence does not explicitly include the recovered original selected checkpoint. Add a short clause stating node2b validates either (i) the newly selected checkpoint after a fresh fit or (ii) the recovered original selected checkpoint when exact-run lineage is verified. This is a remaining wording clarification for a complete release map; it does not reopen the fresh-fit ordering finding.

No code, data, checkpoint, compute, or model operations were performed. The original review and its prior source hashes above are preserved unchanged; this addendum records the new map hash and limited follow-up scope only.

## Revision addendum 2 — 2026-10-16

**Follow-up disposition: the remaining node2b wording item is resolved.** The current map SHA-256 is `ef9dde2e079f362e77993f6194f69220793707017e6538d5af581307876fbb91`. Node2b now explicitly covers either a separately authorized fresh fit followed by validation-only selection, or recovery of the original checkpoint with verified run lineage. In both cases it validates that exact immutable selected candidate through the native path, and test targets remain closed until node2b passes. This resolves the exact-reproduction-path ambiguity while preserving the requirement that original checkpoint and run lineage be recovered before skipping a fresh fit.

Together with the prior addendum, the revised map now clearly defines node2a's deterministic synthetic pre-fit bootstrap, gates new fit/validation selection on node2a, requires node2b before test access, and requires a separately frozen multi-seed replication plan for fresh reconstruction. **The reviewed release-order items are accepted.** The original finding and hashes are retained above as historical review record; this addendum records the current resolution only. No code, data, checkpoint, compute, or model operations were performed.
