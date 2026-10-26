# Independent legacy array audit implementation review

Reviewed 2026-10-14 against the frozen v1 protocol and configuration. The initial
implementation requires two repairs before engineering acceptance. This review
does not authorize production execution or accept a scientific finding.

The protocol SHA-256 is
`9e201ec67ac2906abebbac38e4ad5f989e12db0bb583dd518cbb32dd1a4b4493`;
the configuration SHA-256 is
`a078ec7306fb88cf83b21758e307490e5e025005212bd93f28b92272b83c5409`.
Initial reviewed module, launcher and test hashes are respectively
`89c210486ae2646a22b55105d7635e57f615f36d51d160cba73171443ba60d92`,
`0bd407d0bae24140c3a6fa62c0802eade5fbd2e2b6710471c2b7e14587c8ef6c`,
and `4cc3b2c7482b5a17d0cc24493967155e87270a6f72095dca2d0e819c0f1dd6c4`.
The durable initial receipt is
`runs/review/legacy-array-audit-independent-20261014/review.json`.

The first blocker is the time-row repeat indicator in
`src/representation_collapse/legacy_floquet_audit.py:260`. The frozen protocol
requires byte-repeat indicators; `np.array_equal` compares numerical values.
A width-one complex64 adversarial fixture changes only a zero sign bit. Its
row bytes differ, while the implementation reports exact repetition and zero
absolute difference. The adjacent-prefix `exactly_equal` field similarly
reports true despite different bytes. A repair should preserve chunked byte
comparison for the primary time-row diagnostic and distinguish numerical
from byte equality explicitly. The zero numerical difference remains valid.

The second blocker is descendant cleanup in
`code/ops/legacy_floquet_audit.py:78`. The owning supervisor sends SIGKILL only
to the worker process group on timeout. The installed pinned W&B SDK launches
its service in a separate session (`sdk/lib/service/service_process.py:116,154`).
That service is outside the worker's group. The supervisor neither records an
owned descendant inventory nor reconciles descendants after worker failure or
completion. The repair must own and verify those descendants, including the
SDK service, and terminate/reconcile them before reporting the bounded attempt
complete. No SDK campaign was run to obtain this static evidence.

One independent focused campaign passed all 15 tiny tests. Numerical threads
were fixed to one; fixture width was at most eight. The focused campaign and
width-one signed-zero checks completed in 0.224 seconds and are preserved in
`focused-tests.log` and `adversarial.log` beneath the review directory. No
whole-suite run, archive payload, checkpoint, dataset, model, provider,
network or cluster operation was used.

The remaining static array contracts match the protocol: the fixed forty-file
schema, bounded numeric headers, object/structured-array refusal, exact payload
length, regular-file and link checks, descriptor-backed read-only mappings,
streamed hashes, and before/after/final stat comparisons. Numerical reductions
use chunks of at most 1,024 state coordinates. Complex components are retained
in eigenvalue phases, Hermitian Gram products and state-axis norms. Norm
comparisons use the saved float32 norm as the directed reference with the frozen
absolute and relative tolerances. All nineteen prefixes, failed norm edges and
adjacent comparisons remain in the result. The pinned offline reporter is
entered before audited-array opening or hashing. Scientific acceptance is
always false; full-Jacobian eigenpair residuals and closure remain unavailable.

Production admission still requires the reviewed files to be committed and
byte-equal to the exact published HEAD in a clean isolated worktree. The worker's
tracked-only Git status check excludes untracked files, so that check alone
does not establish that its imported source belongs to HEAD. External exact
source/configuration/manifest/lock admission and manifest freeze must precede
the single local CPU attempt. Independent result reconciliation remains a
separate later step.

At root's request, acquisition code was inspected as text only. Its whitelist
and 468,187,088-byte plan contain forty numerical result files and no numerical
processing, Jacobians, data or weights. Its final trailing-data `os.read` can
block without the declared 300-second deadline; root was notified to replace
it with a deadline-guarded read or an owning outer watchdog before launch.

Root corrected the acquisition EOF guard before launch. A subsequent static
check confirms `select(..., remaining())` now bounds the final read. This
acquisition observation is resolved; the two implementation blockers above
remain pending an exact repair packet.

The first exact repair packet was statically rechecked without rerunning tests.
Module SHA-256 `edfa4336040d5f54856d474d40b8b8eeb06e0089ad0af2c442dba1b7efe3db6a`,
launcher `c51c7ff53f97f1b3a80dcd51374c14563f0037a8f6d3e8fe120a16ccca626a72`,
and tests `70faa3ba5b0951db1135d9add100540620887a9065657342b48ee0bbb7e3745a`
match the author's frozen packet. Its hash-bound 18-test log includes byte
identity, stale-identity signal refusal, and a detached orphan cleanup fixture
that preserves an unrelated process. The byte-repeat blocker is resolved.

Process ownership is substantially repaired: the worker installs a spawn
journal before SDK import, detached child identities survive parent exit, and
cleanup sends signals to verified owned identities. A remaining failure-path
blocker prevents engineering acceptance of this packet. The final cleanup
snapshot's exception is discarded at launcher lines 174–177, and cleanup
returns after SIGKILL without reconciling descendant identities. The caller
waits only for the worker. An unknown snapshot or surviving descendant can
therefore coexist with a completed supervisor receipt. The requested final
repair retains snapshot errors and uses bounded postkill reconciliation, with
unknown or live owned identities marked incomplete. This intermediate review
is in `runs/review/legacy-array-audit-independent-20261014/repair-review.json`.

The final cleanup packet resolves the remaining blocker. Final snapshot
exceptions are retained, each recorded PID/start identity is reconciled after
signaling within 250 milliseconds and the outer deadline, and unknown or live
identities force an incomplete supervisor receipt. Darwin snapshots inspect
the owned parent chain rather than unrelated processes. Per-signal deadline
checks and the elapsed-cap override preserve late/incomplete status.

**Final engineering acceptance: accepted for the exact reviewed implementation.**
The final module remains
`edfa4336040d5f54856d474d40b8b8eeb06e0089ad0af2c442dba1b7efe3db6a`;
launcher SHA-256 is
`8e1f388d4f5ce72e0bfd9a4b5a4b1f8c98f66f2512841d96875822bfd9bffe09`;
tests SHA-256 is
`e621d20e1c4f6a75d91d85217fcb11633928a3ca83b58d8e996d0072fe21ab02`.
All source pins and the final handoff were reverified. The author's four
targeted repair tests pass in 0.334 seconds with log SHA-256
`c2d3873cf07d47de5289a6a9b77076113cf9bc1fddfb50394a5f080f8a9222db`.
The injected snapshot-failure/live-identity test checks the previously missing
failure path; the detached-process test retains unrelated-process protection.
Earlier failed attempts and review receipts remain unchanged. This closeout
used static checks and existing test logs; no additional campaign ran.

The final decision is saved separately in
`runs/review/legacy-array-audit-independent-20261014/final-review.json`.
Acceptance covers engineering only. Source publication, the clean isolated
worktree, exact external admission and frozen manifest must still precede the
one CPU attempt. No numerical finding is accepted here; full-Jacobian residuals,
orbit closure, task behavior, causal memory and representation collapse remain
unavailable, and scientific acceptance stays false.
