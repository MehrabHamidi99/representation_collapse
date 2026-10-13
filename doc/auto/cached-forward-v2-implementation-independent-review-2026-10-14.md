# Cached-state forward v2 — independent implementation review

2026-10-14 local project date; review finished 2026-10-13 UTC.

**Verdict: implementation preparation accepted at the exact hashes below.**
Scientific acceptance and production execution authorization remain false.
The coordinator must publish clean source, issue the external admission and
reconcile actual execution evidence before interpreting a production run.
This review ran no native recurrent forwards, read no checkpoint or target
files, loaded no weights and performed no training, provider, cluster or Git
operations. It used static source inspection, pure NumPy checks and the
author's separately authorized saved tiny fixtures.

The immutable acceptance receipt is
`runs/review/cached-forward-v2-independent/implementation-review.json`, SHA-256
`67bfc920d26ffad8221fcb3feb3f6d25946d15b40bbd288364fc32bc9413da84`.
It binds the exact seven-file source map expected by the engine's admission
guard, rather than relying on a verbal approval.

## Contract and source identity

The unchanged [v2 measurement contract](post-calibration-measurement-contract-v2-2026-10-14.md)
has SHA-256 `83c0b8c61fdefa9eb73e8386deab6c073d5ac04a82b13d251e1be699b25124d7`.
Its config remains non-authorizing, SHA-256
`0170170b7724311a3903e5f1cde1917e91263b57fe45ebfc89f6d90b121a1740`.

- Engine: `7bd9bb6b8d05b1ffdc1264492830745f2e019c7e08af817820ba8898c8bbfd04`.
- Launcher: `957f3e4398ad422997256c4aca6a08a9b4fc23c0f09ddf5d39ab8066e434f9f7`.
- Focused tests: `503e1b20fc182e4e0cff3a12c641f69ea87d260b42eed13392b43c196b82447a`.
- Root-owned outer supervisor:
  `193ecd77d4fa762cba20873b40a605e3e98d8bf64b219b6d40eed94b7f4667a2`.

The receipt also binds reporter, smoke driver, pyproject and dependency lock.
The supervisor is separately bound because it lives in the ignored operations
directory and is outside the engine's seven tracked-source entries.

## Independent evidence

The author ran one tiny campaign: eight conditions, 184 attempted and completed
native calls, with an additional independent hook counting the same 184/184.
Internal elapsed time was 1.607 seconds, within the 200-call/120-second fixture
envelope. These use hidden dimension 3, encoder dimension 4, batch 4 and two
independent histories, with GRU/categorical and LSTM/continuous fixtures in both
dtypes and two repetitions. They are infrastructure evidence, not results
about trained models or the full-size production conditions.

Independent code reduced 1,616 saved arrays, 2,948 comparisons and 176,128
entries. It checked 96 C perturbation-effect rows, 256 construction component
rows and 1,188 registry operand/metadata assertions across every saved native
call. Separate pure checks covered 137 numerical boundaries and 19 admission,
counter and clock boundaries. No author comparator was the sole oracle:
saved-array comparisons, displacement reconstruction, stable norm/gain
reductions, margin certificates and continuous bounds were independently
computed. The numerical helper boundary tests use only selected NumPy function
ASTs; no Torch import or forward enters those checks.

The independent reductions verified complete `h,c` ordering, C S0–S17, the
pre-QUERY S16 origin, native terminal states, full-cache strides, every supplied
state/cache before and after, constructed sham/perturbation states, origin
preservation, duplicate rows and exact same-dtype repeat bytes. Requested and
represented displacements are separate; untargeted LSTM components retain
their bytes. B0 intermediate LSTM cell state remains explicitly unobserved.
Registry input/state hashes match the independently saved operands. Source
inspection preserves input seed 30019, direction seed 30029, declared context
order and exact FP32-to-FP64 promotion; this reviewer did not generate the
prospective production streams.

## Findings resolved before acceptance

The implementation now exports requested complete displacements and records
constructed-state versus origin byte identity, including signed-zero changes.
Stable norm reduction distinguishes a genuinely zero represented displacement
from subnormal arithmetic, undefined ratios and overflow. A zero response at
a nonzero perturbation is unresolved at that perturbation, not lost information.

The final native prehook checks time again after hashing and durable journal
bookkeeping; a blocked attempt retains an incomplete entry with no native-start
timestamp. Production has exactly 32 conditions/736 core calls, 64 encoder and
224 head calls. The shared clock begins in the root supervisor before spawning
the launcher; the launcher passes it to the engine. Native starts are prohibited
at 590 seconds, the leaf child is terminated at 598, and the outer supervisor
enforces 600. Post-spawn receipt errors reach cleanup. Late finalization creates
an explicit censor receipt that overrides an earlier completion flag. These
process controls were statically reviewed, not executed by this reviewer.

One concrete manifest defect was repaired: the last head hook increments after
the final native return journal write, so hashing that journal before persisting
the terminal count would make the manifest stale. The engine now persists the
terminal registry before hashing it. The original executed engine and fixture
receipt remain immutable. An independent whole-module AST comparison proved
that removing this one production bookkeeping insertion recovers the executed
source exactly, including every native capture, hook, comparator, input,
direction and admission definition. No second native campaign was needed.
The separately named additive fixture-equivalence receipt, SHA-256
`c017d4932833f94e9163e5243b79c3fdd789f351ee0e6c532057c3799e26e85f`,
is explicitly accepted only for that insertion. The original fixture receipt
remains `6822dc91292fa688ae325a843969e5ecba781b8b83a276ff235d51e59bb11830`.

Mandatory reporting initializes before input/direction generation or native
work. It binds source, seeds, environment and condition identities; raw arrays
and weights are not passed to the reporter. Existing source-bound SDK 0.30.0
offline lifecycle evidence and the two-record transaction stream were checked
by hashes and receipt content. This supplies durable local reporting evidence;
it does not establish dashboard/server delivery. Entity-null offline remains
permitted and explicitly sync-pending.

## Limits and next admission

Admission remains contingent on exact clean committed source, unique outputs,
the reviewed supervisor/invocation, an immutable root freeze and successful
mandatory reporting. Actual outputs require a new independent results review;
complete registry execution and numerical acceptance are separate decisions.
No failed comparison may be hidden or trigger a precision/path fallback.

This stage measures C effects and baseline numerical alignment. It establishes
no B0 counterfactual bridge, task accuracy, competent retention, tangent or
gradient certificate, collapse, periodicity or Floquet result. The original
eight-query R0/R1/R2 gates, independent sampling units and all older failed or
unknown fixed endpoints remain unchanged. Derivative and new-fit budgets are
zero. A plan or this engineering receipt grants no additional execution.
