# Terminal selected-load attempt repair — 2026-10-16

Status: locally implemented and synthetic fixtures passed; independent review
required before integration acceptance. No native checkpoint loading, model,
dataset, SDK, provider, remote operation or experiment was executed for this
repair. The existing native construction/preflight blocks and selected-load
stub remain unchanged.

## Defect and bounded change

The accepted checkpoint-codec independent review identified that
`CausalController.test_once` previously set `test_opened` only after
`backend.load_selected` and its subsequent budget check returned. A loader
exception, partial component mutation, deadline or owner STOP at that boundary
therefore left the controller able to resume training, validation or another
load attempt. The sealed lazy store cannot prevent an earlier load attempt
because its callback has not yet been invoked.

The controller now maintains two distinct flags:

- `test_attempted` starts false and is set immediately after selection
  eligibility and the initial budget check succeed, **before** the selected
  checkpoint loader is called. It is never reset by controller methods. Training,
  validation and further test attempts reject a charged controller before any
  backend or opener call.
- `test_opened` remains false until the controller attempts the `open_test`
  callback. It becomes true immediately before invocation, even if the callback
  raises. This flag does not establish that test payload bytes were read:
  returning a lazy iterator can leave those bytes unopened until its first
  advancement after conditioning replay.

An initial eligibility rejection or initial budget rejection performs no load,
does not invoke the opener, and does not charge either flag. After charging,
any loader exception (including partial mutation), post-load budget/STOP failure,
callback failure or later evaluation failure leaves the attempt terminal.
There is no rollback, automatic retry or recovery of a partially loaded backend.
An eligible successful call still makes exactly one selected-load call and one
opener invocation and follows the existing reset/replay/scoring chronology.

This is a cooperative API lifecycle guarantee. Public Python fields can be
deliberately changed, backend methods can be called directly, and previously
obtained objects cannot be revoked. The repair neither introduces a security
capability nor proves historical selection finalization. Selection eligibility
checks are preserved as they existed; stronger codec/selection and replay-parity
gates remain separate work.

## Verification

Executed once after the source and regression edits:

```text
PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_graph_reconstruction.py -v
Ran 32 tests in 0.384s
OK
```

The meaningful synthetic spy cases cover exceptions both before and after a
simulated component mutation, charging observed inside the loader, walltime and
owner STOP triggered by loading, callback exceptions, and walltime/STOP
triggered by the opener. After failure, budgets are restored where applicable
and training, validation and a second test attempt all reject without another
load, opener invocation, reset, consume, score or optimizer update. Additional
cases establish that missing selection and initial deadline/STOP rejection
remain uncharged, and an initially budget-rejected controller can subsequently
complete its first eligible attempt. Existing successful-call chronology,
selection, counts, native-construction rejection and dependency side-effect
tests also pass. `git diff --check` passed.

Before/after source copies and an immutable local author receipt are under
`runs/review/graph-terminal-load-attempt-20261016/author/`. The receipt records
AST equality for the native backend, native factory, evaluation implementation
and selection implementation, using only stdlib source inspection.

Source SHA256 before:
`89894f8bd3e8574b5b57c777353fe1aac3a1f123f792467a009b04b8a6f9bcdb`.

Source SHA256 after:
`1e8212b9589709d1d4654d0ebfe98179a41ea5a307dbc2083fe84f06c1aead13`.

Focused tests SHA256 after:
`d5d97f595c44bf89e82f87c1806f24db050d28bad2a910adaba2cc0d326adc71`.

These are engineering fixture results, not checkpoint parity or scientific
evidence about representation collapse. Historical reports and source pins
remain historical; future schema/runner source bindings must use the repaired
source hash after independent acceptance. No commit was made by this author.
