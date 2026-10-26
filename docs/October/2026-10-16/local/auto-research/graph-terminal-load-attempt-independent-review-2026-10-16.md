# Independent review of terminal selected-load charging

2026-10-16. **Accepted for this bounded controller repair.** The change closes
the pre-load terminality gap identified in the
[checkpoint-design review](tgrab-selected-checkpoint-codec-independent-review-2026-10-16.md).
It does not implement a checkpoint loader, selected-state replay parity or a
runnable training experiment. No blocking finding remains within this scope.

Reviewed source `src/representation_collapse/graph_reconstruction.py`: SHA256
`1e8212b9589709d1d4654d0ebfe98179a41ea5a307dbc2083fe84f06c1aead13`.
Reviewed tests `tests/test_graph_reconstruction.py`: SHA256
`d5d97f595c44bf89e82f87c1806f24db050d28bad2a910adaba2cc0d326adc71`.
The [author report](graph-terminal-load-attempt-repair-2026-10-16.md) hashes to
`b544c6f957b975fb48dc036bc20bdabe3f74f0aea53f4abdce89935e14d87489`.
The retained pre-repair source equals the previously accepted
`89894f8bd3e8574b5b57c777353fe1aac3a1f123f792467a009b04b8a6f9bcdb`.

## Ordering and failure behavior

The actual method now checks eligibility and the initial budget first, sets
`test_attempted=True`, and only then calls `backend.load_selected`. No controller
method resets that flag. Training, validation and another test call reject a
charged controller before any backend or opener work. This applies to loader
exceptions, partial backend mutation, post-load deadline/STOP rejection,
callback failure and later replay/evaluation exceptions. Loader exceptions
continue to propagate; partial mutation is retained as a terminal failure,
without rollback or an automatic retry.

`test_opened` has a narrower meaning: it becomes true immediately before the
`open_test` callback is attempted. A callback failure leaves it true; a load or
post-load boundary failure leaves it false while `test_attempted` remains true.
A returned lazy iterator may still have read no test bytes. The existing replay
then occurs before the first target iteration; charging a controller does not
itself read the test payload or alter recurrent chronology.

An initial eligibility or initial budget rejection leaves both flags false,
with no load or opener invocation. A later first eligible call is therefore
permitted. Synthetic budget restoration in the tests demonstrates flag
semantics only; it does not renew a real deadline, remove owner STOP authority
or authorize another allocation.

## Independent verification

The reviewer inspected the exact source diff and replayed:

```text
PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -p test_graph_reconstruction.py -v
Ran 32 tests in 0.144s
OK
```

Source, tests, report and frozen reconstruction-config hashes were identical
immediately before and after execution. The suite uses synthetic arrays and
spies, including exceptions before/after simulated component mutation, charging
observed inside the loader, load-triggered walltime/STOP, callback exceptions,
callback-triggered walltime/STOP and uncharged initial rejections. After failures,
restored synthetic budgets still cannot resume training, validation or retesting;
backend event logs and update/epoch counters remain unchanged by those rejected
calls. Successful load, reset, conditioning and score chronology still passes.

Independent Python3.10 AST comparison proves the complete native backend,
native factory, preflight, checkpoint reference, selection, budget, complete-state
layout and `_consume`/`_evaluate` implementations unchanged. Native selected
loading still raises, preflight still unconditionally records the missing native
checkpoint/reporting runner, and the frozen launch-disabled reconstruction config
retains SHA256 `22bef6bf781c5bfb5493dd5e03169ea0e658b2463ccaa9ba09189af454508f1f`.
No native import/construction or checkpoint operation was exercised.

The static inspection, complete replay log, before/after hashes and immutable
receipt are retained under
`runs/review/graph-terminal-load-attempt-20261016/independent/`.

## Limits and remaining work

This is a cooperative public-controller guarantee. Deliberate flag mutation,
direct backend/private-method calls and previously retained objects remain
outside it. Existing selection eligibility is preserved; this repair does not
authenticate historical finalization, durable selection receipts or checkpoint
integrity. It also adds no validation-state observer or pre-first-target
complete-state parity gate. The future trainer and codec must still implement
and independently verify those controls using this repaired source identity.

Historical source pins/reports continue to describe their original bytes;
future executable bindings must explicitly use the repaired hash after this
acceptance. No target dataset, model, SDK, provider, remote operation or cluster
job was used in this review. Trained competence and representation-collapse
claims remain unverified.
