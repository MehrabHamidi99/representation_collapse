# Reporting correction within the approved single CPU allocation

The owner replied **“submit and continue doing auto research”** to the prepared
one-job Mila v2 calibration request. This authorizes one CPU, 2 GiB, ten minutes,
zero GPUs, account `mila`, QOS `normal`, partition `main-cpu`, without retry,
requeue or race. It grants no training or follow-on allocation.

Before any submission, a native review agent found a second integration defect
in source `e4ab463`: the v2 worker supplied `module_path`, `config_path` and
`protocol_path` as reporting identities, but the actual reporter constructor
requires every identity value to be a lowercase SHA-1/SHA-256 digest. Those
paths would fail before SDK initialization or model execution. The earlier
dependency-free spies and static review missed that actual constructor contract.

The old approved plan was staged but never submitted. At 06:17:32 UTC,
read-only queue inspection and absence of submission claim, admission and output
verified that it consumed no allocation. Its source and plan are preserved;
an additive supersession record and STOP now mark it unusable. This is a
pre-submission repair using the **same single approved allocation**.

The narrow fix moves those three public path strings into compact reporting
configuration metadata, preserving eight hash-valued identities. A new test
exercises the actual reporter constructor without entering it or initializing
the SDK, and reproduces the old invalid-identity failure. Numerical functions,
fixtures, dependency versions, tolerances, counts, protocol and physical
supervision are unchanged. The engine SHA is
`eecb96df752fbb5842685cc423e67494758785792302d7c46fa24e2642741156`.
The additive independent review binds this corrected source; previous reviews
and the old prospective verifier remain preserved, with their limitations.

The launch descriptor now binds this engine and fresh scratch paths:

- Source: `sources/native-gclstm-calibration-v2-reportingfix-20261015-attempt001`
- Run: `runs/native-gclstm-calibration-v2-reportingfix-20261015`
- Logs: `logs/native-gclstm-calibration-v2-reportingfix-20261015`

All are under `/network/scratch/m/mehrab.hamidi/representation-collapse`.
The already verified legacy CPU environment is reused without installation.
The exact published commit and operational plan are bound in the runtime
approval and source-staging receipts before native cluv submission. Automatic
root-environment sync is excluded. The reviewed submission controller and job
materializer are unchanged; only descriptor-bound source/path identities vary.

The immutable v2 scientific protocol and initial allocation request remain
historical preparation records. This additive document records their operational
continuation; their earlier “not authorized” status is not a current refusal
to execute the owner's subsequent approval. The disabled configuration itself
does not confer authority: the exact runtime owner record, one-submission
claim and physical scheduler admission do.

After execution, require actual scheduler outcome, exact native counts,
source/class/environment provenance, owned cleanup, offline W&B transaction
verification and independent saved-array reconciliation. Corrected source
publication and passing tests alone establish no native result. No trained
graph competence, benchmark reconstruction, periodicity effect, Floquet
certification or harmful-collapse claim follows from this engineering assay.
