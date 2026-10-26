# Original primary T-GRAB payload staged on Mila

At **2026-10-15 06:30:09 UTC**, native cluv access to Mila staged and verified
the exact public primary periodicity CSV. No dataset payload was downloaded
to the Mac, parsed, converted or used for training.

The repository is `Gilestel/T-GRAB`, frozen revision
`603388f332c7756f90bc1702b26053c2bc834654`. The primary file is
`periodicity/periodicity_det_(2, 32)_100N_40tr_4vl_4ts_pER(0.01).csv`:
**6,907,326 bytes**, SHA-256
`5a19f9e7949668473bc61cb59b6215f25f093ea0ee2d45b5e143338ded5523e3`.
Both size and hash match the LFS metadata already present in the frozen
October 12 tree. The metadata was available earlier; the payload had not been
verified. This closes that acquisition gap without claiming schema or split
semantics from the filename.

Scratch path:
`/network/scratch/m/mehrab.hamidi/representation-collapse/datasets/tgrab-603388f332c7756f90bc1702b26053c2bc834654-primary/primary.csv`.
The adjacent `payload-receipt.json` records source revision, original path,
byte/hash checks and staging time. The local receipt is ignored
`runs/ops/native-graph-reporting-fix-20261015/primary-dataset-staging-result.json`;
it contains metadata only. Source metadata/tree hashes are respectively
`c6b519445defa651d88e4ff5247498e13d96d6830d3bf51cd184954cc4421e75`
and `93fbd4b15f026a1f6821ca277ab24eee0aea9c7b9cec8fa27d0501756ae71225`.

The bounded streamed download and hashing took 12.483 seconds. Whole-project
scratch before staging was 9,155,355,648 bytes, with reservation for the payload
and temporary copy below the existing 20 GiB staging limit. No package install,
SDK initialization, numerical/model evaluation or scheduler submission occurred.
The original public bytes and receipt are preserved; this operation does not
grant a compute allocation.

The next step is a reviewed CSV-schema/conversion contract and a sealed split
store. Verify timestamps, node IDs, implicit weights/markers, ordered edges,
two actual distinct patterns and train/validation/test boundaries against the
pinned generator and reconstruction protocol. Do not infer these facts from
matching file hashes, regenerate a substitute dataset, or expose test labels
to training/selection. The [training-runner contract](tgrab-training-runner-implementation-contract-2026-10-15.md)
remains prospective; this receipt supersedes only its earlier pending-payload
status. Conversion, trainer, checkpoint codec/load parity and competence remain
unfinished. No benchmark reproduction or collapse claim is accepted.
