# Native graph CPU environment preparation — 2026-10-15

This handoff prepares a locked, isolated legacy environment for a later CPU calibration. The environment has **not been installed by this task**, and native imports, numerical parity, dataset provenance, trained-model reconstruction and scientific acceptance remain unavailable. The main project package, root `pyproject.toml`, root `uv.lock`, and main `.venv` were not changed.

The target is CPython 3.10.19, Linux x86_64, glibc at least 2.28, CPU only. The coordinator selected Python 3.10 as an explicit compatibility deviation from the main project's Python >=3.11 requirement. Use the standalone reviewed source through an explicit `PYTHONPATH`; do not install the main project into this environment or claim its Python requirement has been met.

## Source fidelity and packaging exceptions

The locked native versions are Torch `2.0.0+cpu` (public version 2.0.0), PyG 2.6.1, Temporal 0.54.0, NumPy 1.24.4, torch-cluster `1.6.3+pt20cpu`, torch-scatter `2.1.2+pt20cpu`, torch-sparse `0.6.18+pt20cpu`, and torch-spline-conv `1.2.2+pt20cpu`. CPU local version suffixes identify the exact official artifacts and are retained in installation checks. The native Temporal cell SHA-256 is `c3568d2c1ba8926ab43e11c840213f90ebe782ad158dd76809d1dc1c255797af`, with source commit reference `46abd39abec0cd8de23c0e489a617c320c3fc5e8`.

Literal all-wheel resolution fails for Temporal 0.54.0: [official PyPI release metadata](https://pypi.org/pypi/torch-geometric-temporal/0.54.0/json) lists one 48,092-byte source distribution and no wheel. Its setup requires `pandas<=1.3.5`; [pandas 1.3.5 release metadata](https://pypi.org/pypi/pandas/1.3.5/json) has CPython 3.10 Linux wheels and no CPython 3.11 wheel. The exact Torch and four extension artifacts are available for both interpreters, so the pandas restriction is the recorded Python 3.11 binary-only blocker, not an invented replacement cell.

The coordinator authorized inspecting and packaging only the small pure Python Temporal archive offline within the existing project staging allowance. The source archive SHA-256 is `8e9f94751a6656f49d68c3dc32ae4067221ad886e66649897b859b7770a522d6`. All 70 archive entries are regular files/directories, with 332,794 uncompressed bytes. No C/C++/CUDA/pyx/shared binary or custom build command was found. The setup AST imports setuptools and invokes `find_packages`/`setup`; the inspected source files were hash checked before and after the build. The native cell is byte identical to the frozen cell. This binds the release archive and the cell; it does not claim that every release source file was independently compared with the referenced Git commit.

The offline build used existing Apple Python 3.9.6, setuptools 58.0.4 and wheel 0.37.0. No packages were installed, no native modules were imported, no source was edited, and no compiler/network operation ran during the build. The derived `torch_geometric_temporal-0.54.0-py3-none-any.whl` is 86,746 bytes, SHA-256 `fb5f0bc2c9ef9e7a23a82d53b07490bc8a99adb5e55120559fe5412a8ca70893`; its source cell and dependency metadata were verified. The build receipt is `runs/review/native-graph-environment-20261015/temporal-derived-wheel-attempt-001/receipt.json`. This is the sole packaging exception. Every compiled package must remain an immutable prebuilt wheel.

The [official PyG CPU index](https://data.pyg.org/whl/torch-2.0.0+cpu.html) supplies the four CP310 extension wheels, consistent with [PyG's installation guidance](https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html). Their fully received SHA-256 values are:

- torch-cluster: `c97e7a2f3796890e095b214b0850ebbc5acb40643f03ca700774312b322580ea`, 751,305 bytes.
- torch-scatter: `935a09ea195a6f439816eda886674b43b971adfdf92890a3ebd238673efb49ea`, 494,117 bytes.
- torch-sparse: `45be5c7afa376aba91d89eac13d89fa494d0d49d80cf0ac51f8df7481da21563`, 1,200,431 bytes.
- torch-spline-conv: `da5cd9b071814585de99198ecf4b27087da4adbe451c421bba949801c62e5a66`, 208,100 bytes.

The index does not publish hashes for these four wheels. These are measured hashes from the authorized 2,653,953-byte download, with ZIP path/type, CRC and member hash inspection. No binary was installed, imported or executed. Full URLs and member evidence are in `extension-wheel-inputs-attempt-001/receipt.json`.

Torch's 195,412,802-byte CPU wheel was not downloaded locally. Its official index advertises SHA-256 `7c9e668462cd0813fc4baaeaa137efce2516e73cf381120846edf45d4ec21174`. Only HTTP range ZIP directory/metadata reads totaling 1,275,988 bytes were made; CRC and metadata hashes were verified. Full wheel verification is required during staging. The [official CPU index](https://download.pytorch.org/whl/cpu/torch/) and `torch-metadata-range-receipt.json` bind this distinction.

## Reporting additions and the lock

W&B 0.30.0 is a mandatory reporting addition, not a recovered historical native dependency. Its exact [release metadata](https://pypi.org/pypi/wandb/0.30.0/json) requires Python >=3.10 and the selected Linux wheel requires glibc >=2.28. The source-native eight versions are retained unchanged. Decorator 4.4.2 and pandas 1.3.5 satisfy Temporal's literal metadata; NetworkX 3.0 is explicitly locked for the legacy target.

The target wheel closure has 60 artifacts totaling 314,764,667 bytes. Official release JSON and wheel metadata were resolved with target Python/ABI/platform markers and requested extras. All active version edges were checked. No package payload beyond the approved small Temporal archive/four extension wheels was downloaded locally. `wheel-manifest.json` records URLs, exact bytes, wheel and metadata hashes, local versus official provenance, body-verification status, and active dependency edges. `requirements.lock` pins every distribution and hash.

The main reporting lock cannot be copied literally into Python 3.10: platformdirs 4.13.0 requires Python >=3.11. The isolated closure instead locks platformdirs 4.12.4, and separately resolved other compatible reporting transitives, including pydantic 2.14.1 and pydantic-core 2.50.1. The manifest retains the main reporting closure and marks target-specific transitive resolution explicitly. SDK 0.30.0 is unchanged. This lock is dependency engineering evidence, not evidence that the native extensions load correctly with every transitive package.

Metadata preparation failures remain saved: a denied default network attempt; an initial target conflict from retaining platformdirs 4.13.0; an invalid historical PyPI version label; a bounded metadata timeout; and bounded resolver attempts before properly admitting explicitly required prereleases/extras. The final corrected resolver attempt completed in 6.96 seconds with 60 artifacts, no installation, and no native imports. No failed numerical run or empirical result is inferred from these preparation errors.

## Frozen inputs and executable interface

The final hashes are:

- `code/ops/prepare_native_graph_environment.py`: `98103ab0cc82ded41e1474ed40aedfff50558b7a1480660110c377c8d997b738`.
- `environments/graph-native-cpu/wheel-manifest.json`: `67e27963f2166f8a7e26dba301c9020d2bd7d4c012010fb859e6e43b15ce9842`.
- `environments/graph-native-cpu/requirements.lock`: `ea2f58d2a12f5d8562b31d5fa7b5c646121784867c134afd4a5cf6a8108b8c06`.
- `environments/graph-native-cpu/interpreter-input.json`: `abdda02b28487cae05da9f36c906c896470bee1157e76621fadc5fe67e8e769d`.

The script defaults to read-only check mode:

```sh
python3 code/ops/prepare_native_graph_environment.py \
  --manifest environments/graph-native-cpu/wheel-manifest.json \
  --manifest-sha256 67e27963f2166f8a7e26dba301c9020d2bd7d4c012010fb859e6e43b15ce9842
```

Stage/install require a separately reviewed, hash-pinned admission JSON. Its `kind` is `native_graph_environment_preparation_admission`; `status` is `approved`; it must bind `allowed_modes`, exact script and manifest hashes, canonical project `scratch_root`, unique `run_id`, `wall_seconds`, `storage_cap_bytes=21474836480`, and exact `uv_sha256`/`python_sha256`. No admission is manufactured by this handoff.

Root separately staged the managed interpreter on Mila and supplied its receipt. This task did not install it. The input manifest records the pinned uv 0.9.18 [download metadata](https://raw.githubusercontent.com/astral-sh/uv/0.9.18/crates/uv-python/download-metadata.json), explicit baseline key `cpython-3.10.19-linux-x86_64-gnu`, and archive SHA-256 `d87398fc9292c33ec9f5c85cf72de27e520904d0aff18f0f92221493dd6f7a4c`. Future interpreter staging must use the exact key, project-local install/cache directories, and `--no-bin`; [uv documents these Python install controls](https://docs.astral.sh/uv/reference/cli/#uv-python-install). No HOME shim or global PATH mutation is authorized.

After source review and the admission, the coordinator may execute one preparation attempt using its actual pinned toolchain. For Mila, the root-reported interpreter path/hash is `/network/scratch/m/mehrab.hamidi/representation-collapse/toolchains/python/cpython-3.10.19-linux-x86_64-gnu/bin/python3.10`, `b081289003128d701e48880da11800d7e4410a5777fec6afaa9e31d07419c52f`; uv 0.9.18 SHA is `7f73d66b9767a281c2cae604b7cd0b6eaa1f0dda5ae28b9067e6de745d9593eb`. The root's separate `runs/ops/native-graph-20261015/toolchain-receipt.json` is authoritative for that staging; this handoff did not independently query the host.

```sh
python3 code/ops/prepare_native_graph_environment.py \
  --mode install \
  --manifest environments/graph-native-cpu/wheel-manifest.json \
  --manifest-sha256 67e27963f2166f8a7e26dba301c9020d2bd7d4c012010fb859e6e43b15ce9842 \
  --admission /ABSOLUTE/REVIEWED/admission.json --admission-sha256 ADMISSION_SHA256 \
  --scratch-root /network/scratch/m/mehrab.hamidi/representation-collapse \
  --run-id UNIQUE_APPROVED_ID --wall-seconds 1800 \
  --uv /ABSOLUTE/PINNED/uv \
  --python /network/scratch/m/mehrab.hamidi/representation-collapse/toolchains/python/cpython-3.10.19-linux-x86_64-gnu/bin/python3.10 \
  --local-wheels /ABSOLUTE/DERIVED_TEMPORAL_WHEEL_DIRECTORY
```

The coordinator must enforce an external process deadline of 1800 seconds. Work and child signals stop by 1795 seconds; subprocess checks poll STOP at most one second apart, and child reap is bounded to two seconds. Only bounded receipt cleanup follows. A terminal receipt elapsed time at or beyond the cap overrides success. This script is preparation only and does not invoke SDK/native models; numerical workers require their own reviewed deadline/descendant supervisor.

`--mode stage` downloads/hash checks the complete wheelhouse and exits without installation. `--mode install` stages a new unique attempt and performs offline wheel-only installation in that same attempt; a completed stage attempt is immutable and is not reopened. Reusing an existing run path is refused. Every wheel is byte/hash checked, ZIP paths/types are checked, redirects must remain HTTPS on the official allowlist, and the local pure wheel is prehashed before bounded streaming copy. Existing symlink/nonowned project mutation parents are refused.

The environment/cache/temp/wheelhouse all live under `scratch/environments/graph-native-cpu-UNIQUE_APPROVED_ID/`. A stat-only census includes the entire owned project scratch baseline, including prior vector/source/cache inputs and `toolchains/python`; it counts physical `st_blocks*512`, deduplicates hardlinked inodes, counts at most 200,000 entries, and never traverses symlinks. Any census error fails closed. A conservative reservation adds artifact bytes, three expanded wheel copies and 1 GiB overhead to this complete baseline. Total project staging is limited to 20 GiB per host. No other project directory is scanned or changed. Concurrent installs/downloads/build settings and CPU thread variables are capped at one; no source-build path exists.

Installation uses pinned uv with `--offline`, `--no-index`, `--find-links`, `--no-build`, `--no-deps`, `--require-hashes` and copy linking, then `uv pip check`. The final check reads only distribution metadata and the installed native cell's source bytes. `preparation-receipt.json` atomically records `artifact_manifest_sha256`, `lock_sha256`, target `[3,10]`/`linux-x86_64`, exact installed versions, cell hash, interpreter identity, storage accounting and failures. Successful installation is labeled `installed_metadata_verified_native_smoke_pending`; it cannot set native parity or scientific acceptance. Create the attempt's `STOP` file to stop preparation.

## Validation and remaining acceptance

Read-only real-manifest check and Python 3.10 syntax parsing passed. Nine stdlib guard fixtures passed in 0.004 seconds: exact lock/source binding, URL/redirect policy, symlink output parent, bounded local copy/hash/size/STOP, and duplicate JSON rejection. A distinct fake-process lifecycle fixture passed in 0.006 seconds, verifying STOP during a command, one-second polling, five-second cleanup reserve, two-second reap, and preserving the original stop failure when the process exits between poll and kill. Its initial macOS temporary-path canonicalization failure is preserved separately; only the fixture path was corrected. No real uv, SDK, native process, package install or network call occurred in these fixtures.

A reviewer identified the initial graph-subtree census gap; the original frozen script is retained in `storage-guard-repair-001/`. A distinct whole-project storage fixture passed in 0.002 seconds and verifies prior-baseline inclusion, physical accounting, hardlink deduplication and exclusion of an external symlink target. Independent source review is required before root executes preparation. Actual target metadata installation must precede the separately bounded CPU allocation's native import/cell smoke and calibration. Those checks must initialize mandatory offline W&B 0.30.0 before native numerical calls, preserve the joint hidden/cell state and exact source wrapper, and fail closed on version/hash/dependency discrepancies. No dataset or checkpoint is opened here, no historical training result is reproduced, and no graph/collapse claim is accepted.
