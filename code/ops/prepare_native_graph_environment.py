#!/usr/bin/env python3
"""Prepare a reviewed isolated legacy CPU environment; never run graph models.

Default check mode is read-only. Stage/install require a separately hash-pinned
root admission. All artifacts are wheels; the single locally derived pure
Temporal wheel must already exist. No package/source build is implemented.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit
import urllib.request
import zipfile

CAP = 20 * 1024**3
CELL = "c3568d2c1ba8926ab43e11c840213f90ebe782ad158dd76809d1dc1c255797af"
CORE = {"torch": "2.0.0+cpu", "torch-geometric": "2.6.1", "torch-geometric-temporal": "0.54.0",
        "numpy": "1.24.4", "torch-cluster": "1.6.3+pt20cpu", "torch-scatter": "2.1.2+pt20cpu",
        "torch-sparse": "0.6.18+pt20cpu", "torch-spline-conv": "1.2.2+pt20cpu", "wandb": "0.30.0"}
HOSTS = {"files.pythonhosted.org", "data.pyg.org", "download.pytorch.org", "download-r2.pytorch.org"}
LOCK_HEADER = "# Linux x86_64 CPython3.10, glibc>=2.28; wheels only, isolated legacy environment.\n"


class PreparationError(ValueError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def pinned_json(path, digest):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024 or sha(path) != digest:
        raise PreparationError("metadata identity mismatch")
    def unique_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise PreparationError("duplicate metadata key")
            value[key] = item
        return value
    value = json.loads(path.read_text(), object_pairs_hook=unique_pairs)
    if not isinstance(value, dict):
        raise PreparationError("metadata object required")
    return value


def validate_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in HOSTS or parsed.username or
            parsed.password or parsed.query or parsed.fragment or parsed.port not in (None, 443)):
        raise PreparationError("artifact URL is outside official source allowlist")


class OfficialRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        validate_url(newurl)
        return super().redirect_request(request, response, code, message, headers, newurl)


def owned_directory(path, scratch, create=False):
    """Inspect every project-relative parent before permitting any mutation."""
    relative = path.relative_to(scratch)
    current = scratch
    for part in relative.parts:
        current = current / part
        if not current.exists() and not current.is_symlink() and create:
            current.mkdir()
        value = current.lstat()
        if not stat.S_ISDIR(value.st_mode) or value.st_uid != os.getuid():
            raise PreparationError("project mutation parent must be an owned nonsymlink directory")
    return path


def bounded_local_copy(source, target, artifact, guard):
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != artifact["bytes"]:
            raise PreparationError("local wheel must have exact immutable byte size")
        digest = hashlib.sha256()
        count = 0
        while True:
            guard()
            block = stream.read(min(1024 * 1024, artifact["bytes"] - count + 1))
            if not block:
                break
            count += len(block)
            if count > artifact["bytes"]:
                raise PreparationError("local wheel exceeded immutable byte size")
            digest.update(block)
        if count != artifact["bytes"] or digest.hexdigest() != artifact["sha256"]:
            raise PreparationError("local wheel hash mismatch before copying")
        stream.seek(0)
        count = 0
        with target.open("xb") as output:
            while True:
                guard()
                block = stream.read(min(1024 * 1024, artifact["bytes"] - count + 1))
                if not block:
                    break
                count += len(block)
                if count > artifact["bytes"]:
                    raise PreparationError("local wheel changed during bounded copy")
                output.write(block)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise PreparationError("local wheel changed during copying")


def validate_manifest(value):
    if value.get("kind") != "legacy_native_graph_cpu_wheel_lock" or value.get("schema_version") != 1:
        raise PreparationError("unsupported wheel manifest")
    target = value.get("target", {})
    if (target.get("python_major_minor") != [3, 10] or target.get("os") != "Linux" or
            target.get("machine") != "x86_64" or target.get("glibc_minimum") != [2, 28]):
        raise PreparationError("only the approved legacy-compatible target is supported")
    if (value.get("binary_dependency_closure_resolved") is not True or value.get("source_pins", {}).get("cell_sha256") != CELL
            or value.get("source_pins", {}).get("temporal_commit") != "46abd39abec0cd8de23c0e489a617c320c3fc5e8"
            or value.get("source_pins", {}).get("temporal_version") != "0.54.0"):
        raise PreparationError("unresolved binary closure or cell identity")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not 9 <= len(artifacts) <= 80:
        raise PreparationError("bounded complete artifact list required")
    names, filenames = {}, set()
    for item in artifacts:
        name, filename = item.get("name"), item.get("filename")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or name in names:
            raise PreparationError("invalid or duplicate distribution")
        if (not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".whl")
                or "\\" in filename or filename in filenames):
            raise PreparationError("only unique relative wheel filenames are admitted")
        if not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise PreparationError("every wheel requires immutable SHA256")
        if type(item.get("bytes")) is not int or not 0 < item["bytes"] <= CAP:
            raise PreparationError("every wheel requires exact bounded bytes")
        if item.get("metadata_body_verified") is not True or not re.fullmatch(r"[0-9a-f]{64}", item.get("metadata_sha256", "")):
            raise PreparationError("wheel dependency metadata is unverified")
        url = item.get("url")
        if url is None:
            if name != "torch-geometric-temporal" or filename != "torch_geometric_temporal-0.54.0-py3-none-any.whl":
                raise PreparationError("only the reviewed pure Temporal wheel may be supplied locally")
        else:
            validate_url(url)
        names[name] = item["version"]; filenames.add(filename)
    if any(names.get(name) != version for name, version in CORE.items()):
        raise PreparationError("source-native or reporting version changed")
    if any(name.startswith(("nvidia-", "triton")) for name in names):
        raise PreparationError("GPU payloads are outside this CPU environment")
    if sum(item["bytes"] for item in artifacts) != value.get("artifact_download_bytes") or value["artifact_download_bytes"] > CAP // 4:
        raise PreparationError("artifact size accounting or storage reserve invalid")
    if hashlib.sha256(requirements_text(artifacts).encode()).hexdigest() != value.get("requirements_sha256"):
        raise PreparationError("canonical requirements lock identity differs")
    return artifacts


def requirements_text(artifacts):
    return LOCK_HEADER + "\n".join(f"{item['name']}=={item['version']} --hash=sha256:{item['sha256']}" for item in artifacts) + "\n"


def wheel_shape(path):
    expanded = 0
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) > 40000:
            raise PreparationError("wheel file count exceeds bound")
        for item in archive.infolist():
            parts = item.filename.split("/")
            mode = item.external_attr >> 16
            if item.filename.startswith("/") or ".." in parts or "\\" in item.filename or stat.S_ISLNK(mode):
                raise PreparationError("unsafe wheel member")
            if stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise PreparationError("nonregular wheel member")
            expanded += item.file_size
    return expanded


def staging_usage(scratch):
    """Physical whole-project scratch census; never traverse symbolic links."""
    seen, total, count = set(), 0, 0
    def account(path):
        nonlocal count, total
        value = path.lstat()
        count += 1
        if count > 200000:
            raise PreparationError("project staging census exceeds entry bound")
        identity = (value.st_dev, value.st_ino)
        if identity not in seen:
            seen.add(identity); total += value.st_blocks * 512
    account(scratch)
    for directory, subdirs, files in os.walk(scratch, followlinks=False, onerror=lambda error: (_ for _ in ()).throw(error)):
        for name in subdirs + files:
            account(Path(directory) / name)
        subdirs[:] = [name for name in subdirs if not (Path(directory) / name).is_symlink()]
    return total


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "stage", "install"), default="check")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--admission-sha256")
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--local-wheels", type=Path)
    parser.add_argument("--uv", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--wall-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    manifest = pinned_json(args.manifest, args.manifest_sha256)
    artifacts = validate_manifest(manifest)
    if args.mode == "check":
        print(json.dumps({"manifest_structurally_verified": True, "artifacts": len(artifacts),
                         "artifact_bytes": manifest["artifact_download_bytes"], "target": manifest["target"],
                         "installed_environment_verified": False, "native_import_smoke_verified": False,
                         "scientific_acceptance": False}, sort_keys=True, indent=2))
        return 0
    if (args.admission is None or not args.admission_sha256 or args.scratch_root is None or
            not args.run_id or args.uv is None or args.python is None or args.local_wheels is None):
        parser.error("mutation requires pinned admission, scratch root, run ID, uv, Python and local wheel directory")
    admission = pinned_json(args.admission, args.admission_sha256)
    scratch = args.scratch_root.absolute()
    for item in [scratch, *scratch.parents]:
        if item.is_symlink():
            raise PreparationError("scratch root must be canonical and nonsymlink")
    if scratch.name != "representation-collapse" or not scratch.is_dir() or scratch.stat().st_uid != os.getuid():
        raise PreparationError("existing owned project scratch root required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.run_id) or not 30 <= args.wall_seconds <= 1800:
        raise PreparationError("invalid unique attempt or bounded walltime")
    if (admission.get("kind") != "native_graph_environment_preparation_admission" or admission.get("status") != "approved"
            or args.mode not in admission.get("allowed_modes", []) or admission.get("script_sha256") != sha(Path(__file__))
            or admission.get("manifest_sha256") != args.manifest_sha256 or admission.get("scratch_root") != str(scratch)
            or admission.get("run_id") != args.run_id or admission.get("wall_seconds") != args.wall_seconds
            or admission.get("storage_cap_bytes") != CAP or admission.get("uv_sha256") != sha(args.uv)
            or admission.get("python_sha256") != sha(args.python)):
        raise PreparationError("exact reviewed preparation envelope required")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise PreparationError("Linux x86_64 host required")
    libc, version = platform.libc_ver()
    if libc != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 28):
        raise PreparationError("glibc>=2.28 required by the pinned reporting wheel")
    if scratch not in args.python.resolve().parents:
        raise PreparationError("the pinned legacy interpreter must already be staged under project scratch")
    owned_directory(args.python.resolve().parent, scratch)
    owned_directory(scratch / "environments", scratch, create=True)
    existing_usage = staging_usage(scratch)
    if existing_usage + manifest["artifact_download_bytes"] + 1024**3 > CAP:
        raise PreparationError("initial per-host project staging allowance exhausted")
    attempt = scratch / "environments" / ("graph-native-cpu-" + args.run_id)
    attempt.mkdir(exist_ok=False)
    receipt = {"schema_version": 1, "state": "initializing", "mode": args.mode, "run_id": args.run_id,
               "manifest_sha256": args.manifest_sha256, "admission_sha256": args.admission_sha256,
               "artifact_manifest_sha256": args.manifest_sha256, "lock_sha256": manifest["requirements_sha256"],
               "python_major_minor": [3, 10], "platform": "linux-x86_64", "installed_metadata_verified": False,
               "storage_cap_bytes": CAP, "scientific_acceptance": False, "native_model_calls": 0,
               "native_import_smoke_verified": False, "source_builds": 0, "main_project_installed": False}
    receipt["existing_project_staging_physical_bytes"] = existing_usage
    # Stop work/signals five seconds before the external hard cap, leaving time
    # for the bounded two-second child reap and an atomic terminal receipt.
    deadline = started + args.wall_seconds - 5
    receipt["wall_seconds"] = args.wall_seconds
    receipt["work_seconds"] = args.wall_seconds - 5
    def guard():
        if time.monotonic() >= deadline:
            raise TimeoutError("preparation walltime expired")
        if (attempt / "STOP").exists():
            raise InterruptedError("owner preparation stop")
    def persist():
        receipt["elapsed_seconds"] = time.monotonic() - started
        if receipt["elapsed_seconds"] >= args.wall_seconds and receipt["state"] in ("staged", "installed_metadata_verified_native_smoke_pending"):
            receipt.update(state="failed", error_type="TimeoutError")
        atomic_json(attempt / "preparation-receipt.json", receipt)
    def alarm(signum, frame):
        raise TimeoutError("preparation walltime expired")
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("UV_", "PIP_", "PYTHON")):
            del env[key]
    env.update({"UV_CACHE_DIR": str(attempt / "cache"), "UV_PYTHON_INSTALL_DIR": str(scratch / "toolchains/python"),
                "UV_PYTHON_BIN_DIR": str(attempt / "unused-bin"), "TMPDIR": str(attempt / "tmp"),
                "UV_CONCURRENT_DOWNLOADS": "1", "UV_CONCURRENT_INSTALLS": "1", "UV_CONCURRENT_BUILDS": "1",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    (attempt / "tmp").mkdir()
    def command(argv):
        guard()
        process = subprocess.Popen(argv, cwd=attempt, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, start_new_session=True)
        try:
            while True:
                guard()
                try:
                    output, _ = process.communicate(timeout=max(0.001, min(1, deadline - time.monotonic())))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=2)
            except BaseException as cleanup_error:
                receipt.setdefault("cleanup_errors", []).append(type(cleanup_error).__name__)
            raise
        if process.returncode:
            raise PreparationError("bounded preparation command failed")
        return output
    persist()
    try:
        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.monotonic()))
        if command([str(args.uv), "--version"]).split()[:2] != ["uv", "0.9.18"]:
            raise PreparationError("pinned uv0.9.18 required")
        python_identity = json.loads(command([str(args.python), "-I", "-c",
            "import json,platform,sys;print(json.dumps({'version':list(sys.version_info[:3]),'implementation':platform.python_implementation(),'system':platform.system(),'machine':platform.machine()}))"]))
        if (python_identity["version"][:2] != [3, 10] or python_identity["implementation"] != "CPython"
                or python_identity["system"] != "Linux" or python_identity["machine"] != "x86_64"):
            raise PreparationError("isolated CPython3.10 interpreter required")
        receipt["python_identity"] = python_identity
        wheelhouse = attempt / "wheelhouse"; wheelhouse.mkdir()
        records, expanded = [], 0
        for artifact in artifacts:
            guard()
            target = wheelhouse / artifact["filename"]
            if artifact["url"] is None:
                source = args.local_wheels / artifact["filename"]
                bounded_local_copy(source, target, artifact, guard)
            else:
                remaining = max(0.001, min(20, deadline - time.monotonic()))
                opener = urllib.request.build_opener(OfficialRedirects())
                with opener.open(artifact["url"], timeout=remaining) as response, target.open("xb") as stream:
                    validate_url(response.url)
                    count = 0
                    while True:
                        guard()
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        count += len(block)
                        if count > artifact["bytes"]:
                            raise PreparationError("artifact exceeded its immutable byte size")
                        stream.write(block)
            if target.stat().st_size != artifact["bytes"] or sha(target) != artifact["sha256"]:
                raise PreparationError("complete wheel hash/size mismatch")
            expanded += wheel_shape(target)
            # Reserve three expanded copies, toolchain allowance and evidence.
            if existing_usage + manifest["artifact_download_bytes"] + 3 * expanded + 1024**3 > CAP:
                raise PreparationError("20GiB staging allowance would be exceeded")
            records.append({"filename": artifact["filename"], "bytes": artifact["bytes"], "sha256": artifact["sha256"]})
            receipt["wheels_completed"] = len(records); persist()
        receipt["verified_wheels"] = records
        receipt["conservative_storage_reservation_bytes"] = manifest["artifact_download_bytes"] + 3 * expanded + 1024**3
        requirements = attempt / "requirements.lock"
        requirements.write_text(requirements_text(artifacts))
        receipt["requirements_sha256"] = sha(requirements)
        if args.mode == "install":
            venv = attempt / "venv"
            command([str(args.uv), "--no-config", "--offline", "venv", "--python", str(args.python), "--no-python-downloads", str(venv)])
            command([str(args.uv), "--no-config", "--offline", "pip", "install", "--python", str(venv / "bin/python"),
                     "--no-index", "--find-links", str(wheelhouse), "--no-build", "--no-deps", "--require-hashes",
                     "--link-mode", "copy", "-r", str(requirements)])
            command([str(args.uv), "--no-config", "--offline", "pip", "check", "--python", str(venv / "bin/python")])
            # Distribution metadata and source text only; no torch/native import.
            program = "import hashlib,importlib.metadata as m,json;d=m.distribution('torch-geometric-temporal');p=d.locate_file('torch_geometric_temporal/nn/recurrent/gc_lstm.py');print(json.dumps({'versions':{x.metadata['Name'].lower().replace('_','-'):x.version for x in m.distributions()},'cell_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))"
            observed = json.loads(command([str(venv / "bin/python"), "-I", "-c", program]))
            if observed["cell_sha256"] != CELL or any(observed["versions"].get(item["name"]) != item["version"] for item in artifacts):
                raise PreparationError("installed distribution/cell identities differ")
            receipt["installed_metadata_verified"] = True
            receipt["installed_versions"] = observed["versions"]
            receipt["installed_cell_sha256"] = CELL
            receipt["venv"] = str(venv)
        guard()
        receipt["actual_project_staging_physical_bytes"] = staging_usage(scratch)
        if receipt["actual_project_staging_physical_bytes"] > CAP:
            raise PreparationError("initial per-host project staging allowance exceeded")
        guard()
        receipt["state"] = "staged" if args.mode == "stage" else "installed_metadata_verified_native_smoke_pending"
        signal.setitimer(signal.ITIMER_REAL, 0)
        persist()
        return 0 if receipt["state"] != "failed" else 1
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt.update(state="failed", error_type=type(error).__name__)
        persist()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
