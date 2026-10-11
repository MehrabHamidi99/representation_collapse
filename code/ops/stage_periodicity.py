#!/usr/bin/env python3
"""Preview, then explicitly stage an immutable posthoc source bundle through cluv.

Run on the held Tamia2 controller. This stages source only: it does not run
full cluv sync, reinstall environments, fetch results, or submit jobs. The
controller's existing checkout is preserved even when it has untracked helpers.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tempfile

BASE = "9004a0c114a8b7c5335dca3d5247449fd10b5ffa"
CONTROLLER = Path("/scratch/m/mehrab/representation-collapse/source")
ROOT = "/scratch/mehrab/representation-collapse"
ORIGIN = "git@github.com:MehrabHamidi99/representation_collapse.git"
CHECKPOINTS = {
    "narval": {
        "narval_5234860": "f1db79385be4de085c457f88355242d0e6e62ceefbe79de4963ab725eca2bcd3",
        "narval_5235508": "e35502d2958c78c465ae4a46639965f3e841fc5e6ad4ff490daf1fc4fb74d4f5",
    },
    "rorqual": {
        "rorqual_22980985": "8752a480b2f5a9d674ef624a7cfb8780b67ceb43b51b4bcf3cadb9ecbd298673",
        "rorqual_22980998": "ec63862dbd27657e99c7becdbdd1301d8813d4525426a9a1545768658561c660",
    },
}


def git(*args, cwd=CONTROLLER):
    return subprocess.check_output(["git", *map(str, args)], cwd=cwd, text=True).strip()


def snapshot():
    return {"sha": git("rev-parse", "HEAD"), "porcelain": git("status", "--porcelain")}


def blob(revision, path):
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=CONTROLLER)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


async def probe(remote, expected_sha):
    code = '''import hashlib,json,os,subprocess
from pathlib import Path
root=Path(ROOT); source=root/'source'; checks=CHECKPOINTS
def git(*args):return subprocess.check_output(['git','-C',str(source),*args],text=True).strip()
for path in [root,source,root/'bootstrap',root/'bootstrap/source.git']:
 assert path.exists() and not path.is_symlink() and path.stat().st_uid==os.getuid(), 'unsafe project path '+str(path)
assert git('rev-parse','HEAD')==EXPECTED, 'unexpected source HEAD'
assert not git('status','--porcelain'), 'dirty target checkout'
assert git('remote').splitlines()==['origin'], 'unexpected target remotes'
assert git('config','--get','remote.origin.url')==ORIGIN, 'unexpected source origin'
mirror=root/'bootstrap/source.git'
assert subprocess.check_output(['git','-C',str(mirror),'rev-parse','--is-bare-repository'],text=True).strip()=='true'
mirror_head=subprocess.check_output(['git','-C',str(mirror),'rev-parse','refs/heads/main'],text=True).strip()
assert mirror_head==EXPECTED, 'unexpected private mirror HEAD'
digests={}
for run,expected in checks.items():
 p=root/'logs/cluv'/run/'pilot/checkpoint.pt'
 assert p.is_file() and not p.is_symlink(), 'missing or symlinked checkpoint'
 digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==expected, 'checkpoint hash mismatch'
 digests[run]=digest
q=subprocess.run(['squeue','-u','mehrab','-h','-o','%i|%j'],capture_output=True,text=True,timeout=12)
assert q.returncode==0, 'queue unavailable'
assert not any(l.split('|')[1].startswith(('rc-','cluv-rc-')) for l in q.stdout.splitlines() if '|'in l), 'active project job blocks source staging'
files=['uv.lock','.python-version','.venv/pyvenv.cfg']
fingerprints={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files}
assert (source/'.venv').resolve().is_relative_to(root.resolve()), 'environment outside project scratch'
print(json.dumps({'sha':git('rev-parse','HEAD'),'clean':True,'checkpoints':digests,'preserved_files':fingerprints,'mirror_head':mirror_head}))
'''.replace("ROOT", repr(ROOT)).replace("CHECKPOINTS", repr(CHECKPOINTS[remote.hostname])).replace("EXPECTED", repr(expected_sha)).replace("ORIGIN", repr(ORIGIN))
    command = shlex.join(["bash", "-lc", shlex.join([
        str(PurePosixPath(ROOT) / "source/.venv/bin/python"), "-c", code])])
    result = await asyncio.wait_for(remote.run(command, display=False, hide=True), 35)
    return json.loads(result.stdout)


async def execute(args, report):
    if version("cluster-uv") != "0.1.5":
        raise RuntimeError("require locked cluster-uv 0.1.5")
    if os.uname().nodename != "tamia2.tamia.ecpia.ca" or os.getuid() != 3115794:
        raise RuntimeError("use the existing held Tamia2 controller, without new authentication")
    if Path.cwd().resolve() != CONTROLLER:
        raise RuntimeError("invoke from the original controller source path")
    bundle = args.bundle.resolve(strict=True)
    if args.bundle.is_symlink() or not bundle.is_relative_to(CONTROLLER.parent / "bootstrap"):
        raise RuntimeError("bundle must be a regular file under controller project bootstrap")
    if not bundle.is_file() or bundle.stat().st_uid != os.getuid() or bundle.stat().st_size > 64 * 1024**2:
        raise RuntimeError("bundle ownership/type or 64MiB bound failed")
    data = bundle.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    report.update(bundle=str(bundle), bundle_bytes=len(data), bundle_sha256=digest,
                  controller_before=snapshot(), cluster_uv_version=version("cluster-uv"))
    heads = git("bundle", "list-heads", bundle).splitlines()
    refs = [line.split()[1] for line in heads if line.split()[0] == args.source_commit]
    if "refs/heads/main" not in refs:
        raise RuntimeError("bundle must advertise the exact requested source as refs/heads/main")
    git("bundle", "verify", bundle)

    # Anchor private SSH options to the original controller before changing cwd.
    sys.path.insert(0, str(CONTROLLER / "code/ops"))
    from cluv_controller import configure_ssh
    import cluv.remote as remote_module
    from cluv.cli.login import get_remote_without_2fa_prompt
    from cluv.cache import ProjectStateOnCluster
    configure_ssh(batch=True)
    remotes = []
    for host in CHECKPOINTS:
        remote_module.get_multiplexing_options_to_use(host)
        remote = await asyncio.wait_for(get_remote_without_2fa_prompt(host), 20)
        if remote is None:
            raise RuntimeError("no reusable private master for " + host)
        report["targets"][host] = {"before": await probe(remote, BASE)}
        remotes.append(remote)

    # Import objects only; controller HEAD/index/untracked helpers are untouched.
    git("fetch", "--no-tags", bundle, "refs/heads/main")
    if git("rev-parse", "FETCH_HEAD") != args.source_commit:
        raise RuntimeError("fetched bundle source mismatch")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, args.source_commit], cwd=CONTROLLER, check=True)
    for path in ("uv.lock", ".python-version"):
        if blob(BASE, path) != blob(args.source_commit, path):
            raise RuntimeError("dependency/Python change needs separate environment preparation: " + path)
    for path in args.required_path:
        git("cat-file", "-e", f"{args.source_commit}:{path}")
    stage_parent = Path(tempfile.mkdtemp(prefix="periodicity-stage-" + args.source_commit[:12] + "-", dir=CONTROLLER.parent / "bootstrap"))
    stage = stage_parent / "source"
    git("clone", "--shared", "--no-checkout", CONTROLLER, stage)
    git("checkout", "--detach", args.source_commit, cwd=stage)
    git("config", "remote.origin.url", ORIGIN, cwd=stage)
    report["isolated_controller_source"] = str(stage)

    # Strict transport and URL mapping exist only in this process's Remote.run.
    native_run = remote_module.Remote.run
    async def mirror_run(remote, command, **kwargs):
        env = dict(kwargs.pop("env", None) or {})
        env.update({"GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "url.file://" + ROOT + "/bootstrap/source.git.insteadOf",
                    "GIT_CONFIG_VALUE_0": ORIGIN, "GIT_TERMINAL_PROMPT": "0",
                    "GIT_SSH_COMMAND": "ssh -oBatchMode=yes -oConnectTimeout=8 -oStrictHostKeyChecking=yes"})
        return await native_run(remote, command, env=env, **kwargs)
    remote_module.Remote.run = mirror_run
    sync_module = importlib.import_module("cluv.cli.sync")
    try:
        os.chdir(stage)
        os.environ.pop("GITHUB_HEAD_REF", None)
        os.environ.pop("GITHUB_REF", None)
        for remote in remotes:
            host = remote.hostname
            remote_bundle = ROOT + "/bootstrap/source-" + args.source_commit + ".bundle"
            receive = '''import base64,hashlib,sys
from pathlib import Path
p=Path(PATH);data=base64.b64decode(sys.stdin.buffer.read(),validate=True)
assert hashlib.sha256(data).hexdigest()==DIGEST
assert not p.is_symlink()
if p.exists():assert p.read_bytes()==data, 'existing bundle differs'
else:
 with p.open('xb')as f:f.write(data)
'''.replace("PATH", repr(remote_bundle)).replace("DIGEST", repr(digest))
            await asyncio.wait_for(remote.run(shlex.join([
                ROOT + "/source/.venv/bin/python", "-c", receive]),
                input=base64.b64encode(data).decode(), display=False, hide=True), 90)
            mirror = ROOT + "/bootstrap/source.git"
            previous = report["targets"][host]["before"]["mirror_head"]
            if previous != BASE:
                raise RuntimeError("mirror no longer at approved base: " + host)
            command = shlex.join(["git", "-C", mirror, "fetch", "--no-tags", remote_bundle, "refs/heads/main"])
            command += " && " + shlex.join(["git", "-C", mirror, "update-ref", "refs/heads/main", args.source_commit, previous])
            await asyncio.wait_for(remote.run(command, display=False, hide=True), 50)
            state = ProjectStateOnCluster()
            await asyncio.wait_for(sync_module.clone_project(remote,
                PurePosixPath(ROOT) / "source", state), 90)
            after = await probe(remote, args.source_commit)
            before = report["targets"][host]["before"]
            if after["checkpoints"] != before["checkpoints"] or after["preserved_files"] != before["preserved_files"]:
                raise RuntimeError("preserved checkpoint/environment/dependency fingerprints changed: " + host)
            report["targets"][host].update(after=after, staged=True,
                native_api="cluv.cli.sync.clone_project", cache_state_persisted=False)
            atomic_json(args.journal, report)
    finally:
        remote_module.Remote.run = native_run
        os.chdir(CONTROLLER)
    report["controller_after"] = snapshot()
    if report["controller_after"] != report["controller_before"]:
        raise RuntimeError("controller checkout changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--required-path", action="append", default=[])
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--execute", action="store_true", help="requires explicit authorization for this exact source; never submits")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit) or args.source_commit == BASE:
        parser.error("provide a new full immutable source SHA")
    if not args.execute:
        print(json.dumps({"preview": True, "source_commit": args.source_commit,
            "expected_clean_target_base": BASE, "bundle": str(args.bundle),
            "targets": list(CHECKPOINTS), "native_api": "cluv.cli.sync.clone_project",
            "strategy": "private bundle mirrors + isolated detached controller clone; source-only sync",
            "preserved": ["controller checkout", "checkpoints", "uv.lock", ".python-version", "existing .venv"],
            "submissions": 0}, indent=2))
        return
    if args.journal is None or not args.required_path:
        parser.error("execution requires --journal and explicit --required-path source files")
    if args.journal.exists():
        parser.error("existing staging journal: reconcile prior attempt rather than retry")
    if not args.journal.resolve().is_relative_to(CONTROLLER.parent / "runs/ops"):
        parser.error("execution journal must be in controller project runs/ops")
    report = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
              "source_commit": args.source_commit, "expected_base": BASE,
              "scope": "authorized project source-only staging; zero submissions/environment sync/results fetch",
              "targets": {}, "success": False}
    try:
        asyncio.run(execute(args, report))
        report["success"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(args.journal, report)
    print(json.dumps(report, indent=2))
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
