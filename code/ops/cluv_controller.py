#!/usr/bin/env python3
"""Native cluv with project-specific, node-local SSH control sockets."""
from __future__ import annotations

import hashlib
import asyncio
from importlib.metadata import version
import os
from pathlib import Path
import runpy
import stat
import sys

from cluv.config import find_pyproject
import cluv.remote as remote_module

_native_options = remote_module.get_multiplexing_options_to_use


def configure_ssh(batch=False):
    if version("cluster-uv") != "0.1.5":
        raise SystemExit("invoke through uv run --locked --group ops")
    project = find_pyproject().parent.resolve()
    identifier = hashlib.sha256(str(project).encode()).hexdigest()[:12]
    # UNIX sockets belong on the login node's local filesystem. Shared HOME
    # paths can point at a master running on a different login node.
    runtime = Path("/tmp") / f"rc-cluv-{os.getuid()}-{identifier}"
    if runtime.is_symlink():
        raise SystemExit("refusing a symlink at the control socket directory")
    runtime.mkdir(mode=0o700, exist_ok=True)
    info = runtime.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise SystemExit("control socket directory must be owner-only and owned by this user")
    control_path = str(runtime / "c-%C")
    # Invoke cluv's actual implementation; alter SSH options in this process
    # only. Shared SSH configs and other projects' masters stay untouched.
    remote_module.get_multiplexing_options_to_use = lambda host, native=_native_options: (
        f"-oControlPath={control_path}", "-oControlMaster=auto",
        "-oControlPersist=8h", f"-oBatchMode={'yes' if batch else 'no'}",
        f"-oConnectTimeout={8 if batch else 120}",
        "-oServerAliveInterval=15", "-oStrictHostKeyChecking=yes",
        *native(host),
    )
    return project


async def hold_login(clusters):
    """Keep the native login's event loop and controlling terminal available."""
    from cluv.cli.login import login
    from cluv.config import get_cluv_config
    allowed = get_cluv_config().clusters_names
    if not clusters or any(host not in allowed for host in clusters):
        raise SystemExit("provide explicitly configured cluster aliases")
    for host in dict.fromkeys(clusters):
        configure_ssh(batch=False)
        remotes = await login([host])
        if not remotes:
            print(f"LOGIN_FAILED={host}", flush=True)
            continue
        # A read-only identity check must never request a second authentication.
        configure_ssh(batch=True)
        try:
            probe = await asyncio.wait_for(remotes[0].run(
                "hostname; id -un", display=False, warn=True, hide=True), timeout=25)
            print(f"LOGIN_IDENTITY={host} exit={probe.returncode}\n{probe.stdout}", flush=True)
        except asyncio.TimeoutError:
            print(f"LOGIN_IDENTITY_TIMEOUT={host}", flush=True)
    print("CLUV_CONTROLLER_HOLDING=8h", flush=True)
    await asyncio.sleep(8 * 3600)


def main():
    argv = sys.argv[1:]
    batch = bool(argv and argv[0] == "--batch-mode")
    if batch:
        argv = argv[1:]
    verify = bool(argv and argv[0] == "--verify-access")
    if batch and argv and argv[0] == "--hold-login":
        raise SystemExit("--hold-login requires explicit interactive authentication; use --verify-access for passive checks")
    project = configure_ssh(batch=batch or verify)
    if argv and argv[0] == "--hold-login":
        asyncio.run(hold_login(argv[1:]))
        return
    if verify:
        sys.argv = [str(project / "code/ops/verify_cluv_access.py"), *argv[1:]]
        runpy.run_path(sys.argv[0], run_name="__main__")
    else:
        from cluv.__main__ import main as cluv_main
        cluv_main(argv)


if __name__ == "__main__":
    main()
