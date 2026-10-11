#!/usr/bin/env python3
"""Read-only cluv access checks; never request interactive authentication."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from importlib.metadata import version
import json
import socket

from cluv.cli.login import get_remote_without_2fa_prompt
import cluv.remote as remote_module
from cluv.utils import current_cluster

ALLIANCE = ("tamia", "narval", "rorqual", "fir", "nibi", "trillium",
            "trillium-gpu", "killarney", "vulcan")

# Apply safety options only in this process; never edit shared SSH settings.
_options = remote_module.get_multiplexing_options_to_use
remote_module.get_multiplexing_options_to_use = lambda host: (
    *_options(host), "-oBatchMode=yes", "-oConnectTimeout=8",
    "-oStrictHostKeyChecking=yes",
)

# Authentication and filesystem/scheduler visibility are separate evidence.
# These are read-only commands, with no queue mutation or package installation.
PROBE = """hostname
id -un
printf 'scratch=%s\\n' "${SCRATCH:-}"
if test -n "${SCRATCH:-}" && test -d "$SCRATCH"; then
  printf 'scratch_exists=yes\\n'
else
  printf 'scratch_exists=no\\n'
fi
if command -v sinfo >/dev/null; then
  timeout 8 sinfo --noheader -o '%P|%a|%l' 2>/dev/null
fi
if command -v sacctmgr >/dev/null; then
  timeout 8 sacctmgr --noheader --parsable2 show assoc where user="$USER" format=Cluster,Account,Partition,QOS 2>/dev/null
fi
"""


async def check(cluster, local):
    result = {"cluster": cluster}
    if cluster == local:
        result.update(access="local_controller", reusable_master=None)
        process = await asyncio.create_subprocess_exec(
            "bash", "-c", PROBE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=25)
            result.update(probe_exit_code=process.returncode,
                          probe_stdout=stdout.decode(errors="replace")[:16000])
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            result.update(access="local_probe_timeout")
        return result
    try:
        remote = await asyncio.wait_for(get_remote_without_2fa_prompt(cluster), timeout=10)
        if remote is None:
            result.update(access="no_reusable_master", reusable_master=False)
            return result
        probe = await asyncio.wait_for(remote.run(PROBE, display=False, warn=True, hide=True), timeout=25)
        result.update(access="verified" if probe.returncode == 0 else "probe_failed",
                      reusable_master=True, probe_exit_code=probe.returncode,
                      probe_stdout=probe.stdout[:16000], probe_stderr=probe.stderr[:2000])
    except asyncio.TimeoutError:
        result.update(access="timeout")
    except Exception as error:
        result.update(access="probe_error", error_type=type(error).__name__)
    return result


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clusters", nargs="*", metavar="CLUSTER")
    args = parser.parse_args()
    if any(cluster not in (*ALLIANCE, "mila") for cluster in args.clusters):
        parser.error("clusters must be configured Alliance aliases or mila")
    installed = version("cluster-uv")
    if installed != "0.1.5":
        parser.error("use uv run --locked --group ops to select cluster-uv 0.1.5")
    local = current_cluster()
    targets = list(dict.fromkeys(args.clusters or ALLIANCE))
    records = await asyncio.gather(*(check(cluster, local) for cluster in targets))
    print(json.dumps({"schema_version": 1, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "controller_hostname": socket.gethostname(), "controller_cluster": local,
        "cluv_version": installed, "interactive_authentication": False,
        "remote_mutations": False, "results": records}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
