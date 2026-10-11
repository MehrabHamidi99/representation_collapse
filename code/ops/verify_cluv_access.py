#!/usr/bin/env python3
"""Read-only cluv access checks; never request interactive authentication."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from importlib.metadata import version
import json
import shlex
import socket

from cluv.cli.login import get_remote_without_2fa_prompt
import cluv.remote as remote_module
from cluv.utils import current_cluster

ALLIANCE = ("tamia", "narval", "rorqual", "fir", "nibi", "trillium",
            "trillium-gpu", "killarney", "vulcan")

# Apply safety options only in this process; never edit shared SSH settings.
_options = remote_module.get_multiplexing_options_to_use
remote_module.get_multiplexing_options_to_use = lambda host, native=_options: (
    "-oBatchMode=yes", "-oConnectTimeout=8", "-oStrictHostKeyChecking=yes",
    *native(host),
)

# Authentication and filesystem/scheduler visibility are separate evidence.
# These are read-only commands, with no queue mutation or package installation.
PROBE = """set -e
hostname
id -un
printf 'cluv_identity_verified=yes\\n'
printf 'scratch=%s\\n' "${SCRATCH:-}"
if test -n "${SCRATCH:-}" && test -d "$SCRATCH"; then
  printf 'scratch_exists=yes\\n'
else
  printf 'scratch_exists=no\\n'
fi
if command -v sinfo >/dev/null; then
  if timeout 8 sinfo --noheader -o '%P|%a|%l'; then
    printf 'partition_query_exit=0\\n'
  else
    printf 'partition_query_exit=%s\\n' "$?"
  fi
else
  printf 'partition_query_exit=unavailable\\n'
fi
if command -v sacctmgr >/dev/null; then
  if timeout 8 sacctmgr --noheader --parsable2 show assoc where user="$USER" format=Cluster,Account,Partition,QOS; then
    printf 'account_query_exit=0\\n'
  else
    printf 'account_query_exit=%s\\n' "$?"
  fi
else
  printf 'account_query_exit=unavailable\\n'
fi
"""


def visibility(stdout):
    fields = dict(line.split("=", 1) for line in stdout.splitlines()
                  if line.startswith(("partition_query_exit=", "account_query_exit=")))
    return {"identity_verified": "cluv_identity_verified=yes" in stdout.splitlines(),
            "scheduler_query_exit_codes": fields}


async def check(cluster, local):
    result = {"cluster": cluster}
    if cluster == local:
        result.update(access="local_controller", reusable_master=None)
        process = await asyncio.create_subprocess_exec(
            "bash", "-c", PROBE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=25)
            result.update(probe_exit_code=process.returncode,
                          probe_stdout=stdout.decode(errors="replace")[:16000],
                          probe_stderr=stderr.decode(errors="replace")[:2000],
                          **visibility(stdout.decode(errors="replace")))
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            result.update(access="local_probe_timeout")
        return result
    try:
        remote = await asyncio.wait_for(get_remote_without_2fa_prompt(cluster), timeout=15)
        if remote is None:
            result.update(access="no_reusable_master", reusable_master=False)
            return result
        # Killarney/Vulcan initialize Slurm PATH and SCRATCH in login profiles.
        command = shlex.join(("bash", "-lc", PROBE))
        probe = await asyncio.wait_for(remote.run(command, display=False, warn=True, hide=True), timeout=25)
        seen = visibility(probe.stdout)
        result.update(access="verified" if seen["identity_verified"] else "probe_failed",
                      reusable_master=True, probe_exit_code=probe.returncode,
                      probe_stdout=probe.stdout[:16000], probe_stderr=probe.stderr[:2000], **seen)
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
    # cluv 0.1.5 performs synchronous `ssh -G` configuration resolution during
    # remote acquisition. Concurrent checks block the event loop cumulatively,
    # consuming every host's deadline before its async socket check can finish.
    # Keep bounded host checks sequential so each gets its own acquisition time.
    records = []
    for cluster in targets:
        records.append(await check(cluster, local))
    print(json.dumps({"schema_version": 1, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "controller_hostname": socket.gethostname(), "controller_cluster": local,
        "cluv_version": installed, "interactive_authentication": False,
        "remote_mutations": False, "results": records}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
