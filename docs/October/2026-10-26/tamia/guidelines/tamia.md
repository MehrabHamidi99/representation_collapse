# Tamia execution guide

On 2026-10-13, read-only access through the existing Mila compute route reached
`tamia3.tamia.ecpia.ca`. The inspected user's scratch root is `/scratch/m/mehrab`
and prior project receipts report account `aip-ioannism`. These values need live
verification for each project; account visibility grants no new budget.

The Alliance TamIA wiki returned an access-denied page during this investigation.
The hardware/partition facts below therefore come from the dated live `sinfo`
receipt and local project records, not a successfully fetched current wiki.
[Official TamIA page](https://docs.alliancecan.ca/wiki/TamIA/en).

## Route and environment

Use the owning project's working SSH route and a bounded BatchMode probe. The
existing route uses Mac → Mila allocation → Tamia. Preserve host verification,
shared masters and MFA. If authentication expires, retain jobs and partial work;
normal interactive reauthentication may be required. Never copy private keys.

The first non-login probe did not find `cluv` on PATH. Exporting the existing
`$HOME/.local/bin` is a process-local fix, not a reason to reinstall tools or alter
shared shell configuration. Check actual executable/version/help. Where supported,
Tamia's `httpproxy` module supplies compute-side network access; this does not
guarantee every package or artifact endpoint is reachable.

## Measure the allocation

Live `sinfo` exposed H100 nodes with four GPUs and H200 nodes with eight GPUs.
`gpubase_bynode_b1/b2/b3` reported 3/12/24-hour limits; CPU `bycore` and `bynode`
families also exist. Do not infer one-GPU allocation cost from one training process
on a whole-node partition. Refresh `sinfo`, associations and `sbatch --test-only`.

The four-H100 policy in Skill Inference is that project's policy, not a default
for this template. Use a measured request suited to the experiment. Small independent
runs may pack a node only if isolated and admitted; a distributed run needs a tested
launcher and numerical equivalence checks. Count allocated GPUs, including idle ones.

## Concrete request preparation

For an admitted four-H100 pilot, the historical request shape is:

```bash
# Inspect/validate first. Replace values with this project's verified request.
sbatch --test-only --account="$SLURM_ACCOUNT" --partition=gpubase_bynode_b1 \
  --nodes=1 --ntasks=1 --cpus-per-task=12 --gres=gpu:h100:4 \
  --mem=64G --time=00:20:00 scripts/cluster_job.sh "$PYTHON" -V
```

This is scheduler validation, not a GPU readiness test. For the real admitted
command, add unique logs/comment, export required paths, use `--parsable`, remove
`--test-only`, and run the measured application. The exact request above reserves
4 × 20/60 = 1⅓ GPU-hours even when only one worker is useful.

Packed workers must use Slurm's GPU binding; verify physical GPU UUIDs because
isolated tasks may all report ordinal zero. Prepare caches once, avoid concurrent
environment installs, and keep outputs/checkpoints/cache mutable state isolated.
Record phase timings, useful throughput, per-device evidence and the final actual
allocation cost. Follow [GPU efficiency](../../../../../skills/gpu-efficiency/SKILL.md).

Native source, data, lock and environment checks precede launch. Cluv staging or a
successful login is not training readiness. Keep every attempt, reconcile unknown
submissions, and use the owning campaign's existing controller where one is active.
