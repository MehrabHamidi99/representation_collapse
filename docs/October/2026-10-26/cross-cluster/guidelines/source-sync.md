# Git synchronization between local and cluster workspaces

Git is the primary synchronization mechanism between the local machine and all
cluster checkouts. Use the project's Git remote and commit history for code,
experiment configurations, lockfiles, skills, guidelines, documentation and
compact versioned result summaries. Changes made on either side flow through
reviewed commits and the project's normal push/fetch/pull workflow.

## Routine workflow

1. Inspect the repository, branch, remote and working-tree changes on each machine.
   Preserve independent nested repositories and existing branch ownership.
2. Validate and commit intended changes on their owning branch, then push that
   branch within the user's existing authorization. Cluster changes must return
   through Git too; do not leave a separate source of truth on a cluster.
3. Fetch on the receiving machine. Inspect divergence and local changes before
   integration; use a fast-forward pull when appropriate. Resolve divergent work
   without force-pushing, discarding edits or overwriting live sources.
4. Stage each experiment in a separate checkout/worktree or archive of the exact
   admitted commit and verify its full SHA on the target. Record source/config/lock
   identities. Keep queued and running attempts pinned when working branches advance.

Follow the project's integration-branch policy; synchronization does not itself
authorize merging. Use existing Git authentication. When network access is
unavailable, a verified Git bundle can carry committed history while preserving
Git as the source of truth.

## Cluv, bulk artifacts and template updates

Cluv may stage committed Git revisions through its project workflow. Verify the
actual target commit and avoid automatic commits of unrelated working-tree edits.
Cluv and file mirrors do not replace Git history as the source of truth.

Datasets, large checkpoints, environments, caches and runtime journals remain
outside ordinary Git. Use an artifact store or checksum-verified transfer for those
assets, with manifests linking them to the Git commit. Report results from qualifying monitored scientific jobs in [W&B](results-reporting.md).

The template collector observes changes and prepares shared updates. After review,
commit and push accepted template/project changes and fetch them on the other
machines. Its source snapshots preserve provenance; the collector does not deploy
code between clusters or replace Git synchronization.
