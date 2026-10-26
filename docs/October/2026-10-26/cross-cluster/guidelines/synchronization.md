# Shared skill synchronization

Git is the primary mechanism for synchronizing versioned work between the local
machine and cluster checkouts; see [source synchronization](source-sync.md).
This collector prepares attributed template updates. Commit and push accepted
changes through Git before deploying them to other machines.

## What is linked

`registry.json` lists local project roots and separately attributed cluster
snapshots. A linked project's `.research-template.json` records its template path
and the SHA-256 baseline of each managed shared file. It is a real file, so the
project remains portable and Overleaf-compatible. No Git remotes or hooks change.

The collector reads `skills/`, `.agents/skills/`, `.cursor/skills/`, root
instructions and top-level durable `doc/`/`docs/` notes. It stores eligible regular
text files as content-addressed objects and builds `sources/index.json`. It excludes
symlinks, caches, checkpoints, runtime directories, oversized and secret-named
files. This is an operational documentation collector, not a whole-project backup
or a guarantee of secret detection. Keep secrets out of documentation.

## Three update paths

- **A registered shared file changes in one project:** if the central file still
  matches its recorded base and there is one unambiguous proposal, sync updates
  the central bundle. Different projects proposing identical bytes count as one
  proposal. The updater does not execute imported code.
- **Project-specific guidance changes:** the new object and change event update
  the template's source library immediately. Use `template-maintenance` to extract
  the reusable lesson. Scientific assumptions, ownership and permissions stay local.
- **Competing edits:** keep all source versions and report a conflict. No
  last-writer rule. The entire affected skill update is held for review.

An entrypoint edit that adds or removes bundle files is also held, so a new
reference cannot leave the central skill incomplete. Promote the complete reviewed
bundle together, then explicitly adopt its new file membership where needed.

Missing projects retain their last snapshot and appear unavailable. File deletions
are recorded but never automatically delete central skills. Empty or absent source
roots are not interpreted as permission to erase accepted content.

## Review a change

`sources/changes.json` retains timestamped events. Resolve an object hash from
`sources/index.json` at `sources/objects/<first-two>/<sha256>.txt`. Compare it with
the previous hash, inspect all affected references/scripts, and decide whether the
change is portable. The background service does not make semantic research decisions.

To resolve a managed conflict, reconcile the central file and contributing project
deliberately. If both have the accepted bytes, `pull --apply` records the new shared
baseline. A local-only edit is held rather than discarded. Deletions and newly
introduced bundle files require explicit adoption/review; they are collected but
not silently added to existing managed manifests.

`pull --project PATH` displays a diff and changes nothing. `--apply` updates only
managed files still at their base and refuses the whole update on a conflict,
missing file or unmerged local edit. This prevents mixed versions within a skill.
Existing research runs stay pinned to the versions they actually used.

## Automatic operation

`python3 scripts/template.py service install` installs
`~/Library/LaunchAgents/org.research-template.skill-sync.plist`, running every
300 seconds with a single-writer lock. It makes only local sync writes. No source
checkout is pulled, no cluster contact occurs, and no LLM is invoked.
Inspect `service status` and `.template-sync/status.json`; errors are in
`.template-sync/service.err`. `service stop` unloads it; remove its plist to prevent
loading at the next login. The service resumes after login/wake, not while asleep.

Cluster mirror registrations update when the owning project's inbound sync
refreshes those local mirrors. This service cannot see new remote-only edits until
that happens. Remote freshness is explicitly separate from local sync freshness.
Do not overwrite a live remote source checkout to keep this library current.
