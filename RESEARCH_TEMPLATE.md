# Shared research template

This workspace is linked to [template_project](../template_project/README.md).
Its existing instructions, scientific protocols and budgets remain authoritative.
The link is `.research-template.json`; this is an additive adoption, not a migration.

The template collects changes to registered skills and documentation. Matching
shared files can update the template when there is no competing edit. Different
project-specific instructions become review candidates, never global permissions.
No source project is automatically overwritten and no cluster job is submitted.

Refresh: `python3 "../template_project/scripts/template.py" sync`
Review incoming changes: `../template_project/sources/changes.json`
Preview shared updates here: `python3 "../template_project/scripts/template.py" pull --project .`
Apply an inspected preview by adding `--apply`. Local changes cause conflicts.
For a full additive skill installation use `adopt --project . --install`.
