# Approved repository cleanup

Date: 2026-10-19

The user approved batches A (generated local clutter) and B (unused tracked
images), with one commit after each target's deletion. Batch A targets are
ignored/untracked, so their commits record the local removal here; they do not
represent tracked-file deletions. Batch B deletions are recorded directly in Git.

## Batch A — local removals

| Target removed | Reason |
|---|---|
| `__pycache__/` | Root Python bytecode cache; recreated on ordinary imports |
| `tests/__pycache__/` | Test bytecode cache; recreated on ordinary imports |
| `utils/__pycache__/` | Helper bytecode cache; recreated on ordinary imports |
